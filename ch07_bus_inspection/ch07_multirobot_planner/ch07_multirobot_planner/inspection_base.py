#!/usr/bin/env python3
"""
Tek kol muayene düğümü: ortak taban sınıf.

Her somut muayene düğümü (UR, Kawasaki) TAM OLARAK BİR kolu sürer: o kolu (gerçek
MoveIt ile, çarpışmaları gözeterek) kaplama bakış noktalarının her birine planlar,
doğrudan o kolun denetleyicisine gönderir, ardından kendi kamera(lar)ından en son
SICK bulutunu alır, world çerçevesine dönüştürür ve bakış noktası başına şunları yazar:
    <base>/<sim|real>_pcds/<ur|kawasaki>_data/pcds/<N>.pcd   -- bulut
    <base>/<sim|real>_pcds/<ur|kawasaki>_data/poses/<N>.txt  -- OctoMap başlangıcı
                             "x y z qx qy qz qw" (world çerçevesi)

İki kol AYRI süreçler olarak (her biri bir düğüm) çalışır; kollar arasında bir
bekleme engeli yoktur, yapıları gereği paraleldirler. Kollar arası çarpışma
farkındalığı yine korunur, çünkü bu yürütücüde değil, iki kolun /joint_states
konularıyla beslenen ortak move_group planlama sahnesinde yaşar.

Bu taban, iki kolun paylaştığı her şeyi (planlama, gönderim, yörünge önbelleği, veri
toplama, TF, planlama sahnesi dolgusu, ölçülen varış denetimi yardımcıları) ve bir
şablon `run()` / `run_sequence()` tutar. Alt sınıflar kolun MoveIt2 nesnesini,
denetleyicisini, veri toplama akışlarını, planlama biçimini (poz hedefi ya da eklem
uzayı) ve yerleşme stratejisini sağlar.
"""
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.task import Future
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       qos_profile_sensor_data)

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene, GetPositionIK
from moveit_msgs.msg import LinkPadding, MoveItErrorCodes, PlanningScene, PlanningSceneComponents
from sensor_msgs.msg import PointCloud2, JointState
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from ch07_multirobot_planner import paths
from scipy.spatial.transform import Rotation

from book_pymoveit2.joint_wrap import (describe_changes, JointChange, JointLimit,
                                       parse_joint_limits, wrap_to_reference)

import open3d as o3d

# Plan dosyasındaki tek bir bakış noktası: {"id", "position", "rotation", ...}
Viewpoint = Dict[str, Any]


class InspectionNodeBase(Node):
    """Tek kol muayene düğümü için taban sınıf. `robot_tag` "ur" ya da "kawasaki"dir."""

    # BU sınıfta bildirilen parametrelerin kol başına varsayılanları. Alt sınıf onları
    # düz sınıf öznitelikleri olarak geçersiz kılar; super().__init__() parametreyi
    # bildirdiğinde bunlar zaten bağlıdır. İki kolun farklı değerlere ihtiyacı vardır
    # (rastgele IK tohumları ve küçültülmüş IK grubu UR için değil Kawasaki için önemlidir)
    # ama tek bir başlatma parametresi sözlüğünü paylaşırlar; bu yüzden ayrımın
# yaşayabileceği tek yer kol başına VARSAYILAN değerdir.
    DEFAULT_IK_RANDOM_SEEDS = 0
    DEFAULT_BRANCH_IK_GROUP = ""

    def __init__(self, node_name: str, robot_tag: str) -> None:
        super().__init__(node_name)
        self.robot_tag = robot_tag

        # run()'dan önce alt sınıf tarafından doldurulur: kolun MoveIt2 planlayıcısı,
        # denetleyici eylem istemcisi, okunabilir bir etiket, veri toplama akışları ve plan
        # anahtarları.
        self.moveit = None
        self.ctrl = None
        self.arm_label = robot_tag.upper()
        self.streams = []
        self.viewpoints_key = f"{robot_tag}_viewpoints"
        self.start_pose_param = None  # örneğin "start_pose_ur"
        self.return_home_param = None  # örneğin "return_home_ur"
        self.home_positions_param = None  # örneğin "ur_home_joint_positions"
        # UR, YENİ bir planı başlangıcına hizalamayı atlar (plan zaten güncel pozdan
        # başlar); rayın yersiz ileri-geri hareketinden kaçınılır. Kawasaki eski "her zaman
        # hizala" davranışını korur (_at_pose koruması zaten başlangıçtayken onu etkisiz kılar).
        self.align_on_fresh = True
        # Ayrı bir "kayıtlı başlangıca hizalan" hareketi yapmak yerine her yörüngenin ilk
        # noktasını güncel ölçülen poza TAŞI (kanıtlanmış ur_inspection_scenario oynatması
        # gibi); rayın ileri-geri sarsılmasını durduran budur.
        # Artık iki kol da bunu yapıyor: UR düzeltildikten sonra Kawasaki uzun süre eski ayrı
        # hizalama yolunda bırakılmıştı; bakış noktası başına çok daha fazla yol almasının sebebi
        # buydu.
        # Güvenle geri çekilir: fark başlangıç toleransından büyük olduğunda
        # _rebase_traj_to_current reddeder ve açık yürüyüş yine yapılır.
        self.rebase_to_current = False

        self._declare_shared_params()

        self.plan_file = self.get_parameter("plan_file").value
        self.world_frame = self.get_parameter("world_frame").value
        self.only_sim = bool(self.get_parameter("only_sim").value)

        self._cb = ReentrantCallbackGroup()
        self._plan_timeout = self.get_parameter("plan_timeout_sec").value
        self._motion_timeout = self.get_parameter("motion_timeout_sec").value

        # pymoveit2'ninki yerine KENDİ /compute_ik istemcimiz: o istemci grubun varsayılan
        # ucunu sabit kodlar (ik_link_name'i hiç ayarlamaz; bu yüzden kamera optik çerçevesi
        # yerine ur10e_tool0 için çözerdi) ve engelleyici sarmalayıcısı düğümü içeride
        # döndürür, bu da bu düğümün yürütücü wait set'ini bozar.
        # En yakın dal IK'sını etkinleştirmek için alt sınıfta self.ik_group_name'i ayarlayın.
        self.ik_group_name = None
        # move_group `branch_ik_group`'u reddederse (INVALID_GROUP_NAME) kalıcı olarak
        # temizlenir; böylece küçültülmüş grubunu kaybetmiş bir yapılandırma dal aramasını
        # tamamen kaybetmek yerine planlama grubuna düşer.
        self._branch_ik_group_ok = True
        self._last_ik_error = None  # en son /compute_ik çağrısının error_code.val değeri
        self._ik_rng = None  # gerektiğinde `ik_random_seed` ile tohumlanır
        self._ik_cli = self.create_client(
            GetPositionIK, "compute_ik", callback_group=self._cb)

        # Planlama sahnesi servis istemcileri (çarpışma dolgusu).
        self._get_scene_cli = self.create_client(
            GetPlanningScene, "get_planning_scene", callback_group=self._cb)
        self._apply_scene_cli = self.create_client(
            ApplyPlanningScene, "apply_planning_scene", callback_group=self._cb)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        # Eklem başına en son ölçülen konum; bütün /joint_states yayıncılarından
        # BİRLEŞTİRİLİR (kol yayıncısı ve AGV köprüsü AYRI mesajlarda yayımlar).
        # _wait_until_reached tarafından ilerlemeyi gerçek varışa bağlamak için kullanılır.
        self._joint_pos = {}
        # görülen en yeni /joint_states mesajının başlık zaman damgası (s)
        self._joint_stamp = None
        self._last_plan_error = None  # en son başarısız planın MoveItErrorCodes.val değeri
        self.create_subscription(
            JointState, "/joint_states", self._joint_states_cb, 10, callback_group=self._cb)

        # Bir hedef açının farklı bir 2*pi eşdeğeriyle yeniden ifade edilip edilemeyeceğine
        # karar vermek için kullanılan URDF eklem sınırları (bkz. _wrap_goal_to_current).
        # robot_state_publisher /robot_description'ı kalıcı yayımlar; bu yüzden düğüm çok
        # sonra başlasa bile TRANSIENT_LOCAL bir abonelik onu alır.
        self._urdf_xml = None
        self._joint_limits = None
        self.create_subscription(
            String, "/robot_description", self._robot_description_cb,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       history=HistoryPolicy.KEEP_LAST),
            callback_group=self._cb)

        # Önbelleğe alınmış yörüngelerin yeri: trajectory_cache_dir verilmezse
        # ~/ros2_book_output/ch07/trajectories/.
        tcd = self.get_parameter("trajectory_cache_dir").value
        self._traj_dir = os.path.expanduser(tcd) if tcd else paths.output_path("trajectories")

    # ------------------------------------------------------------------ #
    # Bütün kolların paylaştığı parametreler.
    # ------------------------------------------------------------------ #
    def _declare_shared_params(self) -> None:
        self.declare_parameter("plan_file", paths.shipped_plan_file())
        # only_sim, GERÇEK SICK verisinin benzetimdekiyle birlikte toplanıp toplanmayacağını
        # belirler.
        self.declare_parameter("only_sim", True)
        # GERÇEK DÜNYA testi için tek kol seçimi. Her düğüm çalışıp çalışmayacağına kendisi
        # karar verir (bkz. should_run): only_ur=True -> yalnızca UR düğümü çalışır; ikisi de
        # False (varsayılan) -> iş birlikli koşu; ikisi de True çelişkilidir ve ikisi de çalışır.
        self.declare_parameter("only_ur", False)
        self.declare_parameter("only_kawasaki", False)
        self.declare_parameter("output_base_dir", paths.output_path("pcds"))

        self.declare_parameter("world_frame", "world")
        self.declare_parameter("capture_settle_sec", 1.5)
        self.declare_parameter("capture_timeout_sec", 6.0)
        self.declare_parameter("transform_to_world", True)

        self.declare_parameter("plan_timeout_sec", 30.0)
        self.declare_parameter("motion_timeout_sec", 120.0)
        self.declare_parameter("allowed_planning_time", 5.0)
        # MoveIt bu kadar kez planlar ve EN KISA sonucu tutar; yani yol uzunluğu için
        # eldeki en ucuz ayardır. 2026-09-05'te planlanabilen 8 Kawasaki şase hamlesinde
        # ölçüldü (/plan_kinematic_path, RRTConnect, 5 sn bütçe): 1 deneme 4680 derece,
        # 3 deneme 4364, 10 deneme 3663 (-%22), 20 deneme 3603.
        # ~10'da doyuma ulaşır ve RRTConnect'in 10 denemesi 0.15 sn sürer; bu hücrede
        # herhangi bir asimptotik olarak en iyi planlayıcının TEK denemesinden azdır.
        self.declare_parameter("num_planning_attempts", 10)
        # Boş -> MoveIt'in kendi yedeği, yani RRTConnect (yapılandırma paketlerinin hiçbiri
        # `default_planner_config` ayarlamaz). Başka her değer, moveit_config paketinin
        # ompl_planning.yaml dosyasında bu grubun altında listelenen bir yapılandırma adı
        # olmalıdır; örneğin "RRTstarkConfigDefault" ya da "AnytimePathShorteningkConfigDefault".
        # Ad bilinmiyorsa MoveIt sessizce varsayılana düşer; bu yüzden günlükten doğrulayın.
        # Karşılaştırmalı alternatifler araştırma deposundaki tools/planner_bench.py içindedir.
        self.declare_parameter("planner_id", "")
        self.declare_parameter("plan_attempts", 4)

        # MoveIt'in bir kolu asla zeminin altına planlamaması için zemin çarpışma kutusu.
        self.declare_parameter("add_ground_plane", True)
        self.declare_parameter("ground_plane_z", -0.02)
        self.declare_parameter("ground_plane_size", 6.0)
        self.declare_parameter("ground_plane_thickness", 0.2)
        self.declare_parameter("collision_padding", 0.04)

        # Ölçülen varış denetimi (Kawasaki kullanır; UR denetleyicisine güvenir).
        # SABİT bir bütçedir, yörüngenin süresine bağlı değildir: kullanılan en yavaş hız
        # ölçeğinde en uzun hamleyi kapsamalıdır. 2026-09-10'da kawasaki_velocity
        # 0.015 -> 0.007 olunca uzun hamleler 90 sn'yi aştığı için 90 -> 180 sn'ye çıkarıldı
        # (düğüm o zaman veriyi hareketin ortasında topluyor ve süren hareketi kesiyordu).
        self.declare_parameter("arrival_timeout_sec", 180.0)
        self.declare_parameter("arrival_joint_tol", 0.10)  # radyan, dönel eklemler
        self.declare_parameter("arrival_linear_tol", 0.08)  # m, world_to_agv / UR rayı

        # Kaç TUR atılacak. Buradaki varsayılan 1'dir ve ÖYLE KALMALI: bu sınıfı
        # Kawasaki düğümü de kullanıyor ve o tek tur atar. Çok turu yalnız UR alır;
        # başlatma dosyasındaki ur_tours argümanı (varsayılan 2) bu parametreyi UR'ye geçirir.
        # Neden var: iki kol aynı anda başlıyor ama Kawasaki turunu UR'den ~7.5 dk sonra
        # bitiriyor (ölçüm: UR 26 durak / 4.5 dk hareket, Kawasaki 11 durak / 8.7 dk).
        # UR'nin boş beklediği bu süre, aynı yörüngelerle ikinci bir tur atarak
        # değerlendirilebilir. DİKKAT: veri toplama dosyaları her turda 1'den numaralanır,
        # yani sonraki tur bir öncekinin bulutlarının ÜZERİNE yazar; kaplama değişmez,
        # yalnızca bulutlar daha geç bir anda kaydedilmiş olur.
        self.declare_parameter("tours", 1)

        # Yörünge önbelleği.
        self.declare_parameter("use_trajectory_cache", True)
        self.declare_parameter("force_replan", False)
        self.declare_parameter("trajectory_cache_dir", "")
        self.declare_parameter("traj_start_tol_rad", 0.05)
        self.declare_parameter("traj_start_tol_m", 0.03)

        # Sabit BAŞLANGIÇ pozu + bakış noktası başına ÖNCE-EVE uğrama.
        self.declare_parameter("go_to_start", True)
        # Bir önceki bakış noktasından doğrudan değil, EVE uğranarak yaklaşılan bakış noktası
        # kimlikleri (doğrudan hamle, ROS modelinin bilmediği fiziksel bir kablo kanalını
        # sıkıştırır). dynamic_typing ŞARTTIR: rclpy boş liste varsayılanının tipini
        # BYTE_ARRAY olarak çıkarır ve çıkarılan tip tanımlayıcıyı geçersiz kılar; böylece
        # başlatma dosyasından gelen bir STRING_ARRAY değeri bildirim anında reddedilirdi.
        self.declare_parameter(
            "home_before_viewpoints", [], ParameterDescriptor(dynamic_typing=True))

        # Poz hedefi planlama toleransları (yalnızca bir kol Kartezyen poz hedefine
        # planlarken kullanılır, yani UR). _plan_pose tabanda olduğu için burada durur.
        self.declare_parameter("pose_goal_pos_tol", 0.005)  # m
        self.declare_parameter("pose_goal_ori_tol", 0.02)  # rad
        # |açısı| +/-2*pi sınır kenarına bu pay kadar yakın olan bir eklem hedefi planlamadan
        # önce (-pi, pi] aralığına sarılır (OMPL hedef bölgesini kurtarma).
        self.declare_parameter("normalize_edge_margin", 0.2)  # radyan

        # 2*pi HEDEF SARMA. IK bir çözüm ailesinin rastgele bir dalını döndürür; bu yüzden
        # ardışık bakış noktaları örneğin +100 derece ve -260 derece olarak saklanabilir:
        # aynı fiziksel eklem açısı, ama gidilecek fazladan bir tam tur. Açıkken her dönel
        # hedef, kolun GÜNCEL ölçülen açısına en yakın 2*pi eşdeğeriyle yeniden ifade
        # edilir; ama YALNIZCA o eşdeğer eklemin URDF sınırları içinde kalıyorsa
        # (her kenardan wrap_limit_margin uzakta). Aralığı <= 2*pi olan eklemler
        # (Kawasaki joint1/2/3/5 ve UR dirseği) bu yüzden asla değişmez; eski, sınırları
        # gözetmeyen sürümün güvenli olmadığı yerde bunu güvenli kılan budur.
        self.declare_parameter("wrap_goals_to_current", True)
        self.declare_parameter("wrap_limit_margin", 0.05)  # her sınırdan uzak tutulan radyan
        # Bundan az kazandıran yeniden yazmaları yok say; hedefler boşuna değiştirilmesin.
        self.declare_parameter("wrap_min_gain", 0.35)  # radyan (~20 derece)

        # EN YAKIN DAL IK (yalnızca poz hedefli kollar, yani UR). Bir pozun birkaç IK dalı
        # vardır (ona bambaşka kol duruşlarından ulaşan omuz/dirsek/bilek dönüşleri) ve
        # sarma aralarında yardımcı olamaz, çünkü bunlar gerçekten farklı konfigürasyonlardır.
        # Çıplak bir Kartezyen poz hedefine planlamak, MoveIt'in hedef örnekleyicisinin
        # bunlardan herhangi birini seçmesine izin verir; kolun birkaç santimetre gitmek
        # için yüzlerce derece dönmesi böyle olur. Bunun yerine IK'yı kendimiz, kolun
        # GÜNCEL ölçülen pozuyla tohumlayarak çözer ve gerçekten yol veren en yakın dala
        # planlarız. Hiçbiri yol vermezse düz poz hedefine düşülür.
        self.declare_parameter("nearest_branch_ik", True)
        # örneklenecek global kip yeniden başlatmaları
        self.declare_parameter("ik_seed_attempts", 4)
        self.declare_parameter("ik_timeout", 0.1)  # /compute_ik çağrısı başına saniye
        self.declare_parameter("ik_avoid_collisions", True)
        # Aynı fikrin Kawasaki tarafı. Grubu FAZLADAN eksenlidir (AGV rayı + 6 eklem); yani
        # ray her çözümün serbest bir parametresidir: daha yakın bir kol duruşu bulmak için
        # onu tohum olarak dene, ama bir adayın AGV'yi planlayıcının seçtiği konumdan
        # gerçekten uzaklaştırmasına asla izin verme (o seçim kaplama için yapıldı).
        # branch_max_rail_shift <= 0 rayı tam olarak planlanan değere kilitler.
        self.declare_parameter("branch_rail_probe", 0.25)  # m, tohum denemesi; 0 kapatır
        self.declare_parameter("branch_max_rail_shift", 0.0)  # bir adayın rayı oynatabileceği m
        # YEREL ÇÖZÜCÜ İÇİN RASTGELE YENİDEN BAŞLATMALAR. Yukarıdaki iki ayar bir pozun
        # dallarını yalnızca ÇÖZÜCÜ rastgele yeniden başlıyorsa örnekler; bu UR'nin
        # `mode: global` ile pick_ik'sı için doğrudur, Kawasaki grubunun kullandığı KDL için
        # DEĞİL. Planlanan hedefin etrafında toplanan her tohum doğrudan ona yakınsar; bu kolun
        # her bakış noktasında "nearest-branch IK: 1 distinct branch(es)" yazmasının ve
        # kawa_vp_008'in 876 derecelik hamlesi gibi hamlelerin asla iyileştirilememesinin
        # sebebi buydu. Çalışan move_group'a karşı canlı bir /compute_ik denemesi meseleyi
        # çözdü: 24 RASTGELE tohumla aynı grup ve pozlar 16-20 FARKLI çözüm döndürdü; yani
        # sorun çözücüde değil, tohum listemizdeydi. Bunlar listeye N rastgele, sınır içi kol
        # duruşu ekler.
        # Yalnızca dönel eklemler: doğrusal bir eksen planın kaplama tercihidir, asla
        # rastgeleleştirilecek bir şey değildir. IK çağrısı maliyeti yalnızca yörünge
        # KAYDEDİLİRKEN.
        self.declare_parameter("ik_random_seeds", self.DEFAULT_IK_RANDOM_SEEDS)
        # rastgele sayı tohumu; böylece koşu tekrarlanabilir
        self.declare_parameter("ik_random_seed", 0)
        # Aynı düzeltmenin DİĞER yarısı. `branch_max_rail_shift` AGV'yi korur, ama 7
        # serbestlik dereceli (ray + 6) Kawasaki grubunda ray her çözümün serbest
        # değişkenidir ve tohum onu sabitlese bile KDL onu kaydırır; koruma da tohumların
        # bulduğu dalların neredeyse hepsini atıyordu (kawa_vp_005'te ölçüldü: 19 çözümden
        # 1'i planlanan rayı koruyordu; kawa_vp_002'de 16'dan 3'ü). Rayı içermeyen
        # KÜÇÜLTÜLMÜŞ bir grupta çözmek kaymayı yapısal olarak imkânsız kılar;
        # bu yüzden bulunan dalların %100'ü planlayıcının seçtiği rayda geçerlidir. Boş
        # metin -> planlama grubunda çöz, yani tam olarak eski davranış.
        self.declare_parameter("branch_ik_group", self.DEFAULT_BRANCH_IK_GROUP)
        # Dal adayları hedef uzayındaki uzaklığa (|hedef - güncel| toplamı) göre SIRALANIR;
        # bu yalnızca bir yaklaşıklıktır: OMPL'nin gerçekte döndürdüğü yol o düz çizginin
        # çok ötesine sapabilir. Kapı koşusunda ölçüldü: Kawasaki'nin kayıtlı yolları
        # doğrudan eklem uzaklığının 1.54 katı (en kötü 3.5 katı), UR'ninkiler 1.05 katı;
        # yani yalnızca PLANLANABİLEN ilk adayı almak, ikinciden çok daha uzun bir yolu
        # kilitleyebilir. Bunlardan en fazla bu kadarını planla ve GERÇEK yörüngesi en
        # kısa olanı tut. 1, eski "planlanabilen ilki" davranışını tam olarak geri getirir.
        # Fazladan planlama maliyeti yalnızca KAYIT sırasında; oynatmalar önbellekten gelir.
        # BİLEREK _goal_policy'nin parçası DEĞİLDİR: bu, hangi hedefin geçerli olduğunu
        # değil, hangi yolun kaydedildiğini değiştirir; bu yüzden mevcut bir önbellek
        # geçerli kalır ve tuttuğu yolu oynatmaya devam eder. Yeni ayarın etkisi için
        # force_replan:=true ile yeniden kaydedin; böylece donanımda doğrulanmış şase
        # yörüngeleri buradaki bir ayar değişikliğiyle sessizce atılmaz.
        self.declare_parameter("branch_plan_candidates", 3)

    # ------------------------------------------------------------------ #
    # Alt sınıfın geçersiz kıldığı kancalar.
    # ------------------------------------------------------------------ #
    def should_run(self) -> bool:
        """
        only_ur/only_kawasaki (+ enable) verildiğinde BU kolun çalışıp çalışmayacağını döndürür.

        İkisi de True ise çelişkilidir -> ikisi de çalışır (bir kez uyarılır).
        """
        raise NotImplementedError

    def _setup_planning_scene_extras(self) -> None:
        """
        Kola özgü planlama sahnesi eklemeleri.

        Diziden önce ve zemin düzlemi / dolgudan sonra bir kez uygulanır.
        Varsayılan: hiçbir şey (Kawasaki).
        """
        return

    def _use_pose_goal(self) -> bool:
        """Kartezyen poz hedefi kullanılsın mı: True -> poz hedefi (UR), False -> eklem uzayı."""
        return False

    def _pose_target_link(self) -> Optional[str]:
        """Poz hedefinin target_link'i (kamera optik çerçevesi). Yalnızca UR."""
        return None

    def _ik_target(self, vp: Viewpoint) -> Optional[Tuple[List[float], np.ndarray, str]]:
        """
        Bu bakış noktası için tohumlu bir /compute_ik'nın (position, quat_xyzw, ik_link) üçlüsü.

        Kol bunu ifade edemiyorsa None döner.

        Varsayılan UR'ninkidir: saklanan poz zaten `_pose_target_link()`'in pozudur; bu
        yüzden olduğu gibi verilir. Kawasaki bunu geçersiz kılar, çünkü onun saklanan pozu
        ÜRETİCİNİN görüş çerçevesidir; IK için anlam kazanmadan önce görüş ekseni düzeltmesi
        ve sabit kamera->uç dönüşümü uygulanmalıdır; planlayıcının `_make_kawasaki_reach_fn`
        içinde kullandığı zincirin aynısı.
        """
        if vp.get("position") is None or vp.get("rotation") is None:
            return None
        link = self._pose_target_link()
        if link is None:
            return None
        quat = Rotation.from_matrix(np.array(vp["rotation"], dtype=float)).as_quat()
        return [float(x) for x in vp["position"]], quat, link

    def _settle(self, send: Future, targets: Optional[Dict[str, float]], label: str) -> bool:
        """
        Gönderilen bir hareketin bitmesini bekler (kola özgü).

        UR: yalnızca denetleyici sonucu. Kawasaki: ölçülen varış.
        """
        raise NotImplementedError

    def _viewpoint_targets(
            self, traj: JointTrajectory, vp: Viewpoint) -> Optional[Dict[str, float]]:
        """
        Bir bakış noktası hareketinin varış hedeflerini döndürür.

        Kawasaki için hedefler, UR için None (yalnızca denetleyici sonucu).
        """
        return None

    # ------------------------------------------------------------------ #
    # Veri toplama akışları.
    # ------------------------------------------------------------------ #
    def _make_stream(
            self, name: str, robot: str, topic: str, sensor_frame: str,
            subtree: str) -> Dict[str, Any]:
        pcd_dir = os.path.join(subtree, "pcds")
        pose_dir = os.path.join(subtree, "poses")
        os.makedirs(pcd_dir, exist_ok=True)
        os.makedirs(pose_dir, exist_ok=True)
        return {
            "name": name, "robot": robot, "topic": topic, "sensor_frame": sensor_frame,
            "subtree": subtree, "pcd_dir": pcd_dir, "pose_dir": pose_dir,
            "latest_cloud": None,
        }

    def _make_cloud_cb(self, stream: Dict[str, Any]) -> Callable[[PointCloud2], None]:
        def _cb(msg: PointCloud2) -> None:
            stream["latest_cloud"] = msg
        return _cb

    def _subscribe_streams(self) -> None:
        """
        Her akış için bir PointCloud2 aboneliği oluşturur.

        self.streams ayarlandıktan sonra çağırın.
        """
        for s in self.streams:
            s["sub"] = self.create_subscription(
                PointCloud2, s["topic"], self._make_cloud_cb(s),
                qos_profile_sensor_data, callback_group=self._cb)
        streams_desc = ", ".join(f"{s['name']}('{s['topic']}' -> "
                                 f"{s['subtree']})" for s in self.streams)
        self.get_logger().info(
            f"{self.arm_label} inspection ready [only_sim={self.only_sim}]. "
            f"plan='{self.plan_file}'. Capture streams: {streams_desc}.")

    # ------------------------------------------------------------------ #
    # Planlama sahnesi: zemin düzlemi + çarpışma dolgusu.
    # ------------------------------------------------------------------ #
    def _add_ground_plane(self) -> None:
        if not self.get_parameter("add_ground_plane").value:
            return
        top_z = self.get_parameter("ground_plane_z").value
        size = self.get_parameter("ground_plane_size").value
        thickness = self.get_parameter("ground_plane_thickness").value
        center_z = top_z - thickness / 2.0
        for _ in range(2):
            self.moveit.add_collision_box(
                id="ground_plane",
                size=(size, size, thickness),
                position=(0.0, 0.0, center_z),
                quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id=self.world_frame,
            )
            time.sleep(0.3)
        self.get_logger().info(
            f"Ground plane added: {size}x{size}x{thickness} m box, top face at z={top_z:.3f} "
            f"in '{self.world_frame}'.")

    def _arm_link_names(self, scene: PlanningScene) -> List[str]:
        """
        BU kolun çarpışma dolgusu uygulanan HAREKETLİ bağlantıları.

        Kol başına geçersiz kılınır (UR 'ur10e_*', Kawasaki 'link1'..'link6').
        """
        raise NotImplementedError

    def _apply_collision_padding(self) -> None:
        padding = float(self.get_parameter("collision_padding").value)
        if padding <= 0.0:
            return
        if not self._get_scene_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warning("get_planning_scene unavailable; skipping collision "
                                      "padding.")
            return
        req = GetPlanningScene.Request()
        req.components.components = (PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
                                     | PlanningSceneComponents.WORLD_OBJECT_NAMES)
        future = self._get_scene_cli.call_async(req)
        if not self._wait_future(future, 5.0) or future.result() is None:
            self.get_logger().warning("get_planning_scene failed; skipping collision padding.")
            return
        arm_links = self._arm_link_names(future.result().scene)
        if not arm_links:
            self.get_logger().warning("No moving arm links found; skipping collision padding.")
            return
        diff = PlanningScene()
        diff.is_diff = True
        diff.link_padding = [LinkPadding(link_name=n, padding=padding) for n in arm_links]
        if not self._apply_scene_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warning("apply_planning_scene unavailable; skipping collision "
                                      "padding.")
            return
        af = self._apply_scene_cli.call_async(ApplyPlanningScene.Request(scene=diff))
        if not self._wait_future(af, 5.0) or af.result() is None:
            self.get_logger().warning("apply_planning_scene failed; collision padding may be off.")
            return
        self.get_logger().info(
            f"Collision padding {padding * 100:.1f} cm applied to {len(arm_links)} moving arm "
            "links. "
            f"success={af.result().success}.")

    # ------------------------------------------------------------------ #
    # Veri toplama + kaydetme.
    # ------------------------------------------------------------------ #
    def _lookup_sensor_pose(
            self, frame_id: str, fallback_frame: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        for target in (frame_id, fallback_frame):
            if not target:
                continue
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.world_frame, target, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0))
                t = tf.transform.translation
                q = tf.transform.rotation
                return np.array([t.x, t.y, t.z]), np.array([q.x, q.y, q.z, q.w]), target
            except Exception:
                continue
        return None

    def capture_and_save(
            self, index: int, active_robots: Set[str],
            fallback_by_robot: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> bool:
        """
        Bu adımda robotu hareket eden her akıştan taze bir bulut alır ve kaydeder.

        Her bulut kendi robotunun klasör ağacına yazılır. En az biri kaydedildiyse True döner.
        """
        settle = self.get_parameter("capture_settle_sec").value
        active_streams = [s for s in self.streams if s["robot"] in active_robots]
        for s in active_streams:
            s["latest_cloud"] = None
        time.sleep(settle)

        saved_any = False
        for s in active_streams:
            if self._save_stream(s, index, fallback_by_robot.get(s["robot"])):
                saved_any = True
        return saved_any

    def _save_stream(
            self, stream: Dict[str, Any], index: int,
            fallback_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> bool:
        timeout = self.get_parameter("capture_timeout_sec").value
        tag = f"{stream['name']}][{index}"

        deadline = time.monotonic() + timeout
        while stream["latest_cloud"] is None and time.monotonic() < deadline:
            time.sleep(0.1)
        msg = stream["latest_cloud"]
        if msg is None:
            self.get_logger().error(
                f"[{tag}] No point cloud within {timeout:.1f}s on '{stream['topic']}'. Skipping "
                "save.")
            return False

        pts = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)
        if pts.size == 0:
            self.get_logger().warning(f"[{tag}] Captured cloud is empty; skipping save.")
            return False

        pose = self._lookup_sensor_pose(msg.header.frame_id, stream["sensor_frame"])
        if pose is not None:
            trans, quat, used_frame = pose
        elif fallback_pose is not None:
            trans, quat = fallback_pose
            used_frame = "plan_viewpoint(fallback)"
            self.get_logger().warning(
                f"[{tag}] TF sensor pose unavailable; using the planned viewpoint pose as origin.")
        else:
            self.get_logger().error(
                f"[{tag}] No TF and no fallback pose; saving cloud in its own frame, origin=0.")
            trans, quat, used_frame = np.zeros(3), np.array([0, 0, 0, 1.0]), msg.header.frame_id

        if self.get_parameter("transform_to_world").value and pose is not None:
            R = Rotation.from_quat(quat).as_matrix()
            pts = pts @ R.T + trans  # algılayıcı -> world
            cloud_frame = self.world_frame
        else:
            cloud_frame = msg.header.frame_id

        pcd_path = os.path.join(stream["pcd_dir"], f"{index}.pcd")
        txt_path = os.path.join(stream["pose_dir"], f"{index}.txt")
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(pcd_path, cloud)
        with open(txt_path, "w") as f:
            f.write(f"{trans[0]:.6f} {trans[1]:.6f} {trans[2]:.6f} "
                    f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")

        self.get_logger().info(
            f"[{tag}] Saved {len(pts)} points -> {pcd_path} (frame={cloud_frame}); "
            f"origin -> {txt_path} (from {used_frame}).")
        return True

    # ------------------------------------------------------------------ #
    # Küçük yardımcılar.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _wait_future(future: Future, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if future is not None and future.done():
                return True
            time.sleep(0.05)
        return False

    def _wait_joint_state(self, timeout: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.moveit.joint_state is not None:
                return True
            time.sleep(0.1)
        self.get_logger().error(
            f"[{self.arm_label}] joint states never arrived for joints {self.moveit.joint_names}.")
        return False

    @staticmethod
    def _subset(
            full_names: Sequence[str], full_positions: Sequence[float],
            wanted_names: Sequence[str]) -> Optional[List[float]]:
        lut = dict(zip(full_names, full_positions))
        try:
            return [float(lut[n]) for n in wanted_names]
        except KeyError:
            return None

    def _joint_states_cb(self, msg: JointState) -> None:
        for n, p in zip(msg.name, msg.position):
            self._joint_pos[n] = p
        st = msg.header.stamp
        self._joint_stamp = st.sec + st.nanosec * 1e-9

    def _wait_fresh_joint_state(
            self, label: str, max_age: float = 0.5, timeout: float = 5.0) -> bool:
        """
        En yeni /joint_states mesajı max_age'den genç olana kadar bekler.

        move_group bir execute_trajectory hedefini güncel durum izleyicisine göre doğrular
        ve en yeni durum 1 sn'den eskiyse reddeder ("couldn't receive full current joint
        state within 1s"); hedef denetleyici onu hiç görmeden İPTAL EDİLİR, yani dizi
        ilerlerken kol hiç hareket etmez. Biz de onun gördüğü mesajları görüyoruz; bu yüzden
        burada taze bir mesaj beklemek doğrudan bir denetimdir. Elde taze bir durum varsa
        True döner (False -> yine de gönder ve kararı denetleyiciye bırak; toleranslara
        göre asla engellemeyiz).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            stamp = self._joint_stamp
            if stamp is not None:
                age = self.get_clock().now().nanoseconds * 1e-9 - stamp
                if age < max_age:
                    return True
            time.sleep(0.05)
        self.get_logger().warning(
            f"[{label}] /joint_states is stale (>{max_age:.1f}s old after {timeout:.0f}s of "
            "waiting); move_group may abort this goal before execution.")
        return False

    # ------------------------------------------------------------------ #
    # Planlama.
    # ------------------------------------------------------------------ #
    def _plan(
            self, joint_positions: Sequence[float], joint_names: Sequence[str],
            label: str) -> Optional[JointTrajectory]:
        moveit2 = self.moveit
        attempts = max(1, int(self.get_parameter("plan_attempts").value))
        for attempt in range(1, attempts + 1):
            future = moveit2.plan_async(
                joint_positions=list(joint_positions), joint_names=list(joint_names),
                start_joint_state=moveit2.joint_state)
            if future is None:
                self.get_logger().warning(
                    f"[{label}] plan_async returned None (attempt {attempt}/{attempts}).")
            elif not self._wait_future(future, self._plan_timeout):
                self.get_logger().warning(
                    f"[{label}] planning timed out (attempt {attempt}/{attempts}).")
            else:
                traj = moveit2.get_trajectory(future)
                if traj is not None and traj.points:
                    if attempt > 1:
                        self.get_logger().info(
                            f"[{label}] planning succeeded on attempt {attempt}/{attempts}.")
                    return traj
                code = self._plan_error_code(future)
                self._last_plan_error = code
                self.get_logger().warning(
                    f"[{label}] planning failed / empty trajectory "
                    f"(attempt {attempt}/{attempts}) -- MoveIt reason: "
                    f"{self._moveit_err_name(code)}.")
        self.get_logger().error(
            f"[{label}] planning failed after {attempts} attempts "
            f"(last reason: {self._moveit_err_name(self._last_plan_error)}); "
            "skipping this viewpoint.")
        return None

    def _plan_pose(
            self, position: Sequence[float], quat_xyzw: Sequence[float], target_link: str,
            label: str) -> Optional[JointTrajectory]:
        """
        KARTEZYEN bir poz hedefine planlar (world'de konum+kuaterniyon üzerindeki target_link).

        UR için kullanılır; böylece MoveIt, yanlış yönelimli olabilecek saklı bir eklem
        çözümünü oynatmak yerine IK'yı istenen kamera pozuna göre yeniden çözer.
        """
        moveit2 = self.moveit
        attempts = max(1, int(self.get_parameter("plan_attempts").value))
        pos_tol = float(self.get_parameter("pose_goal_pos_tol").value)
        ori_tol = float(self.get_parameter("pose_goal_ori_tol").value)
        for attempt in range(1, attempts + 1):
            future = moveit2.plan_async(
                position=list(position), quat_xyzw=list(quat_xyzw),
                frame_id=self.world_frame, target_link=target_link,
                tolerance_position=pos_tol, tolerance_orientation=ori_tol,
                start_joint_state=moveit2.joint_state)
            if future is None:
                self.get_logger().warning(
                    f"[{label}] plan_async(pose) returned None (attempt {attempt}/{attempts}).")
            elif not self._wait_future(future, self._plan_timeout):
                self.get_logger().warning(
                    f"[{label}] pose planning timed out (attempt {attempt}/{attempts}).")
            else:
                traj = moveit2.get_trajectory(future)
                if traj is not None and traj.points:
                    if attempt > 1:
                        self.get_logger().info(
                            f"[{label}] pose planning succeeded on attempt {attempt}/{attempts}.")
                    return traj
                code = self._plan_error_code(future)
                self._last_plan_error = code
                self.get_logger().warning(
                    f"[{label}] pose planning failed / empty (attempt {attempt}/{attempts}) "
                    f"-- MoveIt reason: {self._moveit_err_name(code)}.")
        self.get_logger().error(
            f"[{label}] pose planning failed after {attempts} attempts "
            f"(last reason: {self._moveit_err_name(self._last_plan_error)}).")
        return None

    @staticmethod
    def _pose_goal_match(
            cached_pg: Optional[Sequence[float]], pg: Sequence[float], pos_tol: float = 0.01,
            ori_tol: float = 0.05) -> bool:
        """İki [px,py,pz,qx,qy,qz,qw] poz hedefi tolerans içinde uyuşuyorsa True döndürür."""
        if not cached_pg or len(cached_pg) != 7 or len(pg) != 7:
            return False
        if np.linalg.norm(np.array(cached_pg[:3]) - np.array(pg[:3])) > pos_tol:
            return False
        dot = abs(float(np.dot(np.array(cached_pg[3:]), np.array(pg[3:]))))
        ang = 2.0 * np.arccos(min(1.0, dot))
        return ang <= ori_tol

    @staticmethod
    def _is_prismatic(name: str) -> bool:
        # İki gruptaki doğrusal eksenler; açı olarak SARILMAMALIDIR.
        return any(k in name for k in
                   ("mount", "base_to_robot", "world_to_agv", "rail", "wheel"))

    def _normalize_goal(
            self, goal: Sequence[float], names: Sequence[str]) -> Tuple[List[float], bool]:
        """
        YALNIZCA saklı IK dalı +/-2*pi eklem sınırı kenarında duran dönel eklemleri kurtarır.

        Örneğin shoulder_pan = 6.283 = 2*pi ya da joint6 = 6.283; burada OMPL'nin hedef
        bölgesi sınırda çöker ve planlama çıplak bir FAILURE döndürür. Böyle bir eklem
        (-pi, pi] aralığına sarılır: AYNI fiziksel poz (theta ve theta +/- 2*pi aynıdır,
        çarpışmalar/yönelim değişmez) ama OMPL'nin örnekleyebileceği iç bölgenin derinlerinde.

        Diğer her eklem TAM OLARAK PLANLANDIĞI GİBİ BIRAKILIR: planlayıcı o açıları sınır
        içi IK olarak zaten doğruladı; onları yeniden sarmak (eski "en yakın güncel açı"
        kuralı) yalnızca sınır içindeki bir değeri (örneğin joint1 = 1.928 -> -4.355)
        sınırının DIŞINA itme riski taşır; Kawasaki'yi ilgisiz, yanlış yönelimli pozlara
        gönderen de buydu. (new_goal, changed) döndürür.
        """
        out = list(goal)
        changed = False
        two_pi = 2.0 * np.pi
        edge = float(self.get_parameter("normalize_edge_margin").value)
        for i, n in enumerate(names):
            if self._is_prismatic(n):
                continue
            w = float(goal[i])
            if abs(w) < two_pi - edge:
                continue  # rahatça iç bölgede -> güven
            nw = (w + np.pi) % two_pi - np.pi  # -> (-pi, pi], iç bölgenin derinleri
            if abs(nw - w) > 1e-6:
                out[i] = nw
                changed = True
        return out, changed

    # ------------------------------------------------------------------ #
    # 2*pi hedef sarma (sınırları gözeten).
    # ------------------------------------------------------------------ #
    def _robot_description_cb(self, msg: String) -> None:
        if self._urdf_xml is None:
            self._urdf_xml = msg.data

    def _wrap_enabled(self) -> bool:
        return bool(self.get_parameter("wrap_goals_to_current").value)

    def _goal_policy(self) -> str:
        """
        Hedeflerin nasıl seçildiğini tanımlayan etiket.

        Önbellekli yollar bunu taşır; böylece politikayı değiştirmek, eski seçimleri
        sessizce oynatmak yerine onları bir kez geçersiz kılar.
        """
        # `jbranch` eklem uzaylı kolları (Kawasaki) kapsar: eklem yolundaki dal aramasıyla
        # birlikte açıldı ve önbellekteki bir Kawasaki girdisi, o aramanın bilerek yerine
        # koyduğu saklı eklem hedefine göre anahtarlanır; bu etiket olmadan eski tek dallı
        # yollar sonsuza kadar oynatılırdı.
        jbranch = int(bool(self.get_parameter("nearest_branch_ik").value)
                      and self.ik_group_name is not None
                      and not self._use_pose_goal())
        return (f"unwind={int(self._wrap_enabled())},"
                f"branch={int(bool(self.get_parameter('nearest_branch_ik').value))},"
                f"jbranch={jbranch}")

    def _ensure_joint_limits(self, timeout: float = 5.0) -> Dict[str, JointLimit]:
        """
        Eklem sınırlarını URDF'ten bir kez ayrıştırır.

        /robot_description hiç gelmezse ya da ayrıştırılamazsa {} döner; boş bir tablo her
        sarmayı etkisiz kılar, yani eksik bir model kötü bir sarmaya değil, bugünkü
        davranışa düşer.
        """
        if self._joint_limits is not None:
            return self._joint_limits
        deadline = time.monotonic() + timeout
        while self._urdf_xml is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if self._urdf_xml is None:
            self.get_logger().warning(
                "No /robot_description received -- joint limits unknown, so 2*pi goal "
                "unwinding is DISABLED for this run (goals used exactly as planned).")
            self._joint_limits = {}
            return self._joint_limits
        try:
            self._joint_limits = parse_joint_limits(self._urdf_xml)
        except Exception as e:
            self.get_logger().warning(
                f"Could not parse /robot_description ({e}); 2*pi goal unwinding DISABLED.")
            self._joint_limits = {}
            return self._joint_limits
        group = [n for n in (self.moveit.joint_names if self.moveit else [])]
        wrappable = [n for n in group
                     if n in self._joint_limits and self._joint_limits[n].wrappable]
        self.get_logger().info(
            f"Joint limits parsed for {len(self._joint_limits)} joints; "
            f"{len(wrappable)}/{len(group)} of this arm's joints can be unwound: {wrappable}")
        return self._joint_limits

    def _wrap_goal_to_current(
            self, goal: Optional[Sequence[float]], names: Sequence[str],
            label: str = "") -> Tuple[Optional[List[float]], List[JointChange]]:
        """
        Her dönel hedefi kolun güncel ölçülen açısına EN YAKIN 2*pi eşdeğeriyle yeniden ifade eder.

        Eşdeğer, o eklemin URDF sınırları içinde kalmalıdır. theta ve theta +/- 2*pi aynı
        fiziksel konfigürasyondur; kol kısa yoldan dönerek aynı poza (aynı çarpışmalar, aynı
        kamera yönelimi) varır.

        (new_goal, changes) döndürür; hiçbir şey yeniden yazılmadıysa (ya da
        yazılamadıysa) changes boştur ve çağıran özgün hedefi kullanmalıdır.
        """
        if goal is None or not self._wrap_enabled():
            return (list(goal) if goal is not None else None), []
        limits = self._ensure_joint_limits()
        if not limits:
            return list(goal), []
        reference = dict(self._joint_pos)
        if not reference:
            return list(goal), []
        new_goal, changes = wrap_to_reference(
            goal, list(names), reference, limits,
            margin=float(self.get_parameter("wrap_limit_margin").value),
            min_gain=float(self.get_parameter("wrap_min_gain").value))
        if changes:
            self.get_logger().info(f"[{label}] unwinding goal: {describe_changes(changes)}")
        return new_goal, changes

    # ------------------------------------------------------------------ #
    # En yakın dal IK.
    # ------------------------------------------------------------------ #
    def _current_joint_state(self) -> Optional[JointState]:
        """Kolun ölçülen pozunu IK tohumu olarak kullanmak için JointState biçiminde döndürür."""
        pos = dict(self._joint_pos)
        js = JointState()
        js.name = list(pos.keys())
        js.position = [float(v) for v in pos.values()]
        return js if js.name else None

    def _solve_ik(
            self, position: Sequence[float], quat_xyzw: Sequence[float], seed: JointState,
            timeout: float, ik_link: Optional[str] = None,
            group: Optional[str] = None) -> Optional[JointState]:
        """
        Tohumlu tek bir /compute_ik çağrısı yapar; çözüm JointState'ini ya da None döndürür.

        Grubun varsayılan ucu yerine çağıranın `ik_link`'i için çözmek, hedef poza bilek
        flanşının değil KAMERANIN oturmasını sağlar.

        `group` bu çözüm için planlama grubunu geçersiz kılar (dal araması küçültülmüş bir
        grup kullanır; bkz. _active_branch_group). Sonuç `self._last_ik_error` içinde
        bırakılır; böylece çağıran, reddedilen bir GRUP ADINI çözümü olmayan bir pozdan
        ayırt edebilir.
        """
        self._last_ik_error = None
        if self.ik_group_name is None or not self._ik_cli.service_is_ready():
            return None
        link = ik_link or self._pose_target_link()
        if link is None:
            return None
        req = GetPositionIK.Request()
        req.ik_request.group_name = group or self.ik_group_name
        req.ik_request.ik_link_name = link
        req.ik_request.avoid_collisions = bool(
            self.get_parameter("ik_avoid_collisions").value)
        req.ik_request.timeout.sec = int(timeout)
        req.ik_request.timeout.nanosec = int((timeout - int(timeout)) * 1e9)
        if seed is not None:
            req.ik_request.robot_state.joint_state = seed
        ps = PoseStamped()
        ps.header.frame_id = self.world_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (
            float(position[0]), float(position[1]), float(position[2]))
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = (
            float(quat_xyzw[0]), float(quat_xyzw[1]),
            float(quat_xyzw[2]), float(quat_xyzw[3]))
        req.ik_request.pose_stamped = ps

        future = self._ik_cli.call_async(req)
        if not self._wait_future(future, max(2.0, timeout * 4.0)):
            return None
        res = future.result()
        if res is None:
            return None
        self._last_ik_error = res.error_code.val
        if res.error_code.val != res.error_code.SUCCESS:
            return None
        return res.solution.joint_state

    def _active_branch_group(self) -> Optional[str]:
        """
        Dal aramasının çözdüğü grup.

        move_group kabul ediyorsa `branch_ik_group`, değilse planlama grubu.

        Bu yedek bir evham değildir. Bu SRDF'ler git'te DEĞİLDİR ve daha önce sessizce
        üzerine yazılmışlardır (kinematics.yaml dosyalarının KDL'ye dönmesi böyle oldu);
        bu yüzden küçültülmüş grubu eksik bir yapılandırmaya karşı yapılan bir koşu yalnızca
        ray güvencesini kaybetmelidir, dal aramasının tamamını değil.
        """
        if not self._branch_ik_group_ok:
            return self.ik_group_name
        name = (self.get_parameter("branch_ik_group").value or "").strip()
        # "default", "bu kolun DEFAULT_BRANCH_IK_GROUP değeri ne diyorsa" demektir.
        # Başlatma dosyası düz bir grup adı yerine bu işareti geçirir; böylece iki kolun
        # farklı varsayılanları başlatma argümanlarında da tekrarlanıp zamanla ayrışmak
        # yerine TEK bir yerde (düğüm sınıflarında) yaşar.
        if name == "default":
            name = (self.DEFAULT_BRANCH_IK_GROUP or "").strip()
        # "none" boş metinle aynıdır: planlama grubunda çöz. Var olma sebebi,
        # `ros2 launch`'ın boş bir argüman değerini doğrudan reddetmesidir
        # ("malformed launch argument"); yoksa komut satırından eski davranışı
        # istemenin bir yolu olmazdı.
        if name == "none":
            name = ""
        return name or self.ik_group_name

    def _prismatic_names(self, names: Sequence[str]) -> List[str]:
        """
        `names` arasındaki doğrusal eksenler (AGV rayı / UR rayı).

        Model biliniyorsa URDF tipine, bilinmiyorsa adlandırma geleneğine göre belirlenir.
        """
        limits = self._joint_limits or {}
        out = []
        for n in names:
            lim = limits.get(n)
            if (not lim.is_revolute) if lim is not None else self._is_prismatic(n):
                out.append(n)
        return out

    def _branch_candidates(
            self, position: Sequence[float], quat_xyzw: Sequence[float], label: str,
            ik_link: Optional[str] = None,
            planned_goal: Optional[Sequence[float]] = None) -> List[Tuple[List[float], float]]:
        """
        Bu poza ulaşan eklem hedeflerini gereken hareket miktarına göre sıralar.

        Sıralama, kolun şu anki yerinden ne kadar az hareketle varılacağına göredir.

        Farklı bir dala düşmenin iki yolu vardır. Sorguyu tekrarlamak yalnızca çözücü
        rastgele yeniden başlıyorsa bir şey örnekler; bu `mode: global` ile pick_ik (UR)
        için doğrudur, her seferinde aynı yanıtı veren KDL (Kawasaki grubu) için DEĞİL.
        Bu yüzden TOHUM da değiştirilir: ölçülen poz, planlanan hedef, rayı iki yöne
        dürtülmüş planlanan hedef ve `ik_random_seeds` adet rastgele kol duruşu; yerel bir
        çözücüyü planlanan çözümden asıl kurtaran sonuncusudur.

        Kol KÜÇÜLTÜLMÜŞ bir grup (`branch_ik_group`) gösterdiğinde arama orada yapılır:
        Kawasaki'nin rayı o grubun dışındadır; bu yüzden çözüm sırasında kayamaz ve bulunan
        her dal planlayıcının seçtiği AGV konumunda kullanılabilir. Ray girişte (kol tabanının
        nerede OLDUĞUNU belirler; çözüm yalnızca planlanan konumda anlamlıdır) ve çıkışta o
        planlanan değere sabitlenir.

        Her sonuç önce sarılır; böylece bir dal kazara oluşan tam tur farkıyla yargılanmaz;
        sonra güncel pozdan yola göre puanlanır.
        [(goal, travel_rad), ...] döndürür, en yakını başta.
        """
        if not self.get_parameter("nearest_branch_ik").value or self.ik_group_name is None:
            return []
        seed = self._current_joint_state()
        if seed is None:
            return []
        names = list(self.moveit.joint_names)
        timeout = float(self.get_parameter("ik_timeout").value)
        self._ensure_joint_limits()

        group = self._active_branch_group()
        rail_lock = None
        if group != self.ik_group_name and planned_goal is not None:
            pris = set(self._prismatic_names(names))
            rail_lock = {n: float(planned_goal[i])
                         for i, n in enumerate(names) if n in pris}

        out = []
        seen = set()
        for s, repeats in self._ik_seeds(seed, planned_goal, names, rail_lock):
            for _ in range(repeats):
                sol = self._solve_ik(position, quat_xyzw, s, timeout, ik_link=ik_link,
                                     group=group)
                if sol is None:
                    if (self._last_ik_error == -15  # INVALID_GROUP_NAME
                            and group != self.ik_group_name):
                        self.get_logger().error(
                            f"[{label}] move_group does not know the branch IK group "
                            f"'{group}' -- is it missing from this config's SRDF? "
                            f"Falling back to '{self.ik_group_name}' for the rest of "
                            f"this run; branches that move the AGV rail will be "
                            f"rejected again as they were before.")
                        self._branch_ik_group_ok = False
                        return self._branch_candidates(
                            position, quat_xyzw, label, ik_link=ik_link,
                            planned_goal=planned_goal)
                    continue
                goal = self._subset(list(sol.name), list(sol.position), names)
                if goal is None and planned_goal is not None:
                    # Küçültülmüş bir grup yalnızca kendi eklemleriyle yanıt verebilir; hedefin
                    # geri kalanı planındır; ray için tam doğrusu budur.
                    lut = dict(zip(sol.name, sol.position))
                    goal = [float(lut.get(n, planned_goal[i]))
                            for i, n in enumerate(names)]
                if goal is None:
                    continue
                if rail_lock:
                    for i, n in enumerate(names):
                        if n in rail_lock:
                            goal[i] = rail_lock[n]
                goal, _ = self._wrap_goal_to_current(goal, names)
                if not self._rail_shift_ok(goal, planned_goal, names, label):
                    continue
                key = tuple(round(v, 3) for v in goal)
                if key in seen:
                    continue
                seen.add(key)
                out.append((goal, self._goal_travel(goal, names)))
        out.sort(key=lambda t: t[1])
        if out:
            where = "" if group == self.ik_group_name else f" in '{group}'"
            self.get_logger().info(
                f"[{label}] nearest-branch IK{where}: {len(out)} distinct branch(es), "
                f"travel from current pose "
                f"{', '.join(f'{np.rad2deg(t):.0f}' for _, t in out)} deg.")
        return out

    def _ik_seeds(
            self, measured: JointState, planned_goal: Optional[Sequence[float]],
            names: Sequence[str],
            rail_lock: Optional[Dict[str, float]] = None) -> List[Tuple[JointState, int]]:
        """
        Dal araması için tohumlar: (durum, tekrar) çiftleri, en olası başta.

        Kolun gerçekte bulunduğu yer, planlanan hedef, rayı iki yöne denenmiş planlanan
        hedef, ardından `ik_random_seeds` adet rastgele kol duruşu.

        Tekrarlar RASTGELE YENİDEN BAŞLAYAN bir çözücüyü (UR'nin `mode: global` ile
        pick_ik'sı) örneklemek için vardır; rastgele bir tohum tam olarak bir kez denenir,
        çünkü yerel bir çözücüyü aynı tohumdan yeniden çalıştırmak yalnızca aynı yanıtı
        döndürebilir.

        `rail_lock` ({eklem: planlanan değer}) her doğrusal eklemi her tohumda sabitler.
        Arama rayı dışlayan küçültülmüş bir grupta yapıldığında ayarlanır: o zaman tohumun
        ray değeri çözüm sırasında kol tabanının fiziksel olarak nerede durduğunu belirler;
        bu yüzden planlanan değer olmalıdır, yoksa dönen kol açıları gönderildiğinde kamerayı
        başka bir yere yöneltirdi. Bu, aşağıdaki ray denemelerini de anlamsız kılar (hepsi
        koşunun asla bulunmayacağı bir taban konumunu tarif ederdi); bu yüzden atlanırlar.
        """
        attempts = max(1, int(self.get_parameter("ik_seed_attempts").value))

        def as_state(values: Sequence[float]) -> JointState:
            js = JointState()
            js.name = list(names)
            js.position = [float(v) for v in values]
            return js

        def locked(values: Sequence[float]) -> List[float]:
            vals = [float(v) for v in values]
            if rail_lock:
                for i, n in enumerate(names):
                    if n in rail_lock:
                        vals[i] = rail_lock[n]
            return vals

        if rail_lock:
            m = self._subset(list(measured.name), list(measured.position), names)
            if m is not None:
                measured = as_state(locked(m))
        seeds = [(measured, attempts)]
        if planned_goal is None:
            return seeds

        seeds.append((as_state(locked(planned_goal)), attempts))
        probe = float(self.get_parameter("branch_rail_probe").value)
        limits = self._joint_limits or {}
        if probe > 0.0 and not rail_lock:
            for i, n in enumerate(names):
                lim = limits.get(n)
                if lim is None or lim.is_revolute:
                    continue
                for d in (probe, -probe):
                    v = float(np.clip(planned_goal[i] + d,
                                      lim.lower + 1e-6, lim.upper - 1e-6))
                    if abs(v - planned_goal[i]) < 1e-4:
                        continue
                    vals = list(planned_goal)
                    vals[i] = v
                    seeds.append((as_state(vals), attempts))
        seeds.extend(self._random_arm_seeds(planned_goal, names, as_state, locked))
        return seeds

    def _random_arm_seeds(
            self, planned_goal: Optional[Sequence[float]], names: Sequence[str],
            as_state: Callable[[Sequence[float]], JointState],
            locked: Callable[[Sequence[float]], List[float]]) -> List[Tuple[JointState, int]]:
        """
        DÖNEL eklemleri URDF sınırlarından düzgün dağılımla çekilen `ik_random_seeds` adet tohum.

        Diğer her eklem planlanan değerde bırakılır.

        Dal aramasını yerel bir çözücüde çalıştıran parça budur. Doğrusal eksenler asla
        çekilmez: planlayıcı ray konumunu kaplama için seçti ve Kawasaki'de farklı bir rayda
        duran tohum, kolun bulunmayacağı bir taban konumunu tarif eder. Sürekli (continuous)
        bir eklemin çekilecek sınırı yoktur; bu yüzden tam bir tur üzerinden örneklenir.
        """
        n_rand = int(self.get_parameter("ik_random_seeds").value)
        if n_rand < 0:  # işaret: bkz. _active_branch_group()
            n_rand = int(self.DEFAULT_IK_RANDOM_SEEDS)
        if n_rand <= 0:
            return []
        limits = self._ensure_joint_limits()
        if not limits:
            self.get_logger().warning(
                "ik_random_seeds is set but no joint limits are known (no "
                "/robot_description) -- random branch seeds are DISABLED for this run.")
            return []
        if self._ik_rng is None:
            self._ik_rng = np.random.default_rng(
                int(self.get_parameter("ik_random_seed").value))
        draw = []
        for i, n in enumerate(names):
            lim = limits.get(n)
            if lim is None or not lim.is_revolute:
                continue
            lo, hi = float(lim.lower), float(lim.upper)
            if not (np.isfinite(lo) and np.isfinite(hi)):
                lo, hi = -np.pi, np.pi  # sürekli (continuous) eklem
            if hi - lo < 1e-6:
                continue
            draw.append((i, lo, hi))
        if not draw:
            return []
        out = []
        for _ in range(n_rand):
            vals = list(planned_goal)
            for i, lo, hi in draw:
                vals[i] = float(self._ik_rng.uniform(lo, hi))
            out.append((as_state(locked(vals)), 1))
        return out

    def _rail_shift_ok(
            self, goal: Sequence[float], planned_goal: Optional[Sequence[float]],
            names: Sequence[str], label: str) -> bool:
        """
        Planın hiç amaçlamadığı bir yere AGV'yi gönderecek dalı reddeder.

        Ray doğrusaldır; bu yüzden `_goal_travel` onu maliyete katmaz; bu koruma olmadan bir
        aday AGV'yi sessizce metrelerce sürmeyi önerebilirdi. `branch_max_rail_shift` <= 0
        rayı tam olarak planlanan değere kilitler; varsayılan budur, çünkü planlayıcı o AGV
        konumunu kaplama sebepleriyle seçti.
        """
        if planned_goal is None:
            return True
        tol = float(self.get_parameter("branch_max_rail_shift").value)
        limits = self._joint_limits or {}
        for i, n in enumerate(names):
            lim = limits.get(n)
            is_pris = (not lim.is_revolute) if lim is not None else self._is_prismatic(n)
            if not is_pris:
                continue
            d = abs(float(goal[i]) - float(planned_goal[i]))
            if d > max(tol, 1e-3):
                self.get_logger().debug(
                    f"[{label}] branch rejected: {n} would move {d:.3f} m "
                    f"(limit {tol:.3f}).")
                return False
        return True

    def _goal_travel(self, goal: Sequence[float], names: Sequence[str]) -> float:
        """Kolun güncel pozundan `goal`'a dönel eklem yolu (radyan)."""
        pos = dict(self._joint_pos)
        limits = self._joint_limits or {}
        total = 0.0
        for i, n in enumerate(names):
            if n not in pos:
                continue
            if (limits[n].is_revolute if n in limits else not self._is_prismatic(n)):
                total += abs(float(goal[i]) - float(pos[n]))
        return total

    def _traj_travel(self, traj: JointTrajectory) -> float:
        """
        Bir yörünge boyunca dönel eklemlerin toplam yolu (radyan).

        Sarılmış bir yeniden planın özgün yoldan gerçekten daha kısa olup olmadığına karar
        vermek için kullanılır.
        """
        if traj is None or len(traj.points) < 2:
            return 0.0
        limits = self._joint_limits or {}
        keep = [i for i, n in enumerate(traj.joint_names)
                if (limits[n].is_revolute if n in limits else not self._is_prismatic(n))]
        total = 0.0
        prev = traj.points[0].positions
        for p in traj.points[1:]:
            total += sum(abs(float(p.positions[i]) - float(prev[i])) for i in keep)
            prev = p.positions
        return total

    def _shorten_wound_up(
            self, traj: JointTrajectory, goal: List[float], joint_names: Sequence[str],
            label: str) -> Tuple[JointTrajectory, List[float]]:
        """
        Planlanmış bir yörüngenin bitiş konfigürasyonu sarılmış bir dalsa daha kısa olanı seçer.

        Öyleyse en yakın 2*pi eşdeğerine (AYNI poz) ikinci bir yol planlar ve gerçekten daha
        az yol alan yörüngeyi tutar. Sarmayı, bakış noktaları Kartezyen bir poz hedefinden
        planlanan UR'ye getiren budur: MoveIt orada IK'yı yeniden çözer ve herhangi bir dalı
        döndürebilir; bu yüzden düzeltme yalnızca saklı eklem değerlerinde yaşayamaz.

        (trajectory, end_configuration) döndürür; hiçbir şey işe yaramazsa özgün çift döner.
        """
        wrapped, changes = self._wrap_goal_to_current(goal, joint_names, label)
        if not changes:
            return traj, goal
        alt = self._plan(wrapped, joint_names, f"{label} [unwound]")
        if alt is None:
            self.get_logger().info(
                f"[{label}] unwound goal did not plan; keeping the original path.")
            return traj, goal
        orig_travel, alt_travel = self._traj_travel(traj), self._traj_travel(alt)
        if alt_travel >= orig_travel:
            self.get_logger().info(
                f"[{label}] unwound path is not shorter "
                f"({np.rad2deg(alt_travel):.0f} vs {np.rad2deg(orig_travel):.0f} deg); "
                "keeping the original.")
            return traj, goal
        self.get_logger().info(
            f"[{label}] using UNWOUND path: {np.rad2deg(orig_travel):.0f} -> "
            f"{np.rad2deg(alt_travel):.0f} deg of joint travel "
            f"({100.0 * (1.0 - alt_travel / max(orig_travel, 1e-9)):.0f}% less).")
        return alt, wrapped

    def _plan_best_branch(
            self, candidates: List[Tuple[List[float], float]], joint_names: Sequence[str],
            label: str) -> Tuple[Optional[JointTrajectory], Optional[List[float]]]:
        """
        Sıralanmış dal adaylarını planlar ve gerçekten en az yol alanı tutar.

        `candidates` HEDEF uzayındaki uzaklığa göre sıralı gelir; ama bu, OMPL'nin engellerin
        etrafından dolaştırabileceği bir yol için düz çizgi yaklaşıklığıdır: kapı koşusunda
        ölçüldü, Kawasaki'nin kayıtlı yolları doğrudan eklem uzaklıklarının 1.54 katını kat
        eder (biri 3.5 katını); yani en yakın hedef güvenilir biçimde en kısa yolculuk değildir.
        Bu fonksiyon başarılı olanlardan en fazla `branch_plan_candidates` tanesini planlar
        ve _traj_travel'a göre seçer.

        Planlanamayan adaylar bütçeyi TÜKETMEZ; bu yüzden ilk tercihleri erişilemez olan bir
        kol yine listeyi eskisi gibi işler. `branch_plan_candidates: 1` eski "planlanabilen
        ilki al" davranışıdır.

        (traj, goal) ya da (None, None) döndürür.
        """
        budget = max(1, int(self.get_parameter("branch_plan_candidates").value))
        best_traj = best_goal = None
        best_travel = np.inf
        planned = 0
        for cand_goal, goal_travel in candidates:
            traj = self._plan(
                cand_goal, joint_names,
                f"{label} [branch {np.rad2deg(goal_travel):.0f} deg]")
            if traj is None:
                continue
            planned += 1
            travel = self._traj_travel(traj)
            if travel < best_travel:
                best_traj, best_goal, best_travel = traj, cand_goal, travel
            if planned >= budget:
                break
        if best_traj is not None and planned > 1:
            self.get_logger().info(
                f"[{label}] compared {planned} planned branches; kept the one travelling "
                f"{np.rad2deg(best_travel):.0f} deg.")
        return best_traj, best_goal

    @staticmethod
    def _moveit_err_name(code: Optional[int]) -> str:
        return {
            1: "SUCCESS", 99999: "FAILURE", -1: "PLANNING_FAILED",
            -2: "INVALID_MOTION_PLAN",
            -3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
            -4: "CONTROL_FAILED", -5: "UNABLE_TO_AQUIRE_SENSOR_DATA",
            -6: "TIMED_OUT", -7: "PREEMPTED",
            -10: "START_STATE_IN_COLLISION",
            -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
            -12: "GOAL_IN_COLLISION", -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
            -14: "GOAL_CONSTRAINTS_VIOLATED", -15: "INVALID_GROUP_NAME",
            -16: "INVALID_GOAL_CONSTRAINTS", -17: "INVALID_ROBOT_STATE",
            -18: "INVALID_LINK_NAME", -19: "INVALID_OBJECT_NAME",
            -21: "FRAME_TRANSFORM_FAILURE", -22: "COLLISION_CHECKING_UNAVAILABLE",
            -23: "ROBOT_STATE_STALE", -24: "SENSOR_INFO_STALE",
            -25: "COMMUNICATION_FAILURE", -31: "NO_IK_SOLUTION",
        }.get(code, f"UNKNOWN({code})")

    @staticmethod
    def _plan_error_code(future: Future) -> Optional[int]:
        try:
            res = future.result()
            mpr = getattr(res, "motion_plan_response", None)
            if mpr is not None:
                return mpr.error_code.val
            ec = getattr(res, "error_code", None)
            return ec.val if ec is not None else None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Gönderim + tamamlanma.
    # ------------------------------------------------------------------ #
    def _dispatch(self, traj: JointTrajectory, label: str) -> Future:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        return self.ctrl.send_goal_async(goal)

    def _await(self, send_future: Future, label: str) -> bool:
        if send_future is None:
            return False
        if not self._wait_future(send_future, 5.0):
            self.get_logger().error(f"[{label}] controller did not accept the goal in time.")
            return False
        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().error(f"[{label}] goal REJECTED by controller.")
            return False
        result_future = handle.get_result_async()
        if not self._wait_future(result_future, self._motion_timeout):
            self.get_logger().error(f"[{label}] motion timed out.")
            return False
        wrapped = result_future.result()
        status = wrapped.status
        if status != GoalStatus.STATUS_SUCCEEDED:
            res = getattr(wrapped, "result", None)
            err = getattr(res, "error_code", None)
            errstr = getattr(res, "error_string", "") or ""
            self.get_logger().warning(
                f"[{label}] motion did not succeed (status={status}, "
                f"error_code={err} {self._ftj_err_name(err)}"
                + (f", '{errstr}'" if errstr else "") + ").")
            return False
        return True

    @classmethod
    def _ftj_err_name(cls, code: Union[int, MoveItErrorCodes]) -> str:
        """
        Başarısız bir eylemin hata kodunu adlandırır.

        _await İKİ sonuç tipine de hizmet eder: Kawasaki'nin ham FollowJointTrajectory'si
        (error_code düz bir int) ve UR'nin move_group ExecuteTrajectory'si (error_code
        özetlenemeyen (unhashable) bir MoveItErrorCodes MESAJI; onunla bir sözlüğü
        indekslemek TypeError fırlatır ve tam da hatayı RAPORLAMAK için yazılmış satırda
        düğümü öldürüyordu). Hangisi geldiyse ona göre davranılır.
        """
        if hasattr(code, "val"):
            return cls._moveit_err_name(code.val)
        return {
            0: "SUCCESSFUL", -1: "INVALID_GOAL", -2: "INVALID_JOINTS",
            -3: "OLD_HEADER_TIMESTAMP", -4: "PATH_TOLERANCE_VIOLATED",
            -5: "GOAL_TOLERANCE_VIOLATED",
        }.get(code, "UNKNOWN")

    def _confirm_accepted(self, send_future: Future, label: str) -> bool:
        """
        Denetleyicinin hedefi KABUL ettiğini doğrular.

        Ardından eylem sonucuna değil, ölçülen varışa göre ilerlenir. Kabul edildiyse True döner.
        """
        if send_future is None:
            return False
        if not self._wait_future(send_future, 5.0):
            self.get_logger().error(f"[{label}] controller did not accept the goal in time.")
            return False
        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().error(f"[{label}] goal REJECTED by controller.")
            return False
        return True

    # ------------------------------------------------------------------ #
    # Ölçülen varış denetimi (Kawasaki alt sınıfı kullanır).
    # ------------------------------------------------------------------ #
    def _arrival_targets(self, vp: Viewpoint) -> Dict[str, float]:
        """Bu kolun grup eklemlerinin vp'deki hedefleri: {eklem_adı: hedef_konum}."""
        pos = self._subset(vp["joint_names"], vp["joint_positions"], self.moveit.joint_names)
        if pos is None:
            return {}
        return dict(zip(self.moveit.joint_names, pos))

    def _arrival_targets_from_traj(self, traj: JointTrajectory, vp: Viewpoint) -> Dict[str, float]:
        """
        Varış hedefleri: yörüngenin GERÇEK son komut eklem konumları.

        Ham plan joint_positions değerleri DEĞİL. Bu önemlidir, çünkü yürüttüğümüz hedef
        açı olarak normalleştirilmiş olabilir (sarılmış bir 2*pi eklem iç bölgeye alınmış):
        o zaman ham plan değerine göre denetlemek, robota hiç komut verilmemiş bir açıyla
        karşılaştırma yapar (örneğin joint6 hedefi 6.283, komut 0 -> hiç kapanmayan kalıcı
        bir err=-6.283). Yörüngenin TAŞIMADIĞI her grup eklemi (örneğin planlanan bir eklem
        değilse asenkron AGV rayı) ham plan hedefine düşer; böylece varışı yine denetlenir.
        """
        final = {}
        if traj is not None and traj.points:
            final = dict(zip(traj.joint_names, traj.points[-1].positions))
        raw = self._arrival_targets(vp)
        out = {}
        for n in self.moveit.joint_names:
            if n in final:
                out[n] = float(final[n])
            elif n in raw:
                out[n] = float(raw[n])
        return out

    def _arrival_tol_for(self, name: str) -> float:
        """
        Doğrusal eklemler (AGV rayı / UR taban rayı) metre toleransı kullanır.

        Diğer her (dönel) eklem radyan toleransını kullanır.
        """
        if "world_to_agv" in name or "base_to_robot_mount" in name:
            return float(self.get_parameter("arrival_linear_tol").value)
        return float(self.get_parameter("arrival_joint_tol").value)

    def _log_arrival(
            self, tag: str, targets: Dict[str, float], tol: Dict[str, float],
            timed_out: bool = False) -> None:
        pos = dict(self._joint_pos)
        parts = []
        for n, t in targets.items():
            if n in pos:
                err = pos[n] - t
                mark = "" if abs(err) <= tol[n] else "   <-- NOT reached"
                parts.append(f"{n}: err={err:+.3f} (tol {tol[n]:.3f}){mark}")
            else:
                parts.append(f"{n}: NO STATE in /joint_states   <-- missing feedback")
        body = "\n  ".join(parts)
        if timed_out:
            self.get_logger().warning(f"{tag} arrival TIMEOUT -- capturing anyway:\n  {body}")
        else:
            self.get_logger().info(f"{tag} waiting for arrival:\n  {body}")

    def _wait_until_reached(self, targets: Dict[str, float], tag: str) -> bool:
        """
        Komut verilen her eklem hedefinin toleransı içine girene kadar /joint_states'i yoklar.

        Ya da arrival_timeout_sec dolana kadar. Varıldıysa True döner.
        """
        timeout = float(self.get_parameter("arrival_timeout_sec").value)
        tol = {n: self._arrival_tol_for(n) for n in targets}
        deadline = time.monotonic() + timeout
        last_log = 0.0
        while time.monotonic() < deadline:
            pos = dict(self._joint_pos)
            if all(n in pos for n in targets):
                if all(abs(pos[n] - targets[n]) <= tol[n] for n in targets):
                    self.get_logger().info(f"{tag} reached target (all joints within tolerance).")
                    return True
            now = time.monotonic()
            if now - last_log >= 5.0:
                last_log = now
                self._log_arrival(tag, targets, tol)
            time.sleep(0.1)
        self._log_arrival(tag, targets, tol, timed_out=True)
        return False

    @staticmethod
    def _fallback_pose(vp: Viewpoint) -> Tuple[np.ndarray, np.ndarray]:
        return (np.array(vp["position"], dtype=float),
                Rotation.from_matrix(np.array(vp["rotation"], dtype=float)).as_quat())

    # ------------------------------------------------------------------ #
    # Yörünge önbelleği: geçerliyse kayıtlı yolu oynat, değilse planla + kaydet.
    # ------------------------------------------------------------------ #
    def _start_tol_for(self, name: str) -> float:
        if "world_to_agv" in name or "base_to_robot_mount" in name:
            return float(self.get_parameter("traj_start_tol_m").value)
        return float(self.get_parameter("traj_start_tol_rad").value)

    def _traj_cache_path(self, vp_id: str) -> str:
        return os.path.join(self._traj_dir, f"{self.robot_tag}_{vp_id}.json")

    def _goal_close(
            self, joint_names: Sequence[str], a: Sequence[float], b: Sequence[float]) -> bool:
        """
        İki eklem hedefi aynı kol pozunu tarif ediyorsa True döndürür.

        Dönel eklemler 2*pi MODÜLÜNDE karşılaştırılır: theta ve theta +/- 2*pi aynı fiziksel
        konfigürasyondur ve yürütücü, planın turunun uzak ucunda sakladığı bir hedef için
        (joint6 = +5.85) çoğu zaman sarılmış/dal seçilmiş değeri (joint6 = -0.43) saklar. Bu
        ham sayıları karşılaştırmak her koşuda bütün Kawasaki önbelleğini eskimiş ilan ederdi;
        kol kayıtlı yolları oynatmak yerine 8 bakış noktasının hepsini yeniden planlardı.
        Doğrusal eksenler (AGV rayı) olduğu gibi karşılaştırılır: bir metre bir metredir.
        """
        if a is None or b is None or len(a) != len(b) or len(a) != len(joint_names):
            return False
        two_pi = 2.0 * np.pi
        for n, x, y in zip(joint_names, a, b):
            d = abs(float(x) - float(y))
            if not self._is_prismatic(n):
                d = min(d % two_pi, two_pi - (d % two_pi))
            if d > self._start_tol_for(n):
                return False
        return True

    def _cache_valid(
            self, cached: Dict[str, Any], joint_names: Sequence[str], goal: Sequence[float],
            pose_goal: Optional[Sequence[float]] = None) -> bool:
        """
        Önbellekteki bir yolun güncel plan için geçerli olup olmadığını döndürür.

        Yol yalnızca güncel planla AYNI eklemleri ve AYNI hedefi hedefliyorsa geçerlidir.

        Poz kipinde önbellek eşleşen bir pose_goal taşımalıdır.
        """
        if list(cached.get("joint_names", [])) != list(joint_names):
            return False
        # Bu denetimdeki başka hiçbir şey, bir hedefi SEÇME biçimimizin değiştiğini fark
        # edemez: bir UR önbelleği kendi pose_goal'üne göre anahtarlanır ve ne sarma ne de
        # en yakın dal IK'sı pozu değiştirir; yalnızca ona hangi konfigürasyonun ulaştığını
        # değiştirir. Yani eski bir politikayla kaydedilmiş bir yol uzun yolunu sonsuza kadar
        # oynatırdı. Politikayı damgalamak bu yolların tam olarak bir kez yeniden planlanmasını
        # sağlar.
        if cached.get("goal_policy") != self._goal_policy():
            self.get_logger().info(
                f"[{self.arm_label}] cached path for '{cached.get('vp_id')}' was recorded "
                "under a different goal-selection policy; replanning it once.")
            return False
        if pose_goal is not None:
            return self._pose_goal_match(cached.get("pose_goal"), pose_goal)
        # DOĞRU önbellek anahtarı, yürütücünün sonunda verdiği komut değil, PLANIN istediği
        # hedeftir: dal araması ve sarıcı, saklı hedefi bilerek aynı kamera pozunun daha
        # ucuz bir konfigürasyonuyla değiştirir; yani yürütülene göre anahtarlamak, bir yolun
        # kaydedildiği anda kendini geçersiz kılmasına yol açar.
        # `planned_goal` kayıt anında damgalanır; bu alanı olmayan eski önbellekler
        # yürütülen hedefe düşer; 2*pi toleranslı karşılaştırma onu da kabul eder.
        planned = cached.get("planned_goal")
        if planned:
            return self._goal_close(joint_names, planned, goal)
        return self._goal_close(joint_names, cached.get("goal_positions"), goal)

    def _serialize_traj(
            self, jt: JointTrajectory, vp_id: str, goal: Sequence[float],
            pose_goal: Optional[Sequence[float]] = None,
            planned_goal: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        pts = []
        for p in jt.points:
            t = p.time_from_start
            pts.append({
                "positions": [float(x) for x in p.positions],
                "velocities": [float(x) for x in p.velocities],
                "accelerations": [float(x) for x in p.accelerations],
                "time_from_start": t.sec + t.nanosec * 1e-9,
            })
        out = {
            "arm": self.robot_tag,
            "vp_id": vp_id,
            "joint_names": list(jt.joint_names),
            "goal_positions": [float(x) for x in goal] if goal is not None else [],
            "start_positions": [float(x) for x in jt.points[0].positions] if jt.points else [],
            "points": pts,
            # Bu yolun hedefi nasıl seçildi (bkz. _cache_valid).
            "goal_policy": self._goal_policy(),
        }
        if pose_goal is not None:
            out["pose_goal"] = [float(x) for x in pose_goal]
        if planned_goal is not None:
            # PLANIN istediği, yani bu önbellek girdisinin anahtarı.
            out["planned_goal"] = [float(x) for x in planned_goal]
        return out

    @staticmethod
    def _deserialize_traj(cached: Dict[str, Any]) -> JointTrajectory:
        jt = JointTrajectory()
        jt.joint_names = list(cached["joint_names"])
        for pd in cached["points"]:
            p = JointTrajectoryPoint()
            p.positions = [float(x) for x in pd.get("positions", [])]
            p.velocities = [float(x) for x in pd.get("velocities", [])]
            p.accelerations = [float(x) for x in pd.get("accelerations", [])]
            t = float(pd.get("time_from_start", 0.0))
            p.time_from_start = Duration(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))
            jt.points.append(p)
        return jt

    def _get_trajectory(
            self, vp: Viewpoint, label: str, idx: int,
            force_fresh: bool = False) -> Tuple[Optional[JointTrajectory], bool]:
        """
        Önbellekten al ya da planla. (JointTrajectory | None, from_cache: bool) döndürür.

        from_cache, çağıranın YENİ bir planı başlangıcına hizalamayı atlamasını sağlar (plan
        zaten güncel pozdan başlar).
        """
        moveit2 = self.moveit
        vp_id = vp.get("id") or f"{self.robot_tag}_step{idx}"
        goal = self._subset(vp["joint_names"], vp["joint_positions"], moveit2.joint_names)

        # Poz kipi (UR): planlayıcının IK çözdüğü kamera optik çerçevesi POZUNA planla;
        # böylece MoveIt geçerli ve doğru yönelimli bir eklem çözümünü yeniden türetir.
        # Eklem uzaylı kollar (Kawasaki) değişmeden devam eder.
        use_pose = (self._use_pose_goal()
                    and vp.get("position") is not None and vp.get("rotation") is not None)
        pose_goal = position = quat = None
        target_link = self._pose_target_link()
        if use_pose:
            position = [float(x) for x in vp["position"]]
            quat = Rotation.from_matrix(np.array(vp["rotation"], dtype=float)).as_quat()  # xyzw
            pose_goal = list(position) + [float(x) for x in quat]

        if goal is None and not use_pose:
            self.get_logger().warning(f"[{label}] viewpoint {vp_id} missing a group joint; "
                                      "skipping.")
            return None, False

        use_cache = bool(self.get_parameter("use_trajectory_cache").value)
        force = bool(self.get_parameter("force_replan").value)
        path = self._traj_cache_path(vp_id)

        # force_fresh (önce-eve uğrama): doğrudan GÜNCEL pozdan (ev) hedefe planla ve
        # önbelleğe DOKUNMA; bilerek evden yaklaşıyoruz.
        if use_cache and not force and not force_fresh and os.path.exists(path):
            try:
                with open(path) as f:
                    cached = json.load(f)
            except Exception as e:
                cached = None
                self.get_logger().warning(f"[{label}] {vp_id}: could not read cache {path}: {e}")
            if cached and self._cache_valid(cached, moveit2.joint_names, goal,
                                            pose_goal=pose_goal):
                self.get_logger().info(
                    f"[{label}] {vp_id}: using CACHED trajectory ({len(cached['points'])} pts) "
                    "(will align to its start first).")
                return self._deserialize_traj(cached), True
            elif cached is not None:
                self.get_logger().info(f"[{label}] {vp_id}: cached goal differs from plan; "
                                       "replanning.")

        traj = None
        saved_pose_goal = None
        used_goal = goal
        if use_pose:
            # UR için BİRİNCİL. Aşağıdaki iki yol da planlayıcının IK çözdüğü kamera optik
            # çerçevesini hedefler; bu yüzden kamera yönelimi iki durumda da doğrudur;
            # yalnızca kolun pozun HANGİ IK dalına gönderildiğinde ayrışırlar.
            #
            # 1) En yakın dal: IK'yı güncel pozla tohumlayarak kendimiz çöz ve yol veren en
            #    yakın dalı al. Kolun santimetreler ötedeki bir bakış noktasına ulaşmak için
            #    farklı bir omuz/bilek konfigürasyonundan dönmesini durduran budur.
            traj, cand_goal = self._plan_best_branch(
                self._branch_candidates(position, quat, f"{label} {vp_id}"),
                moveit2.joint_names, f"{label} {vp_id}")
            if traj is not None:
                saved_pose_goal = pose_goal
                used_goal = cand_goal

            # 2) Yedek: Kartezyen poz hedefini MoveIt'e ver, hedef örnekleyicisi bir dal
            #    seçsin; ardından en azından döndürdüğü şeyi sar.
            if traj is None:
                traj = self._plan_pose(position, quat, target_link,
                                       f"{label} {vp_id} [pose]")
                if traj is not None:
                    saved_pose_goal = pose_goal
                    used_goal = [float(x) for x in traj.points[-1].positions]
                    traj, used_goal = self._shorten_wound_up(
                        traj, used_goal, list(moveit2.joint_names), f"{label} {vp_id}")

        if traj is None and goal is not None:
            # Eklem uzayı (Kawasaki her zaman; UR yalnızca poz hedefi başarısız olduysa).
            #
            # 0) Kol bir IK hedefi ifade edebiliyorsa önce en yakın dal. Saklı hedef, plan
            #    anında PLANLANAN öncüle göre sabitlenmiş TEK bir daldır.
            #    Kol gerçekte orada değilse (evden ilk bakış noktası, eve uğrama, atlanan
            #    bir bakış noktası, yeniden hizalanan önbellekli bir yol) aynı kamera pozunun
            #    farklı bir dalı çok daha yakın olabilir. Poz her adayda aynıdır;
            #    dolayısıyla tarama verisi değişemez.
            ik_target = self._ik_target(vp)
            if ik_target is not None:
                pos_t, quat_t, link_t = ik_target
                traj, cand_goal = self._plan_best_branch(
                    self._branch_candidates(pos_t, quat_t, f"{label} {vp_id}",
                                            ik_link=link_t, planned_goal=goal),
                    moveit2.joint_names, f"{label} {vp_id}")
                if traj is not None:
                    used_goal = cand_goal

        if traj is None and goal is not None:
            # 1) SARILMIŞ hedef (kolun şu anki yerine en yakın 2*pi eşdeğeri: aynı poz, çok
            #    daha az yol), ardından +/-2*pi sınır kenarına park etmiş bir hedef için açı
            #    normalleştirmeli kurtarma, ardından ham planlanan hedef.
            wgoal, wchanges = self._wrap_goal_to_current(
                goal, moveit2.joint_names, f"{label} {vp_id}")
            ngoal, changed = self._normalize_goal(goal, moveit2.joint_names)
            attempts_order = []
            if wchanges:
                attempts_order.append((wgoal, " [unwound]"))
            if changed:
                attempts_order.append((ngoal, " [normalized]"))
            attempts_order.append((goal, ""))
            for cand, sfx in attempts_order:
                traj = self._plan(cand, moveit2.joint_names, f"{label} {vp_id}{sfx}")
                if traj is not None:
                    used_goal = cand
                    break

        if traj is None:
            return None, False
        if use_cache and not force_fresh:
            try:
                os.makedirs(self._traj_dir, exist_ok=True)
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(self._serialize_traj(traj, vp_id, used_goal,
                                                   pose_goal=saved_pose_goal,
                                                   planned_goal=goal), f, indent=1)
                os.replace(tmp, path)
                self.get_logger().info(f"[{label}] {vp_id}: trajectory saved -> {path}")
            except Exception as e:
                self.get_logger().warning(f"[{label}] {vp_id}: could not save trajectory: {e}")
        return traj, False

    # ------------------------------------------------------------------ #
    # Adlandırılmış geçişler (eve uğramalar / hizalanma hamleleri); bakış noktaları gibi önbelleğe
    # alınır.
    # ------------------------------------------------------------------ #
    def _cached_move(
            self, name: str, target_rad: Sequence[float], joint_names: Sequence[str],
            label: str) -> Tuple[Optional[JointTrajectory], bool]:
        """
        Adlandırılmış yardımcı bir eklem uzayı hareketini önbellekten alır ya da planlar.

        Bir önce-eve-uğra bakış noktasının iki hamlesi için kullanılır: EVE uğrama ve bakış
        noktasının kayıtlı yörünge başlangıcına yaklaşma. Bunlar bakış noktası yörüngeleriyle
        AYNI "bir kez kaydet, hep oynat" sözleşmesini alır (bakış noktası kimliği yerine
        `name` ile anahtarlanır); böylece gerçek robotta canlı yeniden planlanmak yerine
        benzetimde kaydedilen yolun aynısı oynatılır.
        (JointTrajectory | None, from_cache) döndürür.
        """
        use_cache = bool(self.get_parameter("use_trajectory_cache").value)
        force = bool(self.get_parameter("force_replan").value)
        path = self._traj_cache_path(name)

        if use_cache and not force and os.path.exists(path):
            try:
                with open(path) as f:
                    cached = json.load(f)
            except Exception as e:
                cached = None
                self.get_logger().warning(f"[{label}] could not read cache {path}: {e}")
            if cached and self._cache_valid(cached, joint_names, target_rad):
                self.get_logger().info(
                    f"[{label}] using CACHED transition '{name}' ({len(cached['points'])} pts).")
                return self._deserialize_traj(cached), True
            elif cached is not None:
                self.get_logger().info(
                    f"[{label}] cached transition '{name}' targets a different pose; replanning.")

        traj = self._plan(target_rad, joint_names, label)
        if traj is None:
            return None, False
        if use_cache:
            try:
                os.makedirs(self._traj_dir, exist_ok=True)
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(self._serialize_traj(traj, name, target_rad), f, indent=1)
                os.replace(tmp, path)
                self.get_logger().info(f"[{label}] transition '{name}' saved -> {path}")
            except Exception as e:
                self.get_logger().warning(f"[{label}] could not save transition '{name}': {e}")
        return traj, False

    def _run_cached_move(
            self, name: str, target_rad: Sequence[float], joint_names: Sequence[str],
            label: str) -> bool:
        """
        Adlandırılmış bir geçişi planlar ya da oynatır ve kolu onun boyunca sürer.

        Kol zaten hedefteyse hiçbir şey yapmaz.
        """
        if len(target_rad) != len(joint_names):
            self.get_logger().warning(
                f"[{label}] transition '{name}' expects {len(joint_names)} joint values, "
                f"got {len(target_rad)}; skipping it.")
            return False
        # Önce sar: 0 derece olarak yazılmış bir ev/hizalanma hedefi sarılmış bir koldan
        # 350 derecelik bir yolculuktur, +360 derecelik eşdeğeri ise 10 derece ötededir. Bunu
        # _at_pose'dan ÖNCE yapmak, kolun zaten fiilen bulunduğu bir eklemi çözmesini de engeller.
        target_rad, _ = self._wrap_goal_to_current(target_rad, joint_names, f"{label} {name}")
        targets = dict(zip(joint_names, target_rad))
        if self._at_pose(targets):
            self.get_logger().info(f"[{label}] already at the '{name}' target; no move needed.")
            return True
        traj, _ = self._cached_move(name, target_rad, joint_names, label)
        if traj is None:
            self.get_logger().warning(f"[{label}] could not plan transition '{name}'; skipping "
                                      "it.")
            return False
        self._prepare_start(traj, True, label)
        return self._settle(self._dispatch(traj, label), targets, f"[{label}]")

    # ------------------------------------------------------------------ #
    # Hareketler.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pose_to_rad(pose_deg: Sequence[float]) -> List[float]:
        """Yapılandırma pozunu ([eksen METRE, eklemler DERECE]) radyana çevirir."""
        return [float(pose_deg[0])] + [float(np.deg2rad(v)) for v in pose_deg[1:]]

    def _home_arm(self, home_deg: Sequence[float], label: str) -> None:
        """
        Bu kolu ev pozuna planlar (çarpışmaları gözeterek) ve gönderir.

        home_deg'in 0. indeksi METRE cinsinden doğrusal eksen; 1..6 indeksleri DERECE
        cinsinden dönel eklemlerdir.

        Diğer her hareket gibi kaydedilir ve oynatılır: son bakış noktası -> ev ayağı
        adlandırılmış bir geçiştir ('return_home'); bu yüzden gerçek robotta canlı yeniden
        planlanmak yerine bakış noktası yörüngeleriyle aynı "bir kez kaydet, hep oynat"
        sözleşmesini alır. Yörünge klasöründe <kol>_return_home.json altında önbelleğe alınır.
        """
        moveit2, ctrl = self.moveit, self.ctrl
        if moveit2 is None or ctrl is None:
            return
        if len(home_deg) != len(moveit2.joint_names):
            self.get_logger().warning(
                f"[{label} home] expected {len(moveit2.joint_names)} joint values, got "
                f"{len(home_deg)}; skipping homing.")
            return
        self.get_logger().info(
            f"[{label}] returning to home pose (axis0={home_deg[0]} m, joints={home_deg[1:]} "
            "deg)...")
        # _run_cached_move hedefi zaten sarar, kol oradaysa kısa devre yapar, kayıtlı
        # başlangıca hizalanır ve varışta yerleşir.
        if self._run_cached_move("return_home", self._pose_to_rad(home_deg),
                                 moveit2.joint_names, f"{label} home"):
            self.get_logger().info(f"[{label}] returned to home pose.")
        else:
            self.get_logger().warning(f"[{label} home] homing motion did not complete "
                                      "successfully.")

    def _at_pose(self, targets: Dict[str, float]) -> bool:
        """Ölçülen eklemler hedeflere BAŞLANGIÇ toleransı içindeyse True döndürür."""
        pos = dict(self._joint_pos)
        for n, t in targets.items():
            if n not in pos or abs(pos[n] - t) > self._start_tol_for(n):
                return False
        return True

    def _move_to_pose(self, pose_deg: Sequence[float], label: str) -> None:
        """
        Bu kolu, zaten orada değilse pose_deg'e götürür (0. indeks doğrusal m, 1..6 derece).

        Çarpışmaları gözeterek planlanır; tamamlanmaya kolun _settle'ı karar verir.
        """
        moveit2, ctrl = self.moveit, self.ctrl
        if moveit2 is None or ctrl is None:
            return
        if len(pose_deg) != len(moveit2.joint_names):
            self.get_logger().warning(
                f"[{label}] expected {len(moveit2.joint_names)} joint values, got "
                f"{len(pose_deg)}; skipping.")
            return
        target_rad = self._pose_to_rad(pose_deg)
        target_rad, _ = self._wrap_goal_to_current(target_rad, moveit2.joint_names, label)
        targets = dict(zip(moveit2.joint_names, target_rad))
        if self._at_pose(targets):
            self.get_logger().info(f"[{label}] already at start pose; no move needed.")
            return
        self.get_logger().info(
            f"[{label}] NOT at start pose -> moving there "
            f"(axis0={pose_deg[0]} m, joints={list(pose_deg[1:])} deg).")
        traj = self._plan(target_rad, moveit2.joint_names, label)
        if traj is None:
            self.get_logger().warning(f"[{label}] could not plan to start pose; leaving arm "
                                      "where it is.")
            return
        send = self._dispatch(traj, label)
        self._settle(send, targets, f"[{label}]")

    def _unwind_traj_to_current(self, traj: JointTrajectory, label: str) -> List[str]:
        """
        Kayıtlı bir yörüngeden TAM TURLARI çıkararak onu kolun bulunduğu yerden başlatır.

        Bu, _wrap_goal_to_current'in yörünge düzeyindeki karşılığıdır ve önbellekli bir yol
        saklandığı gibi oynatıldığı için vardır: kol kayıtlı başlangıcın bir 2*pi eşdeğerinde
        duruyorsa (aynı fiziksel poz, 360 derece farklı sayı) sonraki her denetim ham açıları
        karşılaştırır, kolun "başlangıçta olmadığına" karar verir ve onu zaten bulunduğu bir
        poza tam bir tur döndürür.

        Düzeltme karşılaştırmaya değil YOLA uygulanır. _at_pose'u 2*pi modülünde
        karşılaştırır yapmak, yörünge hâlâ uzak sayıları tutarken "zaten orada" derdi ve onu
        göndermek tek bir denetleyici döngüsünde 360 derecelik bir adım olurdu. Eklemin her
        noktasını aynı k*2*pi kadar kaydırmak yolu biçim ve fiziksel pozlar bakımından aynı
        tutar (sabit bir ofset hızları ve ivmeleri değiştirmez) ve onu yalnızca kolun
        gerçekte bulunduğu yerde yeniden ifade eder.

        KARARI SINIRLAR VERİR ve kol başına farklıdırlar; buradaki asıl mesele budur:
          * kaydırma yalnızca aralığı 2*pi'yi AŞAN eklemler için düşünülür; bu yüzden
            Kawasaki'de yalnızca joint4 ve joint6'ya (+/-360 derece) dokunulur; joint1
            (+/-180, aralık tam 2*pi), joint2 (-80..135), joint3 (-172..118) ve joint5
            (+/-145) için sınır içinde ikinci bir eşdeğer olamaz ve onlara dokunulmaz;
          * kaydırılan eklem yalnızca başlangıçta değil, yolun TAMAMINDA sınırlar içinde
            kalmalıdır; 270..334 derecelik bir joint4 taraması güvenle -90..-26'ya kayar,
            ama zaten 300..400 dereceye yayılan bir taramanın gidecek yeri yoktur;
          * ayrıştırılamayan bir model sınır vermez ve o zaman bu işlem etkisizdir.
        UR'de (6 eklemin 5'i +/-360 derece) aynı kuralın daha çok yeri vardır.

        `traj`'ı yerinde değiştirir. Değişikliklerin listesini döndürür; hiçbir şey
        kaymadıysa boştur.
        """
        if traj is None or not traj.points:
            return []
        if not self._wrap_enabled():
            return []
        limits = self._ensure_joint_limits()
        if not limits:
            return []
        pos = dict(self._joint_pos)
        if not pos:
            return []

        margin = float(self.get_parameter("wrap_limit_margin").value)
        min_gain = float(self.get_parameter("wrap_min_gain").value)
        two_pi = 2.0 * np.pi
        names = list(traj.joint_names)
        cols = np.array([[float(v) for v in p.positions] for p in traj.points])

        changes = []
        for i, n in enumerate(names):
            lim = limits.get(n)
            # Sınır bilgisi yok, doğrusal eklem ya da ikinci bir eşdeğeri alamayacak bir aralık:
            # kabul edilebilir bir şey yok; asla tahmin etme.
            if lim is None or not lim.wrappable or self._is_prismatic(n):
                continue
            cur = pos.get(n)
            if cur is None:
                continue
            lo, hi = lim.lower + margin, lim.upper - margin
            col_min, col_max = float(cols[:, i].min()), float(cols[:, i].max())
            start = float(cols[0, i])
            best_k, best_dist = 0, abs(start - cur)
            for k in range(-3, 4):
                if k == 0:
                    continue
                off = k * two_pi
                if col_min + off < lo or col_max + off > hi:
                    continue  # yalnızca hedef değil, YOL sınırların dışına çıkardı
                d = abs(start + off - cur)
                if d < best_dist:
                    best_k, best_dist = k, d
            if best_k == 0:
                continue
            gain = abs(start - cur) - best_dist
            if gain < min_gain:
                continue  # kayıtlı yolu değiştirmeye değmez
            cols[:, i] += best_k * two_pi
            changes.append(f"{n} {np.degrees(start):.0f}->{np.degrees(cols[0, i]):.0f} deg "
                           f"(saves {np.degrees(gain):.0f} deg)")

        if not changes:
            return []
        for j, p in enumerate(traj.points):
            p.positions = [float(v) for v in cols[j]]
        self.get_logger().info(
            f"[{label}] recorded path unwound onto the arm's current pose: "
            + "; ".join(changes))
        return changes

    def _rebase_traj_to_current(self, traj: JointTrajectory, label: str) -> bool:
        """
        Yörüngenin İLK noktasının üzerine kolun GÜNCEL ölçülen eklem konumlarını yazar.

        Hızlar/ivmeler 0, t=0; böylece denetleyici robotun gerçekte BULUNDUĞU yerden doğrudan
        hedefe yumuşakça ara değer üretir. Bu, kanıtlanmış ur_inspection_scenario oynatma
        yönteminin aynısıdır ve rayı ileri-geri sarsan ayrı "kayıtlı başlangıca hizalan"
        hareketinin (_align_to_traj_start) YERİNİ alır. Bir eklemin ölçülen değeri eksikse
        kayıtlı başlangıç olduğu gibi bırakılır (başlangıç noktası asla uydurulmaz).

        YALNIZCA KÜÇÜK BİR DÜZELTME OLARAK GEÇERLİDİR. Taşıma point[0]'ı yeniden yazar ama
        point[1]'i ~0.1 sn sonraki kayıtlı değerinde bırakır; bu yüzden güncel poz ile kayıtlı
        başlangıç arasındaki BÜYÜK bir fark, denetleyicinin tek döngüde kapatması gereken bir
        adıma dönüşür: kol gözle görülür biçimde ışınlanır ve scaled_joint_trajectory_controller
        'State tolerances failed ... Aborted due to state tolerance violation' ile durur
        (burada CONTROL_FAILED olarak görünür). Bu yüzden yalnızca her eklem zaten başlangıç
        toleransı içindeyse taşınır; aksi hâlde reddedilir ve False döner, çağıran da kolu
        kayıtlı başlangıca düzgünce yürütür.
        """
        if traj is None or not traj.points:
            return False
        pos = dict(self._joint_pos)
        names = list(traj.joint_names)
        if not all(n in pos for n in names):
            self.get_logger().warning(
                f"[{label}] cannot rebase trajectory start (missing joint feedback); "
                "running recorded start as-is.")
            return False
        recorded = [float(x) for x in traj.points[0].positions]
        far = [(n, pos[n] - r) for n, r in zip(names, recorded)
               if abs(pos[n] - r) > self._start_tol_for(n)]
        if far:
            worst = max(far, key=lambda kv: abs(kv[1]))
            self.get_logger().warning(
                f"[{label}] NOT rebasing: the arm is {abs(worst[1]):.3f} away from this "
                f"trajectory's recorded start on '{worst[0]}' ({len(far)} joint(s) out of "
                "tolerance). Rebasing that gap would command a one-cycle jump; walking to "
                "the recorded start first instead.")
            return False
        p0 = traj.points[0]
        p0.positions = [float(pos[n]) for n in names]
        p0.velocities = [0.0] * len(names)
        p0.accelerations = [0.0] * len(names)
        p0.time_from_start = Duration(sec=0, nanosec=0)
        return True

    def _prepare_start(self, traj: JointTrajectory, from_cache: bool, label: str) -> None:
        """Göndermeden önce kolun gerçek pozunu yörüngenin ilk noktasıyla uyuşturur."""
        # İKİ kol için de HER ZAMAN İLK: kayıtlı yoldan tam turları çıkar; böylece
        # başlangıcı kolun gerçekte bulunduğu yerde ifade edilir. Aşağıdaki her şey ham
        # açıları karşılaştırır; 2*pi uzakta bırakılan bir yol, kolun zaten bulunduğu bir
        # poza tam bir tur döndürülerek "hizalanırdı".
        self._unwind_traj_to_current(traj, label)
        if self.rebase_to_current:
            # Fark, bunun bir sıçrama değil düzeltme olacağı kadar küçükse güncel poza taşı
            # (ayrı bir hareket yok); aksi hâlde açıkça başlangıca yürümeye düş.
            if self._rebase_traj_to_current(traj, label):
                return
            self._align_to_traj_start(traj, True, label)
            self._rebase_traj_to_current(traj, label)  # yürüyüşten sonra kalan artığı topla
            return
        self._align_to_traj_start(traj, from_cache, label)

    def _align_to_traj_start(self, traj: JointTrajectory, from_cache: bool, label: str) -> None:
        """
        Göndermeden önce kolun `traj`'ın İLK noktasında olmasını sağlar.

        ÖNBELLEKLİ bir yörünge biraz uzakta başlayabilir; bu yüzden başlangıcına çarpışmaları
        gözeten TEK bir kısa hareket yapılır. YENİ bir plan zaten güncel pozdan başlar; bu
        yüzden atlanır (UR için self.align_on_fresh False; rayın yersiz ileri-geri hareketinden
        kaçınılır).

        Hedeflerini _at_pose'u denetlemeden önce saran _run_cached_move ve _move_to_pose'un
        aksine burada BİLEREK sarılmaz: bunun hedefi kayıtlı yolun kendi ilk noktasıdır;
        yolu yeniden yazmadan onu yeniden yazmak, kolun bir sayıya yürüyüp sonra başka bir
        sayıdan oynatmasına yol açardı. _prepare_start bunun yerine yolun tamamını zaten
        sarmıştır; bu da başlangıcı ve yolu birlikte düzeltir ve aşağıdaki ham
        karşılaştırmayı doğru bırakır.
        """
        moveit2, ctrl = self.moveit, self.ctrl
        if moveit2 is None or ctrl is None or traj is None or not traj.points:
            return
        if not from_cache and not self.align_on_fresh:
            return
        start = [float(x) for x in traj.points[0].positions]
        targets = dict(zip(list(traj.joint_names), start))
        if self._at_pose(targets):
            return  # zaten kayıtlı başlangıçta (yeni planlar her zaman buraya düşer)
        self.get_logger().info(f"[{label}] aligning to cached trajectory start before replay.")
        plan = self._plan(start, list(traj.joint_names), f"{label} align-start")
        if plan is None:
            self.get_logger().warning(
                f"[{label}] could not plan to the cached start; running cached traj from current "
                "pose.")
            return
        send = self._dispatch(plan, f"{label} align-start")
        self._settle(send, targets, f"[{label} align]")

    def _go_to_start(self) -> None:
        """
        Dizi başlamadan önce bu kolu sabit BAŞLANGIÇ pozuna gönderir.

        Böylece ilk (önbellekten gelebilecek) yörünge bilinen ve eşleşen bir durumdan başlar.
        """
        if not self.get_parameter("go_to_start").value:
            return
        self._move_to_pose(
            list(self.get_parameter(self.start_pose_param).value), f"{self.arm_label} start")

    # ------------------------------------------------------------------ #
    # Dizi.
    # ------------------------------------------------------------------ #
    def run_sequence(self, vps: List[Viewpoint]) -> int:
        """
        BU kolun bakış noktalarının her birini sırayla planlar/hizalar/gönderir/yerleştirir/toplar.

        Bulutu toplanan bakış noktası sayısını döndürür.
        """
        label = self.arm_label
        # `or []`: BOŞ bir liste [] olarak değil None olarak gelir. Başlatma dosyası bu
        # parametreyi bir parametre dosyasıyla geçirir ve rclpy YAML'daki çıplak bir `[]`
        # için tip çıkaramaz; dynamic_typing ile PARAMETER_NOT_SET -> None olarak düşer.
        # (Python API'si üzerinden geçirilen []-değerli bir Parameter [] olarak döner;
        # bu yüzden durum yalnızca başlatma dosyasıyla çalıştırınca ortaya çıkar.)
        home_before = list(self.get_parameter("home_before_viewpoints").value or [])
        start_pose = list(self.get_parameter(self.start_pose_param).value)
        n = len(vps)
        saved = 0
        for idx, vp in enumerate(vps, start=1):
            vp_id = vp.get("id")
            home_first = vp_id in home_before
            if home_first:
                # Önceki bakış noktasından DOĞRUDAN hamle kablo kanalını sıkıştırır; bu yüzden
                # EVE uğranır. Uğramanın iki hamlesi de (buradan->ev, sonra ev->bu bakış noktasının
                # kayıtlı başlangıcı) KAYDEDİLMİŞ geçişlerdir; böylece gerçek robot canlı yeni bir
                # yol planlamak yerine onları oynatır.
                self.get_logger().warning(
                    f"[{label} {idx}/{n}] {vp_id} in home_before_viewpoints -> detouring via HOME "
                    "first, then approaching its recorded trajectory start.")
                self._run_cached_move(
                    f"{vp_id}_home", self._pose_to_rad(start_pose), self.moveit.joint_names,
                    f"{label} {vp_id} home-before")

            has_goal = bool(vp.get("joint_positions")) or (
                self._use_pose_goal() and vp.get("position") is not None
                and vp.get("rotation") is not None)
            if not has_goal:
                self.get_logger().warning(f"[{label} {idx}/{n}] {vp_id} has no goal; skipping.")
                continue

            # NOT: eve uğrama bilerek yeni bir planı ZORLAMAZ; bakış noktası kendi KAYITLI
            # yörüngesini korur; yalnızca ona nasıl yaklaştığımızı değiştiririz.
            traj, from_cache = self._get_trajectory(vp, label, idx)
            if traj is None:
                self.get_logger().warning(f"[{label} {idx}/{n}] {vp_id} could not be planned; "
                                          "skipping.")
                continue

            if home_first:
                # Evden gelen kol kayıtlı başlangıca hiç yakın değildir; bu yüzden bakış noktası
                # yolunu oynatmadan önce onu KAYITLI bir geçişle oraya yürüt.
                self._run_cached_move(
                    f"{vp_id}_align", [float(x) for x in traj.points[0].positions],
                    list(traj.joint_names), f"{label} {vp_id} align-start")

            self._prepare_start(traj, from_cache, label)
            self.get_logger().info(f"[{label} {idx}/{n}] Moving {label}->{vp_id}.")
            send = self._dispatch(traj, label)
            targets = self._viewpoint_targets(traj, vp)
            self._settle(send, targets, f"[{label} {idx}/{n}]")

            fallback = {self.robot_tag: self._fallback_pose(vp)}
            if self.capture_and_save(idx, {self.robot_tag}, fallback):
                saved += 1

        self.get_logger().info(f"[{label}] sequence complete: {saved}/{n} viewpoints captured.")
        return saved

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        if not os.path.exists(self.plan_file):
            self.get_logger().error(f"Plan file not found: {self.plan_file}.")
            return
        if not self.should_run():
            self.get_logger().info(
                f"{self.arm_label} node idle (only_ur/only_kawasaki/enable gating); exiting.")
            return
        with open(self.plan_file) as f:
            plan = json.load(f)
        vps = plan.get(self.viewpoints_key, [])
        if not vps:
            self.get_logger().error(
                f"Plan has no '{self.viewpoints_key}'; nothing to execute for {self.arm_label}.")
            return

        if not self.ctrl.wait_for_server(timeout_sec=15.0):
            self.get_logger().error(
                f"{self.arm_label} controller not available; aborting.")
            return
        if not self._wait_joint_state():
            self.get_logger().error(f"No /joint_states for the {self.arm_label} group; aborting.")
            return

        self._add_ground_plane()
        self._apply_collision_padding()
        self._setup_planning_scene_extras()

        self.get_logger().info(
            f"{self.arm_label} executing {len(vps)} viewpoints (independent single-arm node).")

        tours = max(1, int(self.get_parameter("tours").value))
        t_start = time.monotonic()
        saved = 0
        for tour in range(1, tours + 1):
            # Her turun başında BAŞLANGIÇ pozuna gidilir: ilk (muhtemelen önbellekten gelen)
            # yörünge, kaydedildiği durumdan başlasın. Bir sonraki tur, bir önceki turun
            # son durağından başladığı için bu ikinci turda da şarttır.
            self._go_to_start()
            if tours > 1:
                self.get_logger().info(
                    f"{self.arm_label} TUR {tour}/{tours} başlıyor "
                    "(yakalama dosyaları 1'den numaralanır, öncekinin üzerine yazılır).")
            saved = self.run_sequence(vps)

        if self.get_parameter(self.return_home_param).value:
            self._home_arm(
                list(self.get_parameter(self.home_positions_param).value), self.arm_label)

        trees = " + ".join(s["subtree"] for s in self.streams)
        self.get_logger().info(
            f"{self.arm_label} inspection complete [only_sim={self.only_sim}]: "
            f"{saved}/{len(vps)} viewpoints captured into {trees} "
            f"in {time.monotonic() - t_start:.1f}s"
            + (f" ({tours} tur; diskteki bulutlar SON turdan)." if tours > 1 else "."))

    # ------------------------------------------------------------------ #
    def _resolve_only_flags(self) -> Tuple[bool, bool]:
        """
        (only_ur, only_kawasaki) bayraklarını döndürür.

        Çelişkili ikisi-de-True durumu "ikisini de çalıştır" olarak çözülür; çelişki
        bir kez uyarılır.
        """
        only_ur = bool(self.get_parameter("only_ur").value)
        only_kawa = bool(self.get_parameter("only_kawasaki").value)
        if only_ur and only_kawa:
            self.get_logger().warning(
                "only_ur AND only_kawasaki are both True (contradictory); running BOTH arms.")
            only_ur = only_kawa = False
        return only_ur, only_kawa
