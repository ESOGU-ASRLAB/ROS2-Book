import json
import os
import re
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.client import Client
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from std_msgs.msg import String
from std_srvs.srv import Trigger
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import (CollisionObject, LinkPadding, PlanningScene,
                             PlanningSceneComponents)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene

# Tek kollu ch07_viewpoint_planner çekirdeği olduğu gibi yeniden kullanılır (bu
# paket ona bağımlıdır). Yalnızca iki kol arasındaki TAHSİS yenidir; ortak aday
# havuzu ve kaplama matrisi dahil öncesindeki her şey tek kollu temel çözümle
# aynıdır; bu, tek kol-çok kol karşılaştırmasını dürüst tutar.
from ch07_viewpoint_planner.mesh_analyzer import MeshAnalyzer
from ch07_viewpoint_planner.viewpoint_generator import Candidate, ViewpointGenerator
from ch07_viewpoint_planner.reachability_checker import ReachabilityChecker
from book_pymoveit2.joint_wrap import (describe_changes, JointLimit,
                                       parse_joint_limits, wrap_to_reference)

from ch07_multirobot_planner import paths
from ch07_multirobot_planner.robot_allocator import RobotAllocator

# aday -> (erişilebilir, eklem_çözümü); isteğe bağlı tohum ve hata sayımı bayrağı alır
ReachFn = Callable[..., Tuple[bool, Optional[JointState]]]


class IkSceneBuilder(Node):
    """
    move_group'un planlama sahnesini yürütücülerin kurduğu duruma getirir.

    Bu, burada herhangi bir /compute_ik çalışmadan ÖNCE yapılır. O durum: zemin
    kutusu ve bağlantı başına çarpışma dolgusu.

    Erişilebilirlik `ik_avoid_collisions` ile, yani move_group'un o an tuttuğu sahneye
    göre doğrulanır. Yürütücüler bir zemin düzlemi ekler ve hareketli bağlantılarına
    dolgu uygular (inspection_base._add_ground_plane / _apply_collision_padding); bu
    düğüm uygulamıyordu, dolayısıyla üretici çıplak bir sahneyi, yürütücü ise zeminli ve
    dolgulu bir sahneyi denetliyordu. 2026-09-07 ve 2026-09-08 şase planlarında
    ölçüldü: her koşuda bir UR bakış noktasının (z=0.278'de ur_vp_020, z=0.256'da
    ur_vp_018) yürütme anında çarpışmasız IK dalı YOKTU; 40 tohumlu /compute_ik
    sorgusunun hepsi NO_IK_SOLUTION döndürdü ve her dal forearm/wrist_1/wrist_2'yi
    zeminin içine koyuyordu. İkisi de yine plana yazılmış ve kaplamaları
    `coverage_achieved` içinde sayılmıştı.

    Dolgu İKİ kolun bağlantı kümelerine birden uygulanır; iş birlikli bir koşunun
    ürettiği de tam olarak budur: iki yürütücü aynı ortak sahneye dolgu uygular.

    ReachabilityChecker gibi kendi başına bir düğümdür; böylece senkron servis
    çağrıları planlayıcının servis geri çağırmasını kilitlemek yerine kendi
    yürütücülerinde döner.
    """

    def __init__(self) -> None:
        super().__init__('ik_scene_builder')
        self.apply_cli = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.get_cli = self.create_client(GetPlanningScene, '/get_planning_scene')

    def _call(self, cli: Client, req: Any, timeout: float = 10.0) -> Any:
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    def wait_for_services(self, timeout_sec: float = 10.0) -> bool:
        return (self.apply_cli.wait_for_service(timeout_sec=timeout_sec)
                and self.get_cli.wait_for_service(timeout_sec=timeout_sec))

    def add_ground_plane(
            self, top_z: float, size: float, thickness: float, frame_id: str = 'world') -> bool:
        """Yürütücülerin eklediği kutunun aynısı: dünya orijininde ortalı, ÜST yüzü top_z'de."""
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(size), float(size), float(thickness)]
        pose = Pose()
        pose.position.z = float(top_z) - float(thickness) / 2.0
        pose.orientation.w = 1.0

        obj = CollisionObject()
        obj.id = 'ground_plane'
        obj.header.frame_id = frame_id
        obj.operation = CollisionObject.ADD
        obj.primitives = [box]
        obj.primitive_poses = [pose]

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]
        result = self._call(self.apply_cli, ApplyPlanningScene.Request(scene=scene))
        return bool(result and result.success)

    def apply_padding(self, padding: float, patterns: Sequence[str]) -> int:
        """
        `patterns` (düzenli ifadeler) ile eşleşen her sahne bağlantısına dolgu uygular.

        Dolgu uygulanan bağlantı sayısını, sahne okunamadıysa -1 döndürür.
        """
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        result = self._call(self.get_cli, req)
        if result is None:
            return -1
        rxs = [re.compile(x) for x in patterns]
        links = [n for n in result.scene.allowed_collision_matrix.entry_names
                 if any(rx.search(n) for rx in rxs)]
        if not links:
            return 0
        scene = PlanningScene()
        scene.is_diff = True
        scene.link_padding = [LinkPadding(link_name=n, padding=float(padding))
                              for n in links]
        applied = self._call(self.apply_cli, ApplyPlanningScene.Request(scene=scene))
        return len(links) if (applied and applied.success) else -1


class MultiRobotPlannerNode(Node):
    """
    Ortak bir kaplama kümesi planlar ve onu UR10e ile Kawasaki arasında paylaştırır.

    İki kolda da artık bir SICK kamera var. Ayrı `ur_viewpoints` ve
    `kawasaki_viewpoints` listeleri olan bir plan dosyası yazar
    (varsayılan: ~/ros2_book_output/ch07/multirobot_viewpoint_plan.json); her biri
    kendi IK eklem çözümü olan gerçek bir kaplama bakış noktasıdır (ayna nokta DEĞİL).
    """

    def __init__(self) -> None:
        super().__init__('multirobot_planner_node')

        # --- Ağ / kaplama parametreleri (tek kollu planlayıcıyla aynı ayarlar) ---
        # Boşsa araştırma deposundaki ur_description paketinin şase ağı kullanılır
        self.declare_parameter('mesh_path', paths.chassis_mesh_file())
        self.declare_parameter('mesh_scale', 0.001)
        self.declare_parameter('coverage_threshold', 0.98)
        self.declare_parameter('target_sample_points', 5000)
        self.declare_parameter('num_base_points', 400)
        self.declare_parameter('ik_avoid_collisions', True)
        # KOL başına bütçe (0 = bütçe yok, eşiğe kadar kapla). İki kol paralel
        # çalıştığı için toplam süre ~max(kol başına sayılar) olur; bu yüzden bu ayar
        # "her kol en fazla N durak yapar" sınırını koyar.
        self.declare_parameter('max_viewpoints_per_robot', 0)
        # Yukarıdaki bütçenin KOL başına geçersiz kılınması (0 = geçersiz kılma yok, ortak değer).
        # Kollar eşit hızda olmadığı için vardırlar: bir koşunun toplam süresini yavaş
        # kol belirler; bu yüzden koşuyu kısaltan yalnızca o kolu sınırlamaktır.
        # İkisinin de varsayılanı 0'dır; yani bunlardan söz etmeyen bir yapılandırma
        # eskisi gibi davranır. Şase işi bilerek ortak üst sınırda bırakılmıştır.
        self.declare_parameter('max_viewpoints_ur', 0)
        self.declare_parameter('max_viewpoints_kawasaki', 0)
        # True -> iki kolun da ulaşabildiği aday daha az yüklü kola gider (daha kısa
        # paralel koşu). False -> sabit kol öncelik sırası, önce UR; yani Kawasaki
        # yalnızca UR'nin ulaşamadığı adayları alır.
        self.declare_parameter('balance_load', True)
        # Kuyruk kesici: kalan en iyi bakış noktası bu sayıdan AZ YENİ hedef noktası
        # ekleyecekse bakış noktası eklemeyi durdur. Kaplama eğrisinin keskin bir dirseği
        # ve ardından her biri yalnızca 1-3 nokta ekleyen uzun bir kuyruğu vardır.
        # Kuyruğu kesip çok daha az ara noktayla (küçük kaplama kaybıyla) yetinmek için artırın.
        self.declare_parameter('min_new_points', 1)
        # Oransal kuyruk kesici (tek kollu ch07_viewpoint_planner'ın aynası). Kalan en iyi
        # bakış noktası yeni kaplama olarak bütün hedef noktalarının bu ORANINDAN azını
        # ekleyecekse dur; ağ yoğunluğuyla ölçeklenir (0.005 = %0.5; 5000 hedefli bir ağda
        # ~25 yeni nokta). Etkin eşik max(min_new_points, bu)'dur.
        # 0 kapatır. Tek kollu temel çözümün kullandığı "%0.5'te kes" ayarı budur.
        self.declare_parameter('min_marginal_coverage', 0.005)

        # --- Kamera modeli (ortak: iki kolda da aynı SICK var) ---
        self.declare_parameter('camera.horizontal_fov_deg', 70.0)
        self.declare_parameter('camera.vertical_fov_deg', 60.0)
        self.declare_parameter('camera.fov_safety_margin_deg', 3.0)
        self.declare_parameter('camera.min_range_m', 0.2)
        self.declare_parameter('camera.max_range_m', 1.8)
        self.declare_parameter('camera.frame_rate_hz', 5.0)
        self.declare_parameter('camera.viewpoint_distances', [0.3, 0.6, 1.0, 1.4])
        self.declare_parameter('camera.tilt_variations_deg', [-15.0, 0.0, 15.0])
        self.declare_parameter('camera.max_incidence_angle_deg', 80.0)

        # --- UR10e erişilebilirliği (kamera ur10e_depth_optical_frame üzerinde) ---
        self.declare_parameter('ur_group_name', 'real_ur10e')
        self.declare_parameter('ur_base_frame', 'world')
        self.declare_parameter('ur_tool_frame', 'ur10e_depth_optical_frame')

        # YALNIZCA UR için ÇALIŞMA ALANI YASAK KUTUSU (world çerçevesi, metre). Yalnızca ağı
        # kullanan örtülme ışın izlemesi, UR'nin doğrusal rayının ARKASINDAKİ bir aday
        # kamera pozunun fiziksel olarak engellendiğini ve ZEMİNE yakın bir pozun
        # erişilemez olduğunu (kol yere çarpardı) bilemez. Bu kutu, kamera konumu [min, max]
        # dışında kalan her adayı UR için ERİŞİLEMEZ sayar; böylece tahsis edici onu
        # Kawasaki'ye verir (ya da atar). Kendi tabanındaki Kawasaki bu kutuyla asla
        # kısıtlanmaz; dolayısıyla Kawasaki'nin kaplaması korunur.
        #   X üst sınırı = -0.20: ray world X=-0.158'de durur (hücre dünya orijininde,
        #     Y boyunca ilerler); -0.20 (ray X + ~4 cm pay) rayın arkasındaki pozları reddeder.
        #   Z alt sınırı = 0.20: zemin payı; UR'nin yere çarpmadan ulaşamayacağı zemine
        #     yakın pozları reddeder. Kol hâlâ yere sürtüyorsa artırın; meşru alçak şase
        #     bakış noktalarını yanlışlıkla atıyorsa 0'a doğru düşürün.
        # Kapatmak için enabled False verin. Tek kollu ch07_viewpoint_planner'ın kullandığı kutuyla
        # aynıdır.
        # --- IK doğrulama sahnesi -----------------------------------------------
        # Yürütücüler planlamadan önce bunları move_group'un sahnesine ekler
        # (inspection_base._add_ground_plane / _apply_collision_padding). Aşağıdaki
        # erişilebilirlik taraması AYNI sahneyi görmelidir; yoksa tek IK dalları bir kolu
        # zeminin içine koyan kamera pozlarını onaylar. Bu değerleri koşunun gerçekten
        # kullanacağı başlatma argümanlarıyla aynı tutun.
        self.declare_parameter('ik_scene_setup', True)
        self.declare_parameter('ground_plane_z', -0.02)
        self.declare_parameter('ground_plane_size', 6.0)
        self.declare_parameter('ground_plane_thickness', 0.2)
        self.declare_parameter('collision_padding', 0.04)
        # Yürütücülerin dolgu uyguladığı bağlantıların düzenli ifadeleri: UR düğümü her
        # 'ur10e_*' bağlantısına, Kawasaki düğümü link1..link6'ya dolgu uygular. İki küme
        # de burada uygulanır, çünkü iş birlikli bir koşuda ikisi de tek sahneye uygulanmış olur.
        self.declare_parameter('ik_padding_link_patterns', ['^ur10e_', '^link[1-6]$'])

        self.declare_parameter('ur_workspace_filter_enabled', True)
        self.declare_parameter('ur_workspace_bounds_min', [-100.0, -100.0, 0.20])
        self.declare_parameter('ur_workspace_bounds_max', [-0.20, 100.0, 100.0])

        # --- Kawasaki erişilebilirliği. Bakış noktası, Kawasaki'nin SICK KAMERASININ
        #     bulunması gereken yerdir. Ama MoveIt'in real_kawasaki grubu (zincir ucu
        #     link6) için /compute_ik'sı, link6'nın 4 sabit eklem ötesindeki (büyük joint7
        #     rpy'si dahil) kamera çerçevesi için ik_link_name'i güvenilir biçimde dikkate
        #     almaz; sonunda link6'yı yönlendirir ve kamera yanlış yöne bakar. Bu yüzden IK
        #     grubun KENDİ ucu (link6) için çözülür ve sabit kamera->link6 dönüşümü
        #     (TF'den bulunur) hedef poza önceden uygulanır. Bu, KAMERAYI tam olarak bakış
        #     noktasına yerleştirir; yönelimin doğruluğu ik_link işleyişinden bağımsız
        #     olarak güvence altındadır.
        self.declare_parameter('kawasaki_group_name', 'real_kawasaki')
        self.declare_parameter('kawasaki_base_frame', 'world')
        # Üretilen bakış noktası pozunun ait olduğu çerçeve (kameranın gideceği yer).
        self.declare_parameter('kawasaki_camera_frame', 'camera_rgb_optic_frame')
        # Kawasaki grubunun IK uç bağlantısı (/compute_ik'nın gerçekte çözdüğü).
        self.declare_parameter('kawasaki_group_tip', 'link6')
        # True -> sağlam uç+ofset IK (önerilen). False -> kamera çerçevesini ik_link_name
        # ile doğrudan hedefle (yalnızca MoveIt'iniz bunu dikkate alıyorsa).
        self.declare_parameter('kawasaki_ik_via_tip', True)
        # Kamera optik ekseni gelenek düzeltmesi (YÖNELİM düzeltmesi). Bakış noktası
        # üreticisi pozları, kameranın +Z boyunca baktığı ROS optik geleneğinde üretir.
        # Bu, *gerçek* depth_optical_frame'i (ur_macro.xacro, rpy 0 0 0) görüş ekseni
        # olarak +Z'ye sahip UR için doğrudur.
        # Kawasaki'nin camera_rgb_optic_frame'inin (rs005l_macro.xacro) görüş ekseni ise
        # +X boyuncadır (kaydedilmiş bulutlardan doğrulandı: toplanan geometri o çerçevenin
        # +X'i boyunca durur). Bu yüzden Kawasaki kamerasının hedef yönelimi R_cam = R_view @ C
        # olur; C, üreticinin +Z görüş eksenini çerçevenin gerçek +X görüş eksenine eşler.
        # rpy(-pi/2,-pi/2,0) tam olarak bunu yapar (UR'nin benzetim ve gerçek optik
        # çerçevelerini ilişkilendiren dönüşün kendisidir). Yalnızca Kawasaki hâlâ şaseye
        # dik bakmıyorsa ayarlayın.
        self.declare_parameter('kawasaki_view_axis_correction_rpy',
                               [-1.5707963, -1.5707963, 0.0])

        self.declare_parameter('output_plan_file',
                               paths.output_path('multirobot_viewpoint_plan.json'))

        # Her kolun seçilmiş bakış noktalarını kısa bir Kartezyen ziyaret yoluna
        # (en yakın komşu + 2-opt) yeniden sırala; böylece kol, açgözlü kaplama keşfi
        # sırasıyla şasenin bir ucundan diğerine atlamak yerine komşu durakları tarar.
        # YALNIZCA ziyaret sırasını değiştirir, hangi bakış noktalarının tutulduğunu asla.
        # False -> açgözlü seçim sırasını koru.
        self.declare_parameter('order_by_proximity', True)
        # Ziyaret taraması için kol başına BAŞLANGIÇ çapası. İki kol aynı bölgede
        # buluşmasınlar diye şaseye ZIT uçlardan yaklaşır:
        #   UR       -> 'max_y' (en büyük Y, şasenin ÖNÜ),
        #   Kawasaki -> 'min_y' (en küçük Y, şasenin ARKASI).
        # Seçenekler: 'max_y' | 'min_y' | 'coverage' (en yüksek kaplamalı bakış noktasından başla).
        self.declare_parameter('ur_order_anchor', 'max_y')
        self.declare_parameter('kawasaki_order_anchor', 'min_y')
        # Çapa seçildikten sonra turun NASIL kurulacağı:
        #   'y_bands'   -> tek yönlü tarama: bir Y bandını bitir, sonra sonrakine geç, asla
        #                  geri dönme. Bant içindeki sıralama yine en kısa yoldur.
        #   'proximity' -> düz en kısa 3B yol (özgün davranış); daha ucuzsa şasenin
        #                  başına geri yürümekte serbesttir.
        self.declare_parameter('order_mode', 'y_bands')
        # Bir Y bandının metre cinsinden derinliği. 0, katı bir Y sıralamasına dönüşür;
        # şaseden büyük bir değer 'proximity'ye dönüşür. 2026-09-07/08 UR turlarında
        # ölçüldü: 0.30, en kötü geri adımı 0.22 m'de tutar ('proximity' için 0.76 m),
        # karşılığında yol %13 uzar; daha dar tutmak neredeyse hiç tek yönlülük
        # kazandırmaz ve %15 daha maliyetlidir.
        self.declare_parameter('order_band_width_m', 0.30)

        # Saklanan IK çözümlerinin 2*pi SARILMASI. IK bir çözüm ailesinin rastgele bir
        # dalını döndürür; bu yüzden tur bir durakta +100 derece, sonrakinde -260 derece
        # saklayabilir: aynı fiziksel eklem açısı, ama sürülecek fazladan bir tam tur.
        # Ziyaret sırası kesinleşince tur yürünür ve eklemin URDF sınırları izin verdikçe
        # her açı, ÖNCEKİ durağa en yakın eşdeğeriyle yeniden ifade edilir. Pozlar
        # fiziksel olarak aynıdır; kaplama ve erişilebilirlik değişmez, yalnızca
        # yürütücünün aralarında sürdüğü sayılar değişir.
        # Her bakış noktasının IK'sını bir önceki bakış noktasının çözümüyle tohumlayarak
        # yeniden çöz ve yalnızca daha yakınsa tut; böylece tur, aynı poza zıt kol
        # duruşlarından ulaşan omuz/dirsek/bilek konfigürasyonları arasında atlamak yerine
        # TEK bir IK dalında kalır. Sarmanın aksine bu, HANGİ çözümün saklandığını
        # değiştirir; ama bakış noktası pozunu asla, dolayısıyla kaplama etkilenmez.
        self.declare_parameter('rechain_ik', True)
        # pick_ik rastgele yeniden başlatmalarla çalışır; bu yüzden tohumlu sorguyu
        # tekrarlamak farklı dallar örnekler; çözülenlerin en yakını tutulur.
        self.declare_parameter('rechain_ik_attempts', 4)
        # Yeniden çözüm için tohum çeşitlemeleri (bkz. _seed_variants). Yalnızca tekrar,
        # çözücü rastgele yeniden başlıyorsa dal örnekler; Kawasaki'nin KDL'si bunu
        # yapmaz; bu çeşitlemeler ona yakınsayıp uzaklaşabileceği bir şey verir.
        self.declare_parameter('rechain_rail_joint', 'world_to_agv')
        self.declare_parameter('rechain_rail_probe', 0.25)  # m, +/- ray tohum ofseti; 0 kapatır
        self.declare_parameter('rechain_rail_weight', 2.0)  # metre ray yolu başına radyan
        self.declare_parameter('rechain_yaw_probe', True)  # joint1'i +/- pi çevirerek dene
        self.declare_parameter('rechain_yaw_joint', 'joint1')

        self.declare_parameter('unwind_joint_goals', True)
        self.declare_parameter('wrap_limit_margin', 0.05)  # her sınırdan uzak tutulan radyan
        self.declare_parameter('wrap_min_gain', 0.35)  # radyan (~20 derece) en az kazanç

        # Yukarıdaki sarma için kalıcı (latched) URDF -> eklem sınırları.
        self._urdf_xml = None
        self._joint_limits = None
        self.create_subscription(
            String, '/robot_description', self._robot_description_cb,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       history=HistoryPolicy.KEEP_LAST))

        self.mesh_path = self.get_parameter('mesh_path').value
        self.mesh_scale = self.get_parameter('mesh_scale').value
        self.coverage_threshold = self.get_parameter('coverage_threshold').value

        self.camera_config = {
            'horizontal_fov_deg': self.get_parameter('camera.horizontal_fov_deg').value,
            'vertical_fov_deg': self.get_parameter('camera.vertical_fov_deg').value,
            'fov_safety_margin_deg': self.get_parameter('camera.fov_safety_margin_deg').value,
            'min_range_m': self.get_parameter('camera.min_range_m').value,
            'max_range_m': self.get_parameter('camera.max_range_m').value,
            'frame_rate_hz': self.get_parameter('camera.frame_rate_hz').value,
            'viewpoint_distances': list(self.get_parameter('camera.viewpoint_distances').value),
            'tilt_variations_deg': list(self.get_parameter('camera.tilt_variations_deg').value),
            'max_incidence_angle_deg': self.get_parameter('camera.max_incidence_angle_deg').value,
        }

        self.plan_srv = self.create_service(Trigger, '~/plan', self.plan_callback)
        self.get_logger().info(
            f"MultiRobotPlannerNode ready. mesh='{self.mesh_path}', "
            f"coverage_threshold={self.coverage_threshold}."
        )

    def _lookup_static_offset(
            self, parent_frame: str, child_frame: str,
            timeout: float = 8.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        child_frame'in parent_frame içindeki pozunu /tf_static'ten (R, p) olarak döndürür.

        Yerel olarak döndürülen kısa ömürlü bir düğüm kullanır (servis geri çağırması bu
        düğümün kendi yürütücüsünü bloke ettiği için kendi TF tamponumuza güvenemeyiz).
        Dönüşüm hiç gelmezse None döner.
        """
        from tf2_ros import Buffer, TransformListener
        tmp = rclpy.create_node('mrvp_tf_lookup')
        buf = Buffer()
        TransformListener(buf, tmp)
        tf = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(tmp, timeout_sec=0.1)
            try:
                if buf.can_transform(parent_frame, child_frame, rclpy.time.Time()):
                    tf = buf.lookup_transform(parent_frame, child_frame, rclpy.time.Time())
                    break
            except Exception:
                pass
        tmp.destroy_node()
        if tf is None:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return R, np.array([t.x, t.y, t.z])

    def _analytic_kawasaki_offset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Grup ucunun (link6) kamera optik çerçevesindeki sabit pozunu URDF'ten hesaplar.

        Değer doğrudan rs005l_macro.xacro'daki URDF zincirinden hesaplanır. URDF sabitlerini
        kullanmak (TF araması yerine) uç üzerinden IK'yı sağlam kılar: gerçek (öneksiz)
        kamera/link6 çerçeveleri benzetim koşusunda TF'de YOKTUR; bu yüzden eski /tf_static
        araması sessizce başarısız oluyor ve MoveIt'in grup ucunun ötesinde dikkate almadığı
        eski bir ik_link yoluna düşüyordu. Düzeltmenin kameraya hiç ulaşmaması ve
        Kawasaki'nin yanlış bakmaya devam etmesi bu yüzdendi.

        Zincir:
            link6 --joint7 (rpy 0 0 -1.96)------------------------> link7
                  --camera_joint (xyz .0145 .0125 .185,
                                   rpy 0 -pi/2 pi/2)--------------> kamera(=optik)
            (camera_rgb_joint ve camera_rgb_optic_joint birim dönüşümdür)

        (R_off, p_off) = link6'nın kamera çerçevesindeki pozunu döndürür; yani
        _lookup_static_offset(camera, link6)'nın döndüreceği değerin aynısı.
        """
        def rpy(r: float, p: float, y: float) -> np.ndarray:
            return Rotation.from_euler('xyz', [r, p, y]).as_matrix()
        R67 = rpy(0.0, 0.0, -1.96)  # link6 -> link7
        R7c = rpy(0.0, -np.pi / 2, np.pi / 2)  # link7 -> camera
        p7c = np.array([0.0145, 0.0125, 0.185])
        R6c = R67 @ R7c  # link6 içinde kamera
        p6c = R67 @ p7c  # (joint7'nin ötelemesi sıfırdır)
        R_off = R6c.T  # kamera içinde link6
        p_off = -R6c.T @ p6c
        return R_off, p_off

    def _make_kawasaki_reach_fn(
            self, checker: ReachabilityChecker, offset: Tuple[np.ndarray, np.ndarray],
            correction: np.ndarray) -> Optional[ReachFn]:
        """
        Kawasaki için aday -> (erişilebilir, eklem_çözümü) fonksiyonu üretir.

        Kameranın gerçekten şaseye bakması için saf UR yoluna göre iki düzeltme gerekir:

          1. GÖRÜŞ EKSENİ GELENEĞİ (`correction`, 3x3): üreticinin pozunda görüş +Z
             boyuncadır, ama Kawasaki'nin camera_rgb_optic_frame'i +X boyunca bakar.
             R_cam = R_view @ correction, çerçeveyi gerçek görüş ekseni yüzeye bakacak
             biçimde yeniden yönlendirir.
          2. UÇ ÜZERİNDEN IK (`offset` = (R_off, p_off) = grup ucunun kamera
             çerçevesindeki pozu): IK, sabit kamera->uç dönüşümü önceden uygulanarak
             grubun kendi ucu için çözülür; böylece kamera çerçevesi ucun birkaç sabit
             eklem ötesinde olsa bile KAMERA hedefe oturur. Eski doğrudan ik_link yolu
             için offset = (I, 0) verin.
        """
        if not checker.wait_for_service():
            self.get_logger().warning(
                f"/compute_ik unavailable for group '{checker.group_name}' (Kawasaki). Its "
                "viewpoints "
                "will be selected WITHOUT reachability verification.")
            return None
        R_off, p_off = offset
        C = np.asarray(correction, dtype=np.float64)

        def _fn(
                candidate: Candidate, seed: Optional[JointState] = None,
                count_errors: bool = True) -> Tuple[bool, Optional[JointState]]:
            R_view = np.asarray(candidate['rotation'], dtype=np.float64)
            pos = np.asarray(candidate['position'], dtype=np.float64)
            R_cam = R_view @ C  # kameranın ulaşması gereken yönelim
            R_tip = R_cam @ R_off  # -> grup ucunun yönelimi
            p_tip = pos + R_cam @ p_off  # -> grup ucunun konumu
            quat = Rotation.from_matrix(R_tip).as_quat()  # [x, y, z, w]
            return checker.check_ik(p_tip, quat, seed_joint_state=seed,
                                    count_errors=count_errors)

        return _fn

    def _ur_workspace_gate(self) -> Optional[Callable[[Sequence[float]], bool]]:
        """
        Kamera konumu UR yasak kutusunun içindeyse True döndüren bir fonksiyon üretir.

        Kutu kapalıysa None döner. Yalnızca UR için rayın arkasındaki adayları veto
        etmekte kullanılır (Kawasaki asla engellenmez).
        """
        if not self.get_parameter('ur_workspace_filter_enabled').value:
            return None
        lo = np.asarray(self.get_parameter('ur_workspace_bounds_min').value, dtype=float)
        hi = np.asarray(self.get_parameter('ur_workspace_bounds_max').value, dtype=float)
        if lo.shape != (3,) or hi.shape != (3,):
            self.get_logger().warning(
                "ur_workspace_bounds_min/max must each have 3 values [x,y,z]; UR workspace gate "
                "off.")
            return None
        self.get_logger().info(
            f"UR workspace keep-out box active: camera must be within min={lo.tolist()} "
            f"max={hi.tolist()} (world). Behind-rail candidates are handed to the Kawasaki.")
        return lambda p: bool(np.all(np.asarray(p, dtype=float) >= lo)
                              and np.all(np.asarray(p, dtype=float) <= hi))

    def _prepare_ik_scene(self) -> Dict[str, Any]:
        """
        Yürütücülerin planlama sahnesini yeniden kurar.

        Böylece erişilebilirlik, kolların gerçekten planlayacağı sahneye göre doğrulanır.
        Neyin uygulandığını anlatan bir sözlük döndürür; bu sözlük plan dosyasına
        damgalanır, böylece çıplak bir sahneye göre üretilmiş bir plan asla doğrulanmış
        bir planla karıştırılamaz.
        """
        stamp = {'applied': False, 'reason': 'disabled'}
        if not self.get_parameter('ik_scene_setup').value:
            self.get_logger().warning(
                "ik_scene_setup=False: reachability will be checked against move_group's "
                "CURRENT scene. If the executors add a ground plane or padding, this plan "
                "may contain viewpoints the arms cannot reach.")
            return stamp

        builder = IkSceneBuilder()
        try:
            if not builder.wait_for_services():
                stamp['reason'] = 'apply_planning_scene/get_planning_scene unavailable'
                self.get_logger().error(
                    "Planning-scene services unavailable; the IK sweep will run against a "
                    "BARE scene and may approve unreachable viewpoints.")
                return stamp

            top_z = float(self.get_parameter('ground_plane_z').value)
            size = float(self.get_parameter('ground_plane_size').value)
            thickness = float(self.get_parameter('ground_plane_thickness').value)
            padding = float(self.get_parameter('collision_padding').value)
            patterns = list(self.get_parameter('ik_padding_link_patterns').value)

            if not builder.add_ground_plane(top_z, size, thickness):
                stamp['reason'] = 'ground plane not applied'
                self.get_logger().error(
                    "Ground plane could not be added to the planning scene; the IK sweep "
                    "will not see the floor.")
                return stamp
            time.sleep(0.3)

            padded = builder.apply_padding(padding, patterns) if padding > 0.0 else 0
            if padded < 0:
                stamp['reason'] = 'padding not applied'
                self.get_logger().error(
                    "Collision padding could not be applied; the IK sweep sees the floor "
                    "but not the arms' safety margin.")
                return stamp
            time.sleep(0.3)

            stamp = {'applied': True, 'ground_plane_z': top_z,
                     'ground_plane_size': size, 'ground_plane_thickness': thickness,
                     'collision_padding': padding, 'padded_links': padded,
                     'padding_link_patterns': patterns}
            self.get_logger().info(
                f"IK verification scene ready: floor top face at z={top_z:.3f} "
                f"({size}x{size}x{thickness} m box), {padding * 100:.1f} cm padding on "
                f"{padded} link(s) matching {patterns}. Reachability now matches what the "
                "executors will plan against.")
            return stamp
        finally:
            builder.destroy_node()

    def _make_reachability_fn(
            self, checker: ReachabilityChecker, label: str,
            workspace_gate: Optional[Callable[[Sequence[float]], bool]] = None
    ) -> Optional[ReachFn]:
        """
        Bir kol için aday -> (erişilebilir, eklem_çözümü) fonksiyonu üretir.

        /compute_ik hiç ayağa kalkmazsa None döner (kolun her şeye ulaşabildiği
        varsayılır, DOĞRULANMAMIŞ); doğrulanmamış bir plan asla doğrulanmış bir planla
        karıştırılmasın diye bu durum yüksek sesle günlüğe yazılır. `workspace_gate`
        (isteğe bağlı, callable(position)->bool), IK sorgusundan ÖNCE bu kolun izin verilen
        world çerçevesi bölgesi dışındaki adayları veto eder; kutu dışındaki bir aday bu
        kol için erişilemez olarak raporlanır.
        """
        if not checker.wait_for_service():
            self.get_logger().warning(
                f"/compute_ik unavailable for group '{checker.group_name}' ({label}). Its "
                "viewpoints "
                "will be selected WITHOUT reachability verification and may fail at execution "
                "time."
            )
            return None

        def _fn(
                candidate: Candidate, seed: Optional[JointState] = None,
                count_errors: bool = True) -> Tuple[bool, Optional[JointState]]:
            if workspace_gate is not None and not workspace_gate(candidate['position']):
                return False, None
            quat = Rotation.from_matrix(candidate['rotation']).as_quat()  # [x, y, z, w]
            return checker.check_ik(candidate['position'], quat, seed_joint_state=seed,
                                    count_errors=count_errors)

        return _fn

    def plan_callback(
            self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self.get_logger().info('Multi-robot planning requested.')
        t_start = time.monotonic()

        if not self.mesh_path or not os.path.exists(self.mesh_path):
            self.get_logger().error(f"Mesh path invalid or not set: '{self.mesh_path}'.")
            response.success = False
            response.message = f"Mesh path invalid: {self.mesh_path}"
            return response

        ur_checker = None
        kawa_checker = None
        try:
            # 1. Ağ çözümlemesi (tek kollu temel çözümle aynı).
            self.get_logger().info('Step 1: Mesh Analysis')
            t0 = time.monotonic()
            analyzer = MeshAnalyzer(self.mesh_path, scale=self.mesh_scale,
                                    logger=self.get_logger())
            analyzer.load_mesh()
            target_points, target_normals, _ = analyzer.sample_surface(
                num_points=self.get_parameter('target_sample_points').value)
            self.get_logger().info(f"Step 1 done in {time.monotonic() - t0:.2f}s.")

            # 2. Ortak aday havuzu + kaplama matrisi (tek kolluyla aynı).
            self.get_logger().info('Step 2: Viewpoint Generation (shared pool)')
            t0 = time.monotonic()
            vg = ViewpointGenerator(analyzer, config=self.camera_config, logger=self.get_logger())
            candidates = vg.generate_candidates(
                target_points, target_normals,
                num_base_points=self.get_parameter('num_base_points').value)
            coverage_matrix = vg.compute_coverage_matrix(candidates, target_points, target_normals)
            self.get_logger().info(
                f"Step 2 done in {time.monotonic() - t0:.2f}s. {len(candidates)} candidates.")

            # 3. Her kol için bir erişilebilirlik fonksiyonu kur. Kawasaki, UR gibi,
            #    kendi kamera optik çerçevesi için IK çözer.
            #    Sahne ÖNCE kurulur: aşağıdaki her check_ik ona göre yanıtlanır.
            ik_scene = self._prepare_ik_scene()
            avoid_collisions = self.get_parameter('ik_avoid_collisions').value
            if not avoid_collisions:
                self.get_logger().warning(
                    "ik_avoid_collisions=False: measuring PURE KINEMATIC reachability -- selected "
                    "viewpoints may be in collision at execution time (diagnostic only).")

            ur_checker = ReachabilityChecker(
                group_name=self.get_parameter('ur_group_name').value,
                base_frame=self.get_parameter('ur_base_frame').value,
                tool_frame=self.get_parameter('ur_tool_frame').value,
                avoid_collisions=avoid_collisions,
            )

            kawa_group = self.get_parameter('kawasaki_group_name').value
            kawa_base = self.get_parameter('kawasaki_base_frame').value
            kawa_cam = self.get_parameter('kawasaki_camera_frame').value
            kawa_tip = self.get_parameter('kawasaki_group_tip').value
            via_tip = self.get_parameter('kawasaki_ik_via_tip').value

            # Görüş ekseni gelenek düzeltmesi (camera_rgb_optic_frame üreticinin +Z'si
            # boyunca değil, +X boyunca bakar). Aşağıdaki İKİ IK yolunda da uygulanır.
            corr_rpy = list(self.get_parameter('kawasaki_view_axis_correction_rpy').value)
            C = Rotation.from_euler('xyz', corr_rpy).as_matrix()
            self.get_logger().info(
                f"Kawasaki view-axis correction rpy={corr_rpy} applied "
                "(generator +Z view -> camera_rgb_optic_frame +X view).")

            # offset = grup ucunun kamera çerçevesindeki pozu. Birim dönüşüm "kamera
            # çerçevesini doğrudan hedefle" (eski yol) demektir; gerçek bir dönüşüm ise
            # "sabit kamera->uç dönüşümünü uygulayıp grubun kendi ucu (link6) için IK çöz"
            # demektir; MoveIt bunu her zaman dikkate alır.
            offset = (np.eye(3), np.zeros(3))
            if via_tip:
                # TF'den DEĞİL, URDF zincirinden hesaplanır: gerçek kamera/link6 çerçeveleri
                # benzetim koşusunda TF'de yoktur; TF araması başarısız olur ve bizi sessizce
                # güvenilmez eski yola düşürürdü.
                offset = self._analytic_kawasaki_offset()
                self.get_logger().info(
                    f"Kawasaki IK via tip '{kawa_tip}' with analytic camera->tip offset "
                    f"(link6-in-camera t={offset[1].round(4).tolist()}); "
                    "camera placed exactly at each viewpoint.")

            kawa_checker = ReachabilityChecker(
                group_name=kawa_group,
                base_frame=kawa_base,
                # Uç üzerinden çözerken ik_link uca ayarlanır; aksi hâlde kamera çerçevesi
                # doğrudan hedeflenir (eski yol).
                tool_frame=(kawa_tip if via_tip else kawa_cam),
                avoid_collisions=avoid_collisions,
            )
            kawa_reach_fn = self._make_kawasaki_reach_fn(kawa_checker, offset, C)

            # Yalnızca UR için çalışma alanı kapısı (rayın arkası vetosu). Kawasaki'ye kapı
            # yoktur; böylece UR'nin meşru olarak ulaşamadığı adaylar yine Kawasaki'ye gider.
            ur_ws_gate = self._ur_workspace_gate()
            ur_reach_fn = self._make_reachability_fn(
                ur_checker, 'UR10e', workspace_gate=ur_ws_gate)
            robots = [
                {'name': 'ur', 'reachability_fn': ur_reach_fn},
                {'name': 'kawasaki', 'reachability_fn': kawa_reach_fn},
            ]

            # 4. Ortak havuz üzerinde açgözlü çok robotlu tahsis.
            self.get_logger().info('Step 3: Multi-robot allocation')
            t0 = time.monotonic()
            max_per = self.get_parameter('max_viewpoints_per_robot').value
            # Anahtarlar yukarıdaki `robots` içindeki 'name' alanlarıyla eşleşmelidir.
            per_arm_caps = {
                'ur': self.get_parameter('max_viewpoints_ur').value,
                'kawasaki': self.get_parameter('max_viewpoints_kawasaki').value,
            }
            allocator = RobotAllocator(
                coverage_threshold=self.coverage_threshold,
                max_per_robot=max_per,
                max_per_robot_by_name=per_arm_caps,
                balance=self.get_parameter('balance_load').value,
                min_new_points=self.get_parameter('min_new_points').value,
                min_marginal_coverage=self.get_parameter('min_marginal_coverage').value,
                logger=self.get_logger(),
            )
            assignments, final_coverage = allocator.allocate(candidates, coverage_matrix, robots)
            ur_checker.log_error_breakdown()
            kawa_checker.log_error_breakdown()
            self.get_logger().info(f"Step 3 done in {time.monotonic() - t0:.2f}s.")

            ur_vps = assignments.get('ur', [])
            kawa_vps = assignments.get('kawasaki', [])

            # Her kolu kısa bir Kartezyen ziyaret yoluna yeniden sırala; böylece kol açgözlü
            # keşif sırasıyla atlamak yerine komşu durakları tarar. Seçim değişmez; yalnızca
            # ziyaret sırası (ve dolayısıyla yürütücünün kullandığı 'rank' ile liste sırası)
            # değişir.
            if self.get_parameter('order_by_proximity').value:
                ur_anchor = self.get_parameter('ur_order_anchor').value
                kawa_anchor = self.get_parameter('kawasaki_order_anchor').value
                mode = str(self.get_parameter('order_mode').value)
                width = float(self.get_parameter('order_band_width_m').value)
                if mode == 'y_bands':
                    ur_vps = self._order_by_y_bands(ur_vps, ur_anchor, width)
                    kawa_vps = self._order_by_y_bands(kawa_vps, kawa_anchor, width)
                else:
                    ur_vps = self._order_by_proximity(ur_vps, anchor=ur_anchor)
                    kawa_vps = self._order_by_proximity(kawa_vps, anchor=kawa_anchor)
                how = (f"monotone Y sweep in {width:.2f} m bands (shortest path "
                       "inside each band)" if mode == 'y_bands'
                       else "shortest cartesian path (nearest-neighbour + 2-opt + Or-opt)")
                self.get_logger().info(
                    f"Viewpoints reordered by {how}. UR starts at '{ur_anchor}', "
                    f"Kawasaki at '{kawa_anchor}' (opposite ends of the chassis).")

            for i, vp in enumerate(ur_vps):
                vp['id'] = f'ur_vp_{i:03d}'
                vp['rank'] = i
            for i, vp in enumerate(kawa_vps):
                vp['id'] = f'kawa_vp_{i:03d}'
                vp['rank'] = i

            # Kimlikler var ve ziyaret sırası kesin; iki süreklilik geçişi çalışabilir.
            # Sıra önemlidir: önce her bakış noktası, öncülüne yakın bir IK dalına taşınır
            # (aynı poza ulaşan gerçekten farklı bir kol konfigürasyonu), SONRA o dal
            # içinde kalan tam tur farkları sarılır.
            self._rechain_ik(ur_vps, ur_reach_fn, 'UR tour')
            self._rechain_ik(kawa_vps, kawa_reach_fn, 'Kawasaki tour')
            self._unwind_tour(ur_vps, 'UR tour')
            self._unwind_tour(kawa_vps, 'Kawasaki tour')

            # 5. Planı kaydet.
            plan = {
                "coverage_achieved": float(final_coverage),
                "coverage_threshold": float(self.coverage_threshold),
                "objective": "multi_robot_max_coverage_balanced",
                "total_ur_viewpoints": len(ur_vps),
                "total_kawasaki_viewpoints": len(kawa_vps),
                "camera_config": self.camera_config,
                # Erişilebilirlik taramasının neye göre doğrulandığı. applied=False, aşağıdaki
                # bakış noktalarının yürütücülerin zeminine ve dolgusuna göre DENETLENMEDİĞİNİ,
                # bu yüzden bazılarının yürütme anında erişilemez olabileceğini gösterir.
                "ik_scene": ik_scene,
                "ur_viewpoints": [self._vp_to_dict(vp) for vp in ur_vps],
                "kawasaki_viewpoints": [self._vp_to_dict(vp) for vp in kawa_vps],
            }
            plan_file = os.path.expanduser(self.get_parameter('output_plan_file').value)
            os.makedirs(os.path.dirname(plan_file), exist_ok=True)
            with open(plan_file, 'w') as f:
                json.dump(plan, f, indent=2)

            total_time = time.monotonic() - t_start
            msg = (
                f"Multi-robot plan created. Coverage: {final_coverage * 100:.1f}%. "
                f"UR: {len(ur_vps)} viewpoints, Kawasaki: {len(kawa_vps)} viewpoints "
                f"(parallel run ~= {max(len(ur_vps), len(kawa_vps))} stops). Saved to {plan_file}."
            )
            self.get_logger().info(f"Planning finished in {total_time:.2f}s. {msg}")
            response.success = True
            response.message = msg

        except Exception as e:
            self.get_logger().error(f"Planning failed after {time.monotonic() - t_start:.2f}s: "
                                    f"{e}")
            self.get_logger().error(traceback.format_exc())
            response.success = False
            response.message = f"Error: {e}"
        finally:
            for checker in (ur_checker, kawa_checker):
                if checker is not None:
                    try:
                        checker.destroy_node()
                    except Exception as e:
                        self.get_logger().warning("Failed to destroy reachability checker node: "
                                                  f"{e}")

        return response

    # ------------------------------------------------------------------ #
    # Saklanan IK çözümlerinin 2*pi sarılması.
    # ------------------------------------------------------------------ #
    def _robot_description_cb(self, msg: String) -> None:
        if self._urdf_xml is None:
            self._urdf_xml = msg.data

    def _get_joint_limits(self, timeout: float = 5.0) -> Dict[str, JointLimit]:
        """
        Kalıcı (latched) URDF'i bir kez ayrıştırır.

        Boş bir tablo "hiçbir şeyi sarma" demektir; bu güvenli bir geri çekilmedir:
        hedefler IK'nın ürettiği gibi kalır.
        """
        if self._joint_limits is not None:
            return self._joint_limits
        deadline = time.monotonic() + timeout
        while self._urdf_xml is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if self._urdf_xml is None:
            self.get_logger().warning(
                'No /robot_description available -- joint limits unknown, so the stored '
                'IK solutions are left un-unwound.')
            self._joint_limits = {}
            return self._joint_limits
        try:
            self._joint_limits = parse_joint_limits(self._urdf_xml)
            wrappable = sorted(n for n, v in self._joint_limits.items() if v.wrappable)
            self.get_logger().info(
                f'Joint limits parsed for {len(self._joint_limits)} joints; '
                f'{len(wrappable)} can be unwound: {wrappable}')
        except Exception as e:
            self.get_logger().warning(f'Could not parse /robot_description ({e}); '
                                      'IK solutions left un-unwound.')
            self._joint_limits = {}
        return self._joint_limits

    def _joint_distance(self, sol_a: Optional[JointState], sol_b: Optional[JointState]) -> float:
        """
        İki IK çözümü arasında hareket etmenin maliyeti [radyan].

        Dönel eklemler EN KISA eşdeğerleriyle sayılır; böylece sarma geçişinin zaten
        kaldırdığı saf bir 2*pi fark, dal değişikliği gibi görünmez.

        DOĞRUSAL eklemler (Kawasaki'nin `world_to_agv` rayı) da sayılır; `rechain_rail_weight`
        rad/m ile çevrilir. Eskiden tamamen atlanıyorlardı; bu Kawasaki için yanlıştır:
        grubu FAZLADAN eksenlidir (ray + 6 eklem), yani ray her IK çözümünün serbest bir
        parametresidir ve AGV'yi bir metre süren bir aday bedava puan alıyordu.
        2026-08-03 planında ölçüldü: Kawasaki turu rayı 8 bakış noktası boyunca 3.33 m sürdü.
        """
        if sol_a is None or sol_b is None:
            return float('inf')
        limits = self._joint_limits or {}
        rail_w = float(self.get_parameter('rechain_rail_weight').value)
        b = dict(zip(sol_b.name, sol_b.position))
        total = 0.0
        for n, pa in zip(sol_a.name, sol_a.position):
            if n not in b:
                continue
            lim = limits.get(n)
            d = abs(float(b[n]) - float(pa))
            if lim is not None and not lim.is_revolute:
                total += d * rail_w
                continue
            total += min(d, abs(d - 2.0 * np.pi))
        return total

    def _seed_variants(
            self, prev: Optional[JointState], current: Optional[JointState],
            vp_id: str) -> List[JointState]:
        """
        Bir bakış noktasının IK yeniden çözümü için denenecek tohumlar, en olası başta.

        AYNI tohumlu sorguyu tekrarlamak yalnızca çözücü rastgele yeniden başlıyorsa yeni
        dallar örnekler; bu `mode: global` ile pick_ik (UR) için doğrudur ama KDL
        (Kawasaki grubu) için DEĞİLDİR: KDL yerel bir çözücüdür ve her seferinde aynı
        yanıtı verir; bu yüzden `rechain_ik_attempts` tek başına Kawasaki'ye tam olarak
        bir aday kazandırır. Bu çeşitlemeler onu çözüm uzayının gerçekten farklı
        bölgelerine iter:

          - `prev`, düz zincirleme tohum, ve `current`; böylece mevcut çözüme her zaman
            yeniden yakınsanabilir.
          - RAY DENEMELERİ: rayı +/- `rechain_rail_probe` kaydırılmış prev. Bu fazladan
            serbestlik çözümünün ayarıdır: aynı kamera pozuna bir dizi AGV konumundan
            ulaşılabilir ve en yakını genellikle tohumsuz bir çözümün seçtiği değildir.
          - TABAN YALPA DENEMELERİ: joint1'i +/- pi çevrilmiş prev. 2026-08-03 planında
            tek hamlede 248 derecelik bir joint1 salınımı vardı; joint1 tam olarak +/-180
            aralığında olduğu için sarma ona asla dokunamaz, yalnızca farklı bir dal dokunabilir.

        Her çeşitleme URDF sınırlarına kırpılır; böylece bir deneme asla aralık dışı bir
        tohum öneremez.
        """
        limits = self._joint_limits or {}

        def clamp(name: str, value: float) -> float:
            lim = limits.get(name)
            if lim is None:
                return value
            return float(np.clip(value, lim.lower + 1e-6, lim.upper - 1e-6))

        def shifted(base: Optional[JointState], changes: Dict[str, float]) -> Optional[JointState]:
            if base is None:
                return None
            out = JointState()
            out.name = list(base.name)
            pos = [float(v) for v in base.position]
            touched = False
            for jname, delta in changes.items():
                if jname not in out.name:
                    continue
                i = out.name.index(jname)
                new = clamp(jname, pos[i] + delta)
                if abs(new - pos[i]) > 1e-4:
                    pos[i] = new
                    touched = True
            if not touched:
                return None
            out.position = pos
            return out

        rail = str(self.get_parameter('rechain_rail_joint').value)
        probe = float(self.get_parameter('rechain_rail_probe').value)
        seeds = [prev, current]
        if probe > 0.0:
            seeds += [shifted(prev, {rail: probe}), shifted(prev, {rail: -probe})]
        if bool(self.get_parameter('rechain_yaw_probe').value):
            yaw = str(self.get_parameter('rechain_yaw_joint').value)
            seeds += [shifted(prev, {yaw: np.pi}), shifted(prev, {yaw: -np.pi})]
        return [s for s in seeds if s is not None]

    def _rechain_ik(self, vps: List[Candidate], reach_fn: Optional[ReachFn], label: str) -> None:
        """
        Her bakış noktasının IK'sını bir öncekinin çözümüyle TOHUMLAYARAK yeniden çözer.

        Sonuç yalnızca eklem uzayında elimizdekinden daha yakınsa tutulur.

        Neden var: her pozun ona bambaşka kol konfigürasyonlarından ulaşan birkaç IK dalı
        vardır (omuz/dirsek/bilek dönüşleri). Tahsis her bakış noktasını bağımsız çözer ve
        pick_ik `mode: global` ile, yani rastgele yeniden başlatmalarla çalışır; bu yüzden
        komşu bakış noktaları sık sık farklı dallara düşer. Kol da birkaç santimetre gitmek
        için yüzlerce derece döner; bu, 2*pi sarmanın düzeltebileceği bir şey DEĞİLDİR,
        çünkü konfigürasyonlar gerçekten farklıdır.

        Burada bakış noktası POZU asla değişmez, yalnızca ona hangi çözümün ulaştığı
        değişir; dolayısıyla kaplama etkilenmez. Yeniden çözüm başarısız olursa ya da daha
        iyi değilse bakış noktası özgün çözümünü korur; yani bu yalnızca yolu kısaltabilir.
        """
        if not vps or reach_fn is None or not self.get_parameter('rechain_ik').value:
            return
        attempts = max(1, int(self.get_parameter('rechain_ik_attempts').value))
        self._get_joint_limits()
        rail_before = self._tour_rail_travel(vps)

        improved = 0
        saved = 0.0
        prev = vps[0].get('joint_solution')
        for vp in vps[1:]:
            current = vp.get('joint_solution')
            if prev is None:
                prev = current
                continue
            best, best_d = current, self._joint_distance(prev, current)
            # Farklı bir dala ulaşmanın iki yolu: sorguyu tekrarla (yalnızca çözücü rastgele
            # yeniden başlıyorsa, yani UR'deki pick_ik/global ile bir şey örnekler) ve
            # TOHUMU değiştir (Kawasaki'de KDL'yi oynatan tek şey). İkisi de denenir ve bir
            # sonuç yalnızca kesinlikle daha yakınsa tutulur; bu yüzden bu işlem bir turu
            # asla kötüleştiremez.
            for seed in self._seed_variants(prev, current, vp.get('id', '?')):
                for _ in range(attempts):
                    ok, sol = reach_fn(vp, seed=seed, count_errors=False)
                    if not ok or sol is None:
                        continue
                    d = self._joint_distance(prev, sol)
                    if d < best_d:
                        best, best_d = sol, d
            if best is not current and best is not None:
                before = self._joint_distance(prev, current)
                self.get_logger().info(
                    f"  {vp.get('id', '?')}: closer IK branch found -- "
                    f"{np.rad2deg(before):.0f} -> {np.rad2deg(best_d):.0f} deg from the "
                    "previous viewpoint.")
                vp['joint_solution'] = best
                improved += 1
                saved += before - best_d
            prev = vp.get('joint_solution')

        rail_after = self._tour_rail_travel(vps)
        if improved:
            self.get_logger().info(
                f'{label}: re-seeded {improved} viewpoint(s) onto a nearer IK branch, '
                f'removing {np.rad2deg(saved):.0f} deg of travel.')
        else:
            self.get_logger().info(f'{label}: no viewpoint had a nearer IK branch.')
        if rail_before is not None and rail_after is not None:
            self.get_logger().info(
                f'{label}: rail travel {rail_before:.3f} -> {rail_after:.3f} m.')

    def _tour_rail_travel(self, vps: List[Candidate]) -> Optional[float]:
        """
        Sıralı bir tur boyunca toplam doğrusal (AGV rayı) yol [metre].

        Turda doğrusal eklem yoksa (yani UR için) None döner.
        """
        limits = self._joint_limits or {}
        rail = str(self.get_parameter('rechain_rail_joint').value)
        vals = []
        for vp in vps:
            sol = vp.get('joint_solution')
            if sol is None or rail not in sol.name:
                return None
            vals.append(float(sol.position[list(sol.name).index(rail)]))
        lim = limits.get(rail)
        if lim is not None and lim.is_revolute:
            return None
        return float(np.abs(np.diff(vals)).sum()) if len(vals) > 1 else 0.0

    def _unwind_tour(self, vps: List[Candidate], label: str) -> None:
        """
        SIRALI bir turun eklem açılarını bir önceki durağa göre sarar.

        Turu yürür ve her bakış noktasının eklem açılarını, URDF sınırları içinde
        ÖNCEKİ bakış noktasına en yakın 2*pi eşdeğerleriyle yeniden ifade eder.

        İlk bakış noktası IK'nın ürettiği gibi bırakılır: burada onun için anlamlı bir
        referans yoktur ve yürütücü onu çalışma anında kolun gerçek ölçülen pozuna göre
        zaten sarar. Pozlar boyunca değişmez (theta ve theta +/- 2*pi her bağlantıyı aynı
        yere koyar); bu yüzden bu işlem kaplamayı, erişilebilirliği ya da çarpışmaları
        değiştiremez, yalnızca kolun duraklar arasında ne kadar yol aldığını değiştirir.
        """
        if not vps or not self.get_parameter('unwind_joint_goals').value:
            return
        limits = self._get_joint_limits()
        if not limits:
            return
        margin = float(self.get_parameter('wrap_limit_margin').value)
        min_gain = float(self.get_parameter('wrap_min_gain').value)

        reference = None
        total_saved = 0.0
        rewritten = 0
        for vp in vps:
            sol = vp.get('joint_solution')
            if sol is None:
                continue
            names = list(sol.name)
            positions = [float(x) for x in sol.position]
            if reference is not None:
                positions, changes = wrap_to_reference(
                    positions, names, reference, limits, margin=margin, min_gain=min_gain)
                if changes:
                    rewritten += len(changes)
                    total_saved += sum(c[3] for c in changes)
                    self.get_logger().info(
                        f"  {vp.get('id', '?')}: {describe_changes(changes)}")
                    sol.position = positions
            reference = dict(zip(names, positions))

        if rewritten:
            self.get_logger().info(
                f'{label}: unwound {rewritten} joint goal(s) across the tour, removing '
                f'{np.rad2deg(total_saved):.0f} deg of needless rotation.')
        else:
            self.get_logger().info(f'{label}: no joint goals needed unwinding.')

    @staticmethod
    def _order_by_proximity(vps: List[Candidate], anchor: str = 'coverage') -> List[Candidate]:
        """
        Bakış noktalarını EN KISA Kartezyen ziyaret yoluna göre yeniden sıralar.

        Tarama yönü yok sayılır.

        Yalnızca SIRA değişir; tam olarak aynı bakış noktası kümesi döndürülür.

        anchor, BAŞLANGIÇ bakış noktasını seçer:
          'coverage' -> 0. indeks (açgözlü yöntemin başa koyduğu en yüksek kaplamalı),
          'max_y'    -> en büyük Y'li bakış noktası (şasenin ÖNÜ),
          'min_y'    -> en küçük Y'li bakış noktası (şasenin ARKASI).

        Neyin eniyilendiğine dikkat edin: saf bir en kısa yol turu 3B'de daha ucuzsa
        şasenin başına geri yürümekte serbesttir ve bunu yapar: 2026-09-08 şase planında
        UR'nin 20 hamlesinin 9'u tarama yönünden UZAKLAŞTI ve uzak uca (y=-0.49) vardıktan
        sonra tur y=+1.25'e kadar geri tırmandı. Tek yönlü tarama için 'y_bands' kullanın.
        """
        n = len(vps)
        if n <= 2:
            return list(vps)
        pos = np.array([np.asarray(v['position'], dtype=np.float64) for v in vps])
        dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

        # Çapa (başlangıç) indeksini istenen kurala göre seç.
        if anchor == 'max_y':
            start = int(np.argmax(pos[:, 1]))
        elif anchor == 'min_y':
            start = int(np.argmin(pos[:, 1]))
        else:
            # 'coverage': açgözlü yöntem en iyi kaplamalı bakış noktasını zaten başa koydu
            start = 0

        return [vps[t] for t in MultiRobotPlannerNode._short_path(dist, start)]

    @staticmethod
    def _order_by_y_bands(
            vps: List[Candidate], anchor: str = 'max_y',
            band_width: float = 0.30) -> List[Candidate]:
        """
        Y boyunca tek yönlü tarama: bir sonrakine geçmeden önce bir Y bandını bitirir.

        Kol şasenin çapa ucundan başlar, Y'si ilk `band_width` metreye düşen her bakış
        noktasını ziyaret eder, sonra sonraki bandı, vb.; bıraktığı bir banda asla geri
        dönmez. Bant içindeki duraklar, bir önceki bandın bittiği yere en yakın bakış
        noktasından başlayarak 'proximity' ile aynı en kısa yol yordamıyla sıralanır;
        böylece yerel hareket kısa kalırken genel ilerleme tek yönlü olur.

        Bütün denge band_width'tedir: 0'da katı bir Y sıralamasına (Y'de olası en kısa,
        X/Z'de en uzun tarama), şase uzunluğundan büyükken düz 'proximity'ye dönüşür.
        Kameranın tek bir Y konumundan kapladığı bölgenin derinliği kadar olmalıdır;
        böylece gerçekten birbirine ait duraklar hâlâ mantıklı sıralanabilir.

        anchor yönü verir: 'max_y' önden arkaya (azalan Y), 'min_y' arkadan öne tarar.
        Başka bir değerin yönü yoktur; çağıran bunun yerine 'proximity' kullanmalıdır.
        """
        n = len(vps)
        if n <= 2:
            return list(vps)
        pos = np.array([np.asarray(v['position'], dtype=np.float64) for v in vps])
        y = pos[:, 1]
        descending = (anchor != 'min_y')
        width = max(float(band_width), 1e-6)
        # 0. bant, tarama hangi yöne olursa olsun her zaman taramanın BAŞLADIĞI banttır.
        edge = y.max() if descending else y.min()
        band = np.floor((np.abs(y - edge)) / width).astype(int)

        order, prev = [], None
        for b in sorted(set(band.tolist())):
            idx = [i for i in range(n) if band[i] == b]
            if len(idx) == 1:
                order.append(idx[0])
                prev = pos[idx[0]]
                continue
            sub = pos[idx]
            d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
            if prev is None:
                # İlk durak: uç Y'deki bakış noktası; böylece tarama gerçekten çapanın
                # adlandırdığı şase ucundan başlar.
                start = int(np.argmax(sub[:, 1]) if descending else np.argmin(sub[:, 1]))
            else:
                start = int(np.argmin(np.linalg.norm(sub - prev, axis=1)))
            tour = MultiRobotPlannerNode._short_path(d, start)
            order.extend(idx[k] for k in tour)
            prev = sub[tour[-1]]
        return [vps[i] for i in order]

    @staticmethod
    def _short_path(dist: np.ndarray, start: int) -> List[int]:
        """
        `start`'a sabitlenmiş, `dist` üzerinde açık en kısa yol sırası.

        En yakın komşu tohumu, ardından 2-opt + Or-opt iyileştirmesi (ikisi de turun 0.
        indeksini asla taşımaz). `dist` içine indeks listesi döndürür. n küçüktür; bu
        yüzden anında biter.
        """
        n = dist.shape[0]
        if n <= 2:
            return list(range(n)) if start == 0 else [start] + [i for i in range(n) if i != start]

        # Çapadan başlayan en yakın komşu tohum turu.
        unvisited = set(range(n))
        tour = [start]
        unvisited.discard(start)
        while unvisited:
            last = tour[-1]
            nxt = min(unvisited, key=lambda j: dist[last, j])
            tour.append(nxt)
            unvisited.discard(nxt)

        # 2-opt: hiçbir takas fayda sağlamayana kadar açık yolun kesişen kenarlarını çöz.
        improved = True
        while improved:
            improved = False
            for a in range(n - 1):
                for b in range(a + 1, n - 1):
                    i, j = tour[a], tour[a + 1]
                    k, m = tour[b], tour[b + 1]
                    if dist[i, k] + dist[j, m] + 1e-9 < dist[i, j] + dist[k, m]:
                        tour[a + 1:b + 1] = tour[a + 1:b + 1][::-1]
                        improved = True

        # Or-opt: kısa zincirleri (uzunluk 1-3) daha ucuz bir yere taşı. 2-opt yalnızca
        # bir parçayı TERS ÇEVİREBİLİR; bu yüzden bir bakış noktası ana taramadan uzakta
        # durduğunda oluşan uzun ileri-geri hamleleri bırakır. Or-opt o başıboş bakış
        # noktasını gerçek komşusunun yanına taşır ve yolu daha da kısaltır.
        # tour[0] (en yüksek kaplamalı çapa) asla taşınmaz. n küçük olduğu için aday
        # başına O(n^2) tam yeniden hesap önemsizdir.
        def path_len(t: Sequence[int]) -> float:
            return sum(dist[t[p], t[p + 1]] for p in range(len(t) - 1))

        for seg_len in (1, 2, 3):
            if seg_len >= n:
                break
            improved = True
            while improved:
                improved = False
                base = path_len(tour)
                for a in range(1, n - seg_len + 1):  # çapayı asla kaldırma
                    seg = tour[a:a + seg_len]
                    rest = tour[:a] + tour[a + seg_len:]
                    for b in range(1, len(rest) + 1):  # çapanın önüne asla ekleme
                        cand = rest[:b] + seg + rest[b:]
                        if path_len(cand) + 1e-9 < base:
                            tour = cand
                            improved = True
                            break
                    if improved:
                        break

        return tour

    @staticmethod
    def _vp_to_dict(vp: Candidate) -> Dict[str, Any]:
        d = {
            "id": vp['id'],
            "robot": vp.get('robot'),
            "position": np.asarray(vp['position']).tolist(),
            "rotation": np.asarray(vp['rotation']).tolist(),
            "rank": vp.get('rank'),
            "global_order": vp.get('global_order'),
            "new_points_covered": vp.get('new_points_covered'),
            "total_points_visible": vp.get('total_points_visible'),
            "cumulative_coverage": vp.get('cumulative_coverage'),
        }
        # JointState.name bir list[str]'dir; .position ise json modülünün
        # serileştiremediği bir array.array'dir; ikisini de düz listeye çevir.
        sol = vp.get('joint_solution')
        if sol is not None:
            d['joint_names'] = list(sol.name)
            d['joint_positions'] = list(sol.position)
        return d


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = MultiRobotPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
