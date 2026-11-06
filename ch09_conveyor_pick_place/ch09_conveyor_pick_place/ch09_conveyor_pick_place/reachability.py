#!/usr/bin/env python3
"""
Hedefe GERÇEKTEN gidilebilir mi? MoveIt'e sorar (/compute_ik).

NEDEN VAR: ER 2'nin gösterdiği nokta doğru olabilir ama kolun oraya gitmesi
imkânsız olabilir - ve bunu hareket sırasında öğrenmek pahalıdır. 11 Ağu
2026'da yaşandı: ER, toolkit rafında üstten İKİNCİ sıradaki gözü gösterdi.
Nokta doğruydu; ama o gözün tavanı var (üstündeki raf 211 mm yukarıda) ve
biz yukarıdan dik iniyoruz - emme kabının sapı rafın içinden geçiyordu.
MoveIt bunu "Planning failed! FAILURE" diye reddetti, ama bu ancak parça
KAVRANDIKTAN sonra ortaya çıktı: kol havada parçayla asılı kaldı.

Bu modül aynı soruyu hareketten önce sorar. İki aşamalı, çünkü "olmuyor"un
iki farklı sebebi var ve ikisi farklı şeyler söyler:

    avoid_collisions=True  başarısız  ->  ya erişim yok ya çarpışma
    avoid_collisions=False başarılı   ->  erişim VAR, sorun ÇARPIŞMA
                           başarısız  ->  gerçekten erişilemiyor (kol yetmiyor)

İkinci çağrı yalnızca birincisi başarısız olduğunda yapılır; başarılı yolda
ek maliyet yoktur.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, TYPE_CHECKING

from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK, GetStateValidity
from sensor_msgs.msg import JointState

if TYPE_CHECKING:  # yalnızca tip ipuçları için; çalışma anında içe aktarılmaz
    from rclpy.callback_groups import CallbackGroup
    from rclpy.node import Node

# moveit_msgs/MoveItErrorCodes
SUCCESS = 1
NO_IK_SOLUTION = -31
GOAL_IN_COLLISION = -12


@dataclass
class ReachResult:
    ok: bool
    code: int
    reason: str

    def __bool__(self) -> bool:  # `if result:` okunaklı olsun
        return self.ok


class ReachabilityChecker:
    """/compute_ik üzerinden poz doğrulaması. Servis yoksa sessizce devre dışı."""

    def __init__(
        self,
        node: Node,
        group_name: str,
        ik_link: str,
        planning_frame: str,
        service: str = "/compute_ik",
        timeout_sec: float = 3.0,
        callback_group: Optional[CallbackGroup] = None,
        validity_service: str = "/check_state_validity",
    ) -> None:
        self._node = node
        self._group = group_name
        self._link = ik_link
        self._frame = planning_frame
        self._timeout = float(timeout_sec)
        self._client = node.create_client(
            GetPositionIK, service, callback_group=callback_group
        )
        # Yalnızca TEŞHİS için: "çarpışma" demek yetmiyor, NEYE çarptığını
        # bilmeden hiçbir şey düzeltilemiyor. IK'nın çarpışmasız çözümünü bu
        # servise verip temas eden link çiftlerini isimleriyle alıyoruz.
        self._validity = node.create_client(
            GetStateValidity, validity_service, callback_group=callback_group
        )
        self._available: Optional[bool] = None

    def available(self) -> bool:
        """Servis var mı? Bir kez sorulur; yoksa doğrulama tümden atlanır."""
        if self._available is None:
            # Tek seferlik: keşfin oturması için cömert davran. Kısa tutulunca
            # ÇALIŞAN bir move_group bile bulunamayıp doğrulama sessizce
            # kapanabiliyor - en çok ihtiyaç duyulan anda.
            self._available = self._client.wait_for_service(timeout_sec=5.0)
            if not self._available:
                self._node.get_logger().warn(
                    "/compute_ik yok - hedef doğrulaması KAPALI. Ulaşılamaz bir "
                    "hedef ancak hareket denenirken anlaşılacak."
                )
        return bool(self._available)

    def _call(
        self,
        position: Sequence[float],
        quat_xyzw: Sequence[float],
        seed: Optional[JointState],
        avoid_collisions: bool,
    ) -> Tuple[Optional[int], Optional[JointState]]:
        """(hata kodu, çözüm JointState) döndürür. Kod None ise servis susmuş."""
        request = GetPositionIK.Request()
        request.ik_request.group_name = self._group
        request.ik_request.ik_link_name = self._link
        request.ik_request.avoid_collisions = avoid_collisions
        request.ik_request.timeout.sec = int(self._timeout)
        request.ik_request.timeout.nanosec = int(
            (self._timeout - int(self._timeout)) * 1e9
        )
        # is_diff=True ŞART. False olduğunda MoveIt verilen robot_state'i
        # sahnedeki durumun YERİNE koyar ve İLİŞTİRİLMİŞ NESNELERİ DÜŞÜRÜR -
        # yani taşınan parça çarpışma denetimine hiç girmez. Doğrulama o zaman
        # planlayıcıdan farklı bir dünyaya bakar ve iyimser cevap verir.
        #
        # ÖLÇÜLDÜ (13 Ağu 2026, canlı hücre, APPROACH_PLACE (0.924, 1.660, 1.174)):
        #     is_diff=False -> IK 1 (uygun)      <-- yanlış
        #     is_diff=True  -> IK -31            <-- doğru; temas:
        #                      ur10e_stackable_bin<->gemini_payload
        # Görevin başarısızlığı buydu: doğrulama "gidilebilir" dedi, kol kutuyu
        # aldı, sonra move_group aynı poza plan üretemedi ("Planning failed!
        # FAILURE") - çünkü ELİNDEKİ KUTU rafa giriyordu. Hata mesajı hedefi
        # suçluyordu, oysa hedef yükle birlikte geçersizdi.
        request.ik_request.robot_state.is_diff = True
        if seed is not None:
            # Tohum ÖNEMLİ: UR10e'de aynı poz için birden çok IK dalı var ve
            # tohumsuz çözüm, kolun o an bulunduğu duruştan çok uzak bir dala
            # düşebiliyor. O dal geçerli çıksa bile planlayıcı oraya gitmiyor.
            request.ik_request.robot_state.joint_state = seed

        pose = PoseStamped()
        pose.header.frame_id = self._frame
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = (
            float(q) for q in quat_xyzw)
        request.ik_request.pose_stamped = pose

        future = self._client.call_async(request)
        # spin ETMİYORUZ: düğümü bizim executor'ımız zaten spin ediyor ve
        # rclpy.spin_once burada düğümü global executor'a kaptırırdı
        # (bkz. pick_place_node.install_spin_once_guard).
        deadline = time.monotonic() + self._timeout + 2.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        response = future.result()
        if response is None:
            return None, None
        return int(response.error_code.val), response.solution.joint_state

    def _contacts(self, state: Optional[JointState], limit: int = 4) -> str:
        """
        Bu duruşta temas eden link çiftlerini isimleriyle döndürür.

        Teşhis amaçlı: başarısız olursa boş string döner ve arayan taraf
        eskisi gibi çalışır. Doğrulamanın kendisi buna bağlı DEĞİL.
        """
        if state is None or not self._validity.service_is_ready():
            return ""
        request = GetStateValidity.Request()
        request.group_name = self._group
        # _call ile aynı sebep: is_diff olmadan taşınan parça denetimden düşer
        # ve temas listesi onu HİÇ göstermez. Asıl çarpanı gizleyen buydu.
        request.robot_state.is_diff = True
        request.robot_state.joint_state = state
        future = self._validity.call_async(request)
        deadline = time.monotonic() + 2.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        response = future.result()
        if response is None or not response.contacts:
            return ""
        pairs = []
        for contact in response.contacts:
            pair = f"{contact.contact_body_1}<->{contact.contact_body_2}"
            if pair not in pairs:
                pairs.append(pair)
        extra = "" if len(pairs) <= limit else f" (+{len(pairs) - limit})"
        return ", ".join(pairs[:limit]) + extra

    def check(
        self,
        position: Sequence[float],
        quat_xyzw: Sequence[float],
        seed: Optional[JointState] = None,
        label: str = "",
    ) -> ReachResult:
        if not self.available():
            return ReachResult(True, SUCCESS, "doğrulama kapalı")

        code, _ = self._call(position, quat_xyzw, seed, avoid_collisions=True)
        if code is None:
            # Servis yanıt vermedi: hedefi SUÇLAMA. Doğrulanamayan bir hedefi
            # reddetmek, çalışan bir görevi altyapı gecikmesi yüzünden iptal
            # etmek olurdu.
            return ReachResult(True, SUCCESS, "IK servisi yanıt vermedi, geçildi")
        if code == SUCCESS:
            return ReachResult(True, code, "uygun")

        free, state = self._call(position, quat_xyzw, seed, avoid_collisions=False)
        if free == SUCCESS:
            contacts = self._contacts(state)
            detail = f", temas: {contacts}" if contacts else ""
            return ReachResult(
                False, code,
                f"çarpışma (kol oraya uzanabiliyor ama bir şeye giriyor, "
                f"IK {code}{detail})",
            )
        return ReachResult(
            False, code,
            f"erişim yok (kol o poza hiç ulaşamıyor, IK {code})",
        )
