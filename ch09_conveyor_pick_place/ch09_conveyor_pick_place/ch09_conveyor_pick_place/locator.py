#!/usr/bin/env python3
"""
GeminiLocator: "şunu bul" -> world frame'inde 3D nokta + yüzey normali.

Hem perception_node hem pick_place_node bunu kullanır. Ayrı bir sınıf olmasının
sebebi, iki düğümün topic üzerinden istek/yanıt el sıkışması yapmak zorunda
kalmaması: pick and place akışı sorgunun cevabını beklemek zorunda ve bunu
topic'lerle yapmak yarış koşulu üretir.

ER 2'ye verilen görüntü `er_image_source` ile seçilir:
  render : derinlikten üretilen görüntü (VARSAYILAN). Gerçek SICK Visionary-T
           Mini'nin RGB'si olmadığı için sim ve gerçek arasında taşınabilir tek
           seçenek budur. Bkz. depth_render.py.
  rgb    : Gazebo'nun RGB kamerası. Yalnız sim'de vardır; gerçeğe taşınmaz.

ÖNEMLİ: locate() bloklayıcıdır (ağ çağrısı + TF bekleme). Executor thread'inden
değil, çağıran tarafın kendi worker thread'inden çağrılmalıdır.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import tf2_ros

from . import depth_render
from . import telemetry
from .er_client import Detection, ERClientBase
from .geometry import point_from_cloud, point_from_depth, rotate_vector, transform_point
from .grasp import (
    approach_quaternion, fit_surface, gather_patch_points, matrix_to_quaternion,
    snap_normal_to_axis,
)
from .payload import measure_payload_box, organized_xyz
from .image_utils import imgmsg_to_bgr


# OnRobot VGC10 kauçuk emme kabının ağız çapı. Yalnız teşhis logunda kullanılır:
# eğik oturan kabın bir kenarında kalan boşluğu mm cinsinden yazmak için.
_CUP_DIAMETER_M = 0.039

# Gazebo köprüsü ve SICK sürücüsü sensörleri best-effort yayınlıyor; RELIABLE
# abone olursak hiçbir mesaj gelmez ve düğüm sessizce boş kalır.
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class GeminiLocator:
    def __init__(
        self,
        node: Node,
        er_client: ERClientBase,
        image_topic: str = "/sim/image",
        cloud_topic: str = "/sim/pointcloud",
        depth_topic: str = "/sim/depth/image",
        camera_info_topic: str = "/sim/camera_info",
        intensity_topic: str = "/intensity",
        world_frame: str = "world",
        camera_frame_override: str = "",
        deprojection: str = "auto",
        sample_window: int = 5,
        tf_timeout_sec: float = 1.0,
        er_image_source: str = "render",
        render_mode: str = "normals",
        render_min_m: float = 0.3,
        render_max_m: float = 4.0,
        render_height_lo_m: float = -0.005,
        render_height_hi_m: float = 0.12,
        depth_scale_m: float = depth_render.DEFAULT_DEPTH_SCALE_M,
        depth_is_radial: bool = False,
        publish_er_image: bool = True,
        er_image_rate_hz: float = 2.0,
        # yüzey / kavrama
        patch_radius_px: int = 12,
        patch_depth_band: float = 0.05,
        patch_min_points: int = 25,
        tool_approach_vector: Sequence[float] = (0.0, 0.0, 1.0),
        normal_snap_deg: float = 15.0,
        # Yalnızca TEŞHİS LOGU için: oturtulmamış bir normalin yaklaşma
        # pozunu ne kadar yana kaydırdığını mm cinsinden yazabilmek gerekiyor.
        # Harekete girmez; pick_place_node kendi approach_distance'ını geçer.
        approach_hint_m: float = 0.15,
        # taşınacak parçanın boyut ölçümü
        measure_payload: bool = True,
        payload_margin_m: float = 0.005,
        payload_min_height_m: float = 0.004,
        payload_max_height_m: float = 0.30,
        payload_max_span_m: float = 0.45,
    ) -> None:
        self._node = node
        self._er = er_client
        self._world_frame = world_frame
        self._camera_frame_override = camera_frame_override
        self._deprojection = deprojection
        self._sample_window = int(sample_window)
        self._tf_timeout_sec = float(tf_timeout_sec)

        self._er_image_source = er_image_source
        self._render_mode = render_mode
        self._render_min_m = float(render_min_m)
        self._render_max_m = float(render_max_m)
        # relief yükseklik penceresi. Tarama pozları arasında değişir: bant
        # üstündeki kutu için düzlemin altını görmek gerekmez, toolkit'te BOŞ
        # GÖZ bir çukur olduğu için lo negatif olmalı. Önizleme thread'i de
        # aynı değeri okuduğundan RViz'de tam olarak ER'nin gördüğü kare çıkar.
        self._render_height_lo_m = float(render_height_lo_m)
        self._render_height_hi_m = float(render_height_hi_m)
        # Derinliğin BİRİMİ ve KONVANSİYONU. Sim ile gerçek arasındaki gerçek
        # modalite farkı burada: Gazebo 32FC1/metre/planar yayınlıyor, SICK
        # Visionary-T Mini ham 16UC1'i 0.25 mm/LSB ve IŞIN BOYU mesafe olarak
        # yayınlıyor. İkisini de sabit kabul etmek sim'de görünmüyor, gerçekte
        # ER'ye okunmaz bir kare gönderiyor. Bkz. depth_render.py.
        self._depth_scale_m = float(depth_scale_m)
        self._depth_is_radial = bool(depth_is_radial)

        self._tool_approach_vector = tool_approach_vector
        self._normal_snap_deg = float(normal_snap_deg)
        self._approach_hint = float(approach_hint_m)
        self._measure_payload_enabled = bool(measure_payload)
        self._payload_margin_m = float(payload_margin_m)
        self._payload_min_height_m = float(payload_min_height_m)
        self._payload_max_height_m = float(payload_max_height_m)
        self._payload_max_span_m = float(payload_max_span_m)
        self._patch_radius_px = int(patch_radius_px)
        self._patch_depth_band = float(patch_depth_band)
        self._patch_min_points = int(patch_min_points)

        self._lock = threading.Lock()
        # Model çağrısı sayacı: kaydedilen kareyi sonuç satırıyla eşler.
        self._query_seq = 0
        self._image: Optional[Image] = None
        self._cloud: Optional[PointCloud2] = None
        self._depth: Optional[Image] = None
        self._intensity: Optional[Image] = None
        self._camera_info: Optional[CameraInfo] = None

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

        cbg = ReentrantCallbackGroup()
        node.create_subscription(Image, image_topic, self._on_image, SENSOR_QOS,
                                 callback_group=cbg)
        node.create_subscription(PointCloud2, cloud_topic, self._on_cloud, SENSOR_QOS,
                                 callback_group=cbg)
        node.create_subscription(Image, depth_topic, self._on_depth, SENSOR_QOS,
                                 callback_group=cbg)
        node.create_subscription(Image, intensity_topic, self._on_intensity, SENSOR_QOS,
                                 callback_group=cbg)
        node.create_subscription(
            CameraInfo, camera_info_topic, self._on_camera_info, SENSOR_QOS, callback_group=cbg
        )

        # ER 2'nin GERÇEKTE gördüğü kareyi yayınla. Derinlik render'ı ile
        # çalışırken "model neden orayı gösterdi" sorusunun tek cevabı bu.
        self._er_image_pub = (
            node.create_publisher(Image, "/gemini/er_image", 1) if publish_er_image else None
        )

        # SÜREKLİ yayınla, sadece sorgu anında değil. Sorgu başına tek kare
        # yayınlarsak RViz sorgular arasında boş durur ve "kamera hiçbir şey
        # görmüyor" gibi okunur; oysa asıl ihtiyaç, sorgu GÖNDERMEDEN önce
        # kadrajın doğru olduğunu görebilmek.
        #
        # ROS timer'ı DEĞİL, wall-clock thread'i: use_sim_time true iken ROS
        # timer'ları /clock'a bağlı ve /clock durursa önizleme de susuyor -
        # yani teşhis aracı, tam da teşhise en çok ihtiyaç duyulan anda ölüyor.
        if self._er_image_pub is not None and er_image_rate_hz > 0.0:
            threading.Thread(
                target=self._preview_loop, args=(1.0 / er_image_rate_hz,), daemon=True
            ).start()

        node.get_logger().info(
            f"GeminiLocator hazır | backend={self._er.name} | "
            f"er_image={er_image_source}"
            + (f"/{render_mode} [{render_min_m}-{render_max_m} "
               "m]" if er_image_source == "render" else "")
            + f" | cloud={cloud_topic} depth={depth_topic} world={world_frame}"
        )

    # --- abonelikler -----------------------------------------------------

    def _on_image(self, msg: Image) -> None:
        with self._lock:
            self._image = msg

    def _on_cloud(self, msg: PointCloud2) -> None:
        with self._lock:
            self._cloud = msg

    def _on_depth(self, msg: Image) -> None:
        with self._lock:
            self._depth = msg

    def _on_intensity(self, msg: Image) -> None:
        with self._lock:
            self._intensity = msg

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._camera_info = msg

    # --- ER girdisi -------------------------------------------------------

    def set_height_window(self, lo_m: float, hi_m: float) -> None:
        """Relief çiziminin yükseklik penceresini değiştirir (tarama pozu başına)."""
        with self._lock:
            self._render_height_lo_m = float(lo_m)
            self._render_height_hi_m = float(hi_m)

    def set_render_mode(self, mode: str) -> None:
        """
        Render modunu değiştirir ve ER'nin modalite anlatımını da güncller.

        Poza göre değişmesi gerekiyor: konveyöre TEPEDEN bakışta parça düz bir
        yüzey üstünde bir çıkıntı, orada relief şart. Toolkit'e YANDAN bakışta
        ise sahne metrelerce derinliğe yayılıyor, relief penceresi doyuyor ve
        görüntü okunmaz hale geliyor; orada normals sahneyi 3B gri render gibi
        gösteriyor.
        """
        if mode not in depth_render.RENDER_MODES:
            raise ValueError(
                f"Bilinmeyen render modu {mode!r} "
                f"({', '.join(depth_render.RENDER_MODES)})"
            )
        with self._lock:
            self._render_mode = mode
        setter = getattr(self._er, "set_context", None)
        if callable(setter):
            setter("rgb" if self._er_image_source == "rgb" else mode)

    def has_image(self) -> bool:
        """ER 2'ye verilecek bir kare üretilebiliyor mu."""
        with self._lock:
            if self._er_image_source == "rgb":
                return self._image is not None
            if self._render_mode == "intensity":
                return self._intensity is not None
            return self._depth is not None

    def latest_bgr(self) -> Optional[np.ndarray]:
        """ER 2'ye gönderilecek BGR kareyi üretir."""
        with self._lock:
            image, depth, intensity, info = (
                self._image, self._depth, self._intensity, self._camera_info
            )

        if self._er_image_source == "rgb":
            return imgmsg_to_bgr(image) if image is not None else None

        try:
            rendered = depth_render.render(
                mode=self._render_mode,
                depth_msg=depth,
                camera_info=info,
                intensity_msg=intensity,
                min_m=self._render_min_m,
                max_m=self._render_max_m,
                height_lo_m=self._render_height_lo_m,
                height_hi_m=self._render_height_hi_m,
                depth_scale_m=self._depth_scale_m,
                depth_is_radial=self._depth_is_radial,
            )
        except Exception as exc:
            self._node.get_logger().error(f"Derinlik render'ı başarısız: {exc}")
            return None

        if rendered is not None and self._er_image_pub is not None:
            out = Image()
            out.header.stamp = self._node.get_clock().now().to_msg()
            out.header.frame_id = self._camera_frame(depth) if depth is not None else ""
            out.height, out.width = rendered.shape[:2]
            out.encoding = "bgr8"
            out.step = out.width * 3
            out.data = rendered.tobytes()
            self._er_image_pub.publish(out)

        return rendered

    def _preview_loop(self, period_sec: float) -> None:
        """ER görüntüsünü sorgudan bağımsız olarak, gerçek zamanda yayınlar."""
        while rclpy.ok():
            try:
                self.latest_bgr()
            except Exception as exc:
                self._node.get_logger().debug(f"Önizleme render'ı atlandı: {exc}")
            time.sleep(period_sec)

    # --- yardımcılar -----------------------------------------------------

    def _camera_frame(self, fallback_msg: Any) -> str:
        if self._camera_frame_override:
            return self._camera_frame_override
        return fallback_msg.header.frame_id

    @staticmethod
    def _scale_pixel(
            u: int, v: int, src_shape: Tuple[int, ...], dst_w: int, dst_h: int) -> Tuple[int, int]:
        """
        ER görüntüsü uzayındaki pikseli bulut/derinlik uzayına ölçekler.

        Render derinlikten üretildiği için normalde aynı çözünürlüktedir ve bu
        birim dönüşümdür; RGB kaynağı seçildiğinde ise RGB ve derinlik boyutları
        ayrışabiliyor ve ölçeklemezsek nokta kayar.
        """
        src_h, src_w = src_shape[0], src_shape[1]
        if src_w == dst_w and src_h == dst_h:
            return u, v
        uu = int(round(u * dst_w / float(src_w)))
        vv = int(round(v * dst_h / float(src_h)))
        return (max(0, min(dst_w - 1, uu)), max(0, min(dst_h - 1, vv)))

    def _deproject(self, detection: Detection, er_shape: Tuple[int, ...],
                   measure_payload: bool = True) -> Optional[Dict[str, Any]]:
        """Tek bir 2D tespiti world frame'inde 3D noktaya çevirir."""
        with self._lock:
            cloud, depth, info = self._cloud, self._depth, self._camera_info

        point_cam = None
        source_msg = None
        u_src = v_src = None

        if self._deprojection in ("auto", "cloud") and cloud is not None:
            u_src, v_src = self._scale_pixel(
                detection.u, detection.v, er_shape, cloud.width, cloud.height
            )
            point_cam = point_from_cloud(cloud, u_src, v_src, self._sample_window)
            source_msg = cloud

        if point_cam is None and self._deprojection in ("auto", "depth"):
            if depth is not None and info is not None:
                u_src, v_src = self._scale_pixel(
                    detection.u, detection.v, er_shape, depth.width, depth.height
                )
                point_cam = point_from_depth(
                    depth, info, u_src, v_src, self._sample_window,
                    depth_scale_m=self._depth_scale_m,
                    depth_is_radial=self._depth_is_radial,
                )
                source_msg = depth

        if point_cam is None or source_msg is None:
            return None

        source_frame = self._camera_frame(source_msg)
        point_world = self._to_world_point(point_cam, source_frame, source_msg.header.stamp)
        if point_world is None:
            self._node.get_logger().warn(
                f"TF yok: {source_frame} -> {self._world_frame} ({detection.label})"
            )
            return None

        result = {
            "label": detection.label,
            "pixel": {"u": detection.u, "v": detection.v},
            "camera_frame": source_frame,
            "position_camera": {
                "x": point_cam[0], "y": point_cam[1], "z": point_cam[2],
            },
            "frame_id": self._world_frame,
            "position": {
                "x": point_world[0], "y": point_world[1], "z": point_world[2],
            },
        }

        # Yüzey yaması yalnızca organize buluttan çıkarılabiliyor.
        if cloud is not None and u_src is not None:
            u_c, v_c = self._scale_pixel(
                detection.u, detection.v, er_shape, cloud.width, cloud.height
            )
            surface = self._surface_at(cloud, u_c, v_c, source_frame,
                                       measure_payload=measure_payload,
                                       label=detection.label)
            if surface is not None:
                result["surface"] = surface

        return result

    def _to_world_point(
            self, point_cam: Tuple[float, float, float], source_frame: str,
            stamp: Any) -> Optional[Tuple[float, float, float]]:
        world = transform_point(
            self._tf_buffer, point_cam, source_frame, self._world_frame,
            stamp=stamp, timeout_sec=self._tf_timeout_sec,
        )
        if world is None:
            # Damgalı arama başarısızsa en son transform'a düş: sim başlangıcında
            # TF buffer henüz o anı kapsamayabiliyor.
            world = transform_point(
                self._tf_buffer, point_cam, source_frame, self._world_frame,
                stamp=None, timeout_sec=self._tf_timeout_sec,
            )
        return world

    def _surface_at(
        self, cloud: PointCloud2, u: int, v: int, source_frame: str,
        measure_payload: bool = True, label: str = "",
    ) -> Optional[Dict[str, Any]]:
        """(u, v) çevresine düzlem oturtup world frame'inde normal ve kavrama pozu üretir."""
        points = gather_patch_points(cloud, u, v, self._patch_radius_px)

        # Çapa: ER 2'nin işaret ettiği pikselin kendi 3B noktası. Yamanın
        # medyanı kenar durumlarında iki yüzeyin arasına düşüyor (bkz. fit_surface).
        anchor = point_from_cloud(cloud, u, v, self._sample_window)
        patch = fit_surface(
            points,
            depth_band=self._patch_depth_band,
            min_points=self._patch_min_points,
            anchor=anchor,
        )
        if patch is None:
            return None

        centroid_world = self._to_world_point(patch.centroid, source_frame, cloud.header.stamp)
        # Normal bir YÖN vektörüdür: transform_point kullanılırsa kameranın
        # ötelemesi de eklenir ve sonuç anlamsız olur.
        normal_world = rotate_vector(
            self._tf_buffer, patch.normal, source_frame, self._world_frame,
            stamp=cloud.header.stamp, timeout_sec=self._tf_timeout_sec,
        )
        if normal_world is None:
            normal_world = rotate_vector(
                self._tf_buffer, patch.normal, source_frame, self._world_frame,
                stamp=None, timeout_sec=self._tf_timeout_sec,
            )
        if centroid_world is None or normal_world is None:
            return None

        # Hareket için normali dünya eksenine oturt, ama ÖLÇÜLEN normali de
        # sakla: is_graspable'ın eğim denetimi ölçülen değere bakmalı, yoksa
        # gerçekten eğik bir yüzeyi oturttuktan sonra "düz" sanıp üstüne
        # gideriz. Sıra önemli - önce ölç, sonra denetle, en son oturt.
        try:
            snapped, deviation = snap_normal_to_axis(normal_world, self._normal_snap_deg)
        except ValueError as exc:
            self._node.get_logger().warn(f"Yüzey normali geçersiz: {exc}")
            return None
        if self._normal_snap_deg > 0.0 and deviation <= self._normal_snap_deg:
            self._node.get_logger().info(
                f"Yüzey normali dünya eksenine oturtuldu ({deviation:.2f} derece sapma)"
            )
        elif self._normal_snap_deg > 0.0:
            # (normal_snap_deg = 0 ise oturtma KASITLI olarak kapalıdır - eğik
            # yüzeylerle çalışılıyordur; orada uyarmak gürültü olur.)
            #
            # OTURTULMADI: ölçülen normal olduğu gibi harekete gidiyor. Bu
            # sessiz kalırsa teşhis edilemez, çünkü belirtisi normalin kendisi
            # değil GEOMETRİK SONUCU: yaklaşma pozu temas noktasının tam
            # üstünde değil, yana kaymış olur ve iniş nesneyi süpürür.
            # Sayıyı burada hesaplayıp basıyoruz ki logdan doğrudan okunsun.
            lateral_mm = 1000.0 * self._approach_hint * math.tan(math.radians(deviation))
            gap_mm = 1000.0 * _CUP_DIAMETER_M * math.tan(math.radians(deviation))
            self._node.get_logger().warning(
                f"Yüzey normali dünya eksenine OTURTULMADI: sapma "
                f"{deviation:.1f} derece > {self._normal_snap_deg:.1f}. "
                f"Sonuç: {self._approach_hint * 100:.0f} cm yaklaşmada yaklaşma "
                f"pozu temas noktasından {lateral_mm:.0f} mm YANA düşer, emme "
                f"kabı yüzeye {gap_mm:.1f} mm eğik oturur. "
                f"(yama: {patch.inliers} nokta, artık "
                f"{patch.rms_residual * 1000:.1f} mm, yayılım "
                f"{patch.extent * 1000:.0f} mm)"
            )

        try:
            quat = approach_quaternion(snapped, approach_vector=self._tool_approach_vector)
        except ValueError as exc:
            self._node.get_logger().warn(f"Kavrama kuaterniyonu üretilemedi: {exc}")
            return None

        payload = self._measure_payload(cloud, u, v, patch, source_frame,
                                        enabled=measure_payload)

        # ÖLÇÜM KAYDI: buradaki sayılar zaten hesaplanıyor ama şimdiye kadar
        # yalnız log satırlarında kalıyordu. Yamanın kaç noktayla başlayıp
        # bantta kaça düştüğü, artığın kaç mm olduğu ve normalin oturtulup
        # oturtulmadığı, makalede doğrudan raporlanan büyüklüklerdir.
        telemetry.emit(
            "surface",
            label=str(label),
            u=int(u), v=int(v),
            patch_points=int(getattr(patch, "patch_points", 0)),
            band_points=int(patch.inliers),
            rms_residual_mm=round(patch.rms_residual * 1000.0, 3),
            extent_mm=round(patch.extent * 1000.0, 1),
            normal_measured=[normal_world[0], normal_world[1], normal_world[2]],
            normal_used=[snapped[0], snapped[1], snapped[2]],
            normal_deviation_deg=round(float(deviation), 3),
            snapped=bool(self._normal_snap_deg > 0.0
                         and deviation <= self._normal_snap_deg),
            snap_tolerance_deg=float(self._normal_snap_deg),
            # payload None ise ölçüm reddedilmiş, config'teki boyuta
            # düşülmüştür; kayıtta bu ayrım korunur, yoksa ölçülen ile
            # varsayılan boyut sonradan birbirinden ayrılamaz.
            payload_size=([round(v * 1000.0, 1) for v in payload["size_measured"]]
                          if isinstance(payload, dict)
                          and payload.get("size_measured") else None),
            payload_pixels=(payload.get("pixels")
                            if isinstance(payload, dict) else None),
            payload_source=("measured" if isinstance(payload, dict) else "config"),
        )

        return {
            "payload": payload,
            "centroid": {
                "x": centroid_world[0], "y": centroid_world[1], "z": centroid_world[2],
            },
            # Ölçülen (denetim ve raporlama için)
            "normal_measured": {
                "x": normal_world[0], "y": normal_world[1], "z": normal_world[2],
            },
            # Harekette kullanılan
            "normal": {
                "x": snapped[0], "y": snapped[1], "z": snapped[2],
            },
            "normal_deviation_deg": deviation,
            "quat_xyzw": list(quat),
            "rms_residual": patch.rms_residual,
            "inliers": patch.inliers,
            "extent": patch.extent,
        }

    def _measure_payload(
        self, cloud: PointCloud2, u: int, v: int, patch: Any, source_frame: str,
        enabled: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Parçanın yönlü sınırlayıcı kutusunu bu kareden ölçer.

        NEDEN BURADA: ölçüm TESPİT anında yapılmak zorunda. Kavrama anında
        kamera artık parçayı değil emme kabını görüyor - kol tam üstüne inmiş
        oluyor. Tespitle birlikte ölçüp sonuca iliştiriyoruz.
        """
        if not (enabled and self._measure_payload_enabled):
            return None
        xyz = organized_xyz(cloud)
        if xyz is None:
            return None
        try:
            box = measure_payload_box(
                xyz, (u, v), patch.normal, patch.centroid,
                margin_m=self._payload_margin_m,
                min_height_m=self._payload_min_height_m,
                max_height_m=self._payload_max_height_m,
                max_span_m=self._payload_max_span_m,
            )
        except Exception as exc:  # noqa: BLE001 - ölçüm görevi düşürmesin
            self._node.get_logger().debug(f"Parça ölçülemedi: {exc}")
            return None
        if box is None:
            # Sessiz kalmamalı: ölçüm reddedildiğinde config'teki payload_size
            # kullanılıyor ve gerçek parça ondan büyükse planlayıcı var olmayan
            # boşluklara güvenir. Reddin en sık sebebi ayak izinin
            # payload_max_span_m'i aşması, yani üst düzlemin parçaya değil
            # destek yüzeyine oturması.
            self._node.get_logger().warn(
                "Parça ÖLÇÜLEMEDİ; config'teki payload_size'a düşülüyor. "
                "En olası sebep ölçümün parçadan taşıp destek yüzeyine "
                "(konveyör bandı) yayılması - o durumda ayak izi "
                f"payload_max_span_m={self._payload_max_span_m:.2f} m sınırını "
                "aşıyor ve ölçüm reddediliyor."
            )
            return None

        centre_world = self._to_world_point(box.centre, source_frame, cloud.header.stamp)
        # Eksenler YÖN vektörü: yalnızca döndürülür, ötelenmez.
        axes_world = []
        for column in range(3):
            rotated = rotate_vector(
                self._tf_buffer, tuple(box.axes[:, column]), source_frame,
                self._world_frame, stamp=cloud.header.stamp,
                timeout_sec=self._tf_timeout_sec,
            )
            if rotated is None:
                return None
            axes_world.append(np.asarray(rotated, dtype=np.float64))
        if centre_world is None:
            return None

        matrix = np.column_stack(axes_world)
        if float(np.linalg.det(matrix)) < 0.0:
            matrix[:, 1] *= -1.0
        try:
            quat = matrix_to_quaternion(matrix)
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().debug(f"Parça kuaterniyonu üretilemedi: {exc}")
            return None

        self._node.get_logger().info(
            "Parça ölçüldü: "
            f"{box.size_measured[0] * 1000:.0f}x{box.size_measured[1] * 1000:.0f}x"
            f"{box.size_measured[2] * 1000:.0f} mm ({box.pixels} piksel), "
            f"pay ile {box.size[0] * 1000:.0f}x{box.size[1] * 1000:.0f}x"
            f"{box.size[2] * 1000:.0f} mm"
        )
        return {
            "size": list(box.size),
            "size_measured": list(box.size_measured),
            "centre": list(centre_world),
            "quat_xyzw": list(quat),
            "pixels": box.pixels,
        }

    # --- ana giriş noktası ------------------------------------------------

    def locate(self, query: str, max_items: int = 10,
               measure_payload: bool = True) -> List[Dict[str, Any]]:
        """
        Sorguyu (query) ER 2'ye sorar ve 3D'ye çevrilebilen tespitleri döndürür.

        Bloklayıcıdır. 3D'ye çevrilemeyen tespitler (derinlik yok, TF yok)
        listeden düşer; boş liste "nesne görülmedi VEYA konumlandırılamadı"
        demektir, log'da ikisi ayrışır.
        """
        image_bgr = self.latest_bgr()
        if image_bgr is None:
            self._node.get_logger().warn(
                "ER görüntüsü üretilemedi (kamera verisi henüz gelmedi mi?)"
            )
            return []

        self._begin_query("point", query)
        try:
            detections = self._er.point(image_bgr, query, max_items=max_items)
        except Exception as exc:
            self._node.get_logger().error(f"ER sorgusu başarısız ({query!r}): {exc}")
            self._emit_query("point", query, image_bgr, None, error=str(exc))
            return []

        # Kayıt, tespit 3B'ye çevrilMEDEN önce yapılır: modelin ham çıktısı ile
        # zincirin sonucu ayrı sorulardır ve ikisi ayrı ayrı raporlanmalıdır.
        # Boş sonuç da kaydedilir; "model hiçbir şey bulamadı" bu çalışmada
        # ölçülen bir olaydır (kayıtlı koşuda bırakma hedefinde yaşandı).
        self._emit_query("point", query, image_bgr, detections)

        if not detections:
            self._node.get_logger().info(f"ER hiçbir şey bulamadı: {query!r}")
            return []

        results = []
        for detection in detections:
            located = self._deproject(detection, image_bgr.shape,
                                      measure_payload=measure_payload)
            if located is None:
                self._node.get_logger().warn(
                    f"'{detection.label}' 2D'de bulundu ama 3D'ye çevrilemedi "
                    f"(piksel {detection.u},{detection.v}) - derinlik geçersiz olabilir."
                )
                continue
            results.append(located)

        self._node.get_logger().info(
            f"locate({query!r}) -> {len(results)}/{len(detections)} tespit 3D'ye çevrildi"
        )
        return results

    def plan(self, command: str) -> Optional[Dict[str, str]]:
        """Doğal dil komutunu ER 2'ye plan olarak çözdürür."""
        image_bgr = self.latest_bgr()
        if image_bgr is None:
            self._node.get_logger().warn("ER görüntüsü üretilemedi.")
            return None
        self._begin_query("plan", command)
        try:
            plan = self._er.plan(image_bgr, command)
        except Exception as exc:
            self._node.get_logger().error(f"ER planlaması başarısız ({command!r}): {exc}")
            self._emit_query("plan", command, image_bgr, None, error=str(exc))
            return None
        self._emit_query("plan", command, image_bgr, None, plan=plan)
        return plan

    def _begin_query(self, call: str, query: str) -> None:
        """
        Çağrıdan HEMEN ÖNCE: kayıt düğümü o anki kareleri diske yazsın.

        Kareler sonuç geldiğinde yazılamaz. Model çağrısı saniyeler sürüyor ve
        bu sürede derinlik akışı ilerliyor; sonuçla birlikte kaydedilen kare,
        modele gerçekten gönderilen kare olmazdı. Çevrimdışı ablasyonun anlamı
        da tam olarak buna bağlı.
        """
        self._query_seq += 1
        # Render parametreleri de gider: kayıt düğümü AYNI kareden başka
        # render biçimlerini üretip yan yana saklıyor (görüntü karşılaştırması).
        # Parametreler tarama pozuna göre değişiyor (yükseklik penceresi), yani
        # kayıt tarafında sabit varsayılamaz.
        telemetry.emit("er_query_begin", call=call, seq=self._query_seq,
                       query=str(query), render_mode=self._render_mode,
                       er_image_source=self._er_image_source,
                       render_min_m=self._render_min_m,
                       render_max_m=self._render_max_m,
                       render_height_lo_m=self._render_height_lo_m,
                       render_height_hi_m=self._render_height_hi_m,
                       depth_scale_m=self._depth_scale_m,
                       depth_is_radial=self._depth_is_radial)

    def _emit_query(
            self, call: str, query: str, image_bgr: Optional[np.ndarray],
            detections: List[Detection], error: str = "",
            plan: Optional[Dict[str, Any]] = None) -> None:
        """
        Model çağrısını ölçüm veri yoluna yazar.

        Gecikme istemciden okunur (`last_latency_s`): burada yeniden ölçmek
        render süresini de içine katardı ve makalede bildirilen "model çağrısı
        gecikmesi" ile aynı şey olmazdı.
        """
        try:
            height, width = image_bgr.shape[:2]
        except Exception:  # noqa: BLE001
            height = width = None
        items = []
        for det in (detections or []):
            raw = getattr(det, "raw", None)
            items.append({
                "label": getattr(det, "label", ""),
                "u": getattr(det, "u", None),
                "v": getattr(det, "v", None),
                # Modelin KENDİ çıktısı (0-1000 normalize); piksel değeri
                # görüntü boyutuna bağlı olduğu için tek başına yeniden
                # kullanılamaz, ham nokta ise kullanılabilir.
                "point": (raw or {}).get("point") if isinstance(raw, dict) else None,
            })
        latency = getattr(self._er, "last_latency_s", None)
        telemetry.emit(
            "er_query",
            call=call,
            seq=self._query_seq,
            query=str(query),
            latency_s=round(latency, 3) if isinstance(latency, (int, float)) else None,
            n_detections=(len(detections) if detections is not None else 0),
            image_w=width, image_h=height,
            render_mode=self._render_mode,
            er_image_source=self._er_image_source,
            detections=items,
            plan=plan,
            error=error,
        )
