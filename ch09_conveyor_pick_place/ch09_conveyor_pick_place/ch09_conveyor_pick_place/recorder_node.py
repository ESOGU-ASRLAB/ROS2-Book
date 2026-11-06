#!/usr/bin/env python3
"""
Koşu kaydedici: bir görevden makaleye girebilecek her ölçümü diske yazar.

NEDEN VAR
    İlk gerçek robot koşularında elimizde yalnız ekran kaydı vardı. Aşama
    süreleri, model gecikmesi ve varış sapmaları terminalden KARE KARE, gözle
    okundu. Bu hem yorucu hem de bir daha üretilemez: aynı sayıya ikinci kez
    ihtiyaç duyulduğunda video yeniden izlenmek zorunda. Bu düğüm aynı
    sayıları koşu sırasında yapılandırılmış dosyalara yazar.

NE TOPLAR
    events.jsonl      /gemini/status + /gemini/record ham akışı (her satır bir olay)
    timeline.csv      durum geçişleri ve aralarındaki süreler
    arrivals.csv      hedef ile TF'ten okunan kap ucu; sapma mm cinsinden
    surfaces.csv      yama/bant nokta sayısı, RMS artık, normal, oturtma sapması
    er_queries.csv    model çağrısı: sorgu, gecikme, dönen tespitler
    vacuum.csv        ölçülen bağıl vakum zaman serisi (VGC10 geri beslemesi)
    joint_states.csv  eklem konumları (doğrusal eksen dahil)
    tcp_track.csv     kap ucunun TF'ten sürekli izi
    frames/           model çağrısı ANINDA: modele giden render (PNG), ham
                      derinlik (.npy), IR genlik (PNG), kamera iç parametreleri
    meta.json         parametre anlık görüntüsü, git commit, konu listesi
    run_summary.txt   insan-okur özet (koşu bitince)

ÇEVRİMDIŞI ABLASYON
    frames/ altındaki ham derinlik ve camera_info, render biçimini sonradan
    değiştirip modele yeniden sormaya yeter. Yani "kabartma yerine normal
    gölgeleme verseydik ne olurdu" sorusu robota bir daha dokunmadan
    yanıtlanabilir. Bunun için ham derinliğin PNG'ye SIKIŞTIRILMADAN, float
    olarak saklanması şart; .npy bu yüzden.

KAYIT SINIRI
    Her koşu kendi klasörüne yazar; klasör adı zaman damgasıdır. Görev
    başlamadan da düğüm ayakta durur (vakum ve eklem akışını sürekli tutar),
    ancak dosyalar ilk olayla birlikte açılır: boş klasör bırakmaz.
"""

from __future__ import annotations

import csv
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import CallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String

from .image_utils import imgmsg_to_bgr

try:
    import cv2
    CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    CV_AVAILABLE = False

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

# Görevin bittiğini gösteren durumlar; özet bunlarda yazılır.
_TERMINAL = {"DONE", "FAILED", "ERROR"}


class CsvSink:
    """
    Başlığı ilk satırda yazan, her satırda flush eden basit CSV yazıcı.

    flush ŞART: koşu Ctrl-C ile kesilirse ya da düğüm çökerse tampondaki
    satırlar kaybolur. Kayıt işinin tek görevi veriyi kaybetmemek olduğu için
    burada başarım değil dayanıklılık tercih edilmiştir.
    """

    def __init__(self, path: Path, columns: List[str]) -> None:
        self.path = path
        self.columns = columns
        self._fh = None
        self._writer = None
        self._lock = threading.Lock()
        self.rows = 0

    def write(self, row: Dict[str, Any]) -> None:
        with self._lock:
            if self._writer is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("w", newline="", encoding="utf-8")
                self._writer = csv.DictWriter(
                    self._fh, fieldnames=self.columns, extrasaction="ignore")
                self._writer.writeheader()
            self._writer.writerow(row)
            self._fh.flush()
            self.rows += 1

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


class RecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("gemini_recorder")
        cbg = ReentrantCallbackGroup()

        self.declare_parameter("record_root", "")
        self.declare_parameter("record_run_name", "")
        self.declare_parameter("record_frames", True)
        self.declare_parameter("record_depth_npy", True)
        self.declare_parameter("record_tcp_hz", 5.0)
        self.declare_parameter("record_joint_decimate", 10)
        self.declare_parameter("record_note", "")
        # Model çağrısı anında, AYNI derinlik karesinden ek olarak üretilecek
        # render biçimleri. Görüntü karşılaştırmasını (ve render ablasyonunu)
        # koşu sırasında, elle bir şey yapmadan toplar: modele giden kare ile
        # alternatifleri aynı andan, aynı parametrelerle gelir.
        self.declare_parameter("record_compare_modes", ["normals", "gray"])
        # Bu topic'ler gemini_params.yaml'daki adlarla AYNI anahtarları
        # kullanır; `/**:` joker'i sayesinde kayıt düğümü de aynı değerleri alır
        # ve mode overlay'i (sim/real) burada da kendiliğinden geçerli olur.
        self.declare_parameter("depth_topic", "/depth")
        self.declare_parameter("intensity_topic", "/intensity")
        self.declare_parameter("camera_info_topic", "/camera_info")
        self.declare_parameter("er_image_topic", "/gemini/er_image")
        self.declare_parameter("vacuum_status_topic", "/OnRobotVGInput")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("end_effector", "")

        def get(name: str) -> Any:
            return self.get_parameter(name).value

        root = str(get("record_root") or "").strip()
        if not root:
            # Kullanıcının çıktı klasörü. install/ altına yazmak işe yaramaz
            # (colcon build onu siler); kaynak klasörüne yazmak da depoyu kirletir.
            root = str(Path.home() / "ros2_book_output" / "ch09" / "recordings")
        self.root = Path(root)

        name = str(get("record_run_name") or "").strip()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{stamp}_{name}" if name else stamp
        self.run_dir = self.root / self.run_id
        self.frames_dir = self.run_dir / "frames"

        self.save_frames = bool(get("record_frames")) and CV_AVAILABLE
        self.save_depth_npy = bool(get("record_depth_npy"))
        self.compare_modes = [str(m) for m in (get("record_compare_modes") or [])
                              if str(m).strip()]
        self.joint_decimate = max(1, int(get("record_joint_decimate")))
        self.world_frame = str(get("world_frame"))
        self.end_effector = str(get("end_effector") or "")

        self.t0 = time.time()
        self._joint_seen = 0
        self._last_state: Optional[str] = None
        self._last_state_t: Optional[float] = None
        self._frame_stems: Dict[Any, str] = {}
        self._scan_pose: Any = None
        self._closed = False
        self._lock = threading.Lock()

        # En son görülen kareler; model çağrısı olayı geldiğinde diske yazılır.
        self._depth_msg: Optional[Image] = None
        self._intensity_msg: Optional[Image] = None
        self._info_msg: Optional[CameraInfo] = None
        self._er_msg: Optional[Image] = None

        self.events_path = self.run_dir / "events.jsonl"
        self._events_fh = None
        self.timeline = CsvSink(self.run_dir / "timeline.csv", [
            "t_rel_s", "state", "detail", "prev_state", "prev_duration_s"])
        self.arrivals = CsvSink(self.run_dir / "arrivals.csv", [
            "t_rel_s", "label",
            "target_x", "target_y", "target_z",
            "tf_x", "tf_y", "tf_z",
            "err_x_mm", "err_y_mm", "err_z_mm", "err_norm_mm", "rail_m"])
        self.surfaces = CsvSink(self.run_dir / "surfaces.csv", [
            "t_rel_s", "label", "u_px", "v_px",
            "patch_points", "band_points", "band_kept_ratio",
            "rms_residual_mm", "extent_mm",
            "normal_meas_x", "normal_meas_y", "normal_meas_z",
            "normal_used_x", "normal_used_y", "normal_used_z",
            "normal_deviation_deg", "snapped", "snap_tolerance_deg",
            "payload_h_mm", "payload_w_mm", "payload_d_mm", "payload_source"])
        self.er_queries = CsvSink(self.run_dir / "er_queries.csv", [
            "t_rel_s", "call", "query", "latency_s", "n_detections",
            "image_w", "image_h", "render_mode", "scan_pose", "frame_file",
            "detections_norm"])
        self.scans = CsvSink(self.run_dir / "scan_poses.csv", [
            "t_rel_s", "role", "query", "pose_index", "pose_name", "found"])
        self.vacuum = CsvSink(self.run_dir / "vacuum.csv", [
            "t_rel_s", "gvca", "gvcb", "pct_a", "pct_b"])
        self.joints = CsvSink(self.run_dir / "joint_states.csv", [
            "t_rel_s", "name", "position", "velocity"])
        self.tcp = CsvSink(self.run_dir / "tcp_track.csv", [
            "t_rel_s", "x", "y", "z", "qx", "qy", "qz", "qw"])
        self._sinks = [self.timeline, self.arrivals, self.surfaces,
                       self.er_queries, self.scans, self.vacuum, self.joints,
                       self.tcp]

        self.create_subscription(String, "/gemini/status", self._on_status, 20,
                                 callback_group=cbg)
        self.create_subscription(String, "/gemini/record", self._on_record, 50,
                                 callback_group=cbg)
        self.create_subscription(Image, str(get("depth_topic")), self._on_depth,
                                 SENSOR_QOS, callback_group=cbg)
        self.create_subscription(Image, str(get("intensity_topic")),
                                 self._on_intensity, SENSOR_QOS, callback_group=cbg)
        self.create_subscription(CameraInfo, str(get("camera_info_topic")),
                                 self._on_info, SENSOR_QOS, callback_group=cbg)
        self.create_subscription(Image, str(get("er_image_topic")), self._on_er_image,
                                 SENSOR_QOS, callback_group=cbg)
        self.create_subscription(JointState, str(get("joint_states_topic")),
                                 self._on_joints, 20, callback_group=cbg)
        self._subscribe_vacuum(str(get("vacuum_status_topic")), cbg)

        # TCP izi ancak uç eleman çerçevesi bilindiğinde başlar. Ad
        # parametreden gelirse hemen, gelmezse görev düğümünün künyesiyle
        # (bkz. _rec_config) sonra.
        self._tf_buffer = None
        self._tcp_timer = None
        self._tcp_hz = float(get("record_tcp_hz"))
        self._cbg = cbg
        if self.end_effector:
            self._start_tcp_track()

        self._write_meta(str(get("record_note") or ""))
        self.get_logger().info(f"kayıt -> {self.run_dir}")

    # --- kurulum ---------------------------------------------------------

    def _subscribe_vacuum(self, topic: str, cbg: CallbackGroup) -> None:
        """VGC10 mesajları ayrı bir pakettedir; yoksa kayıt yine de çalışsın."""
        try:
            from vgc10_msgs.msg import OnRobotVGInput
        except ImportError:
            self.get_logger().warn(
                "vgc10_msgs bulunamadı; vakum zaman serisi kaydedilmeyecek.")
            return
        self.create_subscription(OnRobotVGInput, topic, self._on_vacuum, 10,
                                 callback_group=cbg)

    def _start_tcp_track(self) -> None:
        if self._tcp_hz <= 0.0 or self._tcp_timer is not None:
            return
        self._setup_tf()
        if self._tf_buffer is None:
            return
        self._tcp_timer = self.create_timer(
            1.0 / self._tcp_hz, self._sample_tcp, callback_group=self._cbg)
        self.get_logger().info(
            f"kap ucu izi açık: {self.world_frame} -> {self.end_effector} "
            f"@ {self._tcp_hz:g} Hz")

    def _setup_tf(self) -> None:
        try:
            from tf2_ros import Buffer, TransformListener
            self._tf_buffer = Buffer()
            # spin_thread=False: dinleyici KENDİ executor'ını açmasın, bizimki
            # zaten çok iş parçacıklı. Kendi executor'ıyla açıldığında kapanışta
            # tf2_ros.TransformListener.__del__ içinde InvalidHandle yığın izi
            # basıyor - veri kaybı yok ama her koşunun sonunda hata gibi görünüyor.
            self._tf_listener = TransformListener(
                self._tf_buffer, self, spin_thread=False)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"TF dinleyicisi kurulamadı: {exc}")
            self._tf_buffer = None

    def _write_meta(self, note: str) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "run_id": self.run_id,
            "started": datetime.now().isoformat(),
            "started_epoch": self.t0,
            "note": note,
            "world_frame": self.world_frame,
            "end_effector": self.end_effector,
            "git": self._git_state(),
            "opencv": CV_AVAILABLE,
            "topics": {
                "status": "/gemini/status",
                "record": "/gemini/record",
                "depth": self.get_parameter("depth_topic").value,
                "intensity": self.get_parameter("intensity_topic").value,
                "camera_info": self.get_parameter("camera_info_topic").value,
                "er_image": self.get_parameter("er_image_topic").value,
                "vacuum": self.get_parameter("vacuum_status_topic").value,
                "joint_states": self.get_parameter("joint_states_topic").value,
            },
        }
        (self.run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _git_state() -> Dict[str, str]:
        """Kaydın hangi koda ait olduğu, sonradan sorulacak ilk sorudur."""
        here = str(Path(__file__).resolve().parent)
        out = {}
        for key, cmd in (("commit", ["git", "rev-parse", "HEAD"]),
                         ("dirty", ["git", "status", "--porcelain"])):
            try:
                res = subprocess.run(cmd, cwd=here, capture_output=True,
                                     text=True, timeout=5)
                out[key] = res.stdout.strip()[:400] if res.returncode == 0 else ""
            except Exception:  # noqa: BLE001
                out[key] = ""
        out["dirty"] = "yes" if out.get("dirty") else "no"
        return out

    # --- zaman -----------------------------------------------------------

    def _t(self) -> float:
        return time.time() - self.t0

    # --- ham olay akışı --------------------------------------------------

    def _append_event(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                return
            if self._events_fh is None:
                self.run_dir.mkdir(parents=True, exist_ok=True)
                self._events_fh = self.events_path.open("a", encoding="utf-8")
            self._events_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._events_fh.flush()

    def _on_status(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:  # noqa: BLE001
            data = {"raw": msg.data}
        t = self._t()
        data["_t_rel_s"] = round(t, 3)
        data["_source"] = "status"
        self._append_event(data)

        state = str(data.get("state", ""))
        if not state:
            return
        prev, prev_t = self._last_state, self._last_state_t
        self.timeline.write({
            "t_rel_s": round(t, 3),
            "state": state,
            "detail": str(data.get("detail", ""))[:400],
            "prev_state": prev or "",
            "prev_duration_s": round(t - prev_t, 3) if prev_t is not None else "",
        })
        self._last_state, self._last_state_t = state, t
        if state in _TERMINAL:
            self._write_summary(state, str(data.get("detail", "")))

    def _on_record(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:  # noqa: BLE001
            return
        t = self._t()
        data["_t_rel_s"] = round(t, 3)
        data["_source"] = "record"
        self._append_event(data)

        kind = str(data.get("kind", ""))
        handler = {
            "arrival": self._rec_arrival,
            "surface": self._rec_surface,
            "config": self._rec_config,
            "scan_pose": self._rec_scan_pose,
            "scan_result": self._rec_scan_result,
            "er_query_begin": self._rec_er_begin,
            "er_query": self._rec_er_query,
        }.get(kind)
        if handler is not None:
            try:
                handler(t, data)
            except Exception as exc:  # noqa: BLE001 - kayıt görevi düşürmez
                self.get_logger().warn(f"{kind} kaydı yazılamadı: {exc}")

    # --- olay yazıcıları -------------------------------------------------

    def _rec_arrival(self, t: float, d: Dict[str, Any]) -> None:
        target = d.get("target") or [None, None, None]
        tf = d.get("tf") or [None, None, None]
        err = d.get("error_mm") or [None, None, None]
        norm = None
        if all(isinstance(v, (int, float)) for v in err):
            norm = round(float(np.linalg.norm(np.asarray(err, dtype=float))), 2)
        self.arrivals.write({
            "t_rel_s": round(t, 3), "label": d.get("label", ""),
            "target_x": target[0], "target_y": target[1], "target_z": target[2],
            "tf_x": tf[0], "tf_y": tf[1], "tf_z": tf[2],
            "err_x_mm": err[0], "err_y_mm": err[1], "err_z_mm": err[2],
            "err_norm_mm": norm, "rail_m": d.get("rail"),
        })

    def _rec_surface(self, t: float, d: Dict[str, Any]) -> None:
        patch = d.get("patch_points")
        band = d.get("band_points")
        ratio = round(band / patch, 4) if patch and band is not None else ""
        nm = d.get("normal_measured") or [None, None, None]
        nu = d.get("normal_used") or [None, None, None]
        pay = d.get("payload_size") or [None, None, None]
        self.surfaces.write({
            "t_rel_s": round(t, 3), "label": d.get("label", ""),
            "u_px": d.get("u"), "v_px": d.get("v"),
            "patch_points": patch, "band_points": band, "band_kept_ratio": ratio,
            "rms_residual_mm": d.get("rms_residual_mm"),
            "extent_mm": d.get("extent_mm"),
            "normal_meas_x": nm[0], "normal_meas_y": nm[1], "normal_meas_z": nm[2],
            "normal_used_x": nu[0], "normal_used_y": nu[1], "normal_used_z": nu[2],
            "normal_deviation_deg": d.get("normal_deviation_deg"),
            "snapped": d.get("snapped"),
            "snap_tolerance_deg": d.get("snap_tolerance_deg"),
            "payload_h_mm": pay[2] if len(pay) > 2 else None,
            "payload_w_mm": pay[0], "payload_d_mm": pay[1],
            "payload_source": d.get("payload_source", ""),
        })

    def _rec_config(self, t: float, d: Dict[str, Any]) -> None:
        """
        Koşu künyesi: parametreleri diske yaz, TCP izini burada başlat.

        Uç eleman çerçevesinin adı önekten TÜRETİLİYOR (pick_place_node), yani
        kayıt düğümü onu parametrelerden okuyamaz. Künye gelene kadar TF
        dinleyicisi hiç açılmaz: gereksiz bir iş parçacığı ve kapanışta
        gereksiz bir yığın izi olurdu.
        """
        (self.run_dir / "config.json").write_text(
            json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        frame = str(d.get("end_effector") or "")
        world = str(d.get("world_frame") or "")
        if world:
            self.world_frame = world
        if frame and not self.end_effector:
            self.end_effector = frame
            self._start_tcp_track()

    def _rec_scan_pose(self, t: float, d: Dict[str, Any]) -> None:
        """Hangi tarama pozunda olduğumuzu tut; model çağrısı satırına yazılır."""
        self._scan_pose = d.get("name") or d.get("index")

    def _rec_scan_result(self, t: float, d: Dict[str, Any]) -> None:
        self.scans.write({
            "t_rel_s": round(t, 3), "role": d.get("role", ""),
            "query": str(d.get("query", ""))[:200],
            "pose_index": d.get("index"), "pose_name": d.get("name", ""),
            "found": d.get("found"),
        })

    def _rec_er_begin(self, t: float, d: Dict[str, Any]) -> None:
        """Çağrı BAŞLARKEN kareleri diske al; sonuç satırı bunları gösterir."""
        seq = d.get("seq")
        stem = self._save_frames(t, f"{d.get('call', 'er')}{seq if seq else ''}",
                                 render_opts=d)
        if seq is not None:
            self._frame_stems[seq] = stem

    def _rec_er_query(self, t: float, d: Dict[str, Any]) -> None:
        seq = d.get("seq")
        stem = self._frame_stems.pop(seq, "") if seq is not None else ""
        if not stem:
            # begin olayı kaçtıysa yine de bir kare bırak; hiç yoktan iyidir,
            # ama bu karenin modele gidenle aynı olmadığı bilinsin diye adı
            # "late" ile işaretlenir.
            stem = self._save_frames(t, f"{d.get('call', 'er')}_late")
        self.er_queries.write({
            "t_rel_s": round(t, 3), "call": d.get("call", ""),
            "query": str(d.get("query", ""))[:400],
            "latency_s": d.get("latency_s"),
            "n_detections": d.get("n_detections"),
            "image_w": d.get("image_w"), "image_h": d.get("image_h"),
            "render_mode": d.get("render_mode", ""),
            "scan_pose": d.get("scan_pose") or self._scan_pose,
            "frame_file": stem,
            "detections_norm": json.dumps(d.get("detections") or [],
                                          ensure_ascii=False)[:2000],
        })

    # --- kare kaydı ------------------------------------------------------

    def _save_frames(self, t: float, tag: str,
                     render_opts: Optional[Dict[str, Any]] = None) -> str:
        """
        Model çağrısı anındaki kareleri diske yazar; dosya kökünü döndürür.

        Modele GİDEN kare `/gemini/er_image` üzerinden alınır; bu, render'ı
        burada yeniden üretmekten daha güvenilirdir, çünkü render parametreleri
        (yükseklik penceresi, mod) tarama pozuna göre değişiyor ve o an hangi
        değerlerin geçerli olduğunu yalnız locator biliyor.
        """
        if not self.save_frames:
            return ""
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)[:40]
        stem = f"{t:08.3f}_{safe}".replace(".", "p", 1)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        wrote = False

        er, depth, intensity, info = (self._er_msg, self._depth_msg,
                                      self._intensity_msg, self._info_msg)
        if er is not None:
            try:
                cv2.imwrite(str(self.frames_dir / f"{stem}_er.png"),
                            imgmsg_to_bgr(er))
                wrote = True
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"ER karesi yazılamadı: {exc}")
        if depth is not None:
            arr = self._depth_array(depth)
            if arr is not None:
                if self.save_depth_npy:
                    np.save(str(self.frames_dir / f"{stem}_depth.npy"), arr)
                # Göz denetimi için 16-bit mm görüntüsü; ablasyonda .npy kullanılır.
                mm = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                scale = 1000.0 if arr.dtype != np.uint16 else 1.0
                cv2.imwrite(str(self.frames_dir / f"{stem}_depth_mm.png"),
                            np.clip(mm * scale, 0, 65535).astype(np.uint16))
                wrote = True
        if intensity is not None:
            try:
                cv2.imwrite(str(self.frames_dir / f"{stem}_intensity.png"),
                            self._intensity_png(intensity))
                wrote = True
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"Genlik karesi yazılamadı: {exc}")
        if render_opts is not None and self.compare_modes and depth is not None:
            self._save_comparison(stem, depth, intensity, info, render_opts)
        if info is not None:
            (self.frames_dir / f"{stem}_caminfo.json").write_text(json.dumps({
                "width": info.width, "height": info.height,
                "k": list(info.k), "d": list(info.d), "p": list(info.p),
                "distortion_model": info.distortion_model,
                "frame_id": info.header.frame_id,
            }, indent=2), encoding="utf-8")
            wrote = True
        return stem if wrote else ""

    def _save_comparison(
            self, stem: str, depth: np.ndarray, intensity: Optional[np.ndarray], info: Any,
            opts: Dict[str, Any]) -> None:
        """
        Aynı kareden alternatif render biçimlerini üretip yazar.

        Parametreler locator'dan geldiği gibi kullanılır. Burada varsayılan
        değere düşmek sessiz bir hata olurdu: yükseklik penceresi tarama
        pozuna göre değişiyor ve yanlış pencereyle üretilen bir kare, modele
        gidenle karşılaştırılabilir olmaz.
        """
        try:
            from . import depth_render
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"depth_render yüklenemedi: {exc}")
            return
        common = dict(
            depth_msg=depth, camera_info=info, intensity_msg=intensity,
            min_m=float(opts.get("render_min_m", 0.3)),
            max_m=float(opts.get("render_max_m", 4.0)),
            height_lo_m=float(opts.get("render_height_lo_m", -0.005)),
            height_hi_m=float(opts.get("render_height_hi_m", 0.12)),
            depth_scale_m=float(opts.get("depth_scale_m", 0.001)),
            depth_is_radial=bool(opts.get("depth_is_radial", False)),
        )
        used = str(opts.get("render_mode", ""))
        for mode in self.compare_modes:
            if mode == used:
                continue  # modele giden kare zaten _er.png olarak duruyor
            try:
                img = depth_render.render(mode=mode, **common)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"{mode} render'ı başarısız: {exc}")
                continue
            if img is not None:
                cv2.imwrite(str(self.frames_dir / f"{stem}_alt_{mode}.png"), img)

    @staticmethod
    def _depth_array(msg: Image) -> Optional[np.ndarray]:
        """
        Derinlik mesajını ndarray'e çevirir; ölçek DEĞİŞTİRİLMEZ.

        Ölçek burada bilerek uygulanmıyor: 16UC1 kamerada mm, 32FC1'de metredir
        ve hangisinin geçerli olduğu depth_scale_m parametresine bağlı. Ham
        değeri saklayıp yorumu çevrimdışı çözümlemeye bırakmak, yanlış ölçekle
        kaydedip veriyi bozmaktan iyidir.
        """
        try:
            if msg.encoding in ("32FC1", "32FC"):
                return np.frombuffer(msg.data, np.float32).reshape(
                    msg.height, msg.width).copy()
            if msg.encoding in ("16UC1", "mono16"):
                return np.frombuffer(msg.data, np.uint16).reshape(
                    msg.height, msg.width).copy()
        except Exception:  # noqa: BLE001
            return None
        return None

    @staticmethod
    def _intensity_png(msg: Image) -> np.ndarray:
        if msg.encoding in ("16UC1", "mono16"):
            arr = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width)
            return arr.copy()
        if msg.encoding == "mono8":
            return np.frombuffer(msg.data, np.uint8).reshape(
                msg.height, msg.width).copy()
        return imgmsg_to_bgr(msg)

    # --- sensör geri çağrıları -------------------------------------------

    def _on_depth(self, msg: Image) -> None:
        self._depth_msg = msg

    def _on_intensity(self, msg: Image) -> None:
        self._intensity_msg = msg

    def _on_info(self, msg: CameraInfo) -> None:
        self._info_msg = msg

    def _on_er_image(self, msg: Image) -> None:
        self._er_msg = msg

    def _on_vacuum(self, msg: Any) -> None:
        a = int(getattr(msg, "gvca", 0))
        b = int(getattr(msg, "gvcb", 0))
        self.vacuum.write({
            "t_rel_s": round(self._t(), 3), "gvca": a, "gvcb": b,
            "pct_a": round(a / 10.0, 1), "pct_b": round(b / 10.0, 1),
        })

    def _on_joints(self, msg: JointState) -> None:
        # Eklem akışı ~100-500 Hz gelir; hepsini yazmak dosyayı gereksiz
        # şişirir ve makale için gereken çözünürlük çok daha kaba.
        self._joint_seen += 1
        if self._joint_seen % self.joint_decimate:
            return
        t = round(self._t(), 3)
        vel = list(msg.velocity) if msg.velocity else []
        for i, name in enumerate(msg.name):
            self.joints.write({
                "t_rel_s": t, "name": name,
                "position": round(float(msg.position[i]), 6)
                if i < len(msg.position) else "",
                "velocity": round(float(vel[i]), 6) if i < len(vel) else "",
            })

    def _sample_tcp(self) -> None:
        if self._tf_buffer is None or not self.end_effector:
            return
        try:
            from rclpy.duration import Duration
            from rclpy.time import Time
            tf = self._tf_buffer.lookup_transform(
                self.world_frame, self.end_effector, Time(),
                timeout=Duration(seconds=0.2))
        except Exception:  # noqa: BLE001 - TF henüz yoksa sessiz geç
            return
        p, q = tf.transform.translation, tf.transform.rotation
        self.tcp.write({
            "t_rel_s": round(self._t(), 3),
            "x": round(p.x, 5), "y": round(p.y, 5), "z": round(p.z, 5),
            "qx": round(q.x, 6), "qy": round(q.y, 6),
            "qz": round(q.z, 6), "qw": round(q.w, 6),
        })

    # --- özet ------------------------------------------------------------

    def _write_summary(self, final_state: str, detail: str) -> None:
        lines = [
            f"koşu       : {self.run_id}",
            f"bitiş      : {final_state}  {detail}",
            f"süre       : {self._t():.1f} s",
            "",
            f"olay       : {self.events_path.name}",
            f"aşama      : {self.timeline.rows} geçiş",
            f"varış      : {self.arrivals.rows} ölçüm",
            f"yüzey      : {self.surfaces.rows} düzlem oturtma",
            f"model      : {self.er_queries.rows} çağrı",
            f"tarama     : {self.scans.rows} poz denemesi",
            f"vakum      : {self.vacuum.rows} örnek",
            f"eklem      : {self.joints.rows} satır",
            f"kap ucu izi: {self.tcp.rows} örnek",
        ]
        if self.frames_dir.exists():
            lines.append(f"kare       : {len(list(self.frames_dir.iterdir()))} dosya")
        (self.run_dir / "run_summary.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        self.get_logger().info(f"koşu özeti yazıldı: {self.run_dir}")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._events_fh is not None:
                self._events_fh.close()
                self._events_fh = None
        for sink in self._sinks:
            sink.close()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = RecorderNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        # launch kapanırken SIGTERM gelir. Bu bir hata değil, normal çıkıştır;
        # yakalanmazsa terminale yığın izi basar ve asıl önemli olan `finally`
        # bloğu (dosyaların kapatılması) yine de çalışsa bile kayıt kesilmiş
        # gibi görünür.
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
