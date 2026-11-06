#!/usr/bin/env python3
"""
Yörünge kayıt/oynatma deposu (HARMONY doğrulama senaryosu).

Amaç: MoveIt planlayıcısının (OMPL) rastgeleliğini ortadan kaldırmak. Her kip
(sensing / cleaning) için yörüngeler BİR KEZ hesaplanır ve o kipe özel tek bir
JSON dosyasına kaydedilir. Dosya varsa planlama yapılmadan aynı yörüngeler
tekrar tekrar çalıştırılır.

Kullanım:
    store = TrajectoryStore(directory, "sensing", logger=node.get_logger())
    traj = store.get("wp_0")          # kayıtlı varsa JointTrajectory, yoksa None
    ...
    store.record("wp_0", planned_jt)  # kayıt kipinde hesaplananı sakla
    store.save()                      # tur bitince dosyaya yaz (ve oynatmaya geç)

Dosya biçimi (<directory>/<mode>.json):
    {
      "mode": "sensing",
      "created": "<iso zaman>",
      "segments": { "<step_key>": { ...JointTrajectory... }, ... }
    }
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from builtin_interfaces.msg import Duration
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# ---------------------------------------------------------------------------
# JointTrajectory <-> dict dönüştürme
# ---------------------------------------------------------------------------
def joint_trajectory_to_dict(jt: JointTrajectory) -> dict:
    points = []
    for p in jt.points:
        points.append({
            "positions": list(p.positions),
            "velocities": list(p.velocities),
            "accelerations": list(p.accelerations),
            "effort": list(p.effort),
            "time_from_start": {
                "sec": int(p.time_from_start.sec),
                "nanosec": int(p.time_from_start.nanosec),
            },
        })
    return {
        "joint_names": list(jt.joint_names),
        "frame_id": jt.header.frame_id,
        "points": points,
    }


def joint_trajectory_from_dict(d: dict) -> JointTrajectory:
    jt = JointTrajectory()
    jt.header = Header()
    jt.header.frame_id = str(d.get("frame_id", ""))
    jt.joint_names = [str(n) for n in d.get("joint_names", [])]
    for pd in d.get("points", []):
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in pd.get("positions", [])]
        pt.velocities = [float(x) for x in pd.get("velocities", [])]
        pt.accelerations = [float(x) for x in pd.get("accelerations", [])]
        pt.effort = [float(x) for x in pd.get("effort", [])]
        tfs = pd.get("time_from_start", {}) or {}
        dur = Duration()
        dur.sec = int(tfs.get("sec", 0))
        dur.nanosec = int(tfs.get("nanosec", 0))
        pt.time_from_start = dur
        jt.points.append(pt)
    return jt


# ---------------------------------------------------------------------------
# Depo
# ---------------------------------------------------------------------------
class TrajectoryStore:
    """
    Bir kipe ait yörüngelerin tek dosyada kaydı ve oynatılması.

    replay=True  -> dosya vardı, kayıtlı yörüngeler `get()` ile oynatılır.
    replay=False -> dosya yoktu; yörüngeler `record()` ile toplanır, `save()` ile
                    yazılır ve kayıttan sonra aynı süreçte oynatma kipine geçilir.
    """

    def __init__(self, directory: str, mode: str, logger: Optional[Any] = None) -> None:
        self.mode = str(mode)
        self.directory = os.path.expanduser(str(directory))
        self.path = os.path.join(self.directory, f"{self.mode}.json")
        self._logger = logger
        self._loaded: Dict[str, JointTrajectory] = {}
        self._recorded: Dict[str, dict] = {}
        self._replay = False
        self._load()

    # -- günlük yardımcıları --
    def _info(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.info(msg)

    def _warn(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.warning(msg)

    @property
    def replay(self) -> bool:
        return self._replay

    # -- yükleme --
    def _load(self) -> None:
        if not os.path.isfile(self.path):
            self._info(
                f"[TRAJ] '{self.mode}' için kayıt yok ({self.path}). "
                f"Trajectory'ler ilk turda hesaplanıp kaydedilecek."
            )
            self._replay = False
            return
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            segments = data.get("segments", {}) or {}
            self._loaded = {
                str(k): joint_trajectory_from_dict(v) for k, v in segments.items()
            }
            self._replay = len(self._loaded) > 0
            self._info(
                f"[TRAJ] '{self.mode}' kaydı yüklendi: {len(self._loaded)} segment "
                f"({self.path}). Planlama atlanacak, kayıtlı trajectory'ler oynatılacak."
            )
        except Exception as e:
            self._warn(
                f"[TRAJ] '{self.path}' okunamadı ({e}). Bu tur yeniden hesaplanacak."
            )
            self._loaded = {}
            self._replay = False

    # -- oynatma --
    def get(self, step_key: str) -> Optional[JointTrajectory]:
        return self._loaded.get(str(step_key))

    # -- kayıt --
    def record(self, step_key: str, jt: Optional[JointTrajectory]) -> None:
        if jt is None:
            return
        self._recorded[str(step_key)] = joint_trajectory_to_dict(jt)

    def save(self) -> bool:
        """
        Kaydedilenleri dosyaya yazar ve aynı süreçte oynatma kipine geçer.

        Dosyada kayıtlı segmentler varsa KORUNUR; yeni kaydedilenler onlara eklenir
        (aynı anahtar varsa yenisi yazılır). Böylece kayıt eksik kalmışsa (örneğin tur
        yarıda kesildiyse) sonraki çalıştırmada yalnızca eksik adımlar planlanıp dosyaya
        eklenebilir.
        """
        if not self._recorded:
            # Oynatma kipinde yeni kayıt yoksa sessizce çık (normal durum).
            if not self._replay:
                self._warn(f"[TRAJ] '{self.mode}' için kaydedilecek trajectory yok.")
            return False
        try:
            os.makedirs(self.directory, exist_ok=True)
            # Önce dosyadakiler (sözlüğe çevrilerek), üzerine yeni kayıtlar.
            segments = {
                k: joint_trajectory_to_dict(v) for k, v in self._loaded.items()
            }
            segments.update(self._recorded)
            payload = {
                "mode": self.mode,
                "created": datetime.now().isoformat(),
                "segments": segments,
            }
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self.path)
            self._info(
                f"[TRAJ] '{self.mode}' kaydedildi: {len(self._recorded)} yeni segment "
                f"(toplam {len(segments)}) -> {self.path}. "
                f"Bundan sonra bu trajectory'ler oynatılacak."
            )
            # Aynı süreçteki sonraki turlar kayıtlı yörüngeyi oynatsın.
            self._loaded = {
                k: joint_trajectory_from_dict(v) for k, v in segments.items()
            }
            self._recorded = {}
            self._replay = True
            return True
        except Exception as e:
            self._warn(f"[TRAJ] '{self.path}' yazılamadı: {e}")
            return False
