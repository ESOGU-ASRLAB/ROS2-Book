#!/usr/bin/env python3
"""
Gemini Robotics ER 2 istemcisi.

ER 2 robot aksiyonu ÜRETMEZ; sadece metin/koordinat döndürür. Bu modül iki şeyi verir:

  - point(image, query)  -> görüntüdeki nesnelerin 2D piksel konumları
  - plan(image, command) -> doğal dil komutunu {"pick": ..., "place": ...} adımlarına ayırır

İki backend var:
  gemini : Gemini API (GEMINI_API_KEY ortam değişkeni gerekir)
  vertex : Vertex AI üzerinden aynı model (GCP projesi gerekir)

Koordinat sözleşmesi: ER 2 noktaları [y, x] sırasında ve 0-1000'e normalize
döndürür. pixel_from_normalized() bunu (u, v) piksele çevirir; bu dönüşümü tek
yerde tutmak önemli, çünkü y/x sırasının ters çevrilmesi sessizce yanlış ama
makul görünen 3D noktalar üretir.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
    CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    CV_AVAILABLE = False


DEFAULT_MODEL = "gemini-robotics-er-2-preview"

# API anahtarının aranacağı dosyalar (ortam değişkenleri boşsa, sırayla).
#
# NEDEN DOSYA: ortam değişkeni yalnızca onu export eden kabuktan başlatılan
# süreçlere geçer. Düğüm masaüstü kısayolundan, bir IDE terminalinden, systemd
# biriminden ya da .bashrc düzenlenmeden ÖNCE açılmış bir terminalden
# başlatılırsa anahtar yoktur ve hata "sanki her açılışta yeniden girmek
# gerekiyor" gibi görünür. Dosya bu bağlamların hepsinde okunur.
#
# Anahtar HÂLÂ depoya girmez: bu yollar ev dizinindedir, paketin içinde değil.
API_KEY_PATHS = (
    os.path.expanduser("~/.config/gemini/api_key"),
    "/etc/gemini/api_key",
)


def load_api_key() -> Optional[str]:
    """Anahtarı ortamdan, olmazsa dosyadan okur. Bulamazsa None döner."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value

    for path in API_KEY_PATHS:
        try:
            with open(path, "r") as handle:
                for line in handle:
                    line = line.strip()
                    # Yorum ve boş satırları atla; "GEMINI_API_KEY=..." biçimi
                    # de kabul edilsin ki dosya doğrudan source edilebilsin.
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):].strip()
                    if line.startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
                        line = line.split("=", 1)[1].strip()
                    return line.strip("'\"")
        except (OSError, UnicodeDecodeError):
            continue
    return None


# Derinlik render'ı ER 2 için dağılım-dışı bir girdi. Ne baktığını söylemezsek
# model görüntüyü olağan bir fotoğraf sanıp renk/doku üzerinden akıl yürütmeye
# çalışıyor; modaliteyi açıkça tarif etmek geometrik sorgulara yönlendiriyor.
IMAGE_CONTEXT = {
    "rgb": "",
    "relief": (
        "This image is NOT a photograph. It is a time-of-flight depth map rendered "
        "as HEIGHT ABOVE THE DOMINANT FLAT SURFACE in the scene (the table, "
        "conveyor belt or floor the objects rest on). Colour encodes height only, "
        "not the real colour of objects: dark blue/purple is the reference surface "
        "itself, and cyan, green, yellow and red are progressively taller. Any "
        "object lying on the surface therefore appears as a clearly coloured "
        "silhouette against a dark background, and its shape in the image is its "
        "true footprint seen from the camera. Black pixels are missing depth data. "
        "Identify objects by their silhouette and height. Never reason about "
        "colour, texture, printing or labels - none of that is visible here.\n"
    ),
    "normals": (
        "This image is NOT a photograph. It is a surface-normal shaded rendering "
        "of a time-of-flight depth map: brightness encodes surface orientation and "
        "distance, not colour or texture. Nearer surfaces are brighter; black "
        "pixels are missing depth data. Identify objects by their 3D shape, "
        "silhouette and how their flat faces are oriented. Never reason about colour.\n"
    ),
    "turbo": (
        "This image is NOT a photograph. It is a depth map rendered with a colour "
        "map over a fixed metric range: colour encodes DISTANCE only, not the real "
        "colour of objects. Warmer/brighter colours are nearer, black pixels are "
        "missing depth data. Identify objects by their 3D shape and silhouette.\n"
    ),
    "gray": (
        "This image is NOT a photograph. It is a depth map: brightness encodes "
        "DISTANCE only (brighter = nearer), not the real colour of objects. Black "
        "pixels are missing depth data. Identify objects by their 3D shape and "
        "silhouette.\n"
    ),
    "intensity": (
        "This image is the near-infrared intensity (amplitude) image of a "
        "time-of-flight camera, not a colour photograph. Brightness encodes how "
        "strongly the surface reflects the sensor's infrared light. Identify "
        "objects by shape and reflectance, not by colour.\n"
    ),
}

POINTING_PROMPT = """\
{context}Point to {query} in the image.
Return at most {max_items} items.

If several DISTINCT places match the description - for example the query says
"one of the empty compartments" and there are four of them - return ALL of them,
best match first. Do not collapse them into a single answer: the robot checks
each candidate against its own reach and collision model and takes the first one
it can physically get to, so extra candidates are what let it recover when the
best-looking one turns out to be unreachable.

The label returned should be an identifying name for the object detected.
The answer should follow the json format: [{{"point": <point>, "label": <label1>}}, ...].
The points are in [y, x] format normalized to 0-1000.
If nothing matching is visible, return an empty json list [].
"""

PLANNING_PROMPT = """\
{context}You are the high-level planner for a UR10e robot arm fitted with a \
{tool}.
The user command (may be Turkish or English) is: "{command}"

Look at the image and decide:
  - "pick": a short English noun phrase describing the object to grasp
  - "place": a short English noun phrase describing where to release it
  - "reasoning": one short sentence

The "pick" phrase must describe something actually visible in this image.

The "place" target is very likely NOT visible here: the camera is mounted on the
wrist and this is only one of several viewpoints the robot will move through
while searching. Therefore write "place" as a self-contained description taken
from the user's command alone. Do NOT add spatial qualifiers that depend on this
image ("on the left", "in the background", "next to the belt") - the same phrase
will be matched against a completely different view, where such qualifiers would
be wrong.
{tool_hint}
Answer ONLY with json: {{"pick": "...", "place": "...", "reasoning": "..."}}
"""

TOOL_HINTS = {
    "vacuum": (
        "The tool is a vacuum suction cup: it can only pick objects that expose a "
        "flat, smooth, unobstructed surface facing the camera. Prefer the centre of "
        "such a face, away from edges, holes and steep curvature."
    ),
    "parallel": (
        "The tool is a parallel-jaw gripper: prefer objects narrow enough to be "
        "enclosed between two fingers."
    ),
}


@dataclass
class Detection:
    """ER 2'nin döndürdüğü tek bir 2D tespit (henüz 3D değil)."""

    label: str
    u: int  # piksel sütunu
    v: int  # piksel satırı
    raw: Optional[Dict[str, Any]] = None


def pixel_from_normalized(point_yx: Sequence[float], width: int, height: int) -> Tuple[int, int]:
    """
    ER 2'nin [y, x] 0-1000 noktasını (u, v) piksele çevirir.

    Görüntü sınırlarına kırpar: modelin 1000'i biraz aşan değer döndürmesi
    nadir ama mümkün, ve kırpmazsak aşağıda dizi indeksi patlar.
    """
    y_norm, x_norm = float(point_yx[0]), float(point_yx[1])
    u = int(round(x_norm / 1000.0 * (width - 1)))
    v = int(round(y_norm / 1000.0 * (height - 1)))
    return (max(0, min(width - 1, u)), max(0, min(height - 1, v)))


def _extract_json(text: str) -> Any:
    """
    Model çıktısından JSON'u ayıklar.

    thinking_level açıkken model bazen JSON'u ```json ... ``` içine alıyor veya
    önüne bir cümle koyuyor. Önce düz parse dene, olmazsa fence'i soy, o da
    olmazsa ilk [...] / {...} bloğunu yakala.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    for pattern in (r"\[.*\]", r"\{.*\}"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Model çıktısından JSON ayıklanamadı: {text[:300]!r}")


class ERClientBase:
    """Ortak sözleşme. point() ve plan() alt sınıflarda uygulanır."""

    # Ölçüm alanları taban sınıfta tanımlıdır ki kaydı yapan taraf istemci
    # türünü bilmek zorunda kalmasın: gerçek istemcide dolar, uygulamayan bir
    # istemcide None kalır; iki durumda da okunması güvenlidir.
    last_latency_s: Optional[float] = None
    last_response_text: Optional[str] = None

    def point(self, image_bgr: np.ndarray, query: str, max_items: int = 10) -> List[Detection]:
        raise NotImplementedError

    def plan(self, image_bgr: np.ndarray, command: str) -> Dict[str, str]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__


class GeminiERClient(ERClientBase):
    """Gerçek Gemini Robotics ER 2 istemcisi (Gemini API veya Vertex AI)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        thinking_level: str = "high",
        jpeg_quality: int = 90,
        use_vertex: bool = False,
        project: str = "",
        location: str = "us-central1",
        context_key: str = "normals",
        tool: str = "vacuum",
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "google-genai kurulu değil. Kurulum: pip install --upgrade google-genai"
            ) from exc

        if use_vertex:
            self._client = genai.Client(vertexai=True, project=project, location=location)
        else:
            api_key = load_api_key()
            if not api_key:
                raise RuntimeError(
                    "Gemini API anahtarı bulunamadı. İki yoldan biri:\n"
                    f"  1) dosya : {API_KEY_PATHS[0]}  (chmod 600, tek satır)\n"
                    "  2) ortam : export GEMINI_API_KEY=...\n"
                    "Anahtar: aistudio.google.com/apikey"
                )
            self._client = genai.Client(api_key=api_key)

        self._model = model
        self._thinking_level = thinking_level
        self._jpeg_quality = int(jpeg_quality)
        self._context = IMAGE_CONTEXT.get(context_key, "")
        self._tool = tool if tool in TOOL_HINTS else "vacuum"

    def set_context(self, context_key: str) -> None:
        """
        Görüntü modalitesi anlatımını değiştirir.

        Render modu tarama pozları arasında değişiyor (konveyörde relief,
        toolkit'te normals); ER'ye yanlış modaliteyi anlatmak, doğru görüntüyü
        yanlış yorumlatır - relief'te renk YÜKSEKLİK, normals'ta YÖNELİM demek.
        """
        self._context = IMAGE_CONTEXT.get(context_key, "")

    def _encode(self, image_bgr: np.ndarray) -> str:
        """
        BGR görüntüyü base64 JPEG'e çevirir (inline gönderim, dosya upload'suz).

        Inline istek toplamı 20MB ile sınırlı; 512x424 JPEG bunun çok altında,
        ve upload turunu atlamak sorgu gecikmesini belirgin düşürüyor.
        """
        if not CV_AVAILABLE:
            raise RuntimeError("OpenCV yok; görüntü kodlanamıyor.")
        ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not ok:
            raise RuntimeError("cv2.imencode başarısız.")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _ask(self, image_bgr: np.ndarray, prompt: str) -> str:
        # Çağrı gecikmesi ölçülüp saklanır: modelin görev süresindeki payı,
        # sistemin kullanım zarfını belirleyen tek sayıdır ve sonradan
        # hesaplanamaz. `last_latency_s` çağrıdan hemen sonra okunur
        # (locator.locate/plan), bu yüzden örnek başına tek değer yeter.
        started = time.perf_counter()
        response = self._client.interactions.create(
            model=self._model,
            input=[
                {
                    "type": "image",
                    "data": self._encode(image_bgr),
                    "mime_type": "image/jpeg",
                },
                {"type": "text", "text": prompt},
            ],
            generation_config={"thinking_level": self._thinking_level},
        )
        self.last_latency_s = time.perf_counter() - started
        self.last_response_text = response.output_text
        return response.output_text

    def point(self, image_bgr: np.ndarray, query: str, max_items: int = 10) -> List[Detection]:
        height, width = image_bgr.shape[:2]
        text = self._ask(
            image_bgr,
            POINTING_PROMPT.format(context=self._context, query=query, max_items=max_items),
        )
        items = _extract_json(text)
        if not isinstance(items, list):
            raise ValueError(f"Pointing çıktısı liste değil: {items!r}")

        detections: List[Detection] = []
        for item in items:
            if not isinstance(item, dict) or "point" not in item:
                continue
            u, v = pixel_from_normalized(item["point"], width, height)
            detections.append(Detection(label=str(item.get("label", query)), u=u, v=v, raw=item))
        return detections

    def plan(self, image_bgr: np.ndarray, command: str) -> Dict[str, str]:
        text = self._ask(image_bgr, PLANNING_PROMPT.format(
            context=self._context,
            command=command,
            tool="vacuum suction cup" if self._tool == "vacuum" else "parallel-jaw gripper",
            tool_hint=TOOL_HINTS[self._tool],
        ))
        plan = _extract_json(text)
        if not isinstance(plan, dict):
            raise ValueError(f"Plan çıktısı sözlük değil: {plan!r}")
        return {
            "pick": str(plan.get("pick", "")).strip(),
            "place": str(plan.get("place", "")).strip(),
            "reasoning": str(plan.get("reasoning", "")).strip(),
        }


def make_er_client(
    backend: str,
    model: str = DEFAULT_MODEL,
    thinking_level: str = "high",
    jpeg_quality: int = 90,
    project: str = "",
    location: str = "us-central1",
    context_key: str = "normals",
    tool: str = "vacuum",
) -> ERClientBase:
    backend = (backend or "gemini").lower()
    if backend == "gemini":
        return GeminiERClient(
            model=model,
            thinking_level=thinking_level,
            jpeg_quality=jpeg_quality,
            context_key=context_key,
            tool=tool,
        )
    if backend == "vertex":
        return GeminiERClient(
            model=model,
            thinking_level=thinking_level,
            jpeg_quality=jpeg_quality,
            use_vertex=True,
            project=project,
            location=location,
            context_key=context_key,
            tool=tool,
        )
    raise ValueError(f"Bilinmeyen backend: {backend!r} (gemini | vertex)")
