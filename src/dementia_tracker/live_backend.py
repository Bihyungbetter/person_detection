from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .embeddings import AppearanceEmbedder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RUNTIME = PROJECT_ROOT / ".codex_runtime"
DEFAULT_CROP = (0.32, 0.08, 0.68, 0.96)

if LOCAL_RUNTIME.exists() and str(LOCAL_RUNTIME) not in sys.path:
    sys.path.insert(0, str(LOCAL_RUNTIME))

try:
    import cv2  # type: ignore[import-not-found]
except Exception:
    cv2 = None


@dataclass(frozen=True)
class IdentityCrop:
    image: Image.Image
    bbox: tuple[float, float, float, float]
    mode: str


class LiveVisionBackend:
    def __init__(self) -> None:
        self.embedder = AppearanceEmbedder(size=(96, 96), color_bins=24, use_person_mask=False)
        self.embedding_dimensions = int(self.embedder.from_image(Image.new("RGB", (96, 96))).shape[0])
        self._face_detector = self._load_face_detector()
        self._person_detector = self._load_person_detector()
        if self._person_detector is not None:
            self.backend_name = "opencv-hog-person"
        elif self._face_detector is not None:
            self.backend_name = "opencv-haar-face"
        else:
            self.backend_name = "numpy-foreground"

    def extract_identity_crop(
        self,
        image: Image.Image,
        background_path: Path,
        guide_crop: tuple[float, float, float, float],
        require_foreground: bool,
    ) -> IdentityCrop | None:
        person_crop = self._extract_person_crop_hog(image, guide_crop)
        if person_crop is not None:
            return person_crop

        face_crop = self._extract_face_crop(image, guide_crop)
        if face_crop is not None:
            return face_crop

        person_crop_fg, person_bbox = extract_person_crop_with_bbox(
            image,
            background_path,
            guide_crop,
            require_foreground=background_path.exists() and require_foreground,
        )
        if person_crop_fg is None or person_bbox is None:
            return None
        return IdentityCrop(image=person_crop_fg, bbox=person_bbox, mode="foreground")

    def embedding_for_crop(self, crop: IdentityCrop) -> np.ndarray:
        return self.embedder.from_image(crop.image)

    def _load_face_detector(self):
        if cv2 is None:
            return None
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            return None
        return detector

    def _load_person_detector(self):
        if cv2 is None:
            return None
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return hog
        except Exception:
            return None

    def _extract_person_crop_hog(
        self,
        image: Image.Image,
        guide_crop: tuple[float, float, float, float],
    ) -> IdentityCrop | None:
        if cv2 is None or self._person_detector is None:
            return None

        width, height = image.size
        gx1, gy1, gx2, gy2 = normalized_to_box(guide_crop, width, height)
        roi = image.crop((gx1, gy1, gx2, gy2)).convert("RGB")
        roi_w, roi_h = roi.size
        if roi_w < 64 or roi_h < 128:
            return None

        target_w = 320
        scale = target_w / roi_w if roi_w > target_w else 1.0
        if scale != 1.0:
            scaled = roi.resize((target_w, max(128, int(round(roi_h * scale)))), Image.Resampling.BILINEAR)
        else:
            scaled = roi
        scaled_arr = np.asarray(scaled)
        bgr = cv2.cvtColor(scaled_arr, cv2.COLOR_RGB2BGR)

        rects, weights = self._person_detector.detectMultiScale(
            bgr,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        if len(rects) == 0:
            return None

        idx = int(np.argmax(weights)) if len(weights) else 0
        if float(weights[idx]) < 0.35:
            return None
        x, y, bw, bh = rects[idx]

        inv = 1.0 / scale if scale != 0 else 1.0
        x1 = max(0, int(round(x * inv)))
        y1 = max(0, int(round(y * inv)))
        x2 = min(roi_w, int(round((x + bw) * inv)))
        y2 = min(roi_h, int(round((y + bh) * inv)))

        pad_x = max(6, int((x2 - x1) * 0.10))
        pad_y = max(6, int((y2 - y1) * 0.08))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(roi_w, x2 + pad_x)
        y2 = min(roi_h, y2 + pad_y)
        if x2 - x1 < 24 or y2 - y1 < 48:
            return None

        box = (gx1 + x1, gy1 + y1, gx1 + x2, gy1 + y2)
        normalized = (box[0] / width, box[1] / height, box[2] / width, box[3] / height)
        return IdentityCrop(image=image.crop(box), bbox=normalized, mode="person")

    def _extract_face_crop(
        self,
        image: Image.Image,
        guide_crop: tuple[float, float, float, float],
    ) -> IdentityCrop | None:
        if cv2 is None or self._face_detector is None:
            return None

        width, height = image.size
        gx1, gy1, gx2, gy2 = normalized_to_box(guide_crop, width, height)
        roi = image.crop((gx1, gy1, gx2, gy2)).convert("RGB")
        roi_arr = np.asarray(roi)
        gray = cv2.cvtColor(roi_arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._face_detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(42, 42),
        )
        if len(faces) == 0:
            return None

        x, y, face_w, face_h = max(faces, key=lambda face: int(face[2]) * int(face[3]))
        pad_x = int(face_w * 0.42)
        pad_y_top = int(face_h * 0.48)
        pad_y_bottom = int(face_h * 0.72)
        x1 = max(0, int(x) - pad_x)
        y1 = max(0, int(y) - pad_y_top)
        x2 = min(roi.width, int(x + face_w) + pad_x)
        y2 = min(roi.height, int(y + face_h) + pad_y_bottom)
        box = (gx1 + x1, gy1 + y1, gx1 + x2, gy1 + y2)
        normalized = (box[0] / width, box[1] / height, box[2] / width, box[3] / height)
        return IdentityCrop(image=image.crop(box), bbox=normalized, mode="face")


def parse_crop(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        return DEFAULT_CROP
    x1, y1, x2, y2 = (float(item) for item in value)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("Crop must be normalized [x1, y1, x2, y2].")
    return x1, y1, x2, y2


def crop_normalized(image: Image.Image, crop: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    return image.crop(normalized_to_box(crop, width, height))


def extract_person_crop(
    image: Image.Image,
    background_path: Path,
    guide_crop: tuple[float, float, float, float],
    require_foreground: bool,
) -> Image.Image:
    crop, _ = extract_person_crop_with_bbox(image, background_path, guide_crop, require_foreground)
    if crop is None:
        raise ValueError("No foreground person found inside the guide box.")
    return crop


def extract_person_crop_with_bbox(
    image: Image.Image,
    background_path: Path,
    guide_crop: tuple[float, float, float, float],
    require_foreground: bool,
) -> tuple[Image.Image | None, tuple[float, float, float, float] | None]:
    if not background_path.exists():
        return crop_normalized(image, guide_crop), guide_crop

    with Image.open(background_path) as background_image:
        background = background_image.convert("RGB").resize(image.size, Image.Resampling.BILINEAR)

    width, height = image.size
    gx1, gy1, gx2, gy2 = normalized_to_box(guide_crop, width, height)
    current_roi = np.asarray(image.crop((gx1, gy1, gx2, gy2)), dtype=np.float32)
    background_roi = np.asarray(background.crop((gx1, gy1, gx2, gy2)), dtype=np.float32)
    diff = np.abs(current_roi - background_roi).mean(axis=2)

    adaptive_threshold = max(18.0, float(np.percentile(diff, 88)) * 0.7)
    mask = diff > adaptive_threshold
    ys, xs = np.nonzero(mask)
    min_area = max(180, int(mask.size * 0.018))
    if len(xs) < min_area:
        if require_foreground:
            return None, None
        return crop_normalized(image, guide_crop), guide_crop

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    roi_w = gx2 - gx1
    roi_h = gy2 - gy1
    pad_x = max(8, int((x2 - x1) * 0.22))
    pad_y = max(8, int((y2 - y1) * 0.14))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(roi_w, x2 + pad_x)
    y2 = min(roi_h, y2 + pad_y)

    box = (gx1 + x1, gy1 + y1, gx1 + x2, gy1 + y2)
    normalized = (box[0] / width, box[1] / height, box[2] / width, box[3] / height)
    return image.crop(box), normalized


def normalized_to_box(crop: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = crop
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )
