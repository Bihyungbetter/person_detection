from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    label: str = "person_candidate"


class MotionPersonDetector:
    """Motion detector for fixed indoor cameras.

    The first frame becomes the background. Later frames are thresholded against
    that background, then connected components become person candidates.
    """

    def __init__(self, threshold: float = 26.0, min_area: int = 280, expand_px: int = 8) -> None:
        self.threshold = threshold
        self.min_area = min_area
        self.expand_px = expand_px
        self._background: np.ndarray | None = None

    def detect(self, frame: Image.Image) -> list[Detection]:
        arr = np.asarray(frame.convert("RGB"), dtype=np.float32)
        if self._background is None:
            self._background = arr
            return []

        diff = np.abs(arr - self._background).mean(axis=2)
        mask = diff > self.threshold
        mask = _close_mask(mask)
        components = _connected_components(mask)

        height, width = mask.shape
        detections: list[Detection] = []
        for x1, y1, x2, y2, area in components:
            if area < self.min_area:
                continue
            box_width = x2 - x1
            box_height = y2 - y1
            if box_height < 24 or box_width < 10:
                continue
            expanded = (
                max(0, x1 - self.expand_px),
                max(0, y1 - self.expand_px),
                min(width, x2 + self.expand_px),
                min(height, y2 + self.expand_px),
            )
            confidence = min(1.0, area / 5000.0)
            detections.append(Detection(bbox=tuple(int(value) for value in expanded), confidence=float(confidence)))
        return detections


def _close_mask(mask: np.ndarray) -> np.ndarray:
    mask = _dilate(mask)
    mask = _erode(mask)
    return mask


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    output = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            output |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return output


def _erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    output = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            output &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return output


def _connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components: list[tuple[int, int, int, int, int]] = []

    ys, xs = np.nonzero(mask)
    for start_x, start_y in zip(xs, ys):
        if visited[start_y, start_x]:
            continue

        queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
        visited[start_y, start_x] = True
        min_x = max_x = start_x
        min_y = max_y = start_y
        area = 0

        while queue:
            x, y = queue.popleft()
            area += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((nx, ny))

        components.append((min_x, min_y, max_x + 1, max_y + 1, area))

    return components
