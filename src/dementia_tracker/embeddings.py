from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .detector import BBox


class AppearanceEmbedder:
    """Extracts a compact person appearance vector from an image crop."""

    def __init__(
        self,
        size: tuple[int, int] = (64, 128),
        color_bins: int = 16,
        grad_bins: int = 9,
        use_person_mask: bool = False,
    ) -> None:
        self.size = size
        self.color_bins = color_bins
        self.grad_bins = grad_bins
        self.use_person_mask = use_person_mask

    def from_path(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            return self.from_image(image)

    def from_crop(self, frame: Image.Image, bbox: BBox) -> np.ndarray:
        width, height = frame.size
        x1, y1, x2, y2 = bbox
        safe_box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        if safe_box[2] <= safe_box[0] or safe_box[3] <= safe_box[1]:
            raise ValueError(f"Invalid crop box: {bbox}")
        return self.from_image(frame.crop(safe_box))

    def from_image(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize(self.size, Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        mask = self._person_mask(arr.shape[:2]) if self.use_person_mask else None
        color = self._color_features(arr, mask)
        gradients = self._gradient_features(arr, mask)
        vector = np.concatenate([color, gradients]).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _color_features(self, arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        bands = np.array_split(arr, 4, axis=0)
        mask_bands = np.array_split(mask, 4, axis=0) if mask is not None else [None] * len(bands)
        features: list[np.ndarray] = []
        for band, band_mask in zip(bands, mask_bands):
            for channel in range(3):
                values = band[:, :, channel].reshape(-1)
                weights = band_mask.reshape(-1) if band_mask is not None else None
                hist, _ = np.histogram(
                    values,
                    bins=self.color_bins,
                    range=(0.0, 1.0),
                    weights=weights,
                )
                hist = hist.astype(np.float32)
                total = hist.sum()
                features.append(hist / total if total else hist)
            mean, std = _weighted_mean_std(band, band_mask)
            features.extend([mean, std])
        return np.concatenate(features)

    def _gradient_features(self, arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        gray = arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114
        gy, gx = np.gradient(gray)
        magnitude = np.sqrt(gx * gx + gy * gy)
        if mask is not None:
            magnitude = magnitude * mask
        angle = (np.arctan2(gy, gx) + math.pi) % math.pi

        cell_h = 16
        cell_w = 16
        features: list[np.ndarray] = []
        for y in range(0, gray.shape[0], cell_h):
            for x in range(0, gray.shape[1], cell_w):
                cell_angle = angle[y : y + cell_h, x : x + cell_w]
                cell_mag = magnitude[y : y + cell_h, x : x + cell_w]
                hist, _ = np.histogram(
                    cell_angle,
                    bins=self.grad_bins,
                    range=(0.0, math.pi),
                    weights=cell_mag,
                )
                hist = hist.astype(np.float32)
                norm = np.linalg.norm(hist)
                features.append(hist / norm if norm else hist)
        return np.concatenate(features)

    def _person_mask(self, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        ys, xs = np.mgrid[0:height, 0:width]
        nx = (xs + 0.5) / width
        ny = (ys + 0.5) / height
        torso = np.exp(-(((nx - 0.5) / 0.26) ** 2 + ((ny - 0.48) / 0.42) ** 2) / 2.0)
        head = np.exp(-(((nx - 0.5) / 0.18) ** 2 + ((ny - 0.16) / 0.16) ** 2) / 2.0)
        legs = np.exp(-(((nx - 0.5) / 0.21) ** 2 + ((ny - 0.82) / 0.22) ** 2) / 2.0)
        mask = np.maximum.reduce([torso, head, legs]).astype(np.float32)
        mask[mask < 0.08] = 0.0
        total = mask.max()
        return mask / total if total else mask


def _weighted_mean_std(arr: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if mask is None:
        return arr.mean(axis=(0, 1)).astype(np.float32), arr.std(axis=(0, 1)).astype(np.float32)
    weights = mask.astype(np.float32)
    total = weights.sum()
    if total == 0:
        zeros = np.zeros(arr.shape[2], dtype=np.float32)
        return zeros, zeros
    mean = (arr * weights[:, :, None]).sum(axis=(0, 1)) / total
    variance = (((arr - mean) ** 2) * weights[:, :, None]).sum(axis=(0, 1)) / total
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))
