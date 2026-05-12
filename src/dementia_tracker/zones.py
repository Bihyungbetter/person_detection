from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .detector import BBox


@dataclass(frozen=True)
class Zone:
    zone_id: str
    name: str
    zone_type: str
    polygon: list[tuple[float, float]]


def load_zones(path: Path) -> list[Zone]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    zones: list[Zone] = []
    for item in payload.get("zones", []):
        polygon = [(float(x), float(y)) for x, y in item["polygon"]]
        if len(polygon) < 3:
            raise ValueError(f"Zone {item.get('id')} must contain at least three points.")
        zones.append(
            Zone(
                zone_id=str(item["id"]),
                name=str(item.get("name", item["id"])),
                zone_type=str(item.get("type", "restricted")),
                polygon=polygon,
            )
        )
    return zones


def zones_for_bbox(bbox: BBox, frame_size: tuple[int, int], zones: list[Zone]) -> list[Zone]:
    width, height = frame_size
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height)
    return [zone for zone in zones if point_in_polygon(center, zone.polygon)]


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside

