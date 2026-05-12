from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


PATIENT_ID = "patient-001"
PATIENT_NAME = "Demo Patient"
IMAGE_SIZE = (360, 240)


def _draw_room(draw: ImageDraw.ImageDraw) -> None:
    width, height = IMAGE_SIZE
    draw.rectangle([0, 0, width, height], fill=(234, 238, 232))
    draw.rectangle([0, 116, 102, height], fill=(214, 219, 212))
    draw.rectangle([268, 0, width, 144], fill=(218, 226, 235))
    draw.rectangle([284, 18, 352, 128], outline=(90, 105, 122), width=3)
    draw.rectangle([10, 138, 90, 228], outline=(120, 95, 74), width=3)
    draw.line([0, 116, width, 116], fill=(180, 185, 178), width=2)


def _draw_person(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    floor_y: int,
    shirt: tuple[int, int, int],
    pants: tuple[int, int, int],
    skin: tuple[int, int, int] = (183, 134, 104),
) -> None:
    head_r = 10
    head_cy = floor_y - 72
    draw.ellipse(
        [center_x - head_r, head_cy - head_r, center_x + head_r, head_cy + head_r],
        fill=skin,
        outline=(78, 60, 50),
    )
    draw.rounded_rectangle([center_x - 17, floor_y - 60, center_x + 17, floor_y - 24], radius=6, fill=shirt)
    draw.rectangle([center_x - 14, floor_y - 24, center_x - 3, floor_y], fill=pants)
    draw.rectangle([center_x + 3, floor_y - 24, center_x + 14, floor_y], fill=pants)
    draw.line([center_x - 17, floor_y - 54, center_x - 31, floor_y - 35], fill=shirt, width=6)
    draw.line([center_x + 17, floor_y - 54, center_x + 31, floor_y - 35], fill=shirt, width=6)


def _frame_with_people(patient_x: int | None, visitor_x: int | None = None) -> Image.Image:
    image = Image.new("RGB", IMAGE_SIZE)
    draw = ImageDraw.Draw(image)
    _draw_room(draw)
    if visitor_x is not None:
        _draw_person(draw, visitor_x, 208, shirt=(170, 80, 68), pants=(75, 78, 88), skin=(145, 98, 72))
    if patient_x is not None:
        _draw_person(draw, patient_x, 204, shirt=(38, 132, 146), pants=(44, 67, 92))
    return image


def _reference_image() -> Image.Image:
    image = Image.new("RGB", (96, 144), (234, 238, 232))
    draw = ImageDraw.Draw(image)
    _draw_person(draw, 48, 120, shirt=(38, 132, 146), pants=(44, 67, 92))
    return image


def _zones() -> dict[str, object]:
    return {
        "zones": [
            {
                "id": "front_door",
                "name": "Front door",
                "type": "restricted",
                "polygon": [[0.74, 0.0], [1.0, 0.0], [1.0, 1.0], [0.74, 1.0]],
            },
            {
                "id": "stove",
                "name": "Stove area",
                "type": "high_risk",
                "polygon": [[0.0, 0.48], [0.28, 0.48], [0.28, 1.0], [0.0, 1.0]],
            },
        ]
    }


def generate_demo_dataset(out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    reference_dir = out_dir / "reference"
    frames_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    reference_path = reference_dir / "patient_001.png"
    _reference_image().save(reference_path)

    zone_path = out_dir / "zones.json"
    zone_path.write_text(json.dumps(_zones(), indent=2), encoding="utf-8")

    _frame_with_people(patient_x=None, visitor_x=None).save(frames_dir / "frame_0000.png")
    for index in range(1, 27):
        patient_x = 52 + index * 10
        visitor_x = 142 if 6 <= index <= 21 else None
        _frame_with_people(patient_x=patient_x, visitor_x=visitor_x).save(frames_dir / f"frame_{index:04d}.png")

    return {
        "patient_id": PATIENT_ID,
        "patient_name": PATIENT_NAME,
        "frames_dir": str(frames_dir),
        "reference_images": [str(reference_path)],
        "zones_path": str(zone_path),
        "frame_count": 27,
    }
