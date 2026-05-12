from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dementia_tracker.demo_data import generate_demo_dataset
from dementia_tracker.embeddings import AppearanceEmbedder
from dementia_tracker.pipeline import VisionPipeline
from dementia_tracker.registry import PatientRegistry
from dementia_tracker.webapp import DEFAULT_CROP, _extract_person_crop_with_bbox
from dementia_tracker.zones import load_zones, point_in_polygon


class VisionPipelineTests(unittest.TestCase):
    def test_demo_pipeline_emits_patient_zone_alert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo = generate_demo_dataset(root / "input")
            registry = PatientRegistry()
            registry.add_patient_from_images(
                patient_id=demo["patient_id"],
                name=demo["patient_name"],
                image_paths=[Path(path) for path in demo["reference_images"]],
                embedder=AppearanceEmbedder(),
            )
            zones = load_zones(Path(demo["zones_path"]))

            pipeline = VisionPipeline(registry=registry, zones=zones, threshold=0.72)
            summary = pipeline.run(Path(demo["frames_dir"]), root / "results", annotate=False)

            self.assertGreater(summary["detections"], 0)
            self.assertGreater(summary["matched_detections"], 0)
            self.assertGreater(summary["alerts"], 0)

            events_path = Path(summary["events_path"])
            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(event["zone_id"] == "front_door" for event in events))
            self.assertTrue(all(event["patient_id"] == demo["patient_id"] for event in events))

    def test_point_in_polygon(self) -> None:
        square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self.assertTrue(point_in_polygon((0.5, 0.5), square))
        self.assertFalse(point_in_polygon((1.5, 0.5), square))

    def test_person_mask_embedder_builds_vector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo = generate_demo_dataset(root / "input")
            reference = Path(demo["reference_images"][0])

            vector = AppearanceEmbedder(use_person_mask=True).from_path(reference)

            self.assertGreater(vector.shape[0], 0)
            self.assertAlmostEqual(float((vector * vector).sum()), 1.0, places=4)

    def test_live_foreground_crop_uses_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo = generate_demo_dataset(root / "input")
            background_path = root / "live_background.jpg"
            background_frame = Path(demo["frames_dir"]) / "frame_0000.png"
            person_frame = Path(demo["frames_dir"]) / "frame_0016.png"
            background_path.write_bytes(background_frame.read_bytes())

            from PIL import Image

            with Image.open(person_frame) as image:
                crop, bbox = _extract_person_crop_with_bbox(
                    image.convert("RGB"),
                    background_path,
                    DEFAULT_CROP,
                    require_foreground=True,
                )

            self.assertIsNotNone(crop)
            self.assertIsNotNone(bbox)
            self.assertLess(crop.size[0], 180)

            with Image.open(background_frame) as image:
                crop, bbox = _extract_person_crop_with_bbox(
                    image.convert("RGB"),
                    background_path,
                    DEFAULT_CROP,
                    require_foreground=True,
                )

            self.assertIsNone(crop)
            self.assertIsNone(bbox)


if __name__ == "__main__":
    unittest.main()
