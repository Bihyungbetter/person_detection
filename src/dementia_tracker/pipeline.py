from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from .detector import Detection, MotionPersonDetector
from .embeddings import AppearanceEmbedder
from .registry import PatientRegistry
from .tracker import ClassifiedDetection, SimpleTracker
from .zones import Zone, zones_for_bbox


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALERT_ZONE_TYPES = {"restricted", "high_risk"}


class VisionPipeline:
    def __init__(
        self,
        registry: PatientRegistry,
        zones: list[Zone],
        threshold: float = 0.72,
        detector: MotionPersonDetector | None = None,
        embedder: AppearanceEmbedder | None = None,
        tracker: SimpleTracker | None = None,
    ) -> None:
        self.registry = registry
        self.zones = zones
        self.threshold = threshold
        self.detector = detector or MotionPersonDetector()
        self.embedder = embedder or AppearanceEmbedder()
        self.tracker = tracker or SimpleTracker()
        self._active_alerts: set[tuple[int, str, str]] = set()

    def run(self, frames_dir: Path, out_dir: Path, annotate: bool = True) -> dict[str, object]:
        frame_paths = sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not frame_paths:
            raise FileNotFoundError(f"No image frames found in {frames_dir}")

        out_dir.mkdir(parents=True, exist_ok=True)
        annotated_dir = out_dir / "annotated_frames"
        if annotate:
            annotated_dir.mkdir(parents=True, exist_ok=True)
        events_path = out_dir / "events.jsonl"

        event_count = 0
        detection_count = 0
        match_count = 0
        with events_path.open("w", encoding="utf-8") as event_file:
            for frame_index, frame_path in enumerate(frame_paths):
                with Image.open(frame_path) as image:
                    frame = image.convert("RGB")

                detections = self.detector.detect(frame)
                detection_count += len(detections)
                classified = self._classify(frame, detections)
                match_count += sum(1 for item in classified if item.patient_id is not None)
                tracks = self.tracker.update(classified)

                frame_events = self._alert_events(frame_index, frame_path.name, frame.size, tracks)
                for event in frame_events:
                    event_file.write(json.dumps(event) + "\n")
                event_count += len(frame_events)

                if annotate:
                    annotated = self._annotate(frame, tracks)
                    annotated.save(annotated_dir / frame_path.name)

        summary = {
            "frames_processed": len(frame_paths),
            "detections": detection_count,
            "matched_detections": match_count,
            "alerts": event_count,
            "events_path": str(events_path),
            "annotated_frames_dir": str(annotated_dir) if annotate else None,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _classify(self, frame: Image.Image, detections: list[Detection]) -> list[ClassifiedDetection]:
        classified: list[ClassifiedDetection] = []
        for detection in detections:
            embedding = self.embedder.from_crop(frame, detection.bbox)
            match = self.registry.match(embedding, self.threshold)
            classified.append(
                ClassifiedDetection(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    patient_id=match.patient_id,
                    patient_name=match.patient_name,
                    identity_score=match.score,
                )
            )
        return classified

    def _alert_events(
        self,
        frame_index: int,
        frame_name: str,
        frame_size: tuple[int, int],
        tracks: list[ClassifiedDetection],
    ) -> list[dict[str, object]]:
        current_alerts: set[tuple[int, str, str]] = set()
        events: list[dict[str, object]] = []

        for track in tracks:
            if track.track_id is None or track.patient_id is None:
                continue
            for zone in zones_for_bbox(track.bbox, frame_size, self.zones):
                if zone.zone_type not in ALERT_ZONE_TYPES:
                    continue
                key = (track.track_id, track.patient_id, zone.zone_id)
                current_alerts.add(key)
                if key in self._active_alerts:
                    continue
                events.append(
                    {
                        "event": "zone_entry",
                        "frame_index": frame_index,
                        "frame": frame_name,
                        "track_id": int(track.track_id),
                        "patient_id": track.patient_id,
                        "patient_name": track.patient_name,
                        "identity_score": round(track.identity_score, 4),
                        "zone_id": zone.zone_id,
                        "zone_name": zone.name,
                        "zone_type": zone.zone_type,
                        "bbox": [int(value) for value in track.bbox],
                    }
                )

        self._active_alerts = current_alerts
        return events

    def _annotate(self, frame: Image.Image, tracks: list[ClassifiedDetection]) -> Image.Image:
        annotated = frame.copy()
        draw = ImageDraw.Draw(annotated)
        width, height = annotated.size

        for zone in self.zones:
            points = [(int(x * width), int(y * height)) for x, y in zone.polygon]
            color = (215, 64, 64) if zone.zone_type in ALERT_ZONE_TYPES else (80, 150, 90)
            draw.line(points + [points[0]], fill=color, width=3)
            draw.text(points[0], zone.name, fill=color)

        for track in tracks:
            color = (25, 130, 70) if track.patient_id else (70, 70, 70)
            x1, y1, x2, y2 = track.bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f"track {track.track_id}"
            if track.patient_id:
                label = f"{track.patient_id} {track.identity_score:.2f}"
            draw.rectangle([x1, max(0, y1 - 16), x1 + max(90, len(label) * 7), y1], fill=(255, 255, 255))
            draw.text((x1 + 3, max(0, y1 - 15)), label, fill=color)

        return annotated
