from __future__ import annotations

from dataclasses import dataclass

from .detector import BBox


@dataclass
class ClassifiedDetection:
    bbox: BBox
    confidence: float
    patient_id: str | None
    patient_name: str | None
    identity_score: float
    track_id: int | None = None


@dataclass
class _TrackState:
    track_id: int
    detection: ClassifiedDetection
    missed: int = 0


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.18, max_missed: int = 4) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: list[_TrackState] = []

    def update(self, detections: list[ClassifiedDetection]) -> list[ClassifiedDetection]:
        matches = self._match(detections)
        matched_track_indexes = set()
        matched_detection_indexes = set()

        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            detection = detections[detection_index]
            detection.track_id = track.track_id
            track.detection = detection
            track.missed = 0
            matched_track_indexes.add(track_index)
            matched_detection_indexes.add(detection_index)

        for index, track in enumerate(self._tracks):
            if index not in matched_track_indexes:
                track.missed += 1

        for index, detection in enumerate(detections):
            if index in matched_detection_indexes:
                continue
            detection.track_id = self._next_id
            self._tracks.append(_TrackState(track_id=self._next_id, detection=detection))
            self._next_id += 1

        self._tracks = [track for track in self._tracks if track.missed <= self.max_missed]
        return [track.detection for track in self._tracks if track.missed == 0]

    def _match(self, detections: list[ClassifiedDetection]) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                overlap = iou(track.detection.bbox, detection.bbox)
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, track_index, detection_index))

        candidates.sort(reverse=True)
        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_index, detection_index in candidates:
            if track_index in used_tracks or detection_index in used_detections:
                continue
            matches.append((track_index, detection_index))
            used_tracks.add(track_index)
            used_detections.add(detection_index)
        return matches


def iou(left: BBox, right: BBox) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    x1 = max(left_x1, right_x1)
    y1 = max(left_y1, right_y1)
    x2 = min(left_x2, right_x2)
    y2 = min(left_y2, right_y2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = max(0, left_x2 - left_x1) * max(0, left_y2 - left_y1)
    right_area = max(0, right_x2 - right_x1) * max(0, right_y2 - right_y1)
    return intersection / float(left_area + right_area - intersection)

