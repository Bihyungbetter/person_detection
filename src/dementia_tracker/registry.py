from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .embeddings import AppearanceEmbedder, cosine_similarity


@dataclass(frozen=True)
class PatientProfile:
    patient_id: str
    name: str
    embedding: np.ndarray
    reference_count: int
    created_at: str
    updated_at: str

    def to_json(self) -> dict[str, object]:
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "embedding": self.embedding.tolist(),
            "reference_count": self.reference_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "PatientProfile":
        return cls(
            patient_id=str(payload["patient_id"]),
            name=str(payload["name"]),
            embedding=np.asarray(payload["embedding"], dtype=np.float32),
            reference_count=int(payload.get("reference_count", 1)),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
        )


@dataclass(frozen=True)
class MatchResult:
    patient_id: str | None
    patient_name: str | None
    score: float


class PatientRegistry:
    def __init__(self, patients: dict[str, PatientProfile] | None = None) -> None:
        self.patients = patients or {}

    @classmethod
    def load(cls, path: Path) -> "PatientRegistry":
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        patients = {
            item["patient_id"]: PatientProfile.from_json(item)
            for item in payload.get("patients", [])
        }
        return cls(patients=patients)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "patients": [profile.to_json() for profile in self.patients.values()],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add_patient_from_images(
        self,
        patient_id: str,
        name: str,
        image_paths: list[Path],
        embedder: AppearanceEmbedder,
    ) -> PatientProfile:
        if not image_paths:
            raise ValueError("At least one reference image is required.")

        embeddings = [embedder.from_path(path) for path in image_paths]
        combined = np.mean(np.stack(embeddings), axis=0)
        norm = np.linalg.norm(combined)
        if norm:
            combined = combined / norm

        now = datetime.now(UTC).isoformat()
        existing = self.patients.get(patient_id)
        profile = PatientProfile(
            patient_id=patient_id,
            name=name,
            embedding=combined.astype(np.float32),
            reference_count=len(image_paths),
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        self.patients[patient_id] = profile
        return profile

    def add_patient_from_embeddings(
        self,
        patient_id: str,
        name: str,
        embeddings: list[np.ndarray],
    ) -> PatientProfile:
        if not embeddings:
            raise ValueError("At least one embedding is required.")

        combined = np.mean(np.stack(embeddings), axis=0)
        norm = np.linalg.norm(combined)
        if norm:
            combined = combined / norm

        now = datetime.now(UTC).isoformat()
        existing = self.patients.get(patient_id)
        profile = PatientProfile(
            patient_id=patient_id,
            name=name,
            embedding=combined.astype(np.float32),
            reference_count=len(embeddings),
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        self.patients[patient_id] = profile
        return profile

    def match(self, embedding: np.ndarray, threshold: float) -> MatchResult:
        best_profile: PatientProfile | None = None
        best_score = 0.0
        for profile in self.patients.values():
            if profile.embedding.shape != embedding.shape:
                continue
            score = cosine_similarity(embedding, profile.embedding)
            if score > best_score:
                best_score = score
                best_profile = profile

        if best_profile is None or best_score < threshold:
            return MatchResult(patient_id=None, patient_name=None, score=best_score)
        return MatchResult(
            patient_id=best_profile.patient_id,
            patient_name=best_profile.name,
            score=best_score,
        )

    def summary(self) -> dict[str, object]:
        return {
            "patient_count": len(self.patients),
            "patients": [
                {
                    "patient_id": profile.patient_id,
                    "name": profile.name,
                    "reference_count": profile.reference_count,
                    "updated_at": profile.updated_at,
                }
                for profile in self.patients.values()
            ],
        }
