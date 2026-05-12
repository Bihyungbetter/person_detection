from __future__ import annotations

import base64
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from .live_backend import (
    DEFAULT_CROP,
    LiveVisionBackend,
    extract_person_crop,
    extract_person_crop_with_bbox,
    parse_crop,
)
from .registry import PatientRegistry


WEB_ROOT = Path(__file__).resolve().parents[2] / "web"


def run_live_server(host: str, port: int, registry_path: Path, threshold: float) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(
        (host, port),
        _handler_factory(registry_path=registry_path, threshold=threshold),
    )
    print(f"Live detector running at http://{host}:{port}")
    print("Reference data stays local in", registry_path)
    server.serve_forever()


def _handler_factory(registry_path: Path, threshold: float) -> type[BaseHTTPRequestHandler]:
    background_path = registry_path.with_name("live_background.jpg")

    class LiveDetectorHandler(BaseHTTPRequestHandler):
        vision = LiveVisionBackend()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._serve_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
            elif parsed.path == "/app.js":
                self._serve_file(WEB_ROOT / "app.js", "text/javascript; charset=utf-8")
            elif parsed.path == "/styles.css":
                self._serve_file(WEB_ROOT / "styles.css", "text/css; charset=utf-8")
            elif parsed.path == "/api/status":
                registry = PatientRegistry.load(registry_path)
                compatible_count = _compatible_patient_count(registry, self.vision.embedding_dimensions)
                self._json(
                    {
                        "ready": True,
                        "threshold": threshold,
                        "patient_count": len(registry.patients),
                        "compatible_patient_count": compatible_count,
                        "needs_reenrollment": len(registry.patients) > 0 and compatible_count == 0,
                        "background_ready": background_path.exists(),
                        "backend": self.vision.backend_name,
                        "registry_path": str(registry_path),
                    }
                )
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                if parsed.path == "/api/background":
                    self._set_background(payload)
                elif parsed.path == "/api/enroll":
                    self._enroll(payload)
                elif parsed.path == "/api/detect":
                    self._detect(payload)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _serve_file(self, path: Path, content_type: str) -> None:
            if not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def _enroll(self, payload: dict[str, object]) -> None:
            samples = payload.get("samples")
            if not isinstance(samples, list) or not samples:
                raise ValueError("Enrollment requires one or more image samples.")

            crop = parse_crop(payload.get("crop"))
            embeddings = []
            modes: list[str] = []
            for sample in samples:
                if not isinstance(sample, str):
                    continue
                image = _decode_data_url(sample)
                identity_crop = self.vision.extract_identity_crop(
                    image,
                    background_path,
                    crop,
                    require_foreground=False,
                )
                if identity_crop is None:
                    continue
                embeddings.append(self.vision.embedding_for_crop(identity_crop))
                modes.append(identity_crop.mode)

            registry = PatientRegistry.load(registry_path)
            profile = registry.add_patient_from_embeddings(
                patient_id=str(payload.get("patient_id") or "me"),
                name=str(payload.get("name") or "Me"),
                embeddings=embeddings,
            )
            registry.save(registry_path)
            self._json(
                {
                    "patient_id": profile.patient_id,
                    "name": profile.name,
                    "reference_count": profile.reference_count,
                    "threshold": threshold,
                    "background_ready": background_path.exists(),
                    "backend": self.vision.backend_name,
                    "modes": modes,
                }
            )

        def _detect(self, payload: dict[str, object]) -> None:
            image_value = payload.get("image")
            if not isinstance(image_value, str):
                raise ValueError("Detection requires an image.")

            registry = PatientRegistry.load(registry_path)
            if _compatible_patient_count(registry, self.vision.embedding_dimensions) == 0:
                self._json(
                    {
                        "matched": False,
                        "person_present": False,
                        "score": 0.0,
                        "confidence": 0.0,
                        "threshold": threshold,
                        "needs_enrollment": True,
                        "needs_reenrollment": len(registry.patients) > 0,
                    }
                )
                return

            crop = parse_crop(payload.get("crop"))
            image = _decode_data_url(image_value)
            identity_crop = self.vision.extract_identity_crop(
                image,
                background_path,
                crop,
                require_foreground=background_path.exists(),
            )
            if identity_crop is None:
                self._json(
                    {
                        "matched": False,
                        "person_present": False,
                        "score": 0.0,
                        "confidence": 0.0,
                        "threshold": threshold,
                        "crop": list(crop),
                    }
                )
                return

            embedding = self.vision.embedding_for_crop(identity_crop)
            match = registry.match(embedding, threshold=threshold)
            confidence = 0.0
            if threshold < 1:
                confidence = max(0.0, min(1.0, (match.score - threshold) / (1.0 - threshold)))
            self._json(
                {
                    "matched": match.patient_id is not None,
                    "person_present": True,
                    "patient_id": match.patient_id,
                    "patient_name": match.patient_name,
                    "score": round(match.score, 4),
                    "confidence": round(confidence, 4),
                    "threshold": threshold,
                    "crop": list(crop),
                    "identity_bbox": list(identity_crop.bbox),
                    "identity_mode": identity_crop.mode,
                    "backend": self.vision.backend_name,
                }
            )

        def _set_background(self, payload: dict[str, object]) -> None:
            image_value = payload.get("image")
            if not isinstance(image_value, str):
                raise ValueError("Background capture requires an image.")
            image = _decode_data_url(image_value)
            background_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(background_path, quality=88)
            self._json({"background_ready": True, "background_path": str(background_path)})

        def _json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return LiveDetectorHandler


def _compatible_patient_count(registry: PatientRegistry, dimensions: int) -> int:
    return sum(1 for profile in registry.patients.values() if profile.embedding.shape == (dimensions,))


def _decode_data_url(value: str) -> Image.Image:
    if "," in value:
        _, encoded = value.split(",", 1)
    else:
        encoded = value
    data = base64.b64decode(encoded)
    return Image.open(BytesIO(data)).convert("RGB")


def _extract_person_crop(
    image: Image.Image,
    background_path: Path,
    guide_crop: tuple[float, float, float, float],
    require_foreground: bool,
) -> Image.Image:
    return extract_person_crop(image, background_path, guide_crop, require_foreground)


def _extract_person_crop_with_bbox(
    image: Image.Image,
    background_path: Path,
    guide_crop: tuple[float, float, float, float],
    require_foreground: bool,
) -> tuple[Image.Image | None, tuple[float, float, float, float] | None]:
    return extract_person_crop_with_bbox(image, background_path, guide_crop, require_foreground)
