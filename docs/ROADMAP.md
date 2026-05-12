# Person Detection — Roadmap

Living doc. Two tracks. Pick A first (single-patient quality up), then B (multi-patient + zones).

## What ships today (v0 / PROTOTYPE)

- Local webcam → 4-button web UI (Start Camera / Set Empty Background / Enroll Me / Start Detecting).
- Single patient, single match score per frame.
- Pipeline: Haar face → fallback foreground-diff vs saved background → 96×96 color+gradient `AppearanceEmbedder` → cosine match against `live_registry.json`.
- `pipeline.py`, `tracker.py`, `zones.py` exist but **not wired** into the live web app.
- Stack: Python `http.server`, no build step, vanilla JS frontend.

## Known limits

| Limit | Impact |
|-------|--------|
| Color-histogram embedding | Confuses on similar clothing; brittle to lighting. |
| Foreground-diff fallback needs a static empty-room shot | Breaks if camera moves or lighting drifts. |
| Single-patient API | Can't enroll a household / ward. |
| Zones unused live | Pipeline can emit alerts in batch mode but UI never shows them. |
| `http.server` | No streaming, no auth, no schema validation. |

---

## Track A — Stronger re-identification

Goal: confidence stops being a coin flip when clothes are similar.

**Detection swap**
- Replace OpenCV Haar + foreground-diff with **ONNX YOLOv8n-person** (≈6 MB, CPU-real-time) or **MediaPipe Pose** person detector.
- Drop dependency on `Set Empty Background` (button becomes optional / hidden once detection works without a baseline).
- Touch: `src/dementia_tracker/live_backend.py`.

**Embedding swap**
- Replace `AppearanceEmbedder` (24 color bins) with one of:
  - **OSNet-x0.25** via Torchreid → ONNX, 512-dim, purpose-built for person re-ID.
  - **CLIP ViT-B/32** image embedding → 512-dim, robust to lighting, larger model.
- Keep `PatientRegistry.match()` cosine logic — it's shape-agnostic.
- Re-enroll required: bump embedding dim, force users to re-run `Enroll Me`.
- Touch: `src/dementia_tracker/embeddings.py`, `configs/*.yaml`, `README.md`.

**Threshold retune**
- CLIP cosine typical match band ≈ 0.75–0.85; OSNet ≈ 0.6–0.75. Expose per-backend default in `configs/`.

**Acceptance**
- On a 5-person held-out set with similar clothing, top-1 accuracy > 90 % at chosen threshold.
- No `Set Empty Background` click needed for fresh sessions.

---

## Track B — Multi-patient + live zones

Goal: show what `pipeline.py` already does, but live.

**Enroll multiple patients**
- Add a patient-name `<input>` above the buttons in `web/index.html` (keep the 4 buttons unchanged — name field is not a button).
- `Enroll Me` now POSTs `{name, samples, crop}`; backend stores under the typed name.
- Add a small enrolled-list strip in the side panel.
- Touch: `web/index.html`, `web/app.js`, `src/dementia_tracker/webapp.py`.

**Multi-detection per frame**
- `/api/detect` returns a list: `[{patient_id, name, bbox, score, zone_id|null}]`.
- Overlay canvas in `web/app.js` draws a labelled box per detection, color-coded by zone state.
- Use `tracker.py` to smooth flicker across frames (already implemented, just unused).
- Touch: `src/dementia_tracker/webapp.py`, `src/dementia_tracker/pipeline.py` (expose single-frame entry), `web/app.js`.

**Zones live**
- Load `configs/zones.json` server-side; ship polygons in `/api/status`.
- Draw zone outlines on the overlay; flash status pill `IN ZONE: front_door` on entry.
- Reuse `zones.py:point_in_polygon`.
- Touch: `web/app.js`, `configs/zones.json`.

**Acceptance**
- Two enrolled patients in frame both get boxes + names.
- Walking into the demo `front_door` zone fires a visible alert within one poll cycle.

---

## Track C — Plumbing follow-ups (do alongside A/B as friction shows up)

- Migrate `http.server` → **FastAPI** once endpoints exceed ~6 or need JSON schema validation. Track B will hit this first.
- Add `tests/test_webapp.py` (httpx) for `/api/status`, `/api/enroll`, `/api/detect`.
- Drop `_compatible_patient_count` filter in `webapp.py` once the embedding-dim migration in Track A lands and all stored profiles share one shape.
- Replace base64 frame upload with binary `multipart/form-data` to cut payload size ~33 %.

---

## Sequencing

1. **Now:** UI is on the prototype skin; underscore wrappers in `webapp.py` removed.
2. **Next:** Track A detection swap (YOLOv8n) — biggest quality jump per LOC.
3. **Then:** Track A embedding swap (OSNet or CLIP).
4. **Then:** Track B multi-patient enroll + multi-box overlay.
5. **Then:** Track B zones live + tracker smoothing.
6. **Cleanup wave (Track C):** FastAPI migration, webapp tests, payload format.
