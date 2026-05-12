# Dementia Patient Vision Tracker

This repository is focused on the first working slice of the product: registering a patient from reference images, identifying that patient in a static camera feed, tracking their movement, and emitting zone-entry alert events.

The current implementation is deliberately local-first and lightweight:

- no cloud upload is required for video frames
- no pretrained model download is required
- all registry and run artifacts are local JSON/images
- configurable monitored zones are loaded from JSON

It is an MVP vision pipeline, not a clinical-grade biometric system. The next production step should replace the built-in motion detector with a stronger person detector/re-ID model while keeping the registry, tracking, zone, and alert interfaces.

## Quick Demo

Use the bundled Codex Python runtime if `python` is not on PATH:

```powershell
$py = "C:\Users\micha\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$env:PYTHONPATH = "src"
& $py -m dementia_tracker.cli demo --out demo_run
```

Outputs:

- `demo_run/input/frames/`: synthetic camera frames
- `demo_run/input/reference/`: patient reference crop
- `demo_run/patients.json`: local patient registry with appearance embedding
- `demo_run/results/events.jsonl`: alert events
- `demo_run/results/summary.json`: run summary
- `demo_run/results/annotated_frames/`: frames with zones, tracks, and identities drawn

## Detect Yourself With A Webcam

Start the local webcam app:

```powershell
$py = "C:\Users\micha\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$env:PYTHONPATH = "src"
& $py -m dementia_tracker.cli serve-live --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

Click `Start Camera`, move out of the guide rectangle, click `Set Empty Background`, stand inside the guide rectangle, click `Enroll Me`, then click `Start Detecting`.

The live app uses OpenCV when installed. It tries OpenCV Haar face detection first, then falls back to foreground cropping from the empty background. The browser never sends frames to a cloud service. It posts frames to the local Python server, which stores your local reference embedding in `data/live_registry.json` and the empty-frame reference in `data/live_background.jpg`.

If OpenCV is not installed:

```powershell
& $py -m pip install opencv-python-headless --target .codex_runtime
```

## Register A Patient

```powershell
$env:PYTHONPATH = "src"
& $py -m dementia_tracker.cli register `
  --registry data/patients.json `
  --patient-id patient-001 `
  --name "Patient 001" `
  --images .\reference_images\patient_001_*.png
```

Use de-identified IDs in real deployments. Patient names are supported for demo readability, but production alert payloads should avoid PHI when possible.

## Run On Camera Frames

```powershell
$env:PYTHONPATH = "src"
& $py -m dementia_tracker.cli run `
  --frames .\camera_frames `
  --registry data/patients.json `
  --zones configs\zones.example.json `
  --out runs\front-door-check `
  --threshold 0.72
```

Frames are processed in filename order. The default detector assumes a fixed camera and detects moving person candidates against the first frame as the background.

## Zone Format

Zones are normalized polygons in image coordinates where `[0, 0]` is the top-left and `[1, 1]` is the bottom-right.

```json
{
  "zones": [
    {
      "id": "front_door",
      "name": "Front door",
      "type": "restricted",
      "polygon": [[0.74, 0.0], [1.0, 0.0], [1.0, 1.0], [0.74, 1.0]]
    }
  ]
}
```

Alert events are emitted only when a matched patient enters a `restricted` or `high_risk` zone.

## Tests

```powershell
$env:PYTHONPATH = "src"
& $py -m unittest discover -s tests
```
