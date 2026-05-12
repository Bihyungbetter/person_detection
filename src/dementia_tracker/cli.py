from __future__ import annotations

import argparse
import json
from pathlib import Path

from .demo_data import generate_demo_dataset
from .embeddings import AppearanceEmbedder
from .pipeline import VisionPipeline
from .registry import PatientRegistry
from .webapp import run_live_server
from .zones import load_zones


def _expand_image_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        path_text = str(path)
        if any(char in path_text for char in "*?[]"):
            parent = path.parent if str(path.parent) else Path(".")
            expanded.extend(sorted(parent.glob(path.name)))
        else:
            expanded.append(path)
    if not expanded:
        raise FileNotFoundError("No reference images matched the provided --images value.")
    return expanded


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dementia-tracker",
        description="Register patients, identify them in camera frames, and alert on monitored zones.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Generate demo data, register the demo patient, and run the pipeline.")
    demo.add_argument("--out", type=Path, default=Path("demo_run"), help="Output directory for the full demo run.")
    demo.add_argument("--threshold", type=float, default=0.72, help="Identity match threshold.")
    demo.add_argument("--no-annotate", action="store_true", help="Skip annotated frame output.")

    generate = subparsers.add_parser("generate-demo", help="Generate a synthetic static-camera dataset.")
    generate.add_argument("--out", type=Path, default=Path("demo_input"), help="Dataset output directory.")

    register = subparsers.add_parser("register", help="Register or update one patient from reference images.")
    register.add_argument("--registry", type=Path, required=True, help="Local patient registry JSON path.")
    register.add_argument("--patient-id", required=True, help="De-identified patient ID.")
    register.add_argument("--name", required=True, help="Display name for local demos.")
    register.add_argument("--images", type=Path, nargs="+", required=True, help="Reference person crops.")

    inspect = subparsers.add_parser("inspect-registry", help="Print a registry summary.")
    inspect.add_argument("--registry", type=Path, required=True, help="Local patient registry JSON path.")

    run = subparsers.add_parser("run", help="Run identification and zone alerting over a directory of frames.")
    run.add_argument("--frames", type=Path, required=True, help="Directory containing camera frames.")
    run.add_argument("--registry", type=Path, required=True, help="Local patient registry JSON path.")
    run.add_argument("--zones", type=Path, required=True, help="Monitored zones JSON path.")
    run.add_argument("--out", type=Path, required=True, help="Run artifact output directory.")
    run.add_argument("--threshold", type=float, default=0.72, help="Identity match threshold.")
    run.add_argument("--no-annotate", action="store_true", help="Skip annotated frame output.")

    serve = subparsers.add_parser("serve-live", help="Start the local webcam enrollment and detection app.")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host for the local app.")
    serve.add_argument("--port", type=int, default=8765, help="Bind port for the local app.")
    serve.add_argument(
        "--registry",
        type=Path,
        default=Path("data/live_registry.json"),
        help="Local registry path for webcam enrollment.",
    )
    serve.add_argument("--threshold", type=float, default=0.74, help="Live identity match threshold.")
    return parser


def _register_patient(args: argparse.Namespace) -> None:
    embedder = AppearanceEmbedder()
    registry = PatientRegistry.load(args.registry)
    profile = registry.add_patient_from_images(args.patient_id, args.name, _expand_image_paths(args.images), embedder)
    registry.save(args.registry)
    print(
        json.dumps(
            {
                "registry": str(args.registry),
                "patient_id": profile.patient_id,
                "name": profile.name,
                "reference_count": profile.reference_count,
            },
            indent=2,
        )
    )


def _run_pipeline(args: argparse.Namespace) -> None:
    registry = PatientRegistry.load(args.registry)
    zones = load_zones(args.zones)
    pipeline = VisionPipeline(registry=registry, zones=zones, threshold=args.threshold)
    summary = pipeline.run(args.frames, args.out, annotate=not args.no_annotate)
    print(json.dumps(summary, indent=2))


def _run_demo(args: argparse.Namespace) -> None:
    demo_root = args.out / "input"
    result_root = args.out / "results"
    demo_info = generate_demo_dataset(demo_root)

    registry_path = args.out / "patients.json"
    registry = PatientRegistry()
    registry.add_patient_from_images(
        patient_id=demo_info["patient_id"],
        name=demo_info["patient_name"],
        image_paths=[Path(path) for path in demo_info["reference_images"]],
        embedder=AppearanceEmbedder(),
    )
    registry.save(registry_path)

    zones = load_zones(Path(demo_info["zones_path"]))
    pipeline = VisionPipeline(registry=registry, zones=zones, threshold=args.threshold)
    summary = pipeline.run(Path(demo_info["frames_dir"]), result_root, annotate=not args.no_annotate)
    summary["registry_path"] = str(registry_path)
    print(json.dumps(summary, indent=2))


def _generate_demo(args: argparse.Namespace) -> None:
    info = generate_demo_dataset(args.out)
    print(json.dumps(info, indent=2))


def _inspect_registry(args: argparse.Namespace) -> None:
    registry = PatientRegistry.load(args.registry)
    print(json.dumps(registry.summary(), indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        _run_demo(args)
    elif args.command == "generate-demo":
        _generate_demo(args)
    elif args.command == "register":
        _register_patient(args)
    elif args.command == "inspect-registry":
        _inspect_registry(args)
    elif args.command == "run":
        _run_pipeline(args)
    elif args.command == "serve-live":
        run_live_server(host=args.host, port=args.port, registry_path=args.registry, threshold=args.threshold)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
