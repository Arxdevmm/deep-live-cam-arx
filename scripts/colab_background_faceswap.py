#!/usr/bin/env python3
"""Replace a target image's background, then run Deep-Live-Cam face swap."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from modules.background_3d import Background3DEngine


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the 2.5D image background replacement to a target, then "
            "face-swap the result."
        )
    )
    parser.add_argument("--source", required=True, type=Path, help="Source-face image")
    parser.add_argument("--target", required=True, type=Path, help="Target portrait image")
    parser.add_argument("--background", required=True, type=Path, help="Replacement background image")
    parser.add_argument("--output", required=True, type=Path, help="Combined output image")
    parser.add_argument(
        "--execution-provider",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="ONNX execution provider (default: use CUDA when available)",
    )
    parser.add_argument("--parallax", type=float, default=0.35, help="Parallax strength from 0 to 1")
    parser.add_argument("--perspective", type=float, default=0.25, help="Perspective strength from 0 to 1")
    parser.add_argument("--blur", type=float, default=0.05, help="Background blur strength")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {
        "source": args.source.expanduser().resolve(),
        "target": args.target.expanduser().resolve(),
        "background": args.background.expanduser().resolve(),
    }
    output = args.output.expanduser().resolve()

    for label, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label.capitalize()} image not found: {path}")
        if cv2.imread(str(path)) is None:
            raise ValueError(f"Could not decode {label} image: {path}")
    if output.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported output image extension: {output.suffix}")
    if output in inputs.values():
        raise ValueError("Output path must not overwrite an input image.")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; choose a new path: {output}")

    available_providers = set(ort.get_available_providers())
    if args.execution_provider == "cuda" and "CUDAExecutionProvider" not in available_providers:
        raise RuntimeError("CUDA was requested, but CUDAExecutionProvider is unavailable.")
    provider = args.execution_provider
    if provider == "auto":
        provider = "cuda" if "CUDAExecutionProvider" in available_providers else "cpu"

    output.parent.mkdir(parents=True, exist_ok=True)
    target_frame = cv2.imread(str(inputs["target"]))
    engine = Background3DEngine()
    if not engine.available:
        message = engine.error or "Background segmentation is unavailable."
        engine.close()
        raise RuntimeError(message)

    try:
        with tempfile.TemporaryDirectory(prefix="dlc-background-") as temp_dir:
            background_target = Path(temp_dir) / "target_with_background.png"
            composited = engine.process(
                target_frame,
                str(inputs["background"]),
                parallax_strength=args.parallax,
                perspective_strength=args.perspective,
                blur_strength=args.blur,
            )
            depth_available = engine.depth_available
            depth_error = engine.depth_error
            if engine.error:
                raise RuntimeError(engine.error)
            if composited is None or not cv2.imwrite(str(background_target), composited):
                raise RuntimeError("Could not save the background-replaced target image.")
            if not depth_available:
                print("Depth model unavailable; using the perspective/parallax fallback.")
                if depth_error:
                    print(depth_error)

            print(f"Using ONNX execution provider: {provider}")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT_DIR / "run.py"),
                    "--execution-provider",
                    provider,
                    "--source",
                    str(inputs["source"]),
                    "--target",
                    str(background_target),
                    "--output",
                    str(output),
                ],
                cwd=ROOT_DIR,
                check=True,
            )
    finally:
        engine.close()

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Face-swap did not produce an output image: {output}")
    print(f"Combined image saved to: {output}")


if __name__ == "__main__":
    main()
