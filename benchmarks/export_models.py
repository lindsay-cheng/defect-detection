"""export the trained yolo11 detector to CoreML (FP16, INT8) and ONNX (FP32).

# ponytail: stdlib + ultralytics only. ultralytics' YOLO.export already handles ct/onnx,
# NMS baking, and weight palettization; we just drive it sequentially and rename the
# artifacts (ultralytics names outputs after the weights file) to deterministic names.
#
# coreml INT8 here uses quantize="w8a16": coremltools k-means palettization of conv weights
# to 8 bits with FP16 activations (verified: ultralytics 8.4.105 utils.export.coreml.torch2coreml
# `weight_int8 = quantize in {8, "w8a16"}` + `cto.palettize_weights(mode="kmeans", nbits=8)`).
# this is WEIGHT-ONLY — no activation calibration is performed, so no `data=` is required
# (ultralytics validate_args actually rejects `data` for format='coreml'/'mlmodel'). the
# prior-agent note that int8 requires dataset/data.yaml calibration applied to
# activation-quantized formats (onnx/openvino/engine/litert), not CoreML.
#
# NOTE: numpy must be <=2.3.5 for coremltools 9.0; ultralytics' check_requirements will
# auto-downgrade a too-new numpy on first export, but a process that already imported numpy
# 2.4.x will keep the stale version in sys.modules and the export silently crashes with
# `only 0-dimensional arrays can be converted to Python scalars`. we assert numpy<=2.3.5
# up front so a fresh interpreter is in a known good state.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKER_PATH = REPO_ROOT / "src" / "defect_detection" / "inference" / "trackers" / "bytetrack.yaml"

# (final_name, kwargs dict) — ultralytics earns its keep here; we just pass through.
EXPORTS = [
    ("best_fp16.mlpackage", {"format": "coreml", "quantize": 16}),
    ("best_int8.mlpackage", {"format": "coreml", "quantize": "w8a16"}),
    ("best_fp32.onnx", {"format": "onnx", "simplify": True}),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--weights", default="model/weights/best.pt", help="source .pt weights")
    p.add_argument("--outdir", default="benchmarks/models", help="destination dir")
    p.add_argument("--imgsz", type=int, default=640, help="export image size (fixed-shape)")
    return p.parse_args()


def file_size_mb(path: Path) -> float:
    total = 0
    if path.is_dir():
        for root, _, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
    elif path.is_file():
        total = path.stat().st_size
    return total / (1024 * 1024)


def check_numpy_compat() -> None:
    """coremltools 9.0 crashes silently if numpy 2.4.x is already imported in this process."""
    import numpy as np

    major, minor = [int(x) for x in np.__version__.split(".")[:2]]
    if (major, minor) > (2, 3):
        sys.exit(
            f"ERROR: numpy {np.__version__} is too new for coremltools 9.0 "
            f"(needs <=2.3.5). rerun in a fresh interpreter after:\n"
            f"  ./venv/bin/pip install 'numpy<=2.3.5'"
        )


def export_one(weights: str, fmt: str, imgsz: int, extra: dict) -> Path:
    """run a single ultralytics export and return the exported artifact path."""
    from ultralytics import YOLO

    model = YOLO(weights)
    kwargs = {"imgsz": imgsz, "nms": True, **extra}
    t0 = time.perf_counter()
    out = model.export(**kwargs)
    print(f"[export] {fmt} extra={extra} -> {out} ({time.perf_counter() - t0:.1f}s)", flush=True)
    return Path(out)


def main() -> int:
    args = parse_args()
    check_numpy_compat()
    weights = args.weights
    if not os.path.isabs(weights):
        weights = str(REPO_ROOT / weights)
    outdir = args.outdir
    if not os.path.isabs(outdir):
        outdir = REPO_ROOT / outdir
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    common_flags = f"imgsz={args.imgsz}, nms=True"
    pybin = os.path.dirname(sys.executable)
    import subprocess

    def ver(mod):
        return subprocess.check_output(
            [sys.executable, "-c", f"import {mod};print({mod}.__version__)"], text=True
        ).strip()

    manifest = [
        "# Export manifest",
        "",
        f"source weights: `{args.weights}`",
        f"common flags: {common_flags}",
        f"torch: {ver('torch')}",
        f"ultralytics: {ver('ultralytics')}",
        f"coremltools: {ver('coremltools')}",
        "",
        "| artifact | format | quantize | size MB | export flags |",
        "|---|---|---|---:|---|",
    ]
    _ = pybin

    for final_name, extra in EXPORTS:
        quantize = extra.get("quantize")
        print(
            f"\n=== export {final_name} (format={extra['format']}, quantize={quantize!r}) ===",
            flush=True,
        )
        exported = export_one(weights, extra["format"], args.imgsz, extra)
        final_path = outdir / final_name
        if final_path.exists():
            if final_path.is_dir():
                shutil.rmtree(final_path)
            else:
                final_path.unlink()
        shutil.move(str(exported), str(final_path))
        size_mb = file_size_mb(final_path)
        flags = common_flags + "".join(f", {k}={v!r}" for k, v in extra.items())
        manifest.append(
            f"| {final_name} | {extra['format']} | {quantize!r} | {size_mb:.2f} | {flags} |"
        )
        print(f"[export] -> {final_path}  ({size_mb:.2f} MB)", flush=True)

    (outdir / "MANIFEST.md").write_text("\n".join(manifest) + "\n")
    print(f"\n[export] wrote {outdir / 'MANIFEST.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
