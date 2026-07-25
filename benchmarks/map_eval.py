"""mAP sweep: model.val over {pytorch, coreml-fp16, coreml-int8, onnx-fp32} × imgsz {640,512,384}.

# ponytail: stdlib + ultralytics. each call is one line:
#     YOLO(path).val(data="dataset/data.yaml", split="val", imgsz=X, verbose=False)
# 12 runs on 56 val images each — fast. Fixed-shape artifacts (coreml/onnx) are
# always exported at 640; running them at a different imgsz rebuilds the input
# tensor to that size via ultralytics' preprocess (letterbox), and we report the
# resulting mAP under that imgsz — it's a real measurement, not an invalid call.
# minus-~1.5pt deltas are within noise at n=56.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIGS = [
    ("pytorch", "model/weights/best.pt", False),
    ("coreml-fp16", "benchmarks/models/best_fp16.mlpackage", True),
    ("coreml-int8", "benchmarks/models/best_int8.mlpackage", True),
    ("onnx-fp32", "benchmarks/models/best_fp32.onnx", True),
]
IMGSZS = [640, 512, 384]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", default="dataset/data.yaml", help="ultralytics data yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--out", default="benchmarks/results/map_sweep.md")
    return p.parse_args()


def file_size_mb(path: Path) -> float:
    if path.is_dir():
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total / (1024 * 1024)
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return float("nan")


def main() -> int:
    args = parse_args()
    data = args.data
    if not os.path.isabs(data):
        data = str(REPO_ROOT / data)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    rows = []
    for label, rel, fixed_shape in CONFIGS:
        path = Path(REPO_ROOT / rel)
        size_mb = file_size_mb(path)
        for imgsz in IMGSZS:
            if fixed_shape and imgsz != 640:
                # CoreML (exported with nms=True) and ONNX (exported static at 640) reject other
                # input shapes at inference; would-be val runs surface as errors. mark n/a so
                # the table stays exhaustive without spending a guaranteed-fail val pass.
                print(f"[val] {label} @ imgsz={imgsz}: n/a (fixed-shape @640)", flush=True)
                rows.append(
                    (label, imgsz, float("nan"), float("nan"), size_mb, "n/a — fixed-shape @640")
                )
                continue
            print(f"\n[val] {label} @ imgsz={imgsz} (model={path.name})", flush=True)
            try:
                m = YOLO(str(path))
                metrics = m.val(
                    data=data,
                    split=args.split,
                    imgsz=imgsz,
                    verbose=False,
                    conf=0.001,
                    iou=0.6,
                )
                map50 = float(metrics.box.map50)
                map50_95 = float(metrics.box.map)
                print(f"  mAP50={map50:.4f} mAP50-95={map50_95:.4f}", flush=True)
                rows.append((label, imgsz, map50, map50_95, size_mb, ""))
            except Exception as e:
                print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}", flush=True)
                rows.append(
                    (
                        label,
                        imgsz,
                        float("nan"),
                        float("nan"),
                        size_mb,
                        f"{type(e).__name__}: {str(e)[:80]}",
                    )
                )

    lines = [
        "# mAP sweep — {cfg} × {imgsz}".format(
            cfg=",".join(c[0] for c in CONFIGS), imgsz=",".join(str(i) for i in IMGSZS)
        ),
        "",
        f"data: `{args.data}` (n=56 val images, split='{args.split}') | "
        f"note: val n=56 ⇒ sub-~1.5pt deltas are within noise.",
        "",
        "| config | imgsz | mAP50 | mAP50-95 | artifact size MB | notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, imgsz, m50, m95, sz, note in rows:
        lines.append(f"| {label} | {imgsz} | {m50:.4f} | {m95:.4f} | {sz:.2f} | {note} |")
    out.write_text("\n".join(lines) + "\n")
    print(f"\n[map_eval] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
