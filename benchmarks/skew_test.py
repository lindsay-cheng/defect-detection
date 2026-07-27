"""train/serve skew test (Breck-style): pytorch checkpoint vs coreml export.

loads both backends, randomly samples n seeded val images, runs predict on
each with conf=0.25/imgsz=640, and reports top-1 argmax agreement + mean
abs confidence delta on matched top-1 boxes (IoU>=0.5 of the highest-conf
box per image).

# ponytail: top-1 box matching only — simplest correct sanity check. upgrade
# = full Hungarian matching across all boxes plus per-class confusion matrix.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pt", default="model/weights/best.pt")
    p.add_argument("--coreml", default="benchmarks/models/best_fp16.mlpackage")
    p.add_argument("--val-dir", default="dataset/split/val/images")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default=None, help="write report to this path")
    return p.parse_args()


def _top1(result) -> tuple[int, float, tuple[int, int, int, int]] | None:
    """highest-confidence box: (class_id, conf, (x1,y1,x2,y2))."""
    box = result.boxes
    if box is None or len(box) == 0:
        return None
    data = box.data.cpu().numpy()
    # ultralytics row layout: x1,y1,x2,y2,conf,cls (predict, no tracking)
    i = int(data[:, 4].argmax())
    row = data[i]
    return int(row[5]), float(row[4]), (int(row[0]), int(row[1]), int(row[2]), int(row[3]))


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return (inter / union) if union > 0 else 0.0


def main() -> int:
    args = parse_args()
    report_lines = []
    report_lines.append("# train/serve skew test")
    report_lines.append("")
    report_lines.append(f"- n: {args.n} | seed: {args.seed}")
    report_lines.append(f"- pt: `{args.pt}`")
    report_lines.append(f"- coreml: `{args.coreml}`")
    report_lines.append(f"- conf: {args.conf} | imgsz: {args.imgsz}")
    report_lines.append("")

    # coreml artifact is deployment-dependent; missing -> SKIP (exit 0)
    if not os.path.exists(args.coreml):
        msg = (
            f"SKIP: coreml artifact not found at `{args.coreml}` — "
            "skew test is artifact-dependent, not for CI."
        )
        print(msg)
        report_lines.append(msg)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text("\n".join(report_lines) + "\n")
        return 0

    if not os.path.exists(args.pt):
        msg = f"FAIL: pytorch weights not found at `{args.pt}`"
        print(msg)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"FAIL: ultralytics not installed: {e}")
        return 1

    pt_model = YOLO(args.pt)
    core_model = YOLO(args.coreml)

    img_dir = Path(args.val_dir)
    imgs = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        print(f"FAIL: no val images in {img_dir}")
        return 1

    rng = random.Random(args.seed)
    n = min(args.n, len(imgs))
    sample = rng.sample(imgs, n)

    n_agree = 0
    conf_diffs = []
    per_image = []
    for path in sample:
        pt_res = pt_model.predict(str(path), conf=args.conf, imgsz=args.imgsz, verbose=False)
        co_res = core_model.predict(str(path), conf=args.conf, imgsz=args.imgsz, verbose=False)
        pt_top = _top1(pt_res[0])
        co_top = _top1(co_res[0])
        agrees = pt_top is not None and co_top is not None and pt_top[0] == co_top[0]
        if agrees:
            n_agree += 1
            if pt_top[2] and co_top[2] and _iou(pt_top[2], co_top[2]) >= 0.5:
                conf_diffs.append(abs(pt_top[1] - co_top[1]))
        per_image.append(
            (path.name, pt_top[0] if pt_top else None, co_top[0] if co_top else None, agrees)
        )

    agreement = (n_agree / n) if n else 0.0
    mean_abs_diff = (sum(conf_diffs) / len(conf_diffs)) if conf_diffs else 0.0
    pass_agree = agreement >= 0.95
    pass_diff = mean_abs_diff <= 0.10
    verdict = "PASS" if (pass_agree and pass_diff) else "FAIL"

    report_lines.append("## per-image")
    report_lines.append("| image | pt_class | coreml_class | top1 agree |")
    report_lines.append("|---|---|---|---|")
    for name, pc, cc, ag in per_image:
        report_lines.append(f"| {name} | {pc} | {cc} | {ag} |")
    report_lines.append("")
    report_lines.append(
        f"- top-1 agreement: {n_agree}/{n} = {agreement:.3f} (threshold 0.95): "
        f"{'PASS' if pass_agree else 'FAIL'}"
    )
    report_lines.append(
        f"- mean|Δconf| on matched top-1 (IoU≥0.5): {mean_abs_diff:.4f} "
        f"over {len(conf_diffs)} matched boxes (threshold 0.10): "
        f"{'PASS' if pass_diff else 'FAIL'}"
    )
    report_lines.append(f"- verdict: {verdict}")
    report_lines.append("")

    out = "\n".join(report_lines)
    print(out)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out + "\n")
        print(f"\n[wrote] {args.out}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
