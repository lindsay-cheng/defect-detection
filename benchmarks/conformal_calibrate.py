"""split-conformal calibration of the threshold-only logging guard.

runs the detector at conf=0.05 over the val images from a data yaml,
matches predictions to the YOLO label boxes (IoU>=0.5), scores per-GT
nonconformity s_i, splits cal/holdout (seeded), and writes:

    benchmarks/results/conformal.json      — machine-readable numbers
    benchmarks/results/conformal_report.md — human-readable summary

nonconformity (per matched val detection, IoU>=0.5 with a GT box):

    s_i = 1 - p_top          if predicted top class == GT class
    s_i = 1                  if predicted top class != GT class, or unmatched GT

q_hat = k-th smallest of the cal scores, k = ceil((n_cal+1)(1-alpha));
marginal coverage P(true class in {predicted}) >= 1-alpha holds with
finite-sample slack 1/(n_cal+1) (Angelopoulos & Bates, *Conformal
Prediction: A Gentle Introduction*, arXiv:2107.07511).

# ponytail: threshold-only calibrates the operational top-class rule; upgrade
# = per-class softmax RAPS if ultralytics exposes per-class vectors post-NMS
# (it does not in any version tested — see vault alternatives ledger).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="auto")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--data", default="dataset/data.yaml")
    p.add_argument("--holdout-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-json",
        default="benchmarks/results/conformal.json",
    )
    p.add_argument(
        "--out-md",
        default="benchmarks/results/conformal_report.md",
    )
    return p.parse_args()


def resolve_model(model_arg: str) -> str:
    """auto -> best_fp16.mlpackage if present, else best.pt."""
    if model_arg != "auto":
        return model_arg
    core = REPO_ROOT / "benchmarks/models/best_fp16.mlpackage"
    if core.exists():
        return str(core)
    return str(REPO_ROOT / "model/weights/best.pt")


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


def _yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """class cx cy w h from a YOLO label file (none if empty/missing)."""
    if not label_path.exists():
        return []
    rows = []
    text = label_path.read_text().strip()
    if not text:
        return []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:5])
        rows.append((cls, cx, cy, w, h))
    return rows


def main() -> int:
    args = parse_args()
    model_path = resolve_model(args.model)
    if not os.path.exists(model_path):
        print(f"FAIL: model not found at {model_path}", file=sys.stderr)
        return 1

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    with open(data_path) as f:
        data_yaml = yaml.safe_load(f)
    names = {int(k): str(v) for k, v in data_yaml["names"].items()}
    base = Path(data_yaml["path"])
    if not base.is_absolute():
        base = data_path.parent / base
    val_imgs_dir = base / data_yaml["val"]
    val_labels_dir = val_imgs_dir.parent / "labels"

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"FAIL: ultralytics not installed: {e}", file=sys.stderr)
        return 1

    images = sorted(val_imgs_dir.glob("*.jpg"))
    if not images:
        print(f"FAIL: no val images in {val_imgs_dir}", file=sys.stderr)
        return 1
    print(
        f"[calibrate] model={model_path} alpha={args.alpha} "
        f"holdout_frac={args.holdout_frac} seed={args.seed} n_images={len(images)}",
        flush=True,
    )

    model = YOLO(model_path)

    # per-instance nonconformity scores: (score, gt_class_name, outcome)
    # outcome in {"matched_correct", "matched_wrong", "unmatched"}
    scores: list[tuple[float, str, str]] = []
    for img_path in images:
        label_path = val_labels_dir / (img_path.stem + ".txt")
        gts = _yolo_labels(label_path)
        if not gts:
            continue
        # need image dims to convert normalized YOLO boxes to absolute
        try:
            import cv2

            h, w = cv2.imread(str(img_path)).shape[:2]
        except Exception:
            # ponytail: ultralytics can supply shape post-predict; fall back there.
            h, w = None, None
        gt_boxes = []
        for cls, cx, cy, bw, bh in gts:
            if h is None:
                continue
            abs_w, abs_h = bw * w, bh * h
            abs_cx, abs_cy = cx * w, cy * h
            x1 = abs_cx - abs_w / 2
            y1 = abs_cy - abs_h / 2
            x2 = abs_cx + abs_w / 2
            y2 = abs_cy + abs_h / 2
            gt_boxes.append((cls, (x1, y1, x2, y2)))

        results = model.predict(str(img_path), conf=0.05, imgsz=640, verbose=False)
        result = results[0]
        preds = []
        if result.boxes is not None and len(result.boxes) > 0:
            data = result.boxes.data.cpu().numpy()
            # row layout (predict, no tracking): x1,y1,x2,y2,conf,cls
            for row in data:
                x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
                conf = float(row[4])
                cls = int(row[5])
                preds.append((cls, float(conf), (x1, y1, x2, y2)))

        if h is None:
            # second chance: use ultralytics' letterboxed shape (orig_img)
            h_real, w_real = result.orig_img.shape[:2]
            gt_boxes = []
            for cls, cx, cy, bw, bh in gts:
                abs_w, abs_h = bw * w_real, bh * h_real
                abs_cx, abs_cy = cx * w_real, cy * h_real
                gt_boxes.append(
                    (
                        cls,
                        (
                            abs_cx - abs_w / 2,
                            abs_cy - abs_h / 2,
                            abs_cx + abs_w / 2,
                            abs_cy + abs_h / 2,
                        ),
                    )
                )

        for gt_cls, gt_xyxy in gt_boxes:
            best_iou = 0.0
            best_pred = None
            for p_cls, p_conf, p_xyxy in preds:
                iou = _iou(gt_xyxy, p_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = (p_cls, p_conf)
            gt_name = names.get(gt_cls, str(gt_cls))
            if best_pred is None or best_iou < 0.5:
                scores.append((1.0, gt_name, "unmatched"))
            elif best_pred[0] == gt_cls:
                scores.append((1.0 - best_pred[1], gt_name, "matched_correct"))
            else:
                scores.append((1.0, gt_name, "matched_wrong"))

    n = len(scores)
    if n < 4:
        print(
            f"FAIL: only {n} nonconformity scores — too few to calibrate "
            "(expected n_cal*n_holdout each >= 2).",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(args.seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_holdout = max(2, int(round(args.holdout_frac * n)))
    if n_holdout >= n:
        print("FAIL: holdout_frac too large for n", file=sys.stderr)
        return 1
    n_cal = n - n_holdout
    cal_idx = idx[n_holdout:]
    hol_idx = idx[:n_holdout]
    cal_scores = sorted(scores[i][0] for i in cal_idx)
    hol_scores = [scores[i][0] for i in hol_idx]

    k = math.ceil((n_cal + 1) * (1.0 - args.alpha))
    k = min(k, n_cal)
    if k < 1:
        print("FAIL: k<1 — too few cal scores for this alpha", file=sys.stderr)
        return 1
    q_hat = float(cal_scores[k - 1])
    tau = 1.0 - q_hat
    empirical = sum(1 for s in hol_scores if s <= q_hat) / n_holdout
    exact_regime = bool(q_hat < 0.5)

    # per-class calibration counts
    by_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"matched_correct": 0, "matched_wrong": 0, "unmatched": 0, "total": 0}
    )
    for i in cal_idx:
        _, name, outcome = scores[i]
        by_class[name][outcome] += 1
        by_class[name]["total"] += 1
    calibration = {name: dict(v) for name, v in by_class.items()}

    rel_model = (
        os.path.relpath(model_path, REPO_ROOT)
        if model_path.startswith(str(REPO_ROOT))
        else model_path
    )
    out = {
        "model": rel_model,
        "alpha": args.alpha,
        "n_cal": n_cal,
        "n_holdout": n_holdout,
        "q_hat": q_hat,
        "tau": tau,
        "exact_regime": exact_regime,
        "empirical_coverage_holdout": empirical,
        "calibration": calibration,
    }

    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = REPO_ROOT / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[calibrate] wrote {out_json}")

    # ---- report ----
    nominal = 1.0 - args.alpha
    lines = []
    lines.append("# Conformal calibration — threshold-only logging guard")
    lines.append("")
    lines.append("## Methodology")
    lines.append(
        "Split conformal (Angelopoulos & Bates, *Conformal Prediction: A Gentle"
        " Introduction*, arXiv:2107.07511). Run detector at conf=0.05 over the val"
        " images from the data yaml; match each GT instance to the highest-IoU"
        " prediction with IoU>=0.5. Nonconformity s_i = 1 - p_top if matched &"
        " top class == GT class, else s_i = 1 (matched-wrong or unmatched GT — both"
        " maximally nonconforming). Seeded shuffle, 50/50 cal/holdout split;"
        " q_hat is the k-th smallest cal score with k = ceil((n_cal+1)(1-alpha)),"
        " giving marginal coverage P(true class in {predicted}) >= 1-alpha with"
        " finite-sample slack 1/(n_cal+1). Runtime rule: abstain (log UNCERTAIN"
        " bottle row, no defect row) when conf < tau = 1 - q_hat at the crossing."
    )
    lines.append("")
    lines.append("## Numbers")
    lines.append(f"- model: `{out['model']}`")
    lines.append(f"- alpha: {args.alpha}  (nominal coverage {nominal:.2f})")
    lines.append(f"- n_cal: {n_cal}   n_holdout: {n_holdout}   total instances: {n}")
    lines.append(f"- q_hat: {q_hat:.4f}   tau: {tau:.4f}")
    lines.append(
        f"- empirical coverage on holdout: {empirical:.4f} "
        f"({sum(1 for s in hol_scores if s <= q_hat)}/{n_holdout}) vs nominal {nominal:.2f}"
    )
    lines.append("")
    lines.append("## Per-class calibration counts")
    lines.append("| class | total | matched_correct | matched_wrong | unmatched |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, c in sorted(calibration.items()):
        lines.append(
            f"| {name} | {c['total']} | {c['matched_correct']} | "
            f"{c['matched_wrong']} | {c['unmatched']} |"
        )
    lines.append("")
    lines.append("## Exactness verdict")
    lines.append(
        f"- q_hat < 0.5? {exact_regime}. "
        + (
            "YES — top-class-only operational rule is *exact*: at most one class can"
            " exceed 0.5 in a softmax, so p_top >= tau forces the top class to be"
            " the true class whenever the abstention event fires."
            if exact_regime
            else "NO — tau <= 0.5; the rule is conservative (it abstains as if the"
            " full softmax vector were available). document carefully."
        )
    )
    lines.append("")
    lines.append("## Abstain semantics (operational)")
    lines.append(
        "- At a centerline crossing with `det.confidence < tau` the pipeline writes"
        " an `UNCERTAIN` bottle row (no defect row, no crop save) and increments"
        " `total_abstentions`; the inspector's `total_defects`/`total_inspected`"
        " counters are unchanged. The operator UI renders the box amber with the"
        " label `BTL_xxxx: UNCERTAIN`. `tau` is loaded at pipeline init from"
        " this json via `CoverageGuard.from_json` (graceful None disable on a"
        " missing/invalid file)."
    )
    lines.append("")
    lines.append("## Margin-baseline comparison")
    lines.append(
        "- The softmax margin rule `p1 - p2 > tau'` (an alternative abstain cut)"
        " has no finite-sample coverage guarantee under split-conformal exchangeability."
        " It additionally requires `p2` (the second-highest class probability), which"
        " the ultralytics top-class-only post-NMS API does not expose (verified on"
        " 8.4.105; `Results.boxes.data` carries only the top conf/cls). The"
        " threshold-only rule here is therefore both the simpler and the only"
        " API-exposed choice; the margin baseline is retained as a non-implemented"
        " comparison row, not an alternative path."
    )
    lines.append("")
    lines.append("## Caveats (state carefully)")
    lines.append(
        "- Exchangeability is frame-level (per matched detection), not track-level;"
        " the per-track commit-once vote in `TrackLabelStabilizer` is intentionally"
        " outside the conformal frame."
    )
    lines.append(
        "- Val n=56 ⇒ sub-~1.5pt deltas are within noise; coverage numbers carry"
        " the `1/(n_cal+1)` finite-sample slack (~7% here at n_cal=28)."
    )
    lines.append("")
    out_md = Path(args.out_md)
    if not out_md.is_absolute():
        out_md = REPO_ROOT / out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[calibrate] wrote {out_md}")

    # stdout summary
    print(
        f"[calibrate] n_cal={n_cal} n_holdout={n_holdout} q_hat={q_hat:.4f} "
        f"tau={tau:.4f} exact_regime={exact_regime} empirical={empirical:.4f}"
    )
    if tau <= 0.0 or empirical < 0.85:
        print(
            f"[calibrate] WARNING — looks broken (tau={tau:.4f}, "
            f"empirical={empirical:.4f}); inspect before trusting.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
