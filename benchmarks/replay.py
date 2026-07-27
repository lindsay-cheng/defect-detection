"""replay benchmark: measure per-frame class flicker & log-correctness vs ground truth.

# ponytail: one-file harness that drives DetectionPipeline over a full video and
# reconstructs crossing events purely from per-frame detections (first centerline
# hit per track ≈ Inspector semantics). upgrade = centroid interpolation between
# frames for sub-frame crossing timing; Hungarian matching instead of positional
# 1:1 alignment; track-disappearance-aware switch counting instead of adjacent
# observation entries.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict

from defect_detection.config import DetectorConfig
from defect_detection.inspection.pipeline import DetectionPipeline
from defect_detection.video import FrameReader

# ponytail: metrics live alongside this file; import as a sibling module so
# this dir stays runnable standalone without packaging tricks.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import fmt_ci, label_stability, wilson_interval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--conf", type=float, default=0.5)
    return p.parse_args()


def run_pipeline(video: str, model: str, conf: float) -> list[list[dict]]:
    """drive the pipeline over the full video; return per-frame detection lists.
    uses a throwaway temp db so the real database/ is never touched."""
    tmpdir = tempfile.mkdtemp(prefix="replay_db_")
    try:
        config = DetectorConfig(
            model_path=model,
            conf_threshold=conf,
            db_path=os.path.join(tmpdir, "bench.db"),
            save_images=False,
            images_dir=tmpdir,
        )
        pipe = DetectionPipeline(config)
        reader = FrameReader(video, loop=False)
        pipe.start_session()
        per_frame: list[list[dict]] = []
        try:
            while True:
                frame = reader.read()
                if frame is None:
                    break
                _, dets = pipe.detect_frame(frame)
                per_frame.append(list(dets))
        finally:
            reader.release()
            pipe.cleanup()
        return per_frame
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def build_records(per_frame: list[list[dict]]) -> list[dict]:
    """flatten per-frame detections to jsonl records."""
    out = []
    for i, dets in enumerate(per_frame):
        for d in dets:
            x, _y, w, _h = d["bbox"]
            out.append(
                {
                    "frame": i,
                    "track_id": d.get("track_id"),
                    "class": d.get("defect_type"),
                    "conf": d.get("confidence"),
                    "cx": x + w // 2,
                    "on_centerline": bool(d.get("on_centerline", False)),
                    "display_id": d.get("display_id"),
                    "logged": bool(d.get("logged", False)),
                }
            )
    return out


def group_by_track(per_frame: list[list[dict]]) -> dict[int, list[tuple]]:
    """track_id -> ordered (frame, class, conf, on_centerline, display_id, logged)."""
    by_track: dict[int, list[tuple]] = defaultdict(list)
    for i, dets in enumerate(per_frame):
        for d in dets:
            tid = d.get("track_id")
            if tid is None:
                continue
            by_track[tid].append(
                (
                    i,
                    d.get("defect_type"),
                    d.get("confidence"),
                    bool(d.get("on_centerline", False)),
                    d.get("display_id"),
                    bool(d.get("logged", False)),
                )
            )
    return by_track


def classes_by_track(
    per_frame: list[list[dict]],
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """per-track raw vs stabilized class sequences for label_stability.

    raw uses raw_defect_type when the inspector set it, else the (stabilized)
    defect_type — that fallback applies outside the pipeline (no stabilizer).
    # ponytail: second grouping pass is simpler than threading raw through the
    # existing group_by_track tuple;same O(n) cost.
    """
    raw: dict[int, list[str]] = defaultdict(list)
    stable: dict[int, list[str]] = defaultdict(list)
    for dets in per_frame:
        for d in dets:
            tid = d.get("track_id")
            if tid is None:
                continue
            stable[tid].append(d.get("defect_type"))
            raw[tid].append(d.get("raw_defect_type", d.get("defect_type")))
    return raw, stable


def reconstruct_crossings(by_track: dict[int, list[tuple]]) -> list[tuple]:
    """(frame, track_id, display_id, class_at_crossing, conf, logged), ordered by frame.
    # ponytail: first-centerline-frame reconstruction mirrors Inspector._assign_display_ids.
    """
    crossings = []
    for tid, hist in by_track.items():
        for frame, cls, conf, on_c, disp, logged in hist:
            if on_c:
                crossings.append((frame, tid, disp, cls, conf, logged))
                break
    crossings.sort(key=lambda c: c[0])
    return crossings


def flicker_metrics(
    by_track: dict[int, list[tuple]], crossing_frame_by_track: dict[int, int]
) -> dict:
    """per-track & summary flicker stats; timelines only for flickering tracks.
    # ponytail: a 'switch' is a class change between adjacent observations of the
    track (not adjacent video frames), so detection gaps don't double-count.
    """
    total_tracks = len(by_track)
    n_flickering = 0
    total_switches = 0
    near_switches = 0
    timelines = []
    for tid, hist in by_track.items():
        classes = [h[1] for h in hist]
        frames = [h[0] for h in hist]
        distinct_classes = set(classes)
        n_switches = sum(1 for a, b in zip(classes, classes[1:], strict=False) if a != b)
        total_switches += n_switches
        cf = crossing_frame_by_track.get(tid)
        if cf is not None:
            for _f0, f1, c0, c1 in zip(frames, frames[1:], classes, classes[1:], strict=False):
                if c0 != c1 and abs(f1 - cf) <= 10:
                    near_switches += 1
        if len(distinct_classes) > 1:
            n_flickering += 1
            runs = []
            cur_cls, cur_n = classes[0], 0
            for c in classes:
                if c == cur_cls:
                    cur_n += 1
                else:
                    runs.append((cur_cls, cur_n))
                    cur_cls, cur_n = c, 1
            runs.append((cur_cls, cur_n))
            timelines.append(
                {
                    "track_id": tid,
                    "distinct": sorted(distinct_classes),
                    "switches": n_switches,
                    "timeline": " -> ".join(f"{c}(x{n})" for c, n in runs),
                }
            )
    return {
        "total_tracks": total_tracks,
        "flickering_tracks": n_flickering,
        "pct_flickering": (100.0 * n_flickering / total_tracks) if total_tracks else 0.0,
        "total_switches": total_switches,
        "near_crossing_switches": near_switches,
        "timelines": sorted(timelines, key=lambda t: t["track_id"]),
    }


def correctness_vs_gt(crossings: list[tuple], gt_crossings: list[dict]) -> dict:
    """positional 1:1 alignment with GT; classify each crossing. upgrade = Hungarian."""
    n_obs = len(crossings)
    n_gt = len(gt_crossings)
    n_match = min(n_obs, n_gt)
    table = []
    counts = defaultdict(int)
    for i in range(n_match):
        gt_cls = gt_crossings[i]["class"]
        obs_cls = crossings[i][3]
        logged = crossings[i][5]
        if gt_cls == "good":
            verdict = "OK-good" if not logged else "FALSE-POSITIVE"
        else:
            if not logged:
                verdict = "MISS"
            elif obs_cls == gt_cls:
                verdict = "OK-defect"
            else:
                verdict = "WRONG-TYPE"
        table.append((i + 1, gt_cls, obs_cls, logged, verdict))
        counts[verdict] += 1
    return {
        "observed": n_obs,
        "expected": n_gt,
        "extra_observed": max(0, n_obs - n_gt),
        "missing": max(0, n_gt - n_obs),
        "table": table,
        "counts": dict(counts),
    }


def render_report(
    video: str,
    model: str,
    conf: float,
    gt: str,
    crossings: list[tuple],
    flick: dict,
    corr: dict,
    gt_tags: list[str | None],
    stab_raw: dict,
    stab_stable: dict,
) -> str:
    lines = []
    lines.append(f"# Replay Benchmark — `{video}`")
    lines.append("")
    lines.append("## Config")
    lines.append(f"- video: `{video}`")
    lines.append(f"- model: `{model}`")
    lines.append(f"- conf: `{conf}`")
    lines.append(f"- gt: `{gt}`")
    lines.append("")

    lines.append("## Crossing Reconstruction (per-track first-centerline frame)")
    # ponytail: tags column appears only when GT carries tags, so untagged GT
    # reproduces the original table byte-for-byte.
    has_tags = any(t is not None for t in gt_tags)
    if has_tags:
        lines.append(
            "| seq | frame | track_id | display_id | class_at_crossing | conf | logged | tags |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
    else:
        lines.append("| seq | frame | track_id | display_id | class_at_crossing | conf | logged |")
        lines.append("|---|---|---|---|---|---|---|")
    for i, c in enumerate(crossings):
        frame, tid, disp, cls, conf_, logged = c
        row = f"| {i + 1} | {frame} | {tid} | {disp} | {cls} | {conf_:.3f} | {logged} |"
        if has_tags:
            tags = gt_tags[i] if i < len(gt_tags) and gt_tags[i] is not None else ""
            row += f" {tags} |"
        lines.append(row)
    lines.append("")

    lines.append("## Flicker Metrics")
    lines.append(f"- total tracks: {flick['total_tracks']}")
    lines.append(
        f"- tracks with >1 class: {flick['flickering_tracks']} / {flick['total_tracks']} "
        f"({flick['pct_flickering']:.1f}%)"
    )
    lines.append(f"- total class switches: {flick['total_switches']}")
    lines.append(f"- switches within ±10 frames of crossing: {flick['near_crossing_switches']}")
    lines.append("- per-track flicker timelines:")
    if flick["timelines"]:
        for t in flick["timelines"]:
            lines.append(f"  - trk {t['track_id']}: {t['timeline']}")
    else:
        lines.append("  - (none)")
    lines.append("")

    lines.append("## Correctness vs Ground Truth (positional alignment)")
    lines.append(f"- crossings detected: {corr['observed']} / expected: {corr['expected']}")
    lines.append(
        f"- extra observed (unmatched): {corr['extra_observed']} | missing: {corr['missing']}"
    )
    vc = corr["counts"]
    lines.append(
        f"- verdicts: OK-good={vc.get('OK-good', 0)}, OK-defect={vc.get('OK-defect', 0)}, "
        f"FALSE-POSITIVE={vc.get('FALSE-POSITIVE', 0)}, MISS={vc.get('MISS', 0)}, "
        f"WRONG-TYPE={vc.get('WRONG-TYPE', 0)}"
    )
    lines.append("")
    lines.append("| seq | gt_class | obs_class | logged | verdict |")
    lines.append("|---|---|---|---|---|")
    for seq, gt_cls, obs_cls, logged, verdict in corr["table"]:
        lines.append(f"| {seq} | {gt_cls} | {obs_cls} | {logged} | {verdict} |")
    lines.append("")

    # --- additive sections (do not alter any line above) ---
    lines.append("## Correctness confidence intervals")
    expected = corr["expected"]
    detected = min(corr["observed"], expected)
    rate_lo, rate_hi = wilson_interval(detected, expected)
    lines.append(
        f"- crossing-detection rate: {detected}/{expected} = "
        f"{(detected / expected) if expected else 0.0:.3f} "
        f"CI {fmt_ci(rate_lo, rate_hi)}"
    )
    vc = corr["counts"]
    ok_good = vc.get("OK-good", 0)
    ok_defect = vc.get("OK-defect", 0)
    correct = ok_good + ok_defect
    clo, chi = wilson_interval(correct, expected)
    lines.append(
        f"- correct-log rate (OK-good+OK-defect)/expected: {correct}/{expected} = "
        f"{(correct / expected) if expected else 0.0:.3f} "
        f"CI {fmt_ci(clo, chi)}"
    )
    lines.append(f"- per-verdict proportions (n = expected = {expected}):")
    for verdict_name in ("OK-good", "OK-defect", "FALSE-POSITIVE", "MISS", "WRONG-TYPE"):
        k = vc.get(verdict_name, 0)
        vlo, vhi = wilson_interval(k, expected)
        lines.append(
            f"  - {verdict_name}: {k}/{expected} = "
            f"{(k / expected) if expected else 0.0:.3f} CI {fmt_ci(vlo, vhi)}"
        )
    lines.append("")

    lines.append("## Label stability")
    lines.append(
        f"- raw per-frame classes (raw_defect_type): mean stability "
        f"{stab_raw['mean']:.3f}, perfectly-stable {stab_raw['n_perfect']}/{stab_raw['n_tracks']} "
        f"({100.0 * stab_raw['prop_perfect']:.1f}%) "
        f"CI {fmt_ci(*stab_raw['ci_perfect'])}"
    )
    lines.append(
        f"- stabilized classes (defect_type):     mean stability "
        f"{stab_stable['mean']:.3f}, perfectly-stable "
        f"{stab_stable['n_perfect']}/{stab_stable['n_tracks']} "
        f"({100.0 * stab_stable['prop_perfect']:.1f}%) "
        f"CI {fmt_ci(*stab_stable['ci_perfect'])}"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    per_frame = run_pipeline(args.video, args.model, args.conf)
    records = build_records(per_frame)
    by_track = group_by_track(per_frame)
    crossings = reconstruct_crossings(by_track)
    crossing_frame_by_track = {c[1]: c[0] for c in crossings}
    flick = flicker_metrics(by_track, crossing_frame_by_track)
    with open(args.gt) as f:
        gt = json.load(f)
    gt_crossings = gt["crossings"]
    corr = correctness_vs_gt(crossings, gt_crossings)
    # ponytail: tolerate an optional `tags` field per crossing; echoed into the
    # crossing table when present, ignored downstream.
    gt_tags = [c.get("tags") for c in gt_crossings]
    raw_classes, stable_classes = classes_by_track(per_frame)
    stab_raw = label_stability(raw_classes)
    stab_stable = label_stability(stable_classes)
    report = render_report(
        args.video,
        args.model,
        args.conf,
        args.gt,
        crossings,
        flick,
        corr,
        gt_tags,
        stab_raw,
        stab_stable,
    )

    out_jsonl = args.out_prefix + ".jsonl"
    out_md = args.out_prefix + ".md"
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(out_md, "w") as f:
        f.write(report)
    print(report)
    print(f"\n[wrote] {out_jsonl} ({len(records)} detection records)")
    print(f"[wrote] {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
