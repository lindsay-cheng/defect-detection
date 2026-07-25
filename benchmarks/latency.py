"""latency harness: measure model + end-to-end pipeline latency for the inspection stack.

# ponytail: one reusable harness, stdlib + project deps only. two measurement modes share
# the same preload + run loop so the model-timing numbers and the e2e numbers come from the
# same workload. deliberate simplifications:
#   - frames are preloaded into memory once (warmup+frames of them) so decode jitter is
#     removed from the engine-mode timing; this is a model-latency probe, not a decode probe.
#     upgrade = ring buffer to bound RAM for very long runs.
#   - track(persist=True) is run sequentially across all measured frames so ByteTrack track
#     ids accumulate exactly as in production; we do NOT reset between frames within a run.
#   - InferenceEngine.track accepts neither device nor imgsz, so engine mode loads YOLO()
#     directly in the harness (matching the engine's exact track() call signature plus the
#     device/imgsz kwargs). src/ is left untouched.
#   - pipeline mode reports wall-clock per-frame read + detect_frame only; the per-stage
#     ultralytics speed breakdown is NOT accessible without breaking detection_frame's
#     interface, so it is omitted there (engine mode reports it).
#   - reused for coreml/onnx artifacts later: just point --model at the export. the
#     ultralytics YOLO() loader handles .mlpackage / .onnx / .pt transparently.
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKER_PATH = str(
    REPO_ROOT / "src" / "defect_detection" / "inference" / "trackers" / "bytetrack.yaml"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model", required=True, help="ultralytics-loadable model (.pt / .mlpackage / .onnx)"
    )
    p.add_argument(
        "--device",
        default=None,
        help="device override (passed to track()); None = ultralytics auto",
    )
    p.add_argument("--imgsz", type=int, default=640, help="inference image size")
    p.add_argument("--video", default="assets/video5.mov", help="input video")
    p.add_argument("--warmup", type=int, default=30, help="warmup frames discarded (per run)")
    p.add_argument("--frames", type=int, default=300, help="measured frames (per run)")
    p.add_argument("--runs", type=int, default=3, help="fresh model load per run")
    p.add_argument("--out", default=None, help="markdown output path")
    p.add_argument(
        "--append",
        action="store_true",
        help="append a section to existing --out instead of overwriting",
    )
    return p.parse_args()


def quantile(sorted_vals: list[float], q: float) -> float:
    """linear-interpolated quantile of an already-sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def stats_block(vals_ms: list[float]) -> dict:
    """mean / p50 / p95 / p99 / min / max + derived fps from a list of per-call ms."""
    s = sorted(vals_ms)
    p50 = quantile(s, 0.50)
    p95 = quantile(s, 0.95)
    p99 = quantile(s, 0.99)
    return {
        "mean": statistics.fmean(vals_ms),
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "min": s[0],
        "max": s[-1],
        "fps": 1000.0 / p50 if p50 > 0 else float("nan"),
    }


def fmt(ms: float) -> str:
    return f"{ms:7.2f}"


def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    info = {
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def preload_frames(path: str, n: int) -> list[np.ndarray]:
    """read the first n frames into memory once. raises if the video is too short."""
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    try:
        while len(frames) < n:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    if len(frames) < n:
        raise RuntimeError(
            f"video has only {len(frames)} frames, need at least {n} (warmup+frames)"
        )
    return frames


def load_model(model_path: str):
    """load an ultralytics YOLO from any artifact the loader understands."""
    from ultralytics import YOLO

    return YOLO(model_path)


def read_back_device(model) -> str:
    """best-effort read of the device the model actually runs on."""
    dev = getattr(model, "device", None)
    if dev is not None:
        return str(dev)
    # onnx/coreml exporters may not expose .device; fall back to predictor inference device
    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        return str(getattr(predictor, "device", "?"))
    return "unknown"


def engine_run(
    model_path: str,
    frames: list[np.ndarray],
    warmup: int,
    measure: int,
    device,
    imgsz: int,
) -> dict:
    """run a single engine-mode measurement pass.

    returns dict with per-call latency stats + averaged ultralytics stage breakdown.
    """
    model = load_model(model_path)
    track_kwargs = {
        "persist": True,
        "tracker": TRACKER_PATH,
        "conf": 0.5,
        "verbose": False,
    }
    if device is not None:
        track_kwargs["device"] = device
    if imgsz is not None:
        track_kwargs["imgsz"] = imgsz

    # warmup: discarded. NOTE: model.device only reflects the chosen device AFTER the first
    # forward pass actually runs on it, so we read dev_back after warmup, not before.
    for i in range(warmup):
        model.track(frames[i], **track_kwargs)
    dev_back = read_back_device(model)

    latencies_ms: list[float] = []
    speed_acc = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    speed_n = 0

    for i in range(measure):
        frame = frames[warmup + i]
        t0 = time.perf_counter()
        results = model.track(frame, **track_kwargs)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        try:
            spd = results[0].speed
            if spd:
                for k in speed_acc:
                    if k in spd:
                        speed_acc[k] += float(spd[k])
                speed_n += 1
        except Exception:
            pass

    st = stats_block(latencies_ms)
    if speed_n > 0:
        speed_avg = {k: v / speed_n for k, v in speed_acc.items()}
    else:
        speed_avg = {k: float("nan") for k in speed_acc}
    st["device_used"] = dev_back
    st["speed"] = speed_avg
    del model
    gc.collect()
    return st


def pipeline_run(
    model_path: str,
    video: str,
    frames: int,
    device,
    imgsz: int,
) -> dict:
    """run a single pipeline-mode (end-to-end) pass against a fresh DetectionPipeline.

    a throwaway temp db is used so the real database/ is never touched; save_images=False so
    detections/ is never touched either. read-detect breakdown is two wall-clock components
    only (no per-stage subtraction available without re-instrumenting src/).
    """
    from defect_detection.config import DetectorConfig
    from defect_detection.inspection.pipeline import DetectionPipeline
    from defect_detection.video import FrameReader

    tmpdir = tempfile.mkdtemp(prefix="latency_db_")
    try:
        cfg_kwargs = {
            "model_path": model_path,
            "conf_threshold": 0.5,
            "db_path": os.path.join(tmpdir, "bench.db"),
            "save_images": False,
            "images_dir": tmpdir,
            "tracker": TRACKER_PATH,
        }
        config = DetectorConfig(**cfg_kwargs)
        pipe = DetectionPipeline(config)
        reader = FrameReader(video, loop=False)
        pipe.start_session()

        read_ms: list[float] = []
        detect_ms: list[float] = []
        # ponytail: InferenceEngine.track accepts no device/imgsz, so pipeline mode uses
        # ultralytics' auto device (cpu on this mac) regardless of --device. read dev_back
        # after the first forward pass so model.device reflects the actually-used device.
        dev_back = None

        try:
            count = 0
            wall_t0 = time.perf_counter()
            while count < frames:
                tr0 = time.perf_counter()
                frame = reader.read()
                tr1 = time.perf_counter()
                if frame is None:
                    break
                read_ms.append((tr1 - tr0) * 1000.0)

                td0 = time.perf_counter()
                pipe.detect_frame(frame)
                td1 = time.perf_counter()
                detect_ms.append((td1 - td0) * 1000.0)
                count += 1
                if dev_back is None:
                    dev_back = read_back_device(pipe.engine.model)
            wall_t1 = time.perf_counter()
        finally:
            reader.release()
            pipe.cleanup()

        measured = count
        wall_s = wall_t1 - wall_t0
        e2e_fps = measured / wall_s if wall_s > 0 else float("nan")
        return {
            "frames_measured": measured,
            "wall_s": wall_s,
            "e2e_fps": e2e_fps,
            "read": stats_block(read_ms),
            "detect_frame": stats_block(detect_ms),
            "device_used": dev_back or "unknown",
            "speed_breakdown_accessible": False,
        }
    finally:
        # tmpdir cleanup left to gc; benchmarks are throwaway. explicit remove attempted.
        try:
            for f in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, f))
            os.rmdir(tmpdir)
        except OSError:
            pass


def context_block(model_path: str, video: str, device_arg) -> str:
    vi = video_info(video)
    dev_name = (
        "Apple Silicon"
        if platform.machine().startswith(("arm", "aarch"))
        else (platform.processor() or "?")
    )
    machine = "Apple M4 Air 16GB, fanless — sustained numbers may throttle"
    lines = [
        "## Context",
        f"- python: {sys.version.split()[0]}",
        f"- torch: {torch.__version__}",
        f"- ultralytics: {ultralytics.__version__}",
        f"- mps available: {torch.backends.mps.is_available()}",
        f"- device arg: {device_arg if device_arg is not None else 'None (ultralytics auto)'}",
        f"- model: `{model_path}`",
        f"- video: `{video}` — {vi['w']}x{vi['h']} @ {vi['fps']:.2f} fps, {vi['count']} frames",
        f"- machine: {machine}",
        f"- arch: {dev_name}",
    ]
    return "\n".join(lines) + "\n"


def engine_table_row(label: str, r: dict) -> str:
    s = r
    return (
        f"| {label} | {fmt(s['mean'])} | {fmt(s['p50'])} | {fmt(s['p95'])} | "
        f"{fmt(s['p99'])} | {fmt(s['min'])} | {fmt(s['max'])} | {s['fps']:6.1f} | "
        f"{fmt(s['speed']['preprocess'])} / {fmt(s['speed']['inference'])} / "
        f"{fmt(s['speed']['postprocess'])} | "
        f"`{s['device_used']}` |"
    )


def render_engine_section(
    title: str, device_arg, imgsz: int, runs: int, results: list[dict]
) -> str:
    out = [f"## {title}", ""]
    out.append(f"- device arg: `{device_arg if device_arg is not None else 'None (auto)'}`")
    out.append(
        f"- imgsz: {imgsz} | tracker: bytetrack.yaml | conf: 0.5 | runs: {runs} | "
        f"persist=True (sequential, no inter-frame reset)"
    )
    out.append("")
    out.append("### Engine mode — model latency (per `model.track` call, ms)")
    out.append("")
    out.append(
        "| run | mean | p50 | p95 | p99 | min | max | fps(1000/p50) | "
        "speed pre/infer/post (ms) | device |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate(results, 1):
        out.append(engine_table_row(f"run{i}", r))
    # aggregate across runs on p50
    p50s = [r["p50"] for r in results]
    agg = stats_block(p50s)
    # only per-run p50s are retained; pool-all-latencies aggregate would need raw vectors.
    spread = max(p50s) - min(p50s) if p50s else float("nan")
    out.append(f"| **agg** | | **{fmt(agg['mean'])}** | | | | | **{agg['fps']:6.1f}** | | |")
    out.append("")
    out.append(
        f"- p50 across runs: mean={fmt(agg['mean'])} ms, min={fmt(min(p50s))}, "
        f"max={fmt(max(p50s))}, spread={fmt(spread)} ms"
    )
    dev_used = results[0]["device_used"] if results else "?"
    out.append(f"- device read back: `{dev_used}`")
    return "\n".join(out) + "\n"


def render_pipeline_section(
    title: str, device_arg, imgsz: int, runs: int, results: list[dict]
) -> str:
    out = [f"## {title} (pipeline mode)", ""]
    out.append(f"- device arg: `{device_arg if device_arg is not None else 'None (auto)'}`")
    frames_per = results[0]["frames_measured"] if results else "?"
    out.append(f"- imgsz: {imgsz} | frames/run: {frames_per} | runs: {runs}")
    out.append("")
    out.append("### End-to-end pipeline FPS (read + detect_frame, wall clock)")
    out.append("")
    out.append(
        "| run | frames | wall(s) | e2e fps | read mean/p50/p95 (ms) | "
        "detect_frame mean/p50/p95 (ms) | device |"
    )
    out.append("|---|---:|---:|---:|---|---|---|")
    for i, r in enumerate(results, 1):
        rd = r["read"]
        df = r["detect_frame"]
        out.append(
            f"| run{i} | {r['frames_measured']} | {r['wall_s']:.2f} | {r['e2e_fps']:6.1f} | "
            f"{fmt(rd['mean'])}/{fmt(rd['p50'])}/{fmt(rd['p95'])} | "
            f"{fmt(df['mean'])}/{fmt(df['p50'])}/{fmt(df['p95'])} | `{r['device_used']}` |"
        )
    e2e = [r["e2e_fps"] for r in results]
    out.append(f"| **agg** | | | **{statistics.fmean(e2e):6.1f}** (mean of runs) | | | |")
    out.append("")
    out.append(
        "- per-stage ultralytics breakdown not accessible in pipeline mode without breaking "
        "DetectionPipeline.detect_frame's return contract (it wraps engine.track and discards "
        "the underlying results); see engine mode for the pre/infer/post breakdown."
    )
    if device_arg is not None:
        out.append(
            f"- WARNING: pipeline mode could NOT honor --device `{device_arg}`: "
            "InferenceEngine.track accepts no device kwarg, so the underlying YOLO runs on "
            "ultralytics' auto device "
            "(cpu on this mac). The `device` column above reflects what was actually used. "
            "--device only affects engine mode."
        )
    return "\n".join(out) + "\n"


def main() -> int:
    args = parse_args()
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = str(REPO_ROOT / model_path)
    video = args.video
    if not os.path.isabs(video):
        video = str(REPO_ROOT / video)

    print(
        f"[harness] model={model_path} device={args.device} imgsz={args.imgsz} "
        f"video={video} warmup={args.warmup} frames={args.frames} runs={args.runs}",
        flush=True,
    )

    preload_n = args.warmup + args.frames
    print(f"[harness] preloading {preload_n} frames into memory...", flush=True)
    t0 = time.perf_counter()
    frames = preload_frames(video, preload_n)
    # ponytail: in-memory preload removes decode jitter from engine timing; RAM is the cost.
    nbytes = sum(f.nbytes for f in frames)
    print(
        f"[harness] preloaded {len(frames)} frames in {time.perf_counter() - t0:.2f}s "
        f"({nbytes / 1e6:.0f} MB resident)",
        flush=True,
    )

    # ---------- engine mode ----------
    engine_results: list[dict] = []
    for r in range(args.runs):
        print(f"[harness] engine run {r + 1}/{args.runs} (fresh model load)...", flush=True)
        res = engine_run(model_path, frames, args.warmup, args.frames, args.device, args.imgsz)
        print(
            f"  device={res['device_used']} p50={res['p50']:.2f}ms fps={res['fps']:.1f} "
            f"speed={res['speed']['preprocess']:.1f}/{res['speed']['inference']:.1f}/"
            f"{res['speed']['postprocess']:.1f}",
            flush=True,
        )
        engine_results.append(res)

    del frames
    gc.collect()

    # ---------- pipeline mode ----------
    pipe_results: list[dict] = []
    for r in range(args.runs):
        print(
            f"[harness] pipeline run {r + 1}/{args.runs} (fresh pipeline + reader)...", flush=True
        )
        res = pipeline_run(model_path, video, args.frames, args.device, args.imgsz)
        print(
            f"  device={res['device_used']} frames={res['frames_measured']} "
            f"wall={res['wall_s']:.2f}s e2e_fps={res['e2e_fps']:.1f}",
            flush=True,
        )
        pipe_results.append(res)

    # ---------- markdown ----------
    label = f"pytorch {args.device or 'auto (cpu)'}"
    sec_engine = render_engine_section(label, args.device, args.imgsz, args.runs, engine_results)
    sec_pipe = render_pipeline_section(label, args.device, args.imgsz, args.runs, pipe_results)

    body = sec_engine + "\n" + sec_pipe
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.append and out_path.exists():
            existing = out_path.read_text()
            new_doc = existing.rstrip() + "\n\n---\n\n" + body
        else:
            methodology = (
                "## Methodology\n"
                f"- warmup: {args.warmup} discarded frames per run (fresh model load per run)\n"
                f"- frames: {args.frames} measured frames per run (latencies timed with "
                f"time.perf_counter around each model.track / detect_frame call)\n"
                f"- runs: {args.runs} (per-run reported + aggregate); engine frames preloaded "
                f"once into memory (~{(args.warmup + args.frames) * 1080 * 1920 * 3 / 1e9:.1f}GB) "
                f"to strip decode jitter from model timing\n"
                "- quantiles via linear interpolation on sorted per-call latencies; "
                "fps = 1000/p50\n\n"
            )
            doc = context_block(model_path, video, args.device) + methodology + "---\n\n" + body
            new_doc = doc
        out_path.write_text(new_doc)
        print(f"[harness] wrote {out_path}", flush=True)
    else:
        print(body, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
