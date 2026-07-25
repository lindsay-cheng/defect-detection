# Real-Time Bottle Defect Detection System

Automated visual inspection system for identifying defects in Kirkland plastic bottles using YOLO-based computer vision.

## Overview

This system uses YOLOv11s object detection and Bytetrack object tracking to automatically detect and classify defective bottles in real-time. Designed for quality control in manufacturing line environments.

[![Demo video](assets/thumbnail.png)](https://www.youtube.com/watch?v=Nh-hHzurpris)
*Click to Watch Demo*


## Project structure

```
src/defect_detection/
  config.py              # DetectorConfig dataclass + yaml loader
  constants.py           # shared constants and helpers (display ids, db keys)
  video.py               # FrameReader: cv2.VideoCapture loop/restart helper
  inference/engine.py    # InferenceEngine: YOLO load + ByteTrack tracking
  inspection/inspector.py # pure logic: centerline counting, display ids, session state
  inspection/pipeline.py  # DetectionPipeline: composes engine + inspector + database
  inspection/annotator.py # annotate_frame function (boxes, labels, centerline)
  storage/database.py    # thread-safe sqlite worker (DefectDatabase + core)
  cli/detect.py          # `defect-detect` entry point
  cli/db_utils.py        # `defect-db` entry point (export/stats/clear)
  gui/app.py             # `defect-gui` tkinter app (DefectDetectionApp)
```

`inference/engine` loads the YOLO model and runs tracking; `inspection/inspector` holds the pure centerline/counting/session logic; `inspection/pipeline` composes the engine, inspector, and database and owns side effects (db writes, defect crops, frame annotation); `storage/database` is the thread-safe sqlite worker; `cli` and `gui` provide the entry points.

## Training (image collection)

- images were collected by me and annotated in YOLO format using Roboflow
- dataset config: `dataset/data.yaml`
- classes are defined by the trained model and `dataset/data.yaml`
- trained on external Google Colab for GPU access

## Defect classes

- `good`
- `low_water`
- `no_cap`
- `no_label`

## Preliminary Results (Controlled Environment)

The custom YOLO11s model achieves 97.7% mAP@0.5 (93.1% mAP@0.5-0.95) on an 80/20 stratified train/val split. Dataset consists of ~300 images captured with varied camera angles, zoom levels, lighting, and positions. While these metrics demonstrate the model's capability to learn defect patterns, the small dataset size and single-environment capture may limit generalization.

| Class | Precision | Recall | mAP@0.5 | mAP@0.5-0.95 |
|-------|-----------|--------|---------|--------------|
| all | 0.977 | 0.969 | 0.977 | 0.931 |
| good | 0.997 | 0.952 | 0.993 | 0.943 |
| low_water | 0.995 | 0.923 | 0.956 | 0.891 |
| no_cap | 0.996 | 1.000 | 0.995 | 0.967 |
| no_label | 0.918 | 1.000 | 0.963 | 0.925 |

![Training Results](model/results.png)
*Training metrics over 150 epochs*

**Important Limitations:**
- Small dataset size increases risk of overfitting
- Single capture environment may not generalize to diverse production settings
- Model performance on real-world manufacturing data remains to be validated

**WIP:**
- Expand dataset across multiple environments and bottle types
- K-fold cross-validation to better assess model robustness

## Optimization

Two measured problems were addressed in one pass: the pipeline ran at ~10 FPS, and per-frame class labels flickered for the same tracked bottle, so the defect type written to the database was whichever class the model happened to emit on the single frame the bottle crossed the counting line.

Both fixes were measured before and after on a fixed input (`assets/video5.mov`, 924 frames) on an Apple M4 Air 16GB. Methodology, per-run tables, and exact reproduction commands are in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

### Headline

| | before | after |
|---|---:|---:|
| End-to-end pipeline | 10.6 FPS | **40.3 FPS** (3.8×) |
| Model call (engine p50) | 70.60 ms | **16.15 ms** (4.4×) |
| Class switches near a crossing | 6 | **0** |
| Crossings logged correctly vs ground truth | 5/5 | 5/5 |
| mAP50-95 on the local val split | 0.9502 | 0.9567 (within noise) |

### 1. Runtime — CoreML export

The model was exported to CoreML (fp16 and 8-bit weight-palettized) and ONNX, then benchmarked against the PyTorch baseline on CPU and MPS. 30 warmup frames, 300 measured frames, 3 runs per configuration, quantiles over per-call latency.

| config | size MB | mAP50 | mAP50-95 | engine p50 ms | engine p95 ms | FPS |
|---|---:|---:|---:|---:|---:|---:|
| pytorch `.pt` @640 (CPU) — baseline | 18.29 | 0.9950 | 0.9502 | 70.60 | 98.40 | 14.2 |
| pytorch `.pt` @640 (MPS) | 18.29 | 0.9950 | 0.9502 | 31.82 | 45.43 | 31.2 |
| **coreml-fp16 @640** — recommended | 18.17 | 0.9950 | 0.9567 | 16.15 | 20.86 | **62.4** |
| coreml-int8 @640 (`w8a16`) | 9.25 | 0.9950 | 0.9529 | 16.10 | 19.69 | 62.8 |
| onnx-fp32 @640 | 36.21 | 0.9950 | 0.9551 | 94.08 | 126.44 | 10.6 |

The table above is the model call in isolation. End-to-end through the real pipeline — video decode, tracking, centerline logic, database writes, annotation — CoreML fp16 runs at **40.3 FPS against a 10.6 FPS PyTorch-CPU baseline**, both 3-run aggregates over 300 frames.

**CoreML fp16 @640 is the recommended operating point**: 4.4× the CPU baseline, 2.0× MPS, no measurable accuracy change, 18 MB. The int8 artifact ties it on latency at half the size, which is the better pick when bundle size matters. Accuracy is *not* the differentiator here — all four @640 configurations land within 0.7 points of each other on mAP50-95, which is inside the noise floor of a 56-image validation set.

**Negative result, published as such:** ONNX Runtime's CPU provider is *slower* than the PyTorch CPU baseline it was meant to beat (94 ms vs 71 ms). On this host CoreML is the win; "export to ONNX" is not.

### 2. Correctness — zone-gated track label stabilization

No tracker stabilizes class labels: ByteTrack associates detections, and the class is whatever the detector emitted on that frame. Replaying the baseline over `video5` confirmed the instability — 13 of 24 tracks changed class during their lifetime, 56 switches in total, 6 of them within ±10 frames of a counting decision.

`TrackLabelStabilizer` accumulates confidence-weighted class votes per track and commits a label once, permanently (the pattern NVIDIA DeepStream uses for its ID-keyed classifier cache). The critical detail is **which frames are allowed to vote**: only detections whose centroid falls inside an evidence band bracketing the centerline (±15% of frame width). Approach and exit frames — where the bottle is small, off-center, or partly occluded, and the model is systematically biased toward `good` — are excluded by design.

An earlier version that voted over the whole track history is documented in `benchmarks/RESULTS.md` as a failure: it crushed the flicker metrics but committed two defective bottles to `good` before they ever reached the line, silently missing them. The gate is a structural fix, not a tuning knob.

| metric (`video5`, same model, conf 0.5) | baseline | stabilized |
|---|---:|---:|
| crossings detected / expected | 5 / 5 | 5 / 5 |
| correct / missed / false-positive / wrong-type | 5 / 0 / 0 / 0 | 5 / 0 / 0 / 0 |
| tracks showing more than one class | 13 / 24 | 10 / 24 |
| total class switches | 56 | 31 |
| switches within ±10 frames of a crossing | 6 | **0** |

The residual 31 switches all sit outside the decision zone: still visible in the annotation, structurally unable to reach a database row.

### Using it

Exported artifacts are generated, not committed (`benchmarks/models/` is gitignored). Build them with `python benchmarks/export_models.py`, then point the pipeline at one:

```bash
defect-detect --model benchmarks/models/best_fp16.mlpackage --source 0
```

The default stays `model/weights/best.pt`. `DetectorConfig` exposes `model_path`, `device`, and `imgsz`, settable in code or via `DetectorConfig.from_yaml`.

### Limitations

- The validation split is 56 images, so accuracy differences under roughly 1.5 points are noise and are reported as such rather than as gains.
- The stabilizer's three constants are calibrated on one video. A second labelled clip with different camera geometry is the next measurement that matters.
- Latency is burst-after-warmup on a fanless machine; sustained load will throttle and p50 will rise.
- The int8 artifact is weight-only palettization with fp16 activations, not a fully quantized network — full INT8 crashes the current coremltools/torch combination, and ultralytics 8.4.105 offers no activation-calibration path for CoreML.

## Setup

Requires Python >= 3.11.

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## How to run it

Tkinter GUI:

```bash
defect-gui        # or: python app.py
```

CLI:

```bash
defect-detect --model model/weights/best.pt --source 0
```

`--model` accepts any ultralytics-loadable artifact (`.pt`, `.mlpackage`, `.onnx`); for CoreML/ONNX exports see `benchmarks/RESULTS.md`. `DetectorConfig` now exposes `device` and `imgsz` fields (or set `model_path`/`device`/`imgsz` in a config yaml via `DetectorConfig.from_yaml`).

## How to log data

Logging is automatic:

- sqlite database: `database/defects.db`
- defect image crops: `detections/`

Export CSV:

```bash
defect-db export   # stats / clear likewise
```

## Weights

- trained weights: `model/weights/best.pt` (and `last.pt`)

## Roadmap

- Interval inference (detect every Nth frame, tracker-held boxes between), VideoToolbox hardware decode, ReID trackers, and a second-video validation set are deferred — see `benchmarks/RESULTS.md` §Future work.
