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

Export + latency/stability sweep on Apple M4 Air 16GB. Default operating point is **CoreML fp16 @640** (62.4 FPS, 4.4× over the CPU baseline, no measurable accuracy loss); int8 CoreML halves artifact size at tied latency when a bundle matters. Per-track class labels are stabilized by a zone-gated `TrackLabelStabilizer` (commit-once on confidence-weighted votes restricted to a centerline evidence band) — on `assets/video5.mov` it cut near-crossing class switches 6 → 0 with identical 5/5 crossing correctness. Full methodology, negative results (ONNX-CPU is slower than .pt CPU), calibration caveats, and reproduction commands: [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

| config | size MB | mAP50-95 | engine p50 ms | FPS |
|---|---:|---:|---:|---:|
| pytorch .pt @640 (CPU) | 18.29 | 0.9502 | 70.60 | 14.2 |
| pytorch .pt @640 (MPS) | 18.29 | 0.9502 | 31.82 | 31.2 |
| coreml-fp16 @640 | 18.17 | 0.9567 | 16.15 | 62.4 |
| coreml-int8 @640 | 9.25 | 0.9529 | 16.10 | 62.8 |
| onnx-fp32 @640 | 36.21 | 0.9551 | 94.08 | 10.6 |

Stability replay (video5): flickering tracks 13/24 → 10/24, class switches 56 → 31, switches near a crossing 6 → 0, crossings 5/5 unchanged.

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
