# Latency sweep — exported artifacts at imgsz=640

# ponytail: bench latency.py is reused unmodified; its section title template
# `pytorch auto (cpu)` is hard-coded (it does not know about coreml/onnx artifacts),
# so section headings below were post-edited to the actual artifact name. the body
# tables and device-read-back columns are exactly what latency.py emitted.

## Context (pyproject harness)
- python: 3.11.15
- torch: 2.13.0
- ultralytics: 8.4.105
- mps available: True
- device arg: None (ultralytics auto)
- artifact paths: `benchmarks/models/{best_fp16.mlpackage, best_int8.mlpackage, best_fp32.onnx}`
- video: `assets/video5.mov` — 1080x1920 @ 59.94 fps, 924 frames
- machine: Apple M4 Air 16GB, fanless — sustained numbers may throttle
- arch: Apple Silicon
- note: CoreML `device_used = cpu` is the predictor-side label, not the CoreML
  compute unit; ultralytics selects `CPU_AND_NE` internally (requesting `.all`
  / `CPU_AND_GPU` crashes coremltools 9.x). ONNX uses CPUExecutionProvider.

## Methodology
- warmup: 30 discarded frames per run (fresh model load per run)
- frames: 300 measured frames per run (latencies timed with time.perf_counter around each model.track / detect_frame call)
- runs: 3 (per-run reported + aggregate); engine frames preloaded once into memory (~2.1GB) to strip decode jitter from model timing
- quantiles via linear interpolation on sorted per-call latencies; fps = 1000/p50
- artifacts are fixed-shape @640 (CoreML rejects other sizes; ONNX binds 640). Only imgsz=640 is valid for them.

---

## coreml-fp16 (best_fp16.mlpackage) — engine mode

- device arg: `None (auto)`
- imgsz: 640 | tracker: bytetrack.yaml | conf: 0.5 | runs: 3 | persist=True (sequential, no inter-frame reset)

### Engine mode — model latency (per `model.track` call, ms)

| run | mean | p50 | p95 | p99 | min | max | fps(1000/p50) | speed pre/infer/post (ms) | device |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| run1 |   17.34 |   16.57 |   19.79 |   30.99 |   14.88 |   57.66 |   60.3 |    3.04 /   13.62 /    0.14 | `cpu` |
| run2 |   16.18 |   15.84 |   19.24 |   21.63 |   14.56 |   25.33 |   63.1 |    2.02 /   13.44 /    0.14 | `cpu` |
| run3 |   16.79 |   16.04 |   23.54 |   28.84 |   14.22 |   33.43 |   62.4 |    2.79 /   13.34 /    0.13 | `cpu` |
| **agg** | | **  16.15** | | | | | **  62.4** | | |

- p50 across runs: mean=  16.15 ms, min=  15.84, max=  16.57, spread=   0.73 ms
- device read back: `cpu`

## coreml-fp16 (best_fp16.mlpackage) — pipeline mode

- device arg: `None (auto)`
- imgsz: 640 | frames/run: 300 | runs: 3

### End-to-end pipeline FPS (read + detect_frame, wall clock)

| run | frames | wall(s) | e2e fps | read mean/p50/p95 (ms) | detect_frame mean/p50/p95 (ms) | device |
|---|---:|---:|---:|---|---|---|
| run1 | 300 | 7.73 |   38.8 |    3.67/   2.94/   4.46 |   22.11/  15.43/  21.66 | `cpu` |
| run2 | 300 | 7.05 |   42.5 |    2.95/   2.85/   3.25 |   20.55/  15.02/  16.05 | `cpu` |
| run3 | 300 | 7.58 |   39.6 |    3.42/   2.94/   6.00 |   21.84/  15.30/  22.59 | `cpu` |
| **agg** | | | **  40.3** (mean of runs) | | | |

- per-stage ultralytics breakdown not accessible in pipeline mode without breaking DetectionPipeline.detect_frame's return contract (it wraps engine.track and discards the underlying results); see engine mode for the pre/infer/post breakdown.

---

## coreml-int8 (best_int8.mlpackage) — engine mode

- device arg: `None (auto)`
- imgsz: 640 | tracker: bytetrack.yaml | conf: 0.5 | runs: 3 | persist=True (sequential, no inter-frame reset)

### Engine mode — model latency (per `model.track` call, ms)

| run | mean | p50 | p95 | p99 | min | max | fps(1000/p50) | speed pre/infer/post (ms) | device |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| run1 |   17.13 |   16.47 |   21.17 |   23.68 |   14.85 |   27.51 |   60.7 |    3.15 /   13.32 /    0.13 | `cpu` |
| run2 |   16.35 |   15.91 |   18.57 |   23.21 |   14.99 |   28.40 |   62.9 |    2.29 /   13.35 /    0.14 | `cpu` |
| run3 |   16.81 |   15.92 |   19.34 |   38.21 |   14.99 |   74.15 |   62.8 |    2.61 /   13.45 /    0.19 | `cpu` |
| **agg** | | **  16.10** | | | | | **  62.8** | | |

- p50 across runs: mean=  16.10 ms, min=  15.91, max=  16.47, spread=   0.57 ms
- device read back: `cpu`

## coreml-int8 (best_int8.mlpackage) — pipeline mode

- device arg: `None (auto)`
- imgsz: 640 | frames/run: 300 | runs: 3

### End-to-end pipeline FPS (read + detect_frame, wall clock)

| run | frames | wall(s) | e2e fps | read mean/p50/p95 (ms) | detect_frame mean/p50/p95 (ms) | device |
|---|---:|---:|---:|---|---|---|
| run1 | 300 | 7.16 |   41.9 |    2.91/   2.83/   3.19 |   20.97/  14.73/  15.53 | `cpu` |
| run2 | 300 | 7.20 |   41.6 |    2.98/   2.85/   3.25 |   21.03/  14.79/  16.28 | `cpu` |
| run3 | 300 | 9.68 |   31.0 |    5.67/   3.53/  11.55 |   26.58/  17.58/  33.40 | `cpu` |
| **agg** | | | **  38.2** (mean of runs) | | | |

- per-stage ultralytics breakdown not accessible in pipeline mode without breaking DetectionPipeline.detect_frame's return contract (it wraps engine.track and discards the underlying results); see engine mode for the pre/infer/post breakdown.

---

## onnx-fp32 (best_fp32.onnx) — engine mode

- device arg: `None (auto)`
- imgsz: 640 | tracker: bytetrack.yaml | conf: 0.5 | runs: 3 | persist=True (sequential, no inter-frame reset)

### Engine mode — model latency (per `model.track` call, ms)

| run | mean | p50 | p95 | p99 | min | max | fps(1000/p50) | speed pre/infer/post (ms) | device |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| run1 |   91.44 |   87.93 |  112.02 |  139.66 |   77.03 |  172.47 |   11.4 |    4.16 /   86.18 /    0.21 | `cpu` |
| run2 |   98.16 |   92.28 |  122.79 |  179.86 |   77.06 |  258.05 |   10.8 |    4.64 /   92.36 /    0.23 | `cpu` |
| run3 |  106.32 |  102.02 |  144.51 |  194.78 |   79.97 |  266.32 |    9.8 |    4.97 /  100.19 /    0.23 | `cpu` |
| **agg** | | **  94.08** | | | | | **  10.8** | | |

- p50 across runs: mean=  94.08 ms, min=  87.93, max= 102.02, spread=  14.09 ms
- device read back: `cpu`

## onnx-fp32 (best_fp32.onnx) — pipeline mode

- device arg: `None (auto)`
- imgsz: 640 | frames/run: 300 | runs: 3

### End-to-end pipeline FPS (read + detect_frame, wall clock)

| run | frames | wall(s) | e2e fps | read mean/p50/p95 (ms) | detect_frame mean/p50/p95 (ms) | device |
|---|---:|---:|---:|---|---|---|
| run1 | 300 | 29.19 |   10.3 |    6.87/   6.54/   8.50 |   90.44/  88.16/ 105.30 | `cpu` |
| run2 | 300 | 32.02 |    9.4 |    7.14/   6.78/  10.01 |   99.58/  97.18/ 120.58 | `cpu` |
| run3 | 300 | 32.40 |    9.3 |    7.21/   6.73/  10.47 |  100.78/  96.90/ 121.44 | `cpu` |
| **agg** | | | **   9.6** (mean of runs) | | | |

- per-stage ultralytics breakdown not accessible in pipeline mode without breaking DetectionPipeline.detect_frame's return contract (it wraps engine.track and discards the underlying results); see engine mode for the pre/infer/post breakdown.
