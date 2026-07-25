## Context
- python: 3.11.15
- torch: 2.13.0
- ultralytics: 8.4.105
- mps available: True
- device arg: None (ultralytics auto)
- model: `/Users/lindsaycheng/Documents/personal/projects/defect-detection/model/weights/best.pt`
- video: `/Users/lindsaycheng/Documents/personal/projects/defect-detection/assets/video5.mov` — 1080x1920 @ 59.94 fps, 924 frames
- machine: Apple M4 Air 16GB, fanless — sustained numbers may throttle
- arch: Apple Silicon
## Methodology
- warmup: 30 discarded frames per run (fresh model load per run)
- frames: 300 measured frames per run (latencies timed with time.perf_counter around each model.track / detect_frame call)
- runs: 3 (per-run reported + aggregate); engine frames preloaded once into memory (~2.1GB) to strip decode jitter from model timing
- quantiles via linear interpolation on sorted per-call latencies; fps = 1000/p50

---

## pytorch auto (cpu)

- device arg: `None (auto)`
- imgsz: 640 | tracker: bytetrack.yaml | conf: 0.5 | runs: 3 | persist=True (sequential, no inter-frame reset)

### Engine mode — model latency (per `model.track` call, ms)

| run | mean | p50 | p95 | p99 | min | max | fps(1000/p50) | speed pre/infer/post (ms) | device |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| run1 |   75.92 |   70.22 |  107.96 |  158.20 |   60.35 |  308.19 |   14.2 |    3.69 /   71.01 /    0.44 | `cpu` |
| run2 |   73.01 |   70.10 |   89.00 |  117.02 |   62.60 |  138.25 |   14.3 |    3.69 /   68.14 /    0.42 | `cpu` |
| run3 |   75.22 |   71.48 |   98.23 |  146.97 |   61.74 |  224.26 |   14.0 |    3.91 /   70.09 /    0.44 | `cpu` |
| **agg** | | **  70.60** | | | | | **  14.2** | | |

- p50 across runs: mean=  70.60 ms, min=  70.10, max=  71.48, spread=   1.38 ms
- device read back: `cpu`

## pytorch auto (cpu) (pipeline mode)

- device arg: `None (auto)`
- imgsz: 640 | frames/run: 300 | runs: 3

### End-to-end pipeline FPS (read + detect_frame, wall clock)

| run | frames | wall(s) | e2e fps | read mean/p50/p95 (ms) | detect_frame mean/p50/p95 (ms) | device |
|---|---:|---:|---:|---|---|---|
| run1 | 300 | 27.16 |   11.0 |    4.27/   3.90/   6.59 |   86.25/  74.92/ 142.79 | `cpu` |
| run2 | 300 | 31.35 |    9.6 |    4.80/   4.44/   6.94 |   99.68/  93.57/ 132.69 | `cpu` |
| run3 | 300 | 27.18 |   11.0 |    4.20/   4.04/   5.31 |   86.39/  82.65/ 112.92 | `cpu` |
| **agg** | | | **  10.6** (mean of runs) | | | |

- per-stage ultralytics breakdown not accessible in pipeline mode without breaking DetectionPipeline.detect_frame's return contract (it wraps engine.track and discards the underlying results); see engine mode for the pre/infer/post breakdown.

---

## pytorch mps

- device arg: `mps`
- imgsz: 640 | tracker: bytetrack.yaml | conf: 0.5 | runs: 3 | persist=True (sequential, no inter-frame reset)

### Engine mode — model latency (per `model.track` call, ms)

| run | mean | p50 | p95 | p99 | min | max | fps(1000/p50) | speed pre/infer/post (ms) | device |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| run1 |   31.31 |   29.58 |   41.63 |   50.30 |   21.10 |  147.63 |   33.8 |    3.91 /   18.76 /    7.44 | `mps:0` |
| run2 |   32.53 |   32.01 |   44.44 |   47.96 |   21.19 |   61.96 |   31.2 |    4.32 /   19.77 /    7.20 | `mps:0` |
| run3 |   35.77 |   33.87 |   50.22 |   60.45 |   21.05 |  235.84 |   29.5 |    4.47 /   21.66 /    8.20 | `mps:0` |
| **agg** | | **  31.82** | | | | | **  31.2** | | |

- p50 across runs: mean=  31.82 ms, min=  29.58, max=  33.87, spread=   4.30 ms
- device read back: `mps:0`

## pytorch mps (pipeline mode)

- device arg: `mps`
- imgsz: 640 | frames/run: 300 | runs: 3

### End-to-end pipeline FPS (read + detect_frame, wall clock)

| run | frames | wall(s) | e2e fps | read mean/p50/p95 (ms) | detect_frame mean/p50/p95 (ms) | device |
|---|---:|---:|---:|---|---|---|
| run1 | 300 | 24.63 |   12.2 |    3.82/   3.80/   4.40 |   78.26/  76.84/  88.49 | `cpu` |
| run2 | 300 | 23.99 |   12.5 |    3.74/   3.35/   5.26 |   76.23/  66.21/ 108.29 | `cpu` |
| run3 | 300 | 23.00 |   13.0 |    3.70/   3.30/   5.12 |   72.95/  62.96/ 134.40 | `cpu` |
| **agg** | | | **  12.6** (mean of runs) | | | |

- per-stage ultralytics breakdown not accessible in pipeline mode without breaking DetectionPipeline.detect_frame's return contract (it wraps engine.track and discards the underlying results); see engine mode for the pre/infer/post breakdown.
- WARNING: pipeline mode could NOT honor --device `mps`: InferenceEngine.track accepts no device kwarg, so the underlying YOLO runs on ultralytics' auto device (cpu on this mac). The `device` column above reflects what was actually used. --device only affects engine mode.
