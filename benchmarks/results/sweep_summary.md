# Sweep summary — consolidated artifacts × metrics (for docs agent)

Source files: `benchmarks/results/{latency_baseline.md, latency_sweep.md, map_sweep.md}`,
`benchmarks/models/MANIFEST.md`. All latency numbers are engine mode (`model.track`
per-call, p50/p95 across 3 runs of 300 frames). Baselines are copied verbatim from
`latency_baseline.md` (not re-measured here).

## Consolidated table

| config | size MB | mAP50 | mAP50-95 | engine p50 ms | engine p95 ms | FPS | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| pytorch .pt @640 (CPU baseline) | 18.29 | 0.9950 | 0.9502 | 70.60 | 98.40 | 14.2 | from latency_baseline.md; device `cpu` |
| pytorch .pt @640 (MPS baseline) | 18.29 | 0.9950 | 0.9502 | 31.82 | 45.43 | 31.2 | from latency_baseline.md; device `mps:0` |
| pytorch .pt @512 (CPU, val only) | 18.29 | 0.9950 | 0.9595 | — | — | — | dynamic-input; no latency sweep (out of scope) |
| pytorch .pt @384 (CPU, val only) | 18.29 | 0.9950 | 0.9686 | — | — | — | dynamic-input; no latency sweep (out of scope) |
| coreml-fp16 @640 | 18.17 | 0.9950 | 0.9567 | 16.15 | 20.86 | 62.4 | 4.4× CPU / 2.0× MPS; ultralytics picks CPU_AND_NE internally (predictor device label reads `cpu`) |
| coreml-int8 @640 | 9.25 | 0.9950 | 0.9529 | 16.10 | 19.69 | 62.8 | `quantize='w8a16'` kmeans weight-palettized; smallest artifact; +0.27pt vs pytorch 640 (within noise) |
| onnx-fp32 @640 | 36.21 | 0.9950 | 0.9551 | 94.08 | 126.44 | 10.6 | ONNXRuntime 1.28 CPUExecutionProvider; slower than .pt CPU baseline — ORT-CPU is not competitive here |

Notes:
- val n=56 ⇒ sub-~1.5pt deltas are within noise. All four @640 configs land within ±0.7pt
  of each other on mAP50-95 (0.9502–0.9567), so accuracy is not the differentiator; latency
  and artifact size are.
- CoreML/ONNX artifacts are static-shape @640 (exported with `nms=True`; CoreML rejects
  other input sizes, ONNX binds 640). Only the pytorch .pt is dynamic-input — that's why
  the {512, 384} columns are populated for pytorch only.
- self-consistency check (latency_sweep.md): for each artifact p50 ≤ p95 ≤ p99 across all
  3 runs, and fps ≈ 1000/p50 (matches to 2 sig figs).
- CoreML `device_used = cpu` is the ultralytics predictor device label, NOT the CoreML
  compute unit. Per prior-agent research, ultralytics selects `CPU_AND_NE` internally;
  requesting `.all` / `CPU_AND_GPU` crashes coremltools 9.x. No CoreML compute-unit
  fallback warnings were emitted during smoke or sweep runs.

## Candidate operating points (docs agent picks the final)

1. **Balanced — `coreml-fp16.mlpackage` @640.** 4.4× over CPU baseline (62 vs 14 FPS),
   2.0× over MPS (62 vs 31), no measurable accuracy loss (0.9567 vs 0.9502 baseline = +0.65pt,
   within noise), 18 MB artifact, native on Apple Silicon. Default for the on-device
   inspector on this M4 Air.
2. **Max speed / smallest footprint — `coreml-int8.mlpackage` @640.** Tied-fastest p50
   (16.10 ms, 62.8 FPS), half the artifact size (9.25 MB), mAP50-95 0.9529 still within
   0.4pt of FP16 (noise). Pick when the artifact ships in a constrained bundle (e.g. an
   app binary) and FP16's 18 MB matters.
3. **Max accuracy — `best.pt` @384 on CPU/MPS.** mAP50-95 0.9686 (+1.84pt vs 640
   baseline, just above the noise floor) — but no latency sweep was run at 384 and
   inference cost rises on smaller-host GPU. Pick only if a future accuracy budget
   forces it; otherwise the FP16 CoreML point is accuracy-equivalent within noise and
   far faster.
