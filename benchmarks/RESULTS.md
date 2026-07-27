# Optimization Pass — Results

Central evidence artifact for the CoreML/ONNX export + latency/stability sweep. Every number below is traceable to a file under `benchmarks/results/`, `benchmarks/models/MANIFEST.md`, or `src/defect_detection/` (no re-measurement performed here). Hierarchy: this file is the synthesis; `sweep_summary.md` is the prior-agent consolidation; the per-harness `.md` files are raw emission.

## Context & methodology

| | |
|---|---|
| Host | Apple M4 Air 16GB, fanless — sustained load may throttle (thermal settling) |
| Python | 3.11.15 |
| torch | 2.13.0 |
| ultralytics | 8.4.105 |
| numpy | 2.3.5 (downgraded from 2.4.6 — coremltools 9.0 compat; see `benchmarks/export_models.py:63`) |
| coremltools | 9.0 |
| onnxruntime | 1.28 (CPUExecutionProvider) |
| Source weights | `model/weights/best.pt` (18.29 MB, dynamic input) |
| Video probe | `assets/video5.mov` — 1080x1920 @ 59.94 fps, 924 frames |

Measurement protocol (`benchmarks/latency.py`):

- **Warmup**: 30 discarded frames per run; fresh model load per run.
- **Measured frames**: 300 per run; engine frames preloaded once into RAM (~2.1 GB) so decode jitter is excluded from the model-latency probe (a model-latency harness, not a decode probe).
- **Runs**: 3 per config; per-run reported + aggregate.
- **Quantiles**: linear interpolation on sorted per-call latencies; `fps = 1000/p50`.
- **Per-stage source**: engine mode reads `results[0].speed` (ultralytics-reported pre/infer/post, ms) accumulated across measured calls; pipeline mode reports only two wall-clock components (read + `detect_frame`) — the per-stage ultralytics breakdown is not accessible without breaking `DetectionPipeline.detect_frame`'s return contract (it wraps `engine.track` and discards results).
- **Engine vs pipeline mode**: engine = `model.track` per-call (stdlib `time.perf_counter` wrapper); pipeline = `FrameReader.read` + `DetectionPipeline.detect_frame` wall clock over a fresh `DetectionPipeline` with a throwaway temp DB.
- **Device read-back**: `model.device` after warmup (only meaningful post first forward pass); CoreML `device_used = cpu` is the ultralytics predictor-side label, NOT the CoreML compute unit (ultralytics selects `CPU_AND_NE` internally; requesting `.all`/`CPU_AND_GPU` crashes coremltools 9.x).

## Latency baseline — pytorch `.pt`

Engine mode (`model.track` per-call, ms), aggregate p50 across 3 runs. Source: `benchmarks/results/latency_baseline.md`.

| config | device | engine p50 ms | engine p95 ms | engine FPS | e2e pipeline FPS | speed pre/infer/post (ms, run1) |
|---|---|---:|---:|---:|---:|---|
| pytorch .pt @640 (CPU) | `cpu` | 70.60 | 98.40 | 14.2 | 10.6 | 3.69 / 71.01 / 0.44 |
| pytorch .pt @640 (MPS) | `mps:0` | 31.82 | 45.43 | 31.2 | 12.6 | 4.32 / 19.77 / 7.20 |

- Per-run p50 spread: CPU 1.38 ms, MPS 4.30 ms (fanless Air).
- Local val (regenerated 80/20 split, seed 42; `benchmarks/make_split.py`): mAP50 0.9950, mAP50-95 0.9502 — see `benchmarks/results/map_sweep.md`.

## Stability problem & fix

### Root cause

`assets/video5.mov` replay (`benchmarks/results/replay_baseline.md`) reconstructed 5/5 crossings all-correct (a point estimate on n=5 — Wilson 95% CI [0.566, 1.0], indicative not proven), but per-track class labels flickered badly: 13/24 tracks showed >1 class, 56 class switches total, and **6 switches occurred within ±10 frames of a crossing**. The bias is structural, not random:

- Approach frames (small, occluded, off-center bottles) systematically misclassify toward `good` — e.g. track 9 opens `good(x11) -> no_label(x8) -> good(x7) -> ...` before settling; track 15 opens `good(x23) -> low_water(x172) -> good(x2) -> low_water(x9) -> ...`.
- The crossing frame itself (bottle fully visible at the centerline) is the most reliable single observation.

Reading the raw per-frame class at the decision instant therefore risks committing a transient approach-frame label. The fix is to gate evidence by the same decision zone the system already uses for counting: accept votes only from detections whose centroid falls inside a band bracketing the centerline.

### Zone-gated `TrackLabelStabilizer`

`src/defect_detection/inspection/track_labels.py:25` — confidence-weighted per-track class label with one-shot commitment. Knobs: `commit_score=3.0`, `min_frames=2`, `zone_frac=0.15` (accept a vote only when `|cx - mid_x| <= 0.15 * frame_width`). Once cumulative confidence for the argmax class crosses `commit_score` (after `>= min_frames` in-zone observations), the label freezes for that track forever (DeepStream-style id-keyed cache). `raw_defect_type` preserves the instantaneous model class for diagnostics; all downstream logic (`Inspector._log_detections`, annotation, DB) reads the stabilized `defect_type`. Reset on `Inspector.start_session` / `reset_tracks`. Composition: `Inspector._labels = TrackLabelStabilizer()` (`src/defect_detection/inspection/inspector.py:38`).

### Before / after replay (video5; same model, same conf=0.5)

Source: `benchmarks/results/{replay_baseline,replay_stabilized}.md`.

| metric | baseline | stabilized |
|---|---:|---:|
| crossings detected / expected | 5 / 5 | 5 / 5 |
| verdict (OK-good / OK-defect / MISS / FALSE-POS / WRONG-TYPE) | 2 / 3 / 0 / 0 / 0 | 2 / 3 / 0 / 0 / 0 |
| tracks with >1 class | 13 / 24 (54.2%) | 10 / 24 (41.7%) |
| total class switches | 56 | 31 |
| switches within ±10 frames of crossing | 6 | 0 |

Crossing correctness is identical (the baseline already nailed the crossings because the crossing frame happened to be reliable); the win is near-crossing flicker elimination (6 → 0) and a 45% reduction in switches. The 5/5 crossing correctness is a point estimate on n=5 (Wilson 95% CI [0.566, 1.0] — indicative, not proven); the load-bearing measurement is the 6→0 near-crossing switch elimination. The residual 31 switches all sit outside the decision zone — they no longer leak into committed labels.

### The failed ungated attempt (why the zone gate matters)

An earlier version committed on confidence-weighted votes accumulated across **all** frames (no `zone_frac` gate). It accepted approach-frame evidence, which on this video is systematically biased toward `good`, and produced 2 MISSes (defective tracks committed to `good` and were never re-loggable). The zone gate is the structural fix, not merely a tuning knob: evidence quality is spatially correlated with the decision zone, so off-center votes are excluded by design. This is the lesson worth citing — the same commit-once mechanism is correct or destructive depending on *which* frames feed it.

## Model-format sweep (consolidated)

Source: `benchmarks/results/sweep_summary.md` (which consolidates `latency_baseline.md`, `latency_sweep.md`, `map_sweep.md`). Engine p50 across 3 runs of 300 frames; mAP on the regenerated val split (n=56).

| config | size MB | mAP50 | mAP50-95 | engine p50 ms | engine p95 ms | FPS | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| pytorch .pt @640 (CPU) | 18.29 | 0.9950 | 0.9502 | 70.60 | 98.40 | 14.2 | baseline |
| pytorch .pt @640 (MPS) | 18.29 | 0.9950 | 0.9502 | 31.82 | 45.43 | 31.2 | device `mps:0` |
| pytorch .pt @512 (val only) | 18.29 | 0.9950 | 0.9595 | — | — | — | .pt is dynamic-input; no latency sweep at 512 |
| pytorch .pt @384 (val only) | 18.29 | 0.9950 | 0.9686 | — | — | — | .pt is dynamic-input; no latency sweep at 384 |
| **coreml-fp16 @640** | 18.17 | 0.9950 | 0.9567 | 16.15 | 20.86 | 62.4 | 4.4× CPU / 2.0× MPS |
| **coreml-int8 @640** | 9.25 | 0.9950 | 0.9529 | 16.10 | 19.69 | 62.8 | `quantize='w8a16'` kmeans weight-palettized (weight-only) |
| onnx-fp32 @640 | 36.21 | 0.9950 | 0.9551 | 94.08 | 126.44 | 10.6 | ORT-CPU slower than .pt CPU baseline |

End-to-end pipeline mode (`FrameReader.read` + `DetectionPipeline.detect_frame` wall clock — decode, track, centerline logic, DB write, annotate), 3-run aggregate over 300 frames. Source: `benchmarks/results/{latency_baseline,latency_sweep}.md`.

| config | e2e FPS (3-run agg) | vs baseline |
|---|---:|---:|
| pytorch .pt @640 (CPU) | 10.6 | baseline |
| coreml-fp16 @640 | **40.3** | 3.8× |
| coreml-int8 @640 | 38.2 | 3.6× |

- The e2e figure is the number an operator experiences; the engine p50 above is the model call in isolation. A single confirmation run of coreml-fp16 pipeline mode reached 42.3 FPS — quote the 3-run aggregate (40.3), not that run, since the baseline is also a 3-run aggregate.
- All four @640 configs land within ±0.7 pt of each other on mAP50-95 (0.9502–0.9567) — accuracy is **not** the differentiator at n=56; latency and artifact size are.
- Fixed-shape exports (CoreML/ONNX with `nms=True`) reject non-640 input sizes; only the `.pt` is dynamic — that is why the {512, 384} val rows are populated for pytorch only.
- Self-consistency: per artifact p50 ≤ p95 ≤ p99 across all 3 runs; fps ≈ 1000/p50 (2 sig figs).

## Real application path

A fresh `runs=3` measurement of the CoreML fp16 artifact in pipeline mode (this report's data — `benchmarks/latency.py --model benchmarks/models/best_fp16.mlpackage --runs 3`, 30 warmup + 300 measured frames per run, fanless Air):

| metric (coreml-fp16 @640, `runs=3`) | this run | prior 3-run aggregate |
|---|---:|---:|
| engine p50 (ms, mean of runs) | 17.88 | 16.15 |
| e2e pipeline FPS (mean of runs) | 40.2 | 40.3 |

- e2e FPS **40.2 vs the 10.6 FPS CPU `.pt` baseline** → 3.8× on the real read + `detect_frame` path. The earlier **42.3 FPS** figure was a single confirmation run, not a 3-run aggregate — quote 40.2/40.3 (both 3-run aggregates), not 42.3.
- The engine p50 here (17.88 ms) runs slightly hotter than the prior 16.15 ms; this is a one-shot run without dedicated thermal settling on a fanless machine, so treat the ~1.7 ms delta as noise, not regression.
- **The Tkinter GUI polls frames every 30 ms** (`DefectDetectionApp._poll_frames`, `src/defect_detection/gui/app.py:195`) → the on-screen display is capped at ~33 FPS even when inference is faster. At ~40 FPS pipeline throughput the GUI poll interval is the binding constraint, not the model; engine p50 ≈ 18 ms alone leaves ~55 FPS of inference headroom the UI cannot consume without a faster poll.
- **The app default now auto-selects the CoreML fp16 artifact** (V1): `DetectorConfig.model_path = "auto"` and `resolve_model_path` (`src/defect_detection/config.py`) pick `benchmarks/models/best_fp16.mlpackage` when present and fall back to `model/weights/best.pt` with a warning (`# ponytail:`-documented; upgrade = a config flag forcing one path). Build the artifact with `python benchmarks/export_models.py`. Prior phases measured the artifact but the running app never exercised it (default stayed `best.pt`); this phase wires the default to the measured win.

## Operating point recommendation

- **Default — `coreml-fp16.mlpackage` @640.** 4.4× over CPU baseline (62 vs 14 FPS), 2.0× over MPS (62 vs 31), no measurable accuracy loss (0.9567 vs 0.9502 = +0.65 pt, within noise), 18 MB artifact, native on Apple Silicon.
- **Alt when artifact size matters — `coreml-int8.mlpackage` @640.** Tied-fastest p50 (16.10 ms, 62.8 FPS), half the artifact size (9.25 MB), mAP50-95 0.9529 still within 0.4 pt of fp16 (noise). Pick for a size-constrained bundle (e.g. an app binary).
- **Artifacts are generated, not committed.** `benchmarks/models/` is gitignored (export via `benchmarks/export_models.py`); the default config stays `model/weights/best.pt`.
- **Switching**: set `model_path` on `DetectorConfig` (or pass `--model` to `defect-detect`). `DetectorConfig` now has `device` and `imgsz` fields (`src/defect_detection/config.py:23-24`); `InferenceEngine` forwards both to `model.track` (`src/defect_detection/inference/engine.py:62-70`). `.mlpackage` directories are loadable (`engine._load_model` checks `os.path.exists`, not `isfile`).

## Negative results & non-obvious findings

- **ONNX-CPU is slower than the pytorch CPU baseline.** ORT 1.28 CPUExecutionProvider p50 94.08 ms (10.6 FPS) vs pytorch CPU 70.60 ms (14.2 FPS) — ONNX was tried and rejected as a deployment format on this host; CoreML is the win, not "export to ONNX".
- **MPS-vs-pipeline device gap (now fixed via kwarg).** Baseline pipeline mode could not honor `--device mps` because `InferenceEngine.track` forwarded no device kwarg; the underlying YOLO fell back to ultralytics auto (CPU on this mac). Fixed: `InferenceEngine` now accepts `device`/`imgsz` and forwards them (`src/defect_detection/inference/engine.py:62-70`), and `DetectorConfig` exposes both fields. Engine-mode MPS numbers above reflect the fix.
- **numpy downgrade 2.4.6 → 2.3.5.** coremltools 9.0 requires numpy ≤ 2.3.5; a process that already imported numpy 2.4.x keeps the stale version in `sys.modules` and the CoreML export silently crashes with `only 0-dimensional arrays can be converted to Python scalars`. `export_models.py:check_numpy_compat` asserts this up front.
- **INT8 is weight-only (`quantize='w8a16'`, kmeans palettization), no activation calibration.** A full INT8 path crashes coremltools 9.0 + torch 2.13; ultralytics 8.4.105 `validate_args` rejects `data=` for `format='coreml'`/`'mlmodel'`, so no activation calibration is possible through the ultralytics CoreML export at this version. The int8 result is therefore weight-palettized fp16-activations, not a fully quantized network.
- **CoreML compute unit is `CPU_AND_NE`, requested via ultralytics**, never `.all`/`CPU_AND_GPU` (crashes coremltools 9.x). The `device_used = cpu` column in `latency_sweep.md` is the predictor-side label, not the CoreML compute unit.

## Limitations (honest)

- **Val n = 56 ⇒ sub-~1.5 pt deltas are within noise.** Do not read the 0.9502 → 0.9567 fp16 delta as a real accuracy gain; it is noise-floor.
- **Stabilizer calibrated on ONE video** (`assets/video5.mov`, gt in `benchmarks/ground_truth/video5.json`). The `commit_score=3.0 / min_frames=2 / zone_frac=0.15` knobs are not validated across cameras, lighting, or bottle geometries.
- **CoreML host uses `CPU_AND_NE`** — requesting `.all`/`CPU_AND_GPU` crashes coremltools 9.x, so the GPU is not exercised.
- **Latency is burst-after-warmup on a fanless Air.** 300 measured frames after 30 warmup; sustained production load will throttle and the p50 will rise.

## Future work (explicitly deferred)

- **Interval inference / detect-every-N** (DeepStream-style): run the detector every Nth frame and interpolate, instead of every frame. Largest remaining throughput lever, not exercised here.
- **VideoToolbox hardware decode**: `FrameReader` is cv2 software decode; the latency harness explicitly preloads to RAM to strip decode jitter, so a production pipeline still pays it.
- **ReID trackers** (ByteTrack-alternative with appearance features): would help label stability beyond the horizon-based stabilizer; not measured.
- **Re-export-at-smaller-imgsz sweep**: 384/512 val numbers exist for the dynamic `.pt` only; CoreML/ONNX are fixed @640 and would need a fresh export per size.
- **Second-video validation**: the stability fix is calibrated and verified on `video5` alone.
- **k-fold cross-validation**: already called out in `README.md` WIP; the val n=56 noise floor is the binding constraint.

## Conformal logging guard

Calibration: `benchmarks/conformal_calibrate.py --model auto --alpha 0.1 --holdout-frac 0.5 --seed 42` over the 56-image val split (1 GT instance per image ⇒ 56 nonconformity scores). Per-instance rule: matched (IoU≥0.5) & top class == GT class ⇒ `s = 1 − conf`; matched-wrong or unmatched ⇒ `s = 1` (maximally nonconforming). q̂ = k-th smallest cal score, k = ⌈(n_cal+1)(1−α)⌉. Marginal coverage P(true class ∈ {predicted}) ≥ 1−α holds with finite-sample slack 1/(n_cal+1) (Angelopoulos & Bates, arXiv:2107.07511). Source: `benchmarks/results/conformal.json` + `conformal_report.md`.

| metric | value |
|---|---|
| model | `benchmarks/models/best_fp16.mlpackage` (auto) |
| alpha | 0.10 (nominal 0.90) |
| n_cal / n_holdout | 28 / 28 |
| q̂ | 0.0820 |
| τ = 1 − q̂ | **0.9180** |
| exact_regime (q̂ < 0.5) | True — top-class-only rule is *exact* (at most one softmax class > 0.5) |
| empirical coverage (holdout) | 0.9643 (27/28) vs nominal 0.90 — within finite-sample slack |

Per-class calibration: `good` 12 (12/0/0), `low_water` 7 (7/0/0), `no_cap` 6 (6/0/0), `no_label` 3 (3/0/0) — `matched_correct/matched_wrong/unmatched`. Calibrated detector matched every val instance; no matched-wrong or unmatched rows entered the calibration set.

Runtime: `CoverageGuard.from_json("benchmarks/results/conformal.json")` loads τ at `DetectionPipeline.__init__`; a defect at the crossing with `conf < τ` → `UNCERTAIN` bottle row (no defect row, no crop save), `det["coverage"] = "abstained"`, `total_abstentions += 1`. Inspector's `total_defects` / `total_inspected` are unchanged (guard changes persistence, not counting).

Replay regression with guard active (`benchmarks/results/replay_coverage.md`, `benchmarks/models/best_fp16.mlpackage`):

- Crossings reconstructed: 5 / 5 (OK-good=2, OK-defect=3, FALSE-POSITIVE=0, MISS=0, WRONG-TYPE=0).
- Per-crossing conf vs τ=0.918: seq1 `good` 0.823 (no guard — good), seq2 `no_label` 0.934 ≥ τ **covered**, seq3 `low_water` 0.934 ≥ τ **covered**, seq4 `good` 0.924 (no guard — good), seq5 `no_cap` **0.887 < τ → ABSTAINED**.
- **FINDING**: the `no_cap` GT defect at seq 5 abstains (conf 0.887 just below τ=0.918). The inspector's `logged=True` flag still appears in the replay's crossing table because it reads the inspector view (not the persistence) — the *persistence path* rewrote it: an `UNCERTAIN` bottle row was written in place of a defect row, `total_abstentions=1`. The guard is working as designed; the replay harness surfaces the inspector's intent but does not yet drill into the conformal persistence split — a documented harness blind spot, not a regression.

## Safety

See `defect-detection-vault/Safety — ODD & FMEA.md` for the ODD and the 8-row SOTIF-flavored
FMEA. Applicable frame is ISO 21448 (perception capability insufficiency), not ISO 26262
(systematic E/E faults) — see [[Elevation Plan — Rigor, Uncertainty & Validation]] alternatives
ledger for the rejected-standards rationale. Runtime monitor: `DetectionPipeline._check_invariants`
(NaN/Inf confidence + sane detection count; log-only warnings this session) plus the conformal
abstain path in `CoverageGuard` (coverage-guaranteed, not threshold-cut). The n=5 replay Wilson
CI [0.566, 1.0] is still binding; the guard is not framed as a 100% claim.

## Reproduction

All harnesses are stdlib + project deps only; run from repo root with the dev venv active. Commands below are the exact invocations that produced the evidence files.

### 1. Train/val split (regenerates the val set the mAP sweep reads)

```
python benchmarks/make_split.py --src dataset/data --dst dataset/split --val-pct 0.2 --seed 42
```

### 2. Export artifacts (writes `benchmarks/models/{best_fp16.mlpackage, best_int8.mlpackage, best_fp32.onnx}` + `MANIFEST.md`)

```
python benchmarks/export_models.py --weights model/weights/best.pt --outdir benchmarks/models --imgsz 640
```

### 3. mAP sweep (12 val runs; writes `benchmarks/results/map_sweep.md`)

```
python benchmarks/map_eval.py --data dataset/data.yaml --split val --out benchmarks/results/map_sweep.md
```

### 4. Latency baseline (pytorch, CPU + MPS; writes `benchmarks/results/latency_baseline.md`)

```
python benchmarks/latency.py --model model/weights/best.pt --video assets/video5.mov \
  --warmup 30 --frames 300 --runs 3 --out benchmarks/results/latency_baseline.md

python benchmarks/latency.py --model model/weights/best.pt --device mps --video assets/video5.mov \
  --warmup 30 --frames 300 --runs 3 --append --out benchmarks/results/latency_baseline.md
```

### 5. Latency sweep (exported artifacts; appends sections to `benchmarks/results/latency_sweep.md`)

```
for m in best_fp16.mlpackage best_int8.mlpackage best_fp32.onnx; do
  python benchmarks/latency.py --model benchmarks/models/$m --video assets/video5.mov \
    --warmup 30 --frames 300 --runs 3 \
    --append --out benchmarks/results/latency_sweep.md
done
```

### 6. Replay (baseline + stabilized; writes `benchmarks/results/{replay_*,*}.md` + `.jsonl`)

```
python benchmarks/replay.py --video assets/video5.mov --model model/weights/best.pt \
  --gt benchmarks/ground_truth/video5.json --conf 0.5 \
  --out-prefix benchmarks/results/replay_baseline
# (toggle the stabilizer path in src/, re-run, write to replay_stabilized)
python benchmarks/replay.py --video assets/video5.mov --model model/weights/best.pt \
  --gt benchmarks/ground_truth/video5.json --conf 0.5 \
  --out-prefix benchmarks/results/replay_stabilized
```