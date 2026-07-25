# benchmarks/

Latency, mAP, and replay harnesses for the optimization pass. Evidence (raw emission) lives under `results/`; the synthesis is `RESULTS.md`. Generated model artifacts live under `models/` (gitignored — export via `export_models.py`).

| script | does | one-line usage |
|---|---|---|
| `make_split.py` | stratified 80/20 train/val copy split (first-class grouping, `random.Random(seed)`) | `python benchmarks/make_split.py --src dataset/data --dst dataset/split --val-pct 0.2 --seed 42` |
| `export_models.py` | export `.pt` → coreml fp16, coreml int8 (`w8a16`), onnx fp32 + `models/MANIFEST.md` | `python benchmarks/export_models.py --weights model/weights/best.pt --imgsz 640` |
| `map_eval.py` | `model.val` sweep over {pytorch, coreml-fp16, coreml-int8, onnx-fp32} × {640,512,384} | `python benchmarks/map_eval.py --data dataset/data.yaml --split val` |
| `latency.py` | engine mode (per-call `model.track`) + pipeline mode (e2e `detect_frame`); fps = 1000/p50 | `python benchmarks/latency.py --model model/weights/best.pt --device mps --out benchmarks/results/latency_baseline.md` |
| `replay.py` | full-video replay → per-track flicker metrics + crossing correctness vs gt | `python benchmarks/replay.py --video assets/video5.mov --model model/weights/best.pt --gt benchmarks/ground_truth/video5.json --out-prefix benchmarks/results/replay` |

- `models/` is **generated and gitignored** — reproduce with `export_models.py`; never commit artifacts.
- `results/` is **tracked evidence** — raw harness output (do not hand-edit; re-run the harness to regenerate).
- `ground_truth/` is **read-only** — hand-annotated crossing labels per video.
- Full narrative + reproduction sequence: see [`RESULTS.md`](RESULTS.md).