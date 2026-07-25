# Replay Benchmark — `assets/video5.mov`

## Config
- video: `assets/video5.mov`
- model: `benchmarks/models/best_fp16.mlpackage`
- conf: `0.5`
- gt: `benchmarks/ground_truth/video5.json`

## Crossing Reconstruction (per-track first-centerline frame)
| seq | frame | track_id | display_id | class_at_crossing | conf | logged |
|---|---|---|---|---|---|---|
| 1 | 166 | 1 | BTL_00001 | good | 0.823 | False |
| 2 | 308 | 7 | BTL_00002 | no_label | 0.934 | True |
| 3 | 461 | 15 | BTL_00003 | low_water | 0.934 | True |
| 4 | 624 | 23 | BTL_00004 | good | 0.924 | False |
| 5 | 768 | 34 | BTL_00005 | no_cap | 0.887 | True |

## Flicker Metrics
- total tracks: 26
- tracks with >1 class: 8 / 26 (30.8%)
- total class switches: 17
- switches within ±10 frames of crossing: 0
- per-track flicker timelines:
  - trk 7: good(x16) -> no_label(x8) -> good(x9) -> no_label(x140)
  - trk 9: no_label(x1) -> good(x2) -> no_label(x2) -> good(x8) -> no_label(x9)
  - trk 17: no_label(x1) -> good(x1)
  - trk 19: no_label(x1) -> good(x3) -> no_label(x6)
  - trk 23: good(x47) -> no_cap(x10) -> good(x8) -> no_cap(x2) -> good(x136)
  - trk 26: good(x1) -> low_water(x2)
  - trk 28: good(x1) -> low_water(x2)
  - trk 34: good(x1) -> no_cap(x165)

## Correctness vs Ground Truth (positional alignment)
- crossings detected: 5 / expected: 5
- extra observed (unmatched): 0 | missing: 0
- verdicts: OK-good=2, OK-defect=3, FALSE-POSITIVE=0, MISS=0, WRONG-TYPE=0

| seq | gt_class | obs_class | logged | verdict |
|---|---|---|---|---|
| 1 | good | good | False | OK-good |
| 2 | no_label | no_label | True | OK-defect |
| 3 | low_water | low_water | True | OK-defect |
| 4 | good | good | False | OK-good |
| 5 | no_cap | no_cap | True | OK-defect |
