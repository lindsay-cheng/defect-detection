# Replay Benchmark — `assets/video5.mov`

## Config
- video: `assets/video5.mov`
- model: `model/weights/best.pt`
- conf: `0.5`
- gt: `benchmarks/ground_truth/video5.json`

## Crossing Reconstruction (per-track first-centerline frame)
| seq | frame | track_id | display_id | class_at_crossing | conf | logged |
|---|---|---|---|---|---|---|
| 1 | 166 | 1 | BTL_00001 | good | 0.808 | False |
| 2 | 308 | 9 | BTL_00002 | no_label | 0.936 | True |
| 3 | 461 | 15 | BTL_00003 | low_water | 0.936 | True |
| 4 | 624 | 25 | BTL_00004 | good | 0.924 | False |
| 5 | 768 | 36 | BTL_00005 | no_cap | 0.888 | True |

## Flicker Metrics
- total tracks: 24
- tracks with >1 class: 10 / 24 (41.7%)
- total class switches: 31
- switches within ±10 frames of crossing: 0
- per-track flicker timelines:
  - trk 9: good(x11) -> no_label(x8) -> good(x7) -> no_label(x1) -> good(x3) -> no_label(x1) -> good(x1) -> no_label(x14) -> good(x1) -> no_label(x146)
  - trk 10: no_label(x1) -> good(x11) -> no_label(x7) -> good(x1) -> no_label(x3) -> good(x1) -> no_label(x1) -> good(x7)
  - trk 15: good(x23) -> low_water(x192)
  - trk 19: no_label(x1) -> good(x1) -> no_label(x2)
  - trk 20: no_label(x4) -> good(x2)
  - trk 23: no_label(x1) -> good(x13)
  - trk 25: good(x1) -> low_water(x4) -> good(x58) -> no_cap(x10) -> good(x8) -> no_cap(x2) -> good(x137)
  - trk 26: good(x3) -> low_water(x2)
  - trk 28: good(x1) -> low_water(x2)
  - trk 30: good(x1) -> low_water(x6) -> good(x3)

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
