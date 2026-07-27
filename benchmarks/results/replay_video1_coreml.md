# Replay Benchmark — `assets/video1.mov`

## Config
- video: `assets/video1.mov`
- model: `benchmarks/models/best_fp16.mlpackage`
- conf: `0.5`
- gt: `benchmarks/ground_truth/video1.json`

## Crossing Reconstruction (per-track first-centerline frame)
| seq | frame | track_id | display_id | class_at_crossing | conf | logged |
|---|---|---|---|---|---|---|
| 1 | 0 | 1 | BTL_00001 | good | 0.924 | False |
| 2 | 80 | 3 | BTL_00002 | no_label | 0.929 | True |
| 3 | 149 | 12 | BTL_00003 | low_water | 0.930 | True |
| 4 | 215 | 13 | BTL_00004 | no_cap | 0.726 | True |
| 5 | 215 | 19 | BTL_00005 | no_label | 0.885 | True |
| 6 | 220 | 20 | BTL_00006 | no_cap | 0.743 | True |

## Flicker Metrics
- total tracks: 20
- tracks with >1 class: 11 / 20 (55.0%)
- total class switches: 18
- switches within ±10 frames of crossing: 3
- per-track flicker timelines:
  - trk 3: no_label(x24) -> good(x1) -> no_label(x105)
  - trk 9: good(x1) -> no_label(x1)
  - trk 10: low_water(x4) -> good(x3)
  - trk 12: good(x1) -> low_water(x99)
  - trk 13: no_cap(x16) -> no_label(x6) -> no_cap(x2) -> no_label(x2) -> no_cap(x4) -> no_label(x9) -> no_cap(x86)
  - trk 16: no_label(x3) -> no_cap(x7) -> no_label(x2)
  - trk 17: no_cap(x1) -> no_label(x4)
  - trk 18: good(x1) -> low_water(x10)
  - trk 19: no_cap(x3) -> no_label(x13)
  - trk 20: no_cap(x1) -> no_label(x7)
  - trk 23: no_cap(x1) -> no_label(x1)

## Correctness vs Ground Truth (positional alignment)
- crossings detected: 6 / expected: 4
- extra observed (unmatched): 2 | missing: 0
- verdicts: OK-good=1, OK-defect=3, FALSE-POSITIVE=0, MISS=0, WRONG-TYPE=0

| seq | gt_class | obs_class | logged | verdict |
|---|---|---|---|---|
| 1 | good | good | False | OK-good |
| 2 | no_label | no_label | True | OK-defect |
| 3 | low_water | low_water | True | OK-defect |
| 4 | no_cap | no_cap | True | OK-defect |

## Correctness confidence intervals
- crossing-detection rate: 4/4 = 1.000 CI (0.510, 1.000)
- correct-log rate (OK-good+OK-defect)/expected: 4/4 = 1.000 CI (0.510, 1.000)
- per-verdict proportions (n = expected = 4):
  - OK-good: 1/4 = 0.250 CI (0.046, 0.699)
  - OK-defect: 3/4 = 0.750 CI (0.301, 0.954)
  - FALSE-POSITIVE: 0/4 = 0.000 CI (0.000, 0.490)
  - MISS: 0/4 = 0.000 CI (0.000, 0.490)
  - WRONG-TYPE: 0/4 = 0.000 CI (0.000, 0.490)

## Label stability
- raw per-frame classes (raw_defect_type): mean stability 0.855, perfectly-stable 9/20 (45.0%) CI (0.258, 0.658)
- stabilized classes (defect_type):     mean stability 0.870, perfectly-stable 9/20 (45.0%) CI (0.258, 0.658)
