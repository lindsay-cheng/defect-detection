# Conformal calibration — threshold-only logging guard

## Methodology
Split conformal (Angelopoulos & Bates, *Conformal Prediction: A Gentle Introduction*, arXiv:2107.07511). Run detector at conf=0.05 over the val images from the data yaml; match each GT instance to the highest-IoU prediction with IoU>=0.5. Nonconformity s_i = 1 - p_top if matched & top class == GT class, else s_i = 1 (matched-wrong or unmatched GT — both maximally nonconforming). Seeded shuffle, 50/50 cal/holdout split; q_hat is the k-th smallest cal score with k = ceil((n_cal+1)(1-alpha)), giving marginal coverage P(true class in {predicted}) >= 1-alpha with finite-sample slack 1/(n_cal+1). Runtime rule: abstain (log UNCERTAIN bottle row, no defect row) when conf < tau = 1 - q_hat at the crossing.

## Numbers
- model: `benchmarks/models/best_fp16.mlpackage`
- alpha: 0.1  (nominal coverage 0.90)
- n_cal: 28   n_holdout: 28   total instances: 56
- q_hat: 0.0820   tau: 0.9180
- empirical coverage on holdout: 0.9643 (27/28) vs nominal 0.90

## Per-class calibration counts
| class | total | matched_correct | matched_wrong | unmatched |
|---|---:|---:|---:|---:|
| good | 12 | 12 | 0 | 0 |
| low_water | 7 | 7 | 0 | 0 |
| no_cap | 6 | 6 | 0 | 0 |
| no_label | 3 | 3 | 0 | 0 |

## Exactness verdict
- q_hat < 0.5? True. YES — top-class-only operational rule is *exact*: at most one class can exceed 0.5 in a softmax, so p_top >= tau forces the top class to be the true class whenever the abstention event fires.

## Abstain semantics (operational)
- At a centerline crossing with `det.confidence < tau` the pipeline writes an `UNCERTAIN` bottle row (no defect row, no crop save) and increments `total_abstentions`; the inspector's `total_defects`/`total_inspected` counters are unchanged. The operator UI renders the box amber with the label `BTL_xxxx: UNCERTAIN`. `tau` is loaded at pipeline init from this json via `CoverageGuard.from_json` (graceful None disable on a missing/invalid file).

## Margin-baseline comparison
- The softmax margin rule `p1 - p2 > tau'` (an alternative abstain cut) has no finite-sample coverage guarantee under split-conformal exchangeability. It additionally requires `p2` (the second-highest class probability), which the ultralytics top-class-only post-NMS API does not expose (verified on 8.4.105; `Results.boxes.data` carries only the top conf/cls). The threshold-only rule here is therefore both the simpler and the only API-exposed choice; the margin baseline is retained as a non-implemented comparison row, not an alternative path.

## Caveats (state carefully)
- Exchangeability is frame-level (per matched detection), not track-level; the per-track commit-once vote in `TrackLabelStabilizer` is intentionally outside the conformal frame.
- Val n=56 ⇒ sub-~1.5pt deltas are within noise; coverage numbers carry the `1/(n_cal+1)` finite-sample slack (~7% here at n_cal=28).

