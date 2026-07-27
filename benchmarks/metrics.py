"""statistical helpers for replay / skew benchmarks (stdlib only).

# ponytail: wilson score interval + a mode-fraction label stability metric.
# no scipy / numpy — counters and math.sqrt only. upgrade = jacknife CI on
# stability; weighted-by-track-length variant of the mode-fraction.
"""

from __future__ import annotations

import math
from collections import Counter


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """two-sided Wilson score interval for a binomial proportion.

    returns (0.0, 1.0) for n=0 so callers using it as a wide-uncertainty
    placeholder never divide-by-zero or crash. clamps to [0, 1].
    """
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def label_stability(tracks: dict[int, list[str]]) -> dict:
    """per-track mode-class fraction + summary.

    for each track: stab = max_class_count / track_length (1.0 = perfectly
    single-class). summary = mean, min, and a Wilson CI on the proportion of
    tracks whose stability == 1.0. empty input -> all-zero summary with a
    (0, 1) placeholder CI.
    """
    per_track: dict[int, float] = {}
    for tid, classes in tracks.items():
        if not classes:
            continue
        counts = Counter(classes)
        top = max(counts.values())
        per_track[tid] = top / len(classes)

    if not per_track:
        return {
            "per_track": {},
            "n_tracks": 0,
            "mean": 0.0,
            "min": 0.0,
            "n_perfect": 0,
            "prop_perfect": 0.0,
            "ci_perfect": (0.0, 1.0),
        }

    n = len(per_track)
    n_perfect = sum(1 for s in per_track.values() if s == 1.0)
    prop = n_perfect / n
    ci = wilson_interval(n_perfect, n)
    vals = list(per_track.values())
    return {
        "per_track": per_track,
        "n_tracks": n,
        "mean": sum(vals) / n,
        "min": min(vals),
        "n_perfect": n_perfect,
        "prop_perfect": prop,
        "ci_perfect": ci,
    }


def fmt_ci(lo: float, hi: float) -> str:
    """markdown helper: (lo, hi) at 3dp."""
    return f"({lo:.3f}, {hi:.3f})"
