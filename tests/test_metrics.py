"""tests for benchmarks.metrics — wilson CI + label stability, stdlib only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from metrics import fmt_ci, label_stability, wilson_interval


class TestWilson:
    def test_5_of_5_lower_bound(self):
        lo, hi = wilson_interval(5, 5)
        assert abs(lo - 0.566) < 0.01, (lo, hi)
        assert hi == 1.0

    def test_0_of_5_upper_bound(self):
        lo, hi = wilson_interval(0, 5)
        assert lo == 0.0
        assert abs(hi - 0.434) < 0.01, (lo, hi)

    def test_50_of_100_contains_half(self):
        lo, hi = wilson_interval(50, 100)
        assert lo < 0.5 < hi
        assert (hi - lo) < 0.25

    def test_n_zero_wide_interval(self):
        lo, hi = wilson_interval(0, 0)
        assert (lo, hi) == (0.0, 1.0)

    def test_fmt_ci(self):
        assert fmt_ci(0.566, 1.0) == "(0.566, 1.000)"


class TestLabelStability:
    def test_single_class_track_is_perfect(self):
        r = label_stability({1: ["good", "good", "good"]})
        assert r["per_track"][1] == 1.0
        assert r["mean"] == 1.0
        assert r["min"] == 1.0
        assert r["n_perfect"] == 1
        assert r["prop_perfect"] == 1.0

    def test_mixed_track_mode_fraction(self):
        # 3 good + 1 no_cap -> mode fraction 0.75
        r = label_stability({1: ["good", "good", "good", "no_cap"]})
        assert r["per_track"][1] == 0.75
        assert r["n_perfect"] == 0
        assert r["prop_perfect"] == 0.0
        # CI lower bound is 0 here, upper bounded by ~0.6
        lo, hi = r["ci_perfect"]
        assert lo == 0.0
        assert 0.4 < hi < 0.85

    def test_empty_dict_is_graceful(self):
        r = label_stability({})
        assert r["n_tracks"] == 0
        assert r["mean"] == 0.0
        assert r["n_perfect"] == 0
        assert r["ci_perfect"] == (0.0, 1.0)
