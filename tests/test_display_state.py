"""tests for the epistemic display state set by Inspector.process (label_state =
"committed"/"evaluating") and rendered by annotate_frame's two-state path.

label_state communicates per-track certainty to the UI: committed tracks paint
their decided class (green/red); evaluating tracks paint neutral amber so the
operator sees uncertainty instead of a flickering raw per-frame class."""

import numpy as np

from defect_detection.inspection.annotator import annotate_frame
from defect_detection.inspection.inspector import Inspector

FRAME_WIDTH = 200  # mid_x=100, centerline tolerance=15, evidence zone_half=30


def _det(track_id, cls, conf, cx=100):
    """detection with centroid cx (bbox x = cx - 1, w = 2 -> centroid = cx)."""
    return {
        "bbox": (cx - 1, 0, 2, 10),
        "confidence": conf,
        "class_id": 0,
        "defect_type": cls,
        "track_id": track_id,
    }


class TestLabelState:
    def test_no_in_zone_votes_yields_evaluating_and_untouched_defect_type(self):
        insp = Inspector()
        insp.start_session()
        # cx=10 -> |10-100|=90 > zone_half 30 -> off-zone, no vote accumulates.
        det = _det(1, "no_cap", 0.95, cx=10)
        insp.process([det], FRAME_WIDTH)
        assert det["label_state"] == "evaluating"
        assert det["defect_type"] == "no_cap"  # untouched: no stable label to apply

    def test_committed_track_yields_committed_state_and_stable_defect_type(self):
        insp = Inspector()
        insp.start_session()
        # TrackLabelStabilizer defaults: commit_score=3.0, min_frames=2.
        # 4 in-zone frames at 0.95 -> cumulative 3.8 >= 3.0 and frames>=2 -> commit.
        det = None
        for _ in range(4):
            det = _det(1, "no_cap", 0.95, cx=100)
            insp.process([det], FRAME_WIDTH)
        assert det["label_state"] == "committed"
        assert det["defect_type"] == "no_cap"

    def test_track_id_none_yields_evaluating(self):
        insp = Inspector()
        insp.start_session()
        det = _det(1, "no_cap", 0.95, cx=100)
        det["track_id"] = None
        insp.process([det], FRAME_WIDTH)
        assert det["label_state"] == "evaluating"


class TestAnnotatorSmoke:
    def test_one_committed_one_evaluating_renders_without_error(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        committed = {
            "bbox": (10, 10, 20, 20),
            "defect_type": "good",
            "confidence": 0.95,
            "track_id": 1,
            "display_id": "BTL_00001",
            "label_state": "committed",
        }
        evaluating = {
            "bbox": (150, 10, 20, 20),
            "defect_type": "good",  # would-be green if rendered as committed
            "confidence": 0.8,
            "track_id": 2,
            "display_id": "BTL_00002",
            "label_state": "evaluating",
        }
        out = annotate_frame(frame, [committed, evaluating], line_thickness=2)
        assert out is not None

        # sample a border pixel of each box on the top edge (2px stroke at y=10,11),
        # at the horizontal midpoint of the box. both boxes' label backgrounds are
        # drawn above (y<10), so the top edge stays exactly the box border color.
        # committed (good) border is green BGR (0,255,0); evaluating border is amber (0,191,255).
        committed_border = out[11, 20]
        evaluating_border = out[11, 160]
        assert not np.array_equal(committed_border, evaluating_border)
        assert tuple(committed_border) == (0, 255, 0)
        assert tuple(evaluating_border) == (0, 191, 255)
