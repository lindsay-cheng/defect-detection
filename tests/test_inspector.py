"""tests for defect_detection.inspection.inspector — pure centerline logic,
display id assignment, and session state."""

import pytest

from defect_detection.inspection.inspector import Inspector


@pytest.fixture()
def inspector():
    """plain Inspector with default tolerance (15) — no db, no side effects."""
    insp = Inspector()
    insp.start_session()
    return insp


def _detections(frame_width, bboxes_and_types):
    """build detection dicts the way InferenceEngine.track would (minus on_centerline,
    which Inspector.process computes itself)."""
    out = []
    for track_id, (x, y, w, h), defect_type in bboxes_and_types:
        out.append(
            {
                "bbox": (x, y, w, h),
                "confidence": 0.95,
                "class_id": 0,
                "defect_type": defect_type,
                "track_id": track_id,
                "bottle_id": f"BTL_{track_id:05d}",
            }
        )
    return out


def _run(inspector, frame_width, bboxes_and_types):
    """inject synthetic detections through process() and return them."""
    dets = _detections(frame_width, bboxes_and_types)
    inspector.process(dets, frame_width)
    return dets


class TestCenterlineLogic:
    """counting, dedupe, and defect-logging behavior driven via Inspector.process.
    frame_width=200 ⇒ mid_x=100, tolerance 15."""

    def test_centerline_hit_increments_inspected(self, inspector):
        _run(inspector, 200, [(1, (99, 0, 2, 10), "good")])
        assert inspector.total_inspected == 1

    def test_off_center_does_not_count(self, inspector):
        # centroid at x=11, well outside tolerance of mid_x=100
        _run(inspector, 200, [(1, (10, 0, 2, 10), "good")])
        assert inspector.total_inspected == 0

    def test_within_tolerance_counts(self, inspector):
        # centroid at x=90 is 10px from mid_x=100, within tolerance 15
        _run(inspector, 200, [(1, (89, 0, 2, 10), "good")])
        assert inspector.total_inspected == 1

    def test_outside_tolerance_does_not_count(self, inspector):
        # centroid at x=80 is 20px from mid_x=100, outside tolerance 15
        _run(inspector, 200, [(1, (79, 0, 2, 10), "good")])
        assert inspector.total_inspected == 0

    def test_same_track_counted_once(self, inspector):
        for _ in range(3):
            _run(inspector, 200, [(1, (99, 0, 2, 10), "good")])
        assert inspector.total_inspected == 1

    def test_defect_on_centerline_is_logged(self, inspector):
        dets = _run(inspector, 200, [(1, (99, 0, 2, 10), "no_cap")])
        assert inspector.total_defects == 1
        assert dets[0].get("logged") is True

    def test_good_on_centerline_not_logged_as_defect(self, inspector):
        dets = _run(inspector, 200, [(1, (99, 0, 2, 10), "good")])
        assert inspector.total_defects == 0
        assert dets[0].get("logged") is None


class TestDisplayIdAssignment:
    """display ids are assigned on the first centerline hit per track and reused thereafter."""

    def test_first_centerline_hit_gets_display_id(self, inspector):
        det = _detections(100, [(1, (49, 0, 2, 10), "good")])
        inspector.process(det, 100)
        assert det[0]["display_id"] == "BTL_00001"

    def test_off_centerline_gets_no_display_id(self, inspector):
        det = _detections(100, [(2, (10, 0, 2, 10), "good")])
        inspector.process(det, 100)
        assert "display_id" not in det[0]

    def test_consecutive_numbering(self, inspector):
        for tid in range(1, 4):
            det = _detections(100, [(tid, (49, 0, 2, 10), "good")])
            inspector.process(det, 100)
        assert inspector.next_display_number == 4

    def test_same_track_reuses_display_id(self, inspector):
        det1 = _detections(100, [(5, (49, 0, 2, 10), "good")])
        inspector.process(det1, 100)
        id1 = det1[0]["display_id"]

        # second appearance is off-centerline but should still reuse the assigned id
        det2 = _detections(100, [(5, (10, 0, 2, 10), "good")])
        inspector.process(det2, 100)
        assert det2[0]["display_id"] == id1


class TestStartSession:
    def test_resets_counters(self, inspector):
        inspector.total_inspected = 5
        inspector.total_defects = 3
        inspector.counted_tracks.add(1)
        inspector.logged_tracks.add(1)
        inspector.display_number_by_track_id[1] = 1

        inspector.start_session()

        assert inspector.total_inspected == 0
        assert inspector.total_defects == 0
        assert len(inspector.counted_tracks) == 0
        assert len(inspector.logged_tracks) == 0
        assert len(inspector.display_number_by_track_id) == 0
        assert inspector.next_display_number == 1
        assert inspector.session_id != ""
