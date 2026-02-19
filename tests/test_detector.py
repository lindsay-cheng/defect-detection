"""tests for backend.detector — centerline logic, display IDs, image save, validation"""
import os
import tempfile
import numpy as np
import pytest

from backend.detector import DefectDetector


@pytest.fixture()
def detector(tmp_path):
    """detector with no model and a temp DB + images dir"""
    db_path = str(tmp_path / "test.db")
    images_dir = str(tmp_path / "detections")
    det = DefectDetector(
        model_path=None,
        db_path=db_path,
        save_images=True,
        images_dir=images_dir,
    )
    det.start_session()
    yield det
    det.cleanup()


def _make_frame(width=100, height=100):
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestDetectFrameValidation:
    def test_rejects_none(self, detector):
        with pytest.raises(ValueError, match="None or empty"):
            detector.detect_frame(None)

    def test_rejects_empty(self, detector):
        with pytest.raises(ValueError, match="None or empty"):
            detector.detect_frame(np.array([]))

    def test_rejects_2d(self, detector):
        with pytest.raises(ValueError, match="ndim"):
            detector.detect_frame(np.zeros((100, 100), dtype=np.uint8))

    def test_accepts_valid_frame(self, detector):
        frame = _make_frame()
        annotated, detections = detector.detect_frame(frame)
        assert annotated is not None
        assert detections == []


class TestCenterlineLogic:
    """verify that on_centerline, display ID assignment, counting, and logging
    behave consistently when driven with synthetic detections."""

    def _inject_detections(self, detector, frame_width, bboxes_and_types):
        """simulate what detect_frame does after _run_tracking, using synthetic data"""
        mid_x = frame_width // 2

        detections = []
        for track_id, (x, y, w, h), defect_type in bboxes_and_types:
            cx = x + w // 2
            det = {
                'bbox': (x, y, w, h),
                'confidence': 0.95,
                'class_id': 0,
                'defect_type': defect_type,
                'track_id': track_id,
                'bottle_id': f"BTL_{track_id:05d}",
                'on_centerline': abs(cx - mid_x) <= detector.CENTERLINE_TOLERANCE,
            }
            detections.append(det)

        detector._assign_display_ids(detections)
        detector._count_inspected(detections)
        frame = _make_frame(frame_width)
        detector._log_detections(frame, detections)
        return detections

    def test_centerline_hit_increments_inspected(self, detector):
        # place bbox so centroid lands on center of 200px frame (mid_x=100)
        self._inject_detections(detector, 200, [
            (1, (99, 0, 2, 10), "good"),
        ])
        assert detector.total_inspected == 1

    def test_off_center_does_not_count(self, detector):
        # centroid at x=11, well outside tolerance of mid_x=100
        self._inject_detections(detector, 200, [
            (1, (10, 0, 2, 10), "good"),
        ])
        assert detector.total_inspected == 0

    def test_within_tolerance_counts(self, detector):
        # centroid at x=90 is 10 pixels from mid_x=100, within tolerance of 15
        self._inject_detections(detector, 200, [
            (1, (89, 0, 2, 10), "good"),
        ])
        assert detector.total_inspected == 1

    def test_outside_tolerance_does_not_count(self, detector):
        # centroid at x=80 is 20 pixels from mid_x=100, outside tolerance of 15
        self._inject_detections(detector, 200, [
            (1, (79, 0, 2, 10), "good"),
        ])
        assert detector.total_inspected == 0

    def test_same_track_counted_once(self, detector):
        for _ in range(3):
            self._inject_detections(detector, 200, [
                (1, (99, 0, 2, 10), "good"),
            ])
        assert detector.total_inspected == 1

    def test_defect_on_centerline_is_logged(self, detector):
        dets = self._inject_detections(detector, 200, [
            (1, (99, 0, 2, 10), "no_cap"),
        ])
        assert detector.total_defects == 1
        assert dets[0].get('logged') is True

    def test_good_on_centerline_not_logged_as_defect(self, detector):
        dets = self._inject_detections(detector, 200, [
            (1, (99, 0, 2, 10), "good"),
        ])
        assert detector.total_defects == 0
        assert dets[0].get('logged') is None


class TestDisplayIdAssignment:
    def test_first_centerline_hit_gets_display_id(self, detector):
        det = [{
            'track_id': 1, 'on_centerline': True,
            'bbox': (49, 0, 2, 10), 'bottle_id': 'BTL_00001',
        }]
        detector._assign_display_ids(det)
        assert det[0]['display_id'] == "BTL_00001"

    def test_off_centerline_gets_no_display_id(self, detector):
        det = [{
            'track_id': 2, 'on_centerline': False,
            'bbox': (10, 0, 2, 10), 'bottle_id': 'BTL_00002',
        }]
        detector._assign_display_ids(det)
        assert 'display_id' not in det[0]

    def test_consecutive_numbering(self, detector):
        for tid in range(1, 4):
            det = [{
                'track_id': tid, 'on_centerline': True,
                'bbox': (49, 0, 2, 10), 'bottle_id': f'BTL_{tid:05d}',
            }]
            detector._assign_display_ids(det)
        assert detector.next_display_number == 4

    def test_same_track_reuses_display_id(self, detector):
        det1 = [{'track_id': 5, 'on_centerline': True, 'bbox': (49, 0, 2, 10), 'bottle_id': 'BTL_00005'}]
        detector._assign_display_ids(det1)
        id1 = det1[0]['display_id']

        det2 = [{'track_id': 5, 'on_centerline': False, 'bbox': (10, 0, 2, 10), 'bottle_id': 'BTL_00005'}]
        detector._assign_display_ids(det2)
        assert det2[0]['display_id'] == id1


class TestSaveDefectImage:
    def test_saves_valid_image(self, detector):
        frame = _make_frame(200, 200)
        frame[50:60, 90:110] = 255  # white rectangle
        det = {'bbox': (90, 50, 20, 10)}
        path = detector._save_defect_image(frame, det, "BTL_00001")
        assert path is not None
        assert os.path.isfile(path)

    def test_returns_none_on_bad_dir(self, detector):
        detector.images_dir = "/nonexistent_dir_xyz"
        frame = _make_frame(200, 200)
        det = {'bbox': (10, 10, 20, 20)}
        result = detector._save_defect_image(frame, det, "BTL_00001")
        assert result is None


class TestLoadModel:
    def test_raises_on_missing_file(self, detector):
        with pytest.raises(FileNotFoundError):
            detector._load_model("/nonexistent/model.pt")


class TestStartSession:
    def test_resets_counters(self, detector):
        detector.total_inspected = 5
        detector.total_defects = 3
        detector.counted_tracks.add(1)
        detector.logged_tracks.add(1)
        detector.display_number_by_track_id[1] = 1

        detector.start_session()

        assert detector.total_inspected == 0
        assert detector.total_defects == 0
        assert len(detector.counted_tracks) == 0
        assert len(detector.logged_tracks) == 0
        assert len(detector.display_number_by_track_id) == 0
        assert detector.next_display_number == 1
        assert detector.session_id != ""
