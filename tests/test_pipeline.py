"""tests for defect_detection.inspection.pipeline — frame validation,
defect-image saving, and DefectDatabase integration."""

import os

import numpy as np
import pytest

from defect_detection.config import DetectorConfig
from defect_detection.inspection.pipeline import DetectionPipeline


@pytest.fixture()
def pipeline(tmp_path):
    """pipeline with no model, a temp DB and images dir."""
    cfg = DetectorConfig(
        model_path=None,
        db_path=str(tmp_path / "test.db"),
        save_images=True,
        images_dir=str(tmp_path / "detections"),
    )
    p = DetectionPipeline(cfg)
    p.start_session()
    yield p
    p.cleanup()


def _make_frame(width=100, height=100):
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestDetectFrameValidation:
    def test_rejects_none(self, pipeline):
        with pytest.raises(ValueError, match="None or empty"):
            pipeline.detect_frame(None)

    def test_rejects_empty(self, pipeline):
        with pytest.raises(ValueError, match="None or empty"):
            pipeline.detect_frame(np.array([]))

    def test_rejects_2d(self, pipeline):
        with pytest.raises(ValueError, match="ndim"):
            pipeline.detect_frame(np.zeros((100, 100), dtype=np.uint8))

    def test_accepts_valid_frame(self, pipeline):
        frame = _make_frame()
        annotated, detections = pipeline.detect_frame(frame)
        assert annotated is not None
        assert detections == []


class TestSaveDefectImage:
    def test_saves_valid_image(self, pipeline):
        frame = _make_frame(200, 200)
        frame[50:60, 90:110] = 255  # white rectangle
        det = {"bbox": (90, 50, 20, 10)}
        path = pipeline._save_defect_image(frame, det, "BTL_00001")
        assert path is not None
        assert os.path.isfile(path)

    def test_returns_none_on_bad_dir(self, pipeline):
        pipeline.config.images_dir = "/nonexistent_dir_xyz"
        frame = _make_frame(200, 200)
        det = {"bbox": (10, 10, 20, 20)}
        result = pipeline._save_defect_image(frame, det, "BTL_00001")
        assert result is None


class TestLoadModel:
    def test_raises_on_missing_file(self):
        from defect_detection.inference.engine import InferenceEngine

        with pytest.raises(FileNotFoundError):
            InferenceEngine("/nonexistent/model.pt", conf_threshold=0.5)
