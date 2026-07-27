"""tests for defect_detection.config.DetectorConfig"""

import pytest

from defect_detection.config import DetectorConfig


class TestDetectorConfigDefaults:
    def test_defaults(self):
        cfg = DetectorConfig()
        assert cfg.model_path == "auto"  # auto -> CoreML fp16 if present (see resolve_model_path)
        assert cfg.conf_threshold == 0.5
        assert cfg.centerline_tolerance == 15


class TestFromYaml:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text("model_path: weights/x.pt\nconf_threshold: 0.42\ncenterline_tolerance: 7\n")
        cfg = DetectorConfig.from_yaml(path)
        assert cfg.model_path == "weights/x.pt"
        assert cfg.conf_threshold == 0.42
        assert cfg.centerline_tolerance == 7

    def test_unknown_key_raises_typeerror(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text("bogus_key: 1\n")
        with pytest.raises(TypeError):
            DetectorConfig.from_yaml(path)
