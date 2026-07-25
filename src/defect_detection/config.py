"""configuration for the detection pipeline"""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from defect_detection.constants import DEFAULT_CONF_THRESHOLD, DEFAULT_DB_PATH


@dataclass
class DetectorConfig:
    """settings for DetectionPipeline (paths are relative to cwd)"""

    model_path: str | None = "model/weights/best.pt"
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    db_path: str = DEFAULT_DB_PATH
    save_images: bool = True
    images_dir: str = "detections"
    tracker: str = "bytetrack.yaml"  # filename within inference/trackers/, or an absolute path
    centerline_tolerance: int = 15
    line_thickness: int = 3
    device: str | None = None  # ponytail: None = ultralytics auto. no speculative knob.
    imgsz: int = 640
    imgsz: int = 640

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DetectorConfig":
        """load config from a yaml file; unknown keys raise TypeError"""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise TypeError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)
