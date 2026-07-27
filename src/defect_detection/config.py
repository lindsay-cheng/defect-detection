"""configuration for the detection pipeline"""

import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from defect_detection.constants import DEFAULT_CONF_THRESHOLD, DEFAULT_DB_PATH

log = logging.getLogger(__name__)

# ponytail: the CoreML fp16 artifact is ~4.4x faster than the CPU .pt baseline on this
# host (see benchmarks/RESULTS.md); auto-select it when present so the app is fast by
# default. falls back to the committed .pt with a warning. upgrade = a config flag
# forcing one or the other if a user ever wants to override the auto pick.
_COREML_DEFAULT = "benchmarks/models/best_fp16.mlpackage"
_PT_FALLBACK = "model/weights/best.pt"


def resolve_model_path(path: str | None) -> str | None:
    """resolve a DetectorConfig.model_path value to a concrete artifact path.

    "auto"  -> the CoreML fp16 artifact if it exists, else best.pt with a warning.
    a path  -> returned unchanged (caller/engine owns existence + load errors).
    None    -> None (test mode: pipeline runs with no engine).
    """
    if path is None:
        return None
    if path != "auto":
        return path
    # ponytail: .mlpackage is a DIRECTORY; os.path.exists handles both files and dirs,
    # so the CoreML artifact is not rejected by an isfile check.
    if os.path.exists(_COREML_DEFAULT):
        return _COREML_DEFAULT
    log.warning(
        "CoreML artifact %s not found; falling back to %s. run "
        "benchmarks/export_models.py to enable the faster default.",
        _COREML_DEFAULT,
        _PT_FALLBACK,
    )
    return _PT_FALLBACK


@dataclass
class DetectorConfig:
    """settings for DetectionPipeline (paths are relative to cwd).

    model_path defaults to "auto": resolve_model_path picks the CoreML fp16
    artifact if present, else best.pt. an explicit path is used as-is; None
    means no engine (test mode)."""

    model_path: str | None = "auto"
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    db_path: str = DEFAULT_DB_PATH
    save_images: bool = True
    images_dir: str = "detections"
    tracker: str = "bytetrack.yaml"  # filename within inference/trackers/, or an absolute path
    centerline_tolerance: int = 15
    line_thickness: int = 3
    device: str | None = None  # ponytail: None = ultralytics auto. no speculative knob.
    imgsz: int = 640
    coverage_file: str | None = "benchmarks/results/conformal.json"
    # ponytail: path to the calibrated split-conformal threshold json; None disables
    # the guard. when the file is missing/invalid CoverageGuard.from_json logs and
    # returns None so the pipeline runs unguarded rather than failing the run.
    # upgrade = a per-class tau table if ultralytics ever exposes per-class logits.

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
