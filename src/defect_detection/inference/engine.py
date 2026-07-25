"""inference engine: loads a yolo model and runs detection+tracking on frames"""

import logging
import os
from pathlib import Path

import numpy as np

from defect_detection.constants import DEFECT_TYPES
from defect_detection.inspection.types import Detection

log = logging.getLogger(__name__)


class InferenceEngine:
    """loads a yolo model and runs detection+tracking on frames"""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float,
        tracker: str = "bytetrack.yaml",
        device: str | None = None,
        imgsz: int = 640,
    ):
        """device=None uses ultralytics auto; imgsz is the inference image size."""
        self.conf_threshold = conf_threshold
        self.device = device
        self.imgsz = imgsz
        if Path(tracker).is_absolute():
            self.tracker_path = tracker
        else:
            self.tracker_path = str(Path(__file__).parent / "trackers" / tracker)
        self.model = None
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """load trained yolo model for inference. raises on failure so callers
        know immediately that the pipeline cannot run."""
        # ponytail: exists, not isfile — ultralytics-loadable artifacts include .mlpackage
        # directories; rejecting them would silently break CoreML deployment through the pipeline.
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model file not found: {model_path}")
        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)
            log.info("model loaded successfully from: %s", model_path)
        except ImportError as e:
            raise RuntimeError(f"ultralytics package not installed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"failed to load model from {model_path}: {e}") from e

    def track(self, frame: np.ndarray) -> list[Detection]:
        """run yolo tracking on a single frame

        device/imgsz are forwarded to model.track when set; device=None means
        ultralytics auto. returns:
            list of detection dicts with bbox, confidence, class_id, defect_type,
            track_id, and bottle_id
        """
        kwargs = dict(
            persist=True,
            tracker=self.tracker_path,
            conf=self.conf_threshold,
            verbose=False,
            imgsz=self.imgsz,
        )
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.track(frame, **kwargs)

        detections: list[Detection] = []
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        data = result.boxes.data.cpu().numpy()
        is_track = result.boxes.is_track

        for row in data:
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            if is_track:
                track_id = int(row[4])
                conf = float(row[5])
                class_id = int(row[6])
                bottle_id = f"BTL_{track_id:05d}"
            else:
                track_id = None
                conf = float(row[4])
                class_id = int(row[5])
                bottle_id = "UNKNOWN"

            detections.append(
                {
                    "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    "confidence": conf,
                    "class_id": class_id,
                    "defect_type": DEFECT_TYPES.get(class_id, "unknown"),
                    "track_id": track_id,
                    "bottle_id": bottle_id,
                }
            )

        return detections

    def reset(self) -> None:
        """reset tracker internal state so track ids restart"""
        if self.model is not None:
            predictor = getattr(self.model, "predictor", None)
            if predictor is not None:
                trackers = getattr(predictor, "trackers", None)
                if trackers:
                    for tracker in trackers:
                        tracker.reset()
