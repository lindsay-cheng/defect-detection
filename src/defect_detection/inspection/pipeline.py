"""coordinates inference, inspection logic, and persistence"""

import logging
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from defect_detection.config import DetectorConfig
from defect_detection.constants import STATUS_PASS, make_db_key
from defect_detection.inference.engine import InferenceEngine
from defect_detection.inspection.annotator import annotate_frame
from defect_detection.inspection.inspector import Inspector
from defect_detection.inspection.types import Detection
from defect_detection.storage.database import DefectDatabase

log = logging.getLogger(__name__)


class DetectionPipeline:
    """coordinates inference, inspection logic, and persistence"""

    def __init__(self, config: DetectorConfig, database: DefectDatabase | None = None):
        self.config = config
        # ponytail: ownership tracked so cleanup() only closes a db it created; if a caller
        # injects its own DefectDatabase, the caller owns its lifecycle. upgrade: add a
        # shared/connection-pool abstraction if multiple pipelines ever share a db.
        self._owns_database = database is None
        self.database = database or DefectDatabase(config.db_path)
        self.engine = (
            InferenceEngine(
                config.model_path,
                config.conf_threshold,
                config.tracker,
                device=config.device,
                imgsz=config.imgsz,
            )
            if config.model_path
            else None
        )
        self.inspector = Inspector(config.centerline_tolerance)
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()

        if config.save_images:
            os.makedirs(config.images_dir, exist_ok=True)

    def detect_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        """run tracking and detection on a single frame

        args:
            frame: input frame (BGR format from opencv)

        returns:
            tuple of (annotated_frame, detections_list)
        """
        if frame is None or frame.size == 0:
            raise ValueError("frame cannot be None or empty")
        if frame.ndim != 3:
            raise ValueError(f"expected 3-channel frame (H, W, C), got ndim={frame.ndim}")

        detections = self.engine.track(frame) if self.engine else []

        result = self.inspector.process(detections, frame.shape[1])

        for det in result.counted:
            self.database.insert_bottle(
                bottle_id=make_db_key(self.inspector.session_id, det["display_id"]),
                display_id=det["display_id"],
                session_id=self.inspector.session_id,
                status=STATUS_PASS,
            )

        for det in result.defects:
            track_id = det.get("track_id")
            display_id = det.get("display_id")
            image_path = None
            if self.config.save_images:
                image_path = self._save_defect_image(frame, det, display_id or track_id)
            self.database.insert_defect(
                bottle_id=make_db_key(self.inspector.session_id, display_id, track_id),
                defect_type=det.get("defect_type"),
                display_id=display_id,
                session_id=self.inspector.session_id,
                confidence=det.get("confidence"),
                image_path=image_path,
                bbox=det["bbox"],
            )

        annotated_frame = annotate_frame(frame, detections, self.config.line_thickness)

        current_time = time.time()
        self.fps_buffer.append(1.0 / (current_time - self.last_time))
        self.last_time = current_time

        return annotated_frame, detections

    def _save_defect_image(
        self, frame: np.ndarray, detection: Detection, bottle_id: str | int
    ) -> str | None:
        """crop and save image of a defective bottle. returns filepath on
        success, None if the write fails so the caller never stores a bad path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.config.images_dir, f"{bottle_id}_{timestamp}.jpg")

        x, y, w, h = detection["bbox"]
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        try:
            ok = cv2.imwrite(filepath, frame[y1:y2, x1:x2])
            if not ok:
                log.warning("cv2.imwrite returned False for %s", filepath)
                return None
        except Exception as e:
            log.warning("failed to save defect image %s: %s", filepath, e)
            return None
        return filepath

    def get_fps(self) -> float:
        """get current average fps"""
        if not self.fps_buffer:
            return 0.0
        return sum(self.fps_buffer) / len(self.fps_buffer)

    def get_stats(self) -> dict:
        """get current detection statistics"""
        total_inspected = self.inspector.total_inspected
        total_defects = self.inspector.total_defects
        return {
            "fps": self.get_fps(),
            "total_inspected": total_inspected,
            "total_defects": total_defects,
            "defect_rate": (total_defects / total_inspected if total_inspected > 0 else 0.0),
        }

    def start_session(self) -> None:
        """begin a new inspection run — resets stats, tracker state, and display numbering"""
        if self.engine:
            self.engine.reset()
        self.inspector.start_session()

    def reset_tracking_state(self) -> None:
        """reset bytetrack internal state so track ids restart; also clears track-keyed state."""
        if self.engine:
            self.engine.reset()
        self.inspector.reset_tracks()

    def cleanup(self) -> None:
        """cleanup resources owned by this pipeline"""
        if self._owns_database:
            self.database.close()
