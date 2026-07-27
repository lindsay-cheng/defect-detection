"""coordinates inference, inspection logic, and persistence"""

import logging
import math
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from defect_detection.config import DetectorConfig, resolve_model_path
from defect_detection.constants import STATUS_PASS, make_db_key
from defect_detection.inference.engine import InferenceEngine
from defect_detection.inspection.annotator import annotate_frame
from defect_detection.inspection.coverage import CoverageGuard
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
        resolved_model = resolve_model_path(config.model_path)
        if resolved_model is not None:
            log.info("inference model: %s", resolved_model)
        self.engine = (
            InferenceEngine(
                resolved_model,
                config.conf_threshold,
                config.tracker,
                device=config.device,
                imgsz=config.imgsz,
            )
            if resolved_model
            else None
        )
        self.inspector = Inspector(config.centerline_tolerance)
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        # conformal coverage guard — None when coverage_file is unset or
        # conformal.json is missing/invalid (graceful disable, see CoverageGuard).
        self.guard = CoverageGuard.from_json(config.coverage_file) if config.coverage_file else None
        self.total_abstentions = 0

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

        self._check_invariants(detections)

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
            # conformal coverage guard — abstain at conf < tau by writing an
            # UNCERTAIN bottle row instead of a defect row (no insert_defect,
            # no crop save). inspector's counted/total_defects are unchanged
            # so the guard changes persistence only, not the inspector view.
            if (
                self.guard is not None
                and self.guard.verdict(det.get("confidence", 0.0)) == "abstain"
            ):
                self.database.insert_bottle(
                    bottle_id=make_db_key(self.inspector.session_id, display_id, track_id),
                    display_id=display_id,
                    session_id=self.inspector.session_id,
                    status="UNCERTAIN",
                )
                det["coverage"] = "abstained"
                self.total_abstentions += 1
                continue
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
        stats = {
            "fps": self.get_fps(),
            "total_inspected": total_inspected,
            "total_defects": total_defects,
            "defect_rate": (total_defects / total_inspected if total_inspected > 0 else 0.0),
        }
        # include abstentions only when the coverage guard is active — avoid
        # surfacing a forever-zero key in unguarded runs.
        if self.guard is not None:
            stats["abstentions"] = self.total_abstentions
        return stats

    def _check_invariants(self, detections: list[Detection]) -> None:
        """pre-write runtime monitor — log-only warnings on unsafe frame state.

        # ponytail: two invariants (NaN confidence, sane count K_max=20); upgrade
        # = full runtime monitor per vault [[Safety — ODD & FMEA]] (session_id,
        # bottle_id/track-id binding, abstain-honored enforcement).
        """
        if len(detections) > 20:
            log.warning(
                "invariant: detection count %d exceeds 20 (conveyor jam / hand in frame?) "
                "— writes proceed this frame; investigate before trusting counts",
                len(detections),
            )
        for det in detections:
            conf = det.get("confidence")
            if isinstance(conf, float) and (math.isnan(conf) or math.isinf(conf)):
                log.warning(
                    "invariant: NaN/Inf confidence in detection track_id=%s — investigate",
                    det.get("track_id"),
                )

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
