"""
detection pipeline for bottle defect detection
integrates yolo tracking (bytetrack) and database logging
"""
import os
import cv2
import numpy as np
import time
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, TypedDict
from collections import deque

from backend.constants import (
    DEFAULT_DB_PATH, DEFAULT_CONF_THRESHOLD, STATUS_PASS, STATUS_FAIL,
    DEFECT_TYPE_GOOD, get_display_id, make_db_key,
)
from backend.database import DefectDatabase


class Detection(TypedDict, total=False):
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    defect_type: str
    track_id: int | None
    bottle_id: str
    display_id: str
    on_centerline: bool
    logged: bool


class DefectDetector:
    """main detection pipeline coordinating model tracking and logging"""

    DEFECT_TYPES = {
        0: "good",
        1: "low_water",
        2: "no_cap",
        3: "no_label"
    }

    # thickness of the vertical counting line drawn on the frame
    LINE_THICKNESS = 3

    # half-width (in pixels) of the centerline detection zone.
    # must be wide enough that a bottle's centroid cannot skip over it
    # between consecutive frames. independent of the visual line width.
    CENTERLINE_TOLERANCE = 15

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        db_path: str = DEFAULT_DB_PATH,
        save_images: bool = True,
        images_dir: str = "detections",
    ):
        """initialize detector

        args:
            model_path: path to trained model weights
            conf_threshold: confidence threshold for detections
            db_path: path to sqlite database
            save_images: whether to save defect images
            images_dir: directory to save defect images
        """
        self.conf_threshold = conf_threshold
        self.save_images = save_images
        self.images_dir = images_dir

        self.database = DefectDatabase(db_path)

        self.model = None
        if model_path:
            self._load_model(model_path)

        # stats
        self.total_inspected = 0
        self.total_defects = 0
        # dedupe sets keyed by track_id (int)
        self.counted_tracks = set()
        self.logged_tracks = set()
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()

        # operator-facing consecutive numbering; reset each session
        self.session_id: str = ""
        self.next_display_number: int = 1
        self.display_number_by_track_id: Dict[int, int] = {}

        if self.save_images:
            os.makedirs(self.images_dir, exist_ok=True)

    def _load_model(self, model_path: str):
        """load trained yolo model for inference. raises on failure so callers
        know immediately that the pipeline cannot run."""
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"model file not found: {model_path}")
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"model loaded successfully from: {model_path}")
        except ImportError as e:
            raise RuntimeError(f"ultralytics package not installed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"failed to load model from {model_path}: {e}") from e

    def detect_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
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

        detections = self._run_tracking(frame) if self.model else []

        frame_width = frame.shape[1]
        mid_x = frame_width // 2
        for detection in detections:
            cx = detection['bbox'][0] + detection['bbox'][2] // 2
            detection['on_centerline'] = abs(cx - mid_x) <= self.CENTERLINE_TOLERANCE

        self._assign_display_ids(detections)
        self._count_inspected(detections)
        self._log_detections(frame, detections)

        # annotate in-place; _log_detections already saved crops from the raw frame
        annotated_frame = self._annotate_frame(frame, detections)

        current_time = time.time()
        self.fps_buffer.append(1.0 / (current_time - self.last_time))
        self.last_time = current_time

        return annotated_frame, detections

    def _run_tracking(self, frame: np.ndarray) -> List[Detection]:
        """run yolo bytetrack on a single frame

        returns:
            list of detection dicts with bbox, confidence, class_id, defect_type,
            track_id, and bottle_id
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="backend/trackers/bytetrack.yaml",
            conf=self.conf_threshold,
            verbose=False
        )

        detections = []
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # boxes.id is a tensor when tracks are active, None otherwise
        track_ids = result.boxes.id

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])

            if track_ids is not None:
                track_id = int(track_ids[i].cpu())
                bottle_id = f"BTL_{track_id:05d}"
            else:
                track_id = None
                bottle_id = "UNKNOWN"

            detections.append({
                'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                'confidence': float(box.conf[0]),
                'class_id': class_id,
                'defect_type': self.DEFECT_TYPES.get(class_id, 'unknown'),
                'track_id': track_id,
                'bottle_id': bottle_id,
            })

        return detections

    def _assign_display_ids(self, detections: List[Detection]):
        """assign a consecutive operator-facing display_id on the first centerline hit per track"""
        for detection in detections:
            track_id = detection.get('track_id')
            if track_id is None:
                continue
            if track_id in self.display_number_by_track_id:
                n = self.display_number_by_track_id[track_id]
            elif detection.get('on_centerline'):
                n = self.next_display_number
                self.display_number_by_track_id[track_id] = n
                self.next_display_number += 1
            else:
                continue
            detection['display_id'] = f"BTL_{n:05d}"

    def _count_inspected(self, detections: List[Detection]):
        """count unique bottles on the vertical center line and record them in the DB as PASS.
        defective bottles are later upserted to FAIL by _log_detections.
        uses the on_centerline flag computed once in detect_frame().
        """
        for detection in detections:
            if not detection.get('on_centerline'):
                continue
            track_id = detection.get('track_id')
            if track_id is None or track_id in self.counted_tracks:
                continue
            self.counted_tracks.add(track_id)
            self.total_inspected += 1
            display_id = detection.get('display_id')
            if display_id:
                self.database.insert_bottle(
                    bottle_id=make_db_key(self.session_id, display_id),
                    display_id=display_id,
                    session_id=self.session_id,
                    status=STATUS_PASS,
                )

    def _log_detections(self, frame: np.ndarray, detections: List[Detection]):
        """log defective bottles to database when centroid is on the center line.
        uses the on_centerline flag computed once in detect_frame()."""
        for detection in detections:
            if not detection.get('on_centerline'):
                continue
            track_id = detection.get('track_id')
            defect_type = detection.get('defect_type')

            if track_id is None or defect_type == DEFECT_TYPE_GOOD:
                continue
            if track_id in self.logged_tracks:
                continue

            display_id = detection.get('display_id')

            self.logged_tracks.add(track_id)
            self.total_defects += 1
            detection['logged'] = True

            image_path = None
            if self.save_images:
                image_path = self._save_defect_image(frame, detection, display_id or track_id)

            self.database.insert_defect(
                bottle_id=make_db_key(self.session_id, display_id, track_id),
                display_id=display_id,
                session_id=self.session_id,
                defect_type=defect_type,
                confidence=detection.get('confidence'),
                image_path=image_path,
                bbox=detection['bbox']
            )

    def _save_defect_image(
        self, frame: np.ndarray, detection: Detection, bottle_id: str | int
    ) -> Optional[str]:
        """crop and save image of a defective bottle. returns filepath on
        success, None if the write fails so the caller never stores a bad path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.images_dir, f"{bottle_id}_{timestamp}.jpg")

        x, y, w, h = detection['bbox']
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        try:
            ok = cv2.imwrite(filepath, frame[y1:y2, x1:x2])
            if not ok:
                print(f"warning: cv2.imwrite returned False for {filepath}")
                return None
        except Exception as e:
            print(f"warning: failed to save defect image {filepath}: {e}")
            return None
        return filepath

    def _annotate_frame(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """draw bounding boxes, labels, and center counting line on frame"""
        mid_x = frame.shape[1] // 2
        cv2.line(frame, (mid_x, 0), (mid_x, frame.shape[0]),
                 (255, 255, 0), self.LINE_THICKNESS)

        for detection in detections:
            x, y, w, h = detection['bbox']
            defect_type = detection.get('defect_type', 'unknown')
            confidence = detection.get('confidence', 0.0)
            label_id = get_display_id(detection)

            color = (0, 255, 0) if defect_type == DEFECT_TYPE_GOOD else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label = f"{label_id}: {defect_type}"
            if confidence > 0:
                label += f" ({confidence:.2f})"

            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def get_fps(self) -> float:
        """get current average fps"""
        if not self.fps_buffer:
            return 0.0
        return sum(self.fps_buffer) / len(self.fps_buffer)

    def get_stats(self) -> Dict[str, Any]:
        """get current detection statistics"""
        return {
            "fps": self.get_fps(),
            "total_inspected": self.total_inspected,
            "total_defects": self.total_defects,
            "defect_rate": (
                self.total_defects / self.total_inspected
                if self.total_inspected > 0 else 0.0
            )
        }

    def reset_tracking_state(self):
        """reset bytetrack internal state so track IDs restart.
        also clears all track-id-keyed state (dedupe sets, display mapping)
        since those are meaningless once the tracker reassigns IDs from 1.
        session_id and next_display_number are preserved so display numbering
        continues uninterrupted across video loops within the same session.
        """
        if self.model is not None:
            predictor = getattr(self.model, 'predictor', None)
            if predictor is not None:
                trackers = getattr(predictor, 'trackers', None)
                if trackers:
                    for tracker in trackers:
                        tracker.reset()
        self.counted_tracks.clear()
        self.logged_tracks.clear()
        self.display_number_by_track_id.clear()

    def start_session(self):
        """begin a new inspection run — resets stats, tracker state, and display numbering"""
        self.reset_tracking_state()  # also clears track-keyed state
        self.total_inspected = 0
        self.total_defects = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.next_display_number = 1

    def cleanup(self):
        """cleanup resources"""
        self.database.close()
