"""
detection pipeline for bottle defect detection
integrates model inference, tracking, and database logging
"""
import os
import cv2
import numpy as np
import time
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

from backend.tracker import CentroidTracker
from backend.database import DefectDatabase


class DefectDetector:
    """main detection pipeline coordinating model, tracking, and logging"""
    
    DEFECT_TYPES = {
        0: "good",
        1: "low_water",
        2: "no_cap",
        3: "no_label"
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        db_path: str = "database/defects.db",
        save_images: bool = True,
        images_dir: str = "detections",
        min_frames_to_count: int = 2
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
        self.min_frames_to_count = max(1, int(min_frames_to_count))
        
        self.tracker = CentroidTracker(max_disappeared=30, max_distance=50)
        self.database = DefectDatabase(db_path)
        
        self.model = None
        if model_path:
            self._load_model(model_path)
        
        # stats tracking
        self.total_inspected = 0
        self.total_defects = 0
        # counted_bottles: dedupe for inspected count
        # defect_logged_bottles: dedupe for defect logging
        self.counted_bottles = set()
        self.defect_logged_bottles = set()
        # track confirmation to avoid 1-frame jitter counting a "new" bottle
        self._seen_frames_by_object_id: Dict[int, int] = {}
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        
        if self.save_images:
            os.makedirs(self.images_dir, exist_ok=True)
    
    def _load_model(self, model_path: str):
        """load trained yolo model for inference"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"error loading model: {e}")
            self.model = None
    
    def detect_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """run detection on a single frame
        
        args:
            frame: input frame (BGR format from opencv)
        
        returns:
            tuple of (annotated_frame, detections_list)
        """
        # run inference (returns empty list if no model loaded)
        detections = self._run_inference(frame) if self.model else []
        
        # track objects across frames to assign stable bottle IDs
        bboxes = [(d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3]) 
                  for d in detections]
        self.tracker.update(bboxes)
        
        for detection in detections:
            bbox = detection['bbox']
            cx = int(bbox[0] + bbox[2] / 2)
            cy = int(bbox[1] + bbox[3] / 2)
            object_id = self.tracker.get_object_id_by_centroid((cx, cy))
            
            if object_id is not None:
                detection['object_id'] = object_id
                detection['bottle_id'] = self.tracker.format_bottle_id(object_id)
            else:
                detection['object_id'] = None
                detection['bottle_id'] = "UNKNOWN"
        
        # count unique bottles and log defects
        self._count_inspected(detections)
        self._log_detections(frame, detections)
        
        annotated_frame = self._annotate_frame(frame.copy(), detections)
        
        # update fps
        current_time = time.time()
        self.fps_buffer.append(1.0 / (current_time - self.last_time))
        self.last_time = current_time
        
        return annotated_frame, detections
    
    def _run_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """run yolo model inference on a single frame
        
        returns:
            list of detection dicts with bbox, confidence, class, defect_type
        """
        detections = []
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    'confidence': float(box.conf[0]),
                    'class': cls,
                    'defect_type': self.DEFECT_TYPES.get(cls, 'unknown')
                })
        
        return detections
    
    def _count_inspected(self, detections: List[Dict[str, Any]]):
        """count unique bottles seen (good or defective)"""
        # note: we count by stable tracker id after it has been seen for a few frames
        seen_object_ids_this_frame = set()
        for detection in detections:
            bottle_id = detection.get('bottle_id')
            object_id = detection.get('object_id')
            if bottle_id and bottle_id != "UNKNOWN" and object_id is not None:
                seen_object_ids_this_frame.add(object_id)
                self._seen_frames_by_object_id[object_id] = self._seen_frames_by_object_id.get(object_id, 0) + 1

                if bottle_id not in self.counted_bottles and self._seen_frames_by_object_id[object_id] >= self.min_frames_to_count:
                    self.counted_bottles.add(bottle_id)
                    self.total_inspected += 1

        # reset streaks for objects not observed this frame (keeps confirmation local in time)
        for oid in list(self._seen_frames_by_object_id.keys()):
            if oid not in seen_object_ids_this_frame:
                self._seen_frames_by_object_id[oid] = 0
    
    def _log_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]):
        """log defective bottles to database (skips good bottles, dedupes by id)"""
        for detection in detections:
            bottle_id = detection.get('bottle_id')
            defect_type = detection.get('defect_type')
            
            if not bottle_id or bottle_id == "UNKNOWN" or defect_type == "good":
                continue
            if bottle_id in self.defect_logged_bottles:
                continue
            
            self.defect_logged_bottles.add(bottle_id)
            self.total_defects += 1
            
            image_path = None
            if self.save_images:
                image_path = self._save_defect_image(frame, detection, bottle_id)
            
            self.database.insert_defect(
                bottle_id=bottle_id,
                defect_type=defect_type,
                confidence=detection.get('confidence'),
                image_path=image_path,
                bbox=detection['bbox']
            )
    
    def _save_defect_image(self, frame: np.ndarray, detection: Dict[str, Any], bottle_id: str) -> str:
        """crop and save image of a defective bottle, returns the saved file path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.images_dir, f"{bottle_id}_{timestamp}.jpg")
        
        x, y, w, h = detection['bbox']
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        cv2.imwrite(filepath, frame[y1:y2, x1:x2])
        return filepath
    
    def _annotate_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """draw bounding boxes and labels on frame"""
        for detection in detections:
            x, y, w, h = detection['bbox']
            defect_type = detection.get('defect_type', 'unknown')
            confidence = detection.get('confidence', 0.0)
            bottle_id = detection.get('bottle_id', 'N/A')
            
            color = (0, 255, 0) if defect_type == "good" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"{bottle_id}: {defect_type}"
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
    
    def reset_stats(self):
        """reset detection statistics"""
        self.total_inspected = 0
        self.total_defects = 0
        self.counted_bottles.clear()
        self.defect_logged_bottles.clear()
        self._seen_frames_by_object_id.clear()
        self.tracker.reset()
    
    def cleanup(self):
        """cleanup resources"""
        self.database.close()
