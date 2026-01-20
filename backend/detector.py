"""
detection pipeline for bottle defect detection
integrates model inference, tracking, and database logging
"""
import cv2
import numpy as np
import time
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
        images_dir: str = "detections"
    ):
        """initialize detector
        
        args:
            model_path: path to trained model (None for testing without model)
            conf_threshold: confidence threshold for detections
            db_path: path to sqlite database
            save_images: whether to save defect images
            images_dir: directory to save defect images
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.save_images = save_images
        self.images_dir = images_dir
        
        self.tracker = CentroidTracker(max_disappeared=30, max_distance=50)
        self.database = DefectDatabase(db_path)
        self.model = None
        if model_path:
            self._load_model(model_path)
        
        self.total_inspected = 0
        self.total_defects = 0
        self.logged_bottles = set()
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        import os
        if self.save_images and not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
    
    def _load_model(self, model_path: str):
        """load trained model for inference
        
        args:
            model_path: path to model weights
        """
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
        detections = []
        
        if self.model is None:
            detections = self._mock_detection(frame)
        else:
            detections = self._run_inference(frame)
        bboxes = [(d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3]) 
                  for d in detections]
        tracked_objects = self.tracker.update(bboxes)
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
        
        self._log_detections(frame, detections)
        annotated_frame = self._annotate_frame(frame.copy(), detections)
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.fps_buffer.append(fps)
        self.last_time = current_time
        
        return annotated_frame, detections
    
    def _mock_detection(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """mock detection for testing without a trained model
        
        args:
            frame: input frame
        
        returns:
            list of mock detections
        """
        return []
    
    def _run_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """run model inference on frame
        
        args:
            frame: input frame
        
        returns:
            list of detections
        """
        detections = []
        
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                    'confidence': conf,
                    'class': cls,
                    'defect_type': self.DEFECT_TYPES.get(cls, 'unknown')
                })
        
        return detections
    
    def _log_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]):
        """log defective bottles to database
        
        args:
            frame: current frame
            detections: list of detections
        """
        for detection in detections:
            bottle_id = detection.get('bottle_id')
            defect_type = detection.get('defect_type')
            
            if not bottle_id or bottle_id == "UNKNOWN" or defect_type == "good":
                continue
            if bottle_id in self.logged_bottles:
                continue
            self.logged_bottles.add(bottle_id)
            self.total_defects += 1
            image_path = None
            if self.save_images:
                image_path = self._save_defect_image(frame, detection, bottle_id)
            bbox = detection['bbox']
            self.database.insert_defect(
                bottle_id=bottle_id,
                defect_type=defect_type,
                confidence=detection.get('confidence'),
                image_path=image_path,
                bbox=bbox
            )
    
    def _save_defect_image(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        bottle_id: str
    ) -> str:
        """save image of defective bottle
        
        args:
            frame: current frame
            detection: detection info
            bottle_id: bottle identifier
        
        returns:
            path to saved image
        """
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{bottle_id}_{timestamp}.jpg"
        filepath = os.path.join(self.images_dir, filename)
        
        x, y, w, h = detection['bbox']
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        cropped = frame[y1:y2, x1:x2]
        cv2.imwrite(filepath, cropped)
        
        return filepath
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """draw bounding boxes and labels on frame
        
        args:
            frame: input frame
            detections: list of detections
        
        returns:
            annotated frame
        """
        for detection in detections:
            x, y, w, h = detection['bbox']
            defect_type = detection.get('defect_type', 'unknown')
            confidence = detection.get('confidence', 0.0)
            bottle_id = detection.get('bottle_id', 'N/A')
            
            if defect_type == "good":
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{bottle_id}: {defect_type}"
            if confidence > 0:
                label += f" ({confidence:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame, (x, y - text_h - 10), (x + text_w, y), color, -1
            )
            cv2.putText(
                frame, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        return frame
    
    def get_fps(self) -> float:
        """get current average fps
        
        returns:
            average fps over recent frames
        """
        if len(self.fps_buffer) == 0:
            return 0.0
        return sum(self.fps_buffer) / len(self.fps_buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        """get current detection statistics
        
        returns:
            stats dictionary
        """
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
        self.logged_bottles.clear()
        self.tracker.reset()
    
    def cleanup(self):
        """cleanup resources"""
        self.database.close()


if __name__ == "__main__":
    detector = DefectDetector()
    cap = cv2.VideoCapture(1)
    print("press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame, detections = detector.detect_frame(frame)
        
        stats = detector.get_stats()
        cv2.putText(
            annotated_frame, f"FPS: {stats['fps']:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        cv2.imshow("Defect Detector Test", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.cleanup()
