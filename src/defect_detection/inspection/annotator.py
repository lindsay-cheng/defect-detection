"""frame annotation (bounding boxes, labels, center counting line)"""

import cv2
import numpy as np

from defect_detection.constants import DEFECT_TYPE_GOOD, get_display_id
from defect_detection.inspection.types import Detection


def annotate_frame(
    frame: np.ndarray,
    detections: list[Detection],
    line_thickness: int = 3,
) -> np.ndarray:
    """draw bounding boxes, labels, and center counting line on frame"""
    mid_x = frame.shape[1] // 2
    cv2.line(frame, (mid_x, 0), (mid_x, frame.shape[0]), (255, 255, 0), line_thickness)

    for detection in detections:
        x, y, w, h = detection["bbox"]
        defect_type = detection.get("defect_type", "unknown")
        confidence = detection.get("confidence", 0.0)
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
