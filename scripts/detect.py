"""
standalone detection script — runs inference in an opencv window (no gui)
useful for quick testing without the full tkinter dashboard
"""
import cv2
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.detector import DefectDetector


def detect_live(
    model_path: str = None,
    source: int = 0,
    conf_threshold: float = 0.5,
    save_detections: bool = True
):
    """run real-time detection on a video source
    
    args:
        model_path: path to trained model (None runs without a model)
        source: video source (0 for webcam, or path to video file)
        conf_threshold: confidence threshold
        save_detections: save defect images to disk
    """
    detector = DefectDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        save_images=save_detections
    )
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"error: could not open video source {source}")
        return
    
    print("detection started. press 'q' to quit, 'r' to reset stats")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("end of video or cannot read frame")
                break
            
            annotated_frame, detections = detector.detect_frame(frame)
            stats = detector.get_stats()
            
            cv2.putText(
                annotated_frame, f"FPS: {stats['fps']:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.putText(
                annotated_frame,
                f"inspected: {stats['total_inspected']} | defects: {stats['total_defects']}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            cv2.imshow("bottle defect detection", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.reset_stats()
                print("stats reset")
    
    except KeyboardInterrupt:
        print("\ndetection stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()
        
        stats = detector.get_stats()
        print("\n=== final statistics ===")
        print(f"total inspected: {stats['total_inspected']}")
        print(f"total defects: {stats['total_defects']}")
        print(f"defect rate: {stats['defect_rate']:.2%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="run bottle defect detection")
    parser.add_argument("--model", type=str, default=None,
                       help="path to trained model (optional for testing)")
    parser.add_argument("--source", type=str, default="0",
                       help="video source (0 for webcam, or video file path)")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="confidence threshold")
    parser.add_argument("--no-save", action="store_true",
                       help="don't save defect images")
    
    args = parser.parse_args()
    source = 0 if args.source == "0" else args.source
    
    detect_live(
        model_path=args.model,
        source=source,
        conf_threshold=args.conf,
        save_detections=not args.no_save
    )
