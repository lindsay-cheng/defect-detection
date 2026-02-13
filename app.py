"""
main application — integrates tkinter dashboard with detection backend
"""
import tkinter as tk
import cv2
import threading
from queue import Queue

from frontend.dashboard import InspectionDashboard
from backend.detector import DefectDetector


class DefectDetectionApp:
    """main application integrating frontend and backend"""
    
    def __init__(self, video_path="video2.mov"):
        """initialize application
        
        args:
            video_path: path to prerecorded video file
        """
        self.root = tk.Tk()
        self.dashboard = InspectionDashboard(self.root)
        self.detector = DefectDetector(
            model_path="my_model/train/weights/best.pt",
            conf_threshold=0.5,
            save_images=True
        )
        self.video_path = video_path
        self.detection_thread = None
        self.detection_running = False
        self.frame_queue = Queue(maxsize=2)
        
        self._setup_callbacks()
        self._poll_frames()
    
    def _setup_callbacks(self):
        """connect dashboard buttons to backend functionality"""
        self.dashboard.bind_button(
            self.dashboard.start_button, self.dashboard.start_label, self.start_detection)
        self.dashboard.bind_button(
            self.dashboard.stop_button, self.dashboard.stop_label, self.stop_detection)
        self.dashboard.bind_button(
            self.dashboard.stats_button, self.dashboard.stats_label,
            lambda: self.dashboard.show_stats(self.detector.database))
        self.dashboard.bind_button(
            self.dashboard.export_button, self.dashboard.export_label,
            lambda: self.dashboard.export_data(self._export_callback))
    
    def start_detection(self):
        """start the detection thread"""
        if self.detection_running:
            return
        
        self.detection_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        # visual feedback: darken start, brighten stop
        self.dashboard.start_button.config(bg="#a5d6a7")
        self.dashboard.start_label.config(bg="#a5d6a7")
        self.dashboard.stop_button.config(bg="#f44336")
        self.dashboard.stop_label.config(bg="#f44336")
    
    def stop_detection(self):
        """stop the detection thread"""
        self.detection_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        
        # reset button colors
        self.dashboard.start_button.config(bg="#4CAF50")
        self.dashboard.start_label.config(bg="#4CAF50")
        self.dashboard.stop_button.config(bg="#e57373")
        self.dashboard.stop_label.config(bg="#e57373")
    
    def _detection_loop(self):
        """main detection loop running in a background thread"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"error: could not open video file: {self.video_path}")
            self.detection_running = False
            return
        
        try:
            while self.detection_running:
                ret, frame = cap.read()
                
                # loop video when it reaches the end
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        print("error reading frame from video")
                        break
                
                annotated_frame, detections = self.detector.detect_frame(frame)
                
                if not self.frame_queue.full():
                    self.frame_queue.put(annotated_frame)
                
                stats = self.detector.get_stats()
                self._push_stats_to_dashboard(stats, detections)
        
        finally:
            cap.release()
            self.detection_running = False
    
    def _push_stats_to_dashboard(self, stats, detections):
        """schedule ui updates from the detection thread (thread-safe via root.after)"""
        self.root.after(0, lambda: self.dashboard.update_stats(
            fps=stats['fps'],
            inspected=stats['total_inspected'],
            fails=stats['total_defects']
        ))
        
        if not detections:
            return
        
        latest = detections[0]
        bottle_id = latest.get('bottle_id', 'N/A')
        defect_type = latest.get('defect_type', 'unknown')
        confidence = latest.get('confidence', 0.0)
        status = "FAIL" if defect_type != "good" else "PASS"
        
        self.root.after(0, lambda: self.dashboard.update_current_inspection(
            bottle_id=bottle_id,
            fill="N/A",
            defect=defect_type,
            status=status
        ))
        
        if status == "FAIL":
            self.root.after(0, lambda: self.dashboard.add_failure(
                bottle_id=bottle_id,
                defect_desc=f"{defect_type} ({confidence:.2f})"
            ))
    
    def _poll_frames(self):
        """periodically pull annotated frames from the queue and display them"""
        if not self.frame_queue.empty():
            self.dashboard.display_frame(self.frame_queue.get())
        self.root.after(30, self._poll_frames)  # ~33 fps
    
    def _export_callback(self):
        """callback for exporting defect data to csv"""
        from scripts.utils import export_to_csv
        export_to_csv(output_path="defect_report.csv")
    
    def on_closing(self):
        """handle application close"""
        self.stop_detection()
        self.detector.cleanup()
        self.root.destroy()
    
    def run(self):
        """run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


if __name__ == "__main__":
    from backend.database import init_database
    init_database()
    app = DefectDetectionApp(video_path="video2.mov")
    app.run()
