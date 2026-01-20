"""
main application - integrates tkinter dashboard with detection backend
"""
import tkinter as tk
import cv2
import threading
from queue import Queue

from frontend.dashboard import InspectionDashboard
from backend.detector import DefectDetector


class DefectDetectionApp:
    """main application integrating frontend and backend"""
    
    def __init__(self, video_path="video.mp4"):
        """initialize application
        
        args:
            video_path: path to prerecorded video file (default: video.mp4)
        """
        # create tkinter root
        self.root = tk.Tk()
        
        # initialize dashboard
        self.dashboard = InspectionDashboard(self.root)
        
        # initialize detector with trained model
        self.detector = DefectDetector(
            model_path="my_model/train/weights/best.pt",
            conf_threshold=0.5,
            save_images=True
        )
        
        # video source path
        self.video_path = video_path
        
        # detection thread variables
        self.detection_thread = None
        self.detection_running = False
        
        # frame queue for thread-safe communication
        self.frame_queue = Queue(maxsize=2)
        
        # override dashboard button callbacks
        self._setup_callbacks()
        
        # setup periodic updates
        self._setup_update_loop()
    
    def _setup_callbacks(self):
        """connect dashboard buttons to backend functionality"""
        # connect start/stop buttons
        self.dashboard.bind_button(self.dashboard.start_button, self.dashboard.start_label, self.start_detection)
        self.dashboard.bind_button(self.dashboard.stop_button, self.dashboard.stop_label, self.stop_detection)
        
        # connect stats button
        self.dashboard.bind_button(
            self.dashboard.stats_button, self.dashboard.stats_label,
            lambda: self.dashboard.show_stats(self.detector.database)
        )
        
        # connect export button
        self.dashboard.bind_button(
            self.dashboard.export_button, self.dashboard.export_label,
            lambda: self.dashboard.export_data(self._export_callback)
        )
    
    def start_detection(self):
        """start detection thread"""
        if self.detection_running:
            return
        
        self.detection_running = True
        self.detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True
        )
        self.detection_thread.start()
        
        # update dashboard button appearance
        self.dashboard.start_button.config(bg="#a5d6a7")
        self.dashboard.start_label.config(bg="#a5d6a7")
        self.dashboard.stop_button.config(bg="#f44336")
        self.dashboard.stop_label.config(bg="#f44336")
    
    def stop_detection(self):
        """stop detection thread"""
        self.detection_running = False
        
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        
        # update dashboard button appearance
        self.dashboard.start_button.config(bg="#4CAF50")
        self.dashboard.start_label.config(bg="#4CAF50")
        self.dashboard.stop_button.config(bg="#e57373")
        self.dashboard.stop_label.config(bg="#e57373")
    
    def _detection_loop(self):
        """main detection loop running in separate thread"""
        # open video file
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"error: could not open video file: {self.video_path}")
            self.detection_running = False
            return
        
        try:
            while self.detection_running:
                ret, frame = cap.read()
                
                # if video ended, loop back to beginning
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    
                    # if still can't read, something is wrong
                    if not ret:
                        print("error reading frame from video")
                        break
                
                # run detection
                annotated_frame, detections = self.detector.detect_frame(frame)
                
                # put frame in queue for display (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(annotated_frame)
                
                # update stats
                stats = self.detector.get_stats()
                self._update_dashboard_stats(stats, detections)
        
        finally:
            cap.release()
            self.detection_running = False
    
    def _update_dashboard_stats(self, stats, detections):
        """update dashboard with current stats and detections
        
        args:
            stats: statistics dictionary
            detections: list of current detections
        """
        # update top stats bar
        self.root.after(
            0,
            lambda: self.dashboard.update_stats(
                fps=stats['fps'],
                inspected=stats['total_inspected'],
                fails=stats['total_defects']
            )
        )
        
        # update current inspection if there are detections
        if detections:
            latest = detections[0]
            bottle_id = latest.get('bottle_id', 'N/A')
            defect_type = latest.get('defect_type', 'unknown')
            confidence = latest.get('confidence', 0.0)
            
            # determine status
            status = "FAIL" if defect_type != "good" else "PASS"
            
            self.root.after(
                0,
                lambda: self.dashboard.update_current_inspection(
                    bottle_id=bottle_id,
                    fill="N/A",  # can be calculated from detection later
                    defect=defect_type,
                    status=status
                )
            )
            
            # add to failures list if defect
            if status == "FAIL":
                self.root.after(
                    0,
                    lambda: self.dashboard.add_failure(
                        bottle_id=bottle_id,
                        defect_desc=f"{defect_type} ({confidence:.2f})"
                    )
                )
    
    def _setup_update_loop(self):
        """setup periodic ui update loop"""
        self._update_frame()
    
    def _update_frame(self):
        """update video frame in dashboard"""
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            self.dashboard.display_frame(frame)
        
        # schedule next update
        self.root.after(30, self._update_frame)  # ~33 fps
    
    def _export_callback(self):
        """callback for exporting data - integrates with backend"""
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
    # initialize database on first run
    from backend.database import init_database
    init_database()
    
    # create and run app with video file
    # change video_path to your prerecorded video file
    app = DefectDetectionApp(video_path="video.mp4")
    app.run()
