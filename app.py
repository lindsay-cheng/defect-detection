"""
main application — integrates tkinter dashboard with detection backend
"""
import tkinter as tk
import cv2
import threading
from queue import Queue

from frontend.dashboard import InspectionDashboard
from backend.constants import (
    STATUS_PASS, STATUS_FAIL, DEFECT_TYPE_GOOD, get_display_id,
)
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
        self._frame_count = 0
        
        self._setup_callbacks()
        self._poll_frames()
    
    def _setup_callbacks(self):
        """connect dashboard buttons to backend functionality"""
        self.dashboard.bind_button(
            self.dashboard.start_button,
            self.dashboard.start_label,
            self.start_detection,
        )
        self.dashboard.bind_button(
            self.dashboard.stop_button,
            self.dashboard.stop_label,
            self.stop_detection,
        )
        self.dashboard.bind_button(
            self.dashboard.stats_button,
            self.dashboard.stats_label,
            self._show_stats,
        )
        self.dashboard.bind_button(
            self.dashboard.export_button,
            self.dashboard.export_label,
            self._export_data,
        )

    def _show_stats(self):
        self.dashboard.show_stats(self.detector.database)

    def _export_data(self):
        self.dashboard.export_data(self._export_callback)
    
    def start_detection(self):
        """start the detection thread"""
        if self.detection_running:
            return

        # start a new session: resets tracker, stats, and display numbering
        self.detector.start_session()

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
            msg = f"could not open video file: {self.video_path}"
            print(f"error: {msg}")
            self.root.after(0, self.dashboard.show_error, msg)
            self.detection_running = False
            return

        try:
            while self.detection_running:
                ret, frame = cap.read()

                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.detector.reset_tracking_state()
                    ret, frame = cap.read()
                    if not ret:
                        msg = "cannot read frames from video"
                        print(f"error: {msg}")
                        self.root.after(0, self.dashboard.show_error, msg)
                        break

                annotated_frame, detections = self.detector.detect_frame(frame)

                # drop stale frame to keep the display current (backpressure)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Exception:
                        pass
                self.frame_queue.put_nowait(annotated_frame)

                self._frame_count += 1
                # throttle the status-bar refresh to every 3rd frame
                stats = self.detector.get_stats() if self._frame_count % 3 == 0 else None
                self._push_stats_to_dashboard(stats, detections)

        finally:
            cap.release()
            self.detection_running = False
    
    def _push_stats_to_dashboard(self, stats, detections):
        """schedule ui updates from the detection thread (thread-safe via root.after).
        stats may be None on throttled frames."""
        if stats is not None:
            self.root.after(
                0,
                self.dashboard.update_stats,
                stats['fps'],
                stats['total_inspected'],
                stats['total_defects'],
            )
        self._push_current_inspection(detections)
        self._push_logged_failures(detections)

    def _push_current_inspection(self, detections):
        """update the 'current inspection' panel with the latest centerline bottle"""
        if not detections:
            return
        centerline = [d for d in detections if d.get('on_centerline')]
        if not centerline:
            return
        latest = centerline[0]
        display_id = get_display_id(latest)
        defect_type = latest.get('defect_type', 'unknown')
        status = STATUS_FAIL if defect_type != DEFECT_TYPE_GOOD else STATUS_PASS
        self.root.after(
            0,
            self.dashboard.update_current_inspection,
            display_id,
            "N/A",
            defect_type,
            status,
        )

    def _push_logged_failures(self, detections):
        """append newly-logged defect entries to the failures panel"""
        for det in detections:
            if det.get('logged'):
                bid = get_display_id(det)
                desc = f"{det['defect_type']} ({det['confidence']:.2f})"
                self.root.after(0, self.dashboard.add_failure, bid, desc)
    
    def _poll_frames(self):
        """periodically pull annotated frames from the queue and display them"""
        if not self.frame_queue.empty():
            self.dashboard.display_frame(self.frame_queue.get())
        self.root.after(30, self._poll_frames)  # ~33 fps
    
    def _export_callback(self):
        """callback for exporting defect data to csv (runs in background to keep UI responsive)"""
        def export_task():
            from scripts.utils import export_to_csv
            try:
                export_to_csv(output_path="defect_report.csv")
                # schedule UI update on main thread
                self.root.after(0, self.dashboard._show_export_success)
            except Exception as e:
                # schedule UI update on main thread
                self.root.after(0, self.dashboard._show_export_error, str(e))
        
        # run export in background thread to avoid blocking UI
        export_thread = threading.Thread(target=export_task, daemon=True)
        export_thread.start()
    
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
    app = DefectDetectionApp(video_path="assets/video5.mov")
    app.run()
