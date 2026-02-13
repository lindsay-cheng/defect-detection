"""
tkinter dashboard for the bottle inspection system
"""
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk


class InspectionDashboard:
    """gui dashboard showing live feed, stats, and controls"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Industrial Inspection System")
        self.root.geometry("1280x720")
        self.root.configure(bg="#e0e0e0")
        
        self.fps = 0
        self.inspected = 0
        self.fails = 0
        self.current_id = "BTL_00000"
        self.current_fill = "0.0%"
        self.current_defect = ""
        self.current_status = ""
        
        self._setup_ui()
        
    def _setup_ui(self):
        """build the full dashboard layout"""
        self._setup_status_bar()
        
        content_frame = tk.Frame(self.root, bg="#e0e0e0")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self._setup_left_panel(content_frame)
        self._setup_right_panel(content_frame)
    
    def _setup_status_bar(self):
        """top bar with fps, inspected count, and fail count"""
        status_frame = tk.Frame(self.root, bg="#4a7c59", height=40)
        status_frame.pack(fill=tk.X, side=tk.TOP)
        status_frame.pack_propagate(False)
        
        tk.Label(status_frame, text="REC", bg="#4a7c59", fg="white", 
                font=("Arial", 10, "bold"), padx=10).pack(side=tk.LEFT)
        
        self.fps_label = tk.Label(status_frame, text="FPS: 0.0", bg="#4a7c59", 
                                 fg="white", font=("Arial", 10))
        self.fps_label.pack(side=tk.LEFT, padx=10)
        
        tk.Label(status_frame, text="|", bg="#4a7c59", fg="white").pack(side=tk.LEFT)
        
        self.inspected_label = tk.Label(status_frame, text="Inspected: 0", 
                                       bg="#4a7c59", fg="white", font=("Arial", 10))
        self.inspected_label.pack(side=tk.LEFT, padx=10)
        
        tk.Label(status_frame, text="|", bg="#4a7c59", fg="white").pack(side=tk.LEFT)
        
        self.fails_label = tk.Label(status_frame, text="Fails: 0", bg="#4a7c59", 
                                   fg="white", font=("Arial", 10))
        self.fails_label.pack(side=tk.LEFT, padx=10)
    
    def _setup_left_panel(self, parent):
        """left side: live video feed + recent failures log"""
        left_frame = tk.Frame(parent, bg="#e0e0e0")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # video feed
        feed_frame = tk.LabelFrame(left_frame, text="Live Feed", bg="#f5f5f5", 
                                  fg="black", font=("Arial", 10), relief=tk.GROOVE, bd=2)
        feed_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_label = tk.Label(feed_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # recent failures
        failures_frame = tk.LabelFrame(left_frame, text="Recent Failures", 
                                      bg="#f5f5f5", fg="black", font=("Arial", 10), relief=tk.GROOVE, bd=2)
        failures_frame.pack(fill=tk.BOTH, expand=True)
        
        scroll = tk.Scrollbar(failures_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.failures_text = tk.Text(failures_frame, bg="white", fg="black", 
                                    font=("Courier", 9), yscrollcommand=scroll.set,
                                    height=8)
        self.failures_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll.config(command=self.failures_text.yview)
    
    def _setup_right_panel(self, parent):
        """right side: current inspection info + control buttons"""
        right_frame = tk.Frame(parent, bg="#e0e0e0", width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        self._setup_inspection_panel(right_frame)
        self._setup_controls_panel(right_frame)
    
    def _setup_inspection_panel(self, parent):
        """current inspection details (id, fill, defect, status)"""
        inspection_frame = tk.LabelFrame(parent, text="Current Inspection", 
                                        bg="#f5f5f5", fg="black", font=("Arial", 10), relief=tk.GROOVE, bd=2)
        inspection_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        info_frame = tk.Frame(inspection_frame, bg="#f5f5f5")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        labels = [
            ("ID:", self.current_id, "black", "bold"),
            ("Fill:", self.current_fill, "#4CAF50", "bold"),
            ("Defect:", self.current_defect, "black", None),
            ("Status:", self.current_status, "#f44336", "bold"),
        ]
        
        self.id_label = None
        self.fill_label = None
        self.defect_label = None
        self.status_label = None
        
        label_attrs = ['id_label', 'fill_label', 'defect_label', 'status_label']
        
        for row, (title, value, fg_color, weight) in enumerate(labels):
            font = ("Arial", 10, weight) if weight else ("Arial", 10)
            anchor_sticky = "nw" if title == "Defect:" else "w"
            
            tk.Label(info_frame, text=title, bg="#f5f5f5", fg="black",
                    font=("Arial", 10), anchor="w").grid(row=row, column=0, sticky=anchor_sticky, pady=5)
            
            extra = {"wraplength": 280, "justify": "left"} if title == "Defect:" else {}
            value_label = tk.Label(info_frame, text=value, bg="#f5f5f5",
                                  fg=fg_color, font=font, anchor="w", **extra)
            value_label.grid(row=row, column=1, sticky="w", pady=5, padx=(10, 0))
            setattr(self, label_attrs[row], value_label)
    
    def _setup_controls_panel(self, parent):
        """start/stop/stats/export buttons"""
        controls_frame = tk.LabelFrame(parent, text="Controls", bg="#f5f5f5", 
                                      fg="black", font=("Arial", 10), relief=tk.GROOVE, bd=2)
        controls_frame.pack(fill=tk.BOTH, expand=True)
        
        button_frame = tk.Frame(controls_frame, bg="#f5f5f5")
        button_frame.pack(pady=20)
        
        # start/stop row
        control_buttons_frame = tk.Frame(button_frame, bg="#f5f5f5")
        control_buttons_frame.pack(pady=(0, 15))
        
        # use frames with labels as buttons for cross-platform colored buttons
        self.start_button, self.start_label = self._create_button(
            control_buttons_frame, "Start", "#4CAF50", side=tk.LEFT)
        self.stop_button, self.stop_label = self._create_button(
            control_buttons_frame, "Stop", "#e57373", side=tk.LEFT)
        
        # stats and export
        self.stats_button, self.stats_label = self._create_button(
            button_frame, "Stats", "#f5f5f5", fg="black", bordered=True)
        self.export_button, self.export_label = self._create_button(
            button_frame, "Export CSV", "#f5f5f5", fg="black", bordered=True)
    
    def _create_button(self, parent, text, bg, fg="white", side=None, bordered=False):
        """create a frame-based button (workaround for cross-platform colored buttons)"""
        extra = {}
        if bordered:
            extra = {"highlightbackground": "#999999", "highlightthickness": 1}
        
        frame = tk.Frame(parent, bg=bg, cursor="hand2", **extra)
        if side:
            frame.pack(side=side, padx=5)
        else:
            frame.pack(pady=5)
        
        label = tk.Label(frame, text=text, bg=bg, fg=fg,
                        font=("Arial", 11 if not bordered else 10),
                        padx=20 if not bordered else 20, pady=5 if not bordered else 3)
        label.pack()
        
        return frame, label
        
    def bind_button(self, button_frame, label, command):
        """bind click command to a frame-based button"""
        button_frame.bind("<Button-1>", lambda e: command())
        label.bind("<Button-1>", lambda e: command())
        
    def display_frame(self, frame):
        """display a video frame in the live feed panel (expects BGR from opencv)"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        scale = min(740 / w, 420 / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        
        img = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.video_label.imgtk = img
        self.video_label.configure(image=img)
                
    def update_stats(self, fps, inspected, fails):
        """update the status bar counters"""
        self.fps = fps
        self.inspected = inspected
        self.fails = fails
        self.fps_label.config(text=f"FPS: {fps:.1f}")
        self.inspected_label.config(text=f"Inspected: {inspected}")
        self.fails_label.config(text=f"Fails: {fails}")
        
    def update_current_inspection(self, bottle_id, fill, defect, status):
        """update the current inspection info panel"""
        self.id_label.config(text=bottle_id)
        self.fill_label.config(text=fill)
        self.defect_label.config(text=defect)
        self.status_label.config(text=status)
        self.status_label.config(fg="#f44336" if status == "FAIL" else "#4CAF50")
            
    def add_failure(self, bottle_id, defect_desc):
        """append a failure entry to the recent failures log"""
        self.failures_text.insert(tk.END, f"{bottle_id} - {defect_desc}\n")
        self.failures_text.see(tk.END)
    
    def show_stats(self, database):
        """open a popup window showing database statistics"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("statistics")
        stats_window.geometry("400x300")
        stats_window.configure(bg="#f5f5f5")
        
        db_stats = database.get_statistics(hours=24)
        
        text = tk.Text(stats_window, bg="white", fg="black",
                      font=("Courier", 10), padx=20, pady=20)
        text.pack(fill=tk.BOTH, expand=True)
        
        stats_text = "=== statistics (last 24 hours) ===\n\n"
        stats_text += f"total defects: {db_stats['total_defects']}\n\n"
        stats_text += "defects by type:\n"
        for defect_type, count in db_stats['defects_by_type'].items():
            stats_text += f"  {defect_type}: {count}\n"
        
        text.insert("1.0", stats_text)
        text.config(state=tk.DISABLED)
    
    def export_data(self, export_callback):
        """trigger a csv export via the provided callback"""
        try:
            export_callback()
            messagebox.showinfo("export successful", "data exported to defect_report.csv")
        except Exception as e:
            messagebox.showerror("export failed", f"error: {str(e)}")
