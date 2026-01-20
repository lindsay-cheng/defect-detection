import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

class InspectionDashboard:
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
        self.failures = []
        
        self.setup_ui()
        
    def setup_ui(self):
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
        
        content_frame = tk.Frame(self.root, bg="#e0e0e0")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = tk.Frame(content_frame, bg="#e0e0e0")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        feed_frame = tk.LabelFrame(left_frame, text="Live Feed", bg="#f5f5f5", 
                                  fg="black", font=("Arial", 10), relief=tk.GROOVE, bd=2)
        feed_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_label = tk.Label(feed_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
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
        
        right_frame = tk.Frame(content_frame, bg="#e0e0e0", width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        inspection_frame = tk.LabelFrame(right_frame, text="Current Inspection", 
                                        bg="#f5f5f5", fg="black", font=("Arial", 10), relief=tk.GROOVE, bd=2)
        inspection_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        info_frame = tk.Frame(inspection_frame, bg="#f5f5f5")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(info_frame, text="ID:", bg="#f5f5f5", fg="black", 
                font=("Arial", 10), anchor="w").grid(row=0, column=0, sticky="w", pady=5)
        self.id_label = tk.Label(info_frame, text=self.current_id, bg="#f5f5f5", 
                                fg="black", font=("Arial", 10, "bold"), anchor="w")
        self.id_label.grid(row=0, column=1, sticky="w", pady=5, padx=(10, 0))
        
        tk.Label(info_frame, text="Fill:", bg="#f5f5f5", fg="black", 
                font=("Arial", 10), anchor="w").grid(row=1, column=0, sticky="w", pady=5)
        self.fill_label = tk.Label(info_frame, text=self.current_fill, bg="#f5f5f5", 
                                  fg="#4CAF50", font=("Arial", 10, "bold"), anchor="w")
        self.fill_label.grid(row=1, column=1, sticky="w", pady=5, padx=(10, 0))
        
        tk.Label(info_frame, text="Defect:", bg="#f5f5f5", fg="black", 
                font=("Arial", 10), anchor="w").grid(row=2, column=0, sticky="nw", pady=5)
        self.defect_label = tk.Label(info_frame, text=self.current_defect, bg="#f5f5f5", 
                                    fg="black", font=("Arial", 10), anchor="w", 
                                    wraplength=280, justify="left")
        self.defect_label.grid(row=2, column=1, sticky="w", pady=5, padx=(10, 0))
        
        tk.Label(info_frame, text="Status:", bg="#f5f5f5", fg="black", 
                font=("Arial", 10), anchor="w").grid(row=3, column=0, sticky="w", pady=5)
        self.status_label = tk.Label(info_frame, text=self.current_status, bg="#f5f5f5", 
                                    fg="#f44336", font=("Arial", 10, "bold"), anchor="w")
        self.status_label.grid(row=3, column=1, sticky="w", pady=5, padx=(10, 0))
        
        controls_frame = tk.LabelFrame(right_frame, text="Controls", bg="#f5f5f5", 
                                      fg="black", font=("Arial", 10), relief=tk.GROOVE, bd=2)
        controls_frame.pack(fill=tk.BOTH, expand=True)
        
        button_frame = tk.Frame(controls_frame, bg="#f5f5f5")
        button_frame.pack(pady=20)
        
        control_buttons_frame = tk.Frame(button_frame, bg="#f5f5f5")
        control_buttons_frame.pack(pady=(0, 15))
        
        # use frames with labels as buttons for cross-platform colored buttons
        self.start_button = tk.Frame(control_buttons_frame, bg="#4CAF50", cursor="hand2")
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.start_label = tk.Label(self.start_button, text="Start", bg="#4CAF50", fg="white",
                                   font=("Arial", 11), padx=20, pady=5)
        self.start_label.pack()
        
        self.stop_button = tk.Frame(control_buttons_frame, bg="#e57373", cursor="hand2")
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_label = tk.Label(self.stop_button, text="Stop", bg="#e57373", fg="white",
                                  font=("Arial", 11), padx=20, pady=5)
        self.stop_label.pack()
        
        self.stats_button = tk.Frame(button_frame, bg="#f5f5f5", highlightbackground="#999999",
                                    highlightthickness=1, cursor="hand2")
        self.stats_button.pack(pady=5)
        self.stats_label = tk.Label(self.stats_button, text="Stats", bg="#f5f5f5", fg="black",
                                   font=("Arial", 10), padx=30, pady=3)
        self.stats_label.pack()
        
        self.export_button = tk.Frame(button_frame, bg="#f5f5f5", highlightbackground="#999999",
                                     highlightthickness=1, cursor="hand2")
        self.export_button.pack(pady=5)
        self.export_label = tk.Label(self.export_button, text="Export CSV", bg="#f5f5f5", fg="black",
                                    font=("Arial", 10), padx=20, pady=3)
        self.export_label.pack()
        
    def bind_button(self, button_frame, label, command):
        """bind click command to frame-based button"""
        button_frame.bind("<Button-1>", lambda e: command())
        label.bind("<Button-1>", lambda e: command())
        
    def display_frame(self, frame):
        """display a frame in the video feed (expects BGR format from opencv)
        
        args:
            frame: opencv frame in BGR format
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        max_width = 740
        max_height = 420
        scale = min(max_width/w, max_height/h)
        new_w, new_h = int(w*scale), int(h*scale)
        frame = cv2.resize(frame, (new_w, new_h))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
                
    def update_stats(self, fps, inspected, fails):
        self.fps = fps
        self.inspected = inspected
        self.fails = fails
        self.fps_label.config(text=f"FPS: {fps:.1f}")
        self.inspected_label.config(text=f"Inspected: {inspected}")
        self.fails_label.config(text=f"Fails: {fails}")
        
    def update_current_inspection(self, bottle_id, fill, defect, status):
        self.current_id = bottle_id
        self.current_fill = fill
        self.current_defect = defect
        self.current_status = status
        
        self.id_label.config(text=bottle_id)
        self.fill_label.config(text=fill)
        self.defect_label.config(text=defect)
        self.status_label.config(text=status)
        
        if status == "FAIL":
            self.status_label.config(fg="#f44336")
        else:
            self.status_label.config(fg="#4CAF50")
            
    def add_failure(self, bottle_id, defect_desc):
        """add a failure entry to the recent failures list
        
        args:
            bottle_id: id of the bottle
            defect_desc: description of the defect
        """
        failure_text = f"{bottle_id} - {defect_desc}\n"
        self.failures_text.insert(tk.END, failure_text)
        self.failures_text.see(tk.END)
    
    def show_stats(self, database):
        """show statistics window
        
        args:
            database: database instance to fetch stats from
        """
        stats_window = tk.Toplevel(self.root)
        stats_window.title("statistics")
        stats_window.geometry("400x300")
        stats_window.configure(bg="#f5f5f5")
        
        db_stats = database.get_statistics(hours=24)
        text = tk.Text(
            stats_window,
            bg="white",
            fg="black",
            font=("Courier", 10),
            padx=20,
            pady=20
        )
        text.pack(fill=tk.BOTH, expand=True)
        stats_text = "=== statistics (last 24 hours) ===\n\n"
        stats_text += f"total defects: {db_stats['total_defects']}\n\n"
        stats_text += "defects by type:\n"
        
        for defect_type, count in db_stats['defects_by_type'].items():
            stats_text += f"  {defect_type}: {count}\n"
        
        text.insert("1.0", stats_text)
        text.config(state=tk.DISABLED)
    
    def export_data(self, export_callback):
        """export data to csv using provided callback
        
        args:
            export_callback: function that performs the export
        """
        try:
            export_callback()
            messagebox.showinfo(
                "export successful",
                "data exported to defect_report.csv"
            )
        except Exception as e:
            messagebox.showerror(
                "export failed",
                f"error: {str(e)}"
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = InspectionDashboard(root)
    root.mainloop()