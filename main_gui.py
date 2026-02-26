import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
import platform
import os
try:
    from PIL import Image as PILImage, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
try:
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.io import decode_jpeg
    TORCH_GUI_AVAILABLE = True
except Exception:
    torch = None
    TF = None
    decode_jpeg = None
    TORCH_GUI_AVAILABLE = False
from src.camera import Camera
from src.detector import MocapDetector
from src.visualizer import Visualizer
from src.database import MocapDB
from src.visualizer_3d import Visualizer3D
from src.report_generator import ReportGenerator
from src.pose_corrector import PoseCorrector
from src.calculations import Calculations
import config
from config import (
    DRAW_LANDMARKS, MULTI_CAMERA_MODE, REMOTE_CAMERA_IP,
    CUDA_ENABLED, MPS_ENABLED, DEVICE, TORCH_AVAILABLE
)
# Mac MPS + CUDA share the same tensor pipeline; only decode_jpeg differs
_GPU_AVAILABLE = (CUDA_ENABLED or MPS_ENABLED)

# Multi-camera imports (conditional)
if MULTI_CAMERA_MODE == 'server':
    from src.camera_server import CameraServer
elif MULTI_CAMERA_MODE == 'master':
    from src.master_coordinator import MasterCoordinator
    from src.triangulation import Triangulator
    from src.live_visualizer_3d import LiveVisualizer3D

class MocapGUI:
    def __init__(self):
        # Initialize components
        self.camera = Camera()
        self.detector = MocapDetector()
        self.visualizer = Visualizer()
        self.db = MocapDB() 
        self.viz_3d = Visualizer3D(self.db) 
        self.reporter = ReportGenerator(self.db) 
        self.corrector = PoseCorrector() # Init Physics Engine
        
        # Multi-camera network components
        self.network_server = None
        self.coordinator = None
        self.network_server = None
        self.coordinator = None
        self.triangulator = None
        self.live_viz = None # Real-time Matplotlib Window
        self.remote_frame = None  # Buffer for remote camera frame
        
        # Initialize network based on mode
        if MULTI_CAMERA_MODE == 'server':
            from config import NETWORK_CAMERA_ID
            self.network_server = CameraServer(NETWORK_CAMERA_ID)
            self.network_server.start()
            print("[GUI] Camera Server started - broadcasting to network")
            
        elif MULTI_CAMERA_MODE == 'master':
            if not REMOTE_CAMERA_IP:
                print("[GUI] WARNING: REMOTE_CAMERA_IP not set in config!")
            else:
                self.coordinator = MasterCoordinator(num_cameras=2)
                self.coordinator.start()
                self.coordinator.discover_cameras_manual([REMOTE_CAMERA_IP])
                # Triangulator with no calibration for now (will load when available)
                self.triangulator = Triangulator(calibration=None)
                self.live_viz = LiveVisualizer3D()
                print(f"[GUI] Master Coordinator started - connecting to {REMOTE_CAMERA_IP}")
        
        # State
        self.running = True
        self.is_recording = False
        self.session_id = None
        self.frame_count = 0

        # Tkinter camera display window (replaces cv2.imshow on macOS)
        self._cam_window = None
        self._cam_label = None
        self._cam_photo = None  # keep reference to avoid GC
        self._latest_display_frame = None
        self._latest_display_title = "Camera Feed"
        self._display_update_pending = False
        self._latest_metrics = None
        self._metrics_update_pending = False

        # Network send queue — latest-frame only to minimize streaming latency
        self._send_queue = queue.Queue(maxsize=1)
        self._send_worker_thread = threading.Thread(target=self._send_worker, daemon=True)
        self._send_worker_thread.start()

        # Remote JPEG decode queue — latest-only to keep display loop non-blocking
        self._remote_decode_queue = queue.Queue(maxsize=1)
        self._remote_decode_worker_thread = threading.Thread(target=self._remote_decode_worker, daemon=True)
        self._remote_decode_worker_thread.start()

        # Thread-safe GUI State Caches (Initialize defaults)
        self.mirror_active = True
        self.markers_active = True
        self.roi_active = False
        self.face_det_active = True
        self.hand_det_active = True
        self.gamma_val = 1.0
        self.exposure_active = False
        self.model_type = "FULL"
        
        # Kinematics State
        self.prev_lm = []
        self.prev_metrics = {}
        self.prev_time = None
        
        # Create tkinter window
        self.root = tk.Tk()
        
        # Set title based on mode
        mode_name = {
            'single': '',
            'server': ' - Camera Server (PC1)', 
            'master': ' - Master Coordinator (PC2)'
        }.get(MULTI_CAMERA_MODE, '')
        self.root.title(f"MoCap Live Dashboard{mode_name}")
        
        self.root.geometry("500x700")
        self.root.configure(bg='#0f0f1e')  # VS2 Dark Theme
        # Note: db and reporter already initialized above; do not re-init here (leaked connection)

        self.frame_count = 0 # Throttling counter
        
        
        # Setup GUI First (Important: Initialize vars before thread starts)
        self.setup_gui()
        
        # Mac OpenCV fix: Start window thread before video loop
        if platform.system() == 'Darwin':  # macOS
            cv2.startWindowThread()
        
        # Start Video Thread
        self.running = True
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        
    def _display_frame_tkinter(self, frame_bgr, title="Camera Feed"):
        """Display a BGR numpy frame in a tkinter Toplevel window (main-thread safe)."""
        if not PIL_AVAILABLE:
            return
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = PILImage.fromarray(rgb)
            photo = ImageTk.PhotoImage(image=img)

            if self._cam_window is None or not self._cam_window.winfo_exists():
                self._cam_window = tk.Toplevel(self.root)
                self._cam_window.title(title)
                self._cam_window.configure(bg='black')
                self._cam_label = tk.Label(self._cam_window, bg='black')
                self._cam_label.pack()

            self._cam_window.title(title)
            self._cam_label.configure(image=photo)
            self._cam_photo = photo  # prevent GC
        except Exception as e:
            if self.frame_count % 100 == 0:
                print(f"[Display] Frame render error: {e}")

    def _flush_display_tkinter(self):
        """Render only the latest queued frame on the Tk main thread."""
        self._display_update_pending = False
        frame = self._latest_display_frame
        title = self._latest_display_title
        if frame is not None:
            self._display_frame_tkinter(frame, title)

    def _schedule_display_tkinter(self, frame_bgr, title="Camera Feed"):
        """Queue newest frame for display; never let Tkinter callback backlog build."""
        self._latest_display_frame = frame_bgr
        self._latest_display_title = title
        if not self._display_update_pending:
            self._display_update_pending = True
            self.root.after(0, self._flush_display_tkinter)

    def _flush_metrics_gui(self):
        """Apply latest metrics update on Tk main thread."""
        self._metrics_update_pending = False
        metrics = self._latest_metrics
        if metrics is not None:
            self.update_metrics_gui(metrics)

    def _schedule_metrics_gui(self, metrics):
        """Coalesce metrics updates so GUI never backlogs."""
        self._latest_metrics = metrics
        if not self._metrics_update_pending:
            self._metrics_update_pending = True
            self.root.after(0, self._flush_metrics_gui)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def setup_gui(self):
        # VS2 Style Styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview", 
                        background="#1a1a2e",
                        foreground="#e0e0e0",
                        fieldbackground="#1a1a2e")
        style.configure("Treeview.Heading", 
                        background="#0f0f1e",
                        foreground="#00d4ff",
                        font=("Arial", 10, "bold"))
                        
        # --- SCROLLABLE CONTAINER ---
        # Create a container frame
        self.container = tk.Frame(self.root, bg='#0f0f1e')
        self.container.pack(fill="both", expand=True)

        # Create canvas
        self.canvas = tk.Canvas(self.container, bg='#0f0f1e', highlightthickness=0)
        
        # Create scrollbar
        self.scrollbar = ttk.Scrollbar(self.container, orient="vertical", command=self.canvas.yview)
        
        # Create scrollable frame INSIDE canvas
        self.scrollable_frame = tk.Frame(self.canvas, bg='#0f0f1e')

        # Configure scrollable frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Draw frame on canvas
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=480) # Width slightly less than window

        # Configure canvas
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack scrollbar and canvas
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Bind MouseWheel for scrolling
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # --- WIDGET PARENT CHANGED TO self.scrollable_frame ---
        parent = self.scrollable_frame
        
        # Header
        header_frame = tk.Frame(parent, bg='#0f0f1e')
        header_frame.pack(fill=tk.X, pady=20)
        
        tk.Label(header_frame, text="🎬 MoCap Live Dashboard", 
                 font=("Arial", 22, "bold"),
                 bg='#0f0f1e', fg='#00d4ff').pack(side=tk.LEFT, padx=20)
                 
        # Model Selector (Top Right)
        model_frame = tk.Frame(header_frame, bg='#0f0f1e')
        model_frame.pack(side=tk.RIGHT, padx=20)
        
        tk.Label(model_frame, text="Model:", bg='#0f0f1e', fg='#aaa', font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="FULL")
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                     values=["LITE", "FULL", "HEAVY"], 
                     state="readonly", width=8)
        self.model_combo.pack(side=tk.LEFT)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)
        
        # Status Bar
        status_frame = tk.Frame(parent, bg='#1a1a2e', padx=10, pady=5)
        status_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.status_label = tk.Label(status_frame, text="● READY", 
                                     font=("Arial", 12, "bold"),
                                     bg='#1a1a2e', fg='#00ff88')
        self.status_label.pack(side=tk.LEFT)
        
        self.fps_label = tk.Label(status_frame, text="FPS: 00.0", 
                                 font=("Courier", 12),
                                 bg='#1a1a2e', fg='#e0e0e0')
        self.fps_label.pack(side=tk.RIGHT)
        
        # Live Data Table (Matches VS2 Data Preview)
        tk.Label(parent, text="📊 Live Data Feed ( Angles )", 
                 font=("Arial", 12), bg='#0f0f1e', fg='#F5F5DC').pack(anchor=tk.W, padx=20, pady=(20,5))
        
        # Columns: Time + Key Angles + Velocities
        cols = ("Time", "Angle_L_Shoulder", "Angle_R_Shoulder", "Angle_L_Elbow", "Angle_R_Elbow", "Angle_L_Knee", "Angle_R_Knee", "Vel_Elbow", "Vel_Wrist")
        self.tree = ttk.Treeview(parent, columns=cols, show='headings', height=8)
        
        self.tree.heading("Time", text="Time")
        self.tree.column("Time", width=60, anchor=tk.CENTER)
        
        headers = ["L Shldr", "R Shldr", "L Elbow", "R Elbow", "L Knee", "R Knee", "V Elb", "V Wri"]
        for col, title in zip(cols[1:], headers):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=50, anchor=tk.CENTER)
            
        self.tree.pack(padx=20, fill=tk.X)
        

        
        # Live Biometrics Panel
        metrics_frame = tk.LabelFrame(parent, text="Live Biometrics (Degrees)", 
                                    bg='#0f0f1e', fg='#00d4ff', font=("Arial", 12, "bold"))
        metrics_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.angle_labels = {}
        # Format: (Label Text, Metric Key, Unit)
        metric_keys = [
            # Angles
            ("L Elbow", "Angle_Elbow_L", "°"), ("R Elbow", "Angle_Elbow_R", "°"),
            ("L Knee", "Angle_Knee_L", "°"),   ("R Knee", "Angle_Knee_R", "°"),
            ("L Shoulder", "Angle_Shoulder_L", "°"), ("R Shoulder", "Angle_Shoulder_R", "°"),
            # Lengths
            ("L Arm", "Length_UpperArm_L", ""), ("R Arm", "Length_UpperArm_R", ""),
            ("L Leg", "Length_UpperLeg_L", ""), ("R Leg", "Length_UpperLeg_R", "")
        ]
        
        for i, (label_text, key, unit) in enumerate(metric_keys):
            row = i // 2
            col = i % 2
            
            # Label
            tk.Label(metrics_frame, text=label_text, bg='#0f0f1e', fg='#e0e0e0', font=("Arial", 10)).grid(row=row, column=col*2, padx=10, pady=5, sticky="e")
            
            # Value
            val_label = tk.Label(metrics_frame, text="0.0", bg='#0f0f1e', fg='#00ff88', font=("Courier", 12, "bold"))
            val_label.grid(row=row, column=col*2+1, padx=10, pady=5, sticky="w")
            
            # Save widget and unit for updating
            self.angle_labels[key] = (val_label, unit)
            
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.columnconfigure(1, weight=1)
        metrics_frame.columnconfigure(2, weight=1)
        metrics_frame.columnconfigure(3, weight=1)
        
        # Controls
        control_frame = tk.Frame(parent, bg='#0f0f1e')
        control_frame.pack(pady=10)
        
        # Toggles
        toggle_frame = tk.Frame(parent, bg='#0f0f1e')
        toggle_frame.pack(pady=5)
        
        self.mirror_var = tk.BooleanVar(value=True) # Default Mirror: ON
        tk.Checkbutton(toggle_frame, text="Mirror Camera", variable=self.mirror_var,
                       bg='#0f0f1e', fg='#e0e0e0', selectcolor='#0f0f1e',
                       activebackground='#0f0f1e', activeforeground='#e0e0e0',
                       command=self.update_toggles).pack(side=tk.LEFT, padx=10)
                       
        self.markers_var = tk.BooleanVar(value=True) # Default Markers: ON
        tk.Checkbutton(toggle_frame, text="Show Markers", variable=self.markers_var,
                       bg='#0f0f1e', fg='#e0e0e0', selectcolor='#0f0f1e',
                       activebackground='#0f0f1e', activeforeground='#e0e0e0',
                       command=self.update_toggles).pack(side=tk.LEFT, padx=10)
        
        self.record_btn = tk.Button(control_frame, text="▶ Start Capture",
                                    command=self.toggle_recording,
                                    font=("Arial", 14, "bold"),
                                    bg='#1a1a2e', fg='#00ff88',
                                    activebackground='#00d4ff',
                                    width=18, bd=0, relief=tk.FLAT)
        self.record_btn.pack(pady=10)

        # Imaging Controls (VS5)
        # Gamma, Exposure, Toggles
        image_frame = tk.LabelFrame(parent, text="Imaging & AI", 
                                    bg='#0f0f1e', fg='#00d4ff', font=("Arial", 10, "bold"))
        image_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # Gamma
        tk.Label(image_frame, text="Gamma:", bg='#0f0f1e', fg='#aaa').grid(row=0, column=0, padx=5, sticky='e')
        self.gamma_var = tk.DoubleVar(value=1.0)
        gamma_scale = tk.Scale(image_frame, from_=1.0, to=1.4, resolution=0.1, 
                               variable=self.gamma_var, orient=tk.HORIZONTAL, 
                               bg='#0f0f1e', fg='#e0e0e0', highlightthickness=0,
                               command=self.update_imaging)
        gamma_scale.grid(row=0, column=1, sticky='w')
        
        # Features
        self.exposure_var = tk.BooleanVar(value=False)
        tk.Checkbutton(image_frame, text="Face Exposure", variable=self.exposure_var,
                       bg='#0f0f1e', fg='#e0e0e0', selectcolor='#0f0f1e',
                       activebackground='#0f0f1e', activeforeground='#e0e0e0',
                       command=self.update_imaging).grid(row=1, column=0, columnspan=2, sticky='w', padx=5)
        
        self.roi_var = tk.BooleanVar(value=False)
        tk.Checkbutton(image_frame, text="ROI Cropping", variable=self.roi_var,
                       bg='#0f0f1e', fg='#e0e0e0', selectcolor='#0f0f1e',
                       activebackground='#0f0f1e', activeforeground='#e0e0e0',
                       command=self.update_imaging).grid(row=2, column=0, columnspan=2, sticky='w', padx=5)
                       
        self.face_det_var = tk.BooleanVar(value=True)
        tk.Checkbutton(image_frame, text="Face Detect", variable=self.face_det_var,
                       bg='#0f0f1e', fg='#e0e0e0', selectcolor='#0f0f1e',
                       activebackground='#0f0f1e', activeforeground='#e0e0e0',
                       command=self.update_imaging).grid(row=3, column=0, sticky='w', padx=5)
                       
        self.hand_det_var = tk.BooleanVar(value=True)
        tk.Checkbutton(image_frame, text="Hand Detect", variable=self.hand_det_var,
                       bg='#0f0f1e', fg='#e0e0e0', selectcolor='#0f0f1e',
                       activebackground='#0f0f1e', activeforeground='#e0e0e0',
                       command=self.update_imaging).grid(row=3, column=1, sticky='w', padx=5)

        self.download_btn = tk.Button(control_frame, text="📥 Download Dataset",
                                      command=self.download_data,
                                      font=("Arial", 12),
                                      bg='#1a1a2e', fg='#e0e0e0',
                                      width=18, bd=0, relief=tk.FLAT)
        self.download_btn.pack(pady=5)
        
        self.viz_btn = tk.Button(control_frame, text="📊 Visualize Session",
                                      command=self.show_visualization,
                                      font=("Arial", 12),
                                      bg='#1a1a2e', fg='#00d4ff', # Cyan for Viz
                                      width=18, bd=0, relief=tk.FLAT)
        self.viz_btn.pack(pady=5)

        if MULTI_CAMERA_MODE == 'master':
             self.live_btn = tk.Button(control_frame, text="Current Live 3D",
                                          command=self.toggle_live_3d,
                                          font=("Arial", 12),
                                          bg='#1a1a2e', fg='#ffa500', # Orange
                                          width=18, bd=0, relief=tk.FLAT)
             self.live_btn.pack(pady=5)
        
        # Initial attribute sync
        self.update_toggles()
        self.update_imaging()
        


    def update_toggles(self):
        """Cache simple toggles for thread safety."""
        try:
            self.mirror_active = self.mirror_var.get()
            self.markers_active = self.markers_var.get()
        except: pass

    def update_imaging(self, _=None):
        """Update detector imaging params from GUI."""
        try:
            # Cache values for thread safety check (optional, but good practice)
            self.gamma_val = self.gamma_var.get()
            self.exposure_active = self.exposure_var.get()
            self.face_det_active = self.face_det_var.get()
            self.hand_det_active = self.hand_det_var.get()
            self.roi_active = self.roi_var.get()
            
            self.detector.set_imaging_params(
                gamma=self.gamma_val,
                face_exposure=self.exposure_active,
                enable_face=self.face_det_active,
                enable_hand=self.hand_det_active,
                enable_roi=self.roi_active
            )
        except Exception as e:
            print(f"Error updating imaging: {e}")

    def on_model_change(self, event):
        """Handle model complexity change."""
        try:
            model_type = self.model_var.get()
            self.detector.reload(model_type)
        except Exception as e:
            print(f"Error reloading model: {e}")

    def show_visualization(self):
        """Callback to launch Session Dashboard (HTML)."""
        try:
            # 1. Open Interactive Dashboard (HTML)
            self.viz_3d.plot_latest_session()
            
            # 2. Generate Static Report (Images) in Background
            path = self.reporter.generate_report()
            if path:
                messagebox.showinfo("Report Ready", f"Full analysis saved to:\n{path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Viz failed: {e}")

    def toggle_live_3d(self):
        """Enable/Disable Live Matplotlib Window."""
        if self.live_viz:
             if self.live_viz.initialized:
                 self.live_viz.close()
             else:
                 self.live_viz.init_plot()

    def toggle_recording(self):
        if not self.is_recording:
            try:
                self.session_id = self.db.start_recording()
                self.is_recording = True
                
                # Reset Tree
                for item in self.tree.get_children():
                    self.tree.delete(item)
                    
                self.status_label.config(text="● RECORDING", fg='#ff0055') # VS2 Red
                self.record_btn.config(text="⏹ Stop Capture", fg='#ff0055')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start recording: {e}")
        else:
            self.db.stop_recording()
            self.is_recording = False
            self.status_label.config(text="● READY", fg='#00ff88')
            self.record_btn.config(text="▶ Start Capture", fg='#00ff88')
            
            messagebox.showinfo("Session Complete", 
                              f"Recording saved.\nSession ID: {self.session_id[:8]}...")

    def update_table(self, data):
        """Update the table with new frame data (VS2 style live feed)."""
        pass # Not used directly, using update_table_safe

    def download_data(self):
        csv_content = self.db.export_latest_session_csv()
        if not csv_content:
            messagebox.showwarning("No Data", "No recordings found to export")
            return
        
        filename = f"mocap_{int(time.time())}.csv"
        with open(filename, 'w') as f:
            f.write(csv_content)
        messagebox.showinfo("Export Complete", f"Dataset saved to {filename}")

    def _send_worker(self):
        """Single background thread that drains the network send queue.
        Replaces the per-frame threading.Thread spawn which caused thread buildup."""
        while self.running:
            try:
                item = self._send_queue.get(timeout=0.5)
                if item is None:
                    break
                frame_number, timestamp, results, frame_copy = item
                self.network_server.send_frame_data(frame_number, timestamp, results, frame_copy)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[SERVER] Send worker error: {e}")

    def _remote_decode_worker(self):
        """Background worker that decodes remote JPEG frames without blocking video_loop."""
        while self.running:
            try:
                item = self._remote_decode_queue.get(timeout=0.5)
                if item is None:
                    break

                jpeg, display_width, display_height = item
                if not jpeg:
                    continue

                jpg_np = np.frombuffer(bytes(jpeg), dtype=np.uint8)
                decoded = cv2.imdecode(jpg_np, cv2.IMREAD_COLOR)
                if decoded is not None:
                    self.remote_frame = cv2.resize(decoded, (display_width, display_height))
            except queue.Empty:
                continue
            except Exception as e:
                if self.frame_count % 100 == 0:
                    print(f"[Display] Remote decode error: {e}")

    def video_loop(self):
        while self.running:
            frame = self.camera.read()
            if frame is None:
                continue
            
            # 1. Mirroring (Horizontal Flip)
            if self.mirror_active:
                frame = cv2.flip(frame, 1)
            
            # Process & Save
            # Note: If mirrored, landmarks will be mirrored too logic-wise
            results = self.detector.process(frame)
            
            # --- PHYSICS CORRECTION ---
            # Corrects bone lengths and smooths jitter
            results = self.corrector.process(results)
            # --------------------------
            
            # --- CALCULATE METRICS ---
            metrics = {}
            if results.get('pose') and results.get('pose').pose_landmarks:
                try:
                    # Prioritize World Landmarks for Physics/Metrics (Meters)
                    if results.get('pose').pose_world_landmarks:
                         plm = results.get('pose').pose_world_landmarks[0]
                    else:
                         plm = results.get('pose').pose_landmarks[0]
                         
                    lm_list = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'v': lm.visibility} for lm in plm]
                    
                    # Core
                    body_metrics = Calculations.get_body_metrics(lm_list)
                    
                    # Face Metrics
                    face_metrics = {}
                    if results.get('face') and results.get('face').face_landmarks:
                        flm = results.get('face').face_landmarks[0]
                        flm_dict = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in flm]
                        face_metrics = Calculations.get_face_metrics(flm_dict)
                        
                    raw_metrics = {**body_metrics, **face_metrics}
                    
                    # --- ADVANCED PHYSICS (Normalization & Filter) ---
                    # 1. Normalize Lengths (Height-independent)
                    norm_metrics = Calculations.normalize_metrics(raw_metrics, lm_list)
                    
                    # 2. Smooth & Reject Outliers (Temporal)
                    metrics = Calculations.filter_and_smooth(norm_metrics, self.prev_metrics)
                    # -------------------------------------------------
                    
                    # Kinematics
                    now = time.time()
                    if self.prev_time is not None:
                        dt = now - self.prev_time
                        if dt > 0:
                            kinematics = Calculations.get_kinematics(lm_list, self.prev_lm, metrics, self.prev_metrics, dt)
                            metrics.update(kinematics)
                    
                    # Update State
                    self.prev_lm = lm_list
                    self.prev_metrics = metrics
                    self.prev_time = now

                    if self.frame_count % 2 == 0:
                        self._schedule_metrics_gui(metrics)
                except: pass
            # -------------------------
            
            # --- DRAW VISUALIZATION FIRST (for network transmission) ---
            if self.markers_active and results:
                frame = self.visualizer.draw_landmarks(frame, results)
            # -----------------------------------------------------------
            
            # --- NETWORK BROADCASTING (Server Mode) - via send queue ---
            if self.network_server:
                timestamp = int(time.time() * 1e9)
                try:
                    # Non-blocking enqueue
                    self._send_queue.put_nowait((self.frame_count, timestamp, results, frame.copy()))
                except queue.Full:
                    # Queue already has a stale frame waiting to be encoded/sent; replace with newest
                    try:
                        self._send_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._send_queue.put_nowait((self.frame_count, timestamp, results, frame.copy()))
                    except queue.Full:
                        pass

                # Debug only first 3 frames
                if self.frame_count <= 3:
                    print(f"[SERVER] Queued frame {self.frame_count}")
            # -----------------------------------------
            
            # --- RECEIVE REMOTE CAMERA (Master Mode) ---
            if self.coordinator:
                # Master mode: Add our LOCAL camera to sync buffer too!
                # The coordinator receives PC2's frames automatically,
                # but we need to add PC1's local frames manually
                # Use wall-clock time (epoch nanoseconds) for cross-machine sync
                timestamp = int(time.time() * 1e9)
                
                # Create a local frame data entry
                from src.master_coordinator import FrameData
                local_frame_data = FrameData(
                    camera_id='local_cam',
                    frame_number=self.frame_count,
                    timestamp=timestamp,
                    results=results,
                    received_at=timestamp  # Add received_at (same as timestamp for local)
                )
                
                # Add to coordinator's buffer manually
                if hasattr(self.coordinator, 'frame_buffers'):
                    # Use frame_buffers (plural) - it's a dict of deques
                    self.coordinator.frame_buffers['local_cam'].append(local_frame_data)
                    
                    # Debug only on errors
                    pass
                else:
                    if self.frame_count <= 3:
                        print(f"[MASTER ERROR] Coordinator has no frame_buffers attribute!")
            # -----------------------------------------
            
            if self.is_recording:
                # Save first - BUT ONLY if not in master mode (master saves synced batches)
                if MULTI_CAMERA_MODE != 'master':
                    self.db.save_frame(results)
                
                # Update GUI safely (Throttled)
                if self.frame_count % 5 == 0:
                     self.root.after(0, self.update_table_safe, results)
                     
            self.frame_count += 1

            # Draw FPS overlay (landmarks already drawn above before network send)
            frame = self.visualizer.draw_fps(frame)
            fps = self.visualizer.get_fps()
            
            # Update FPS label
            try:
                self.fps_label.config(text=f"FPS: {fps:04.1f}")
            except: pass 
            
            # --- DUAL CAMERA DISPLAY (Master Mode) ---
            # Display uses CPU only — MediaPipe already uses Metal for inference.
            # GPU tensor path was causing 3GB/s MPS memory leak (never freed per-frame tensors).
            if MULTI_CAMERA_MODE == 'master' and self.coordinator:
                display_width, display_height = 640, 480
                sync_label = "WAITING"

                # 1. Local frame — CPU resize
                local_display = cv2.resize(frame, (display_width, display_height))

                # 2. Remote frame — CPU JPEG decode + resize
                remote_display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
                if 'cam_0' in self.coordinator.frame_buffers:
                    buf = self.coordinator.frame_buffers['cam_0']
                    if len(buf) > 0:
                        latest = buf[-1]
                        jpeg = latest.results.get('frame_jpeg')
                        if jpeg:
                            try:
                                self._remote_decode_queue.put_nowait((jpeg, display_width, display_height))
                            except queue.Full:
                                try:
                                    self._remote_decode_queue.get_nowait()
                                except queue.Empty:
                                    pass
                                try:
                                    self._remote_decode_queue.put_nowait((jpeg, display_width, display_height))
                                except queue.Full:
                                    pass

                        if self.remote_frame is not None:
                            remote_display = self.remote_frame
                            sync_label = "LIVE" if jpeg else "BUFFERED"
                    # Debug: print every 5s when no frames arriving
                    if self.frame_count % 150 == 0 and sync_label == "WAITING":
                        total = sum(self.coordinator.stats['frames_received'].values())
                        print(f"[Display] cam_0 buf size={len(buf)}, total_rx={total} — check Windows firewall (ports 6000,6001)")

                # 3. Combine side-by-side and push to tkinter window
                cv2.putText(local_display, f"LOCAL-MAC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(remote_display, f"REMOTE-WIN ({sync_label})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                combined = np.hstack([local_display, remote_display])
                self._schedule_display_tkinter(combined, "Dual Camera — Master")
                
                # --- DISPLAY SYNC STATUS & 3D CALC ---
                # Check sync status and RUN 3D TRIANGULATION
                synced_batch = self.coordinator.get_synchronized_batch()
                
                # DEBUG SYNC FAILURE (Conditional print)
                if not synced_batch and self.frame_count % 30 == 0:
                     if 'cam_0' in self.coordinator.frame_buffers and 'local_cam' in self.coordinator.frame_buffers:
                         c0_buf = self.coordinator.frame_buffers['cam_0']
                         lc_buf = self.coordinator.frame_buffers['local_cam']
                         if len(c0_buf) > 0 and len(lc_buf) > 0:
                             t_remote = c0_buf[-1].timestamp
                             t_local = lc_buf[-1].timestamp
                             diff_ms = abs(t_remote - t_local) / 1e6
                             print(f"[Sync Debug] Latest Frame Delta: {diff_ms:.1f}ms (Threshold: {config.SYNC_TIME_THRESHOLD_MS}ms)")

                if synced_batch and len(synced_batch) >= 2:
                    sync_label = "SYNCED ✓"
                    
                    # Compute 3D Pose
                    pose_3d = self.coordinator.get_synced_3d_pose(synced_batch)
                    
                    # SAVE SYNCHRONIZED DATA (Master Mode Recording)
                    if self.is_recording:
                         pc1_res = next((f.results for f in synced_batch if f.camera_id == 'local_cam'), None)
                         pc2_res = next((f.results for f in synced_batch if f.camera_id == 'cam_0'), None)
                         self.db.save_synced_frame(time.time(), pc1_res, pc2_res, pose_3d)

                    if pose_3d:
                        if self.frame_count % 100 == 0:
                            print(f"✅ 3D Pose Computed! {len(pose_3d.get('pose_3d',[]))} landmarks")
                        if self.live_viz and self.live_viz.initialized:
                             try: self.live_viz.update(pose_3d)
                             except: pass

                # FINAL GPU DISPLAY (or CPU Fallback defined earlier)
                # Note: Labels are already applied in the GPU/CPU blocks above
            else:
                # Single camera or server mode — show in tkinter on Mac, cv2 on Windows
                if platform.system() == 'Darwin':
                    _frame_copy = frame.copy()
                    self._schedule_display_tkinter(_frame_copy, "MoCap Live Feed")
                else:
                    cv2.imshow("MoCap Live Feed", frame)

            # cv2.waitKey crashes on macOS when called from a background thread
            if platform.system() != 'Darwin':
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
        
        self.cleanup()

    def update_table_safe(self, results):
        # Update Table with Key Angles (Matches Headers)
        row_values = ["-"] * 8 
        
        if results.get('pose') and results.get('pose').pose_landmarks:
            try:
                # Prioritize World Landmarks for Physics/Metrics (Meters)
                if results.get('pose').pose_world_landmarks:
                     plm = results.get('pose').pose_world_landmarks[0]
                else:
                     plm = results.get('pose').pose_landmarks[0]
                     
                lm_list = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'v': lm.visibility} for lm in plm]
                # Note: Metrics are calculated in video_loop and passed via self.latest_metrics or similar mechanism usually.
                # However, here we are recalculating or need access to the *latest* metrics which contains velocity.
                # Since video_loop calls update_metrics_gui with full metrics, but update_table_safe only gets 'results' (Pose),
                # we are missing the 'kinematics' which depends on state.
                # Accessing self.prev_metrics is the best way since video_loop updates it before calling this (mostly).
                # Actually, video_loop calls update_metrics_gui, then saves frame, then calls update_table_safe.
                # So self.prev_metrics should have the LATEST calculation including kinematics.
                metrics = self.prev_metrics 
                
                # Headers: ["L Shldr", "R Shldr", "L Elbow", "R Elbow", "L Knee", "R Knee", "V Elb", "V Wri"]
                keys = ["Angle_Shoulder_L", "Angle_Shoulder_R", 
                        "Angle_Elbow_L", "Angle_Elbow_R", 
                        "Angle_Knee_L", "Angle_Knee_R",
                        "Velocity_Angle_Elbow_R", "Velocity_Wrist_R"]

                for i, key in enumerate(keys):
                    if key in metrics:
                        row_values[i] = f"{metrics[key]:.1f}"
            except: pass

        ts = time.strftime("%H:%M:%S")
        try:
            self.tree.insert("", 0, values=(ts, *row_values))
        except: pass
        
        children = self.tree.get_children()
        if len(children) > 10:
            self.tree.delete(children[-1])

    def run(self):
        """Start the application (blocking)."""
        # Thread is already started in __init__
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.running = False
            self.cleanup()

    def cleanup(self):
        self.running = False
        # Signal the send worker thread to exit gracefully
        try:
            self._send_queue.put_nowait(None)
        except Exception:
            pass
        if hasattr(self, '_send_worker_thread'):
            self._send_worker_thread.join(timeout=2.0)
        # Signal remote decode worker to exit gracefully
        try:
            self._remote_decode_queue.put_nowait(None)
        except Exception:
            pass
        if hasattr(self, '_remote_decode_worker_thread'):
            self._remote_decode_worker_thread.join(timeout=2.0)
        if self.camera: self.camera.release()
        if self.db: self.db.stop_recording()
        if self.network_server: self.network_server.stop()
        if self.coordinator: self.coordinator.stop()
        cv2.destroyAllWindows()
        if self.root: self.root.quit()

    def update_metrics_gui(self, metrics):
        """Update angle labels with latest metrics."""
        for key, (label_widget, unit) in self.angle_labels.items():
            if key in metrics:
                val = metrics[key]
                # angles are usually > 1, lengths are 0-1 (normalized)
                # But lengths from calculations are Euclidean distance of normalized coords.
                # Let's show 2 decimal places for better precision.
                label_widget.config(text=f"{val:.2f}{unit}")
            else:
                label_widget.config(text="0.0")


if __name__ == "__main__":
    app = MocapGUI()
    app.run()
