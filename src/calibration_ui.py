"""
Calibration UI Module
Interactive chessboard stereo calibration with Tkinter GUI.
Guides the user through capture, detection, and calibration computation.
"""

import cv2
import numpy as np
import json
import os
import time
import threading
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False


@dataclass
class CalibrationCapture:
    """A single chessboard capture from one or two cameras."""
    index: int
    timestamp: float
    cam1_frame: Optional[np.ndarray] = None
    cam2_frame: Optional[np.ndarray] = None
    cam1_corners: Optional[np.ndarray] = None
    cam2_corners: Optional[np.ndarray] = None
    cam1_found: bool = False
    cam2_found: bool = False


@dataclass
class CalibrationResult:
    """Result of stereo calibration procedure."""
    rms_error: float = 0.0
    camera_matrix_1: Optional[np.ndarray] = None
    dist_coeffs_1: Optional[np.ndarray] = None
    camera_matrix_2: Optional[np.ndarray] = None
    dist_coeffs_2: Optional[np.ndarray] = None
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None
    essential: Optional[np.ndarray] = None
    fundamental: Optional[np.ndarray] = None
    image_size: Tuple[int, int] = (1280, 720)
    calibration_id: str = ''
    captures_used: int = 0


class ChessboardDetector:
    """Detect chessboard corners in images for calibration."""

    def __init__(self, board_size: Tuple[int, int] = (9, 6),
                 square_size_mm: float = 25.0):
        """
        Args:
            board_size: Inner corners (columns, rows) of the chessboard
            square_size_mm: Physical size of each square in millimeters
        """
        self.board_size = board_size
        self.square_size_mm = square_size_mm

        # Prepare object points (3D points in real-world space)
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_mm

        # Sub-pixel refinement criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Detection flags
        self.flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                      cv2.CALIB_CB_FAST_CHECK +
                      cv2.CALIB_CB_NORMALIZE_IMAGE)

    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect chessboard corners in a frame.

        Returns:
            (found, corners) where corners is Nx1x2 float32 array or None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        found, corners = cv2.findChessboardCorners(gray, self.board_size, self.flags)

        if found and corners is not None:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

        return found, corners

    def draw_corners(self, frame: np.ndarray, corners: np.ndarray, found: bool) -> np.ndarray:
        """Draw detected corners on a frame (modifies in place)."""
        vis = frame.copy()
        cv2.drawChessboardCorners(vis, self.board_size, corners, found)
        return vis


class StereoCalibrator:
    """Compute stereo calibration from matched chessboard captures."""

    def __init__(self, board_size: Tuple[int, int] = (9, 6),
                 square_size_mm: float = 25.0):
        self.detector = ChessboardDetector(board_size, square_size_mm)
        self.captures: List[CalibrationCapture] = []
        self.result: Optional[CalibrationResult] = None

    @property
    def num_captures(self) -> int:
        return len(self.captures)

    @property
    def num_valid_pairs(self) -> int:
        return sum(1 for c in self.captures if c.cam1_found and c.cam2_found)

    def add_capture(self, cam1_frame: np.ndarray,
                    cam2_frame: Optional[np.ndarray] = None) -> CalibrationCapture:
        """
        Add a capture pair. Detects chessboard in both frames.

        Returns:
            CalibrationCapture with detection results
        """
        idx = len(self.captures)
        capture = CalibrationCapture(
            index=idx,
            timestamp=time.time()
        )

        # Camera 1
        capture.cam1_frame = cam1_frame.copy()
        found1, corners1 = self.detector.detect(cam1_frame)
        capture.cam1_found = found1
        capture.cam1_corners = corners1

        # Camera 2 (if stereo)
        if cam2_frame is not None:
            capture.cam2_frame = cam2_frame.copy()
            found2, corners2 = self.detector.detect(cam2_frame)
            capture.cam2_found = found2
            capture.cam2_corners = corners2

        self.captures.append(capture)
        return capture

    def remove_capture(self, index: int):
        """Remove a capture by index."""
        self.captures = [c for c in self.captures if c.index != index]

    def calibrate_single(self, camera: int = 1) -> Optional[CalibrationResult]:
        """
        Run single-camera calibration.

        Args:
            camera: 1 or 2

        Returns:
            CalibrationResult or None if insufficient captures
        """
        obj_points = []
        img_points = []
        image_size = None

        for cap in self.captures:
            frame = cap.cam1_frame if camera == 1 else cap.cam2_frame
            corners = cap.cam1_corners if camera == 1 else cap.cam2_corners
            found = cap.cam1_found if camera == 1 else cap.cam2_found

            if not found or corners is None or frame is None:
                continue

            obj_points.append(self.detector.objp)
            img_points.append(corners)
            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])

        if len(obj_points) < 3:
            return None

        rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )

        result = CalibrationResult(
            rms_error=rms,
            image_size=image_size,
            captures_used=len(obj_points)
        )
        if camera == 1:
            result.camera_matrix_1 = mtx
            result.dist_coeffs_1 = dist
        else:
            result.camera_matrix_2 = mtx
            result.dist_coeffs_2 = dist

        return result

    def calibrate_stereo(self) -> Optional[CalibrationResult]:
        """
        Run stereo calibration using all valid capture pairs.

        Returns:
            CalibrationResult with full stereo parameters, or None
        """
        obj_points = []
        img_points_1 = []
        img_points_2 = []
        image_size = None

        for cap in self.captures:
            if not (cap.cam1_found and cap.cam2_found):
                continue
            if cap.cam1_corners is None or cap.cam2_corners is None:
                continue

            obj_points.append(self.detector.objp)
            img_points_1.append(cap.cam1_corners)
            img_points_2.append(cap.cam2_corners)

            if image_size is None and cap.cam1_frame is not None:
                image_size = (cap.cam1_frame.shape[1], cap.cam1_frame.shape[0])

        if len(obj_points) < 5:
            print(f"[CalibrationUI] Need at least 5 valid pairs, have {len(obj_points)}")
            return None

        if image_size is None:
            image_size = (1280, 720)

        # Individual camera calibrations first
        rms1, mtx1, dist1, _, _ = cv2.calibrateCamera(
            obj_points, img_points_1, image_size, None, None
        )
        rms2, mtx2, dist2, _, _ = cv2.calibrateCamera(
            obj_points, img_points_2, image_size, None, None
        )

        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

        rms_stereo, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_1, img_points_2,
            mtx1, dist1, mtx2, dist2, image_size,
            criteria=criteria, flags=flags
        )

        cal_id = f"cal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        result = CalibrationResult(
            rms_error=rms_stereo,
            camera_matrix_1=mtx1,
            dist_coeffs_1=dist1,
            camera_matrix_2=mtx2,
            dist_coeffs_2=dist2,
            rotation=R,
            translation=T,
            essential=E,
            fundamental=F,
            image_size=image_size,
            calibration_id=cal_id,
            captures_used=len(obj_points)
        )
        self.result = result
        return result

    def save_calibration(self, filepath: str = 'calibration.json') -> bool:
        """
        Save calibration result to JSON file compatible with StereoCalibration loader.
        """
        if self.result is None:
            return False

        r = self.result
        data = {
            'calibration_id': r.calibration_id,
            'rms_error': float(r.rms_error),
            'image_size': list(r.image_size),
            'captures_used': r.captures_used,
            'timestamp': datetime.now().isoformat(),
            'cameras': {}
        }

        # Camera 1 (cam_0)
        if r.camera_matrix_1 is not None:
            data['cameras']['cam_0'] = {
                'intrinsic_matrix': r.camera_matrix_1.tolist(),
                'distortion_coefficients': r.dist_coeffs_1.tolist(),
                'rotation': np.eye(3).tolist(),  # Reference camera
                'translation': np.zeros((3, 1)).tolist(),
                'image_size': list(r.image_size)
            }

        # Camera 2 (cam_1)
        if r.camera_matrix_2 is not None and r.rotation is not None:
            data['cameras']['cam_1'] = {
                'intrinsic_matrix': r.camera_matrix_2.tolist(),
                'distortion_coefficients': r.dist_coeffs_2.tolist(),
                'rotation': r.rotation.tolist(),
                'translation': r.translation.tolist(),
                'image_size': list(r.image_size)
            }

        # Stereo pair info
        if r.essential is not None:
            data['stereo'] = {
                'essential_matrix': r.essential.tolist(),
                'fundamental_matrix': r.fundamental.tolist(),
                'baseline_mm': float(np.linalg.norm(r.translation))
            }

        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[CalibrationUI] Saved calibration to {filepath}")
            return True
        except Exception as e:
            print(f"[CalibrationUI] Save error: {e}")
            return False


class CalibrationUI:
    """
    Tkinter-based interactive calibration wizard.
    Provides live preview, capture triggering, and calibration computation.
    """

    MIN_CAPTURES = 5
    RECOMMENDED_CAPTURES = 15

    def __init__(self, parent: Optional[tk.Tk] = None,
                 board_size: Tuple[int, int] = (9, 6),
                 square_size_mm: float = 25.0,
                 output_path: str = 'calibration.json'):
        if not TK_AVAILABLE:
            raise RuntimeError("Tkinter not available")

        self.output_path = output_path
        self.calibrator = StereoCalibrator(board_size, square_size_mm)
        self._running = False
        self._capture_requested = False

        # Build UI
        if parent is None:
            self.root = tk.Tk()
            self._owns_root = True
        else:
            self.root = tk.Toplevel(parent)
            self._owns_root = False

        self.root.title("Stereo Calibration Wizard")
        self.root.geometry("700x520")
        self.root.resizable(False, False)
        self._build_ui()

    def _build_ui(self):
        """Build the calibration wizard UI."""
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(main, text="Stereo Calibration Wizard",
                  font=('Helvetica', 16, 'bold')).pack(pady=(0, 5))

        # Instructions
        instr = (
            "1. Hold a chessboard visible to both cameras\n"
            "2. Click 'Capture' or press SPACE to grab a pair\n"
            "3. Move the board to different positions and angles\n"
            "4. Capture at least 15 pairs, then click 'Calibrate'"
        )
        ttk.Label(main, text=instr, justify=tk.LEFT,
                  font=('Helvetica', 11)).pack(anchor=tk.W, pady=5)

        # Settings frame
        settings = ttk.LabelFrame(main, text="Settings", padding=5)
        settings.pack(fill=tk.X, pady=5)

        row = ttk.Frame(settings)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Board size (cols × rows):").pack(side=tk.LEFT)
        self._board_cols = tk.StringVar(value=str(self.calibrator.detector.board_size[0]))
        self._board_rows = tk.StringVar(value=str(self.calibrator.detector.board_size[1]))
        ttk.Entry(row, textvariable=self._board_cols, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="×").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._board_rows, width=4).pack(side=tk.LEFT, padx=2)

        ttk.Label(row, text="  Square (mm):").pack(side=tk.LEFT, padx=(10, 0))
        self._sq_size = tk.StringVar(
            value=str(self.calibrator.detector.square_size_mm))
        ttk.Entry(row, textvariable=self._sq_size, width=6).pack(side=tk.LEFT, padx=2)

        # Status frame
        status = ttk.LabelFrame(main, text="Status", padding=5)
        status.pack(fill=tk.X, pady=5)

        self._status_var = tk.StringVar(value="Ready. Press 'Start Preview' to begin.")
        ttk.Label(status, textvariable=self._status_var,
                  font=('Helvetica', 11)).pack(anchor=tk.W)

        self._capture_lbl = tk.StringVar(value="Captures: 0 / 0 valid pairs")
        ttk.Label(status, textvariable=self._capture_lbl).pack(anchor=tk.W)

        self._rms_lbl = tk.StringVar(value="RMS Error: —")
        ttk.Label(status, textvariable=self._rms_lbl).pack(anchor=tk.W)

        # Progress bar
        self._progress = ttk.Progressbar(main, maximum=self.RECOMMENDED_CAPTURES, mode='determinate')
        self._progress.pack(fill=tk.X, pady=5)

        # Capture log
        log_frame = ttk.LabelFrame(main, text="Capture Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self._log_text = tk.Text(log_frame, height=6, state=tk.DISABLED,
                                 font=('Courier', 10))
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                  command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scrollbar.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=5)

        self._btn_preview = ttk.Button(btn_frame, text="Start Preview",
                                       command=self._toggle_preview)
        self._btn_preview.pack(side=tk.LEFT, padx=2)

        self._btn_capture = ttk.Button(btn_frame, text="Capture (Space)",
                                       command=self._request_capture, state=tk.DISABLED)
        self._btn_capture.pack(side=tk.LEFT, padx=2)

        self._btn_calibrate = ttk.Button(btn_frame, text="Calibrate",
                                         command=self._run_calibration, state=tk.DISABLED)
        self._btn_calibrate.pack(side=tk.LEFT, padx=2)

        self._btn_save = ttk.Button(btn_frame, text="Save",
                                    command=self._save, state=tk.DISABLED)
        self._btn_save.pack(side=tk.LEFT, padx=2)

        ttk.Button(btn_frame, text="Close",
                   command=self._close).pack(side=tk.RIGHT, padx=2)

        # Keyboard bindings
        self.root.bind('<space>', lambda e: self._request_capture())
        self.root.bind('<Escape>', lambda e: self._close())

    def _log(self, msg: str):
        """Append message to log text widget."""
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + '\n')
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    def _toggle_preview(self):
        """Start/stop live preview (placeholder — camera feed driven externally)."""
        if self._running:
            self._running = False
            self._btn_preview.configure(text="Start Preview")
            self._btn_capture.configure(state=tk.DISABLED)
            self._status_var.set("Preview stopped.")
        else:
            self._running = True
            self._btn_preview.configure(text="Stop Preview")
            self._btn_capture.configure(state=tk.NORMAL)
            self._status_var.set("Preview active. Hold chessboard in view and capture.")

    def _request_capture(self):
        """Flag a capture request (polled by external video loop)."""
        if not self._running:
            return
        self._capture_requested = True

    def process_frames(self, cam1_frame: np.ndarray,
                       cam2_frame: Optional[np.ndarray] = None) -> bool:
        """
        Called by external video loop. Detects board and captures if requested.

        Returns True if a capture was stored.
        """
        if not self._capture_requested:
            return False

        self._capture_requested = False
        capture = self.calibrator.add_capture(cam1_frame, cam2_frame)

        # Update UI
        n = self.calibrator.num_captures
        nv = self.calibrator.num_valid_pairs
        self._capture_lbl.set(f"Captures: {n} / {nv} valid pairs")
        self._progress['value'] = min(nv, self.RECOMMENDED_CAPTURES)

        if capture.cam1_found and (cam2_frame is None or capture.cam2_found):
            self._log(f"#{n}: ✅ Board detected in {'both cameras' if capture.cam2_found else 'camera 1'}")
            self._status_var.set(f"Capture #{n} — board found!")
        else:
            cams_ok = []
            if capture.cam1_found:
                cams_ok.append('cam1')
            if capture.cam2_found:
                cams_ok.append('cam2')
            missing = 'cam2' if not capture.cam2_found else 'cam1'
            self._log(f"#{n}: ⚠️  Board NOT found in {missing}")
            self._status_var.set(f"Capture #{n} — board not found in {missing}")

        # Enable calibrate button when enough captures
        if nv >= self.MIN_CAPTURES:
            self._btn_calibrate.configure(state=tk.NORMAL)

        return capture.cam1_found and (cam2_frame is None or capture.cam2_found)

    def _run_calibration(self):
        """Run stereo calibration in a background thread."""
        self._status_var.set("Calibrating... please wait.")
        self._btn_calibrate.configure(state=tk.DISABLED)

        def _worker():
            result = self.calibrator.calibrate_stereo()
            self.root.after(0, lambda: self._on_calibration_done(result))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_calibration_done(self, result: Optional[CalibrationResult]):
        """Handle calibration completion."""
        if result is None:
            self._status_var.set("Calibration failed — need more valid pairs.")
            self._btn_calibrate.configure(state=tk.NORMAL)
            self._log("❌ Calibration failed")
            return

        self._rms_lbl.set(f"RMS Error: {result.rms_error:.4f} px")
        baseline = np.linalg.norm(result.translation) if result.translation is not None else 0
        self._status_var.set(
            f"Calibration complete! RMS={result.rms_error:.4f}px, "
            f"baseline={baseline:.1f}mm, {result.captures_used} pairs used"
        )
        self._log(f"✅ Stereo calibration: RMS={result.rms_error:.4f}, "
                  f"baseline={baseline:.1f}mm")
        self._btn_save.configure(state=tk.NORMAL)

    def _save(self):
        """Save calibration to file."""
        path = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[('JSON', '*.json')],
            initialfile=os.path.basename(self.output_path),
            title="Save Calibration"
        )
        if not path:
            return
        ok = self.calibrator.save_calibration(path)
        if ok:
            self._log(f"💾 Saved to {path}")
            self._status_var.set(f"Saved to {path}")
            messagebox.showinfo("Saved", f"Calibration saved to:\n{path}")
        else:
            self._log("❌ Save failed")
            messagebox.showerror("Error", "Failed to save calibration.")

    def _close(self):
        """Close the calibration UI."""
        self._running = False
        if self._owns_root:
            self.root.destroy()
        else:
            self.root.destroy()

    def show(self):
        """Show the calibration UI (blocking if owns root)."""
        if self._owns_root:
            self.root.mainloop()
