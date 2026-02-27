"""
Camera Setup Viewer
Visualizes camera positions and orientations in 3D space using Matplotlib.
Helps verify calibration and plan camera placement for optimal coverage.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict, Optional
from src.stereo_calibration import StereoCalibration, CameraCalibration


class CameraSetupViewer:
    """
    3D viewer showing camera positions, orientations, and field-of-view cones.
    """
    
    def __init__(self, calibration: StereoCalibration = None):
        """
        Initialize camera setup viewer.
        
        Args:
            calibration: StereoCalibration object with camera parameters
        """
        self.calibration = calibration
        self.fig = None
        self.ax = None
    
    def show(self, calibration: StereoCalibration = None):
        """
        Display 3D visualization of camera setup.
        
        Args:
            calibration: Optional override calibration (uses self.calibration if None)
        """
        cal = calibration or self.calibration
        if not cal or not cal.cameras:
            print("[CameraSetupViewer] No calibration data available")
            return
        
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("Camera Setup — 3D View", fontsize=14, fontweight='bold')
        
        colors = ['#00d4ff', '#ff6666', '#00ff88', '#ffa500', '#ff00ff', '#ffff00']
        
        for i, (cam_id, cam) in enumerate(cal.cameras.items()):
            color = colors[i % len(colors)]
            self._draw_camera(cam, color, cam_id)
        
        # Draw world origin
        self._draw_origin()
        
        # Draw subject region (approximate)
        self._draw_subject_zone()
        
        # Labels and formatting
        self.ax.set_xlabel('X (Right)', fontsize=11)
        self.ax.set_ylabel('Y (Up)', fontsize=11)
        self.ax.set_zlabel('Z (Forward)', fontsize=11)
        
        # Auto-scale
        self._auto_scale(cal)
        
        self.ax.legend(loc='upper left', fontsize=9)
        plt.tight_layout()
        plt.show(block=False)
    
    def _draw_camera(self, cam: CameraCalibration, color: str, label: str):
        """Draw a single camera with position, orientation, and FOV cone."""
        # Camera position in world space: C = -R^T @ t
        if cam.rotation is None or cam.translation is None:
            return
        
        R = cam.rotation
        t = cam.translation
        C = -R.T @ t  # Camera center in world coordinates
        
        cx, cy, cz = float(C[0]), float(C[1]), float(C[2])
        
        # Plot camera center
        self.ax.scatter([cx], [cy], [cz], c=color, s=100, marker='o',
                       edgecolors='white', linewidths=1, label=label, zorder=5)
        
        # Camera axes (columns of R^T = rows of R mapped to world)
        axis_length = 0.3
        # Camera looks along +Z in camera frame → world direction = R^T @ [0,0,1]
        cam_x = R.T @ np.array([[1], [0], [0]])  # Right
        cam_y = R.T @ np.array([[0], [1], [0]])  # Down (camera convention)
        cam_z = R.T @ np.array([[0], [0], [1]])  # Forward (optical axis)
        
        # Draw optical axis (forward direction)
        self.ax.quiver(cx, cy, cz,
                      float(cam_z[0]) * axis_length,
                      float(cam_z[1]) * axis_length,
                      float(cam_z[2]) * axis_length,
                      color=color, arrow_length_ratio=0.15, linewidth=2)
        
        # Draw FOV cone (simplified)
        self._draw_fov_cone(cam, C, R, color)
    
    def _draw_fov_cone(self, cam: CameraCalibration, C: np.ndarray,
                       R: np.ndarray, color: str, depth: float = 2.0):
        """Draw a simplified field-of-view pyramid for a camera."""
        if cam.intrinsic_matrix is None:
            return
        
        K = cam.intrinsic_matrix
        fx, fy = K[0, 0], K[1, 1]
        cx_img, cy_img = K[0, 2], K[1, 2]
        
        if cam.image_size:
            w, h = cam.image_size
        else:
            w, h = 1280, 720
        
        # Compute FOV angles
        fov_h = 2 * np.arctan(w / (2 * fx))
        fov_v = 2 * np.arctan(h / (2 * fy))
        
        # Corner rays in camera frame at distance 'depth'
        half_w = depth * np.tan(fov_h / 2)
        half_h = depth * np.tan(fov_v / 2)
        
        corners_cam = np.array([
            [-half_w, -half_h, depth],
            [ half_w, -half_h, depth],
            [ half_w,  half_h, depth],
            [-half_w,  half_h, depth]
        ]).T  # 3x4
        
        # Transform to world frame
        corners_world = R.T @ corners_cam + C  # 3x4
        
        # Draw edges from camera center to corners
        cam_pos = C.flatten()
        for i in range(4):
            corner = corners_world[:, i].flatten()
            self.ax.plot([cam_pos[0], corner[0]],
                        [cam_pos[1], corner[1]],
                        [cam_pos[2], corner[2]],
                        color=color, alpha=0.3, linewidth=1)
        
        # Draw FOV rectangle at depth
        for i in range(4):
            j = (i + 1) % 4
            ci = corners_world[:, i].flatten()
            cj = corners_world[:, j].flatten()
            self.ax.plot([ci[0], cj[0]], [ci[1], cj[1]], [ci[2], cj[2]],
                        color=color, alpha=0.4, linewidth=1)
    
    def _draw_origin(self):
        """Draw world origin with XYZ axes."""
        origin = np.array([0, 0, 0])
        length = 0.5
        
        self.ax.quiver(*origin, length, 0, 0, color='red', arrow_length_ratio=0.1,
                      linewidth=2, label='X (Right)')
        self.ax.quiver(*origin, 0, length, 0, color='green', arrow_length_ratio=0.1,
                      linewidth=2, label='Y (Up)')
        self.ax.quiver(*origin, 0, 0, length, color='blue', arrow_length_ratio=0.1,
                      linewidth=2, label='Z (Forward)')
    
    def _draw_subject_zone(self, center=(0, 0, 2), radius=0.5):
        """Draw approximate subject standing zone."""
        theta = np.linspace(0, 2 * np.pi, 30)
        x = center[0] + radius * np.cos(theta)
        z = center[2] + radius * np.sin(theta)
        y = np.full_like(x, center[1])
        
        self.ax.plot(x, y, z, color='#888888', alpha=0.4, linewidth=1, linestyle='--')
        self.ax.text(center[0], center[1] + 0.1, center[2], "Subject Zone",
                    fontsize=8, color='#888888', ha='center')
    
    def _auto_scale(self, cal: StereoCalibration):
        """Auto-scale axes based on camera positions."""
        positions = []
        for cam in cal.cameras.values():
            if cam.rotation is not None and cam.translation is not None:
                C = -cam.rotation.T @ cam.translation
                positions.append(C.flatten())
        
        if not positions:
            self.ax.set_xlim(-3, 3)
            self.ax.set_ylim(-3, 3)
            self.ax.set_zlim(-1, 5)
            return
        
        positions = np.array(positions)
        center = positions.mean(axis=0)
        spread = max(np.ptp(positions, axis=0).max(), 2.0) * 1.5
        
        self.ax.set_xlim(center[0] - spread, center[0] + spread)
        self.ax.set_ylim(center[1] - spread, center[1] + spread)
        self.ax.set_zlim(center[2] - spread, center[2] + spread)
    
    def get_camera_info(self) -> Dict[str, dict]:
        """
        Get summary info about each camera for display.
        
        Returns dict: cam_id -> {position, rotation_euler, fov_h, fov_v, baseline}
        """
        if not self.calibration:
            return {}
        
        info = {}
        positions = {}
        
        for cam_id, cam in self.calibration.cameras.items():
            if cam.rotation is None or cam.translation is None:
                continue
            
            C = -cam.rotation.T @ cam.translation
            positions[cam_id] = C.flatten()
            
            # FOV from intrinsics
            K = cam.intrinsic_matrix
            w, h = cam.image_size or (1280, 720)
            fov_h = float(np.degrees(2 * np.arctan(w / (2 * K[0, 0]))))
            fov_v = float(np.degrees(2 * np.arctan(h / (2 * K[1, 1]))))
            
            # Euler angles (approximate)
            R = cam.rotation
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            if sy > 1e-6:
                pitch = float(np.degrees(np.arctan2(-R[2, 0], sy)))
                yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                roll = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
            else:
                pitch = float(np.degrees(np.arctan2(-R[2, 0], sy)))
                yaw = 0.0
                roll = float(np.degrees(np.arctan2(-R[1, 2], R[1, 1])))
            
            info[cam_id] = {
                'position': [float(x) for x in C.flatten()],
                'euler_deg': {'pitch': pitch, 'yaw': yaw, 'roll': roll},
                'fov_h_deg': fov_h,
                'fov_v_deg': fov_v
            }
        
        # Compute baselines between camera pairs
        cam_ids = list(positions.keys())
        for i, id_a in enumerate(cam_ids):
            for id_b in cam_ids[i + 1:]:
                baseline = float(np.linalg.norm(positions[id_a] - positions[id_b]))
                info[id_a]['baseline_to'] = {id_b: baseline}
                if 'baseline_to' not in info[id_b]:
                    info[id_b]['baseline_to'] = {}
                info[id_b]['baseline_to'][id_a] = baseline
        
        return info
