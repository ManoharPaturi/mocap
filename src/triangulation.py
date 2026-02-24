"""
Triangulation Module
Computes 3D world coordinates from 2D landmarks across multiple camera views.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from src.stereo_calibration import StereoCalibration, CameraCalibration
from config import (
    TRIANGULATION_MIN_VIEWS, REPROJECTION_ERROR_THRESHOLD,
    CONFIDENCE_WEIGHT_VISIBILITY, CONFIDENCE_WEIGHT_REPROJ,
    CUDA_ENABLED, DEVICE
)


@dataclass
class Point3D:
    """3D point in world coordinates."""
    x: float
    y: float
    z: float
    confidence: float = 1.0
    reprojection_error: float = 0.0
    num_views: int = 0


@dataclass
class Landmark3D:
    """3D landmark with metadata."""
    position: Point3D
    landmark_id: int
    visibility_scores: Dict[str, float]  # Per-camera visibility


class Triangulator:
    """
    Triangulates 3D points from multi-view 2D observations.
    Uses Direct Linear Transform (DLT) for triangulation.
    """
    
    def __init__(self, calibration: StereoCalibration = None):
        """
        Initialize triangulator with camera calibration.
        
        Args:
            calibration: StereoCalibration object with camera parameters (optional)
        """
        self.calibration = calibration
        if calibration is not None:
            print(f"[Triangulator] Initialized with {len(calibration.cameras)} cameras")
        else:
            print("[Triangulator] Initialized without calibration (will load later)")
    
    def triangulate_point(
        self,
        observations: Dict[str, np.ndarray],
        visibility_weights: Dict[str, float] = None,
        undistort: bool = True
    ) -> Optional[Point3D]:
        """
        Triangulate a single 3D point from multiple 2D observations.
        
        Args:
            observations: Dict mapping camera_id to 2D point (x, y)
            visibility_weights: Weights per camera (for weighted DLT)
            undistort: Whether to undistort points before triangulation
            
        Returns:
            Point3D object if successful, None otherwise
        """
        if len(observations) < TRIANGULATION_MIN_VIEWS:
            return None
        
        # Prepare projection matrices and points
        projection_matrices = []
        points_2d = []
        camera_ids = []
        
        for camera_id, point_2d in observations.items():
            if camera_id not in self.calibration.cameras:
                continue
            
            # Undistort point if requested
            if undistort:
                point_2d = self.calibration.undistort_point(camera_id, point_2d)[0]
            
            # Get projection matrix
            P = self.calibration.get_projection_matrix(camera_id)
            
            projection_matrices.append(P)
            points_2d.append(point_2d)
            camera_ids.append(camera_id)
        
        if len(projection_matrices) < TRIANGULATION_MIN_VIEWS:
            return None
        
        # Triangulate using DLT (PyTorch CUDA accelerated if enabled)
        point_3d_homogeneous = self._triangulate_dlt(projection_matrices, points_2d, visibility_weights, camera_ids)
        
        # Convert from homogeneous to 3D coordinates
        point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]
        
        # Calculate reprojection error
        reproj_error = self._calculate_reprojection_error(
            point_3d, projection_matrices, points_2d
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            reproj_error.item() if hasattr(reproj_error, 'item') else reproj_error,
            visibility_scores=list(visibility_weights.values()) if visibility_weights else None,
            num_views=len(observations)
        )
        
        # Check if reprojection error is acceptable
        if reproj_error > REPROJECTION_ERROR_THRESHOLD:
            return None
        
        return Point3D(
            x=float(point_3d[0]),
            y=float(point_3d[1]),
            z=float(point_3d[2]),
            confidence=float(confidence),
            reprojection_error=float(reproj_error),
            num_views=len(observations)
        )
    
    def triangulate_pose(
        self,
        multi_view_landmarks: Dict[str, List[Dict]],
        undistort: bool = True
    ) -> List[Landmark3D]:
        """
        Triangulate full pose from multi-view 2D landmarks.
        
        Args:
            multi_view_landmarks: Dict mapping camera_id to list of landmark dicts
                                 Each landmark has 'x', 'y', 'visibility' keys
            undistort: Whether to undistort points
            
        Returns:
            List of Landmark3D objects (one per body landmark)
        """
        # Determine number of landmarks (assume all cameras detect same number)
        num_landmarks = max(len(landmarks) for landmarks in multi_view_landmarks.values())
        
        landmarks_3d = []
        
        for landmark_idx in range(num_landmarks):
            # Collect observations for this landmark across all cameras
            observations = {}
            visibility_scores = {}
            
            for camera_id, landmarks in multi_view_landmarks.items():
                if landmark_idx >= len(landmarks):
                    continue
                
                landmark = landmarks[landmark_idx]
                visibility = landmark.get('visibility', 1.0)
                
                # Only use landmarks with sufficient visibility
                if visibility > 0.5:
                    point_2d = np.array([landmark['x'], landmark['y']], dtype=np.float32)
                    observations[camera_id] = point_2d
                    visibility_scores[camera_id] = visibility
            
            # Triangulate this landmark
            point_3d = self.triangulate_point(observations, undistort=undistort)
            
            if point_3d:
                # Update confidence with visibility information
                avg_visibility = np.mean(list(visibility_scores.values()))
                point_3d.confidence = self._calculate_confidence(
                    point_3d.reprojection_error,
                    visibility_scores=list(visibility_scores.values()),
                    num_views=point_3d.num_views
                )
                
                landmark_3d = Landmark3D(
                    position=point_3d,
                    landmark_id=landmark_idx,
                    visibility_scores=visibility_scores
                )
                landmarks_3d.append(landmark_3d)
            else:
                # Triangulation failed - use fallback or skip
                landmarks_3d.append(None)
        
        return landmarks_3d
    
    def _triangulate_dlt(
        self,
        projection_matrices: List[np.ndarray],
        points_2d: List[np.ndarray],
        visibility_weights: Dict[str, float] = None,
        camera_ids: List[str] = None
    ) -> np.ndarray:
        """Direct Linear Transform triangulation with PyTorch acceleration."""
        num_views = len(projection_matrices)
        
        if CUDA_ENABLED:
            # Move data to GPU
            P_torch = torch.tensor(np.stack(projection_matrices), dtype=torch.float32, device=DEVICE)
            pts_torch = torch.tensor(np.stack(points_2d), dtype=torch.float32, device=DEVICE)
            
            # Build A matrix on GPU
            # Each view gives 2 rows:
            # x * P[2] - P[0]
            # y * P[2] - P[1]
            A = torch.zeros((2 * num_views, 4), device=DEVICE)
            for i in range(num_views):
                x, y = pts_torch[i]
                P = P_torch[i]
                
                weight = 1.0
                if visibility_weights and camera_ids:
                    weight = visibility_weights.get(camera_ids[i], 1.0)
                
                A[2*i] = weight * (x * P[2] - P[0])
                A[2*i + 1] = weight * (y * P[2] - P[1])
            
            # Solve using SVD on GPU
            U, S, Vh = torch.linalg.svd(A)
            X = Vh[-1]  # Last row of Vh corresponds to smallest singular value
            
            return X.detach().cpu().numpy()
        else:
            # Fallback to NumPy
            A = np.zeros((2 * num_views, 4))
            for i, (P, point) in enumerate(zip(projection_matrices, points_2d)):
                x, y = point
                
                weight = 1.0
                if visibility_weights and camera_ids:
                    weight = visibility_weights.get(camera_ids[i], 1.0)
                    
                A[2*i] = weight * (x * P[2] - P[0])
                A[2*i + 1] = weight * (y * P[2] - P[1])
            
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            return X
    
    def _calculate_reprojection_error(
        self,
        point_3d: np.ndarray,
        projection_matrices: List[np.ndarray],
        points_2d: List[np.ndarray]
    ) -> float:
        """Calculate mean reprojection error with PyTorch acceleration."""
        if CUDA_ENABLED:
            point_3d_h = torch.tensor(np.append(point_3d, 1.0), dtype=torch.float32, device=DEVICE)
            P_torch = torch.tensor(np.stack(projection_matrices), dtype=torch.float32, device=DEVICE)
            pts_obs = torch.tensor(np.stack(points_2d), dtype=torch.float32, device=DEVICE)
            
            # Project 3D points to 2D: [N, 3, 4] @ [4, 1] -> [N, 3, 1]
            pts_proj_h = torch.matmul(P_torch, point_3d_h.unsqueeze(-1)).squeeze(-1)
            pts_proj = pts_proj_h[:, :2] / pts_proj_h[:, 2:3]
            
            # Distance error
            errors = torch.norm(pts_proj - pts_obs, dim=1)
            return torch.mean(errors).item()
        else:
            errors = []
            point_3d_h = np.append(point_3d, 1.0)
            for P, point_2d_observed in zip(projection_matrices, points_2d):
                point_2d_projected_h = P @ point_3d_h
                point_2d_projected = point_2d_projected_h[:2] / point_2d_projected_h[2]
                error = np.linalg.norm(point_2d_projected - point_2d_observed)
                errors.append(error)
            return np.mean(errors)
    
    def _calculate_confidence(
        self,
        reproj_error: float,
        visibility_scores: Optional[List[float]] = None,
        num_views: int = 2
    ) -> float:
        """
        Calculate confidence score for triangulated point.
        
        Args:
            reproj_error: Reprojection error in pixels
            visibility_scores: List of visibility scores from each camera
            num_views: Number of views used
            
        Returns:
            Confidence score [0, 1]
        """
        # Reprojection error component (lower is better)
        # Map error to [0, 1] using sigmoid-like function
        reproj_confidence = np.exp(-reproj_error / 5.0)
        
        # Visibility component
        if visibility_scores:
            vis_confidence = np.mean(visibility_scores)
        else:
            vis_confidence = 1.0
        
        # Number of views bonus
        view_bonus = min(1.0, num_views / 4.0)  # Max bonus at 4+ views
        
        # Weighted combination
        confidence = (
            CONFIDENCE_WEIGHT_REPROJ * reproj_confidence +
            CONFIDENCE_WEIGHT_VISIBILITY * vis_confidence
        ) * view_bonus
        
        return np.clip(confidence, 0.0, 1.0)
    
    def triangulate_with_cv2(
        self,
        camera_id_1: str,
        camera_id_2: str,
        points_1: np.ndarray,
        points_2: np.ndarray,
        undistort: bool = True
    ) -> np.ndarray:
        """
        Triangulate using OpenCV's built-in function (for verification).
        
        Args:
            camera_id_1: First camera ID
            camera_id_2: Second camera ID
            points_1: Nx2 array of 2D points from camera 1
            points_2: Nx2 array of 2D points from camera 2
            undistort: Whether to undistort points
            
        Returns:
            Nx3 array of 3D points
        """
        P1 = self.calibration.get_projection_matrix(camera_id_1)
        P2 = self.calibration.get_projection_matrix(camera_id_2)
        
        if undistort:
            points_1 = self.calibration.undistort_point(camera_id_1, points_1)
            points_2 = self.calibration.undistort_point(camera_id_2, points_2)
        
        # OpenCV requires points in shape (1, N, 2)
        points_1 = points_1.reshape(1, -1, 2).astype(np.float32)
        points_2 = points_2.reshape(1, -1, 2).astype(np.float32)
        
        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, points_1, points_2)
        
        # Convert from homogeneous to 3D
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T


if __name__ == "__main__":
    print("Triangulation Module - Test")
    
    # Create mock calibration
    from src.stereo_calibration import CameraCalibration
    
    # Create synthetic camera setup
    cal = StereoCalibration()
    
    # Camera 0: at origin
    cal.cameras['cam_0'] = CameraCalibration(
        camera_id='cam_0',
        intrinsic_matrix=np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]]),
        distortion_coeffs=np.zeros(5),
        rotation=np.eye(3),
        translation=np.zeros((3, 1)),
        image_size=(1280, 720)
    )
    
    # Camera 1: 1 meter to the right
    cal.cameras['cam_1'] = CameraCalibration(
        camera_id='cam_1',
        intrinsic_matrix=np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]]),
        distortion_coeffs=np.zeros(5),
        rotation=np.eye(3),
        translation=np.array([[1.0], [0.0], [0.0]]),
        image_size=(1280, 720)
    )
    
    triangulator = Triangulator(cal)
    
    # Test triangulation of a point at (0, 0, 2) meters
    observations = {
        'cam_0': np.array([640.0, 360.0]),  # Center of image
        'cam_1': np.array([240.0, 360.0])   # Shifted due to baseline
    }
    
    point_3d = triangulator.triangulate_point(observations, undistort=False)
    
    if point_3d:
        print(f"\nTriangulated point:")
        print(f"  Position: ({point_3d.x:.3f}, {point_3d.y:.3f}, {point_3d.z:.3f})")
        print(f"  Confidence: {point_3d.confidence:.3f}")
        print(f"  Reprojection error: {point_3d.reprojection_error:.3f} px")
    else:
        print("Triangulation failed")
