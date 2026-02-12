"""
Stereo Calibration Module
Handles intrinsic and extrinsic camera calibration for multi-view 3D reconstruction.
"""

import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CameraCalibration:
    """Calibration data for a single camera."""
    camera_id: str
    intrinsic_matrix: np.ndarray  # 3x3 camera matrix
    distortion_coeffs: np.ndarray  # (k1, k2, p1, p2, k3)
    rotation: Optional[np.ndarray] = None  # 3x3 rotation matrix (relative to world)
    translation: Optional[np.ndarray] = None  # 3x1 translation vector
    image_size: Tuple[int, int] = None  # (width, height)
    reprojection_error: float = 0.0


class StereoCalibration:
    """
    Handles camera calibration for multi-view motion capture.
    Supports both intrinsic (lens) and extrinsic (position/orientation) calibration.
    """
    
    def __init__(self):
        """Initialize stereo calibration system."""
        self.cameras: Dict[str, CameraCalibration] = {}
        self.calibration_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
        print("[StereoCalibration] Initialized")
    
    def calibrate_intrinsic(
        self,
        camera_id: str,
        images: List[np.ndarray],
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025  # meters
    ) -> CameraCalibration:
        """
        Calibrate intrinsic parameters (camera matrix, distortion) using checkerboard.
        
        Args:
            camera_id: Unique camera identifier
            images: List of grayscale/color images of checkerboard
            checkerboard_size: Number of internal corners (cols, rows)
            square_size: Size of checkerboard squares in meters
            
        Returns:
            CameraCalibration object with intrinsic parameters
        """
        print(f"[StereoCalibration] Calibrating intrinsics for {camera_id}...")
        
        # Prepare object points (3D points in real world space)
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:checkerboard_size[0],
            0:checkerboard_size[1]
        ].T.reshape(-1, 2)
        objp *= square_size
        
        # Arrays to store object points and image points
        obj_points = []  # 3D points in real world space
        img_points = []  # 2D points in image plane
        
        image_size = None
        found_count = 0
        
        for i, img in enumerate(images):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            image_size = gray.shape[::-1]
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            
            if ret:
                obj_points.append(objp)
                
                # Refine corner positions
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self.calibration_criteria
                )
                img_points.append(corners_refined)
                found_count += 1
                print(f"  Found corners in image {i+1}/{len(images)}")
            else:
                print(f"  Failed to find corners in image {i+1}/{len(images)}")
        
        if found_count < 10:
            raise ValueError(
                f"Not enough valid calibration images. Found {found_count}, need at least 10"
            )
        
        print(f"  Successfully processed {found_count}/{len(images)} images")
        
        # Perform calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(obj_points)):
            img_points_reprojected, _ = cv2.projectPoints(
                obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(img_points[i], img_points_reprojected, cv2.NORM_L2) / len(
                img_points_reprojected
            )
            mean_error += error
        
        mean_error /= len(obj_points)
        
        print(f"  Calibration complete. Reprojection error: {mean_error:.3f} pixels")
        
        # Create calibration object
        calibration = CameraCalibration(
            camera_id=camera_id,
            intrinsic_matrix=camera_matrix,
            distortion_coeffs=dist_coeffs,
            image_size=image_size,
            reprojection_error=mean_error
        )
        
        self.cameras[camera_id] = calibration
        return calibration
    
    def calibrate_stereo(
        self,
        camera_id_1: str,
        camera_id_2: str,
        images_1: List[np.ndarray],
        images_2: List[np.ndarray],
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025
    ) -> Tuple[CameraCalibration, CameraCalibration]:
        """
        Calibrate extrinsic parameters between two cameras.
        
        Args:
            camera_id_1: First camera ID
            camera_id_2: Second camera ID
            images_1: Checkerboard images from camera 1
            images_2: Corresponding checkerboard images from camera 2
            checkerboard_size: Number of internal corners
            square_size: Size of squares in meters
            
        Returns:
            Tuple of calibration objects for both cameras
        """
        print(f"[StereoCalibration] Calibrating stereo pair: {camera_id_1} <-> {camera_id_2}")
        
        # Ensure both cameras have intrinsic calibration
        if camera_id_1 not in self.cameras:
            self.calibrate_intrinsic(camera_id_1, images_1, checkerboard_size, square_size)
        if camera_id_2 not in self.cameras:
            self.calibrate_intrinsic(camera_id_2, images_2, checkerboard_size, square_size)
        
        cal1 = self.cameras[camera_id_1]
        cal2 = self.cameras[camera_id_2]
        
        # Prepare object points
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:checkerboard_size[0],
            0:checkerboard_size[1]
        ].T.reshape(-1, 2)
        objp *= square_size
        
        # Find corners in both image sets
        obj_points = []
        img_points_1 = []
        img_points_2 = []
        
        for i, (img1, img2) in enumerate(zip(images_1, images_2)):
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
            
            ret1, corners1 = cv2.findChessboardCorners(gray1, checkerboard_size, None)
            ret2, corners2 = cv2.findChessboardCorners(gray2, checkerboard_size, None)
            
            if ret1 and ret2:
                obj_points.append(objp)
                
                corners1 = cv2.cornerSubPix(
                    gray1, corners1, (11, 11), (-1, -1), self.calibration_criteria
                )
                corners2 = cv2.cornerSubPix(
                    gray2, corners2, (11, 11), (-1, -1), self.calibration_criteria
                )
                
                img_points_1.append(corners1)
                img_points_2.append(corners2)
                print(f"  Found corners in pair {i+1}/{len(images_1)}")
        
        print(f"  Successfully processed {len(obj_points)} image pairs")
        
        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC  # Use pre-calibrated intrinsics
        
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_points,
            img_points_1,
            img_points_2,
            cal1.intrinsic_matrix,
            cal1.distortion_coeffs,
            cal2.intrinsic_matrix,
            cal2.distortion_coeffs,
            cal1.image_size,
            criteria=self.calibration_criteria,
            flags=flags
        )
        
        print(f"  Stereo calibration complete. RMS error: {ret:.3f}")
        
        # Set camera 1 as world origin (identity rotation/translation)
        cal1.rotation = np.eye(3)
        cal1.translation = np.zeros((3, 1))
        
        # Camera 2 position relative to camera 1
        cal2.rotation = R
        cal2.translation = T
        
        self.cameras[camera_id_1] = cal1
        self.cameras[camera_id_2] = cal2
        
        return cal1, cal2
    
    def save_calibration(self, filepath: str):
        """
        Save calibration data to JSON file.
        
        Args:
            filepath: Path to save calibration file
        """
        data = {}
        
        for camera_id, cal in self.cameras.items():
            data[camera_id] = {
                'intrinsic_matrix': cal.intrinsic_matrix.tolist(),
                'distortion_coeffs': cal.distortion_coeffs.tolist(),
                'rotation': cal.rotation.tolist() if cal.rotation is not None else None,
                'translation': cal.translation.tolist() if cal.translation is not None else None,
                'image_size': list(cal.image_size) if cal.image_size else None,
                'reprojection_error': cal.reprojection_error
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[StereoCalibration] Saved calibration to {filepath}")
    
    def load_calibration(self, filepath: str):
        """
        Load calibration data from JSON file.
        
        Args:
            filepath: Path to calibration file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for camera_id, cal_data in data.items():
            calibration = CameraCalibration(
                camera_id=camera_id,
                intrinsic_matrix=np.array(cal_data['intrinsic_matrix']),
                distortion_coeffs=np.array(cal_data['distortion_coeffs']),
                rotation=np.array(cal_data['rotation']) if cal_data['rotation'] else None,
                translation=np.array(cal_data['translation']) if cal_data['translation'] else None,
                image_size=tuple(cal_data['image_size']) if cal_data['image_size'] else None,
                reprojection_error=cal_data['reprojection_error']
            )
            self.cameras[camera_id] = calibration
        
        print(f"[StereoCalibration] Loaded calibration for {len(self.cameras)} cameras from {filepath}")
    
    def undistort_point(self, camera_id: str, point: np.ndarray) -> np.ndarray:
        """
        Undistort a 2D image point using camera calibration.
        
        Args:
            camera_id: Camera identifier
            point: 2D point as (x, y) or [[x, y]]
            
        Returns:
            Undistorted point
        """
        if camera_id not in self.cameras:
            raise ValueError(f"No calibration found for camera: {camera_id}")
        
        cal = self.cameras[camera_id]
        
        # Ensure point is in correct format
        if point.ndim == 1:
            point = point.reshape(1, 1, 2)
        elif point.ndim == 2 and point.shape[0] == 1:
            point = point.reshape(1, 1, 2)
        
        undistorted = cv2.undistortPoints(
            point,
            cal.intrinsic_matrix,
            cal.distortion_coeffs,
            P=cal.intrinsic_matrix
        )
        
        return undistorted.reshape(-1, 2)
    
    def get_projection_matrix(self, camera_id: str) -> np.ndarray:
        """
        Get 3x4 projection matrix for a camera.
        P = K [R | t]
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            3x4 projection matrix
        """
        if camera_id not in self.cameras:
            raise ValueError(f"No calibration found for camera: {camera_id}")
        
        cal = self.cameras[camera_id]
        
        if cal.rotation is None or cal.translation is None:
            raise ValueError(f"Extrinsic parameters not calibrated for camera: {camera_id}")
        
        # P = K [R | t]
        RT = np.hstack([cal.rotation, cal.translation])
        P = cal.intrinsic_matrix @ RT
        
        return P

    def create_default_calibration(self, width: int = 1280, height: int = 720) -> None:
        """
        Create a default/approximate calibration for immediate use.
        Assumes two cameras side-by-side, ~1 meter apart.
        """
        print("[StereoCalibration] Creating default calibration (Approximate)")
        
        # Approximate intrinsics for a standard webcam (FOV ~60 deg)
        focal_length = width  # Rough estimate: fx = width
        center_x = width / 2
        center_y = height / 2
        
        K = np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist = np.zeros(5)  # Assume no distortion
        
        # Camera 1 (Origin)
        self.cameras['local_cam'] = CameraCalibration(
            camera_id='local_cam',
            intrinsic_matrix=K,
            distortion_coeffs=dist,
            rotation=np.eye(3),
            translation=np.zeros((3, 1)),
            image_size=(width, height)
        )
        
        # Camera 2 (Remote - PC2)
        # Positioned 1.0 meter to the right of Camera 1
        self.cameras['cam_0'] = CameraCalibration(
            camera_id='cam_0',
            intrinsic_matrix=K,
            distortion_coeffs=dist,
            rotation=np.eye(3),  # Facing same direction
            translation=np.array([[-1.0], [0.0], [0.0]]),  # Translated -1m on X axis (relative to Cam1)
            image_size=(width, height)
        )
        print("[StereoCalibration] Default calibration created (Baseline: 1.0m)")


if __name__ == "__main__":
    print("Stereo Calibration Module - Test")
    cal = StereoCalibration()
    cal.create_default_calibration()
