"""
Uncertainty Estimation Module
Multi-source error modeling and confidence scoring for 3D landmark estimation.

Provides:
  - Reprojection-based uncertainty (pixel error → 3D error)
  - Visibility variance across cameras
  - Occlusion penalty integration
  - Calibration error propagation
  - Combined 3×3 covariance matrix estimation
  - ErrorMetricsCalculator for aggregate quality metrics
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict


# ── Data Structures ──

@dataclass
class PointUncertainty:
    """Uncertainty estimate for a single 3D landmark."""
    landmark_id: int
    reproj_error_px: float = 0.0        # Mean reprojection error (pixels)
    inter_camera_disagreement_m: float = 0.0  # Max pairwise distance between per-camera 3D estimates
    visibility_variance: float = 0.0     # Variance of visibility scores across cameras
    occlusion_penalty: float = 0.0       # Additional uncertainty from occlusion state
    calibration_error_m: float = 0.0     # Estimated calibration-induced error
    combined_uncertainty_m: float = 0.0  # Combined scalar uncertainty
    covariance_3x3: Optional[np.ndarray] = None  # 3×3 covariance matrix
    confidence: float = 1.0              # Final confidence score (0-1)
    method: str = 'unknown'              # How the 3D point was obtained


@dataclass
class CalibrationQuality:
    """Estimated quality of stereo calibration."""
    rms_error_px: float = 0.0            # RMS reprojection error from calibration
    baseline_m: float = 1.0              # Inter-camera baseline distance
    focal_length_px: float = 1000.0      # Average focal length in pixels
    depth_error_at_2m: float = 0.0       # Estimated depth error at 2m distance
    angular_error_deg: float = 0.0       # Estimated angular error between cameras


class UncertaintyEstimator:
    """
    Estimates uncertainty for triangulated 3D landmarks.

    Models multiple error sources and combines them into a single
    confidence metric and optional covariance matrix.

    Usage:
        estimator = UncertaintyEstimator(calibration_rms=0.5, baseline_m=1.0)
        uncertainties = estimator.estimate_frame(
            landmarks_3d={0: {'x':..., 'y':..., 'z':..., 'visibility':...}, ...},
            per_camera_points={0: [np.array([x,y,z]), ...], ...},
            visibility_scores={0: {'cam_0': 0.9, 'cam_1': 0.7}, ...},
            occlusion_states={0: 'VISIBLE', ...}
        )
    """

    def __init__(self,
                 calibration_rms: float = 0.5,
                 baseline_m: float = 1.0,
                 focal_length_px: float = 1000.0,
                 image_size: Tuple[int, int] = (1280, 720)):
        """
        Args:
            calibration_rms:  RMS reprojection error from calibration (pixels)
            baseline_m:       Inter-camera baseline distance (meters)
            focal_length_px:  Average focal length (pixels)
            image_size:       Camera resolution
        """
        self.calibration_rms = calibration_rms
        self.baseline_m = baseline_m
        self.focal_length_px = focal_length_px
        self.image_size = image_size

        # Precompute calibration quality
        self.cal_quality = CalibrationQuality(
            rms_error_px=calibration_rms,
            baseline_m=baseline_m,
            focal_length_px=focal_length_px,
            depth_error_at_2m=self._estimate_depth_error(2.0),
            angular_error_deg=math.degrees(
                math.atan2(calibration_rms, focal_length_px)
            )
        )

        # Weight factors for combining uncertainties
        self._w_reproj = 0.30
        self._w_disagreement = 0.35
        self._w_visibility = 0.15
        self._w_occlusion = 0.10
        self._w_calibration = 0.10

    def _estimate_depth_error(self, depth_m: float) -> float:
        """
        Estimate depth error from stereo geometry.

        ΔZ ≈ Z² × Δd / (f × B)

        where Δd is disparity error (≈ calibration RMS), f is focal length,
        B is baseline, Z is depth.
        """
        if self.focal_length_px <= 0 or self.baseline_m <= 0:
            return 0.1  # Fallback

        delta_d = self.calibration_rms  # pixels
        delta_z = (depth_m ** 2) * delta_d / (self.focal_length_px * self.baseline_m)
        return abs(delta_z)

    def estimate_frame(
        self,
        landmarks_3d: Dict[int, Dict[str, Any]],
        per_camera_points: Optional[Dict[int, List[np.ndarray]]] = None,
        visibility_scores: Optional[Dict[int, Dict[str, float]]] = None,
        occlusion_states: Optional[Dict[int, str]] = None,
        reproj_errors: Optional[Dict[int, float]] = None,
    ) -> Dict[int, PointUncertainty]:
        """
        Estimate uncertainty for all landmarks in a frame.

        Args:
            landmarks_3d:      3D landmarks {lm_id: {x, y, z, visibility, ...}}
            per_camera_points:  Per-camera 3D estimates {lm_id: [array, array, ...]}
            visibility_scores:  Per-camera visibility {lm_id: {cam_id: vis_score}}
            occlusion_states:   Occlusion state per landmark {lm_id: state_str}
            reproj_errors:      Reprojection error per landmark {lm_id: error_px}

        Returns:
            Dict mapping landmark_id to PointUncertainty
        """
        per_camera_points = per_camera_points or {}
        visibility_scores = visibility_scores or {}
        occlusion_states = occlusion_states or {}
        reproj_errors = reproj_errors or {}

        uncertainties: Dict[int, PointUncertainty] = {}

        for lm_id, lm_data in landmarks_3d.items():
            unc = PointUncertainty(
                landmark_id=lm_id,
                method=lm_data.get('method', 'unknown')
            )

            # 1. Reprojection error
            unc.reproj_error_px = reproj_errors.get(
                lm_id, lm_data.get('reproj_error', 0.0)
            )

            # 2. Inter-camera disagreement
            if lm_id in per_camera_points and len(per_camera_points[lm_id]) >= 2:
                unc.inter_camera_disagreement_m = self._compute_disagreement(
                    per_camera_points[lm_id]
                )

            # 3. Visibility variance
            if lm_id in visibility_scores:
                vis_values = list(visibility_scores[lm_id].values())
                if len(vis_values) >= 2:
                    mean_vis = sum(vis_values) / len(vis_values)
                    unc.visibility_variance = sum(
                        (v - mean_vis) ** 2 for v in vis_values
                    ) / len(vis_values)

            # 4. Occlusion penalty
            state = occlusion_states.get(lm_id, 'VISIBLE')
            unc.occlusion_penalty = self._occlusion_penalty(state)

            # 5. Calibration error propagation
            # Estimate depth from 3D position
            lm_pos = np.array([
                lm_data.get('x', 0), lm_data.get('y', 0), lm_data.get('z', 0)
            ])
            depth_m = max(0.5, float(np.linalg.norm(lm_pos)))
            unc.calibration_error_m = self._estimate_depth_error(depth_m)

            # 6. Combined uncertainty (weighted sum)
            reproj_metric = self._reproj_to_meters(
                unc.reproj_error_px, depth_m
            )

            unc.combined_uncertainty_m = (
                self._w_reproj * reproj_metric +
                self._w_disagreement * unc.inter_camera_disagreement_m +
                self._w_visibility * unc.visibility_variance * 0.1 +  # Scale down
                self._w_occlusion * unc.occlusion_penalty * 0.05 +
                self._w_calibration * unc.calibration_error_m
            )

            # 7. Covariance matrix (isotropic approximation)
            sigma = max(0.001, unc.combined_uncertainty_m)
            unc.covariance_3x3 = np.eye(3) * (sigma ** 2)

            # 8. Confidence score (inverse of uncertainty, clamped 0-1)
            raw_conf = lm_data.get('visibility', 1.0)
            uncertainty_penalty = min(1.0, unc.combined_uncertainty_m / 0.1)
            unc.confidence = max(0.0, raw_conf * (1.0 - 0.5 * uncertainty_penalty))

            uncertainties[lm_id] = unc

        return uncertainties

    @staticmethod
    def _compute_disagreement(estimates: List[np.ndarray]) -> float:
        """Max pairwise Euclidean distance between camera 3D estimates."""
        max_dist = 0.0
        for i in range(len(estimates)):
            for j in range(i + 1, len(estimates)):
                d = float(np.linalg.norm(estimates[i] - estimates[j]))
                if d > max_dist:
                    max_dist = d
        return max_dist

    @staticmethod
    def _occlusion_penalty(state: str) -> float:
        """
        Return a penalty factor [0, 1] based on occlusion state.

        VISIBLE=0, PARTIAL=0.2, PREDICTED=0.5, OCCLUDED=0.8, LOST=1.0
        """
        penalties = {
            'VISIBLE': 0.0,
            'PARTIAL': 0.2,
            'PREDICTED': 0.5,
            'OCCLUDED': 0.8,
            'LOST': 1.0,
        }
        return penalties.get(state, 0.5)

    def _reproj_to_meters(self, reproj_px: float, depth_m: float) -> float:
        """
        Convert reprojection error (pixels) to approximate 3D error (meters).

        Δx_3D ≈ Z × Δx_px / f
        """
        if self.focal_length_px <= 0:
            return 0.01
        return depth_m * reproj_px / self.focal_length_px


# ── Error Metrics Calculator ──

class ErrorMetricsCalculator:
    """
    Aggregates per-frame uncertainty estimates into session-level
    quality metrics.

    Tracks running statistics and provides summaries for the dashboard.
    """

    def __init__(self, window_size: int = 300):
        """
        Args:
            window_size: Number of recent frames to keep for running stats
        """
        self.window_size = window_size
        self._frame_uncertainties: List[Dict[int, PointUncertainty]] = []
        self._mean_uncertainties: List[float] = []
        self._consistency_scores: List[float] = []

    def add_frame(self, uncertainties: Dict[int, PointUncertainty]):
        """Add uncertainty estimates from one frame."""
        self._frame_uncertainties.append(uncertainties)

        # Keep bounded
        if len(self._frame_uncertainties) > self.window_size:
            self._frame_uncertainties = self._frame_uncertainties[-self.window_size:]

        # Compute frame-level mean uncertainty
        if uncertainties:
            mean_u = sum(u.combined_uncertainty_m for u in uncertainties.values()) / len(uncertainties)
            self._mean_uncertainties.append(mean_u)
        if len(self._mean_uncertainties) > self.window_size:
            self._mean_uncertainties = self._mean_uncertainties[-self.window_size:]

    def add_consistency_score(self, score: float):
        """Add a multi-view consistency score (0-1)."""
        self._consistency_scores.append(score)
        if len(self._consistency_scores) > self.window_size:
            self._consistency_scores = self._consistency_scores[-self.window_size:]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get aggregate quality metrics.

        Returns dict with:
          - mean_uncertainty_m: Average uncertainty across recent frames
          - p95_uncertainty_m:  95th percentile uncertainty
          - worst_landmarks:    Top 5 landmarks with highest average uncertainty
          - consistency_mean:   Mean multi-view consistency score
          - quality_grade:      A/B/C/D/F quality grade
        """
        if not self._mean_uncertainties:
            return {
                'mean_uncertainty_m': 0.0,
                'p95_uncertainty_m': 0.0,
                'worst_landmarks': [],
                'consistency_mean': 0.0,
                'quality_grade': 'N/A'
            }

        sorted_u = sorted(self._mean_uncertainties)
        n = len(sorted_u)
        mean_u = sum(sorted_u) / n
        p95_u = sorted_u[int(n * 0.95)] if n >= 20 else sorted_u[-1]

        # Find worst landmarks
        lm_totals: Dict[int, List[float]] = defaultdict(list)
        for frame_u in self._frame_uncertainties[-100:]:
            for lm_id, unc in frame_u.items():
                lm_totals[lm_id].append(unc.combined_uncertainty_m)

        worst = sorted(
            [(lm_id, sum(vals) / len(vals)) for lm_id, vals in lm_totals.items()],
            key=lambda x: x[1], reverse=True
        )[:5]

        # Consistency
        consistency_mean = (
            sum(self._consistency_scores) / len(self._consistency_scores)
            if self._consistency_scores else 0.0
        )

        # Quality grade
        grade = self._compute_grade(mean_u, consistency_mean)

        return {
            'mean_uncertainty_m': round(mean_u, 4),
            'p95_uncertainty_m': round(p95_u, 4),
            'worst_landmarks': [{'id': lm_id, 'mean_uncertainty_m': round(u, 4)}
                               for lm_id, u in worst],
            'consistency_mean': round(consistency_mean, 3),
            'quality_grade': grade,
            'total_frames_analyzed': len(self._mean_uncertainties),
        }

    @staticmethod
    def _compute_grade(mean_uncertainty: float, consistency: float) -> str:
        """
        Compute quality grade from uncertainty and consistency.

        A: excellent (<5mm uncertainty, >0.9 consistency)
        B: good (<10mm, >0.7)
        C: acceptable (<20mm, >0.5)
        D: poor (<50mm, >0.3)
        F: unusable
        """
        score = (1.0 - min(1.0, mean_uncertainty / 0.05)) * 0.6 + consistency * 0.4

        if score >= 0.85:
            return 'A'
        elif score >= 0.70:
            return 'B'
        elif score >= 0.50:
            return 'C'
        elif score >= 0.30:
            return 'D'
        else:
            return 'F'

    def reset(self):
        """Clear all accumulated data."""
        self._frame_uncertainties.clear()
        self._mean_uncertainties.clear()
        self._consistency_scores.clear()
