"""
Occlusion Fusion Engine
Multi-view arbitration with temporal consistency for handling occluded landmarks.

Provides:
  - Single-view fallback with scaled confidence
  - Weighted multi-view fusion (triangulation + monocular blend)
  - Temporal history tracking with exponential decay
  - Uncertainty scoring based on inter-view disagreement
  - State machine: VISIBLE → PREDICTED → OCCLUDED → LOST
"""

import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum


class OcclusionState(Enum):
    """Per-landmark occlusion state."""
    VISIBLE = 'VISIBLE'       # Seen by ≥2 cameras with good triangulation
    PARTIAL = 'PARTIAL'       # Seen by 1 camera only (monocular fallback)
    PREDICTED = 'PREDICTED'   # Not seen, using temporal prediction
    OCCLUDED = 'OCCLUDED'     # Lost for too long, holding last known position
    LOST = 'LOST'             # Lost beyond recovery, dropped from output


@dataclass
class LandmarkHistory:
    """Temporal history for a single landmark."""
    landmark_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=30))  # (timestamp, x, y, z, confidence)
    state: OcclusionState = OcclusionState.LOST
    frames_hidden: int = 0
    last_visible_time: float = 0.0
    last_position: Optional[Dict[str, float]] = None
    velocity: Optional[Tuple[float, float, float]] = None  # Estimated velocity for prediction
    uncertainty: float = 0.0  # Current uncertainty in meters

    @property
    def is_active(self) -> bool:
        return self.state in (OcclusionState.VISIBLE, OcclusionState.PARTIAL,
                              OcclusionState.PREDICTED, OcclusionState.OCCLUDED)


@dataclass
class FusionConfig:
    """Configuration for the occlusion fusion engine."""
    max_predicted_frames: int = 15        # ~500ms at 30fps
    max_occluded_frames: int = 90         # 3 seconds before LOST
    confidence_decay_rate: float = 0.9    # Multiply confidence each predicted frame
    velocity_damping: float = 0.95        # Damp velocity during prediction
    min_confidence_output: float = 0.05   # Below this, mark as LOST
    disagreement_threshold_m: float = 0.1 # Inter-camera disagreement threshold
    monocular_confidence_scale: float = 0.7  # Scale factor for monocular fallback
    history_window: int = 30              # Number of frames to keep in history


class OcclusionFusionEngine:
    """
    Multi-view landmark fusion with occlusion handling.

    Designed to be called per-frame with multi-camera landmark observations.
    Maintains temporal state and produces fused 3D output with confidence
    and occlusion metadata.

    Usage:
        engine = OcclusionFusionEngine()
        fused = engine.fuse_frame(
            triangulated={0: {'x':..., 'y':..., 'z':..., 'visibility':...}, ...},
            monocular={0: {'x':..., ...}, ...},
            world_landmarks={0: {'x':..., ...}, ...},
            timestamp_ns=time.time_ns()
        )
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self._history: Dict[int, LandmarkHistory] = {}
        self._frame_count = 0

    def _get_history(self, lm_id: int) -> LandmarkHistory:
        """Get or create history for a landmark."""
        if lm_id not in self._history:
            self._history[lm_id] = LandmarkHistory(landmark_id=lm_id)
        return self._history[lm_id]

    def fuse_frame(
        self,
        triangulated: Dict[int, Dict[str, Any]],
        monocular: Optional[Dict[int, Dict[str, Any]]] = None,
        world_landmarks: Optional[Dict[int, Dict[str, Any]]] = None,
        per_camera_estimates: Optional[Dict[int, List[np.ndarray]]] = None,
        timestamp_ns: int = 0
    ) -> Dict[str, Any]:
        """
        Fuse landmarks from multiple sources with occlusion handling.

        Args:
            triangulated:       Multi-view triangulated 3D landmarks (Tier 1)
            monocular:          Single-view monocular fallbacks (Tier 2)
            world_landmarks:    MediaPipe world landmarks (Tier 3)
            per_camera_estimates: Per-camera 3D estimates for disagreement
            timestamp_ns:       Frame timestamp in nanoseconds

        Returns:
            Dict with 'landmarks_3d', 'occlusion_states', 'uncertainty',
            'method_counts', 'active_count'
        """
        if timestamp_ns == 0:
            timestamp_ns = time.time_ns()

        monocular = monocular or {}
        world_landmarks = world_landmarks or {}
        per_camera_estimates = per_camera_estimates or {}

        self._frame_count += 1

        fused_landmarks: Dict[int, Dict[str, Any]] = {}
        occlusion_states: Dict[int, str] = {}
        uncertainty: Dict[int, float] = {}
        method_counts = {'triangulated': 0, 'monocular': 0, 'world': 0,
                         'predicted': 0, 'occluded': 0, 'lost': 0}

        # Collect all known landmark IDs
        all_ids = set()
        all_ids.update(triangulated.keys())
        all_ids.update(monocular.keys())
        all_ids.update(world_landmarks.keys())
        all_ids.update(self._history.keys())

        for lm_id in all_ids:
            hist = self._get_history(lm_id)
            result = None
            method = 'lost'

            # ── TIER 1: Triangulated (multi-view) ──
            if lm_id in triangulated:
                pt = triangulated[lm_id]
                result = self._make_output(pt, 'triangulated')
                method = 'triangulated'

                # Compute inter-camera disagreement
                if lm_id in per_camera_estimates:
                    estimates = per_camera_estimates[lm_id]
                    if len(estimates) >= 2:
                        disagree = self._compute_disagreement(estimates)
                        uncertainty[lm_id] = disagree
                        hist.uncertainty = disagree

                # Update temporal history
                self._update_history_visible(hist, result, timestamp_ns)

            # ── TIER 2: Monocular fallback ──
            elif lm_id in monocular:
                pt = monocular[lm_id]
                result = self._make_output(pt, 'monocular')
                result['visibility'] *= self.config.monocular_confidence_scale
                method = 'monocular'
                self._update_history_visible(hist, result, timestamp_ns)

            # ── TIER 3: World landmarks ──
            elif lm_id in world_landmarks:
                pt = world_landmarks[lm_id]
                result = self._make_output(pt, 'world_landmarks')
                result['visibility'] *= 0.5  # Lower confidence
                method = 'world'
                self._update_history_visible(hist, result, timestamp_ns)

            # ── TIER 4: Temporal prediction ──
            elif hist.is_active and hist.last_position is not None:
                hist.frames_hidden += 1

                if hist.frames_hidden <= self.config.max_predicted_frames:
                    result = self._predict_position(hist, timestamp_ns)
                    method = 'predicted'
                    hist.state = OcclusionState.PREDICTED

                elif hist.frames_hidden <= self.config.max_occluded_frames:
                    # Hold last known position with very low confidence
                    result = hist.last_position.copy()
                    result['method'] = 'occluded'
                    result['visibility'] = max(
                        self.config.min_confidence_output,
                        result.get('visibility', 0.1) * 0.5
                    )
                    method = 'occluded'
                    hist.state = OcclusionState.OCCLUDED

                else:
                    # Lost — too long without observation
                    hist.state = OcclusionState.LOST
                    method = 'lost'

            else:
                hist.state = OcclusionState.LOST
                method = 'lost'

            # Store result
            if result is not None and result.get('visibility', 0) >= self.config.min_confidence_output:
                fused_landmarks[lm_id] = result
            
            occlusion_states[lm_id] = hist.state.value
            method_counts[method] += 1

        active_count = sum(1 for s in occlusion_states.values()
                          if s in ('VISIBLE', 'PARTIAL', 'PREDICTED'))

        return {
            'landmarks_3d': fused_landmarks,
            'occlusion_states': occlusion_states,
            'uncertainty': uncertainty,
            'method_counts': method_counts,
            'active_count': active_count,
            'total_tracked': len(all_ids),
            'frame_index': self._frame_count,
        }

    def _make_output(self, pt: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Create a standardized output dict from a landmark point."""
        return {
            'x': float(pt.get('x', 0.0)),
            'y': float(pt.get('y', 0.0)),
            'z': float(pt.get('z', 0.0)),
            'visibility': float(pt.get('visibility', pt.get('v', 1.0))),
            'method': method,
            'views': int(pt.get('views', 1)),
            'reproj_error': float(pt.get('reproj_error', 0.0)),
        }

    def _update_history_visible(self, hist: LandmarkHistory,
                                 result: Dict[str, Any], timestamp_ns: int):
        """Update history when landmark is visible."""
        prev_pos = hist.last_position

        hist.last_position = result.copy()
        hist.last_visible_time = timestamp_ns / 1e9
        hist.frames_hidden = 0

        method = result.get('method', '')
        if method == 'triangulated':
            hist.state = OcclusionState.VISIBLE
        elif method in ('monocular', 'world_landmarks'):
            hist.state = OcclusionState.PARTIAL
        else:
            hist.state = OcclusionState.VISIBLE

        # Store in position history
        hist.positions.append((
            timestamp_ns,
            result['x'], result['y'], result['z'],
            result.get('visibility', 1.0)
        ))

        # Estimate velocity from last two positions
        if prev_pos is not None and len(hist.positions) >= 2:
            prev_entry = hist.positions[-2]
            dt_s = (timestamp_ns - prev_entry[0]) / 1e9
            if dt_s > 0.001:
                hist.velocity = (
                    (result['x'] - prev_entry[1]) / dt_s,
                    (result['y'] - prev_entry[2]) / dt_s,
                    (result['z'] - prev_entry[3]) / dt_s,
                )

    def _predict_position(self, hist: LandmarkHistory,
                          timestamp_ns: int) -> Dict[str, Any]:
        """Predict landmark position using velocity model with damping."""
        pos = hist.last_position.copy()
        confidence = pos.get('visibility', 0.5)

        # Apply confidence decay
        confidence *= (self.config.confidence_decay_rate ** hist.frames_hidden)
        confidence = max(self.config.min_confidence_output, confidence)

        # Apply velocity prediction (if available)
        if hist.velocity is not None:
            dt_frames = hist.frames_hidden
            # Damp velocity progressively
            damping = self.config.velocity_damping ** dt_frames
            vx, vy, vz = hist.velocity

            # Assume ~33ms per frame at 30fps
            dt_s = dt_frames * (1.0 / 30.0)
            pos['x'] += vx * dt_s * damping
            pos['y'] += vy * dt_s * damping
            pos['z'] += vz * dt_s * damping

        pos['visibility'] = confidence
        pos['method'] = 'predicted'
        pos['predicted_frames'] = hist.frames_hidden

        return pos

    @staticmethod
    def _compute_disagreement(estimates: List[np.ndarray]) -> float:
        """
        Compute inter-camera disagreement as max pairwise distance.

        Args:
            estimates: List of 3D position arrays from different cameras

        Returns:
            Maximum pairwise Euclidean distance in meters
        """
        max_dist = 0.0
        for i in range(len(estimates)):
            for j in range(i + 1, len(estimates)):
                d = float(np.linalg.norm(estimates[i] - estimates[j]))
                if d > max_dist:
                    max_dist = d
        return max_dist

    # ── Queries ──

    def get_occlusion_summary(self) -> Dict[str, int]:
        """Get count of landmarks in each state."""
        counts = defaultdict(int)
        for hist in self._history.values():
            counts[hist.state.value] += 1
        return dict(counts)

    def get_landmark_state(self, lm_id: int) -> Optional[str]:
        """Get the current state of a specific landmark."""
        hist = self._history.get(lm_id)
        return hist.state.value if hist else None

    def get_landmark_history(self, lm_id: int,
                             max_frames: int = 30) -> List[Tuple]:
        """Get recent position history for a landmark."""
        hist = self._history.get(lm_id)
        if not hist:
            return []
        return list(hist.positions)[-max_frames:]

    def get_uncertainty_map(self) -> Dict[int, float]:
        """Get current uncertainty for all tracked landmarks."""
        return {
            lm_id: hist.uncertainty
            for lm_id, hist in self._history.items()
            if hist.is_active
        }

    def get_consistency_score(self) -> float:
        """
        Compute overall multi-view consistency score (0.0 = poor, 1.0 = excellent).
        Based on fraction of landmarks that are VISIBLE (not fallback/predicted).
        """
        if not self._history:
            return 0.0

        active = sum(1 for h in self._history.values() if h.is_active)
        visible = sum(1 for h in self._history.values()
                     if h.state == OcclusionState.VISIBLE)

        if active == 0:
            return 0.0
        return visible / active

    def reset(self):
        """Clear all tracking state."""
        self._history.clear()
        self._frame_count = 0
