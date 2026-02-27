"""
Advanced Kinematics Module
Extended biomechanical analysis beyond the base KinematicsEngine.

Provides:
  - KinematicState dataclass for per-frame state
  - 14+ anatomical joint angles (including trunk, neck, wrist)
  - Multi-view consistency checking
  - Skeleton integrity metrics (bone length validation)
  - Smoothed derivative computation with configurable filters
  - Center-of-mass estimation
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque


# ── MediaPipe Pose Landmark Indices ──

LANDMARK_NAMES = {
    0: 'nose', 1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
    4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
    7: 'left_ear', 8: 'right_ear', 9: 'mouth_left', 10: 'mouth_right',
    11: 'left_shoulder', 12: 'right_shoulder',
    13: 'left_elbow', 14: 'right_elbow',
    15: 'left_wrist', 16: 'right_wrist',
    17: 'left_pinky', 18: 'right_pinky',
    19: 'left_index', 20: 'right_index',
    21: 'left_thumb', 22: 'right_thumb',
    23: 'left_hip', 24: 'right_hip',
    25: 'left_knee', 26: 'right_knee',
    27: 'left_ankle', 28: 'right_ankle',
    29: 'left_heel', 30: 'right_heel',
    31: 'left_foot_index', 32: 'right_foot_index',
}

# Anatomical bone connections for skeleton integrity checks
BONE_CONNECTIONS = [
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (11, 12),            # Shoulders
    (23, 24),            # Hips
    (0, 11), (0, 12),   # Neck approximation
]

# Segment mass fractions (Winter, 2009) for center-of-mass
SEGMENT_MASS_FRACTIONS = {
    'head': 0.081,    # nose (proxy)
    'trunk': 0.497,   # mid-torso
    'upper_arm_l': 0.028, 'upper_arm_r': 0.028,
    'forearm_l': 0.016, 'forearm_r': 0.016,
    'hand_l': 0.006, 'hand_r': 0.006,
    'thigh_l': 0.100, 'thigh_r': 0.100,
    'lower_leg_l': 0.047, 'lower_leg_r': 0.047,
    'foot_l': 0.014, 'foot_r': 0.014,
}


@dataclass
class KinematicState:
    """Complete kinematic state for one frame."""
    timestamp_ns: int = 0
    frame_index: int = 0

    # Joint positions (landmark_id -> (x, y, z))
    positions: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)

    # Joint velocities (landmark_id -> (vx, vy, vz))
    velocities: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)

    # Joint accelerations (landmark_id -> (ax, ay, az))
    accelerations: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)

    # Anatomical angles
    joint_angles: Dict[str, float] = field(default_factory=dict)

    # Angular velocities
    angular_velocities: Dict[str, float] = field(default_factory=dict)

    # Spine vector (midshoulders -> midhips)
    spine_vector: Optional[Tuple[float, float, float]] = None

    # Center of mass
    center_of_mass: Optional[Tuple[float, float, float]] = None

    # Skeleton metrics
    bone_lengths: Dict[str, float] = field(default_factory=dict)

    # Quality
    consistency_score: float = 0.0


class AdvancedKinematics:
    """
    Extended kinematics engine with 14+ angle definitions,
    skeleton integrity validation, and center-of-mass estimation.

    Usage:
        ak = AdvancedKinematics()
        state = ak.process_frame(pose_3d, timestamp_ns)
    """

    # Extended angle definitions: (name, (point_a_id, vertex_id, point_c_id))
    ANGLE_DEFINITIONS = {
        # Standard joint angles
        'elbow_right':    (12, 14, 16),
        'elbow_left':     (11, 13, 15),
        'knee_right':     (24, 26, 28),
        'knee_left':      (23, 25, 27),
        'shoulder_right': (24, 12, 14),
        'shoulder_left':  (23, 11, 13),
        'hip_right':      (12, 24, 26),
        'hip_left':       (11, 23, 25),

        # Extended angles
        'neck_right':     (0, 12, 14),   # Nose-RShoulder-RElbow
        'neck_left':      (0, 11, 13),   # Nose-LShoulder-LElbow
        'trunk_lateral':  (12, 24, 26),  # RShoulder-RHip-RKnee (lateral trunk angle)
        'ankle_right':    (26, 28, 30),  # RKnee-RAnkle-RHeel
        'ankle_left':     (25, 27, 29),  # LKnee-LAnkle-LHeel
        'wrist_right':    (14, 16, 20),  # RElbow-RWrist-RIndex
        'wrist_left':     (13, 15, 19),  # LElbow-LWrist-LIndex
    }

    def __init__(self, max_velocity: float = 6.0,
                 history_size: int = 30,
                 smoothing_alpha: float = 0.5):
        """
        Args:
            max_velocity:    Max linear velocity (m/s) for clamping
            history_size:    Number of frames to keep in history
            smoothing_alpha: EMA alpha for derivative smoothing
        """
        self.max_velocity = max_velocity
        self.smoothing_alpha = smoothing_alpha

        # History
        self._history: deque = deque(maxlen=history_size)
        self._prev_state: Optional[KinematicState] = None

        # Calibration: learned bone lengths (running average)
        self._bone_lengths_avg: Dict[str, float] = {}
        self._bone_length_samples: Dict[str, int] = defaultdict(int)

    def process_frame(
        self,
        pose_3d: Dict[int, Dict[str, Any]],
        timestamp_ns: int,
        min_confidence: float = 0.3
    ) -> KinematicState:
        """
        Process a frame of 3D pose data and compute full kinematic state.

        Args:
            pose_3d:        3D landmarks {lm_id: {x, y, z, visibility, ...}}
            timestamp_ns:   Frame timestamp in nanoseconds
            min_confidence: Minimum visibility to use a landmark

        Returns:
            KinematicState with positions, velocities, angles, etc.
        """
        state = KinematicState(
            timestamp_ns=timestamp_ns,
            frame_index=len(self._history)
        )

        # Extract positions
        for lm_id, lm_data in pose_3d.items():
            lm_id = int(lm_id)
            vis = float(lm_data.get('visibility', lm_data.get('v', 1.0)))
            if vis < min_confidence:
                continue
            state.positions[lm_id] = (
                float(lm_data.get('x', 0)),
                float(lm_data.get('y', 0)),
                float(lm_data.get('z', 0))
            )

        # Compute angles
        state.joint_angles = self._compute_all_angles(state.positions)

        # Compute velocities and accelerations
        if self._prev_state is not None:
            dt = (timestamp_ns - self._prev_state.timestamp_ns) / 1e9
            if dt > 0.001:  # At least 1ms
                state.velocities = self._compute_velocities(
                    state.positions, self._prev_state.positions, dt
                )
                state.accelerations = self._compute_accelerations(
                    state.velocities, self._prev_state.velocities, dt
                )
                state.angular_velocities = self._compute_angular_velocities(
                    state.joint_angles,
                    self._prev_state.joint_angles, dt
                )

        # Spine vector
        state.spine_vector = self._compute_spine_vector(state.positions)

        # Center of mass
        state.center_of_mass = self._estimate_center_of_mass(state.positions)

        # Bone lengths and skeleton integrity
        state.bone_lengths = self._compute_bone_lengths(state.positions)
        state.consistency_score = self._compute_consistency(state.bone_lengths)

        # Update calibration bone lengths
        self._update_bone_calibration(state.bone_lengths)

        # Store history
        self._prev_state = state
        self._history.append(state)

        return state

    def _compute_all_angles(self, positions: Dict[int, Tuple[float, float, float]]) -> Dict[str, float]:
        """Compute all defined anatomical angles."""
        angles = {}
        for name, (a_id, b_id, c_id) in self.ANGLE_DEFINITIONS.items():
            pa = positions.get(a_id)
            pb = positions.get(b_id)
            pc = positions.get(c_id)
            if pa is not None and pb is not None and pc is not None:
                angle = self._angle_3pt(pa, pb, pc)
                angles[name] = angle
        return angles

    @staticmethod
    def _angle_3pt(a: Tuple[float, float, float],
                   b: Tuple[float, float, float],
                   c: Tuple[float, float, float]) -> float:
        """Compute angle at vertex b between segments ba and bc (degrees)."""
        v1 = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
        v2 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])

        n1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        n2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0

        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        cos_angle = max(-1.0, min(1.0, dot / (n1 * n2)))
        return math.degrees(math.acos(cos_angle))

    def _compute_velocities(
        self,
        curr: Dict[int, Tuple[float, float, float]],
        prev: Dict[int, Tuple[float, float, float]],
        dt: float
    ) -> Dict[int, Tuple[float, float, float]]:
        """Compute joint velocities with clamping."""
        velocities = {}
        for lm_id in curr:
            if lm_id not in prev:
                continue
            cp, pp = curr[lm_id], prev[lm_id]
            vx = (cp[0] - pp[0]) / dt
            vy = (cp[1] - pp[1]) / dt
            vz = (cp[2] - pp[2]) / dt

            mag = math.sqrt(vx*vx + vy*vy + vz*vz)
            if mag > self.max_velocity and mag > 1e-9:
                scale = self.max_velocity / mag
                vx, vy, vz = vx * scale, vy * scale, vz * scale

            velocities[lm_id] = (vx, vy, vz)
        return velocities

    @staticmethod
    def _compute_accelerations(
        curr_vel: Dict[int, Tuple[float, float, float]],
        prev_vel: Dict[int, Tuple[float, float, float]],
        dt: float
    ) -> Dict[int, Tuple[float, float, float]]:
        """Compute joint accelerations."""
        accelerations = {}
        for lm_id in curr_vel:
            if lm_id not in prev_vel:
                continue
            cv, pv = curr_vel[lm_id], prev_vel[lm_id]
            accelerations[lm_id] = (
                (cv[0] - pv[0]) / dt,
                (cv[1] - pv[1]) / dt,
                (cv[2] - pv[2]) / dt,
            )
        return accelerations

    @staticmethod
    def _compute_angular_velocities(
        curr_angles: Dict[str, float],
        prev_angles: Dict[str, float],
        dt: float
    ) -> Dict[str, float]:
        """Compute angular velocities (deg/s)."""
        omega = {}
        for name in curr_angles:
            if name in prev_angles:
                omega[name] = (curr_angles[name] - prev_angles[name]) / dt
        return omega

    @staticmethod
    def _compute_spine_vector(
        positions: Dict[int, Tuple[float, float, float]]
    ) -> Optional[Tuple[float, float, float]]:
        """Compute spine vector from midshoulders to midhips."""
        ls = positions.get(11)
        rs = positions.get(12)
        lh = positions.get(23)
        rh = positions.get(24)

        if not all([ls, rs, lh, rh]):
            return None

        ms = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2, (ls[2]+rs[2])/2)
        mh = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2, (lh[2]+rh[2])/2)

        return (ms[0]-mh[0], ms[1]-mh[1], ms[2]-mh[2])

    @staticmethod
    def _estimate_center_of_mass(
        positions: Dict[int, Tuple[float, float, float]]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Estimate whole-body center of mass using segment mass fractions.

        Uses a simplified model with segment proximal points as segment CoM proxies.
        """
        # Segment definitions: (landmark_id, mass_fraction_key)
        segments = [
            (0, 'head'),
            (13, 'upper_arm_l'), (14, 'upper_arm_r'),
            (15, 'forearm_l'), (16, 'forearm_r'),
            (15, 'hand_l'), (16, 'hand_r'),  # Wrist as hand proxy
            (25, 'thigh_l'), (26, 'thigh_r'),
            (27, 'lower_leg_l'), (28, 'lower_leg_r'),
            (27, 'foot_l'), (28, 'foot_r'),
        ]

        # Trunk: average of shoulders and hips
        trunk_ids = [11, 12, 23, 24]
        trunk_positions = [positions.get(i) for i in trunk_ids]
        if all(trunk_positions):
            trunk_com = tuple(sum(p[ax] for p in trunk_positions) / 4 for ax in range(3))
        else:
            trunk_com = None

        total_mass = 0.0
        com = [0.0, 0.0, 0.0]

        # Add trunk
        if trunk_com is not None:
            mass = SEGMENT_MASS_FRACTIONS['trunk']
            for ax in range(3):
                com[ax] += trunk_com[ax] * mass
            total_mass += mass

        # Add other segments
        for lm_id, segment_key in segments:
            pos = positions.get(lm_id)
            if pos is None:
                continue
            mass = SEGMENT_MASS_FRACTIONS.get(segment_key, 0.0)
            for ax in range(3):
                com[ax] += pos[ax] * mass
            total_mass += mass

        if total_mass < 0.01:
            return None

        return (com[0] / total_mass, com[1] / total_mass, com[2] / total_mass)

    @staticmethod
    def _compute_bone_lengths(
        positions: Dict[int, Tuple[float, float, float]]
    ) -> Dict[str, float]:
        """Compute all bone segment lengths."""
        lengths = {}
        for (a, b) in BONE_CONNECTIONS:
            pa, pb = positions.get(a), positions.get(b)
            if pa is not None and pb is not None:
                d = math.sqrt(
                    (pa[0]-pb[0])**2 + (pa[1]-pb[1])**2 + (pa[2]-pb[2])**2
                )
                name = f"{LANDMARK_NAMES.get(a, str(a))}_{LANDMARK_NAMES.get(b, str(b))}"
                lengths[name] = d
        return lengths

    def _update_bone_calibration(self, bone_lengths: Dict[str, float]):
        """Update running average of bone lengths for skeleton integrity."""
        for name, length in bone_lengths.items():
            if length < 0.001:
                continue  # Skip near-zero lengths
            n = self._bone_length_samples[name]
            if n == 0:
                self._bone_lengths_avg[name] = length
            else:
                # Running average
                alpha = 1.0 / min(n + 1, 100)
                self._bone_lengths_avg[name] = (
                    (1 - alpha) * self._bone_lengths_avg[name] + alpha * length
                )
            self._bone_length_samples[name] += 1

    def _compute_consistency(self, bone_lengths: Dict[str, float]) -> float:
        """
        Compute skeleton consistency score (0-1).

        Measures how well current bone lengths match calibrated averages.
        Score of 1.0 means perfect match.
        """
        if not self._bone_lengths_avg or not bone_lengths:
            return 1.0  # No reference yet

        deviations = []
        for name, length in bone_lengths.items():
            avg = self._bone_lengths_avg.get(name)
            if avg is None or avg < 0.001:
                continue
            relative_error = abs(length - avg) / avg
            deviations.append(relative_error)

        if not deviations:
            return 1.0

        mean_deviation = sum(deviations) / len(deviations)
        # Map 0% deviation -> 1.0, 50%+ deviation -> 0.0
        score = max(0.0, 1.0 - 2.0 * mean_deviation)
        return score

    # ── Multi-view Consistency ──

    @staticmethod
    def check_multi_view_consistency(
        per_camera_poses: Dict[str, Dict[int, Dict[str, Any]]],
        max_bone_length_deviation: float = 0.3
    ) -> Dict[str, Any]:
        """
        Check consistency of 3D pose estimates across multiple cameras.

        Compares bone lengths and joint angles from each camera's monocular
        estimate. Large deviations indicate calibration issues or poor views.

        Args:
            per_camera_poses:          {camera_id: {lm_id: {x,y,z,...}}}
            max_bone_length_deviation: Max relative bone length difference (fraction)

        Returns:
            Dict with per-camera scores and overall consistency
        """
        if len(per_camera_poses) < 2:
            return {'overall_consistency': 1.0, 'per_camera': {}}

        # Compute bone lengths per camera
        camera_bones: Dict[str, Dict[str, float]] = {}
        for cam_id, pose in per_camera_poses.items():
            positions = {}
            for lm_id, data in pose.items():
                positions[int(lm_id)] = (
                    float(data.get('x', 0)),
                    float(data.get('y', 0)),
                    float(data.get('z', 0))
                )
            camera_bones[cam_id] = AdvancedKinematics._compute_bone_lengths(positions)

        # Compare all pairs
        cam_ids = list(camera_bones.keys())
        consistent_count = 0
        total_comparisons = 0

        for i in range(len(cam_ids)):
            for j in range(i + 1, len(cam_ids)):
                bones_a = camera_bones[cam_ids[i]]
                bones_b = camera_bones[cam_ids[j]]
                common = set(bones_a.keys()) & set(bones_b.keys())

                for bone_name in common:
                    la, lb = bones_a[bone_name], bones_b[bone_name]
                    if la < 0.001 or lb < 0.001:
                        continue
                    deviation = abs(la - lb) / max(la, lb)
                    total_comparisons += 1
                    if deviation <= max_bone_length_deviation:
                        consistent_count += 1

        overall = consistent_count / total_comparisons if total_comparisons > 0 else 1.0

        return {
            'overall_consistency': round(overall, 3),
            'total_comparisons': total_comparisons,
            'consistent_bones': consistent_count,
            'per_camera_bone_lengths': camera_bones,
        }

    # ── Accessors ──

    def get_history(self, max_frames: int = 30) -> List[KinematicState]:
        """Get recent kinematic state history."""
        return list(self._history)[-max_frames:]

    def get_bone_calibration(self) -> Dict[str, float]:
        """Get calibrated average bone lengths."""
        return dict(self._bone_lengths_avg)

    def reset(self):
        """Reset all state."""
        self._history.clear()
        self._prev_state = None
        self._bone_lengths_avg.clear()
        self._bone_length_samples.clear()

    def export_state_dict(self, state: KinematicState) -> Dict[str, Any]:
        """Export a KinematicState to a flat dict for CSV/DB storage."""
        d: Dict[str, Any] = {
            'timestamp_ns': state.timestamp_ns,
            'frame_index': state.frame_index,
            'consistency_score': state.consistency_score,
        }

        # Positions
        for lm_id, pos in state.positions.items():
            d[f'pos_{lm_id}_x'] = pos[0]
            d[f'pos_{lm_id}_y'] = pos[1]
            d[f'pos_{lm_id}_z'] = pos[2]

        # Velocities
        for lm_id, vel in state.velocities.items():
            mag = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
            d[f'vel_{lm_id}_mag'] = mag

        # Angles
        for name, angle in state.joint_angles.items():
            d[f'angle_{name}_deg'] = angle

        # Angular velocities
        for name, omega in state.angular_velocities.items():
            d[f'omega_{name}_deg_s'] = omega

        # Spine
        if state.spine_vector:
            d['spine_x'] = state.spine_vector[0]
            d['spine_y'] = state.spine_vector[1]
            d['spine_z'] = state.spine_vector[2]

        # Center of mass
        if state.center_of_mass:
            d['com_x'] = state.center_of_mass[0]
            d['com_y'] = state.center_of_mass[1]
            d['com_z'] = state.center_of_mass[2]

        # Bone lengths
        for name, length in state.bone_lengths.items():
            d[f'bone_{name}_m'] = length

        return d
