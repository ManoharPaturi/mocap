from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import math

from config import MAX_LINEAR_VELOCITY


@dataclass
class JointKinematics:
    position: Tuple[float, float, float]
    velocity: Optional[Tuple[float, float, float]] = None
    acceleration: Optional[Tuple[float, float, float]] = None
    velocity_magnitude: Optional[float] = None
    acceleration_magnitude: Optional[float] = None
    confidence: Optional[float] = None


@dataclass
class AngleKinematics:
    angle: float
    angular_velocity: Optional[float] = None


class KinematicsEngine:
    POSE_IDX = {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
    }

    def __init__(self, max_linear_velocity: float = MAX_LINEAR_VELOCITY):
        self.max_linear_velocity = float(max_linear_velocity)
        self.prev_timestamp_ns: Optional[int] = None
        self.prev_positions: Dict[int, Tuple[float, float, float]] = {}
        self.prev_velocities: Dict[int, Tuple[float, float, float]] = {}
        self.prev_angles: Dict[str, float] = {}

    @staticmethod
    def _point_tuple(point: Dict[str, Any]) -> Tuple[float, float, float]:
        return (float(point['x']), float(point['y']), float(point['z']))

    @staticmethod
    def _sub(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    @staticmethod
    def _norm(v: Tuple[float, float, float]) -> float:
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    @staticmethod
    def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def compute_velocity(
        self,
        pos_current: Tuple[float, float, float],
        pos_previous: Tuple[float, float, float],
        delta_t: float
    ) -> Tuple[Tuple[float, float, float], float]:
        if delta_t <= 0:
            return (0.0, 0.0, 0.0), 0.0

        v = (
            (pos_current[0] - pos_previous[0]) / delta_t,
            (pos_current[1] - pos_previous[1]) / delta_t,
            (pos_current[2] - pos_previous[2]) / delta_t,
        )
        mag = self._norm(v)

        if mag > self.max_linear_velocity and mag > 1e-9:
            scale = self.max_linear_velocity / mag
            v = (v[0] * scale, v[1] * scale, v[2] * scale)
            mag = self.max_linear_velocity

        return v, mag

    @staticmethod
    def compute_acceleration(
        vel_current: Tuple[float, float, float],
        vel_previous: Tuple[float, float, float],
        delta_t: float
    ) -> Tuple[Tuple[float, float, float], float]:
        if delta_t <= 0:
            return (0.0, 0.0, 0.0), 0.0

        a = (
            (vel_current[0] - vel_previous[0]) / delta_t,
            (vel_current[1] - vel_previous[1]) / delta_t,
            (vel_current[2] - vel_previous[2]) / delta_t,
        )
        return a, math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

    @staticmethod
    def compute_joint_angle(
        point_a: Tuple[float, float, float],
        point_b: Tuple[float, float, float],
        point_c: Tuple[float, float, float]
    ) -> float:
        v1 = (point_a[0] - point_b[0], point_a[1] - point_b[1], point_a[2] - point_b[2])
        v2 = (point_c[0] - point_b[0], point_c[1] - point_b[1], point_c[2] - point_b[2])

        n1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
        n2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])

        if n1 <= 1e-9 or n2 <= 1e-9:
            return 0.0

        cosine = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
        cosine = max(-1.0, min(1.0, cosine))
        angle_deg = math.degrees(math.acos(cosine))
        return max(0.0, min(180.0, angle_deg))

    @staticmethod
    def compute_angular_velocity(angle_current: float, angle_previous: float, delta_t: float) -> float:
        if delta_t <= 0:
            return 0.0
        return (angle_current - angle_previous) / delta_t

    def _get_spine_vector(self, joints_3d: Dict[int, Dict[str, Any]]) -> Optional[Tuple[float, float, float]]:
        ls = joints_3d.get(self.POSE_IDX['left_shoulder'])
        rs = joints_3d.get(self.POSE_IDX['right_shoulder'])
        lh = joints_3d.get(self.POSE_IDX['left_hip'])
        rh = joints_3d.get(self.POSE_IDX['right_hip'])

        if not all([ls, rs, lh, rh]):
            return None

        ms = (
            (float(ls['x']) + float(rs['x'])) / 2.0,
            (float(ls['y']) + float(rs['y'])) / 2.0,
            (float(ls['z']) + float(rs['z'])) / 2.0,
        )
        mh = (
            (float(lh['x']) + float(rh['x'])) / 2.0,
            (float(lh['y']) + float(rh['y'])) / 2.0,
            (float(lh['z']) + float(rh['z'])) / 2.0,
        )
        return self._sub(ms, mh)

    def _compute_angles(self, joints_3d: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
        def p(name: str) -> Optional[Tuple[float, float, float]]:
            lm = joints_3d.get(self.POSE_IDX[name])
            if not lm:
                return None
            return self._point_tuple(lm)

        angles: Dict[str, float] = {}

        elbow_right = (p('right_shoulder'), p('right_elbow'), p('right_wrist'))
        elbow_left = (p('left_shoulder'), p('left_elbow'), p('left_wrist'))
        knee_right = (p('right_hip'), p('right_knee'), p('right_ankle'))
        knee_left = (p('left_hip'), p('left_knee'), p('left_ankle'))
        shoulder_right = (p('right_hip'), p('right_shoulder'), p('right_elbow'))
        shoulder_left = (p('left_hip'), p('left_shoulder'), p('left_elbow'))
        hip_right = (p('right_shoulder'), p('right_hip'), p('right_knee'))
        hip_left = (p('left_shoulder'), p('left_hip'), p('left_knee'))

        triples = {
            'elbow_right': elbow_right,
            'elbow_left': elbow_left,
            'knee_right': knee_right,
            'knee_left': knee_left,
            'shoulder_right': shoulder_right,
            'shoulder_left': shoulder_left,
            'hip_right': hip_right,
            'hip_left': hip_left,
        }

        for key, triple in triples.items():
            if all(triple):
                angles[key] = self.compute_joint_angle(triple[0], triple[1], triple[2])

        return angles

    def process_frame(
        self,
        joints_3d: Dict[int, Dict[str, Any]],
        timestamp: int,
        compute_derivatives: bool = True,
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        filtered_joints = {
            int(joint_id): joint
            for joint_id, joint in joints_3d.items()
            if float(joint.get('visibility', 1.0)) >= min_confidence
        }

        joint_kinematics: Dict[int, JointKinematics] = {}
        for joint_id, point in filtered_joints.items():
            position = self._point_tuple(point)
            joint_kinematics[joint_id] = JointKinematics(
                position=position,
                confidence=float(point.get('visibility', 1.0))
            )

        dt = None
        if self.prev_timestamp_ns is not None:
            dt = (int(timestamp) - int(self.prev_timestamp_ns)) / 1_000_000_000.0

        if compute_derivatives and dt and dt > 0:
            for joint_id, jkin in joint_kinematics.items():
                prev_pos = self.prev_positions.get(joint_id)
                if prev_pos is None:
                    continue

                velocity, vel_mag = self.compute_velocity(jkin.position, prev_pos, dt)
                jkin.velocity = velocity
                jkin.velocity_magnitude = vel_mag

                prev_vel = self.prev_velocities.get(joint_id)
                if prev_vel is not None:
                    acceleration, acc_mag = self.compute_acceleration(velocity, prev_vel, dt)
                    jkin.acceleration = acceleration
                    jkin.acceleration_magnitude = acc_mag

        raw_angles = self._compute_angles(filtered_joints)
        angle_kinematics: Dict[str, AngleKinematics] = {}
        for angle_name, angle_value in raw_angles.items():
            omega = None
            if compute_derivatives and dt and dt > 0 and angle_name in self.prev_angles:
                omega = self.compute_angular_velocity(angle_value, self.prev_angles[angle_name], dt)
            angle_kinematics[angle_name] = AngleKinematics(angle=angle_value, angular_velocity=omega)

        self.prev_timestamp_ns = int(timestamp)
        self.prev_positions = {
            joint_id: jkin.position for joint_id, jkin in joint_kinematics.items()
        }
        self.prev_velocities = {
            joint_id: jkin.velocity
            for joint_id, jkin in joint_kinematics.items()
            if jkin.velocity is not None
        }
        self.prev_angles = {name: angle.angle for name, angle in angle_kinematics.items()}

        return {
            'timestamp_ns': int(timestamp),
            'joint_kinematics': joint_kinematics,
            'angles': angle_kinematics,
            'spine_vector': self._get_spine_vector(filtered_joints),
        }

    @staticmethod
    def export_frame_data(frame_result: Dict[str, Any]) -> Dict[str, Any]:
        exported: Dict[str, Any] = {
            'timestamp_ns': frame_result.get('timestamp_ns')
        }

        joint_kinematics = frame_result.get('joint_kinematics', {})
        for joint_id, item in joint_kinematics.items():
            prefix = f'joint_{joint_id}'
            exported[f'{prefix}_x'] = item.position[0]
            exported[f'{prefix}_y'] = item.position[1]
            exported[f'{prefix}_z'] = item.position[2]
            exported[f'{prefix}_confidence'] = item.confidence

            if item.velocity is not None:
                exported[f'{prefix}_vx'] = item.velocity[0]
                exported[f'{prefix}_vy'] = item.velocity[1]
                exported[f'{prefix}_vz'] = item.velocity[2]
                exported[f'{prefix}_v'] = item.velocity_magnitude

            if item.acceleration is not None:
                exported[f'{prefix}_ax'] = item.acceleration[0]
                exported[f'{prefix}_ay'] = item.acceleration[1]
                exported[f'{prefix}_az'] = item.acceleration[2]
                exported[f'{prefix}_a'] = item.acceleration_magnitude

        angles = frame_result.get('angles', {})
        for name, item in angles.items():
            exported[f'angle_{name}_deg'] = item.angle
            if item.angular_velocity is not None:
                exported[f'angle_{name}_omega_deg_s'] = item.angular_velocity

        spine = frame_result.get('spine_vector')
        if spine is not None:
            exported['spine_x'] = spine[0]
            exported['spine_y'] = spine[1]
            exported['spine_z'] = spine[2]

        return exported
