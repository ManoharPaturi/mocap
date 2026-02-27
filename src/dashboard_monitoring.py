"""
Dashboard Monitoring Module
Real-time performance monitoring for the motion capture pipeline.

Features:
  - LatencyMonitor: tracks 7+ pipeline stages with timing breakdown
  - Bottleneck identification and performance verdicts
  - CameraPlacementVisualizer: top-down view of camera layout + PLY export
  - Pipeline throughput tracking with ring-buffer history
  - Alert system for anomalous conditions
"""

import time
import math
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


# ── Pipeline Stages ──

class PipelineStage:
    """Enumeration of trackable pipeline stages."""
    CAPTURE = 'capture'           # camera frame acquisition
    DETECTION = 'detection'       # MediaPipe pose/face/hand detection
    SERIALIZE = 'serialize'       # msgpack encode + ZMQ send
    NETWORK = 'network'           # network transit (capture→master receive)
    SYNC = 'sync'                 # frame synchronization
    TRIANGULATION = 'triangulation'  # 3D reconstruction
    KINEMATICS = 'kinematics'     # angle / velocity computation
    FUSION = 'fusion'             # occlusion + multi-view fusion
    FILTER = 'filter'             # 1-Euro filter smoothing
    DATABASE = 'database'         # database write
    RENDER = 'render'             # visualization / GUI update

    ALL = [CAPTURE, DETECTION, SERIALIZE, NETWORK, SYNC,
           TRIANGULATION, KINEMATICS, FUSION, FILTER, DATABASE, RENDER]


# ── Data Structures ──

@dataclass
class StageTiming:
    """Timing statistics for a single pipeline stage."""
    stage: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    last_ms: float = 0.0

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    def record(self, duration_ms: float):
        self.count += 1
        self.total_ms += duration_ms
        self.last_ms = duration_ms
        if duration_ms < self.min_ms:
            self.min_ms = duration_ms
        if duration_ms > self.max_ms:
            self.max_ms = duration_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stage': self.stage,
            'count': self.count,
            'mean_ms': round(self.mean_ms, 2),
            'min_ms': round(self.min_ms, 2) if self.min_ms != float('inf') else 0.0,
            'max_ms': round(self.max_ms, 2),
            'last_ms': round(self.last_ms, 2),
        }


@dataclass
class FrameLatencyRecord:
    """End-to-end latency for a single frame through the pipeline."""
    frame_id: int
    timestamp: float
    stage_durations: Dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'frame_id': self.frame_id,
            'total_ms': round(self.total_ms, 2),
            'stages': {k: round(v, 2) for k, v in self.stage_durations.items()},
        }


@dataclass
class PerformanceVerdict:
    """Overall system performance assessment."""
    overall: str  # 'excellent', 'good', 'acceptable', 'poor', 'critical'
    fps: float
    e2e_latency_ms: float
    bottleneck: Optional[str]
    bottleneck_ms: float
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall': self.overall,
            'fps': round(self.fps, 1),
            'e2e_latency_ms': round(self.e2e_latency_ms, 1),
            'bottleneck': self.bottleneck,
            'bottleneck_ms': round(self.bottleneck_ms, 1),
            'warnings': self.warnings,
            'recommendations': self.recommendations,
        }


# ── Latency Monitor ──

class LatencyMonitor:
    """
    Tracks per-stage and end-to-end latency across the motion capture pipeline.

    Usage:
        monitor = LatencyMonitor()

        # Time individual stages
        monitor.start_stage('detection')
        ... detect pose ...
        monitor.end_stage('detection')

        # Or record directly
        monitor.record_stage('network', 12.5)

        # Get bottleneck
        verdict = monitor.get_verdict()
    """

    def __init__(self, history_size: int = 300, alert_threshold_ms: float = 100.0):
        self.history_size = history_size
        self.alert_threshold_ms = alert_threshold_ms

        # Per-stage accumulators
        self.stage_stats: Dict[str, StageTiming] = {
            stage: StageTiming(stage=stage) for stage in PipelineStage.ALL
        }

        # Active timers
        self._active_timers: Dict[str, float] = {}

        # End-to-end frame records (ring buffer)
        self.frame_history: deque = deque(maxlen=history_size)

        # Current frame being assembled
        self._current_frame: Optional[FrameLatencyRecord] = None
        self._frame_counter: int = 0

        # FPS tracking
        self._fps_timestamps: deque = deque(maxlen=120)
        self._current_fps: float = 0.0

        # Alerts
        self.alerts: deque = deque(maxlen=50)

    def begin_frame(self):
        """Start tracking a new frame through the pipeline."""
        self._frame_counter += 1
        now = time.time()
        self._current_frame = FrameLatencyRecord(
            frame_id=self._frame_counter,
            timestamp=now,
        )
        self._fps_timestamps.append(now)
        self._update_fps()

    def start_stage(self, stage: str):
        """Start timing a pipeline stage."""
        self._active_timers[stage] = time.perf_counter()

    def end_stage(self, stage: str):
        """End timing a pipeline stage and record the duration."""
        start = self._active_timers.pop(stage, None)
        if start is None:
            return
        duration_ms = (time.perf_counter() - start) * 1000.0
        self.record_stage(stage, duration_ms)

    def record_stage(self, stage: str, duration_ms: float):
        """Record a stage duration (when timing is done externally)."""
        if stage not in self.stage_stats:
            self.stage_stats[stage] = StageTiming(stage=stage)
        self.stage_stats[stage].record(duration_ms)

        if self._current_frame is not None:
            self._current_frame.stage_durations[stage] = duration_ms

        # Alert on slow stages
        if duration_ms > self.alert_threshold_ms:
            self._add_alert(f"Stage '{stage}' took {duration_ms:.1f}ms (threshold: {self.alert_threshold_ms}ms)")

    def end_frame(self):
        """Finalize the current frame's latency record."""
        if self._current_frame is None:
            return

        total = sum(self._current_frame.stage_durations.values())
        self._current_frame.total_ms = total
        self.frame_history.append(self._current_frame)
        self._current_frame = None

    def _update_fps(self):
        """Compute current FPS from timestamp history."""
        if len(self._fps_timestamps) < 2:
            return
        dt = self._fps_timestamps[-1] - self._fps_timestamps[0]
        if dt > 0:
            self._current_fps = (len(self._fps_timestamps) - 1) / dt

    def _add_alert(self, message: str):
        """Add a performance alert."""
        self.alerts.append({
            'time': time.time(),
            'message': message,
            'frame_id': self._frame_counter,
        })

    def get_fps(self) -> float:
        """Get current frames per second."""
        return self._current_fps

    def get_stage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all pipeline stages."""
        return {
            stage: stats.to_dict()
            for stage, stats in self.stage_stats.items()
            if stats.count > 0
        }

    def get_bottleneck(self) -> Tuple[Optional[str], float]:
        """Identify the slowest pipeline stage."""
        worst_stage = None
        worst_mean = 0.0

        for stage, stats in self.stage_stats.items():
            if stats.count > 0 and stats.mean_ms > worst_mean:
                worst_mean = stats.mean_ms
                worst_stage = stage

        return worst_stage, worst_mean

    def get_e2e_latency(self) -> float:
        """Get average end-to-end latency (ms)."""
        if not self.frame_history:
            return 0.0
        recent = list(self.frame_history)[-30:]
        return sum(f.total_ms for f in recent) / len(recent)

    def get_verdict(self) -> PerformanceVerdict:
        """Generate an overall performance assessment."""
        fps = self.get_fps()
        e2e = self.get_e2e_latency()
        bottleneck, bottleneck_ms = self.get_bottleneck()

        warnings = []
        recommendations = []

        # FPS assessment
        if fps < 10:
            warnings.append(f"Very low FPS: {fps:.1f}")
            recommendations.append("Reduce camera resolution or detection model complexity.")
        elif fps < 20:
            warnings.append(f"Low FPS: {fps:.1f}")

        # Latency assessment
        if e2e > 100:
            warnings.append(f"High end-to-end latency: {e2e:.1f}ms")
        if e2e > 200:
            recommendations.append("Consider disabling some processing stages or using lighter models.")

        # Bottleneck recommendations
        if bottleneck == PipelineStage.DETECTION and bottleneck_ms > 30:
            recommendations.append("Switch to 'lite' pose model for faster detection.")
        elif bottleneck == PipelineStage.NETWORK and bottleneck_ms > 20:
            recommendations.append("Check network connection; consider reducing payload size.")
        elif bottleneck == PipelineStage.TRIANGULATION and bottleneck_ms > 10:
            recommendations.append("Fewer landmarks or skip low-visibility points.")
        elif bottleneck == PipelineStage.DATABASE and bottleneck_ms > 15:
            recommendations.append("Enable async database writes or reduce save frequency.")

        # Overall grade
        if fps >= 28 and e2e < 40:
            overall = 'excellent'
        elif fps >= 22 and e2e < 70:
            overall = 'good'
        elif fps >= 15 and e2e < 120:
            overall = 'acceptable'
        elif fps >= 8:
            overall = 'poor'
        else:
            overall = 'critical'

        return PerformanceVerdict(
            overall=overall,
            fps=fps,
            e2e_latency_ms=e2e,
            bottleneck=bottleneck,
            bottleneck_ms=bottleneck_ms,
            warnings=warnings,
            recommendations=recommendations,
        )

    def get_recent_history(self, n: int = 30) -> List[Dict[str, Any]]:
        """Get the last N frame latency records."""
        recent = list(self.frame_history)[-n:]
        return [r.to_dict() for r in recent]

    def reset(self):
        """Reset all counters and history."""
        for stats in self.stage_stats.values():
            stats.count = 0
            stats.total_ms = 0.0
            stats.min_ms = float('inf')
            stats.max_ms = 0.0
            stats.last_ms = 0.0
        self.frame_history.clear()
        self._fps_timestamps.clear()
        self._current_fps = 0.0
        self._frame_counter = 0
        self.alerts.clear()


# ── Camera Placement Visualizer ──

@dataclass
class CameraPlacement:
    """Camera position and orientation in the measurement volume."""
    camera_id: str
    position: Tuple[float, float, float]  # (x, y, z) in meters
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # target point
    fov_deg: float = 60.0
    label: str = ''
    status: str = 'active'  # 'active', 'disconnected', 'calibrating'

    @property
    def direction(self) -> Tuple[float, float, float]:
        """Unit direction vector from position to look_at."""
        dx = self.look_at[0] - self.position[0]
        dy = self.look_at[1] - self.position[1]
        dz = self.look_at[2] - self.position[2]
        mag = math.sqrt(dx * dx + dy * dy + dz * dz)
        if mag < 1e-9:
            return (0.0, 0.0, -1.0)
        return (dx / mag, dy / mag, dz / mag)


class CameraPlacementVisualizer:
    """
    Manages and visualizes multi-camera placement configuration.

    Supports:
      - Add/remove cameras with 3D position + orientation
      - Compute overlap zones and coverage metrics
      - Export to PLY for 3D visualization
      - Generate top-down ASCII diagram
    """

    def __init__(self):
        self.cameras: Dict[str, CameraPlacement] = {}
        self.measurement_volume: Tuple[float, float, float] = (3.0, 3.0, 2.5)  # WxDxH meters

    def add_camera(self, camera_id: str,
                   position: Tuple[float, float, float],
                   look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                   fov_deg: float = 60.0,
                   label: str = '') -> CameraPlacement:
        """Add or update a camera placement."""
        cam = CameraPlacement(
            camera_id=camera_id,
            position=position,
            look_at=look_at,
            fov_deg=fov_deg,
            label=label or camera_id,
        )
        self.cameras[camera_id] = cam
        return cam

    def remove_camera(self, camera_id: str):
        """Remove a camera."""
        self.cameras.pop(camera_id, None)

    def set_camera_status(self, camera_id: str, status: str):
        """Update a camera's connection status."""
        if camera_id in self.cameras:
            self.cameras[camera_id].status = status

    def compute_baseline(self, cam_a: str, cam_b: str) -> float:
        """Compute baseline distance between two cameras (meters)."""
        a = self.cameras.get(cam_a)
        b = self.cameras.get(cam_b)
        if not a or not b:
            return 0.0
        dx = a.position[0] - b.position[0]
        dy = a.position[1] - b.position[1]
        dz = a.position[2] - b.position[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def compute_stereo_angle(self, cam_a: str, cam_b: str) -> float:
        """
        Compute the convergence angle between two cameras looking at the origin.
        Returns angle in degrees. Ideal is 60-90°.
        """
        a = self.cameras.get(cam_a)
        b = self.cameras.get(cam_b)
        if not a or not b:
            return 0.0

        dir_a = a.direction
        dir_b = b.direction
        dot = dir_a[0] * dir_b[0] + dir_a[1] * dir_b[1] + dir_a[2] * dir_b[2]
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))

    def get_coverage_summary(self) -> Dict[str, Any]:
        """Summarize camera coverage characteristics."""
        if len(self.cameras) < 2:
            return {
                'num_cameras': len(self.cameras),
                'baselines': [],
                'stereo_angles': [],
                'quality': 'insufficient' if len(self.cameras) < 2 else 'unknown',
            }

        cam_ids = list(self.cameras.keys())
        baselines = []
        angles = []

        for i in range(len(cam_ids)):
            for j in range(i + 1, len(cam_ids)):
                bl = self.compute_baseline(cam_ids[i], cam_ids[j])
                ang = self.compute_stereo_angle(cam_ids[i], cam_ids[j])
                baselines.append({
                    'pair': (cam_ids[i], cam_ids[j]),
                    'baseline_m': round(bl, 3),
                })
                angles.append({
                    'pair': (cam_ids[i], cam_ids[j]),
                    'angle_deg': round(ang, 1),
                })

        # Quality assessment
        avg_baseline = sum(b['baseline_m'] for b in baselines) / len(baselines)
        avg_angle = sum(a['angle_deg'] for a in angles) / len(angles)

        if avg_baseline >= 0.5 and 40 <= avg_angle <= 120:
            quality = 'good'
        elif avg_baseline >= 0.3 and 20 <= avg_angle <= 140:
            quality = 'acceptable'
        else:
            quality = 'suboptimal'

        return {
            'num_cameras': len(self.cameras),
            'baselines': baselines,
            'stereo_angles': angles,
            'avg_baseline_m': round(avg_baseline, 3),
            'avg_stereo_angle_deg': round(avg_angle, 1),
            'quality': quality,
        }

    def export_ply(self, filepath: str):
        """
        Export camera positions as a PLY point cloud for 3D visualization.
        Each camera is a point with a direction indicator.
        """
        points = []
        for cam in self.cameras.values():
            # Camera position (red)
            x, y, z = cam.position
            points.append((x, y, z, 255, 0, 0))

            # Look-at direction indicator (green, 0.3m along direction)
            d = cam.direction
            tx = x + d[0] * 0.3
            ty = y + d[1] * 0.3
            tz = z + d[2] * 0.3
            points.append((tx, ty, tz, 0, 255, 0))

        header = [
            'ply',
            'format ascii 1.0',
            f'element vertex {len(points)}',
            'property float x',
            'property float y',
            'property float z',
            'property uchar red',
            'property uchar green',
            'property uchar blue',
            'end_header',
        ]

        with open(filepath, 'w') as f:
            f.write('\n'.join(header) + '\n')
            for p in points:
                f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {p[3]} {p[4]} {p[5]}\n')

        logger.info(f"Camera layout exported to PLY: {filepath}")

    def export_json(self, filepath: str):
        """Export camera placements as JSON."""
        data = {
            'measurement_volume': self.measurement_volume,
            'cameras': {},
        }
        for cam_id, cam in self.cameras.items():
            data['cameras'][cam_id] = {
                'position': cam.position,
                'look_at': cam.look_at,
                'fov_deg': cam.fov_deg,
                'label': cam.label,
                'status': cam.status,
            }
        data['coverage'] = self.get_coverage_summary()

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Camera layout exported to JSON: {filepath}")

    def generate_topdown_ascii(self, width: int = 40, height: int = 20) -> str:
        """
        Generate a top-down ASCII diagram of camera positions (X-Z plane).
        Origin is at center. Scale adapts to measurement volume.
        """
        vol_w, vol_d, _ = self.measurement_volume
        half_w = vol_w / 2
        half_d = vol_d / 2

        grid = [['.' for _ in range(width)] for _ in range(height)]

        # Draw border
        for x in range(width):
            grid[0][x] = '-'
            grid[height - 1][x] = '-'
        for y in range(height):
            grid[y][0] = '|'
            grid[y][width - 1] = '|'

        # Mark origin
        cx, cy = width // 2, height // 2
        grid[cy][cx] = '+'

        # Place cameras
        for cam in self.cameras.values():
            px, _, pz = cam.position
            # Map world coords to grid
            gx = int((px + half_w) / vol_w * (width - 2)) + 1
            gy = int((pz + half_d) / vol_d * (height - 2)) + 1
            gx = max(1, min(width - 2, gx))
            gy = max(1, min(height - 2, gy))

            marker = cam.camera_id[0].upper() if cam.camera_id else 'C'
            if cam.status == 'disconnected':
                marker = 'x'
            grid[gy][gx] = marker

        lines = [''.join(row) for row in grid]
        lines.append(f"  + = origin, letters = cameras, x = disconnected")
        lines.append(f"  Volume: {vol_w}m × {vol_d}m")
        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all placements."""
        return {
            cam_id: {
                'position': cam.position,
                'look_at': cam.look_at,
                'fov_deg': cam.fov_deg,
                'label': cam.label,
                'status': cam.status,
            }
            for cam_id, cam in self.cameras.items()
        }


# ── Combined Dashboard Monitor ──

class DashboardMonitor:
    """
    Combines latency monitoring and camera placement visualization
    into a single dashboard-ready interface.

    Usage:
        monitor = DashboardMonitor()
        monitor.latency.begin_frame()
        monitor.latency.start_stage('detection')
        ...
        monitor.latency.end_stage('detection')
        monitor.latency.end_frame()

        monitor.cameras.add_camera('cam0', (1.0, 0.0, 1.5))
        verdict = monitor.get_status()
    """

    def __init__(self, history_size: int = 300):
        self.latency = LatencyMonitor(history_size=history_size)
        self.cameras = CameraPlacementVisualizer()
        self._start_time = time.time()

    def get_status(self) -> Dict[str, Any]:
        """Get a complete dashboard status snapshot."""
        verdict = self.latency.get_verdict()
        coverage = self.cameras.get_coverage_summary()

        return {
            'uptime_seconds': round(time.time() - self._start_time, 1),
            'performance': verdict.to_dict(),
            'stage_stats': self.latency.get_stage_stats(),
            'camera_coverage': coverage,
            'active_cameras': sum(
                1 for c in self.cameras.cameras.values() if c.status == 'active'
            ),
            'total_cameras': len(self.cameras.cameras),
            'alerts': list(self.latency.alerts)[-10:],
        }

    def get_summary_text(self) -> str:
        """Generate a human-readable status summary."""
        v = self.latency.get_verdict()
        lines = [
            f"Performance: {v.overall.upper()}",
            f"FPS: {v.fps:.1f}  |  Latency: {v.e2e_latency_ms:.1f}ms",
            f"Bottleneck: {v.bottleneck or 'none'} ({v.bottleneck_ms:.1f}ms)",
            f"Cameras: {len(self.cameras.cameras)} configured",
        ]
        if v.warnings:
            lines.append(f"Warnings: {'; '.join(v.warnings)}")
        if v.recommendations:
            lines.append(f"Tips: {'; '.join(v.recommendations)}")
        return '\n'.join(lines)
