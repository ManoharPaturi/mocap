"""
Enhanced Message Format Module
Defines rich payload schemas for multi-camera communication,
with camera metadata, multiple message types, and dataset packaging helpers.
"""

import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False


# ── Constants ──

SCHEMA_VERSION = 2
MESSAGE_MAGIC = 0x4D435632  # "MCV2" in hex


class MessageType(Enum):
    """Types of messages exchanged between camera servers and master."""
    DISCOVERY = 'discovery'
    FRAME_DATA = 'frame_data'
    CLOCK_PING = 'clock_ping'
    CLOCK_PONG = 'clock_pong'
    QUALITY_FEEDBACK = 'quality_feedback'
    CALIBRATION_CMD = 'calibration_cmd'
    STATUS_REPORT = 'status_report'
    GPU_COMPUTE = 'gpu_compute'


# ── Camera Metadata ──

@dataclass
class CameraMetadata:
    """Metadata about a camera's current state."""
    camera_id: str
    ip: str = ''
    capture_fps: float = 30.0
    resolution: Tuple[int, int] = (1280, 720)
    exposure_us: float = 0.0
    gain: float = 1.0
    white_balance: int = 0
    gamma: float = 1.0
    inference_backend: str = 'cpu'
    model_complexity: str = 'FULL'
    calibration_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'camera_id': self.camera_id,
            'ip': self.ip,
            'capture_fps': self.capture_fps,
            'resolution': list(self.resolution),
            'exposure_us': self.exposure_us,
            'gain': self.gain,
            'white_balance': self.white_balance,
            'gamma': self.gamma,
            'inference_backend': self.inference_backend,
            'model_complexity': self.model_complexity,
            'calibration_id': self.calibration_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CameraMetadata':
        res = d.get('resolution', [1280, 720])
        return cls(
            camera_id=d.get('camera_id', ''),
            ip=d.get('ip', ''),
            capture_fps=d.get('capture_fps', 30.0),
            resolution=tuple(res) if isinstance(res, (list, tuple)) else (1280, 720),
            exposure_us=d.get('exposure_us', 0.0),
            gain=d.get('gain', 1.0),
            white_balance=d.get('white_balance', 0),
            gamma=d.get('gamma', 1.0),
            inference_backend=d.get('inference_backend', 'cpu'),
            model_complexity=d.get('model_complexity', 'FULL'),
            calibration_id=d.get('calibration_id'),
        )


# ── Timing Info ──

@dataclass
class TimingInfo:
    """Per-frame timing breakdown (all in milliseconds)."""
    capture_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_pose_ms: float = 0.0
    inference_face_ms: float = 0.0
    inference_hand_ms: float = 0.0
    postprocess_ms: float = 0.0
    encode_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'capture_ms': self.capture_ms,
            'preprocess_ms': self.preprocess_ms,
            'inference_pose_ms': self.inference_pose_ms,
            'inference_face_ms': self.inference_face_ms,
            'inference_hand_ms': self.inference_hand_ms,
            'postprocess_ms': self.postprocess_ms,
            'encode_ms': self.encode_ms,
            'total_ms': self.total_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TimingInfo':
        return cls(**{k: float(d.get(k, 0.0)) for k in [
            'capture_ms', 'preprocess_ms', 'inference_pose_ms',
            'inference_face_ms', 'inference_hand_ms',
            'postprocess_ms', 'encode_ms', 'total_ms'
        ]})


# ── Enhanced Frame Message ──

@dataclass
class EnhancedFrameMessage:
    """
    Rich frame payload sent from camera server to master coordinator.
    Superset of the basic frame_data message format.
    """
    # Header
    schema_version: int = SCHEMA_VERSION
    message_type: str = MessageType.FRAME_DATA.value

    # Identity
    camera_id: str = ''
    frame_number: int = 0
    timestamp_ns: int = 0        # Capture timestamp (nanoseconds)
    sequence_number: int = 0     # Monotonically increasing per camera

    # Detection results
    landmarks: Optional[List[Dict]] = None        # Compact 2D landmarks [{x, y, conf}, ...]
    pose_landmarks: Optional[List] = None          # Full pose landmarks (person list)
    pose_world_landmarks: Optional[List] = None    # MediaPipe world landmarks
    face_landmarks: Optional[List] = None
    hand_landmarks: Optional[List] = None

    # Frame data
    frame_jpeg: Optional[bytes] = None             # JPEG-encoded frame

    # Metadata
    camera_metadata: Optional[Dict] = None
    timing: Optional[Dict] = None
    calibration_id: Optional[str] = None
    capture_fps: float = 30.0

    # GPU compute stats (from detector.py profiling)
    gpu_compute: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        d = {
            'schema_version': self.schema_version,
            'type': self.message_type,
            'camera_id': self.camera_id,
            'frame_number': self.frame_number,
            'timestamp': self.timestamp_ns,
            'sequence_number': self.sequence_number,
            'calibration_id': self.calibration_id,
            'capture_fps': self.capture_fps,
        }

        if self.landmarks is not None:
            d['landmarks'] = self.landmarks
        if self.pose_landmarks is not None:
            d['results'] = {'pose_landmarks': self.pose_landmarks}
            if self.pose_world_landmarks:
                d['results']['pose_world_landmarks'] = self.pose_world_landmarks
            if self.face_landmarks:
                d['results']['face_landmarks'] = self.face_landmarks
            if self.hand_landmarks:
                d['results']['hand_landmarks'] = self.hand_landmarks
        if self.frame_jpeg is not None:
            d['frame_jpeg'] = self.frame_jpeg
        if self.camera_metadata is not None:
            d['camera_metadata'] = self.camera_metadata
        if self.timing is not None:
            d['timing'] = self.timing
        if self.gpu_compute is not None:
            d['gpu_compute'] = self.gpu_compute

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EnhancedFrameMessage':
        """Parse from received dict."""
        msg = cls()
        msg.schema_version = d.get('schema_version', 1)
        msg.message_type = d.get('type', MessageType.FRAME_DATA.value)
        msg.camera_id = d.get('camera_id', '')
        msg.frame_number = d.get('frame_number', 0)
        msg.timestamp_ns = d.get('timestamp', 0)
        msg.sequence_number = d.get('sequence_number', 0)
        msg.calibration_id = d.get('calibration_id')
        msg.capture_fps = d.get('capture_fps', 30.0)
        msg.landmarks = d.get('landmarks')
        msg.frame_jpeg = d.get('frame_jpeg')
        msg.camera_metadata = d.get('camera_metadata')
        msg.timing = d.get('timing')
        msg.gpu_compute = d.get('gpu_compute')

        results = d.get('results', {})
        if isinstance(results, dict):
            msg.pose_landmarks = results.get('pose_landmarks')
            msg.pose_world_landmarks = results.get('pose_world_landmarks')
            msg.face_landmarks = results.get('face_landmarks')
            msg.hand_landmarks = results.get('hand_landmarks')

        return msg

    def serialize(self, use_msgpack: bool = True) -> bytes:
        """Serialize to bytes."""
        d = self.to_dict()
        if use_msgpack and MSGPACK_AVAILABLE:
            return msgpack.packb(d)
        return json.dumps(d).encode('utf-8')

    @classmethod
    def deserialize(cls, data: bytes, use_msgpack: bool = True) -> 'EnhancedFrameMessage':
        """Deserialize from bytes."""
        if use_msgpack and MSGPACK_AVAILABLE:
            d = msgpack.unpackb(data, raw=False)
        else:
            d = json.loads(data.decode('utf-8'))
        return cls.from_dict(d)


# ── Message Builders ──

class MessageBuilder:
    """Factory for creating properly-formatted messages."""

    @staticmethod
    def discovery_message(camera_id: str, ip: str, port: int,
                          metadata: Optional[CameraMetadata] = None) -> Dict[str, Any]:
        """Build a discovery broadcast message."""
        msg = {
            'schema_version': SCHEMA_VERSION,
            'type': MessageType.DISCOVERY.value,
            'camera_id': camera_id,
            'ip': ip,
            'port': port,
            'timestamp': time.time(),
        }
        if metadata:
            msg['camera_metadata'] = metadata.to_dict()
        return msg

    @staticmethod
    def quality_feedback(hints: Dict[int, float],
                         compute_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a quality feedback message from master to cameras."""
        msg = {
            'schema_version': SCHEMA_VERSION,
            'type': MessageType.QUALITY_FEEDBACK.value,
            'timestamp': time.time_ns(),
            'hints': hints,
        }
        if compute_hints:
            msg['compute_hints'] = compute_hints
        return msg

    @staticmethod
    def gpu_compute_report(camera_id: str,
                           inference_times: Dict[str, float],
                           device: str = 'cpu',
                           utilization: float = 0.0) -> Dict[str, Any]:
        """Build a GPU compute stats report."""
        return {
            'schema_version': SCHEMA_VERSION,
            'type': MessageType.GPU_COMPUTE.value,
            'camera_id': camera_id,
            'timestamp': time.time_ns(),
            'device': device,
            'inference_times_ms': inference_times,
            'utilization_pct': utilization,
        }

    @staticmethod
    def status_report(camera_id: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Build a status report message."""
        return {
            'schema_version': SCHEMA_VERSION,
            'type': MessageType.STATUS_REPORT.value,
            'camera_id': camera_id,
            'timestamp': time.time(),
            'stats': stats,
        }


# ── Dataset Packager ──

class DatasetPackager:
    """
    Collects enhanced frame messages and packages them for research export.
    Accumulates frames in memory and exports to structured JSON.
    """

    def __init__(self, session_id: str = ''):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.frames: List[Dict[str, Any]] = []
        self.camera_metadata: Dict[str, Dict] = {}
        self._start_time = time.time()

    def add_frame(self, msg: EnhancedFrameMessage):
        """Add a frame message to the dataset."""
        d = msg.to_dict()
        # Remove binary data for JSON export
        d.pop('frame_jpeg', None)
        self.frames.append(d)

        # Track camera metadata
        if msg.camera_metadata and msg.camera_id:
            self.camera_metadata[msg.camera_id] = msg.camera_metadata

    def export_json(self) -> str:
        """Export dataset as JSON string."""
        dataset = {
            'session_id': self.session_id,
            'start_time': self._start_time,
            'export_time': time.time(),
            'total_frames': len(self.frames),
            'cameras': self.camera_metadata,
            'schema_version': SCHEMA_VERSION,
            'frames': self.frames
        }
        return json.dumps(dataset, indent=2, default=str)

    def compute_checksum(self) -> str:
        """Compute SHA256 checksum of the dataset."""
        data = self.export_json().encode('utf-8')
        return hashlib.sha256(data).hexdigest()

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def duration_seconds(self) -> float:
        if not self.frames:
            return 0.0
        timestamps = [f.get('timestamp', 0) for f in self.frames]
        if not timestamps:
            return 0.0
        span_ns = max(timestamps) - min(timestamps)
        return span_ns / 1e9 if span_ns > 1e6 else 0.0


# ── Utility: pose_3d → landmark list conversion ──

def pose_3d_to_landmark_list(pose_3d: Dict[int, Dict[str, Any]],
                              num_landmarks: int = 33) -> List[Dict[str, Any]]:
    """
    Convert a pose_3d dict (lm_id → {x, y, z, visibility, ...}) to an
    ordered list of 33 landmark dicts compatible with calculations.py.

    Missing landmarks are filled with zeros and visibility=0.

    Args:
        pose_3d: Dict mapping landmark index to 3D coordinates
        num_landmarks: Total expected landmarks (33 for MediaPipe Pose)

    Returns:
        List of length num_landmarks with dicts {x, y, z, v, ...}
    """
    result = []
    for idx in range(num_landmarks):
        lm = pose_3d.get(idx)
        if lm is not None:
            result.append({
                'x': float(lm.get('x', 0.0)),
                'y': float(lm.get('y', 0.0)),
                'z': float(lm.get('z', 0.0)),
                'v': float(lm.get('visibility', lm.get('v', 1.0))),
                'visibility': float(lm.get('visibility', lm.get('v', 1.0))),
                'method': lm.get('method', 'unknown'),
            })
        else:
            result.append({
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'v': 0.0, 'visibility': 0.0,
                'method': 'missing',
            })
    return result
