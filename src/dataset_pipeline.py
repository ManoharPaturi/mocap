"""
Dataset Pipeline Module
Session lifecycle management, structured export, and research-friendly archival.

Provides:
  - DatasetSession: Manages recording lifecycle with metadata
  - DatasetManager: Multi-session management and listing
  - Export formats: JSON, CSV, HDF5-ready dicts
  - Archive creation with ZIP bundling
  - Raw frame management
"""

import os
import io
import csv
import json
import time
import uuid
import zipfile
import hashlib
import shutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from pathlib import Path


@dataclass
class SessionMetadata:
    """Metadata for a recording session."""
    session_id: str = ''
    start_time: str = ''
    end_time: str = ''
    duration_seconds: float = 0.0
    total_frames: int = 0
    cameras: List[str] = field(default_factory=list)
    calibration_id: Optional[str] = None
    capture_fps: float = 30.0
    resolution: Tuple[int, int] = (1280, 720)
    sync_mode: str = 'loose'
    notes: str = ''
    tags: List[str] = field(default_factory=list)
    schema_version: int = 2
    export_format: str = 'json'

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['resolution'] = list(self.resolution)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SessionMetadata':
        res = d.get('resolution', [1280, 720])
        return cls(
            session_id=d.get('session_id', ''),
            start_time=d.get('start_time', ''),
            end_time=d.get('end_time', ''),
            duration_seconds=d.get('duration_seconds', 0.0),
            total_frames=d.get('total_frames', 0),
            cameras=d.get('cameras', []),
            calibration_id=d.get('calibration_id'),
            capture_fps=d.get('capture_fps', 30.0),
            resolution=tuple(res) if isinstance(res, (list, tuple)) else (1280, 720),
            sync_mode=d.get('sync_mode', 'loose'),
            notes=d.get('notes', ''),
            tags=d.get('tags', []),
            schema_version=d.get('schema_version', 2),
            export_format=d.get('export_format', 'json'),
        )


@dataclass
class FrameRecord:
    """A single frame record in the dataset."""
    frame_index: int
    timestamp: float
    timestamp_ns: int = 0
    camera_id: Optional[str] = None
    pose_2d: Optional[List[Dict]] = None
    pose_3d: Optional[Dict[int, Dict]] = None
    face_landmarks: Optional[List] = None
    hand_landmarks: Optional[List] = None
    kinematics: Optional[Dict[str, Any]] = None
    confidence: Optional[Dict[int, float]] = None
    occlusion_states: Optional[Dict[int, str]] = None
    uncertainty: Optional[Dict[int, float]] = None
    method_map: Optional[Dict[int, str]] = None
    timing: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if v is not None:
                if isinstance(v, dict):
                    # Convert int keys to strings for JSON
                    d[k] = {str(kk): vv for kk, vv in v.items()} if any(isinstance(kk, int) for kk in v) else v
                else:
                    d[k] = v
        return d


class DatasetSession:
    """
    Manages a single recording session with frame accumulation and export.

    Usage:
        session = DatasetSession()
        session.start()
        session.add_frame(frame_record)
        session.stop()
        json_str = session.export_json()
    """

    def __init__(self, output_dir: str = 'results',
                 session_id: Optional[str] = None,
                 metadata: Optional[SessionMetadata] = None):
        self.output_dir = output_dir
        self.session_id = session_id or str(uuid.uuid4())
        self.metadata = metadata or SessionMetadata(session_id=self.session_id)
        self.metadata.session_id = self.session_id

        self._frames: List[FrameRecord] = []
        self._raw_frame_dir: Optional[str] = None
        self._running = False
        self._start_epoch = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    def start(self):
        """Start the recording session."""
        if self._running:
            return

        self._running = True
        self._start_epoch = time.time()
        self.metadata.start_time = datetime.now().isoformat()

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Create raw frame directory
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._raw_frame_dir = os.path.join(
            self.output_dir, 'raw_frames', f'session_{date_str}'
        )

        print(f"[DatasetPipeline] Session started: {self.session_id}")

    def stop(self):
        """Stop the recording session and finalize metadata."""
        if not self._running:
            return

        self._running = False
        self.metadata.end_time = datetime.now().isoformat()
        self.metadata.duration_seconds = time.time() - self._start_epoch
        self.metadata.total_frames = len(self._frames)

        print(f"[DatasetPipeline] Session stopped: {self.session_id} "
              f"({self.num_frames} frames, {self.metadata.duration_seconds:.1f}s)")

    def add_frame(self, record: FrameRecord):
        """Add a frame record to the session."""
        if not self._running:
            return
        record.frame_index = len(self._frames)
        self._frames.append(record)

    def add_frame_dict(self, data: Dict[str, Any]):
        """Add a frame from a raw dict (convenience method)."""
        record = FrameRecord(
            frame_index=len(self._frames),
            timestamp=data.get('timestamp', time.time()),
            timestamp_ns=data.get('timestamp_ns', 0),
            camera_id=data.get('camera_id'),
            pose_2d=data.get('pose_2d'),
            pose_3d=data.get('pose_3d'),
            face_landmarks=data.get('face_landmarks'),
            hand_landmarks=data.get('hand_landmarks'),
            kinematics=data.get('kinematics'),
            confidence=data.get('confidence'),
            occlusion_states=data.get('occlusion_states'),
            uncertainty=data.get('uncertainty'),
            method_map=data.get('method_map'),
            timing=data.get('timing'),
        )
        self.add_frame(record)

    def save_raw_frame(self, frame_bgr, camera_id: str = 'local',
                       frame_number: int = -1):
        """
        Save a raw JPEG frame to disk.

        Args:
            frame_bgr:    BGR numpy array (OpenCV format)
            camera_id:    Camera identifier
            frame_number: Frame number for filename
        """
        if not self._running or self._raw_frame_dir is None:
            return

        try:
            import cv2 as _cv2
            os.makedirs(self._raw_frame_dir, exist_ok=True)

            if frame_number < 0:
                frame_number = len(self._frames)

            filename = f"{camera_id}_{frame_number:06d}.jpg"
            filepath = os.path.join(self._raw_frame_dir, filename)
            _cv2.imwrite(filepath, frame_bgr, [_cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception as e:
            if frame_number % 100 == 0:
                print(f"[DatasetPipeline] Raw frame save error: {e}")

    # ── Export Methods ──

    def export_json(self) -> str:
        """Export session as a structured JSON string."""
        data = {
            'metadata': self.metadata.to_dict(),
            'total_frames': len(self._frames),
            'frames': [f.to_dict() for f in self._frames],
        }
        return json.dumps(data, indent=2, default=str)

    def export_csv(self) -> str:
        """Export session as a flat CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        header = ['frame_index', 'timestamp', 'timestamp_ns']
        # Determine all pose landmark indices present
        all_lm_ids = set()
        for frame in self._frames:
            if frame.pose_3d:
                all_lm_ids.update(frame.pose_3d.keys())

        sorted_lm_ids = sorted(all_lm_ids)
        for lm_id in sorted_lm_ids:
            header.extend([f'lm{lm_id}_x', f'lm{lm_id}_y', f'lm{lm_id}_z',
                          f'lm{lm_id}_vis', f'lm{lm_id}_method'])

        # Add kinematics columns
        header.extend(['elbow_right_deg', 'elbow_left_deg', 'knee_right_deg',
                       'knee_left_deg', 'shoulder_right_deg', 'shoulder_left_deg',
                       'hip_right_deg', 'hip_left_deg'])

        writer.writerow(header)

        # Data rows
        for frame in self._frames:
            row = [frame.frame_index, frame.timestamp, frame.timestamp_ns]

            for lm_id in sorted_lm_ids:
                lm = (frame.pose_3d or {}).get(lm_id)
                if lm:
                    row.extend([
                        lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0),
                        lm.get('visibility', 0.0), lm.get('method', '')
                    ])
                else:
                    row.extend([0.0, 0.0, 0.0, 0.0, 'missing'])

            # Kinematics angles
            kin = frame.kinematics or {}
            angles = kin.get('joint_angles_deg', {})
            for angle_name in ['elbow_right', 'elbow_left', 'knee_right',
                              'knee_left', 'shoulder_right', 'shoulder_left',
                              'hip_right', 'hip_left']:
                row.append(angles.get(angle_name, ''))

            writer.writerow(row)

        return output.getvalue()

    def export_to_file(self, filepath: Optional[str] = None,
                       fmt: str = 'json') -> str:
        """
        Export session to a file.

        Args:
            filepath: Output file path (auto-generated if None)
            fmt:      'json' or 'csv'

        Returns:
            Path to saved file
        """
        if filepath is None:
            date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            ext = 'json' if fmt == 'json' else 'csv'
            filepath = os.path.join(self.output_dir, f'session_{date_str}.{ext}')

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        if fmt == 'csv':
            content = self.export_csv()
        else:
            content = self.export_json()

        with open(filepath, 'w') as f:
            f.write(content)

        print(f"[DatasetPipeline] Exported to {filepath}")
        return filepath

    def archive(self, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Create a ZIP archive of the session.

        Contains:
          - session_metadata.json
          - data.json
          - data.csv
          - raw_frames/ (if saved)

        Returns:
            Path to ZIP file, or None on failure
        """
        out_dir = output_dir or self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_name = f'session_{date_str}_archive.zip'
        zip_path = os.path.join(out_dir, zip_name)

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Metadata
                zf.writestr('session_metadata.json',
                           json.dumps(self.metadata.to_dict(), indent=2))

                # JSON export
                zf.writestr('data.json', self.export_json())

                # CSV export
                zf.writestr('data.csv', self.export_csv())

                # Raw frames
                if self._raw_frame_dir and os.path.isdir(self._raw_frame_dir):
                    for fname in sorted(os.listdir(self._raw_frame_dir)):
                        fpath = os.path.join(self._raw_frame_dir, fname)
                        if os.path.isfile(fpath):
                            zf.write(fpath, os.path.join('raw_frames', fname))

                # Checksum
                checksum = hashlib.sha256(
                    self.export_json().encode('utf-8')
                ).hexdigest()
                zf.writestr('checksum.txt', f'sha256:{checksum}\n')

            print(f"[DatasetPipeline] Archived to {zip_path}")
            return zip_path
        except Exception as e:
            print(f"[DatasetPipeline] Archive error: {e}")
            return None

    def get_summary(self) -> Dict[str, Any]:
        """Get a brief summary of the session."""
        methods = defaultdict(int)
        for frame in self._frames:
            if frame.method_map:
                for method in frame.method_map.values():
                    methods[method] += 1

        return {
            'session_id': self.session_id,
            'frames': len(self._frames),
            'duration_s': self.metadata.duration_seconds,
            'cameras': self.metadata.cameras,
            'methods': dict(methods),
        }


class DatasetManager:
    """
    Manages multiple recording sessions with listing and retrieval.

    Persists session index to a JSON file in the output directory.
    """

    INDEX_FILENAME = 'sessions_index.json'

    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        self._sessions: Dict[str, SessionMetadata] = {}
        self._load_index()

    def _index_path(self) -> str:
        return os.path.join(self.output_dir, self.INDEX_FILENAME)

    def _load_index(self):
        """Load session index from disk."""
        path = self._index_path()
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                for entry in data.get('sessions', []):
                    meta = SessionMetadata.from_dict(entry)
                    self._sessions[meta.session_id] = meta
            except Exception as e:
                print(f"[DatasetManager] Failed to load index: {e}")

    def _save_index(self):
        """Save session index to disk."""
        os.makedirs(self.output_dir, exist_ok=True)
        data = {
            'sessions': [m.to_dict() for m in self._sessions.values()],
            'total_sessions': len(self._sessions),
            'last_updated': datetime.now().isoformat(),
        }
        try:
            with open(self._index_path(), 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[DatasetManager] Failed to save index: {e}")

    def register_session(self, metadata: SessionMetadata):
        """Register a completed session in the index."""
        self._sessions[metadata.session_id] = metadata
        self._save_index()

    def create_session(self, **kwargs) -> DatasetSession:
        """Create and return a new DatasetSession."""
        session = DatasetSession(
            output_dir=self.output_dir,
            **kwargs
        )
        return session

    def finalize_session(self, session: DatasetSession):
        """Stop a session and register it in the index."""
        if session.is_running:
            session.stop()
        self.register_session(session.metadata)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all registered sessions."""
        return [
            {
                'session_id': m.session_id,
                'start_time': m.start_time,
                'end_time': m.end_time,
                'duration_s': m.duration_seconds,
                'frames': m.total_frames,
                'cameras': m.cameras,
                'tags': m.tags,
            }
            for m in sorted(self._sessions.values(),
                           key=lambda m: m.start_time, reverse=True)
        ]

    def get_session_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """Get metadata for a specific session."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Remove a session from the index."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save_index()
            return True
        return False

    @property
    def num_sessions(self) -> int:
        return len(self._sessions)
