"""
Master Coordinator Module
Runs on the master laptop to discover cameras, receive frame data, and coordinate multi-view fusion.
"""

import zmq
import msgpack
import json
import time
from typing import Dict, List, Optional, Any
from threading import Thread, Event
from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np
from config import (
    DISCOVERY_PORT, DATA_PORT, NUM_CAMERAS, COMPRESS_NETWORK_DATA,
    FRAME_BUFFER_SIZE, SYNC_TIME_THRESHOLD_MS, CALIBRATION_FILE
)
from src.stereo_calibration import StereoCalibration
from src.triangulation import Triangulator


@dataclass
class CameraInfo:
    """Information about a discovered camera."""
    camera_id: str
    ip: str
    port: int
    last_seen: float


@dataclass
class FrameData:
    """Frame data from a single camera."""
    camera_id: str
    frame_number: int
    timestamp: float  # nanoseconds
    results: Dict[str, Any]
    received_at: float


class MasterCoordinator:
    """
    Master Coordinator for multi-camera setup.
    Discovers cameras, receives frame data, and synchronizes frames across cameras.
    """
    
    def __init__(self, num_cameras: int = NUM_CAMERAS):
        """
        Initialize master coordinator.
        
        Args:
            num_cameras: Expected number of cameras
        """
        self.num_cameras = num_cameras
        self.running = False
        self.discovered_cameras: Dict[str, CameraInfo] = {}
        
        # Frame buffers per camera
        self.frame_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=FRAME_BUFFER_SIZE)
        )
        
        # Threads
        self.discovery_thread = None
        self.data_thread = None
        self.stop_event = Event()
        
        # ZMQ context and sockets
        self.context = zmq.Context()
        self.discovery_socket = None
        self.data_sockets: Dict[str, zmq.Socket] = {}
        
        # Statistics
        self.stats = {
            'frames_received': defaultdict(int),
            'frames_synced': 0,
            'sync_failures': 0
        }
        
        print("[MasterCoordinator] Initialized")
    
    def start(self):
        """Start the master coordinator."""
        if self.running:
            print("[MasterCoordinator] Already running")
            return
        
        self.running = True
        self.stop_event.clear()
        
        # Setup discovery socket
        self._setup_discovery_socket()
        
        # Start discovery listener thread
        self.discovery_thread = Thread(target=self._discovery_listener, daemon=True)
        self.discovery_thread.start()
        
        # Start data receiver thread
        self.data_thread = Thread(target=self._data_receiver, daemon=True)
        self.data_thread.start()
        
        # Initialize Triangulator with default calibration (if no file exists)
        try:
            # Try to load existing calibration
            calibration = StereoCalibration()
            try:
                calibration.load_calibration(CALIBRATION_FILE)
                print(f"[MasterCoordinator] Loaded calibration from {CALIBRATION_FILE}")
            except (FileNotFoundError, Exception):
                print("[MasterCoordinator] No calibration file found. Using DEFAULT calibration (1.0m baseline).")
                calibration.create_default_calibration(width=1280, height=720)
            
            self.triangulator = Triangulator(calibration)
        except Exception as e:
            print(f"[MasterCoordinator] Failed to initialize triangulator: {e}")
            self.triangulator = None
        
        print(f"[MasterCoordinator] Started, listening for {self.num_cameras} cameras")
    
    def stop(self):
        """Stop the master coordinator."""
        if not self.running:
            return
        
        print("[MasterCoordinator] Stopping...")
        self.running = False
        self.stop_event.set()
        
        # Wait for threads
        if self.discovery_thread:
            self.discovery_thread.join(timeout=2.0)
        if self.data_thread:
            self.data_thread.join(timeout=2.0)
        
        # Close sockets
        if self.discovery_socket:
            self.discovery_socket.close()
        
        for socket in self.data_sockets.values():
            socket.close()
        
        self.context.term()
        print("[MasterCoordinator] Stopped")
    
    def _setup_discovery_socket(self):
        """Setup ZMQ socket for receiving discovery broadcasts."""
        # NOTE: Discovery uses TCP instead of UDP because ZMQ doesn't support
        # UDP with PUB/SUB sockets. We'll need to know camera IPs in advance
        # or use a different discovery mechanism.
        # For now, we'll connect to known camera IPs
        pass  # Discovery will be handled differently
    
    def _discovery_listener(self):
        """
        Listen for camera connections.
        Since ZMQ PUB/SUB doesn't support UDP broadcast, we use a simpler approach:
        - Cameras are configured with known IPs in config.py
        - Master connects to these IPs directly
        """
        # This method is kept for compatibility but discovery is now manual
        # via discover_cameras_manual()
        while self.running and not self.stop_event.is_set():
            time.sleep(1.0)
    
    def discover_cameras_manual(self, camera_ips: List[str]):
        """
        Manually connect to cameras at specified IP addresses.
        
        Args:
            camera_ips: List of camera IP addresses (e.g., ['192.168.1.101', '192.168.1.102'])
        """
        for i, ip in enumerate(camera_ips):
            camera_id = f"cam_{i}"
            print(f"[MasterCoordinator] Connecting to camera at {ip}...")
            
            # Try to connect to discovery port first to verify camera is alive
            try:
                test_socket = self.context.socket(zmq.SUB)
                test_socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout
                test_socket.connect(f"tcp://{ip}:{DISCOVERY_PORT}")
                test_socket.setsockopt(zmq.SUBSCRIBE, b"")
                
                # Try to receive one message
                try:
                    data = test_socket.recv()
                    if COMPRESS_NETWORK_DATA:
                        msg = msgpack.unpackb(data)
                    else:
                        msg = json.loads(data.decode('utf-8'))
                    
                    if msg.get('type') == 'discovery':
                        camera_id = msg.get('camera_id', camera_id)
                        print(f"[MasterCoordinator] Discovered camera: {camera_id} at {ip}")
                except zmq.Again:
                    # Timeout, but camera might still be there
                    print(f"[MasterCoordinator] No discovery message, but connecting anyway...")
                
                test_socket.close()
                
                # Add to discovered cameras
                self.discovered_cameras[camera_id] = CameraInfo(
                    camera_id=camera_id,
                    ip=ip,
                    port=DATA_PORT,
                    last_seen=time.time()
                )
                
                # Connect to data stream
                self._connect_to_camera(camera_id, ip, DATA_PORT)
                
            except Exception as e:
                print(f"[MasterCoordinator] Failed to connect to {ip}: {e}")
    
    def _process_discovery(self, msg: Dict[str, Any]):
        """Process a camera discovery message."""
        camera_id = msg.get('camera_id')
        ip = msg.get('ip')
        port = msg.get('port')
        
        if not camera_id or not ip or not port:
            return
        
        # Update or add camera info
        if camera_id not in self.discovered_cameras:
            print(f"[MasterCoordinator] Discovered camera: {camera_id} at {ip}:{port}")
            
            # Connect to camera's data stream
            self._connect_to_camera(camera_id, ip, port)
        
        # Update last seen time
        self.discovered_cameras[camera_id] = CameraInfo(
            camera_id=camera_id,
            ip=ip,
            port=port,
            last_seen=time.time()
        )
    
    def _connect_to_camera(self, camera_id: str, ip: str, port: int):
        """Connect to a camera's data stream."""
        try:
            socket = self.context.socket(zmq.SUB)
            socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s receive timeout
            socket.setsockopt(zmq.LINGER, 0)        # Don't block on close
            socket.connect(f"tcp://{ip}:{port}")
            socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all topics
            # ZMQ slow joiner fix: give the SUB socket time to stabilize
            time.sleep(0.5)
            self.data_sockets[camera_id] = socket
            print(f"[MasterCoordinator] Connected to {camera_id} data stream at {ip}:{port}")
        except Exception as e:
            print(f"[MasterCoordinator] Error connecting to {camera_id}: {e}")
    
    def _data_receiver(self):
        """Receive frame data from all connected cameras."""
        print("[MasterCoordinator] Data receiver started")
        msg_count = 0
        last_log_time = time.time()
        
        while self.running and not self.stop_event.is_set():
            try:
                if not self.data_sockets:
                    time.sleep(0.1)  # Wait until sockets are connected
                    continue

                # Poll all data sockets
                for camera_id, socket in list(self.data_sockets.items()):
                    if socket.poll(timeout=10):
                        data = socket.recv()
                        msg_count += 1
                        
                        # Debug first 5 messages
                        if msg_count <= 5:
                            print(f"[MasterCoordinator] ✅ Receiving data from {camera_id} (msg #{msg_count})")
                        
                        # Deserialize
                        if COMPRESS_NETWORK_DATA:
                            msg = msgpack.unpackb(data, raw=False)
                        else:
                            msg = json.loads(data.decode('utf-8'))
                        
                        # Process frame data
                        if msg.get('type') == 'frame_data':
                            self._process_frame_data(msg)

                # Periodic heartbeat log every 5 seconds
                now = time.time()
                if now - last_log_time >= 5.0:
                    total = sum(self.stats['frames_received'].values())
                    print(f"[MasterCoordinator] Heartbeat: {total} total frames received from {list(self.data_sockets.keys())}")
                    last_log_time = now
                
            except Exception as e:
                if self.running:
                    print(f"[MasterCoordinator] Data receiver error: {e}")
    
    def _process_frame_data(self, msg: Dict[str, Any]):
        """Process incoming frame data from a camera."""
        camera_id = msg.get('camera_id')
        frame_number = msg.get('frame_number')
        timestamp = msg.get('timestamp')
        results = msg.get('results')
        
        # Allow empty results dict (no detection) - still valid for sync
        if not all([camera_id, frame_number is not None, timestamp, results is not None]):
            if self.stats['frames_received'].get(camera_id, 0) < 3:
                print(f"[MasterCoordinator] Skipping invalid frame from {camera_id}")
            return
        
        # Include frame_jpeg in results for decoding later
        if 'frame_jpeg' in msg and msg['frame_jpeg']:
            results['frame_jpeg'] = msg['frame_jpeg']
        
        # Create FrameData object
        frame_data = FrameData(
            camera_id=camera_id,
            frame_number=frame_number,
            timestamp=timestamp,
            results=results,
            received_at=time.time()
        )
        
        # Add to buffer
        self.frame_buffers[camera_id].append(frame_data)
        self.stats['frames_received'][camera_id] += 1
        
        # Debug only first 3 frames
        if self.stats['frames_received'][camera_id] <= 3:
            has_jpeg = 'frame_jpeg' in results
            print(f"[MasterCoordinator] Buffered frame {frame_number} from {camera_id} (JPEG: {has_jpeg})")
    
    def get_synchronized_batch(self) -> Optional[List[FrameData]]:
        """
        Get a synchronized batch of frames from all cameras.
        Returns None if not all cameras have matching frames.
        
        Returns:
            List of FrameData objects, one per camera, with matching timestamps
        """
        if len(self.frame_buffers) < self.num_cameras:
            # Not all cameras connected yet
            return None
        
        # Get the oldest frame from each buffer
        reference_frames = {}
        for camera_id, buffer in self.frame_buffers.items():
            if len(buffer) == 0:
                return None  # At least one camera has no frames
            reference_frames[camera_id] = buffer[0]
        
        # Find the camera with the latest timestamp (slowest camera)
        latest_timestamp = max(f.timestamp for f in reference_frames.values())
        
        # Try to match frames within time threshold
        synced_frames = []
        threshold_ns = SYNC_TIME_THRESHOLD_MS * 1_000_000  # Convert ms to ns
        
        for camera_id, buffer in self.frame_buffers.items():
            matched = False
            
            for frame in buffer:
                time_diff = abs(frame.timestamp - latest_timestamp)
                
                if time_diff <= threshold_ns:
                    synced_frames.append(frame)
                    matched = True
                    break
            
            if not matched:
                # No matching frame found for this camera
                self.stats['sync_failures'] += 1
                return None
        
        if len(synced_frames) == self.num_cameras:
            # We found a match for every camera!
            
            # Remove used frames from buffers
            for frame in synced_frames:
                # Remove this frame and all older frames
                while len(self.frame_buffers[frame.camera_id]) > 0:
                    old_frame = self.frame_buffers[frame.camera_id].popleft()
                    if old_frame == frame:
                        break
            
            self.stats['frames_synced'] += 1
            return synced_frames
            
        return None

    def _extract_pose_landmarks(self, results: Dict[str, Any]) -> Optional[list]:
        """
        Normalize pose landmarks from either a raw MediaPipe result or a serialized dict.
        
        Local camera: results['pose'] is a PoseLandmarkerResult object
        Remote camera: results['pose'] is a list of dicts or list of lists
        
        Returns: list of dicts with keys 'x', 'y', 'z', 'visibility', or None
        """
        pose = results.get('pose')
        if not pose:
            return None

        # Case 1: Raw MediaPipe PoseLandmarkerResult object
        # It has a .pose_landmarks attribute (list of lists of NormalizedLandmark)
        if hasattr(pose, 'pose_landmarks'):
            if pose.pose_landmarks and len(pose.pose_landmarks) > 0:
                # pose_landmarks[0] = first detected person
                return [
                    {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                    for lm in pose.pose_landmarks[0]
                ]
            return None

        # Case 2: Already-serialized list (from remote camera via network)
        if isinstance(pose, list):
            if len(pose) == 0:
                return None
            first = pose[0]
            # Sub-case 2a: list of dicts (JSON deserialized)
            if isinstance(first, dict):
                return pose
            # Sub-case 2b: list of lists [[x,y,z,vis], ...]
            if isinstance(first, (list, tuple)):
                return [
                    {'x': lm[0], 'y': lm[1], 'z': lm[2], 'visibility': lm[3] if len(lm) > 3 else 1.0}
                    for lm in pose
                ]

        # Also handle results['pose_landmarks'] key (alternative serialization)
        pose_lms = results.get('pose_landmarks')
        if pose_lms and isinstance(pose_lms, list) and len(pose_lms) > 0:
            first = pose_lms[0]
            if isinstance(first, dict):
                return pose_lms
            if isinstance(first, list) and len(first) > 0:
                if isinstance(first[0], dict):
                    return first  # pose_landmarks[0] = person 0's landmarks

        return None

    def get_synced_3d_pose(self, synced_frames: List['FrameData']) -> Optional[Dict[str, Any]]:
        """
        Compute 3D pose from a list of synchronized 2D frames.
        
        Args:
            synced_frames: List of FrameData objects (must be synced)
            
        Returns:
            Dictionary containing 3D landmarks if successful, None otherwise
        """
        if not self.triangulator:
            return None
            
        # Collect 2D observations for each landmark
        landmark_observations = defaultdict(dict)
        
        for frame in synced_frames:
            landmarks = self._extract_pose_landmarks(frame.results)
            if not landmarks:
                continue
                
            # Iterate through all pose landmarks
            for idx, lm in enumerate(landmarks):
                # lm is now always a dict
                vis = lm.get('visibility', 1.0)
                if vis > 0.5:
                    w, h = 1280, 720
                    x_px = lm['x'] * w
                    y_px = lm['y'] * h
                    landmark_observations[idx][frame.camera_id] = np.array([x_px, y_px])
        
        # Triangulate each landmark
        landmarks_3d = {}
        for lm_id, observations in landmark_observations.items():
            if len(observations) >= 2:
                point_3d = self.triangulator.triangulate_point(observations)
                if point_3d:
                    landmarks_3d[lm_id] = {
                        'x': point_3d.x,
                        'y': point_3d.y,
                        'z': point_3d.z,
                        'visibility': point_3d.confidence
                    }
        
        if not landmarks_3d:
            return None
            
        return {'pose_3d': landmarks_3d}
    
    def discover_cameras(self, timeout: float = 5.0) -> List[CameraInfo]:
        """
        Wait for cameras to be discovered.
        
        Args:
            timeout: Maximum time to wait (seconds)
            
        Returns:
            List of discovered cameras
        """
        if not self.running:
            self.start()
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if len(self.discovered_cameras) >= self.num_cameras:
                break
            time.sleep(0.1)
        
        return list(self.discovered_cameras.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            'num_cameras_discovered': len(self.discovered_cameras),
            'cameras': list(self.discovered_cameras.keys()),
            'frames_received': dict(self.stats['frames_received']),
            'frames_synced': self.stats['frames_synced'],
            'sync_failures': self.stats['sync_failures'],
            'buffer_sizes': {
                cam_id: len(buffer) 
                for cam_id, buffer in self.frame_buffers.items()
            }
        }
    
    def clear_buffers(self):
        """Clear all frame buffers."""
        for buffer in self.frame_buffers.values():
            buffer.clear()


if __name__ == "__main__":
    # Test coordinator
    print("Starting Master Coordinator test...")
    coordinator = MasterCoordinator(num_cameras=2)
    coordinator.start()
    
    try:
        # Discover cameras
        print("Discovering cameras...")
        cameras = coordinator.discover_cameras(timeout=10.0)
        print(f"Discovered {len(cameras)} cameras: {[c.camera_id for c in cameras]}")
        
        # Monitor for synced frames
        print("Monitoring synchronized frames...")
        for _ in range(100):
            batch = coordinator.get_synchronized_batch()
            if batch:
                print(f"Synced batch: {[f.camera_id for f in batch]}")
            time.sleep(0.1)
        
        # Print stats
        print("\nStatistics:")
        print(json.dumps(coordinator.get_stats(), indent=2))
        
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.stop()
