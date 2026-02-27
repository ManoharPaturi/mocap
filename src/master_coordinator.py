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
import statistics
from config import (
    DISCOVERY_PORT, DATA_PORT, NUM_CAMERAS, COMPRESS_NETWORK_DATA,
    FRAME_BUFFER_SIZE, SYNC_TIME_THRESHOLD_MS, CALIBRATION_FILE,
    FEEDBACK_PORT, FEEDBACK_ENABLED, FEEDBACK_INTERVAL_FRAMES,
    WORLD_AXIS_TRANSFORM, STEREO_POINT_MIN_INPUT_CONFIDENCE,
    KINEMATICS_MIN_POINT_CONFIDENCE, ENABLE_3D_ONE_EURO_FILTER,
    FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF,
    ENABLE_CLOCK_SYNC, CLOCK_SYNC_PORT, CLOCK_SYNC_SAMPLES,
    CLOCK_SYNC_INTERVAL_SEC, CLOCK_SYNC_RTT_OUTLIER_FACTOR
)
from src.stereo_calibration import StereoCalibration
from src.triangulation import Triangulator
from src.one_euro_filter import OneEuroFilter
from src.kinematics_engine import KinematicsEngine
from src.occlusion_fusion import OcclusionFusionEngine
from src.uncertainty_estimation import UncertaintyEstimator, ErrorMetricsCalculator
from src.advanced_kinematics import AdvancedKinematics
from src.dashboard_monitoring import DashboardMonitor


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
        
        # Fusion modules (Level 1 & 2)
        self.triangulator: Optional[Triangulator] = None
        self._load_calibration()
        
        # Feedback Loop (Level 3)
        self.feedback_socket = None
        if FEEDBACK_ENABLED:
            # We use the same self.context
            self.feedback_socket = self.context.socket(zmq.PUB)
            try:
                self.feedback_socket.bind(f"tcp://*:{FEEDBACK_PORT}")
                print(f"[MasterCoordinator] Feedback loop active on port {FEEDBACK_PORT}")
            except Exception as e:
                print(f"[MasterCoordinator] Could not bind feedback port: {e}")
        
        self.frame_count = 0
        self.kinematics_engine = KinematicsEngine()
        self.pose_3d_filters: Dict[int, Dict[str, OneEuroFilter]] = defaultdict(dict)
        self.occlusion_fusion_engine = OcclusionFusionEngine()
        self.advanced_kinematics_engine = AdvancedKinematics()
        self.uncertainty_estimator = UncertaintyEstimator(
            calibration_rms=0.5,
            baseline_m=1.0,
            focal_length_px=1000.0
        )
        self.error_metrics_calculator = ErrorMetricsCalculator(window_size=300)
        self.dashboard_monitor = DashboardMonitor(history_size=300)
        
        # Clock synchronization
        self.clock_offsets: Dict[str, int] = {}  # camera_id -> offset in nanoseconds
        self._clock_sync_thread = None
        self._last_sync_time = 0.0
        
        # Sequence gap detection per camera
        self._last_frame_number: Dict[str, int] = {}
        self._sequence_gaps: Dict[str, int] = defaultdict(int)
        
        # Latency tracking
        self._latency_accum: Dict[str, list] = defaultdict(list)  # stage -> [durations_ms]
        
        # Uncertainty tracking (inter-camera disagreement)
        self._landmark_disagreements: Dict[int, float] = {}  # lm_id -> disagreement_m
        self._gpu_reports: Dict[str, Dict[str, Any]] = {}
        
        # Occlusion state machine per landmark
        self._occlusion_state: Dict[int, str] = {}  # lm_id -> 'VISIBLE'|'OCCLUDED'|'PREDICTED'
        self._occlusion_last_position: Dict[int, Dict] = {}  # lm_id -> last known 3D pos
        self._occlusion_frames_hidden: Dict[int, int] = defaultdict(int)
        
        print("[MasterCoordinator] Initialized")

    def _load_calibration(self):
        """Load stereo calibration file and initialize triangulator."""
        try:
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
        
        # Run clock synchronization after all cameras are connected
        if ENABLE_CLOCK_SYNC:
            try:
                self.sync_camera_clocks(camera_ips, 
                                         [info.camera_id for info in self.discovered_cameras.values()])
            except Exception as e:
                print(f"[ClockSync] ⚠️  Clock synchronization failed: {e}")
                print(f"[ClockSync] Continuing without clock correction.")
    
    def estimate_clock_offset(self, camera_ip: str, camera_id: str) -> Optional[int]:
        """
        Estimate clock offset to a remote camera using Cristian's Algorithm.
        
        Sends CLOCK_SYNC_SAMPLES ping requests and uses the median of non-outlier
        round-trip samples to compute the offset.
        
        Args:
            camera_ip: IP address of the camera server
            camera_id: Camera identifier for logging
            
        Returns:
            Clock offset in nanoseconds (add to camera timestamps to align with master),
            or None if sync failed.
        """
        samples = []  # list of (rtt_ns, offset_ns)
        
        for i in range(CLOCK_SYNC_SAMPLES):
            # Create a fresh REQ socket for each sample to avoid
            # state corruption after recv timeouts (REQ enforces
            # strict send→recv→send→recv ordering).
            sock = self.context.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, 2000)  # 2s timeout per sample
            sock.connect(f"tcp://{camera_ip}:{CLOCK_SYNC_PORT}")
            
            try:
                ping_msg = {
                    'type': 'clock_ping',
                    'master_time_ns': time.time_ns(),
                    'sample': i
                }
                t0 = time.time_ns()
                sock.send(msgpack.packb(ping_msg))
                
                try:
                    reply_data = sock.recv()
                    t1 = time.time_ns()
                    reply = msgpack.unpackb(reply_data, raw=False)
                    
                    if reply.get('type') == 'clock_pong':
                        server_time = reply['server_time_ns']
                        rtt = t1 - t0
                        # Cristian's Algorithm: offset = server_time - (t0 + RTT/2)
                        offset = server_time - (t0 + rtt // 2)
                        samples.append((rtt, offset))
                except zmq.Again:
                    print(f"[ClockSync] Timeout on sample {i} for {camera_id}")
                    continue
                
                time.sleep(0.01)  # Small delay between samples
            finally:
                sock.close()
        
        if not samples:
            print(f"[ClockSync] No valid samples for {camera_id}")
            return None
        
        # Reject RTT outliers: keep only samples with RTT <= factor * min_rtt
        min_rtt = min(s[0] for s in samples)
        threshold = min_rtt * CLOCK_SYNC_RTT_OUTLIER_FACTOR
        good_samples = [s for s in samples if s[0] <= threshold]
        
        if not good_samples:
            good_samples = samples  # Fallback: use all if filter is too strict
        
        # Use median offset for robustness
        offsets = [s[1] for s in good_samples]
        median_offset = int(statistics.median(offsets))
        
        offset_ms = median_offset / 1_000_000
        min_rtt_ms = min_rtt / 1_000_000
        print(f"[ClockSync] {camera_id}: offset = {offset_ms:+.2f} ms "
              f"(RTT min = {min_rtt_ms:.2f} ms, {len(good_samples)}/{len(samples)} samples used)")
        
        return median_offset
    
    def sync_camera_clocks(self, camera_ips: List[str], camera_ids: List[str]):
        """
        Estimate clock offsets for all connected cameras.
        
        Args:
            camera_ips: List of camera IP addresses
            camera_ids: List of corresponding camera IDs
        """
        print(f"[ClockSync] Starting clock synchronization for {len(camera_ips)} camera(s)...")
        
        for ip, cam_id in zip(camera_ips, camera_ids):
            offset = self.estimate_clock_offset(ip, cam_id)
            if offset is not None:
                self.clock_offsets[cam_id] = offset
                print(f"[ClockSync] ✅ {cam_id} synchronized (offset: {offset / 1_000_000:+.2f} ms)")
            else:
                print(f"[ClockSync] ⚠️  {cam_id} sync failed — timestamps will be uncorrected")
        
        self._last_sync_time = time.time()
        
        if self.clock_offsets:
            print(f"[ClockSync] Synchronization complete. {len(self.clock_offsets)} camera(s) corrected.")
        else:
            print(f"[ClockSync] No cameras synchronized. Falling back to raw timestamps.")
    
    def _check_sync_quality(self, synced_frames: List) -> Dict[str, float]:
        """
        Diagnostic: analyze timestamp spread in a synchronized batch.
        
        Args:
            synced_frames: List of FrameData from get_synchronized_batch()
            
        Returns:
            Dict with 'spread_ms' and per-camera offsets from mean
        """
        if not synced_frames or len(synced_frames) < 2:
            return {}
        
        timestamps = [f.timestamp for f in synced_frames]
        mean_ts = sum(timestamps) / len(timestamps)
        spread_ns = max(timestamps) - min(timestamps)
        spread_ms = spread_ns / 1_000_000
        
        per_camera = {}
        for f in synced_frames:
            offset_ms = (f.timestamp - mean_ts) / 1_000_000
            per_camera[f.camera_id] = offset_ms
        
        if spread_ms > 10.0:
            print(f"[ClockSync] ⚠️  Timestamp spread = {spread_ms:.1f} ms (>10 ms) — "
                  f"consider re-synchronizing clocks")
            for cam_id, off_ms in per_camera.items():
                print(f"  {cam_id}: {off_ms:+.2f} ms from mean")
        
        return {'spread_ms': spread_ms, 'per_camera_offset_ms': per_camera}
    
    def check_resync_needed(self) -> bool:
        """
        Check if periodic clock re-synchronization is needed.
        
        Should be called periodically (e.g., in the main processing loop).
        If CLOCK_SYNC_INTERVAL_SEC has elapsed since the last sync, triggers
        a re-sync for all connected cameras.
        
        Returns:
            True if re-sync was performed, False otherwise.
        """
        if not ENABLE_CLOCK_SYNC or CLOCK_SYNC_INTERVAL_SEC <= 0:
            return False
        
        elapsed = time.time() - self._last_sync_time
        if elapsed < CLOCK_SYNC_INTERVAL_SEC:
            return False
        
        print(f"[ClockSync] Periodic re-sync triggered ({elapsed:.0f}s since last sync)")
        camera_ips = [info.ip for info in self.discovered_cameras.values()]
        camera_ids = [info.camera_id for info in self.discovered_cameras.values()]
        
        if camera_ips:
            self.sync_camera_clocks(camera_ips, camera_ids)
            return True
        return False
    
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
            socket.setsockopt(zmq.RCVHWM, 1)
            socket.setsockopt(zmq.CONFLATE, 1)
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
                    latest_data = None
                    if socket.poll(timeout=2):
                        latest_data = socket.recv()
                        while socket.poll(timeout=0):
                            latest_data = socket.recv()

                    if latest_data is not None:
                        msg_count += 1

                        # Debug first 5 messages
                        if msg_count <= 5:
                            print(f"[MasterCoordinator] ✅ Receiving data from {camera_id} (msg #{msg_count})")

                        # Deserialize
                        if COMPRESS_NETWORK_DATA:
                            msg = msgpack.unpackb(latest_data, raw=False)
                        else:
                            msg = json.loads(latest_data.decode('utf-8'))

                        # Process frame data
                        if msg.get('type') == 'frame_data':
                            self._process_frame_data(msg)

                # Periodic heartbeat log every 5 seconds
                now = time.time()
                if now - last_log_time >= 5.0:
                    total = sum(self.stats['frames_received'].values())
                    print(f"[MasterCoordinator] Heartbeat: {total} total frames received from {list(self.data_sockets.keys())}")
                    last_log_time = now

                time.sleep(0.001)
                
            except Exception as e:
                if self.running:
                    print(f"[MasterCoordinator] Data receiver error: {e}")

    # =========================================================================
    # Latency Instrumentation
    # =========================================================================
    
    def record_latency(self, stage: str, duration_ms: float):
        """
        Record a latency sample for a named pipeline stage.
        
        Stages: 'capture', 'detection', 'network', 'sync', 'triangulation',
                'filtering', 'kinematics', 'display', 'total'
        """
        self._latency_accum[stage].append(duration_ms)
        # Keep bounded
        if len(self._latency_accum[stage]) > 500:
            self._latency_accum[stage] = self._latency_accum[stage][-250:]

        # Mirror into dashboard monitoring module
        try:
            if stage == 'capture':
                self.dashboard_monitor.latency.begin_frame()
            self.dashboard_monitor.latency.record_stage(stage, duration_ms)
            if stage == 'total':
                self.dashboard_monitor.latency.end_frame()
        except Exception:
            pass
    
    def get_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get latency statistics per pipeline stage.
        
        Returns dict: stage -> {'mean_ms': ..., 'p95_ms': ..., 'max_ms': ...}
        """
        stats = {}
        for stage, samples in self._latency_accum.items():
            if not samples:
                continue
            arr = sorted(samples)
            n = len(arr)
            stats[stage] = {
                'mean_ms': sum(arr) / n,
                'p95_ms': arr[int(n * 0.95)] if n >= 20 else arr[-1],
                'max_ms': arr[-1],
                'samples': n
            }
        return stats
    
    def _log_latency_summary(self):
        """Print latency summary to console (called periodically)."""
        from config import ENABLE_LATENCY_TRACKING
        if not ENABLE_LATENCY_TRACKING:
            return
        stats = self.get_latency_stats()
        if not stats:
            return
        lines = ["[Latency Summary]"]
        for stage, s in sorted(stats.items()):
            lines.append(f"  {stage:15s}: mean={s['mean_ms']:.1f}ms  "
                         f"p95={s['p95_ms']:.1f}ms  max={s['max_ms']:.1f}ms  (n={s['samples']})")
        print("\n".join(lines))

    def _process_frame_data(self, msg: Dict[str, Any]):
        """Process incoming frame data from a camera."""
        t_recv = time.time_ns()
        camera_id = msg.get('camera_id')
        frame_number = msg.get('frame_number')
        timestamp = msg.get('timestamp')
        results = msg.get('results')
        packet_landmarks = msg.get('landmarks')
        
        # Schema version check
        schema_ver = msg.get('schema_version', 1)
        if schema_ver > 2:
            print(f"[MasterCoordinator] Warning: unknown schema v{schema_ver} from {camera_id}")
        
        # Allow empty results dict (no detection) - still valid for sync
        if not all([camera_id, frame_number is not None, timestamp, results is not None]):
            if self.stats['frames_received'].get(camera_id, 0) < 3:
                print(f"[MasterCoordinator] Skipping invalid frame from {camera_id}")
            return

        # Sequence gap detection
        if camera_id in self._last_frame_number:
            expected = self._last_frame_number[camera_id] + 1
            if frame_number > expected:
                gap = frame_number - expected
                self._sequence_gaps[camera_id] += gap
                if self._sequence_gaps[camera_id] <= 5:  # Only warn first few
                    print(f"[Sync] {camera_id}: sequence gap detected "
                          f"(expected #{expected}, got #{frame_number}, {gap} frames dropped)")
        self._last_frame_number[camera_id] = frame_number

        if packet_landmarks is not None:
            results['packet_landmarks'] = packet_landmarks

        gpu_compute = msg.get('gpu_compute')
        if isinstance(gpu_compute, dict):
            results['gpu_compute'] = gpu_compute
            self._gpu_reports[camera_id] = gpu_compute
        
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
        
        # Apply clock offset correction if available
        if ENABLE_CLOCK_SYNC and camera_id in self.clock_offsets:
            frame_data.timestamp = timestamp + self.clock_offsets[camera_id]
        
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
        
        Includes stale-frame eviction: if a camera's newest frame is older
        than STALE_FRAME_TIMEOUT_MS compared to the freshest camera, its
        buffer is cleared to prevent blocking sync forever when a camera
        drops out.
        
        Returns:
            List of FrameData objects, one per camera, with matching timestamps
        """
        from config import STALE_FRAME_TIMEOUT_MS
        
        if len(self.frame_buffers) < self.num_cameras:
            # Not all cameras connected yet
            return None
        
        # --- Stale frame eviction ---
        # Find the newest timestamp across ALL cameras
        global_newest_ns = None
        for camera_id, buffer in self.frame_buffers.items():
            if len(buffer) > 0:
                newest_ts = buffer[-1].timestamp  # deque: last element is newest
                if global_newest_ns is None or newest_ts > global_newest_ns:
                    global_newest_ns = newest_ts
        
        if global_newest_ns is not None:
            stale_threshold_ns = STALE_FRAME_TIMEOUT_MS * 1_000_000
            for camera_id, buffer in list(self.frame_buffers.items()):
                if len(buffer) > 0:
                    newest_in_buffer = buffer[-1].timestamp
                    age_ns = global_newest_ns - newest_in_buffer
                    if age_ns > stale_threshold_ns:
                        stale_ms = age_ns / 1_000_000
                        if not hasattr(self, '_stale_warned'):
                            self._stale_warned = set()
                        if camera_id not in self._stale_warned:
                            print(f"[Sync] ⚠️  {camera_id} stale by {stale_ms:.0f}ms — "
                                  f"clearing {len(buffer)} buffered frames")
                            self._stale_warned.add(camera_id)
                        buffer.clear()
                    else:
                        # Camera is alive again — reset stale warning
                        if hasattr(self, '_stale_warned'):
                            self._stale_warned.discard(camera_id)
        
        # --- Standard sync logic ---
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
            packet_landmarks = results.get('packet_landmarks')
            if isinstance(packet_landmarks, list) and len(packet_landmarks) > 0:
                return [
                    {
                        'x': lm.get('x', 0.0),
                        'y': lm.get('y', 0.0),
                        'z': lm.get('z', 0.0),
                        'visibility': lm.get('conf', lm.get('visibility', 1.0))
                    }
                    for lm in packet_landmarks if isinstance(lm, dict)
                ]
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

        packet_landmarks = results.get('packet_landmarks')
        if isinstance(packet_landmarks, list) and len(packet_landmarks) > 0:
            return [
                {
                    'x': lm.get('x', 0.0),
                    'y': lm.get('y', 0.0),
                    'z': lm.get('z', 0.0),
                    'visibility': lm.get('conf', lm.get('visibility', 1.0))
                }
                for lm in packet_landmarks if isinstance(lm, dict)
            ]

        return None

    def _extract_world_landmarks(self, results: Dict[str, Any]) -> Optional[list]:
        """
        Extract MediaPipe pose_world_landmarks (hip-relative, in meters).
        Used as Tier 3 fallback when triangulation and monocular both fail.
        
        Returns: list of dicts with 'x','y','z','visibility', or None
        """
        pose = results.get('pose')
        if not pose:
            return None
        
        # Raw MediaPipe result with pose_world_landmarks
        if hasattr(pose, 'pose_world_landmarks'):
            if pose.pose_world_landmarks and len(pose.pose_world_landmarks) > 0:
                return [
                    {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                    for lm in pose.pose_world_landmarks[0]
                ]
        
        # Serialized world landmarks from remote camera
        world_lms = results.get('pose_world_landmarks') or results.get('world_landmarks')
        if isinstance(world_lms, list) and len(world_lms) > 0:
            first = world_lms[0]
            if isinstance(first, dict):
                return world_lms
            if isinstance(first, (list, tuple)):
                return [
                    {'x': lm[0], 'y': lm[1], 'z': lm[2],
                     'visibility': lm[3] if len(lm) > 3 else 0.5}
                    for lm in world_lms
                ]
        
        return None

    def _filter_pose_3d(self, pose_3d: Dict[int, Dict[str, Any]], timestamp_ns: int) -> Dict[int, Dict[str, Any]]:
        """Apply 1-Euro filtering to 3D joint positions only (never filter angles)."""
        if not ENABLE_3D_ONE_EURO_FILTER:
            return pose_3d

        t_s = timestamp_ns / 1_000_000_000.0
        filtered = {}

        for lm_id, point in pose_3d.items():
            out = dict(point)
            axis_filters = self.pose_3d_filters[lm_id]

            for axis in ('x', 'y', 'z'):
                value = float(point[axis])
                if axis not in axis_filters:
                    axis_filters[axis] = OneEuroFilter(
                        t0=t_s,
                        x0=value,
                        min_cutoff=FILTER_MIN_CUTOFF,
                        beta=FILTER_BETA,
                        d_cutoff=FILTER_D_CUTOFF
                    )
                else:
                    value = axis_filters[axis](t_s, value)

                out[axis] = float(value)

            filtered[lm_id] = out

        return filtered

    def _apply_world_axis_convention(self, point: Dict[str, Any]) -> Dict[str, Any]:
        """Map triangulated camera-style coordinates to locked world convention."""
        x = float(point['x'])
        y = float(point['y'])
        z = float(point['z'])

        if WORLD_AXIS_TRANSFORM.get('flip_x', False):
            x = -x
        if WORLD_AXIS_TRANSFORM.get('flip_y', False):
            y = -y
        if WORLD_AXIS_TRANSFORM.get('flip_z', False):
            z = -z

        point['x'] = x
        point['y'] = y
        point['z'] = z
        return point

    def _resolve_calibration_camera_id(self, camera_id: str) -> str:
        """Resolve runtime camera IDs to calibration IDs."""
        if not self.triangulator or not self.triangulator.calibration:
            return camera_id
        cameras = self.triangulator.calibration.cameras
        if camera_id in cameras:
            return camera_id
        if camera_id == 'local_cam' and 'cam_1' in cameras:
            return 'cam_1'
        return camera_id

    def _compute_pose_3d_kinematics(self, pose_3d: Dict[int, Dict[str, Any]], timestamp_ns: int) -> Dict[str, Any]:
        """Compute linear velocity, acceleration, angles, and angular velocity from filtered 3D pose."""
        frame_result = self.kinematics_engine.process_frame(
            joints_3d=pose_3d,
            timestamp=timestamp_ns,
            compute_derivatives=True,
            min_confidence=KINEMATICS_MIN_POINT_CONFIDENCE,
        )

        joint_velocity = {}
        joint_acceleration = {}
        for lm_id, kinematics in frame_result.get('joint_kinematics', {}).items():
            if kinematics.velocity is not None:
                joint_velocity[lm_id] = {
                    'vx': kinematics.velocity[0],
                    'vy': kinematics.velocity[1],
                    'vz': kinematics.velocity[2],
                    'v': kinematics.velocity_magnitude,
                }
            if kinematics.acceleration is not None:
                joint_acceleration[lm_id] = {
                    'ax': kinematics.acceleration[0],
                    'ay': kinematics.acceleration[1],
                    'az': kinematics.acceleration[2],
                    'a': kinematics.acceleration_magnitude,
                }

        joint_angles = {
            name: data.angle
            for name, data in frame_result.get('angles', {}).items()
        }
        angular_velocity = {
            name: data.angular_velocity
            for name, data in frame_result.get('angles', {}).items()
            if data.angular_velocity is not None
        }

        spine_vector = frame_result.get('spine_vector')
        if spine_vector is None:
            spine_payload = None
        else:
            spine_payload = {'x': spine_vector[0], 'y': spine_vector[1], 'z': spine_vector[2]}

        return {
            'joint_velocity_3d': joint_velocity,
            'joint_acceleration_3d': joint_acceleration,
            'joint_angles_deg': joint_angles,
            'angular_velocity_deg_s': angular_velocity,
            'spine_vector': spine_payload,
            'flat_export': self.kinematics_engine.export_frame_data(frame_result),
        }

    def _get_monocular_fallback(self, lm: Dict[str, Any], camera_id: str) -> Optional[Dict[str, Any]]:
        """
        Estimate 3D position from a single view (Level 1 Fallback).
        Uses MediaPipe's relative 'z' and an assumed distance to subject.
        """
        if not self.triangulator or camera_id not in self.triangulator.calibration.cameras:
            return None
            
        from config import MONOCULAR_SUBJECT_DISTANCE_M
        
        # Get camera parameters
        cam = self.triangulator.calibration.cameras[camera_id]
        fx = cam.intrinsic_matrix[0, 0]
        fy = cam.intrinsic_matrix[1, 1]
        cx = cam.intrinsic_matrix[0, 2]
        cy = cam.intrinsic_matrix[1, 2]
        
        # 1. 2D Normalize -> Pixel -> Camera Space
        # MediaPipe x,y are [0,1]. z is relative to hip (approx 0 centered)
        x_px = lm['x'] * cam.image_size[0]
        y_px = lm['y'] * cam.image_size[1]
        
        # Assumed depth: subject distance + MediaPipe's relative z
        # MediaPipe z is scaled such that it's roughly in meters relative to hip
        z_cam = MONOCULAR_SUBJECT_DISTANCE_M + lm.get('z', 0)
        
        # Back-project to 3D in camera space
        x_cam = (x_px - cx) * z_cam / fx
        y_cam = (y_px - cy) * z_cam / fy
        
        # Transform from camera space to world space: P_world = R.T @ (P_cam - T)
        P_cam = np.array([[x_cam], [y_cam], [z_cam]])
        P_world = cam.rotation.T @ (P_cam - cam.translation)
        
        return {
            'x': float(P_world[0, 0]),
            'y': float(P_world[1, 0]),
            'z': float(P_world[2, 0]),
            'visibility': lm.get('visibility', 0.5) * 0.7, # Lower confidence for fallback
            'method': f'monocular_{camera_id}'
        }

    def get_synced_3d_pose(self, synced_frames: List['FrameData']) -> Optional[Dict[str, Any]]:
        """
        Compute 3D pose from a list of synchronized 2D frames.
        
        Pipeline:
          Tier 1 — Multi-view triangulation (DLT) + inter-camera disagreement
          Tier 2 — Monocular back-projection fallback
          Tier 3 — MediaPipe world-landmarks fallback (hip-relative, scaled)
          Occlusion state machine — VISIBLE → OCCLUDED → PREDICTED
        """
        if not self.triangulator:
            return None
            
        from config import OCCLUSION_FILL_ENABLED
        
        OCCLUSION_MAX_PREDICTED_FRAMES = 15  # ~500ms @ 30fps
            
        # ── Collect 2D observations for each landmark ──
        landmark_observations = defaultdict(dict)
        landmark_visibilities = defaultdict(dict)
        landmark_raw_data = defaultdict(dict)
        world_landmark_data = defaultdict(dict)  # Tier 3: world landmarks per camera
        
        for frame in synced_frames:
            landmarks = self._extract_pose_landmarks(frame.results)
            if not landmarks:
                continue

            cal_camera_id = self._resolve_calibration_camera_id(frame.camera_id)
            
            # Also extract world landmarks for Tier 3 fallback
            world_lms = self._extract_world_landmarks(frame.results)
                
            for idx, lm in enumerate(landmarks):
                vis = lm.get('visibility', 1.0)
                landmark_raw_data[idx][cal_camera_id] = lm
                if vis >= STEREO_POINT_MIN_INPUT_CONFIDENCE:
                    w, h = 1280, 720
                    if cal_camera_id in self.triangulator.calibration.cameras:
                        image_size = self.triangulator.calibration.cameras[cal_camera_id].image_size
                        if image_size and len(image_size) == 2:
                            w, h = image_size
                    x_px = lm['x'] * w
                    y_px = lm['y'] * h
                    landmark_observations[idx][cal_camera_id] = np.array([x_px, y_px])
                    landmark_visibilities[idx][cal_camera_id] = vis
                    
            # Store world landmarks for Tier 3
            if world_lms:
                for idx, wlm in enumerate(world_lms):
                    world_landmark_data[idx][cal_camera_id] = wlm
        
        # ── Track quality for feedback loop ──
        quality_feedback = {}
        
        # ── Triangulate each landmark (3-tier fallback) ──
        landmarks_3d = {}
        low_reliability_landmarks = []
        uncertainty = {}  # lm_id -> disagreement in meters
        
        for lm_id in set(list(landmark_observations.keys()) + list(world_landmark_data.keys())):
            observations = landmark_observations.get(lm_id, {})
            
            # ── TIER 1: MULTI-VIEW TRIANGULATION + DISAGREEMENT ──
            if len(observations) >= 2:
                point_3d = self.triangulator.triangulate_point(
                    observations, 
                    visibility_weights=landmark_visibilities.get(lm_id, {})
                )
                if point_3d:
                    landmarks_3d[lm_id] = {
                        'x': point_3d.x,
                        'y': point_3d.y,
                        'z': point_3d.z,
                        'visibility': point_3d.confidence,
                        'method': 'triangulated',
                        'views': point_3d.num_views,
                        'reproj_error': point_3d.reprojection_error
                    }
                    if point_3d.confidence < KINEMATICS_MIN_POINT_CONFIDENCE:
                        low_reliability_landmarks.append(lm_id)
                    
                    # ── Inter-camera disagreement (uncertainty) ──
                    # Compute per-camera monocular 3D estimates and measure spread
                    per_cam_estimates = []
                    for cam_id in observations:
                        if cam_id in landmark_raw_data.get(lm_id, {}):
                            mono = self._get_monocular_fallback(
                                landmark_raw_data[lm_id][cam_id], cam_id
                            )
                            if mono:
                                per_cam_estimates.append(
                                    np.array([mono['x'], mono['y'], mono['z']])
                                )
                    if len(per_cam_estimates) >= 2:
                        # Disagreement = max pairwise distance between estimates
                        max_dist = 0.0
                        for i in range(len(per_cam_estimates)):
                            for j in range(i + 1, len(per_cam_estimates)):
                                d = float(np.linalg.norm(
                                    per_cam_estimates[i] - per_cam_estimates[j]
                                ))
                                max_dist = max(max_dist, d)
                        uncertainty[lm_id] = max_dist
                        self._landmark_disagreements[lm_id] = max_dist
                    
                    quality_feedback[lm_id] = point_3d.reprojection_error
                    continue  # Success — skip lower tiers
            
            # ── TIER 2: MONOCULAR FALLBACK ──
            if OCCLUSION_FILL_ENABLED and len(observations) >= 1:
                best_cam = max(
                    landmark_visibilities.get(lm_id, {}),
                    key=lambda k: landmark_visibilities.get(lm_id, {}).get(k, 0),
                    default=None
                )
                if best_cam and landmark_visibilities.get(lm_id, {}).get(best_cam, 0) > 0.5:
                    est_3d = self._get_monocular_fallback(
                        landmark_raw_data[lm_id][best_cam], best_cam
                    )
                    if est_3d:
                        landmarks_3d[lm_id] = est_3d
                        if est_3d.get('visibility', 0.0) < KINEMATICS_MIN_POINT_CONFIDENCE:
                            low_reliability_landmarks.append(lm_id)
                        continue  # Success
            
            # ── TIER 3: WORLD LANDMARKS FALLBACK ──
            # MediaPipe pose_world_landmarks are hip-relative in meters.
            # Average across cameras when available.
            if lm_id in world_landmark_data and world_landmark_data[lm_id]:
                wx, wy, wz, wv = [], [], [], []
                for cam_id, wlm in world_landmark_data[lm_id].items():
                    wx.append(wlm.get('x', 0))
                    wy.append(wlm.get('y', 0))
                    wz.append(wlm.get('z', 0))
                    wv.append(wlm.get('visibility', 0.5))
                if wx:
                    landmarks_3d[lm_id] = {
                        'x': float(np.mean(wx)),
                        'y': float(np.mean(wy)),
                        'z': float(np.mean(wz)),
                        'visibility': float(np.mean(wv)) * 0.5,  # Penalize
                        'method': 'world_landmarks',
                        'views': len(wx)
                    }
                    low_reliability_landmarks.append(lm_id)
                    continue
        
        # ── OCCLUSION STATE MACHINE ──
        # Track per-landmark state: VISIBLE → OCCLUDED → PREDICTED
        all_lm_ids = set(list(landmarks_3d.keys()) + list(self._occlusion_state.keys()))
        for lm_id in all_lm_ids:
            if lm_id in landmarks_3d:
                # Landmark found → VISIBLE
                self._occlusion_state[lm_id] = 'VISIBLE'
                self._occlusion_last_position[lm_id] = landmarks_3d[lm_id].copy()
                self._occlusion_frames_hidden[lm_id] = 0
            elif lm_id in self._occlusion_last_position:
                # Not found but we have history
                self._occlusion_frames_hidden[lm_id] += 1
                hidden = self._occlusion_frames_hidden[lm_id]
                
                if hidden <= OCCLUSION_MAX_PREDICTED_FRAMES:
                    # Use last known position (hold/predict)
                    predicted = self._occlusion_last_position[lm_id].copy()
                    predicted['method'] = 'predicted'
                    predicted['visibility'] = max(0.1, predicted.get('visibility', 0.5) * 0.9)
                    landmarks_3d[lm_id] = predicted
                    self._occlusion_state[lm_id] = 'PREDICTED'
                    low_reliability_landmarks.append(lm_id)
                else:
                    # Too long — drop and mark as lost
                    self._occlusion_state[lm_id] = 'OCCLUDED'

        if not landmarks_3d:
            return None
            
        # ── Feedback loop ──
        self.frame_count += 1
        if FEEDBACK_ENABLED and self.feedback_socket and self.frame_count % FEEDBACK_INTERVAL_FRAMES == 0:
            self._send_quality_feedback(quality_feedback)
            
        # ── World axis convention ──
        for lm_id in list(landmarks_3d.keys()):
            landmarks_3d[lm_id] = self._apply_world_axis_convention(landmarks_3d[lm_id])

        fused_timestamp_ns = int(np.mean([f.timestamp for f in synced_frames]))
        landmarks_3d = self._filter_pose_3d(landmarks_3d, fused_timestamp_ns)
        kinematics = self._compute_pose_3d_kinematics(landmarks_3d, fused_timestamp_ns)

        # ── Optional standalone module outputs (additive, non-breaking) ──
        # Build per-camera 3D estimates for uncertainty/fusion modules
        per_camera_points = defaultdict(list)
        for lm_id in landmarks_3d.keys():
            raw_map = landmark_raw_data.get(lm_id, {})
            for cam_id, raw_lm in raw_map.items():
                mono = self._get_monocular_fallback(raw_lm, cam_id)
                if mono:
                    per_camera_points[lm_id].append(np.array([
                        mono.get('x', 0.0), mono.get('y', 0.0), mono.get('z', 0.0)
                    ], dtype=float))

        triangulated_only = {
            lm_id: pt for lm_id, pt in landmarks_3d.items()
            if pt.get('method') == 'triangulated'
        }
        monocular_only = {
            lm_id: pt for lm_id, pt in landmarks_3d.items()
            if str(pt.get('method', '')).startswith('monocular_')
        }
        world_only = {
            lm_id: pt for lm_id, pt in landmarks_3d.items()
            if pt.get('method') == 'world_landmarks'
        }

        fusion_output = self.occlusion_fusion_engine.fuse_frame(
            triangulated=triangulated_only,
            monocular=monocular_only,
            world_landmarks=world_only,
            per_camera_estimates=per_camera_points,
            timestamp_ns=fused_timestamp_ns
        )

        reproj_errors = {
            lm_id: float(pt.get('reproj_error', 0.0))
            for lm_id, pt in landmarks_3d.items()
            if isinstance(pt, dict)
        }

        # ── Build occlusion summary ──
        occlusion_summary = {}
        for lm_id in landmarks_3d:
            occlusion_summary[lm_id] = self._occlusion_state.get(lm_id, 'VISIBLE')

        uncertainty_detailed = self.uncertainty_estimator.estimate_frame(
            landmarks_3d=landmarks_3d,
            per_camera_points=per_camera_points,
            visibility_scores=landmark_visibilities,
            occlusion_states=occlusion_summary,
            reproj_errors=reproj_errors
        )
        self.error_metrics_calculator.add_frame(uncertainty_detailed)
        self.error_metrics_calculator.add_consistency_score(
            self.occlusion_fusion_engine.get_consistency_score()
        )
        uncertainty_summary = self.error_metrics_calculator.get_summary()

        advanced_state = self.advanced_kinematics_engine.process_frame(
            landmarks_3d, fused_timestamp_ns,
            min_confidence=KINEMATICS_MIN_POINT_CONFIDENCE
        )
        advanced_kinematics = self.advanced_kinematics_engine.export_state_dict(advanced_state)
        
        return {
            'pose_3d': landmarks_3d,
            'kinematics_3d': kinematics,
            'low_reliability_landmarks': sorted(set(low_reliability_landmarks)),
            'timestamp_ns': fused_timestamp_ns,
            'uncertainty': uncertainty,
            'occlusion_states': occlusion_summary,
            'fusion': fusion_output,
            'uncertainty_detailed': {
                lm_id: {
                    'combined_uncertainty_m': u.combined_uncertainty_m,
                    'confidence': u.confidence,
                    'reproj_error_px': u.reproj_error_px,
                    'inter_camera_disagreement_m': u.inter_camera_disagreement_m,
                    'method': u.method
                }
                for lm_id, u in uncertainty_detailed.items()
            },
            'uncertainty_summary': uncertainty_summary,
            'advanced_kinematics': advanced_kinematics,
            'dashboard_status': self.dashboard_monitor.get_status()
        }

    def get_dashboard_status(self) -> Dict[str, Any]:
        """Expose consolidated dashboard monitoring status."""
        return self.dashboard_monitor.get_status()

    def _send_quality_feedback(self, quality_hints: Dict[int, float]):
        """Broadcast quality metrics to all cameras."""
        if not self.feedback_socket:
            return

        compute_hints = self._build_compute_hints()
            
        msg = {
            'type': 'quality_feedback',
            'timestamp': time.time_ns(),
            'hints': quality_hints,  # {lm_id: reproj_error}
            'compute_hints': compute_hints
        }
        
        try:
            payload = msgpack.packb(msg)
            self.feedback_socket.send(payload)
        except Exception as e:
            print(f"[MasterCoordinator] Feedback send error: {e}")

    def _build_compute_hints(self) -> Dict[str, Any]:
        """
        Build lightweight scheduling hints for camera nodes based on bottlenecks
        and recent per-camera inference reports.
        """
        stats = self.get_latency_stats()
        stage_means = {
            stage: s.get('mean_ms', 0.0)
            for stage, s in stats.items()
            if isinstance(s, dict)
        }
        if not stage_means:
            return {
                'suggested_action': 'none',
                'target_stage': 'unknown',
                'reason': 'insufficient_latency_samples'
            }

        target_stage = max(stage_means, key=lambda k: stage_means[k])
        target_ms = float(stage_means.get(target_stage, 0.0))

        # Conservative policy: only suggest downshift on clearly slow stages.
        if target_stage == 'detection' and target_ms > 35.0:
            action = 'reduce_detector_load'
        elif target_stage == 'network' and target_ms > 20.0:
            action = 'reduce_network_payload'
        elif target_stage == 'triangulation' and target_ms > 12.0:
            action = 'reduce_3d_load'
        else:
            action = 'none'

        camera_compute = {}
        for camera_id, report in self._gpu_reports.items():
            if not isinstance(report, dict):
                continue
            inf = report.get('inference_times_ms', {}) or {}
            camera_compute[camera_id] = {
                'pose_ms': float(inf.get('pose', 0.0)),
                'face_ms': float(inf.get('face', 0.0)),
                'hand_ms': float(inf.get('hand', 0.0)),
                'total_ms': float(inf.get('total', 0.0)),
                'device': report.get('device', 'unknown')
            }

        return {
            'suggested_action': action,
            'target_stage': target_stage,
            'target_stage_mean_ms': round(target_ms, 2),
            'camera_compute': camera_compute,
            'generated_at_ns': time.time_ns()
        }
    
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
