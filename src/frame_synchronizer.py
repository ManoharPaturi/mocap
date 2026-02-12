"""
Frame Synchronizer Module
Matches frames from different cameras based on timestamps for multi-view processing.
"""

import time
from typing import Dict, List, Optional
from collections import deque, defaultdict
from dataclasses import dataclass
from config import SYNC_TIME_THRESHOLD_MS, FRAME_BUFFER_SIZE


@dataclass
class SyncedFrameData:
    """Synchronized frame data from a single camera."""
    camera_id: str
    frame_number: int
    timestamp: float  # nanoseconds
    results: Dict
    time_offset: float  # Offset from earliest frame in batch (ms)


class FrameSynchronizer:
    """
    Synchronizes frames from multiple cameras based on timestamps.
    Uses a sliding window buffer approach.
    """
    
    def __init__(self, time_threshold_ms: float = SYNC_TIME_THRESHOLD_MS):
        """
        Initialize frame synchronizer.
        
        Args:
            time_threshold_ms: Maximum time difference for frame matching (milliseconds)
        """
        self.time_threshold_ms = time_threshold_ms
        self.time_threshold_ns = time_threshold_ms * 1_000_000  # Convert to nanoseconds
        
        # Frame buffers per camera
        self.frame_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=FRAME_BUFFER_SIZE)
        )
        
        # Statistics
        self.stats = {
            'frames_added': defaultdict(int),
            'frames_synced': 0,
            'sync_attempts': 0,
            'sync_failures': 0,
            'buffer_overflows': 0
        }
        
        print(f"[FrameSynchronizer] Initialized with threshold: {time_threshold_ms}ms")
    
    def add_frame(self, camera_id: str, frame_number: int, timestamp: float, results: Dict):
        """
        Add a frame to the buffer.
        
        Args:
            camera_id: Unique camera identifier
            frame_number: Sequential frame number
            timestamp: High-precision timestamp (nanoseconds)
            results: Detection results dict
        """
        frame_data = SyncedFrameData(
            camera_id=camera_id,
            frame_number=frame_number,
            timestamp=timestamp,
            results=results,
            time_offset=0.0  # Will be calculated during sync
        )
        
        # Check for buffer overflow
        if len(self.frame_buffers[camera_id]) >= FRAME_BUFFER_SIZE:
            self.stats['buffer_overflows'] += 1
        
        self.frame_buffers[camera_id].append(frame_data)
        self.stats['frames_added'][camera_id] += 1
    
    def get_synced_frames(self) -> Optional[List[SyncedFrameData]]:
        """
        Get a synchronized batch of frames from all cameras.
        
        Returns:
            List of SyncedFrameData objects (one per camera) if sync successful, None otherwise
        """
        self.stats['sync_attempts'] += 1
        
        if len(self.frame_buffers) < 1:
            return None
        
        # Get list of camera IDs that have frames
        cameras_with_frames = [
            cam_id for cam_id, buffer in self.frame_buffers.items() if len(buffer) > 0
        ]
        
        if len(cameras_with_frames) < 2:
            # Need at least 2 cameras for multi-view
            return None
        
        # Find the oldest unsynced frame across all cameras (reference frame)
        reference_frame = None
        reference_camera = None
        
        for camera_id in cameras_with_frames:
            frame = self.frame_buffers[camera_id][0]
            if reference_frame is None or frame.timestamp < reference_frame.timestamp:
                reference_frame = frame
                reference_camera = camera_id
        
        if reference_frame is None:
            return None
        
        # Try to find matching frames from other cameras
        synced_batch = []
        matched_cameras = set()
        
        for camera_id in cameras_with_frames:
            buffer = self.frame_buffers[camera_id]
            matched = False
            
            for frame in buffer:
                time_diff = abs(frame.timestamp - reference_frame.timestamp)
                
                if time_diff <= self.time_threshold_ns:
                    # Calculate time offset for stats
                    offset_ms = (frame.timestamp - reference_frame.timestamp) / 1_000_000.0
                    frame.time_offset = offset_ms
                    
                    synced_batch.append(frame)
                    matched_cameras.add(camera_id)
                    matched = True
                    break
            
            if not matched:
                # This camera has no matching frame
                # We could either:
                # 1. Discard this batch (strict sync)
                # 2. Continue with partial batch (allowing some cameras to miss frames)
                
                # For now, use strict sync - all cameras must have matching frames
                self.stats['sync_failures'] += 1
                return None
        
        if len(synced_batch) < 2:
            # Not enough cameras synchronized
            self.stats['sync_failures'] += 1
            return None
        
        # Remove consumed frames from buffers
        for frame in synced_batch:
            buffer = self.frame_buffers[frame.camera_id]
            # Remove all frames up to and including the matched frame
            while buffer and buffer[0].timestamp <= frame.timestamp:
                buffer.popleft()
        
        self.stats['frames_synced'] += 1
        return synced_batch
    
    def get_synced_frames_flexible(self, min_cameras: int = 2) -> Optional[List[SyncedFrameData]]:
        """
        Get synchronized frames with flexible matching (allows partial batches).
        
        Args:
            min_cameras: Minimum number of cameras required for a valid batch
            
        Returns:
            List of SyncedFrameData objects if enough cameras matched, None otherwise
        """
        self.stats['sync_attempts'] += 1
        
        if len(self.frame_buffers) < min_cameras:
            return None
        
        # Get cameras with frames
        cameras_with_frames = [
            cam_id for cam_id, buffer in self.frame_buffers.items() if len(buffer) > 0
        ]
        
        if len(cameras_with_frames) < min_cameras:
            return None
        
        # Find reference frame (oldest)
        reference_frame = None
        for camera_id in cameras_with_frames:
            frame = self.frame_buffers[camera_id][0]
            if reference_frame is None or frame.timestamp < reference_frame.timestamp:
                reference_frame = frame
        
        if reference_frame is None:
            return None
        
        # Match frames within threshold
        synced_batch = []
        
        for camera_id in cameras_with_frames:
            buffer = self.frame_buffers[camera_id]
            
            for frame in buffer:
                time_diff = abs(frame.timestamp - reference_frame.timestamp)
                
                if time_diff <= self.time_threshold_ns:
                    offset_ms = (frame.timestamp - reference_frame.timestamp) / 1_000_000.0
                    frame.time_offset = offset_ms
                    synced_batch.append(frame)
                    break
        
        # Check if we have enough cameras
        if len(synced_batch) < min_cameras:
            self.stats['sync_failures'] += 1
            return None
        
        # Remove consumed frames
        for frame in synced_batch:
            buffer = self.frame_buffers[frame.camera_id]
            while buffer and buffer[0].timestamp <= frame.timestamp:
                buffer.popleft()
        
        self.stats['frames_synced'] += 1
        return synced_batch
    
    def clear(self):
        """Clear all frame buffers."""
        for buffer in self.frame_buffers.values():
            buffer.clear()
        print("[FrameSynchronizer] Buffers cleared")
    
    def get_stats(self) -> Dict:
        """Get synchronizer statistics."""
        return {
            'time_threshold_ms': self.time_threshold_ms,
            'num_cameras': len(self.frame_buffers),
            'frames_added': dict(self.stats['frames_added']),
            'frames_synced': self.stats['frames_synced'],
            'sync_attempts': self.stats['sync_attempts'],
            'sync_failures': self.stats['sync_failures'],
            'buffer_overflows': self.stats['buffer_overflows'],
            'sync_success_rate': (
                self.stats['frames_synced'] / max(1, self.stats['sync_attempts'])
            ) * 100,
            'buffer_sizes': {
                cam_id: len(buffer)
                for cam_id, buffer in self.frame_buffers.items()
            }
        }
    
    def get_buffer_info(self) -> Dict:
        """Get detailed buffer information for debugging."""
        info = {}
        for camera_id, buffer in self.frame_buffers.items():
            if buffer:
                timestamps = [f.timestamp for f in buffer]
                info[camera_id] = {
                    'size': len(buffer),
                    'oldest_timestamp': min(timestamps),
                    'newest_timestamp': max(timestamps),
                    'span_ms': (max(timestamps) - min(timestamps)) / 1_000_000.0
                }
        return info


if __name__ == "__main__":
    # Test frame synchronizer
    import time as _time
    
    print("Testing FrameSynchronizer...")
    sync = FrameSynchronizer(time_threshold_ms=33.0)
    
    # Simulate two cameras capturing frames
    base_timestamp = _time.perf_counter_ns()
    frame_interval_ns = 33_333_333  # ~30 FPS
    
    print("\nAdding frames from two cameras...")
    for i in range(10):
        # Camera 0 - perfectly timed
        ts0 = base_timestamp + i * frame_interval_ns
        sync.add_frame('cam_0', i, ts0, {'data': f'cam0_frame{i}'})
        
        # Camera 1 - slightly offset (+5ms)
        ts1 = base_timestamp + i * frame_interval_ns + 5_000_000  # +5ms
        sync.add_frame('cam_1', i, ts1, {'data': f'cam1_frame{i}'})
        
        # Try to get synced frames
        synced = sync.get_synced_frames()
        if synced:
            print(f"Synced batch {i}: {[(f.camera_id, f.time_offset) for f in synced]}")
    
    print("\nStatistics:")
    import json
    print(json.dumps(sync.get_stats(), indent=2))
    
    print("\nBuffer Info:")
    print(json.dumps(sync.get_buffer_info(), indent=2))
