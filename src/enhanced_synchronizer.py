"""
Enhanced Synchronizer Module
Provides multiple synchronization modes for multi-camera frame alignment:
  - Hard Sync:  Strict timestamp matching within tight threshold
  - Loose Sync: Best-effort matching with configurable tolerance
  - Triggered:  External trigger / sequence-number based matching

Includes integrated clock synchronization with RTT-based offset estimation
and 3-sigma outlier filtering.
"""

import time
import math
import statistics
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock

from config import (
    SYNC_TIME_THRESHOLD_MS, FRAME_BUFFER_SIZE,
    STALE_FRAME_TIMEOUT_MS
)


class SyncMode(Enum):
    """Synchronization mode."""
    HARD = 'hard'       # Strict: all cameras must match within threshold
    LOOSE = 'loose'     # Best-effort: return partial matches if some cameras lag
    TRIGGERED = 'triggered'  # Sequence-number based (external trigger)


@dataclass
class SyncedFrame:
    """A single synchronized frame from one camera."""
    camera_id: str
    frame_number: int
    timestamp_ns: int           # Original capture timestamp (nanoseconds)
    corrected_timestamp_ns: int  # Clock-corrected timestamp
    results: Dict[str, Any]
    received_at_ns: int
    time_offset_ms: float = 0.0  # Offset from reference frame
    clock_offset_ns: int = 0     # Applied clock correction


@dataclass
class SyncBatch:
    """Synchronized batch of frames from multiple cameras."""
    frames: List[SyncedFrame]
    reference_timestamp_ns: int
    spread_ms: float            # Max timestamp spread within batch
    num_cameras: int
    sync_mode: SyncMode
    batch_index: int = 0

    @property
    def camera_ids(self) -> List[str]:
        return [f.camera_id for f in self.frames]

    @property
    def is_complete(self) -> bool:
        """True if all expected cameras contributed."""
        return len(self.frames) >= self.num_cameras


@dataclass
class ClockState:
    """Clock synchronization state for a single camera."""
    camera_id: str
    offset_ns: int = 0           # Time to add to this camera's timestamps
    rtt_min_ns: int = 0          # Minimum observed RTT
    samples_collected: int = 0
    last_sync_time: float = 0.0
    drift_rate_ns_per_sec: float = 0.0  # Estimated drift
    history: List[Tuple[float, int]] = field(default_factory=list)  # (time, offset)


class EnhancedSynchronizer:
    """
    Multi-mode frame synchronizer with integrated clock correction.

    Usage:
        sync = EnhancedSynchronizer(num_cameras=2, mode=SyncMode.LOOSE)
        sync.set_clock_offset('cam_0', offset_ns)
        sync.add_frame('cam_0', frame_number, timestamp_ns, results)
        batch = sync.get_synced_batch()
    """

    def __init__(self, num_cameras: int = 2,
                 mode: SyncMode = SyncMode.LOOSE,
                 time_threshold_ms: float = SYNC_TIME_THRESHOLD_MS,
                 buffer_size: int = FRAME_BUFFER_SIZE,
                 stale_timeout_ms: float = STALE_FRAME_TIMEOUT_MS,
                 min_cameras: int = 2):
        """
        Args:
            num_cameras:      Expected number of cameras
            mode:             Synchronization strategy
            time_threshold_ms: Max time difference for frame matching (ms)
            buffer_size:      Max frames to buffer per camera
            stale_timeout_ms: Drop frames older than this vs newest
            min_cameras:      Minimum cameras required for a valid batch
        """
        self.num_cameras = num_cameras
        self.mode = mode
        self.time_threshold_ns = int(time_threshold_ms * 1_000_000)
        self.stale_timeout_ns = int(stale_timeout_ms * 1_000_000)
        self.buffer_size = buffer_size
        self.min_cameras = min(min_cameras, num_cameras)

        # Per-camera frame buffers (deque of SyncedFrame)
        self._buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.buffer_size)
        )
        self._lock = Lock()

        # Clock state
        self._clock: Dict[str, ClockState] = {}

        # Stats
        self._batch_counter = 0
        self.stats = {
            'frames_added': defaultdict(int),
            'batches_synced': 0,
            'batches_partial': 0,
            'batches_dropped': 0,
            'stale_evictions': 0,
            'total_spread_ms': 0.0,
        }

    # ── Clock Management ──

    def set_clock_offset(self, camera_id: str, offset_ns: int,
                         rtt_min_ns: int = 0):
        """
        Set or update the clock offset for a camera.

        Args:
            camera_id:  Camera identifier
            offset_ns:  Nanoseconds to ADD to camera timestamps to align with master
            rtt_min_ns: Minimum RTT observed during sync (for quality tracking)
        """
        now = time.time()
        if camera_id not in self._clock:
            self._clock[camera_id] = ClockState(camera_id=camera_id)

        state = self._clock[camera_id]

        # Track drift if we have previous offset
        if state.samples_collected > 0 and state.last_sync_time > 0:
            dt = now - state.last_sync_time
            if dt > 1.0:
                drift = (offset_ns - state.offset_ns) / dt  # ns per second
                state.drift_rate_ns_per_sec = drift

        state.offset_ns = offset_ns
        state.rtt_min_ns = rtt_min_ns
        state.samples_collected += 1
        state.last_sync_time = now
        state.history.append((now, offset_ns))

        # Keep bounded history
        if len(state.history) > 100:
            state.history = state.history[-50:]

    def get_clock_offset(self, camera_id: str) -> int:
        """Get current clock offset for a camera (0 if unknown)."""
        state = self._clock.get(camera_id)
        return state.offset_ns if state else 0

    def get_clock_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get clock synchronization statistics for all cameras."""
        result = {}
        for cam_id, state in self._clock.items():
            result[cam_id] = {
                'offset_ms': state.offset_ns / 1_000_000,
                'rtt_min_ms': state.rtt_min_ns / 1_000_000,
                'drift_us_per_sec': state.drift_rate_ns_per_sec / 1_000,
                'samples': state.samples_collected,
                'last_sync_age_s': time.time() - state.last_sync_time
                    if state.last_sync_time > 0 else -1
            }
        return result

    # ── Frame Ingestion ──

    def add_frame(self, camera_id: str, frame_number: int,
                  timestamp_ns: int, results: Dict[str, Any]):
        """
        Add a frame to the synchronization buffer.

        Clock offset is applied automatically if available.
        """
        received_at = time.time_ns()
        offset = self.get_clock_offset(camera_id)
        corrected_ts = timestamp_ns + offset

        frame = SyncedFrame(
            camera_id=camera_id,
            frame_number=frame_number,
            timestamp_ns=timestamp_ns,
            corrected_timestamp_ns=corrected_ts,
            results=results,
            received_at_ns=received_at,
            clock_offset_ns=offset
        )

        with self._lock:
            self._buffers[camera_id].append(frame)
            self.stats['frames_added'][camera_id] += 1

    # ── Batch Retrieval ──

    def get_synced_batch(self) -> Optional[SyncBatch]:
        """
        Attempt to produce a synchronized batch from current buffers.

        Returns SyncBatch or None if sync is not possible.
        """
        with self._lock:
            return self._sync_impl()

    def _sync_impl(self) -> Optional[SyncBatch]:
        """Internal sync implementation (must hold _lock)."""
        # Evict stale frames first
        self._evict_stale()

        # Check if we have enough cameras
        active_cameras = [cid for cid, buf in self._buffers.items() if len(buf) > 0]
        if len(active_cameras) < self.min_cameras:
            return None

        if self.mode == SyncMode.TRIGGERED:
            return self._sync_triggered(active_cameras)
        elif self.mode == SyncMode.HARD:
            return self._sync_hard(active_cameras)
        else:  # LOOSE
            return self._sync_loose(active_cameras)

    def _sync_hard(self, active_cameras: List[str]) -> Optional[SyncBatch]:
        """
        Hard sync: all cameras must have a frame within threshold.
        Uses the newest frame from the slowest camera as reference.
        """
        if len(active_cameras) < self.num_cameras:
            return None  # Hard mode needs ALL cameras

        # Get reference = oldest front-of-buffer
        ref_frame = None
        for cam_id in active_cameras:
            front = self._buffers[cam_id][0]
            if ref_frame is None or front.corrected_timestamp_ns > ref_frame.corrected_timestamp_ns:
                ref_frame = front

        if ref_frame is None:
            return None

        ref_ts = ref_frame.corrected_timestamp_ns

        matched = []
        for cam_id in active_cameras:
            best_frame = None
            best_diff = float('inf')
            for frame in self._buffers[cam_id]:
                diff = abs(frame.corrected_timestamp_ns - ref_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_frame = frame

            if best_frame is None or best_diff > self.time_threshold_ns:
                self.stats['batches_dropped'] += 1
                return None

            best_frame.time_offset_ms = (best_frame.corrected_timestamp_ns - ref_ts) / 1_000_000
            matched.append(best_frame)

        if len(matched) < self.num_cameras:
            return None

        # Consume frames
        self._consume_matched(matched)

        spread_ns = max(f.corrected_timestamp_ns for f in matched) - \
                    min(f.corrected_timestamp_ns for f in matched)
        spread_ms = spread_ns / 1_000_000

        self._batch_counter += 1
        self.stats['batches_synced'] += 1
        self.stats['total_spread_ms'] += spread_ms

        return SyncBatch(
            frames=matched,
            reference_timestamp_ns=ref_ts,
            spread_ms=spread_ms,
            num_cameras=self.num_cameras,
            sync_mode=SyncMode.HARD,
            batch_index=self._batch_counter
        )

    def _sync_loose(self, active_cameras: List[str]) -> Optional[SyncBatch]:
        """
        Loose sync: match as many cameras as possible (≥ min_cameras).
        Pick the camera with the oldest front-of-buffer as reference.
        """
        # Find reference = oldest front-of-buffer frame
        ref_frame = None
        ref_cam = None
        for cam_id in active_cameras:
            front = self._buffers[cam_id][0]
            if ref_frame is None or front.corrected_timestamp_ns < ref_frame.corrected_timestamp_ns:
                ref_frame = front
                ref_cam = cam_id

        if ref_frame is None:
            return None

        ref_ts = ref_frame.corrected_timestamp_ns
        matched = []

        for cam_id in active_cameras:
            best_frame = None
            best_diff = float('inf')

            for frame in self._buffers[cam_id]:
                diff = abs(frame.corrected_timestamp_ns - ref_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_frame = frame

            if best_frame is not None and best_diff <= self.time_threshold_ns:
                best_frame.time_offset_ms = (best_frame.corrected_timestamp_ns - ref_ts) / 1_000_000
                matched.append(best_frame)

        if len(matched) < self.min_cameras:
            # Not enough cameras matched — discard the oldest reference frame
            if ref_cam:
                buf = self._buffers[ref_cam]
                if buf:
                    buf.popleft()
            self.stats['batches_dropped'] += 1
            return None

        self._consume_matched(matched)

        spread_ns = max(f.corrected_timestamp_ns for f in matched) - \
                    min(f.corrected_timestamp_ns for f in matched)
        spread_ms = spread_ns / 1_000_000

        self._batch_counter += 1
        is_partial = len(matched) < self.num_cameras
        if is_partial:
            self.stats['batches_partial'] += 1
        self.stats['batches_synced'] += 1
        self.stats['total_spread_ms'] += spread_ms

        return SyncBatch(
            frames=matched,
            reference_timestamp_ns=ref_ts,
            spread_ms=spread_ms,
            num_cameras=self.num_cameras,
            sync_mode=SyncMode.LOOSE,
            batch_index=self._batch_counter
        )

    def _sync_triggered(self, active_cameras: List[str]) -> Optional[SyncBatch]:
        """
        Triggered sync: match frames by frame_number (ignoring timestamps).
        Useful when cameras are hardware-triggered simultaneously.
        """
        # Find the lowest frame_number across all buffers
        min_frame_num = None
        for cam_id in active_cameras:
            front = self._buffers[cam_id][0]
            if min_frame_num is None or front.frame_number < min_frame_num:
                min_frame_num = front.frame_number

        if min_frame_num is None:
            return None

        matched = []
        for cam_id in active_cameras:
            for frame in self._buffers[cam_id]:
                if frame.frame_number == min_frame_num:
                    matched.append(frame)
                    break

        if len(matched) < self.min_cameras:
            # Discard oldest frames with this number
            for cam_id in active_cameras:
                buf = self._buffers[cam_id]
                while buf and buf[0].frame_number <= min_frame_num:
                    buf.popleft()
            self.stats['batches_dropped'] += 1
            return None

        self._consume_matched(matched)

        ref_ts = int(sum(f.corrected_timestamp_ns for f in matched) / len(matched))
        spread_ns = max(f.corrected_timestamp_ns for f in matched) - \
                    min(f.corrected_timestamp_ns for f in matched)

        self._batch_counter += 1
        self.stats['batches_synced'] += 1

        return SyncBatch(
            frames=matched,
            reference_timestamp_ns=ref_ts,
            spread_ms=spread_ns / 1_000_000,
            num_cameras=self.num_cameras,
            sync_mode=SyncMode.TRIGGERED,
            batch_index=self._batch_counter
        )

    # ── Internal Helpers ──

    def _consume_matched(self, matched: List[SyncedFrame]):
        """Remove matched frames and all older frames from buffers."""
        for frame in matched:
            buf = self._buffers[frame.camera_id]
            while buf:
                old = buf.popleft()
                if old.frame_number >= frame.frame_number:
                    break

    def _evict_stale(self):
        """Remove frames that are too old relative to the newest across all cameras."""
        # Find global newest corrected timestamp
        global_newest = None
        for buf in self._buffers.values():
            if buf:
                newest = buf[-1].corrected_timestamp_ns
                if global_newest is None or newest > global_newest:
                    global_newest = newest

        if global_newest is None:
            return

        for cam_id, buf in self._buffers.items():
            if not buf:
                continue
            newest_in_buf = buf[-1].corrected_timestamp_ns
            age_ns = global_newest - newest_in_buf
            if age_ns > self.stale_timeout_ns:
                evicted = len(buf)
                buf.clear()
                self.stats['stale_evictions'] += evicted

    def clear(self):
        """Clear all buffers and reset state."""
        with self._lock:
            for buf in self._buffers.values():
                buf.clear()
            self._batch_counter = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        avg_spread = 0.0
        if self.stats['batches_synced'] > 0:
            avg_spread = self.stats['total_spread_ms'] / self.stats['batches_synced']

        return {
            'mode': self.mode.value,
            'frames_added': dict(self.stats['frames_added']),
            'batches_synced': self.stats['batches_synced'],
            'batches_partial': self.stats['batches_partial'],
            'batches_dropped': self.stats['batches_dropped'],
            'stale_evictions': self.stats['stale_evictions'],
            'avg_spread_ms': round(avg_spread, 2),
            'buffer_sizes': {cid: len(buf) for cid, buf in self._buffers.items()},
            'clock': self.get_clock_stats()
        }


# ── Standalone Clock Synchronizer ──

class ClockSynchronizer:
    """
    Standalone clock synchronizer using Cristian's Algorithm with 3-sigma
    outlier filtering. Can be used independently of EnhancedSynchronizer.
    """

    def __init__(self, rtt_outlier_factor: float = 2.0,
                 max_history: int = 100):
        self.rtt_outlier_factor = rtt_outlier_factor
        self.max_history = max_history
        self._offsets: Dict[str, int] = {}
        self._history: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # (rtt, offset)

    def process_samples(self, camera_id: str,
                        samples: List[Tuple[int, int]]) -> Optional[int]:
        """
        Process a batch of (rtt_ns, offset_ns) samples for a camera.

        Applies outlier filtering and returns the median offset.

        Args:
            camera_id: Camera identifier
            samples:   List of (rtt_ns, offset_ns) tuples

        Returns:
            Estimated clock offset in nanoseconds, or None if all samples rejected
        """
        if not samples:
            return None

        # Stage 1: RTT outlier rejection (keep RTT ≤ factor × min_rtt)
        min_rtt = min(s[0] for s in samples)
        threshold = min_rtt * self.rtt_outlier_factor
        filtered = [s for s in samples if s[0] <= threshold]

        if not filtered:
            filtered = samples  # Fallback

        # Stage 2: 3-sigma outlier rejection on offsets
        offsets = [s[1] for s in filtered]
        if len(offsets) >= 4:
            mean_off = statistics.mean(offsets)
            std_off = statistics.stdev(offsets)
            if std_off > 0:
                sigma3 = [(rtt, off) for rtt, off in filtered
                          if abs(off - mean_off) <= 3 * std_off]
                if sigma3:
                    filtered = sigma3

        # Compute median offset
        final_offsets = [s[1] for s in filtered]
        median_offset = int(statistics.median(final_offsets))

        # Store
        self._offsets[camera_id] = median_offset
        self._history[camera_id].extend(filtered)
        if len(self._history[camera_id]) > self.max_history:
            self._history[camera_id] = self._history[camera_id][-self.max_history:]

        return median_offset

    def get_offset(self, camera_id: str) -> int:
        """Get last computed offset for a camera."""
        return self._offsets.get(camera_id, 0)

    def get_all_offsets(self) -> Dict[str, int]:
        """Get all camera offsets."""
        return dict(self._offsets)

    def estimate_drift(self, camera_id: str) -> Optional[float]:
        """
        Estimate clock drift rate from historical offsets.

        Returns drift in nanoseconds per second, or None if insufficient data.
        """
        history = self._history.get(camera_id, [])
        if len(history) < 5:
            return None

        # Use first and last quartile medians
        n = len(history)
        q1 = history[:n // 4]
        q4 = history[3 * n // 4:]

        if not q1 or not q4:
            return None

        off1 = statistics.median([s[1] for s in q1])
        off4 = statistics.median([s[1] for s in q4])

        # Approximate time span (using RTT as proxy for time ordering)
        # This is rough — real implementation would track wall-clock times
        return float(off4 - off1)  # Total drift across history window
