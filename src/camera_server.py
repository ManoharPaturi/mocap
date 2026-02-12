"""
Camera Server Module
Runs on each laptop to broadcast camera presence and stream detection results to master.
"""

import zmq
import msgpack
import time
import json
import socket
import cv2
from typing import Optional, Dict, Any
from threading import Thread, Event
from config import (
    CAMERA_ID, DISCOVERY_PORT, DATA_PORT, COMPRESS_NETWORK_DATA,
    NETWORK_PROTOCOL
)


class CameraServer:
    """
    Camera Server for multi-camera setup.
    Broadcasts discovery packets and streams detection results to master.
    """
    
    def __init__(self, camera_id: str = CAMERA_ID, master_ip: Optional[str] = None):
        """
        Initialize camera server.
        
        Args:
            camera_id: Unique identifier for this camera
            master_ip: IP address of master coordinator (None for auto-discovery)
        """
        self.camera_id = camera_id
        self.master_ip = master_ip
        self.running = False
        self.discovery_thread = None
        self.data_thread = None
        self.stop_event = Event()
        
        # Get local IP address
        self.local_ip = self._get_local_ip()
        
        # ZMQ context and sockets
        self.context = zmq.Context()
        self.discovery_socket = None
        self.data_socket = None
        
        # Frame counter
        self.frame_count = 0
        
        print(f"[CameraServer] Initialized with ID: {camera_id} on IP: {self.local_ip}")
    
    def _get_local_ip(self) -> str:
        """Get local IP address of this machine."""
        try:
            # Create a UDP socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Connect to a public DNS server (doesn't actually send data)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            # Fallback to localhost
            return '127.0.0.1'
    
    def start(self):
        """Start the camera server (discovery and data streaming)."""
        if self.running:
            print("[CameraServer] Already running")
            return
        
        self.running = True
        self.stop_event.clear()
        
        # Setup sockets
        self._setup_sockets()
        
        # Start discovery broadcast thread
        self.discovery_thread = Thread(target=self._discovery_loop, daemon=True)
        self.discovery_thread.start()
        
        print(f"[CameraServer] Started broadcasting on port {DISCOVERY_PORT}")
    
    def stop(self):
        """Stop the camera server."""
        if not self.running:
            return
        
        print("[CameraServer] Stopping...")
        self.running = False
        self.stop_event.set()
        
        # Wait for threads
        if self.discovery_thread:
            self.discovery_thread.join(timeout=2.0)
        
        # Close sockets
        if self.discovery_socket:
            self.discovery_socket.close()
        if self.data_socket:
            self.data_socket.close()
        
        self.context.term()
        print("[CameraServer] Stopped")
    
    def _setup_sockets(self):
        """Setup ZMQ sockets for discovery and data transmission."""
        # Discovery socket (TCP broadcast using PUB socket)
        self.discovery_socket = self.context.socket(zmq.PUB)
        self.discovery_socket.bind(f"tcp://*:{DISCOVERY_PORT}")
        
        # Data socket (PUB-SUB pattern)
        self.data_socket = self.context.socket(zmq.PUB)
        self.data_socket.bind(f"tcp://*:{DATA_PORT}")
    
    def _discovery_loop(self):
        """Continuously broadcast discovery packets."""
        while self.running and not self.stop_event.is_set():
            try:
                discovery_msg = {
                    'type': 'discovery',
                    'camera_id': self.camera_id,
                    'ip': self.local_ip,
                    'port': DATA_PORT,
                    'timestamp': time.time()
                }
                
                # Serialize
                if COMPRESS_NETWORK_DATA:
                    data = msgpack.packb(discovery_msg)
                else:
                    data = json.dumps(discovery_msg).encode('utf-8')
                
                # Broadcast
                self.discovery_socket.send(data)
                
            except Exception as e:
                print(f"[CameraServer] Discovery error: {e}")
            
            # Broadcast every 2 seconds
            time.sleep(2.0)
    
    def send_frame_data(self, frame_number: int, timestamp: float, results: Dict[str, Any], frame=None):
        """
        Send detection results and frame to master.
        
        Args:
            frame_number: Sequential frame number
            timestamp: High-precision timestamp (from time.time() * 1e9)
            results: Detection results from MocapDetector
            frame: Optional numpy array of the camera frame (will be JPEG encoded)
        """
        if not self.running:
            return
        
        try:
            # Encode frame as JPEG if provided
            frame_jpeg = None
            if frame is not None:
                # Compress to JPEG (quality 70 for faster encoding)
                success, jpeg_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if success:
                    frame_jpeg = jpeg_buffer.tobytes()
            
            # Package frame data
            frame_data = {
                'type': 'frame_data',
                'camera_id': self.camera_id,
                'frame_number': frame_number,
                'timestamp': timestamp,
                'results': self._serialize_results(results),
                'frame_jpeg': frame_jpeg  # JPEG-encoded frame bytes
            }
            
            # Serialize
            if COMPRESS_NETWORK_DATA:
                data = msgpack.packb(frame_data)
            else:
                data = json.dumps(frame_data).encode('utf-8')
            
            # Send
            self.data_socket.send(data)
            self.frame_count += 1
            
        except Exception as e:
            print(f"[CameraServer] Error sending frame data: {e}")
    
    def _serialize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize MediaPipe results to JSON-compatible format.
        Only send landmark coordinates, not the full results object.
        """
        serialized = {}
        
        # Serialize pose landmarks
        if 'pose_landmarks' in results and results['pose_landmarks']:
            # Check if already serialized (list of dicts)
            if isinstance(results['pose_landmarks'][0], list) and isinstance(results['pose_landmarks'][0][0], dict):
                serialized['pose_landmarks'] = results['pose_landmarks']
            else:
                serialized['pose_landmarks'] = [
                    self._serialize_landmark_list(lm_list) 
                    for lm_list in results['pose_landmarks']
                ]
        
        # Serialize pose world landmarks
        if 'pose_world_landmarks' in results and results['pose_world_landmarks']:
            if isinstance(results['pose_world_landmarks'][0], list) and isinstance(results['pose_world_landmarks'][0][0], dict):
                serialized['pose_world_landmarks'] = results['pose_world_landmarks']
            else:
                serialized['pose_world_landmarks'] = [
                    self._serialize_landmark_list(lm_list) 
                    for lm_list in results['pose_world_landmarks']
                ]
        
        # Serialize face landmarks (optional)
        if 'face_landmarks' in results and results['face_landmarks']:
            serialized['face_landmarks'] = results['face_landmarks']
        
        # Serialize hand landmarks (optional)
        if 'hand_landmarks' in results and results['hand_landmarks']:
            serialized['hand_landmarks'] = results['hand_landmarks']
        
        return serialized
    
    def _serialize_landmark_list(self, landmark_list) -> list:
        """Convert MediaPipe landmark list to simple dict format."""
        return [
            {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': getattr(lm, 'visibility', 1.0)
            }
            for lm in landmark_list
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            'camera_id': self.camera_id,
            'ip': self.local_ip,
            'running': self.running,
            'frames_sent': self.frame_count
        }


if __name__ == "__main__":
    # Test server
    print("Starting Camera Server test...")
    server = CameraServer("test_cam")
    server.start()
    
    try:
        # Run for 10 seconds
        time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
