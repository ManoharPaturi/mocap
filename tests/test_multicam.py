"""
Multi-Camera Test Script
Quick test of the camera server and master coordinator modules.
"""

from src.camera_server import CameraServer
from src.master_coordinator import MasterCoordinator
from src.detector import MocapDetector
from src.camera import Camera
import time
import argparse


def test_camera_server():
    """Test the camera server standalone."""
    print("=" * 60)
    print("CAMERA SERVER TEST")
    print("=" * 60)
    
    server = CameraServer('test_cam')
    server.start()
    
    print(f"\nServer Info:")
    stats = server.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nBroadcasting for 10 seconds...")
    time.sleep(10)
    
    server.stop()
    print("\n✅ Camera server test complete!\n")


def test_camera_server_with_detection():
    """Test camera server with actual detection."""
    print("=" * 60)
    print("CAMERA SERVER + DETECTION TEST")
    print("=" * 60)
    
    try:
        server = CameraServer('cam_0')
        server.start()
        
        print("\nInitializing camera and detector...")
        camera = Camera()  # Uses CAMERA_ID from config (integer 0)
        detector = MocapDetector()
        
        print("Processing 30 frames and sending to network...")
        frame_count = 0
        
        while frame_count < 30 and camera.is_opened():
            frame = camera.read()
            if frame is None:
                break
            
            # Detect
            results = detector.process(frame)
            
            # Send over network
            timestamp = time.perf_counter_ns()
            server.send_frame_data(frame_count, timestamp, results)
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"  Sent {frame_count} frames...")
        
        print(f"\n✅ Sent {frame_count} frames successfully!")
        
        stats = server.get_stats()
        print(f"\nServer Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        camera.release()
        server.stop()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_master_coordinator(camera_ips):
    """
    Test master coordinator with manual camera IPs.
    
    Args:
        camera_ips: List of camera IP addresses (e.g., ['192.168.1.101'])
    """
    print("=" * 60)
    print("MASTER COORDINATOR TEST")
    print("=" * 60)
    
    coordinator = MasterCoordinator(num_cameras=len(camera_ips))
    coordinator.start()
    
    print(f"\nConnecting to cameras: {camera_ips}")
    coordinator.discover_cameras_manual(camera_ips)
    
    print(f"\nListening for synchronized frames (30 seconds)...")
    print("(Make sure camera servers are running on those IPs!)")
    
    start_time = time.time()
    synced_count = 0
    
    while time.time() - start_time < 30:
        batch = coordinator.get_synchronized_batch()
        if batch:
            synced_count += 1
            if synced_count % 10 == 0:
                print(f"  Received {synced_count} synchronized batches...")
        time.sleep(0.033)  # ~30 FPS
    
    print(f"\n✅ Received {synced_count} synchronized batches!")
    
    stats = coordinator.get_stats()
    print(f"\nCoordinator Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    coordinator.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test multi-camera modules")
    parser.add_argument(
        '--mode',
        choices=['server', 'server-detect', 'master'],
        required=True,
        help='Test mode: server, server-detect, or master'
    )
    parser.add_argument(
        '--camera-ips',
        nargs='+',
        help='Camera IP addresses for master mode (e.g., 192.168.1.101 192.168.1.102)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'server':
        test_camera_server()
    elif args.mode == 'server-detect':
        test_camera_server_with_detection()
    elif args.mode == 'master':
        if not args.camera_ips:
            print("❌ Error: --camera-ips required for master mode")
            print("Example: python test_multicam.py --mode master --camera-ips 192.168.1.101")
        else:
            test_master_coordinator(args.camera_ips)
