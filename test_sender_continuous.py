"""
Continuous test sender - runs until Ctrl+C
"""

from src.camera_server import CameraServer
import time

print("Starting continuous mock camera server...")
print("Press Ctrl+C to stop")
server = CameraServer('cam_0')
server.start()

print(f"\nBroadcasting on IP: {server.local_ip}")
print("Sending mock frames continuously...")

frame_num = 0
try:
    while True:
        # Mock detection results
        mock_results = {
            'pose_landmarks': [[
                {
                    'x': 0.5 + (frame_num % 10) * 0.01,
                    'y': 0.5,
                    'z': 0.1,
                    'visibility': 0.99
                }
                for _ in range(33)
            ]],
            'face_landmarks': [],
            'left_hand_landmarks': [],
            'right_hand_landmarks': [],
            'num_people': 1,
            'timestamp': time.perf_counter_ns()
        }
        
        timestamp = time.perf_counter_ns()
        server.send_frame_data(frame_num, timestamp, mock_results)
        
        if frame_num % 30 == 0:
            print(f"  Sent {frame_num} frames...")
        
        frame_num += 1
        time.sleep(0.033)  # ~30 FPS

except KeyboardInterrupt:
    print(f"\n✅ Sent {frame_num} total frames")
    server.stop()
