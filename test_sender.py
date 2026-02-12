"""
Simple test to send mock frame data without needing a camera.
"""

from src.camera_server import CameraServer
import time

print("Starting mock camera server...")
server = CameraServer('cam_0')
server.start()

print(f"Broadcasting on IP: {server.local_ip}")
print("Sending mock frames for 30 seconds...")

for i in range(300):  # 300 frames = ~30 seconds
    # Mock detection results - simple dict format
    mock_results = {
        'pose_landmarks': [[  # List of people
            {
                'x': 0.5 + (i % 10) * 0.01,  # Slight movement
                'y': 0.5,
                'z': 0.1,
                'visibility': 0.99
            }
            for _ in range(33)  # 33 pose landmarks
        ]],
        'face_landmarks': [],
        'left_hand_landmarks': [],
        'right_hand_landmarks': [],
        'num_people': 1,
        'timestamp': time.perf_counter_ns()
    }
    
    timestamp = time.perf_counter_ns()
    server.send_frame_data(i, timestamp, mock_results)
    
    if i % 30 == 0:
        print(f"  Sent {i} frames...")
    
    time.sleep(0.033)  # ~30 FPS

print("\n✅ Finished sending mock data!")
server.stop()
