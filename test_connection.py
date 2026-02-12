"""Quick test to check master coordinator receives data"""
from src.master_coordinator import MasterCoordinator
import time

coordinator = MasterCoordinator(num_cameras=2)
coordinator.start()
coordinator.discover_cameras_manual(['10.51.179.38'])

print("Listening for 10 seconds...")
time.sleep(2)

for i in range(20):
    batch = coordinator.get_synchronized_batch()
    if batch:
        print(f"✅ Received {len(batch)} synchronized frames!")
        for frame_data in batch:
            print(f"   Camera: {frame_data.camera_id}, Frame: {frame_data.frame_number}")
    else:
        print(f"⏳ No synchronized batch yet (attempt {i+1}/20)")
    time.sleep(0.5)

coordinator.stop()
print("Done!")
