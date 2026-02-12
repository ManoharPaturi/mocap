# Corrected Multi-Camera Setup

## PC1 (Windows) - MASTER
```powershell
.\venv\Scripts\Activate.ps1
python launch_multi_camera.py --mode master --remote-ip <PC2_IP>
```

**What PC1 shows:**
- Dual camera view (local + remote side-by-side)
- Synchronized frames from both cameras
- 3D triangulation (future)

---

## PC2 (Mac) - SERVER
```bash
source venv/bin/activate
python launch_multi_camera.py --mode server
```

**What PC2 shows:**
- Local camera feed with landmarks
- Broadcasts to PC1 in background

---

## Current Status
✅ Network connection working
✅ Frame synchronization (310+ batches)
✅ Both systems running full GUI
🔄 Dual display on PC1 (showing placeholder for remote - will decode actual frames next)

## Next: Get PC2's IP
On PC2 (Mac):
```bash
ifconfig en0 | grep "inet " | awk '{print $2}'
```

Then relaunch PC1 with that IP!
