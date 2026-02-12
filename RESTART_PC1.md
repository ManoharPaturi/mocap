# PC1 (Master) - Restart Instructions

**On PC1 (Windows):**

```powershell
# Stop current process (Ctrl+C)

# Pull latest code
git pull origin multicamera-v5

# Clear cache
Remove-Item -Recurse -Force .\__pycache__, .\src\__pycache__ -ErrorAction SilentlyContinue

# Restart with master mode
python launch_multi_camera.py --mode master --remote-ip 10.51.179.38
```

---

## What You Should See:

**OpenCV Window**: "Dual Camera View - Master"
- **Left half**: Your local camera (PC1) with live detection
- **Right half**: Remote camera placeholder (black for now)
- Labels on each side

**Tkinter GUI**: Full dashboard with metrics, FPS, recording controls

---

## If still showing single camera:

Check the console output for:
```
[GUI] Master Coordinator started - connecting to 10.51.179.38
[MasterCoordinator] Connected to cam_0 data stream
```

If NOT connected, make sure:
1. PC2 is running in server mode
2. Both on same network (10.51.179.x)
3. Firewall ports 5000-5001 open
