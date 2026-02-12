# VS5 Multi-Camera Quick Start

## Single Camera Mode (Normal)

```bash
python main_gui.py
```

## Multi-Camera Setup

### PC1 (Camera Server):
```bash
python launch_multi_camera.py --mode server
```

### PC2 (Master Coordinator):
```bash
python launch_multi_camera.py --mode master --remote-ip 10.51.179.228
```

Replace `10.51.179.228` with PC1's actual IP address (check with `ipconfig` on Windows).

---

## What Happens

**PC1 Server:**
- Full GUI with detection, visualization, database, reports
- Broadcasts detection results to PC2 over network

**PC2 Master:**
- Full GUI with detection from local camera
- Receives + synchronizes frames from PC1
- Shows both cameras (future: side-by-side display)
- 3D pose triangulation from stereo views

---

## Firewall Note

Make sure ports 5000-5001 are allowed in Windows Firewall (see README).
