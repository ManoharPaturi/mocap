# Multi-Camera Setup Guide

Step-by-step instructions for running the VS5 Motion Capture System across **two PCs**.

---

## Architecture

```
PC1 (Server)                    PC2 (Master)
──────────────                  ──────────────
Camera → Detector          →    MasterCoordinator
     ↓                              ↓
CameraServer               →    FrameSynchronizer
(Port 5000 Discovery)                ↓
(Port 5001 Data Stream)         Triangulator → 3D Pose
                                     ↓
                              Dual Camera View
                              (Local + Remote)
```

| Role | What it does |
|------|-------------|
| **Server** | Captures video, runs pose detection locally, and broadcasts frames over the network |
| **Master** | Captures its own video + receives the Server's frames, synchronizes them, and computes 3D poses |

---

## Prerequisites

### Both PCs

1. **Python 3.8–3.10** installed
2. Clone the repository:
   ```bash
   git clone https://github.com/Mrudula-itsjuzme/Motion-capture.git
   cd Motion-capture/vs5
   ```
3. Create and activate virtual environment:
   ```powershell
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Network Requirements

- Both PCs must be on the **same Wi-Fi network** (same subnet, e.g. `10.137.227.x`)
- Ports **5000** (discovery) and **5001** (data) must be open on the **Server PC**

---

## Step 1: Find IP Addresses

Run on **both PCs**:

```powershell
# Windows
ipconfig
```
```bash
# Mac/Linux
ifconfig en0 | grep "inet " | awk '{print $2}'
```

Note down the IPv4 address from the **Wi-Fi** adapter. Example:
- PC1 (Server): `10.137.227.228`
- PC2 (Master): `10.137.227.217`

> **Tip:** Both IPs should share the same first three octets (e.g. `10.137.227.x`).

---

## Step 2: Open Firewall (Server PC Only)

On the **Server PC**, run **PowerShell as Administrator**:

```powershell
New-NetFirewallRule -DisplayName "MoCap Port 5000" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "MoCap Port 5001" -Direction Inbound -LocalPort 5001 -Protocol TCP -Action Allow
```

Or run the provided script:
```powershell
# Right-click PowerShell → Run as Administrator
.\scripts\allow_firewall.ps1
```

---

## Step 3: Start the Server PC

On the **Server PC**:

```powershell
.\venv\Scripts\Activate.ps1
python launch_multi_camera.py --mode server
```

You should see:
```
[CameraServer] Initialized with ID: cam_0 on IP: 10.137.227.228
[CameraServer] Started broadcasting on port 5000
[GUI] Camera Server started - broadcasting to network
[SERVER] Queued frame 0
[SERVER] Queued frame 1
...
```

> **Important:** Wait until frames start queuing before starting the Master PC.

---

## Step 4: Start the Master PC

On the **Master PC**, replace `<SERVER_IP>` with the Server PC's IP:

```powershell
.\venv\Scripts\Activate.ps1
python launch_multi_camera.py --mode master --remote-ip <SERVER_IP>
```

Example:
```powershell
python launch_multi_camera.py --mode master --remote-ip 10.137.227.228
```

Or use the helper script:
```powershell
.\scripts\run_master.bat
```

You should see:
```
[MasterCoordinator] Connecting to camera at 10.137.227.228...
[MasterCoordinator] Discovered camera: cam_0 at 10.137.227.228
[MasterCoordinator] Connected to cam_0 data stream
```

---

## Step 5: Verify

On the **Master PC**, you should see:
- **OpenCV Window**: "Dual Camera View - Master"
  - Left half: Local camera (Master PC)
  - Right half: Remote camera (Server PC) with "LIVE" or "SYNCED ✓" label
- **Tkinter GUI**: Full dashboard with metrics, FPS, recording controls

If the remote feed shows **"NO DATA"** or **"WAITING"**, see Troubleshooting below.

---

## Quick Reference

| Action | Server PC | Master PC |
|--------|-----------|-----------|
| Activate venv | `.\venv\Scripts\Activate.ps1` | `.\venv\Scripts\Activate.ps1` |
| Launch | `python launch_multi_camera.py --mode server` | `python launch_multi_camera.py --mode master --remote-ip <IP>` |
| Stop | `Ctrl+C` | Press `q` in OpenCV window or `Ctrl+C` |
| Test ports | — | `python tests\test_remote_ports.py` |

---

## Troubleshooting

### "NO DATA" on remote feed
1. **Check the Server** is running and shows `[SERVER] Queued frame X`
2. **Check the IP** — run `ipconfig` on both PCs, confirm same subnet
3. **Check firewall** — run `scripts\allow_firewall.ps1` as Admin on Server PC

### Port 5001 closed
```powershell
# Run on Server PC as Admin
New-NetFirewallRule -DisplayName "MoCap Data 5001" -Direction Inbound -LocalPort 5001 -Protocol TCP -Action Allow
```

### "Access Denied" on firewall
Right-click PowerShell → **Run as Administrator**, then retry.

### High latency / frame drops
- Move PCs closer to the Wi-Fi router
- Close bandwidth-heavy apps (downloads, streaming)
- Try wired Ethernet if available

### Camera not opening
- Close Zoom, Teams, Skype, or any app using the webcam
- On Mac: System Settings → Privacy → Camera → Allow Terminal/Python

---

## Recording a Session

1. Start both PCs as above
2. On the **Master PC GUI**, click **"▶ Start Capture"**
3. Perform the motion capture session
4. Click **"⏹ Stop Capture"**
5. Click **"📥 Download Dataset"** to export CSV
6. Click **"📊 Visualize Session"** for interactive 3D playback

> In Master mode, recordings include **synchronized data from both cameras** with triangulated 3D poses.
