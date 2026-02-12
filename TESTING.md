# Quick Fix Commands

## PC2 (Mac) - Pull Latest Code

```bash
cd ~/motion-capture
git pull origin multicamera-v5
```

Then try again:
```bash
python launch_multi_camera.py --mode master --remote-ip 10.51.179.228
```

## PC1 (Windows) - Launch Server

```powershell
.\venv\Scripts\Activate.ps1
python launch_multi_camera.py --mode server
```

---

## What Was Fixed

- Separated `CAMERA_ID = 0` (integer for cv2.VideoCapture)
- From `NETWORK_CAMERA_ID = 'cam_0'` (string for network)
- Both systems use camera device 0, but identify as 'cam_0' on network

## If Camera Still Doesn't Open

Check if another app is using the camera:
- Close Zoom, Teams, Skype, etc.
- On Mac: System Settings → Privacy & Security → Camera
- Grant permission to Terminal/Python if needed
