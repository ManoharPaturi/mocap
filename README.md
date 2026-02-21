# VS5 Motion Capture - Multi-Camera Enhanced Edition

Real-time multi-person pose, face, and hand tracking with **dual-laptop stereo 3D reconstruction**.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)

---

## 🌟 Features

### Core Motion Capture
- **Multi-Person Tracking**: Detect up to 5 people simultaneously
- **Full Body Analysis**: 33 pose landmarks, 468 face landmarks, 21 hand landmarks per hand
- **Real-time Processing**: 25-30 FPS on modern hardware
- **Physics-Based Correction**: 1-Euro filtering and skeletal constraints for realistic motion
- **Comprehensive Metrics**: Joint angles, limb lengths, velocities, kinematics

### NEW: Multi-Camera System ✨
- **Stereo 3D Reconstruction**: Use 2+ laptops for triangulated 3D poses
- **Improved Accuracy**: ±1-2cm precision (vs ±5-10cm single camera)
- **Network Synchronization**: Frame-level timestamp matching
- **Occlusion Handling**: One camera sees what the other misses
- **Metric-Scale Depth**: True distance measurements (meters)

### Advanced Features
- **ROI-Based Detection**: Crop to person for higher effective resolution
- **Image Preprocessing**: Gamma correction, CLAHE, face-guided exposure
- **Dual Interfaces**: Desktop GUI (Tkinter) and Web interface (React)
- **Session Recording**: SQLite/PostgreSQL database with CSV export
- **Automated Reports**: Generated biomechanical analysis with Plotly visualizations
- **3D Visualization**: Interactive skeleton playback

---

## 📋 Requirements

- **OS**: Windows / macOS / Linux
- **Python**: 3.8 - 3.10
- **Webcam**: Built-in or USB camera (1280x720 recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **For Multi-Camera**: 2+ laptops on the same WiFi network

---

## 🚀 Quick Start

### Single Camera Mode

```bash
# Clone repository
git clone https://github.com/Mrudula-itsjuzme/Motion-capture.git
cd Motion-capture/vs5

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1
# Or (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run GUI application
python main_gui.py
```

### Multi-Camera Mode (2 Laptops)

**Laptop 1 (Server)**:
```bash
python launch_multi_camera.py --mode server
```

**Laptop 2 (Master)**:
```bash
# Replace with Laptop 1's IP address
python launch_multi_camera.py --mode master --remote-ip <SERVER_IP>
```

See [SETUP.md](SETUP.md) for detailed multi-camera setup.

---

## 📁 Project Structure

```
vs5/
├── config.py                  # Central configuration
├── main_gui.py                # Desktop GUI (Tkinter)
├── launch_multi_camera.py     # Multi-camera launcher
├── requirements.txt
├── SETUP.md                   # Multi-camera setup guide
│
├── src/                       # Core modules
│   ├── camera.py              # Camera capture
│   ├── detector.py            # MediaPipe detection
│   ├── pose_corrector.py      # Physics-based correction
│   ├── calculations.py        # Biomechanical metrics
│   ├── visualizer.py          # 2D rendering
│   ├── visualizer_3d.py       # 3D Plotly visualization
│   ├── live_visualizer_3d.py  # Real-time 3D display
│   ├── database.py            # Session recording
│   ├── report_generator.py    # Automated reports
│   ├── streamer.py            # Video streaming
│   ├── one_euro_filter.py     # Adaptive smoothing
│   ├── camera_server.py       # Network broadcaster
│   ├── master_coordinator.py  # Frame aggregator
│   ├── frame_synchronizer.py  # Timestamp matching
│   ├── stereo_calibration.py  # Camera calibration
│   └── triangulation.py       # 3D reconstruction
│
├── scripts/                   # Helper scripts
│   ├── run_gui.bat            # Launch single-camera GUI
│   ├── run_master.bat         # Launch master mode
│   └── allow_firewall.ps1     # Open network ports
│
├── tests/                     # Network & integration tests
│   ├── test_connection.py
│   ├── test_multicam.py
│   ├── test_ports.py
│   ├── test_remote_ports.py
│   ├── test_receive_debug.py
│   ├── test_sender.py
│   └── test_sender_continuous.py
│
├── models/                    # MediaPipe model files
├── frontend/                  # React web interface
├── output/                    # Generated dashboards & exports
└── results/                   # Generated report images
```

---

## 🎯 Usage

### Configuration

Edit `config.py` to customize:

```python
# Camera
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Detection
POSE_MODEL_COMPLEXITY = 'FULL'  # LITE, FULL, or HEAVY
NUM_POSES = 5  # Max people to track

# Multi-Camera
ENABLE_MULTI_CAMERA = True
CAMERA_ROLE = 'server'  # 'single', 'server', or 'master'
```

### Desktop GUI

```bash
python main_gui.py
```

Features:
- Live video preview with landmark overlay
- Real-time metrics display
- Recording start/stop
- Model complexity selector
- Image preprocessing controls
- Session export
- 3D visualization launcher




---

## 📊 Multi-Camera System

### Architecture

```
Laptop 1 (Server)         Laptop 2 (Master + Server)
─────────────────         ─────────────────────────
Camera → Detector    →    MasterCoordinator
     ↓                         ↓
CameraServer         →    FrameSynchronizer
(Port 5000-5001)              ↓
                          Triangulator
                              ↓
                          Unified 3D Pose
```

### Setup Steps

1. **Find IP Addresses**: Run `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
2. **Open Firewall**: Run `scripts\allow_firewall.ps1` as Admin on Server PC
3. **Run Server**: `python launch_multi_camera.py --mode server`
4. **Run Master**: `python launch_multi_camera.py --mode master --remote-ip <SERVER_IP>`
5. **Calibrate** (optional): Use checkerboard pattern for accurate 3D

> See [SETUP.md](SETUP.md) for full step-by-step instructions.

### Expected Improvements

| Metric | Single Camera | Dual Camera |
|--------|--------------|-------------|
| 3D Accuracy | ±5-10cm | ±1-2cm |
| Depth Quality | Approximate | Metric-scale |
| Occlusion Handling | Poor | Good |
| Confidence Score | 0.5-0.8 | 0.7-0.95 |

---

## 🔬 Algorithms & Technologies

- **Detection**: MediaPipe Pose/Face/Hand Landmarker
- **Smoothing**: 1-Euro Filter (adaptive low-pass)
- **Correction**: Bone length constraints, joint limits
- **Triangulation**: Direct Linear Transform (DLT)
- **Synchronization**: Timestamp-based frame matching
- **Database**: SQLite (default) or PostgreSQL

---

## 📈 Performance

- **Single Camera**: 25-30 FPS @ 1280x720
- **Multi-Camera**: 15-25 FPS (network overhead)
- **Memory**: ~500 MB (FULL model)
- **CPU**: 40-60% (quad-core)

Optimizations:
- ROI cropping reduces inference cost ~40%
- Frame skipping configurable
- LITE model achieves 60+ FPS

---

## 🛠️ Development

### Running Tests

```bash
# Network layer
python -m src.camera_server
python -m src.master_coordinator

# Frame synchronization
python -m src.frame_synchronizer

# Triangulation
python -m src.triangulation
```

### Adding New Features

See `implementation_plan.md` for architecture details.

---

## 📚 Documentation

- **Setup Guide**: [SETUP.md](SETUP.md) - Detailed dual-PC setup instructions
- **Configuration**: [config.py](config.py) - All tunable parameters

---

## 🎓 Use Cases

- Biomechanical research and analysis
- Physical therapy assessment
- Sports performance analysis
- Animation reference capture
- Human-computer interaction research
- Fitness and exercise tracking

---

## 🚧 Known Issues

- Multi-camera GUI integration pending
- Calibration wizard in development
- Multi-person association across cameras needs work
- Network latency varies with WiFi quality

---

## 🔜 Roadmap

- [ ] GUI integration for multi-camera mode
- [ ] Interactive calibration wizard
- [ ] Real-time 3D visualization in GUI
- [ ] BVH/FBX export for animation software
- [ ] GPU acceleration (CUDA)
- [ ] Cloud recording with PostgreSQL
- [ ] 4+ camera support

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **MediaPipe** by Google for pose/face/hand detection
- **OpenCV** for computer vision utilities
- **ZeroMQ** for high-performance networking

---

## 📧 Contact

**Author**: Mrudula  
**GitHub**: [@Mrudula-itsjuzme](https://github.com/Mrudula-itsjuzme)  
**Repository**: [Motion-capture](https://github.com/Mrudula-itsjuzme/Motion-capture)

---

## 🌟 Star This Repo!

If you find this project useful, please give it a star ⭐ on GitHub!

---

**Built with ❤️ for the motion capture community**
