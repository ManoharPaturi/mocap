# Motion Capture System — Complete Project Documentation

**Version:** VS5 (Multi-Camera Enhanced Edition)  
**Author:** Mrudula  
**Last Updated:** 27 February 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Installation & Setup](#3-installation--setup)
4. [Configuration Reference](#4-configuration-reference)
5. [Entry Points & Launchers](#5-entry-points--launchers)
6. [Core Module Reference](#6-core-module-reference)
7. [Multi-Camera Pipeline](#7-multi-camera-pipeline)
    - [Clock Synchronization Requirements](#clock-synchronization-requirements)
8. [Coordinate System & Kinematics Conventions](#8-coordinate-system--kinematics-conventions)
9. [Frontend Web Dashboard](#9-frontend-web-dashboard)
10. [Desktop GUI (Tkinter)](#10-desktop-gui-tkinter)
11. [Database & Persistence](#11-database--persistence)
12. [Report Generation & Visualization](#12-report-generation--visualization)
13. [Testing & Validation](#13-testing--validation)
14. [Algorithms & Mathematical Foundations](#14-algorithms--mathematical-foundations)
    - [14.1 Joint Angle Computation](#141-joint-angle-computation)
    - [14.2 Distance & Limb Length Computation](#142-distance--limb-length-computation)
    - [14.3 Face Metrics](#143-face-metrics)
    - [14.4 Linear Velocity](#144-linear-velocity)
    - [14.5 Linear Acceleration](#145-linear-acceleration)
    - [14.6 Angular Velocity](#146-angular-velocity)
    - [14.7 Body Segment Vector Computation](#147-body-segment-vector-computation)
    - [14.8 1-Euro Filter (Adaptive Noise Smoothing)](#148-1-euro-filter-adaptive-noise-smoothing)
    - [14.9 Visibility Hard Gate](#149-visibility-hard-gate)
    - [14.10 Bone Length Constraint](#1410-bone-length-constraint-skeleton-physics)
    - [14.11 Outlier Rejection & Temporal Smoothing](#1411-outlier-rejection--temporal-smoothing)
    - [14.12 DLT Triangulation](#1412-direct-linear-transform-dlt-triangulation)
    - [14.13 Projection Matrix Construction](#1413-projection-matrix-construction)
    - [14.14 Point Undistortion](#1414-point-undistortion)
    - [14.15 Reprojection Error](#1415-reprojection-error)
    - [14.16 3D Point Confidence Score](#1416-3d-point-confidence-score)
    - [14.17 World Axis Convention Transform](#1417-world-axis-convention-transform)
    - [14.18 Monocular Depth Fallback](#1418-monocular-depth-fallback)
    - [14.19 Image Preprocessing Pipeline](#1419-image-preprocessing-pipeline)
    - [14.20 Intrinsic Camera Calibration](#1420-intrinsic-camera-calibration)
    - [14.21 Stereo Extrinsic Calibration](#1421-stereo-extrinsic-calibration)
    - [14A Complete Output Metrics Reference](#14a-complete-output-metrics-reference)
15. [Known Limitations](#15-known-limitations)
16. [Error Sources & Budget](#16-error-sources--budget)
17. [Units Consistency Reference](#17-units-consistency-reference)
18. [Performance & Optimization](#18-performance--optimization)
19. [Troubleshooting](#19-troubleshooting)
20. [File-by-File Reference](#20-file-by-file-reference)
21. [Glossary](#21-glossary)
22. [Development Roadmap](#22-development-roadmap)
23. [Calibration, Quality, and Metric Policy (Normative)](#23-calibration-quality-and-metric-policy-normative)

---

## 1. Project Overview

This is a real-time multi-person motion capture system that uses **MediaPipe** landmark detection across one or two cameras to produce:

- **2D pose, face, and hand landmarks** (33 pose, 468 face, 21 per hand)
- **Biomechanical metrics** (joint angles, limb lengths, velocities)
- **Stereo-triangulated 3D poses** (when two cameras are configured)
- **Full 3D kinematics** (position, velocity, acceleration, angular velocity)
- **Session recording** (SQLite/PostgreSQL) with CSV export
- **Automated analysis reports** (20+ plots: FFT, spectrograms, symmetry, correlation)
- **Interactive 3D visualization** (Plotly HTML dashboards)

### Key Capabilities

| Feature | Single Camera | Dual Camera |
|---------|---------------|-------------|
| People tracked simultaneously | Up to 5 | Up to 5 |
| Pose landmarks per person | 33 | 33 (3D triangulated) |
| Face landmarks | 468 | 468 |
| Hand landmarks | 21 per hand | 21 per hand |
| 3D accuracy | Approximate (relative-Z) | ±1–2 cm (metric-scale) |
| Depth quality | Monocular estimate | Triangulated (true meters) |
| Occlusion handling | Poor | Good (multi-view) |
| Frame rate | 25–30 FPS | 15–25 FPS |
| Kinematics engine | 2D angles only | Full 3D (v, a, theta, omega) |

### Technology Stack

| Component | Technology |
|-----------|------------|
| Detection | MediaPipe Pose/Face/Hand Landmarker (Task API) |
| Smoothing | 1-Euro Filter (adaptive low-pass) |
| Physics correction | Bone length constraints, joint limits, visibility gating |
| Triangulation | Direct Linear Transform (DLT) with optional CUDA |
| Networking | ZeroMQ (PUB/SUB) + msgpack serialization |
| Desktop GUI | Tkinter (macOS-compatible with PIL rendering) |
| Web frontend | React 19 + Vite + Tailwind CSS |
| Database | SQLite (default) or PostgreSQL |
| Visualization | Plotly (3D HTML), Matplotlib (live 3D), Seaborn (reports) |
| Acceleration | Metal (MPS on macOS), CUDA (NVIDIA), CPU fallback |

---

## 2. System Architecture

### Single-Camera Mode

```
Camera
  │
  ▼
MocapDetector (MediaPipe pose/face/hand)
  │
  ▼
PoseCorrector (1-Euro filter + bone constraints)
  │
  ▼
Calculations (angles, distances, kinematics)
  │
  ├──► Visualizer (landmark overlay on frame)
  ├──► MocapDB (SQLite/PostgreSQL recording)
  └──► VideoStreamer (MJPEG HTTP stream)
           │
           ▼
       React Frontend / Tkinter GUI
```

### Multi-Camera Mode (Two Laptops)

```
PC1 (Server)                          PC2 (Master)
┌──────────────────────┐             ┌───────────────────────────────────┐
│ Camera               │             │ Camera (local)                    │
│   ▼                  │   ZMQ       │   ▼                              │
│ MocapDetector        │  PUB/SUB    │ MocapDetector                    │
│   ▼                  │────────────►│   ▼                              │
│ CameraServer         │             │ MasterCoordinator                │
│  • Discovery (6000)  │             │   ├─ Timestamp sync (±20 ms)     │
│  • Data stream (6001)│             │   ├─ StereoCalibration           │
│                      │◄─Feedback───│   ├─ Triangulator (DLT)          │
└──────────────────────┘   (6002)    │   ├─ OneEuroFilter (3D smooth)   │
                                     │   ├─ KinematicsEngine            │
                                     │   └─► MocapDB                    │
                                     │        ├─ ReportGenerator        │
                                     │        └─ Visualizer3D           │
                                     │                                  │
                                     │ LiveVisualizer3D (Matplotlib)    │
                                     │ Dual Camera View (side-by-side)  │
                                     └───────────────────────────────────┘
```

### Data Flow (Multi-Camera Frame Lifecycle)

```
1. Capture     Camera frames on PC1 + PC2 (1280×720 @ 30 FPS)
       │
2. Detect      MediaPipe Pose Landmarker → 33 joints (x, y, confidence)
       │
3. Transmit    PC1 packages {timestamp, landmarks, JPEG} → ZMQ PUB → PC2 SUB
       │
4. Synchronize Match frames within ±20 ms window, drop unmatched
       │
5. Undistort   Pixel coords → undistort via camera intrinsics/distortion
       │
6. Triangulate Weighted DLT across 2+ views → 3D point (X, Y, Z)
       │
7. Validate    Per-point confidence + reprojection error check
       │
8. Transform   OpenCV camera frame → World frame (+Y up, +Z forward)
       │
9. Filter      1-Euro filter on 3D positions (not angles)
       │
10. Kinematics  Position → velocity → acceleration, joint angles → angular velocity
       │
11. Render     Dashboard (Tkinter/React) + live 3D skeleton (Matplotlib)
       │
12. Store      SQLite: per-frame {PC1, PC2, 3D, kinematics, confidence}
```

---

## 3. Installation & Setup

### Prerequisites

- **Python 3.8–3.10** (MediaPipe constraint)
- **Webcam** (built-in or USB, 1280×720 recommended)
- **4 GB RAM** minimum, 8 GB recommended
- For multi-camera: two laptops on the same Wi-Fi subnet

### Installation

```bash
# Clone
git clone https://github.com/Mrudula-itsjuzme/Motion-capture.git
cd Motion-capture

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# .\venv\Scripts\Activate.ps1   # Windows

# Dependencies
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`)

| Package | Purpose |
|---------|---------|
| `opencv-python >=4.8.0` | Camera capture, image processing, calibration |
| `mediapipe ==0.10.9` | Pose/face/hand landmark detection |
| `numpy >=1.21.0` | Linear algebra, array operations |
| `protobuf <4` | MediaPipe serialization compatibility |
| `fastapi` | Web API backend (video feed, metrics, recording) |
| `uvicorn` | ASGI server for FastAPI |
| `python-multipart` | Form data handling |
| `psycopg2-binary` | PostgreSQL adapter (optional) |
| `pandas` | Data analysis, CSV handling |
| `plotly` | Interactive 3D visualization (HTML dashboards) |
| `matplotlib` | Live 3D skeleton display + report images |
| `seaborn` | Statistical plot styling for reports |
| `scipy` | FFT, spectrograms for frequency analysis |
| `pyzmq >=25.0.0` | ZeroMQ networking (multi-camera) |
| `msgpack >=1.0.0` | Binary serialization (network packets) |

### Quick Start

**Single camera:**
```bash
python main_gui.py
```

**Multi-camera (2 laptops):**
```bash
# Laptop 1 (Server) — captures + broadcasts
python launch_multi_camera.py --mode server

# Laptop 2 (Master) — captures + receives + triangulates
python launch_multi_camera.py --mode master --remote-ip <SERVER_IP>
```

---

## 4. Configuration Reference

All configuration lives in `config.py`. Parameters are grouped by subsystem.

### Camera Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_ID` | `0` | OpenCV device index |
| `NETWORK_CAMERA_ID` | `'cam_0'` | Network identifier string |
| `FRAME_WIDTH` | `1280` | Capture width (pixels) |
| `FRAME_HEIGHT` | `720` | Capture height (pixels) |
| `FPS` | `30` | Target capture frame rate |

### Detection Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_DETECTION_CONFIDENCE` | `0.5` | MediaPipe detection threshold |
| `MIN_TRACKING_CONFIDENCE` | `0.5` | MediaPipe tracking threshold |
| `POSE_MODEL_COMPLEXITY` | `'FULL'` | Model: `LITE` (fast), `FULL` (balanced), `HEAVY` (accurate) |
| `NUM_POSES` | `1` | Max simultaneous people |
| `NUM_FACES` | `1` | Max simultaneous faces |
| `NUM_HANDS` | `2` | Max simultaneous hands |

### Hardware Acceleration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INFERENCE_BACKEND` | `'mps'` | `'mps'` (macOS Metal), `'gpu'`, `'cpu'`, `'auto'` |
| `PREFER_GPU_DELEGATE` | `True` | Legacy toggle for older code paths |
| `DEVICE` | Auto-detected | `'mps'`, `'cuda'`, or `'cpu'` — set automatically via PyTorch |

### Image Preprocessing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GAMMA_DEFAULT` | `1.0` | Gamma correction (1.0 = disabled) |
| `GAMMA_MIN` / `GAMMA_MAX` | `1.0` / `1.4` | Slider range |
| `FACE_EXPOSURE_TARGET` | `140.0` | Target face brightness (0–255) |
| `CLAHE_CLIP_LIMIT` | `2.0` | Contrast enhancement strength |
| `ENABLE_ROI_CROPPING` | `True` | Crop to person bounding box |
| `ROI_EXPANSION_FACTOR` | `0.25` | Bbox expansion margin |
| `ROI_TARGET_SIZE` | `640` | Resize ROI for inference |

### Physics Engine

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VISIBILITY_HARD_GATE` | `0.5` | Below this, landmark is frozen |
| `FILTER_MIN_CUTOFF` | `1.0` | 1-Euro: base smoothing frequency (Hz) |
| `FILTER_BETA` | `0.005` | 1-Euro: speed responsiveness |
| `CALIBRATION_FRAMES` | `30` | Frames to learn bone lengths |
| `BONE_LENGTH_STRICTNESS_HIGH_VIS` | `0.85` | Bone constraint strength (high visibility) |
| `BONE_LENGTH_STRICTNESS_LOW_VIS` | `0.99` | Bone constraint strength (low visibility) |

### Metric Calculations

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ANGLE_OUTLIER_BASE_THRESHOLD` | `50.0` | Max angle change per frame (degrees) |
| `MAX_LINEAR_VELOCITY` | `6.0` | Velocity sanity cap (m/s) |
| `SMOOTHING_ALPHA_DEFAULT` | `0.5` | EMA smoothing for body metrics |
| `SMOOTHING_ALPHA_FACE` | `0.3` | Stronger smoothing for face metrics |

### Multi-Camera Network

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLE_MULTI_CAMERA` | `False` | Enable multi-camera subsystem |
| `CAMERA_ROLE` | `'single'` | `'single'`, `'server'`, or `'master'` |
| `MULTI_CAMERA_MODE` | `'single'` | Same as above (GUI integration) |
| `REMOTE_CAMERA_IP` | `None` | IP of the other laptop |
| `DISCOVERY_PORT` | `6000` | ZMQ discovery broadcast port |
| `DATA_PORT` | `6001` | ZMQ data stream port |
| `FEEDBACK_PORT` | `6002` | Quality feedback channel |
| `NETWORK_JPEG_QUALITY` | `50` | JPEG quality for network (balance quality vs bandwidth) |
| `NETWORK_STREAM_WIDTH` | `640` | Transmit resolution width |
| `NETWORK_STREAM_HEIGHT` | `360` | Transmit resolution height |
| `NETWORK_FRAMERATE_LIMIT` | `30` | Cap network transmission to this FPS |
| `SYNC_TIME_THRESHOLD_MS` | `100.0` | Frame match window (±ms); 100 ms tolerates WiFi jitter |
| `FRAME_BUFFER_SIZE` | `10` | Frames kept per camera for sync matching |
| `STALE_FRAME_TIMEOUT_MS` | `2000` | Evict frames older than 2 s vs the freshest camera |
| `COMPRESS_NETWORK_DATA` | `True` | Use msgpack compression |

### Stereo 3D Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CALIBRATION_FILE` | `'calibration.json'` | Stereo calibration path |
| `TRIANGULATION_MIN_VIEWS` | `2` | Min cameras for triangulation |
| `REPROJECTION_ERROR_THRESHOLD` | `15.0` | Max reprojection error (px) |
| `CONFIDENCE_WEIGHT_VISIBILITY` | `0.6` | Visibility weight in confidence |
| `CONFIDENCE_WEIGHT_REPROJ` | `0.4` | Reprojection weight in confidence |
| `STEREO_POINT_MIN_INPUT_CONFIDENCE` | `0.5` | Per-camera landmark gate |
| `KINEMATICS_MIN_POINT_CONFIDENCE` | `0.5` | 3D point gate for kinematics |
| `ENABLE_3D_ONE_EURO_FILTER` | `True` | Filter 3D positions |
| `MONOCULAR_SUBJECT_DISTANCE_M` | `2.5` | Fallback depth for monocular |

### Presets

```python
PRESETS = {
    'indoor':     {'gamma': 1.2, 'face_exposure': True,  'clahe_clip': 2.0},
    'outdoor':    {'gamma': 1.0, 'face_exposure': False, 'clahe_clip': 2.0},
    'low_light':  {'gamma': 1.3, 'face_exposure': True,  'clahe_clip': 3.0},
    'high_speed': {'gamma': 1.0, 'face_exposure': False, 'clahe_clip': 2.0,
                   'enable_face': False, 'enable_hand': False, 'filter_beta': 0.01}
}
```

---

## 5. Entry Points & Launchers

### `main_gui.py` — Desktop Application

The primary entry point for single-camera mode. Instantiates `MocapGUI` which creates:
- A Tkinter dashboard (500×700 dark-themed window)
- Camera capture thread
- Detection, correction, and metric computation pipeline
- Optional CameraServer or MasterCoordinator (based on config)

```bash
python main_gui.py
```

### `launch_multi_camera.py` — Multi-Camera Launcher

CLI wrapper that sets `config.MULTI_CAMERA_MODE` and `config.REMOTE_CAMERA_IP` before importing `MocapGUI`.
Suppresses noisy TensorFlow Lite / MediaPipe / abseil log messages (`GL version`, `Created TensorFlow Lite delegate`, `FaceBlendshapesGraph acceleration`) via environment variables set before any library imports.

```bash
# Single camera (default)
python launch_multi_camera.py --mode single

# Server (broadcasts frames)
python launch_multi_camera.py --mode server

# Master (receives + triangulates)
python launch_multi_camera.py --mode master --remote-ip 10.137.227.228
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--mode` | No (default: `single`) | `single`, `server`, or `master` |
| `--remote-ip` | Yes (master mode) | Server laptop's IP address |

### Helper Scripts (`scripts/`)

| Script | Platform | Purpose |
|--------|----------|---------|
| `run_gui.bat` | Windows | Activates venv + runs `main_gui.py` |
| `run_master.bat` | Windows | Activates venv + runs master mode |
| `allow_firewall.ps1` | Windows | Opens ports 5000–5001 for ZMQ |

---

## 6. Core Module Reference

### `src/camera.py` — Camera Capture

Thread-safe camera capture using OpenCV.

| Class | `Camera` |
|-------|----------|
| **Purpose** | Continuously reads frames in a background thread |
| **Init** | Opens `cv2.VideoCapture(camera_id)`, configures resolution and FPS |
| **Thread** | Daemon thread polls at ~200 Hz, stores latest frame |
| **API** | `read()` → latest BGR frame, `release()` → cleanup, `is_opened()` → status |

### `src/detector.py` — MediaPipe Detection

Orchestrates MediaPipe Pose, Face, and Hand Landmarkers with preprocessing.

| Class | `MocapDetector` |
|-------|-----------------|
| **Purpose** | Full detection pipeline: preprocessing → inference → coordinate mapping |
| **Models** | Pose (LITE/FULL/HEAVY), Face (468 landmarks), Hand (21 landmarks × 2) |
| **GPU** | macOS Metal delegate, CUDA, or CPU auto-selection |
| **Preprocessing** | ROI cropping → resize → gamma correction → face exposure → CLAHE |
| **ROI** | Tracks person bbox across frames, crops + upscales for higher resolution |
| **Output** | `{'pose': [...], 'face': [...], 'hand': [...], 'roi': (x,y,w,h)}` |

Key methods:
- `process(frame)` — main pipeline, returns detection dict
- `reload(model_type)` — hot-swap between LITE/FULL/HEAVY
- `set_imaging_params(...)` — runtime update of gamma, toggles

### `src/pose_corrector.py` — Physics Engine

Post-processes MediaPipe landmarks with physics-based constraints.

| Class | `PoseCorrector` |
|-------|-----------------|
| **Purpose** | Smooth jitter + enforce skeletal realism |
| **Pipeline** | 1-Euro filter → calibration → bone length constraints |
| **Calibration** | Learns reference bone lengths over 30 frames, then enforces |
| **Gating** | Landmarks below `VISIBILITY_HARD_GATE` are frozen to previous position |
| **Bones** | 8 limb bones defined as parent→child (upper arms, forearms, thighs, shins) |

### `src/calculations.py` — Biomechanical Metrics

Static utility class for computing all biomechanical measurements.

| Class | `Calculations` |
|-------|----------------|
| **Angles** | Elbow (L/R), shoulder (L/R), hip (L/R), knee (L/R) — all 0°–180° |
| **Lengths** | Upper arm, forearm, thigh, shin, shoulder width, hip width |
| **Face** | Mouth openness, smile ratio, eye openness (normalized by IPD) |
| **Kinematics** | Angular velocity (deg/s), linear velocity (m/s) with 6 m/s cap |
| **Filtering** | Velocity-dependent outlier rejection + EMA temporal smoothing |

Key constant: `POSE_IDX` maps joint names → MediaPipe landmark indices.

### `src/one_euro_filter.py` — Adaptive Smoothing

Implementation of the [1€ Filter](https://cristal.univ-lille.fr/~casiez/1euro/).

| Class | `OneEuroFilter` |
|-------|-----------------|
| **Purpose** | Noise reduction with speed-adaptive cutoff |
| **Parameters** | `min_cutoff` (base smoothing), `beta` (speed response), `d_cutoff` (derivative smoothing) |
| **Formula** | Adaptive cutoff: $f_c = f_{min} + \beta \cdot |v|$, then EMA with $\alpha = \frac{2\pi f_c \Delta t}{2\pi f_c \Delta t + 1}$ |
| **Usage** | 2D landmarks (PoseCorrector) and 3D positions (MasterCoordinator) |

### `src/visualizer.py` — 2D Rendering

Draws landmark overlays on video frames.

| Class | `Visualizer` |
|-------|-------------|
| **Draws** | Pose skeleton connections, face tessellation mesh, hand landmarks |
| **FPS** | Rolling 30-frame average displayed on frame |
| **Uses** | MediaPipe drawing utilities (`mp.solutions.drawing_utils`) |

### `src/streamer.py` — Video Streamer

Orchestrates the single-camera pipeline in a background thread.

| Class | `VideoStreamer` |
|-------|----------------|
| **Pipeline** | Camera → Detector → PoseCorrector → Calculations → Visualizer → JPEG encode |
| **Output** | `generate()` yields MJPEG multipart frames for HTTP streaming |
| **State** | Maintains `prev_lm`, `prev_metrics`, `prev_time` for temporal computation |
| **Recording** | Calls `MocapDB.save_frame()` each frame when recording |

### `src/camera_server.py` — Network Broadcaster

Runs on remote camera laptops. Broadcasts detection results over the network.

| Class | `CameraServer` |
|-------|----------------|
| **Protocol** | ZMQ PUB/SUB on two ports (discovery + data) |
| **Discovery** | Broadcasts `{camera_id, ip, port}` every 2 seconds |
| **Data** | Sends `{timestamp, landmarks, JPEG frame}` via msgpack |
| **Optimization** | `CONFLATE` socket option (latest-frame-only), JPEG resized to 640×360 |
| **Feedback** | SUB socket for master's quality hints |
| **Clock Sync** | REP socket on port 6003 (`_clock_sync_handler`). After each `recv()`, a reply is **always** sent — even on parse failure — via nested try/except to maintain ZMQ REP strict alternation (recv→send) and prevent socket state corruption |

### `src/master_coordinator.py` — Central Hub

Receives, synchronizes, triangulates, and computes kinematics for multi-camera data.

| Class | `MasterCoordinator` |
|-------|---------------------|
| **Receives** | ZMQ SUB sockets connected to all camera servers |
| **Syncs** | Timestamp-based matching within ±100 ms (configurable) with stale-frame eviction at 2 s |
| **Triangulates** | Weighted DLT via `Triangulator` |
| **Fallback** | Monocular back-projection for occluded joints |
| **Filtering** | Per-joint 1-Euro filter on 3D X/Y/Z |
| **Kinematics** | Delegates to `KinematicsEngine` |
| **Output** | `{pose_3d, kinematics_3d, low_reliability_landmarks, timestamp_ns}` |

### `src/frame_synchronizer.py` — Timestamp Matching

Alternative/complementary frame synchronizer.

| Class | `FrameSynchronizer` |
|-------|---------------------|
| **Strict mode** | `get_synced_frames()` — all cameras must match |
| **Flexible mode** | `get_synced_frames_flexible(min_cameras)` — partial matches allowed |
| **Buffer** | Per-camera deque with configurable depth |
| **Stats** | Sync rate, failure count, per-camera buffer info |

### `src/stereo_calibration.py` — Camera Calibration

OpenCV-based intrinsic and stereo extrinsic calibration.

| Class | `StereoCalibration` |
|-------|---------------------|
| **Intrinsic** | Checkerboard detection → `cv2.calibrateCamera()` |
| **Stereo** | Common checkerboard frames → `cv2.stereoCalibrate()` (fixed intrinsics) |
| **Projection** | Computes $P = K[R|t]$ (3×4 projection matrix) per camera |
| **Default** | Creates approximate calibration: 1m baseline, focal ≈ image width |
| **Storage** | JSON serialization (matrices converted to lists) |

### `src/triangulation.py` — 3D Reconstruction

Direct Linear Transform (DLT) triangulation with optional GPU acceleration.

| Class | `Triangulator` |
|-------|----------------|
| **Algorithm** | DLT: builds $A$ matrix (2N×4), solves via SVD |
| **GPU** | PyTorch CUDA path for SVD + reprojection |
| **Confidence** | $\text{conf} = (w_r \cdot e^{-err/5} + w_v \cdot \bar{v}) \cdot \min(1, \frac{n}{4})$ |
| **Gating** | Points with reprojection error > threshold are rejected |
| **Output** | List of `Landmark3D` objects with position, confidence, visibility |

### `src/kinematics_engine.py` — 3D Kinematics

Stateful engine for computing derivatives and angles from 3D joint positions.

| Class | `KinematicsEngine` |
|-------|---------------------|
| **Velocity** | $(P_t - P_{t-1}) / \Delta t$, clamped to `MAX_LINEAR_VELOCITY` |
| **Acceleration** | $(V_t - V_{t-1}) / \Delta t$ |
| **Angles** | 8 canonical joints (elbows, knees, shoulders, hips), 0°–180° |
| **Angular velocity** | $(\theta_t - \theta_{t-1}) / \Delta t$ |
| **Spine vector** | Mid-shoulder − mid-hip (trunk axis reference) |
| **Export** | `export_frame_data()` flattens to `{joint_0_x, angle_elbow_r_deg, ...}` dict |

---

## 7. Multi-Camera Pipeline

### Network Topology

```
Server (PC1)                                Master (PC2)
─────────────                               ─────────────
ZMQ PUB :6000  ──── Discovery ────────►  ZMQ SUB
ZMQ PUB :6001  ──── Frame Data ──────►  ZMQ SUB
ZMQ SUB        ◄─── Quality Feedback ── ZMQ PUB :6002
```

All sockets use TCP. The `CONFLATE` option ensures only the latest message is buffered (prevents queue buildup on slow networks).

### Frame Packet Format

Each network packet (msgpack-encoded) contains:

```python
{
    'camera_id': 'cam_0',           # Server's network ID
    'frame_number': 42,             # Sequential frame counter
    'timestamp': 1740700000000000,  # Epoch nanoseconds
    'pose': [[{x, y, conf}, ...×33]],  # Per-person 2D landmarks
    'face': [...],                  # Face landmarks (optional)
    'hand': [...],                  # Hand landmarks (optional)
    'stereo_landmarks': [{x, y, conf}, ...×33],  # Compact stereo format
    'frame_jpeg': b'...'           # JPEG-encoded frame (640×360)
}
```

### Synchronization Algorithm

1. Each camera tags frames with `time.time_ns()` (wall clock)
2. Master buffers incoming frames per camera ID (up to `FRAME_BUFFER_SIZE` = 10 frames each)
3. **Stale-frame eviction:** before each sync attempt, `get_synchronized_batch()` compares each camera's newest frame against the global newest timestamp. If any camera's newest frame is older than `STALE_FRAME_TIMEOUT_MS` (2 000 ms), its entire buffer is cleared with a one-time warning. This prevents a disconnected or stalled camera from permanently blocking sync.
4. The oldest frame from one camera is used as the reference timestamp
5. Searches all other cameras for frames within ±`SYNC_TIME_THRESHOLD_MS` (100 ms — tuned for WiFi jitter)
6. **Strict mode:** all cameras must have a match, otherwise the reference frame is discarded
7. Matched frames are consumed from buffers; unmatched accumulate (and are eventually evicted by step 3)

### Clock Synchronization Requirements

> **Status: IMPLEMENTED.** Cristian's Algorithm-based clock offset estimation was added to `camera_server.py` and `master_coordinator.py`. Controlled by `ENABLE_CLOCK_SYNC` in `config.py`.

#### Why This Matters

DLT triangulation assumes that matched 2D observations correspond to the **same physical instant**. If Camera A captures at $t_0$ and Camera B captures at $t_0 + \Delta$, the subject may have moved between the two frames. For fast motion (e.g., a hand swing at 3 m/s), even small offsets cause significant error:

| Clock Offset ($\Delta$) | Position Error at 3 m/s | Effect |
|------------------------|------------------------|--------|
| 5 ms | 1.5 cm | Negligible |
| 10 ms | 3.0 cm | Visible jitter in 3D |
| 20 ms | 6.0 cm | Ghost motion artifacts |
| 50 ms | 15.0 cm | Unusable triangulation |

The ±20 ms `SYNC_TIME_THRESHOLD_MS` window ensures frames are *approximately* concurrent, but without offset correction it cannot detect or compensate a *systematic* clock skew between machines.

#### Implementation: Cristian's Algorithm

The system implements automatic clock offset estimation using a ping-pong handshake on ZMQ REQ/REP sockets (port `CLOCK_SYNC_PORT` = 6003):

```
Master (REQ)                     Server (REP)
  │                                │
  ├── PING {master_time_ns} ────► │
  │          t0 = time_ns()        │── server_time = time_ns()
  │ ◄── PONG {server_time_ns} ────┤
  │          t1 = time_ns()        │
  │                                │
  RTT = t1 - t0
  offset = server_time - (t0 + RTT/2)
```

**Robustness measures:**

1. **Multiple samples:** `CLOCK_SYNC_SAMPLES` (10) pings are collected per camera
2. **Fresh socket per sample:** Each ping uses a new ZMQ REQ socket to avoid state corruption after `recv` timeouts (REQ enforces strict send→recv→send→recv ordering; after a timeout the socket is stuck)
3. **RTT outlier rejection:** Samples with RTT > `CLOCK_SYNC_RTT_OUTLIER_FACTOR` (2.0) × min RTT are discarded (removes WiFi jitter spikes)
4. **Median aggregation:** The median of remaining offsets is used (robust to asymmetric latency)
5. **Periodic re-sync:** Every `CLOCK_SYNC_INTERVAL_SEC` (300s) to compensate for clock drift
6. **Graceful failure:** If clock sync fails entirely (e.g., remote doesn't support it), the system logs a warning and continues with uncorrected timestamps instead of crashing

**Where it's applied:** In `MasterCoordinator._process_frame_data()`, before the frame enters the sync buffer:

```python
if ENABLE_CLOCK_SYNC and camera_id in self.clock_offsets:
    frame_data.timestamp = timestamp + self.clock_offsets[camera_id]
```

The `FrameSynchronizer` itself needs no changes — it just sees corrected timestamps.

#### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLE_CLOCK_SYNC` | `True` | Enable automatic clock offset estimation |
| `CLOCK_SYNC_PORT` | `6003` | ZMQ REQ/REP port for ping-pong |
| `CLOCK_SYNC_SAMPLES` | `10` | Number of ping samples per sync |
| `CLOCK_SYNC_INTERVAL_SEC` | `300` | Re-sync interval (0 = startup only) |
| `CLOCK_SYNC_RTT_OUTLIER_FACTOR` | `2.0` | RTT outlier rejection threshold |

#### Diagnostics

`MasterCoordinator._check_sync_quality(synced_frames)` analyzes timestamp spread in each synchronized batch:
- Warns if spread exceeds 10 ms
- Reports per-camera offsets from the batch mean
- Returns spread and offset data for logging

#### Additional Mitigations

**NTP Synchronization (complementary)**

For best results, also sync both laptops to the same NTP server before each session:

```bash
# macOS
sudo sntp -sS time.apple.com

# Windows
w32tm /resync /force

# Linux
sudo ntpdate -s pool.ntp.org
```

**Hardware Sync (research-grade)**

For high-speed motion (>5 m/s), consider a shared trigger signal (GPIO pulse, audio click, or flash) to synchronize camera shutters directly.

---

### 3D Fusion Pipeline (per synced batch)

```
For each of 33 pose landmarks:
  1. Collect 2D observations from all cameras
  2. Check per-camera confidence ≥ STEREO_POINT_MIN_INPUT_CONFIDENCE
  3. If ≥2 views pass: Triangulate (weighted DLT)
  4. If 1 view passes: Monocular fallback (reduced confidence)
  5. If 0 views pass: Skip landmark
  6. Apply world axis convention transform (flip Y for +Y up)
  7. Apply 1-Euro filter to smoothed 3D position (if enabled)
  8. Compute kinematics (velocity, acceleration, angles, angular velocity)
  9. Record low-reliability landmark IDs
```

### Calibration

| Approach | When |
|----------|------|
| **Default** | No calibration file — assumes 1m horizontal baseline, identity rotation, focal length ≈ image width |
| **Manual** | `StereoCalibration.calibrate_stereo()` with checkerboard images, saves to `calibration.json` |
| **Auto-load** | If `calibration.json` exists and `AUTO_LOAD_CALIBRATION = True`, loaded at startup |

---

## 8. Coordinate System & Kinematics Conventions

### Global World Frame (W)

The system uses a locked **right-handed world frame**:

| Axis | Direction |
|------|-----------|
| +X | Right |
| +Y | Up |
| +Z | Forward (away from front camera) |

**Origin:** Camera A (front camera) optical center.

### Camera Frame (OpenCV)

Per-camera coordinates follow OpenCV convention:

| Axis | Direction |
|------|-----------|
| +X | Right in image |
| +Y | Down in image |
| +Z | Forward from lens |

The `WORLD_AXIS_TRANSFORM` config flips Y to convert camera→world: `{flip_x: false, flip_y: true, flip_z: false}`.

### Body Segment Conventions

Canonical segment vectors (distal − proximal):

| Segment | Vector |
|---------|--------|
| Upper Arm (R) | Elbow − Shoulder |
| Forearm (R) | Wrist − Elbow |
| Trunk Axis | MidShoulder − MidHip |
| Pelvic Axis | RightHip − LeftHip |

### Angle Convention

- **Unit:** Degrees
- **Range:** 0° to 180° (unsigned)
- **Formula:** $\theta = \arccos\left(\frac{v_1 \cdot v_2}{|v_1| \cdot |v_2|}\right)$

**Interpretation:**
| Angle | Meaning |
|-------|---------|
| 180° | Full extension |
| 90° | Right angle (e.g., elbow at 90°) |
| < 90° | Deep flexion |

**Example (Right Elbow):**
- $v_1$ = Shoulder − Elbow
- $v_2$ = Wrist − Elbow
- $\theta$ = angle between $v_1$ and $v_2$

### Kinematics Definitions

| Quantity | Formula | Units |
|----------|---------|-------|
| Position | $(X, Y, Z)$ from triangulation | meters |
| Linear velocity | $(P_t - P_{t-1}) / \Delta t$ | m/s |
| Linear acceleration | $(V_t - V_{t-1}) / \Delta t$ | m/s² |
| Joint angle | $\arccos$ of normalized dot product | degrees |
| Angular velocity | $(\theta_t - \theta_{t-1}) / \Delta t$ | deg/s |

All derivatives are computed from **filtered** 3D positions (post 1-Euro). Angles are computed from filtered positions but are **not** themselves filtered.

---

## 9. Frontend Web Dashboard

### Stack

- **React 19.2** with functional components + hooks
- **Vite 7.2** (dev server + bundler)
- **Tailwind CSS 4.1** with custom theme
- **lucide-react** icons

### Running the Frontend

```bash
cd frontend
npm install
npm run dev    # Dev server on localhost:5173
npm run build  # Production build
```

### API Endpoints (Backend at `http://localhost:8000`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | System status + recording state (polled every 2s) |
| `/metrics` | GET | Live biometric angles (polled every 200ms) |
| `/video_feed` | GET | MJPEG video stream |
| `/record/start` | POST | Start recording session |
| `/record/stop` | POST | Stop recording session |
| `/record/export` | GET | Download CSV |
| `/report/generate` | POST | Generate analysis report |
| `/viz/3d` | POST | Launch 3D visualization |

### User-Facing Features

1. Live MJPEG video feed with recording border (red pulse when active)
2. Connection status indicator (green = online, red = disconnected)
3. Recording badge (animated pulse in header)
4. Session stats (resolution, DB status)
5. Live biometrics panel (L/R Elbow, L/R Knee, L/R Shoulder angles)
6. Start/Stop Recording button
7. Download CSV button
8. Visualize 3D button
9. Generate Report button

### Theme

Custom coffee-inspired dark palette:
| Name | Hex | Usage |
|------|-----|-------|
| Espresso | `#4B3621` | Primary dark |
| Latte | `#D6C0B3` | Light accent |
| Cream | `#F5F5DC` | Background |
| Mocha | `#967969` | Mid-tone |

---

## 10. Desktop GUI (Tkinter)

### Layout

```
Root Window (500×700, dark theme #0f0f1e)
└── Scrollable Container
    ├── Header
    │   ├── "MoCap Live Dashboard" title
    │   └── Model Combobox (LITE / FULL / HEAVY)
    │
    ├── Status Bar
    │   ├── Status indicator (READY / RECORDING)
    │   └── FPS counter
    │
    ├── Treeview Data Table
    │   └── 9 columns: Time, 6 angles, 2 velocities (8 visible rows)
    │
    ├── Biometrics Panel
    │   └── 10 labels: 6 joint angles + 4 normalized limb lengths
    │
    ├── Toggle Frame
    │   ├── Mirror Camera checkbox
    │   └── Show Markers checkbox
    │
    ├── Action Buttons
    │   ├── ▶ Start / ⏹ Stop Capture (green/red)
    │   ├── 📥 Download Dataset
    │   ├── 📊 Visualize Session
    │   └── 🔮 Live 3D (master mode only, orange)
    │
    └── Imaging Controls
        ├── Gamma slider
        ├── Face Exposure checkbox
        ├── ROI Cropping checkbox
        ├── Face Detection checkbox
        └── Hand Detection checkbox
```

### Key Behaviors

- **Camera display:** Rendered in a separate `Toplevel` window using PIL (macOS-compatible, avoids `cv2.imshow` threading issues)
- **Frame coalescing:** Only the latest frame/metrics are displayed (prevents UI lag)
- **Multi-camera display:** Side-by-side composite frame (local left, remote right)
- **Thread safety:** GUI updates scheduled via `root.after()`, metrics and frames use single-slot queues
- **Network send thread:** Dedicated `_send_worker` drains a latest-only queue (prevents per-frame thread spawning)
- **Remote decode thread:** `_remote_decode_worker` JPEG-decodes remote frames off the main loop

#### Camera Thread Polling

The video loop uses `root.after(1)` scheduling (~200 Hz polling rate) rather than a tight `while True` busy-wait. This yields CPU time to the OS scheduler between iterations, keeping CPU usage at 40–60% instead of a full core. The actual frame rate is bounded by camera capture speed (typically 30 fps), so the 200 Hz poll simply ensures frames are consumed promptly without spinlock overhead.

#### Queue Backpressure Design

All inter-thread queues in the GUI use a **latest-only (depth-1)** strategy: the producer overwrites the single slot, and the consumer reads whatever is current. This design:

- **Prevents latency accumulation** — if the consumer is slow, it skips stale frames rather than processing a growing backlog
- **Bounds memory** — queue depth is O(1) regardless of producer rate
- **Trades completeness for freshness** — acceptable because the GUI only needs the latest state for display

This applies to: frame display queue, metrics display queue, network send queue, and remote frame decode queue.

### Recording Flow

```
1. User clicks "▶ Start Capture"
   → db.start_recording() → new session UUID → status = RECORDING (red)

2. Per frame (video_loop):
   → Single/Server: db.save_frame(results)
   → Master: db.save_synced_frame(timestamp, pc1, pc2, pose_3d)
   → Every 5th frame: update data table + biometrics panel

3. User clicks "⏹ Stop Capture"
   → db.stop_recording() → confirmation dialog with session ID

4. User clicks "📥 Download Dataset"
   → db.export_latest_session_csv() → saves mocap_<timestamp>.csv

5. User clicks "📊 Visualize Session"
   → viz_3d.plot_latest_session() → opens HTML dashboard in browser
   → reporter.generate_report() → saves 20+ PNG plots to results/
```

---

## 11. Database & Persistence

### Schema

**Sessions table** (master table):

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,        -- UUID
    start_time TEXT,            -- ISO timestamp
    end_time TEXT,              -- ISO timestamp (set on stop)
    num_frames INTEGER DEFAULT 0,
    table_name TEXT             -- Dynamic per-session table name
)
```

**Per-session table** (created dynamically as `session_YYYYMMDD_HHMMSS`):

```sql
CREATE TABLE session_20260227_143022 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    -- PC1 data
    pose_data TEXT,             -- JSON: [[{x,y,z,v}, ...×33], ...]
    face_data TEXT,             -- JSON: face landmarks
    hand_data TEXT,             -- JSON: hand landmarks
    derived_data TEXT,          -- JSON: angles, lengths, velocities
    -- PC2 data (multi-camera only)
    pc2_pose_data TEXT,
    pc2_face_data TEXT,
    pc2_hand_data TEXT,
    pc2_derived_data TEXT,
    -- 3D reconstructed data (multi-camera only)
    pose_3d_data TEXT,          -- JSON: [{id, x, y, z, confidence}, ...×33]
    combined_derived_data TEXT,  -- JSON: fused metrics
    -- Kinematics (multi-camera only)
    kinematics_flat_data TEXT,  -- JSON: {joint_0_x, joint_0_vx, angle_elbow_r_deg, ...}
    confidence_data TEXT        -- JSON: per-joint confidence + low-reliability flags
)
```

### Recording Engine

- Background writer thread (`_worker_loop`) batches up to 50 frames per DB commit
- Thread-safe queue between processing loop and DB writer
- Automatic schema migration adds `table_name` column if missing

> **Maintenance note — SQLite fragmentation.** Because each recording session creates a new table dynamically, long-term use with many short sessions can fragment the SQLite file. After deleting old sessions or archiving data, run:
>
> ```sql
> VACUUM;
> ```
>
> This rebuilds the database file, reclaims unused space, and restores sequential page layout for faster reads. For databases exceeding ~500 MB, consider periodic VACUUM as part of a maintenance routine.

### Export

`export_latest_session_csv()` produces a flat CSV with columns:

```
Timestamp, Source, Person, Key, Value_X, Value_Y, Value_Z, Conf
```

Where `Source` is one of: `PC1`, `PC2`, `3D`, `KIN`.

---

## 12. Report Generation & Visualization

### Interactive 3D Dashboard (`Visualizer3D`)

Generated from latest DB session as a self-contained HTML file:

- 3D animated skeleton replay with frame slider
- Arm angle time series (elbow L/R)
- Leg angle time series (knee L/R)
- Mouth openness plot
- Smile ratio plot

Opens automatically in the default browser.

### Static Report Images (`ReportGenerator`)

Generates 20+ analysis plots saved to `results/`:

| Analysis | Plots Generated |
|----------|----------------|
| Per-body-part (7 nodes) | XYZ time series, velocity profile, FFT spectrum, spectrogram |
| Joint angles | Angle time series + box-plot distribution |
| Kinematics | Angular velocity heatmap |
| Symmetry | L vs R scatter (elbows, knees) with symmetry line |
| Correlation | Full metric correlation matrix |
| Hand trajectory | 2D hand path (L vs R) |
| Face trajectory | Chin vertical movement (talking detection) |

### Live 3D (Matplotlib)

Real-time 3D skeleton window (master mode only):
- Scatter + line plot of 33 joints + 12 bone connections
- Y/Z axis swap for conventional viewing (+Y up)
- Fixed axis limits [-1, 1] meters
- Updated per synchronized frame

---

## 13. Testing & Validation

### Network Tests

| Test | File | Purpose |
|------|------|---------|
| TCP port scan | `test_connection.py` | Check ports 5000/5001 reachable on server IP |
| Port scan alt | `test_ports.py` | Same, different target IP |
| Remote ports | `test_remote_ports.py` | Parameterized port reachability |
| ZMQ receive | `test_receive_debug.py` | Direct ZMQ SUB → msgpack decode for 10s |

### Integration Tests

| Test | File | Purpose |
|------|------|---------|
| Multi-camera pipeline | `test_multicam.py` | Three sub-tests: server broadcast, server+detect, master receive |
| Mock sender | `test_sender.py` | 300 mock frames with synthetic landmarks |
| Continuous sender | `test_sender_continuous.py` | Indefinite mock broadcast stress test |

### Validation Tests

| Test | File | What It Validates | Pass Criteria |
|------|------|-------------------|---------------|
| Static jitter | `test_static_jitter.py` | 3D position stability | Per-joint std ≤ 0.005 m |
| Known angle | `test_known_angle.py` | Angle computation accuracy | Error ≤ 5° |
| Known distance | `test_known_distance.py` | Distance measurement accuracy | Error ≤ 0.05 m |

All validation tests output structured results to `tests/results/` as both JSON (machine-readable) and Markdown (human-readable).

### Running Tests

```bash
# Network connectivity
python tests/test_connection.py

# Multi-camera integration
python tests/test_multicam.py --mode server
python tests/test_multicam.py --mode master --ips 10.137.227.228

# Validation
python tests/test_static_jitter.py
python tests/test_known_angle.py
python tests/test_known_distance.py
```

---

## 14. Algorithms & Mathematical Foundations

This section defines every computation in the system, the exact formulas used, which source file implements them, and the output metrics produced.

---

### 14.1 Joint Angle Computation

**Source:** `src/calculations.py` → `Calculations.calculate_angle()`, `src/kinematics_engine.py` → `KinematicsEngine.compute_joint_angle()`

**Method:** Unsigned angle at a vertex joint $B$ between two adjacent segments $BA$ and $BC$.

Given three 3D joint positions $A$, $B$, $C$:

$$\vec{v_1} = A - B \qquad \vec{v_2} = C - B$$

$$\theta = \arccos\left(\frac{\vec{v_1} \cdot \vec{v_2}}{|\vec{v_1}| \cdot |\vec{v_2}|}\right)$$

The cosine is clamped to $[-1, 1]$ before `acos` to prevent floating-point domain errors. Result is converted to degrees and clamped to $[0°, 180°]$.

**Interpretation:**
- $180°$ = fully extended (straight limb)
- $90°$ = right angle
- $0°$ = fully collapsed (segments parallel, same direction)

#### 14.1.1 Standard Joint Angles (3-point)

These use the formula above directly with three landmark positions:

| Metric Key | Vertex (B) | A (proximal) | C (distal) | What It Measures |
|------------|-----------|--------------|------------|------------------|
| `Angle_Elbow_L` | Left Elbow (13) | Left Shoulder (11) | Left Wrist (15) | Left elbow flexion/extension |
| `Angle_Elbow_R` | Right Elbow (14) | Right Shoulder (12) | Right Wrist (16) | Right elbow flexion/extension |
| `Angle_Knee_L` | Left Knee (25) | Left Hip (23) | Left Ankle (27) | Left knee flexion/extension |
| `Angle_Knee_R` | Right Knee (26) | Right Hip (24) | Right Ankle (28) | Right knee flexion/extension |

Numbers in parentheses are MediaPipe Pose landmark indices.

#### 14.1.2 Spine-Relative Joint Angles

For shoulders and hips, $\vec{v_1}$ is replaced by the **spine vector** (trunk axis) instead of a proximal bone, making the measurement independent of torso lean.

**Spine vector computation:**

$$\text{MidShoulder} = \frac{P_{11} + P_{12}}{2} \qquad \text{MidHip} = \frac{P_{23} + P_{24}}{2}$$

$$\vec{S} = \text{MidShoulder} - \text{MidHip} \quad \text{(points upward along trunk)}$$

Then for shoulder/hip angles:

$$\theta = \arccos\left(\frac{\vec{S} \cdot \vec{v_2}}{|\vec{S}| \cdot |\vec{v_2}|}\right)$$

| Metric Key | Vertex (B) | $\vec{v_1}$ | C (distal) | What It Measures |
|------------|-----------|-------------|------------|------------------|
| `Angle_Shoulder_L` | Left Shoulder (11) | Spine vector $\vec{S}$ | Left Elbow (13) | Left shoulder abduction/flexion relative to trunk |
| `Angle_Shoulder_R` | Right Shoulder (12) | Spine vector $\vec{S}$ | Right Elbow (14) | Right shoulder abduction/flexion relative to trunk |
| `Angle_Hip_L` | Left Hip (23) | Spine vector $\vec{S}$ | Left Knee (25) | Left hip flexion relative to trunk |
| `Angle_Hip_R` | Right Hip (24) | Spine vector $\vec{S}$ | Right Knee (26) | Right hip flexion relative to trunk |

**Why spine-relative?** Without the spine reference, shoulder and hip angles would change when the subject leans forward/backward, even if the limbs stay in the same relative position. The spine vector provides a stable trunk-local reference frame.

---

### 14.2 Distance & Limb Length Computation

**Source:** `src/calculations.py` → `Calculations.calculate_distance()`

**Method:** 3D Euclidean distance between two landmark positions:

$$d = \sqrt{(x_a - x_b)^2 + (y_a - y_b)^2 + (z_a - z_b)^2}$$

Result is rounded to 4 decimal places.

#### 14.2.1 Limb Length Metrics

| Metric Key | Landmark A | Landmark B | What It Measures |
|------------|-----------|------------|------------------|
| `Length_UpperArm_L` | Left Shoulder (11) | Left Elbow (13) | Left upper arm length |
| `Length_LowerArm_L` | Left Elbow (13) | Left Wrist (15) | Left forearm length |
| `Length_UpperArm_R` | Right Shoulder (12) | Right Elbow (14) | Right upper arm length |
| `Length_LowerArm_R` | Right Elbow (14) | Right Wrist (16) | Right forearm length |
| `Length_UpperLeg_L` | Left Hip (23) | Left Knee (25) | Left thigh length |
| `Length_LowerLeg_L` | Left Knee (25) | Left Ankle (27) | Left shin length |
| `Length_UpperLeg_R` | Right Hip (24) | Right Knee (26) | Right thigh length |
| `Length_LowerLeg_R` | Right Knee (26) | Right Ankle (28) | Right shin length |
| `Width_Shoulder` | Left Shoulder (11) | Right Shoulder (12) | Shoulder width |
| `Width_Hip` | Left Hip (23) | Right Hip (24) | Hip width |

**Units:** In single-camera mode, coordinates are normalized (0–1 range relative to frame), so lengths are in **normalized units**. In multi-camera mode, coordinates come from triangulation in **meters**.

#### 14.2.2 Body-Height Normalization

**Source:** `src/calculations.py` → `Calculations.normalize_metrics()`

To compare limb proportions across subjects of different sizes, all length/width metrics are normalized by an estimated body height:

$$\text{MidHip} = \frac{P_{23} + P_{24}}{2} \qquad \text{MidAnkle} = \frac{P_{27} + P_{28}}{2}$$

$$\text{BodyScale} = ||\text{MidHip} - \text{MidAnkle}|| \times k$$

Where $k$ = `BODY_HEIGHT_MULTIPLIER` (default 2.0, since leg-length ≈ half of full stature).

$$\text{Normalized\_Length\_X} = \frac{\text{Length\_X}}{\text{BodyScale}}$$

**Output metrics added:** `Normalized_Length_UpperArm_L`, `Normalized_Width_Shoulder`, etc., plus `Body_Height` (the scale reference itself).

---

### 14.3 Face Metrics

**Source:** `src/calculations.py` → `Calculations.get_face_metrics()`

All face metrics are computed from MediaPipe Face Mesh (468 landmarks) and **normalized by interpupillary distance (IPD)** to be scale-invariant. The landmark indices below are MediaPipe Face Mesh canonical indices.

#### 14.3.1 Interpupillary Distance (IPD)

$$\text{IPD} = ||P_{159} - P_{386}|| \quad \text{(or fallback: } ||P_{33} - P_{263}|| \times 0.5 \text{)}$$

If IPD = 0, it defaults to `IPD_DEFAULT` (0.065 m). IPD serves as the normalization denominator for all face metrics.

#### 14.3.2 Mouth Openness

$$\text{mouth\_h} = ||P_{13} - P_{14}|| \quad \text{(upper lip center to lower lip center)}$$

$$\text{Face\_Mouth\_Openness} = \frac{\text{mouth\_h}}{\text{IPD}}$$

- $\approx 0$ → mouth closed
- $> 0.3$ → mouth wide open (talking, yawning)

#### 14.3.3 Smile Ratio

$$\text{mouth\_w} = ||P_{61} - P_{291}|| \quad \text{(left to right mouth corner)}$$

$$\text{Face\_Smile\_Ratio} = \frac{\text{mouth\_w}}{\text{mouth\_h} + \epsilon}$$

Where $\epsilon = 10^{-6}$ prevents division by zero.

- $> 5$ → wide smile (mouth corners spread, lips closed)
- $\approx 1$ → neutral or open mouth

#### 14.3.4 Eye Openness

**Left Eye:**

$$\text{Face\_Eye\_L\_Openness} = \frac{||P_{159} - P_{145}||}{||P_{33} - P_{133}|| + \epsilon}$$

Vertical opening divided by horizontal span. The right eye uses landmarks 386/374 (vertical) and 362/263 (horizontal).

- $\approx 0.3$ → eye fully open
- $\approx 0$ → eye closed (blink detection)

---

### 14.4 Linear Velocity

**Source:** `src/calculations.py` → `Calculations.get_kinematics()` (2D mode), `src/kinematics_engine.py` → `KinematicsEngine.compute_velocity()` (3D mode)

#### 14.4.1 2D Mode (Single Camera)

For key joints (wrists and ankles), velocity magnitude is computed as:

$$v = \frac{\sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2 + (z_t - z_{t-1})^2}}{\Delta t}$$

Where $\Delta t$ is the time between consecutive frames in seconds.

**Sanity cap:** If $v > 6.0$ m/s, the measurement is discarded.

> **Justification:** The 6 m/s cap is calibrated for **normal locomotion and clinical scenarios** (walking, sit-to-stand, rehabilitation exercises). For reference, typical human limb speeds in context:
>
> | Activity | Peak Wrist/Ankle Speed | Reference |
> |----------|----------------------|----------|
> | Normal walking | 1–2 m/s | Winter (2009), *Biomechanics of Human Movement* |
> | Fast gesturing | 2–4 m/s | Empirical |
> | Boxing punch | 8–12 m/s | Walilko et al. (2005) |
> | Tennis serve (elite) | 20–25 m/s | Fleisig et al. (2003) |
>
> For sports or high-velocity motion capture, increase `MAX_LINEAR_VELOCITY` in `config.py` accordingly (e.g., 15–30 m/s for racket sports).

**Visibility gate:** Both current and previous landmarks must have visibility $\geq$ `VISIBILITY_MIN_METRIC` (0.5).

| Metric Key | Joint | Landmark Index |
|------------|-------|----------------|
| `Velocity_Wrist_L` | Left Wrist | 15 |
| `Velocity_Wrist_R` | Right Wrist | 16 |
| `Velocity_Ankle_L` | Left Ankle | 27 |
| `Velocity_Ankle_R` | Right Ankle | 28 |

#### 14.4.2 3D Mode (Multi-Camera)

Per-axis velocity vector for **every** joint:

$$\vec{v}_t = \frac{P_t - P_{t-1}}{\Delta t} = \left(\frac{x_t - x_{t-1}}{\Delta t}, \; \frac{y_t - y_{t-1}}{\Delta t}, \; \frac{z_t - z_{t-1}}{\Delta t}\right)$$

Magnitude:

$$|\vec{v}_t| = \sqrt{v_x^2 + v_y^2 + v_z^2}$$

**Clamping:** If $|\vec{v}_t|$ > `MAX_LINEAR_VELOCITY` (6.0 m/s), the vector is rescaled:

$$\vec{v}_t' = \vec{v}_t \cdot \frac{v_{max}}{|\vec{v}_t|}$$

**Output keys (per joint $j$):** `joint_{j}_vx`, `joint_{j}_vy`, `joint_{j}_vz`, `joint_{j}_v`

---

### 14.5 Linear Acceleration

**Source:** `src/kinematics_engine.py` → `KinematicsEngine.compute_acceleration()` (3D mode only)

Finite-difference acceleration from consecutive velocity vectors:

$$\vec{a}_t = \frac{\vec{v}_t - \vec{v}_{t-1}}{\Delta t}$$

$$|\vec{a}_t| = \sqrt{a_x^2 + a_y^2 + a_z^2}$$

Requires at least 3 frames (frame 1 → position, frame 2 → velocity, frame 3 → acceleration).

**Output keys (per joint $j$):** `joint_{j}_ax`, `joint_{j}_ay`, `joint_{j}_az`, `joint_{j}_a`

**Units:** m/s² (in 3D mode with calibrated cameras)

---

### 14.6 Angular Velocity

**Source:** `src/calculations.py` → `Calculations.get_kinematics()` (2D), `src/kinematics_engine.py` → `KinematicsEngine.compute_angular_velocity()` (3D)

First-order finite difference of joint angle over time:

$$\omega_t = \frac{\theta_t - \theta_{t-1}}{\Delta t}$$

**Units:** deg/s

**2D mode output keys:**

| Key | Derived From |
|-----|-------------|
| `Velocity_Angle_Elbow_L` | `Angle_Elbow_L` |
| `Velocity_Angle_Elbow_R` | `Angle_Elbow_R` |
| `Velocity_Angle_Shoulder_L` | `Angle_Shoulder_L` |
| `Velocity_Angle_Shoulder_R` | `Angle_Shoulder_R` |
| `Velocity_Angle_Hip_L` | `Angle_Hip_L` |
| `Velocity_Angle_Hip_R` | `Angle_Hip_R` |
| `Velocity_Angle_Knee_L` | `Angle_Knee_L` |
| `Velocity_Angle_Knee_R` | `Angle_Knee_R` |

**3D mode output keys:** `angle_{name}_omega_deg_s` (e.g., `angle_elbow_right_omega_deg_s`)

> **Edge case — angular discontinuity.** Although joint angles are clamped to $[0°, 180°]$ (which avoids the $360° \to 0°$ wraparound problem of full-circle angles), discontinuities can still occur near the boundary values during occlusion events. If a landmark briefly disappears and reappears at a different pose, $\theta$ may jump from $170°$ to $20°$, producing $\omega = -150° / \Delta t$ — a large spurious spike.
>
> **Current protection:** The outlier rejection stage (§14.11) catches these spikes. For each angle, if $|\theta_t - \theta_{t-1}|$ exceeds the dynamic threshold ($T_{base} + k \cdot |\omega_{prev}|$), the new value is rejected and the previous value is held. This effectively acts as an unwrap-by-rejection strategy: impossible jumps are suppressed, and the angle resumes tracking when valid readings return.

---

### 14.7 Body Segment Vector Computation

**Source:** `src/calculations.py` → `Calculations.get_segment_vectors_from_pose_3d()`

Canonical body segment vectors (distal − proximal convention):

| Segment Key | Formula | Anatomical Meaning |
|-------------|---------|---------------------|
| `upper_arm_r` | $P_{\text{Elbow}} - P_{\text{Shoulder}}$ | Right upper arm direction |
| `forearm_r` | $P_{\text{Wrist}} - P_{\text{Elbow}}$ | Right forearm direction |
| `trunk_axis` | $\text{MidShoulder} - \text{MidHip}$ | Trunk longitudinal axis (upward) |
| `pelvic_axis` | $P_{\text{RightHip}} - P_{\text{LeftHip}}$ | Pelvis lateral axis (left → right) |

These vectors are used for spine-relative angle computation and for anatomical reference frame construction.

---

### 14.8 1-Euro Filter (Adaptive Noise Smoothing)

**Source:** `src/one_euro_filter.py` → `OneEuroFilter`

The [1€ Filter](https://cristal.univ-lille.fr/~casiez/1euro/) is an adaptive low-pass filter that reduces jitter in slow motion while preserving responsiveness during fast motion.

#### Step 1: Estimate signal derivative

$$\hat{\dot{x}}_t = \alpha_d \cdot \frac{x_t - \hat{x}_{t-1}}{\Delta t} + (1 - \alpha_d) \cdot \hat{\dot{x}}_{t-1}$$

Where $\alpha_d$ is the EMA coefficient for the derivative channel:

$$\alpha_d = \frac{2\pi f_{d} \Delta t}{2\pi f_{d} \Delta t + 1}$$

$f_d$ = `FILTER_D_CUTOFF` (default 1.0 Hz).

#### Step 2: Compute adaptive cutoff frequency

$$f_c = f_{min} + \beta \cdot |\hat{\dot{x}}_t|$$

- $f_{min}$ = `FILTER_MIN_CUTOFF` (1.0 Hz) — controls smoothness during slow motion
- $\beta$ = `FILTER_BETA` (0.005) — controls how much fast motion reduces smoothing

When $\beta = 0$, the filter behaves as a static low-pass. When $\beta$ is large, fast movements pass through with minimal lag.

#### Step 3: Filter the signal

$$\alpha = \frac{2\pi f_c \Delta t}{2\pi f_c \Delta t + 1}$$

$$\hat{x}_t = \alpha \cdot x_t + (1 - \alpha) \cdot \hat{x}_{t-1}$$

This is an Exponential Moving Average (EMA) where $\alpha$ adapts per frame.

#### Where it's applied

| Context | Applied To | Config Toggle |
|---------|-----------|---------------|
| `PoseCorrector` (2D) | Each of 33 landmarks × (x, y, z) | Always active after init |
| `MasterCoordinator` (3D) | Each of 33 3D joints × (x, y, z) | `ENABLE_3D_ONE_EURO_FILTER` |

> **Tradeoff — phase lag on derived kinematics.** Because the 1-Euro filter is applied to **positions before** velocity/acceleration are computed, the filtered position signal has a causal low-pass characteristic. This means:
>
> - Velocity estimates are **slightly delayed** (phase-lagged) relative to ground truth.
> - Peak acceleration is **attenuated** — the filter rounds off sharp transients.
> - The effect scales with $f_{min}$: higher minimum cutoff → less lag but more jitter.
>
> This is the standard tradeoff in causal filtering: **jitter reduction at the cost of slight temporal delay**. For clinical gait analysis and general motion capture this is acceptable. For applications requiring precise peak acceleration (e.g., impact detection), consider computing velocity from raw positions and filtering the velocity signal separately.

---

### 14.9 Visibility Hard Gate

**Source:** `src/pose_corrector.py` → `PoseCorrector._correct_skeleton()`

For each landmark, if MediaPipe's visibility confidence drops below a threshold, the landmark position is **frozen** to its last known good value:

$$P_t = \begin{cases} \text{Filter}(P_t^{\text{raw}}) & \text{if } v_t \geq v_{\text{gate}} \\ P_{t-1}^{\text{filtered}} & \text{if } v_t < v_{\text{gate}} \end{cases}$$

Where $v_{\text{gate}}$ = `VISIBILITY_HARD_GATE` (0.5).

**Effect:** Prevents erratic jumps when a body part is occluded or poorly detected. The filter's internal timestamp is not updated during a freeze, ensuring smooth resumption when visibility returns.

> **Known side-effect — velocity spike on reappearance.** When a landmark is frozen for $N$ frames and then reappears at a new position, the finite-difference velocity computation sees the *accumulated* displacement all at once:
>
> $$v_{\text{spike}} = \frac{||P_{\text{reappear}} - P_{\text{frozen}}||}{\Delta t_{\text{single\ frame}}}$$
>
> This produces a transient velocity spike that can be orders of magnitude above the true velocity. **Current mitigations:**
>
> 1. The velocity **sanity cap** (6.0 m/s in 2D, `MAX_LINEAR_VELOCITY` in 3D) clamps or discards the spike.
> 2. The **outlier rejection** stage (§14.11) rejects angle changes exceeding the dynamic threshold.
> 3. The **EMA smoothing** stage attenuates surviving spikes over subsequent frames.
>
> **Recommended future improvement:** Reset the velocity integrator (set $v = 0$) when a landmark's visibility transitions from below-gate to above-gate, or apply a linear interpolation ramp over 3–5 frames to bridge the position gap smoothly.

---

### 14.10 Bone Length Constraint (Skeleton Physics)

**Source:** `src/pose_corrector.py` → `PoseCorrector._calibrate()`, `_apply_constraints()`

#### Phase 1: Calibration (first 30 frames)

For each of 8 limb bones (child→parent pairs), the reference length is learned by averaging over `CALIBRATION_FRAMES`:

$$L_{\text{ref}}^{(b)} = \frac{1}{N} \sum_{t=1}^{N} ||P_{\text{child},t} - P_{\text{parent},t}||$$

**Bones tracked:**

| Child | Parent | Segment |
|-------|--------|---------|
| Right Elbow (14) | Right Shoulder (12) | Right upper arm |
| Right Wrist (16) | Right Elbow (14) | Right forearm |
| Left Elbow (13) | Left Shoulder (11) | Left upper arm |
| Left Wrist (15) | Left Elbow (13) | Left forearm |
| Right Knee (26) | Right Hip (24) | Right thigh |
| Right Ankle (28) | Right Knee (26) | Right shin |
| Left Knee (25) | Left Hip (23) | Left thigh |
| Left Ankle (27) | Left Knee (25) | Left shin |

#### Phase 2: Constraint enforcement (after calibration)

Each bone is corrected to match its reference length. The blending uses a **visibility-squared** weight:

$$\alpha = v_{\text{child}}^2$$

The "strictness" (how much to trust the model vs the sensor) is $s = 1 - \alpha$:

$$P_{\text{ideal}} = P_{\text{parent}} + \frac{P_{\text{child}} - P_{\text{parent}}}{||P_{\text{child}} - P_{\text{parent}}||} \cdot L_{\text{ref}}$$

$$P_{\text{child}}' = (1 - s) \cdot P_{\text{child}} + s \cdot P_{\text{ideal}}$$

**Effect:** When visibility is high ($v \to 1$), $\alpha \to 1$, $s \to 0$ → trusts the sensor. When visibility is low ($v \to 0$), $\alpha \to 0$, $s \to 1$ → forces bone to reference length. This prevents unrealistic stretching or compression of limbs.

---

### 14.11 Outlier Rejection & Temporal Smoothing

**Source:** `src/calculations.py` → `Calculations.filter_and_smooth()`

Applied to all angle and length metrics per frame, in two stages:

#### Stage 1: Velocity-Dependent Outlier Rejection (angles only)

For each angle metric, if the frame-to-frame change exceeds a dynamic threshold, the new value is **rejected** and the previous value is held:

$$\text{threshold} = T_{\text{base}} + k \cdot |\omega_{\text{prev}}|$$

Where:
- $T_{\text{base}}$ = `ANGLE_OUTLIER_BASE_THRESHOLD` (50°)
- $k$ = `ANGLE_OUTLIER_VELOCITY_COEFF` (0.1)
- $|\omega_{\text{prev}}|$ = absolute angular velocity from previous frame (deg/s)

$$\text{accepted} = \begin{cases} \text{true} & \text{if } |\theta_t - \theta_{t-1}| \leq \text{threshold} \\ \text{false (hold)} & \text{otherwise} \end{cases}$$

**Logic:** Fast-moving joints get a larger threshold (allows bigger changes), while stationary joints reject even moderate spikes.

#### Stage 2: Exponential Moving Average (EMA)

For all accepted values:

$$\hat{m}_t = \alpha \cdot m_t + (1 - \alpha) \cdot \hat{m}_{t-1}$$

| Metric type | $\alpha$ | Config parameter |
|-------------|----------|-----------------|
| Body angles/lengths | 0.5 | `SMOOTHING_ALPHA_DEFAULT` |
| Face metrics | 0.3 | `SMOOTHING_ALPHA_FACE` |

Smaller $\alpha$ = heavier smoothing (more weight on history). Face metrics use stronger smoothing because micro-expressions create more noise.

---

### 14.12 Direct Linear Transform (DLT) Triangulation

**Source:** `src/triangulation.py` → `Triangulator._triangulate_dlt()`

Given 2D observations $(u_i, v_i)$ from $N$ cameras, each with a $3 \times 4$ projection matrix $P_i$:

#### Step 1: Build the linear system

Each camera contributes two rows to an $A$ matrix (size $2N \times 4$):

$$A_{2i} = w_i \cdot \left(u_i \cdot P_i^{(3)} - P_i^{(1)}\right)$$
$$A_{2i+1} = w_i \cdot \left(v_i \cdot P_i^{(3)} - P_i^{(2)}\right)$$

Where $P_i^{(k)}$ is the $k$-th row of projection matrix $P_i$, and $w_i$ is the per-camera visibility weight.

#### Step 2: Solve via SVD

$$A = U \Sigma V^T$$

The solution $X$ is the last row of $V^T$ (corresponding to the smallest singular value).

#### Step 3: Convert from homogeneous coordinates

The SVD solution $X_h = (X, Y, Z, W)^T$ is in **homogeneous coordinates** — a projective representation where any scalar multiple $k \cdot X_h$ represents the same 3D point. To recover Euclidean coordinates, divide by the homogeneous scale factor $W$:

$$P_{3D} = \left(\frac{X_h}{W}, \frac{Y_h}{W}, \frac{Z_h}{W}\right)$$

If $W \approx 0$, the point is at infinity (degenerate case — cameras nearly co-planar with the point). In practice this is caught by the reprojection error threshold (§14.15).

#### GPU Acceleration

When `CUDA_ENABLED = True`, the entire $A$ matrix construction, SVD, and reprojection error calculation run on GPU using PyTorch tensors on the configured `DEVICE`.

---

### 14.13 Projection Matrix Construction

**Source:** `src/stereo_calibration.py` → `StereoCalibration.get_projection_matrix()`

Each camera's $3 \times 4$ projection matrix is composed from intrinsic and extrinsic parameters:

$$P = K \cdot [R | t]$$

Where:
- $K$ is the $3 \times 3$ **intrinsic matrix**: $K = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}$
- $f_x, f_y$ = focal lengths (pixels)
- $c_x, c_y$ = principal point (image center)
- $R$ is the $3 \times 3$ rotation matrix (camera orientation relative to world)
- $t$ is the $3 \times 1$ translation vector (camera position relative to world)

**Default calibration** (when no calibration file exists):
- Camera A (`local_cam`): at origin, identity rotation
- Camera B (`cam_0`): 1 meter to the right, identity rotation
- Focal length: $f = \text{image\_width}$ (approximation)
- Principal point: image center
- Distortion: zero

---

### 14.14 Point Undistortion

**Source:** `src/stereo_calibration.py` → `StereoCalibration.undistort_point()`

Before triangulation, 2D points are undistorted using the camera's lens model:

$$\hat{p} = \text{cv2.undistortPoints}(p, K, D)$$

Where $D = (k_1, k_2, p_1, p_2, k_3)$ are the radial and tangential distortion coefficients. This reverses lens distortion so that 2D→3D projection geometry is accurate.

---

### 14.15 Reprojection Error

**Source:** `src/triangulation.py` → `Triangulator._calculate_reprojection_error()`

Measures how well a triangulated 3D point $X$ projects back to the observed 2D positions:

$$\hat{p}_i = P_i \cdot X_h \quad \rightarrow \quad \hat{u}_i = \frac{\hat{p}_{i,x}}{\hat{p}_{i,z}}, \; \hat{v}_i = \frac{\hat{p}_{i,y}}{\hat{p}_{i,z}}$$

$$\text{error} = \frac{1}{N} \sum_{i=1}^{N} \sqrt{(u_i - \hat{u}_i)^2 + (v_i - \hat{v}_i)^2}$$

**Units:** Pixels. Points with error > `REPROJECTION_ERROR_THRESHOLD` (15 px) are rejected.

---

### 14.16 3D Point Confidence Score

**Source:** `src/triangulation.py` → `Triangulator._calculate_confidence()`

Composite confidence for each triangulated point:

$$C = \left(w_r \cdot e^{-\text{err}/5} + w_v \cdot \bar{v}\right) \cdot \min\left(1, \frac{n}{4}\right)$$

where $\text{err}$ is the **reprojection error in pixels** (see §14.15). The exponential decay constant of 5 means that at 5 px error the reprojection term contributes $e^{-1} \approx 0.37$, and at 15 px (the rejection threshold) it contributes $e^{-3} \approx 0.05$ — effectively zero.

| Component | Symbol | Default | Meaning |
|-----------|--------|---------|---------|
| Reprojection weight | $w_r$ | 0.4 | How much reprojection error matters |
| Visibility weight | $w_v$ | 0.6 | How much landmark visibility matters |
| Reprojection error | $\text{err}$ | — | Mean pixel error (lower = better) |
| Mean visibility | $\bar{v}$ | — | Average landmark visibility across cameras |
| View count | $n$ | — | Number of cameras that saw this point |
| View bonus | $\min(1, n/4)$ | — | More views = higher confidence (saturates at 4) |

Result is clamped to $[0, 1]$.

---

### 14.17 World Axis Convention Transform

**Source:** `src/master_coordinator.py` → `MasterCoordinator._apply_world_axis_convention()`

After triangulation produces 3D points in OpenCV camera frame (Y-down), they are transformed to the world frame (Y-up):

$$P_{\text{world}} = \begin{pmatrix} x \cdot s_x \\ y \cdot s_y \\ z \cdot s_z \end{pmatrix}$$

Where $s_x, s_y, s_z \in \{-1, +1\}$ are sign flips from `WORLD_AXIS_TRANSFORM`:
- `flip_x = False` → $s_x = +1$
- `flip_y = True` → $s_y = -1$ (converts Y-down to Y-up)
- `flip_z = False` → $s_z = +1$

---

### 14.18 Monocular Depth Fallback

**Source:** `src/master_coordinator.py` → `MasterCoordinator._get_monocular_fallback()`

When a landmark is visible in only one camera (stereo triangulation requires ≥2), a rough 3D position is estimated using assumed subject distance:

$$X = \frac{(u - c_x) \cdot d}{f_x} \qquad Y = \frac{(v - c_y) \cdot d}{f_y} \qquad Z = z_{\text{rel}} \cdot d + d$$

Where:
- $(u, v)$ = 2D landmark position (normalized, then scaled to pixels)
- $(c_x, c_y)$ = image center (principal point)
- $(f_x, f_y)$ = focal lengths from calibration
- $d$ = `MONOCULAR_SUBJECT_DISTANCE_M` (2.5 m) — assumed distance from camera
- $z_{\text{rel}}$ = MediaPipe's relative Z (depth relative to hip center)

**Confidence penalty:** Monocular fallback points get confidence = $0.3$ (vs. stereo confidence typically $0.7$–$0.95$).

---

### 14.19 Image Preprocessing Pipeline

**Source:** `src/detector.py` → `MocapDetector.process()`

Applied to each frame before MediaPipe inference, in order:

#### 14.19.1 ROI Cropping

If a person was detected in the previous frame, the current frame is cropped to the person's bounding box (expanded by `ROI_EXPANSION_FACTOR` = 25%) and resized to `ROI_TARGET_SIZE` (640 px). This provides higher effective resolution for inference. After detection, landmark coordinates are reprojected back to the original frame.

#### 14.19.2 Gamma Correction

$$I_{\text{out}} = 255 \cdot \left(\frac{I_{\text{in}}}{255}\right)^{1/\gamma}$$

Implemented via a 256-entry lookup table (LUT). `GAMMA_DEFAULT` = 1.0 (no change), adjustable to 1.4 for brightening dark scenes.

#### 14.19.3 Face-Guided Exposure

Detects the face region in HSV, measures mean brightness $B_{\text{face}}$, then adjusts the full frame:

$$\text{gain} = \frac{B_{\text{target}}}{B_{\text{face}}}$$

Clamped to [`FACE_EXPOSURE_MIN_GAIN` (0.5), `FACE_EXPOSURE_MAX_GAIN` (2.5)].

$$I_{\text{out}} = \text{clip}(I_{\text{in}} \times \text{gain}, \; 0, \; 255)$$

#### 14.19.4 CLAHE (Contrast Limited Adaptive Histogram Equalization)

Applied to the L channel of LAB color space:

$$L_{\text{out}} = \text{CLAHE}(L_{\text{in}}, \; \text{clipLimit}=2.0, \; \text{tileGridSize}=(8,8))$$

Enhances local contrast without amplifying global noise.

---

### 14.20 Intrinsic Camera Calibration

**Source:** `src/stereo_calibration.py` → `StereoCalibration.calibrate_intrinsic()`

Uses OpenCV's `calibrateCamera()` with a checkerboard pattern:

1. Detect checkerboard corners in ≥10 images (`findChessboardCorners`)
2. Refine corners to sub-pixel accuracy (`cornerSubPix`)
3. Compute optimal $K$ and $D$ via Zhang's method (`calibrateCamera`)
4. Report mean reprojection error across all images

**Output:** `CameraCalibration` object containing $K$ (3×3), $D$ (5 coefficients), and image size.

### 14.21 Stereo Extrinsic Calibration

**Source:** `src/stereo_calibration.py` → `StereoCalibration.calibrate_stereo()`

Given intrinsic parameters from two cameras and simultaneous checkerboard images:

1. Find common checkerboard detections across both cameras
2. Run `cv2.stereoCalibrate()` with fixed intrinsics (flag `CALIB_FIX_INTRINSIC`)
3. Compute relative rotation $R$ and translation $T$ between cameras
4. Camera A is set as world origin ($R = I$, $T = 0$); Camera B gets $R_{AB}$ and $T_{AB}$

---

## 14A. Complete Output Metrics Reference

This section catalogs **every metric** the system produces, organized by source.

### Body Angle Metrics (8 angles)

| Metric Key | Range | Units | Description | Computed In |
|------------|-------|-------|-------------|-------------|
| `Angle_Elbow_L` | 0–180 | degrees | Left elbow flexion/extension | `calculations.py`, `kinematics_engine.py` |
| `Angle_Elbow_R` | 0–180 | degrees | Right elbow flexion/extension | `calculations.py`, `kinematics_engine.py` |
| `Angle_Shoulder_L` | 0–180 | degrees | Left shoulder angle (spine-relative) | `calculations.py` |
| `Angle_Shoulder_R` | 0–180 | degrees | Right shoulder angle (spine-relative) | `calculations.py` |
| `Angle_Hip_L` | 0–180 | degrees | Left hip flexion (spine-relative) | `calculations.py` |
| `Angle_Hip_R` | 0–180 | degrees | Right hip flexion (spine-relative) | `calculations.py` |
| `Angle_Knee_L` | 0–180 | degrees | Left knee flexion/extension | `calculations.py`, `kinematics_engine.py` |
| `Angle_Knee_R` | 0–180 | degrees | Right knee flexion/extension | `calculations.py`, `kinematics_engine.py` |

### Limb Length Metrics (10 lengths)

| Metric Key | Units | Description |
|------------|-------|-------------|
| `Length_UpperArm_L` | normalized / meters | Left shoulder → elbow |
| `Length_LowerArm_L` | normalized / meters | Left elbow → wrist |
| `Length_UpperArm_R` | normalized / meters | Right shoulder → elbow |
| `Length_LowerArm_R` | normalized / meters | Right elbow → wrist |
| `Length_UpperLeg_L` | normalized / meters | Left hip → knee |
| `Length_LowerLeg_L` | normalized / meters | Left knee → ankle |
| `Length_UpperLeg_R` | normalized / meters | Right hip → knee |
| `Length_LowerLeg_R` | normalized / meters | Right knee → ankle |
| `Width_Shoulder` | normalized / meters | Left shoulder → right shoulder |
| `Width_Hip` | normalized / meters | Left hip → right hip |

### Normalized Length Metrics (12 metrics)

Each `Length_*` and `Width_*` metric produces a corresponding `Normalized_*` version divided by `Body_Height`.

| Metric Key | Range | Description |
|------------|-------|-------------|
| `Normalized_Length_UpperArm_L` | 0–1 | Ratio to body height |
| ... (one per length/width metric above) | | |
| `Body_Height` | varies | Reference scale (MidHip–MidAnkle × 2.0) |

### Face Metrics (4 metrics)

| Metric Key | Range | Units | Description |
|------------|-------|-------|-------------|
| `Face_Mouth_Openness` | 0–1+ | ratio | Lip separation / IPD |
| `Face_Smile_Ratio` | 0–20+ | ratio | Mouth width / mouth height |
| `Face_Eye_L_Openness` | 0–1 | ratio | Left eye vertical / horizontal |
| `Face_Eye_R_Openness` | 0–1 | ratio | Right eye vertical / horizontal |

### Angular Velocity Metrics (8 metrics, 2D mode)

| Metric Key | Units | Description |
|------------|-------|-------------|
| `Velocity_Angle_Elbow_L` | deg/s | Left elbow angular velocity |
| `Velocity_Angle_Elbow_R` | deg/s | Right elbow angular velocity |
| `Velocity_Angle_Shoulder_L` | deg/s | Left shoulder angular velocity |
| `Velocity_Angle_Shoulder_R` | deg/s | Right shoulder angular velocity |
| `Velocity_Angle_Hip_L` | deg/s | Left hip angular velocity |
| `Velocity_Angle_Hip_R` | deg/s | Right hip angular velocity |
| `Velocity_Angle_Knee_L` | deg/s | Left knee angular velocity |
| `Velocity_Angle_Knee_R` | deg/s | Right knee angular velocity |

### Linear Velocity Metrics (4 metrics, 2D mode)

| Metric Key | Units | Cap | Description |
|------------|-------|-----|-------------|
| `Velocity_Wrist_L` | units/s | 6.0 | Left wrist speed |
| `Velocity_Wrist_R` | units/s | 6.0 | Right wrist speed |
| `Velocity_Ankle_L` | units/s | 6.0 | Left ankle speed |
| `Velocity_Ankle_R` | units/s | 6.0 | Right ankle speed |

### 3D Kinematics Metrics (per-joint, exported flat)

For each of 33 joints (indexed 0–32), the KinematicsEngine exports:

| Key Pattern | Units | Available From |
|-------------|-------|----------------|
| `joint_{id}_x`, `_y`, `_z` | meters | Frame 1 |
| `joint_{id}_confidence` | 0–1 | Frame 1 |
| `joint_{id}_vx`, `_vy`, `_vz` | m/s | Frame 2+ |
| `joint_{id}_v` | m/s | Frame 2+ (magnitude) |
| `joint_{id}_ax`, `_ay`, `_az` | m/s² | Frame 3+ |
| `joint_{id}_a` | m/s² | Frame 3+ (magnitude) |

### 3D Angular Kinematics (8 angles, exported flat)

| Key Pattern | Units | Description |
|-------------|-------|-------------|
| `angle_elbow_right_deg` | degrees | Right elbow angle |
| `angle_elbow_right_omega_deg_s` | deg/s | Right elbow angular velocity |
| `angle_elbow_left_deg` | degrees | Left elbow angle |
| `angle_elbow_left_omega_deg_s` | deg/s | Left elbow angular velocity |
| `angle_knee_right_deg` | degrees | Right knee angle |
| `angle_knee_right_omega_deg_s` | deg/s | Right knee angular velocity |
| `angle_knee_left_deg` | degrees | Left knee angle |
| `angle_knee_left_omega_deg_s` | deg/s | Left knee angular velocity |
| `angle_shoulder_right_deg` | degrees | Right shoulder angle |
| `angle_shoulder_right_omega_deg_s` | deg/s | Right shoulder angular velocity |
| `angle_shoulder_left_deg` | degrees | Left shoulder angle |
| `angle_shoulder_left_omega_deg_s` | deg/s | Left shoulder angular velocity |
| `angle_hip_right_deg` | degrees | Right hip angle |
| `angle_hip_right_omega_deg_s` | deg/s | Right hip angular velocity |
| `angle_hip_left_deg` | degrees | Left hip angle |
| `angle_hip_left_omega_deg_s` | deg/s | Left hip angular velocity |

### Spine Vector (3D mode)

| Key | Units | Description |
|-----|-------|-------------|
| `spine_x` | meters | Trunk axis X component |
| `spine_y` | meters | Trunk axis Y component |
| `spine_z` | meters | Trunk axis Z component |

### Confidence & Reliability (per-frame)

| Key | Type | Description |
|-----|------|-------------|
| `low_reliability_landmarks` | list of ints | Joint IDs where confidence < threshold |
| `confidence_data` | dict | Per-joint confidence scores + flags |

### Summary: Total Metrics Per Frame

| Mode | Metric Count | Breakdown |
|------|-------------|-----------|
| **Single camera** | ~34 | 8 angles + 10 lengths + 12 normalized + 4 face |
| **Single camera + kinematics** | ~46 | Above + 8 angular velocities + 4 linear velocities |
| **Multi-camera (3D)** | ~400+ | 33 joints × (x,y,z,v,vx,vy,vz,ax,ay,az,a,conf) + 16 angles/omega + spine + confidence |

---

## 15. Known Limitations

This section documents known constraints and edge cases that users and reviewers should be aware of.

### 15.1 Camera Hardware Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **Rolling shutter** | Consumer webcams expose rows sequentially (top-to-bottom). Fast lateral motion causes skew distortion in single frames. | Use global-shutter cameras for high-speed capture, or limit to moderate motion speeds (<3 m/s). |
| **Motion blur** | At 30 fps with typical 33 ms exposure, fast limbs (>2 m/s) produce blur that degrades MediaPipe landmark localization. | Reduce exposure time (increase lighting), use higher frame rate cameras, or accept reduced accuracy during fast transients. |
| **Auto-exposure / auto-white-balance** | Camera firmware adjustments cause inter-frame brightness shifts that affect landmark detection consistency. | Lock exposure and white balance manually where possible. The face-guided exposure module (§14.19.3) partially compensates. |
| **Low-light noise** | Noisy images in dim environments reduce MediaPipe confidence and increase landmark jitter. | Use CLAHE preprocessing (§14.19.4), add lighting, or lower detection confidence thresholds. |

### 15.2 Pose Estimation Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **Self-occlusion** | When a limb passes behind the torso (e.g., arm behind back), MediaPipe visibility drops sharply and landmark positions become unreliable. | Visibility hard gate (§14.9) freezes position. Bone length constraints (§14.10) prevent skeleton distortion. Multi-camera setup provides alternative viewpoints. |
| **Multi-person overlap** | When two subjects overlap in the image, MediaPipe's person-assignment may swap IDs or merge landmarks. | Keep subjects spatially separated. Current system uses `NUM_POSES = 1` by default. |
| **Extreme poses** | Unusual body configurations (inverted, fetal position, extreme contortion) are underrepresented in MediaPipe's training data. | Accept reduced accuracy for non-standard poses. HEAVY model performs best. |
| **Loose clothing / accessories** | Baggy clothes, capes, or held objects shift apparent joint positions away from skeletal joints. | Wear fitted clothing during capture. |

### 15.3 Multi-Camera Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **Two-camera baseline** | With only 2 views, depth resolution is limited and degenerate viewing angles (subject on camera baseline) cause poor triangulation. | Position cameras with 60°–120° angular separation. Add more cameras for better coverage. |
| **WiFi latency jitter** | Variable network delay adds noise to timestamp matching beyond clock offset correction. Typical WiFi deltas: 30–170 ms. | Use wired Ethernet for consistent <1 ms latency. Clock sync (§7) corrects systematic offset. Sync threshold widened to 100 ms to tolerate random jitter. Stale-frame eviction (2 s) prevents dead cameras from blocking sync. |
| **Uncalibrated deployment** | Default calibration assumes 1 m horizontal baseline with identity rotation — acceptable for demos but metrically inaccurate. | Perform stereo calibration with checkerboard before quantitative sessions. |
| **Single-room scale** | System designed for indoor capture at 2–5 m range. Outdoor or large-venue scenarios are untested. | Keep subjects within 1–5 m of cameras. |

### 15.4 Algorithm Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **Velocity spike on visibility recovery** | When a frozen landmark reappears, accumulated displacement causes a transient velocity spike (see §14.9). | Sanity cap (6 m/s) clamps spikes. Future: velocity reset on visibility transition. |
| **1-Euro filter phase lag** | Causal low-pass filtering delays velocity/acceleration estimates and attenuates peak acceleration (see §14.8). | Acceptable for clinical use. For impact detection, consider filtering velocity separately. |
| **Angular velocity near boundaries** | Angle jumps during occlusion can produce spurious angular velocity spikes (see §14.6). | Outlier rejection (§14.11) catches these. |
| **No absolute scale without calibration** | Single-camera mode produces normalized coordinates (0–1), not metric values. | Use multi-camera calibrated setup for metric measurements. |

---

## 16. Error Sources & Budget

This section identifies and quantifies the major sources of error in the 3D reconstruction pipeline.

### 16.1 Error Source Taxonomy

```
3D Position Error
├── Pose Detector Noise (MediaPipe)
│   ├── Landmark localization: ±2–5 px (FULL), ±3–8 px (LITE)
│   ├── Visibility estimation uncertainty
│   └── Frame-to-frame jitter (temporal noise)
├── Camera Calibration Error
│   ├── Intrinsic parameters (focal length, principal point): ±0.5–2 px RMS
│   ├── Lens distortion model residual: ±0.1–1 px
│   └── Extrinsic parameters (R, T): depends on checkerboard quality
├── Timestamp Mismatch
│   ├── Clock offset (corrected by Cristian's Algorithm): ±1–5 ms residual
│   ├── Network jitter (WiFi): ±2–10 ms
│   └── Rolling shutter skew: ±1–5 ms effective
├── Triangulation Geometry
│   ├── Baseline-to-distance ratio (B/D < 0.1 → poor depth)
│   ├── Viewing angle (near-parallel rays → depth ambiguity)
│   └── Numerical conditioning of SVD
└── Post-Processing
    ├── 1-Euro filter phase lag: 1–3 frames
    ├── EMA smoothing delay: 1–2 frames
    └── Bone constraint distortion: ±1–3 mm when active
```

### 16.2 Typical Error Budget (calibrated 2-camera setup at 3 m)

| Error Source | Contribution | Magnitude |
|-------------|-------------|----------|
| MediaPipe landmark noise | Dominant | ±1–3 cm per joint |
| Calibration residual | Systematic bias | ±0.5–1 cm |
| Timestamp mismatch (after sync) | Motion-dependent | ±0.3–1.5 cm at 1 m/s |
| Triangulation geometry | Depth-dependent | ±1–5 cm (depth axis) |
| 1-Euro filter smoothing | Temporal | 1–3 frame delay on peaks |
| **Combined (RSS)** | **Total** | **±2–6 cm per joint** |

> **Note:** The dominant error source is MediaPipe's landmark localization noise, not the triangulation or calibration. Improving the pose detector (e.g., using a more accurate model or higher resolution input) would have the largest impact on overall system accuracy.

### 16.3 Validation Metrics

The test suite (§13) provides quantitative validation:

| Test | What It Measures | Acceptance Criterion |
|------|-----------------|---------------------|
| `test_static_jitter.py` | Position stability (std dev) | ≤0.005 m |
| `test_known_angle.py` | Joint angle accuracy | ≤5° error vs ground truth |
| `test_known_distance.py` | Inter-joint distance accuracy | ≤0.05 m error vs measured |

---

## 17. Units Consistency Reference

The system uses different unit conventions depending on the operating mode. This table provides a quick reference.

### 17.1 Coordinate Units by Mode

| Quantity | Single Camera | Multi-Camera (uncalibrated) | Multi-Camera (calibrated) |
|----------|--------------|---------------------------|---------------------------|
| Landmark position (x, y) | 0–1 (fraction of frame) | 0–1 (fraction of frame) | 0–1 (fraction of frame) |
| Landmark position (z) | Relative to hip center | Relative to hip center | Relative to hip center |
| 3D position (X, Y, Z) | N/A | Arbitrary units (approx. meters) | **Meters** |
| Limb lengths | Normalized (0–1 range) | Arbitrary | **Meters** |
| Joint angles | **Degrees** (0–180) | **Degrees** (0–180) | **Degrees** (0–180) |
| Linear velocity | Normalized units/s | Arbitrary units/s | **m/s** |
| Linear acceleration | N/A | N/A | **m/s²** |
| Angular velocity | **deg/s** | **deg/s** | **deg/s** |
| Timestamps | **Nanoseconds** (epoch) | **Nanoseconds** (epoch) | **Nanoseconds** (epoch) |
| Reprojection error | N/A | **Pixels** | **Pixels** |
| Confidence scores | 0–1 (unitless) | 0–1 (unitless) | 0–1 (unitless) |

### 17.2 Face Metric Units

All face metrics are **unitless ratios** normalized by interpupillary distance (IPD):

| Metric | Numerator | Denominator | Typical Range |
|--------|-----------|-------------|---------------|
| `Face_Mouth_Openness` | Lip separation (px) | IPD (px) | 0–0.5 |
| `Face_Smile_Ratio` | Mouth width (px) | Mouth height (px) | 1–15 |
| `Face_Eye_L_Openness` | Vertical opening (px) | Horizontal span (px) | 0–0.4 |
| `Face_Eye_R_Openness` | Vertical opening (px) | Horizontal span (px) | 0–0.4 |

### 17.3 Normalized vs Absolute Lengths

| Metric Form | Example | Units | When Available |
|-------------|---------|-------|----------------|
| Raw length | `Length_UpperArm_L = 0.142` | Normalized (single cam) or meters (3D) | Always |
| Normalized length | `Normalized_Length_UpperArm_L = 0.078` | Ratio to `Body_Height` (unitless) | When body scale computable |
| Body height reference | `Body_Height = 1.82` | Same as raw coordinates | When hips + ankles visible |

> **Key rule:** If you need metric (SI) units, use multi-camera mode with stereo calibration. Single-camera output is in frame-normalized coordinates and should not be interpreted as physical distances.

---

## 18. Performance & Optimization

### Benchmarks

| Mode | FPS | Memory | CPU |
|------|-----|--------|-----|
| Single camera (FULL model) | 25–30 | ~500 MB | 40–60% |
| Single camera (LITE model) | 60+ | ~300 MB | 20–30% |
| Multi-camera (2 laptops) | 15–25 | ~700 MB | 50–70% |

### System Requirements Justification

| Resource | Why It's Needed | Impact of Reduction |
|----------|----------------|--------------------|
| **~500 MB RAM** (FULL model) | MediaPipe loads pose (9 MB) + face (5 MB) + hand (12 MB) models into GPU/CPU memory, plus OpenCV frame buffers (1280×720×3 = 2.7 MB/frame × 3 pipeline stages), plus ZMQ send/receive buffers | Disabling face/hand detection saves ~100 MB. Using LITE model saves ~200 MB. |
| **40–60% single-core CPU** | MediaPipe inference (15–25 ms/frame), OpenCV preprocessing (2–5 ms), metric calculations (1–2 ms), visualization overlay (1–3 ms), all on the main thread | ROI cropping reduces inference cost by ~40%. Frame skip trades accuracy for CPU. |
| **GPU memory (optional)** | When Metal (macOS) or CUDA (Linux/Windows) delegates are active: model weights + inference workspace ≈ 200–400 MB VRAM. Face mesh is the largest GPU consumer. | Disabling `ENABLE_FACE_DETECTION` reduces GPU memory by ~30%. |
| **Network bandwidth** | Each frame packet: ~15 KB (JPEG at quality 35) + ~2 KB (landmarks) ≈ 17 KB × 30 fps ≈ **500 KB/s** per camera | Lower `NETWORK_JPEG_QUALITY` or reduce `NETWORK_STREAM_WIDTH` for slower networks. |

### Key Optimizations

| Optimization | Impact | Configuration |
|-------------|--------|---------------|
| ROI cropping | ~40% faster inference | `ENABLE_ROI_CROPPING = True` |
| Frame skip | Linear FPS boost | `FRAME_SKIP = N` |
| LITE model | 2× FPS vs FULL | `POSE_MODEL_COMPLEXITY = 'LITE'` |
| GPU delegate (Metal/CUDA) | 20–50% faster inference | `INFERENCE_BACKEND = 'mps'` |
| Network JPEG compression | 60–80% bandwidth reduction | `NETWORK_JPEG_QUALITY = 35` |
| ZMQ CONFLATE | Prevents queue buildup | Always enabled |
| Background DB writer | Non-blocking recording | Automatic (50-frame batches) |
| Frame coalescing (GUI) | Prevents UI lag | Automatic |
| Dedicated send/decode threads | Non-blocking network I/O | Automatic in master mode |
| Disable face/hand detection | ~30% faster per-frame | `ENABLE_FACE_DETECTION = False` |

### Tuning Tips

- For **low-end hardware**: Use LITE model, disable face/hand, set `FRAME_SKIP = 1`
- For **best accuracy**: Use HEAVY model, calibrate cameras, reduce `SYNC_TIME_THRESHOLD_MS`
- For **low latency network**: Lower `NETWORK_JPEG_QUALITY`, reduce stream resolution
- For **smooth visualization**: Increase `FILTER_MIN_CUTOFF` (more smoothing), decrease `FILTER_BETA`
- For **responsive tracking**: Increase `FILTER_BETA`, decrease `FILTER_MIN_CUTOFF`

---

## 19. Troubleshooting

### Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| "NO DATA" on remote feed | Server not running or firewall blocking | Start server, run `allow_firewall.ps1` as admin |
| Port 5001 closed | Windows firewall | `New-NetFirewallRule -LocalPort 5001 -Protocol TCP -Action Allow` |
| Camera not opening | Another app using webcam | Close Zoom/Teams/Skype |
| High latency / frame drops | Wi-Fi congestion | Move closer to router, use Ethernet if possible |
| "Access Denied" on firewall | Not admin | Run PowerShell as Administrator |
| IndentationError on startup | Python file corruption | Run `python -c "import py_compile; py_compile.compile('src/<file>.py', doraise=True)"` to find issues |
| `mediapipe` import error | Wrong Python version | Use Python 3.8–3.10 |
| Metal/MPS errors on macOS | PyTorch version mismatch | Set `INFERENCE_BACKEND = 'cpu'` |

### Diagnostic Commands

```bash
# Check IP address
ifconfig en0 | grep "inet " | awk '{print $2}'   # Mac
ipconfig                                            # Windows

# Test port connectivity
python tests/test_connection.py

# Test ZMQ reception
python tests/test_receive_debug.py

# Verify Python file syntax
python -c "import py_compile; py_compile.compile('src/database.py', doraise=True)"

# Test mock sender (no camera needed)
python tests/test_sender.py
```

---

## 20. File-by-File Reference

### Root Directory

| File | Purpose |
|------|---------|
| `config.py` | All configuration parameters (259 lines) |
| `main_gui.py` | Tkinter desktop application (914 lines) |
| `launch_multi_camera.py` | CLI launcher for multi-camera modes |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview and usage guide |
| `SETUP.md` | Step-by-step multi-camera setup (210 lines) |
| `UPDATE_NOTES.md` | Recent change log |
| `IMPLEMENTATION_PROGRESS.md` | Pipeline implementation status |
| `COORDINATE_SYSTEM_SPEC.md` | Locked coordinate/angle conventions |
| `SYSTEM_ARCHITECTURE.md` | Normative architecture + calibration and quality policy |
| `DOCUMENTATION.md` | This file |

### `src/` — Core Modules (16 files)

| File | Lines | Primary Class | Role |
|------|-------|---------------|------|
| `camera.py` | ~60 | `Camera` | Thread-safe OpenCV capture |
| `detector.py` | ~350 | `MocapDetector` | MediaPipe landmark detection + preprocessing |
| `pose_corrector.py` | ~200 | `PoseCorrector` | 1-Euro filtering + bone length constraints |
| `calculations.py` | ~400 | `Calculations` | Biomechanical metrics (angles, lengths, kinematics) |
| `one_euro_filter.py` | ~60 | `OneEuroFilter` | Adaptive noise filter |
| `visualizer.py` | ~100 | `Visualizer` | 2D landmark overlay rendering |
| `streamer.py` | ~150 | `VideoStreamer` | Single-camera pipeline orchestrator |
| `database.py` | ~474 | `MocapDB` | SQLite/PostgreSQL recording engine |
| `report_generator.py` | ~400 | `ReportGenerator` | 20+ analysis plots |
| `visualizer_3d.py` | ~200 | `Visualizer3D` | Plotly interactive HTML dashboard |
| `live_visualizer_3d.py` | ~100 | `LiveVisualizer3D` | Real-time Matplotlib 3D skeleton |
| `camera_server.py` | ~250 | `CameraServer` | ZMQ broadcaster (remote cameras) |
| `master_coordinator.py` | ~500 | `MasterCoordinator` | Frame sync + triangulation + kinematics |
| `frame_synchronizer.py` | ~150 | `FrameSynchronizer` | Timestamp-based frame matching |
| `stereo_calibration.py` | ~300 | `StereoCalibration` | Camera intrinsic/extrinsic calibration |
| `triangulation.py` | ~250 | `Triangulator` | DLT 3D reconstruction (CPU/CUDA) |
| `kinematics_engine.py` | ~250 | `KinematicsEngine` | 3D velocity, acceleration, angles |

### `frontend/` — React Web Interface

| File | Purpose |
|------|---------|
| `package.json` | Dependencies: React 19, Vite 7, Tailwind 4 |
| `vite.config.js` | Dev server + API proxy configuration |
| `tailwind.config.js` | Custom coffee-themed color palette |
| `index.html` | HTML shell |
| `src/main.jsx` | React mount point |
| `src/App.jsx` | Single-page dashboard component (~273 lines) |
| `src/App.css` | Global styles |
| `src/index.css` | Tailwind imports |

### `models/` — MediaPipe Model Files

| File | Size | Purpose |
|------|------|---------|
| `pose_landmarker_lite.task` | ~4 MB | Fastest pose model |
| `pose_landmarker_full.task` | ~9 MB | Balanced accuracy/speed |
| `pose_landmarker_heavy.task` | ~26 MB | Most accurate pose model |
| `face_landmarker.task` | ~5 MB | 468-landmark face mesh |
| `hand_landmarker.task` | ~12 MB | 21-landmark hand mesh |

### `tests/` — Test Suite (10 files)

| File | Category | Tests |
|------|----------|-------|
| `test_connection.py` | Network | TCP port scan + coordinator sync |
| `test_ports.py` | Network | Port reachability |
| `test_remote_ports.py` | Network | Parameterized port check |
| `test_receive_debug.py` | Network | ZMQ message decoding |
| `test_multicam.py` | Integration | Server/master pipeline |
| `test_sender.py` | Integration | 300 mock frames (10s) |
| `test_sender_continuous.py` | Integration | Infinite mock broadcast |
| `test_static_jitter.py` | Validation | Position stability (≤0.005 m std) |
| `test_known_angle.py` | Validation | Angle accuracy (≤5° error) |
| `test_known_distance.py` | Validation | Distance accuracy (≤0.05 m error) |

### `scripts/` — Helpers

| File | Platform | Purpose |
|------|----------|---------|
| `run_gui.bat` | Windows | Launch single-camera GUI |
| `run_master.bat` | Windows | Launch master mode |
| `allow_firewall.ps1` | Windows | Open ZMQ ports |

---

## 21. Glossary

| Term | Definition |
|------|------------|
| **DLT** | Direct Linear Transform — algorithm for 3D triangulation from 2D correspondences |
| **1-Euro Filter** | Adaptive low-pass filter that adjusts cutoff frequency based on signal speed |
| **EMA** | Exponential Moving Average — weighted average giving more weight to recent values |
| **Landmark** | A detected body keypoint (e.g., left elbow) with (x, y, z, visibility) |
| **Visibility** | MediaPipe's confidence that a landmark is visible (0–1) |
| **Triangulation** | Computing 3D position from two or more 2D observations |
| **Reprojection error** | Pixel distance between observed 2D point and 3D point projected back to 2D |
| **SVD** | Singular Value Decomposition — used to solve the DLT system |
| **ROI** | Region of Interest — cropped bounding box around detected person |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization — local contrast enhancement |
| **ZMQ** | ZeroMQ — high-performance messaging library for inter-process communication |
| **msgpack** | MessagePack — binary serialization format (faster/smaller than JSON) |
| **PUB/SUB** | Publish/Subscribe — ZMQ messaging pattern (broadcast to all subscribers) |
| **CONFLATE** | ZMQ option that keeps only the latest message in the buffer |
| **Intrinsic matrix** | 3×3 camera matrix $K$ encoding focal length and principal point |
| **Extrinsic parameters** | Rotation $R$ and translation $T$ between cameras |
| **Projection matrix** | $P = K[R|t]$ — 3×4 matrix mapping 3D world points to 2D image pixels |
| **Homogeneous coordinates** | 4D representation $(X, Y, Z, W)$ of 3D points for projective geometry |
| **Bone constraint** | Enforcement of learned bone lengths to prevent impossible skeletal deformations |
| **Frame synchronization** | Matching frames from multiple cameras by timestamp within a tolerance window |
| **Kinematics** | Study of motion: position, velocity, acceleration, angles, angular velocity |
| **IPD** | Interpupillary distance — distance between eye centers (face normalization) |
| **FFT** | Fast Fourier Transform — frequency analysis of temporal signals |
| **Spectrogram** | Time-frequency visualization of signal energy distribution |
| **Cristian's Algorithm** | Clock synchronization protocol using ping-pong RTT estimation to compute clock offset between two machines |
| **RTT** | Round-Trip Time — total time for a message to travel to a server and back |
| **NTP** | Network Time Protocol — internet protocol for synchronizing computer clocks to a reference |
| **Rolling shutter** | Camera sensor readout mode where rows are exposed sequentially, causing motion skew |
| **Phase lag** | Temporal delay introduced by causal (real-time) filtering — filtered signal trails the true signal |
| **RSS** | Root Sum of Squares — method for combining independent error sources: $\sqrt{e_1^2 + e_2^2 + \cdots}$ |

---

*Generated 27 February 2026 — Motion Capture VS5 Multi-Camera Enhanced Edition*

---

## 22. Development Roadmap

This section documents 14 feature areas reviewed against the current codebase, classifying each as **implemented**, **partially implemented**, or **pending**. Each entry describes what exists today, what is missing, and what the target implementation should include.

### Status Summary

| # | Feature Area | Status | Priority |
|---|---|---|---|
| 1 | Camera Calibration Module | ✅ Implemented (GUI recalibration workflow added) | — |
| 2 | True Triangulation Engine | ✅ Fully implemented | — |
| 3 | Hard Frame Synchronization | ✅ Implemented | P3 — NTP/PTP optional |
| 4 | Message Format Extension | ✅ Implemented (schema_version, calibration_id, capture_fps, gap detection) | — |
| 5 | Occlusion Fusion | ✅ Implemented (3-tier + state machine) | — |
| 6 | Dataset Pipeline | ✅ Implemented (JSON export, ZIP archive, raw frames, session list) | — |
| 7 | Replacing World Landmarks | ✅ Implemented (Tier 3 fallback chain) | — |
| 8 | Kinematics Engine Upgrade | ✅ Mostly implemented | P3 |
| 9 | Uncertainty + Error Model | ✅ Implemented (inter-camera disagreement) | — |
| 10 | Dashboard Help System | ✅ Implemented (help dialog with equations) | — |
| 11 | Latency Visibility | ✅ Implemented (pipeline panel + instrumentation) | — |
| 12 | Camera Placement Visualization | ✅ Implemented (3D FOV viewer) | — |
| 13 | Recalibration | ✅ Implemented (guided chessboard workflow) | — |
| 14 | GPU Scheduling | ✅ Implemented (per-model profiling) | — |

---

### 22.1 Camera Calibration Module

**Status: ✅ Implemented**

**What exists:**
- `stereo_calibration.py` includes full intrinsic calibration via `cv2.findChessboardCorners` + `cv2.calibrateCamera` → outputs `CameraCalibration` dataclass with `intrinsic_matrix`, `distortion_coeffs`, `reprojection_error`
- Stereo extrinsic calibration via `cv2.stereoCalibrate` → rotation `R` and translation `T` between two cameras
- `undistort_point()` using `cv2.undistortPoints`
- `get_projection_matrix()` computing $P = K[R|t]$
- JSON save/load for calibration files
- **NEW:** GUI "🔧 Recalibrate" button → guided chessboard capture workflow (see §22.13)
- **NEW:** Calibration profiles saved with timestamps (`calibration_{epoch}.json`)
- **NEW:** Live calibration reload into active triangulation pipeline

**Remaining improvements:**
- No `cv2.solvePnP` — no single-camera extrinsic calibration from a known object
- No multi-camera (N>2) chain calibration or bundle adjustment

---

### 22.2 True Triangulation Engine

**What exists (COMPLETE):**
- `triangulation.py` — Full DLT (Direct Linear Transform) via SVD, with both PyTorch CUDA and NumPy backends
- `triangulate_point()` — multi-view weighted DLT with undistortion, reprojection error check, confidence scoring
- `triangulate_pose()` — per-landmark triangulation across all cameras
- `triangulate_with_cv2()` — OpenCV `cv2.triangulatePoints` verification path
- Reprojection error calculation with GPU acceleration
- Confidence scoring combining reprojection error, visibility, and view count

**What could be improved:**
- No RANSAC or robust triangulation (outlier-resilient multi-view)
- No ray-intersection (midpoint) method — only DLT (sufficient for 2 cameras)

> **Assessment:** This module is production-ready. DLT triangulation with confidence scoring, GPU acceleration, and fallback paths is a complete implementation.

---

### 22.3 Hard Frame Synchronization

**What exists:**
- Cristian's Algorithm clock offset estimation (`estimate_clock_offset` on ZMQ REQ/REP, port 6003)
- Clock sync responder (`_clock_sync_handler`) with REP socket state protection
- `FrameSynchronizer` with configurable threshold (100 ms), 10-frame buffers
- Stale-frame eviction at 2 s timeout
- Periodic re-sync every 300 s
- Per-sample fresh socket creation to prevent REQ state corruption after timeouts

**What is missing:**
- No NTP client integration (documented as complementary — manual command only)
- No PTP (Precision Time Protocol) support
- No hardware trigger / genlock for shutter synchronization
- No clock drift *rate* estimation (only absolute offset)

**Target implementation:**
- Automatic NTP pre-check at startup: warn if system clock offset > 50 ms
- Drift rate estimation: fit linear model to offset samples over time
- For research-grade: GPIO/audio trigger documentation and integration path

---

### 22.4 Message Format Extension

**Status: ✅ Implemented**

**What exists:**
- Frame payload includes: `type`, `camera_id`, `frame_number`, `timestamp`, `landmarks`, `results`, `frame_jpeg`
- **NEW:** `schema_version` (currently `2`) for forward compatibility
- **NEW:** `calibration_id` to verify sender/receiver use the same calibration
- **NEW:** `capture_fps` field in frame payloads
- **NEW:** Receiver-side sequence gap detection — warns when frames are dropped
- Config params: `MESSAGE_SCHEMA_VERSION`, `CALIBRATION_ID`

**Implementation details:**
- `camera_server.py` sends extended payload with all new fields
- `master_coordinator.py` checks `schema_version`, tracks `frame_number` per camera, logs gaps

---

### 22.5 Occlusion Fusion

**Status: ✅ Implemented**

**What exists:**
- Three-tier fusion in `get_synced_3d_pose()`:
  - **Tier 1:** Multi-view weighted triangulation (DLT) with inter-camera disagreement measurement
  - **Tier 2:** Monocular back-projection fallback for landmarks seen by only 1 camera
  - **Tier 3:** MediaPipe `pose_world_landmarks` fallback (hip-relative, averaged across cameras)
- **NEW:** Per-landmark occlusion state machine: `VISIBLE` → `OCCLUDED` → `PREDICTED`
- **NEW:** Predicted state holds last known 3D position for up to 15 frames (~500ms @ 30fps)
- **NEW:** Occlusion states returned in output dict and available to GUI/database
- Visibility hard gate, bone-length constraints, per-point confidence scoring

**Implementation details:**
- `_occlusion_state`, `_occlusion_last_position`, `_occlusion_frames_hidden` tracked per landmark
- Predicted positions have decaying visibility (×0.9 per frame) and are flagged as low-reliability
- After 15 frames of prediction, landmark transitions to `OCCLUDED` (dropped)

---

### 22.6 Dataset Pipeline

**Status: ✅ Implemented**

**What exists:**
- Session management: `start_recording()` / `stop_recording()` with UUID and dynamic per-session tables
- `save_synced_frame()`: PC1 data, PC2 data, 3D pose, kinematics, confidence per frame
- `export_latest_session_csv()`: CSV export with all metrics
- **NEW:** `export_session_json()`: Full structured JSON export of any session
- **NEW:** `archive_session()`: ZIP bundle containing metadata.json, data.csv, data.json, and raw frames
- **NEW:** `save_raw_frame()`: JPEG archival to `results/raw_frames/{session}/` directory
- **NEW:** `get_session_list()`: List all sessions with metadata
- `report_generator.py`: 20+ plots (time series, FFT, kinematics, symmetry, correlation)
- SQLite and PostgreSQL backends
- Config: `SAVE_RAW_FRAMES`, `SESSION_EXPORT_FORMAT`

**Implementation details:**
- Raw frame saving wired into GUI video_loop when `SAVE_RAW_FRAMES=True`
- Archive creates `{table_name}_archive.zip` in `results/` directory
- JSON export includes full schema columns, frame data, and session metadata

---

### 22.7 Replacing World Landmarks

**Status: ✅ Implemented**

**What exists:**
- Triangulated 3D points are primary source with `method: 'triangulated'`
- Monocular back-projection as Tier 2 with `method: 'monocular_{camera_id}'`
- **NEW:** MediaPipe `pose_world_landmarks` as Tier 3 fallback with `method: 'world_landmarks'`
- **NEW:** `_extract_world_landmarks()` helper extracts world landmarks from both local and remote frames
- Quality-based fallback chain:
  ```
  IF ≥2 cameras see landmark AND triangulation succeeds → USE triangulated 3D
  ELIF 1 camera sees landmark AND monocular fallback enabled → USE monocular back-projection
  ELSE → USE MediaPipe world_landmarks (hip-relative, averaged across cameras)
  ```
- World axis convention transform applied uniformly to all sources
- Low visibility penalty (×0.5) applied to Tier 3 landmarks

---

### 22.8 Kinematics Engine Upgrade

**What exists:**
- `compute_velocity()`: 3D finite-difference with `MAX_LINEAR_VELOCITY` clamping
- `compute_acceleration()`: 3D finite-difference
- `compute_joint_angle()`: 3-point angle for 8 joints (elbows, knees, shoulders, hips)
- `compute_angular_velocity()`: angle derivative
- `process_frame()`: full-frame processing with state tracking
- `export_frame_data()`: flat dictionary for DB storage

**What is missing:**
- No inverse kinematics (IK) solver
- No multi-source consistency check (comparing kinematics from different camera subsets)
- No jerk computation (derivative of acceleration)
- No segment/limb velocity (only joint-level)
- No kinematic chain model

**Target additions:**
- Jerk = $\frac{d\vec{a}}{dt}$ — useful for smoothness metrics in clinical gait analysis
- Segment velocities for limb endpoint speed (e.g., hand speed relative to torso)
- Multi-source consistency: compute kinematics from Camera A alone and Camera B alone, compare with fused result

---

### 22.9 Uncertainty + Error Model

**Status: ✅ Implemented**

**What exists:**
- Per-point confidence score: $\text{conf} = (w_r \cdot e^{-\text{err}/5} + w_v \cdot \bar{v}) \cdot \min(1, n/4)$
- Reprojection error per triangulated point
- Calibration-level `reprojection_error` per camera
- Low-reliability landmark flagging (below `KINEMATICS_MIN_POINT_CONFIDENCE`)
- `confidence_data` stored per frame in database
- **NEW:** Inter-camera disagreement metric — for each triangulated landmark, computes individual monocular 3D estimates per camera, then measures $\text{uncertainty} = \max\|\mathbf{p}_A - \mathbf{p}_B\|$ across all camera pairs
- **NEW:** `uncertainty` dict returned in `get_synced_3d_pose()` output (landmark_id → meters)
- **NEW:** `_landmark_disagreements` tracked in coordinator state

**Implementation details:**
- Disagreement computed before DLT fusion for honest uncertainty estimate
- Max pairwise distance used (conservative — captures worst-case disagreement)
- Only computed when ≥2 cameras observe the landmark

---

### 22.10 Dashboard Help System

**Status: ✅ Implemented**

**What exists:**
- Comprehensive external DOCUMENTATION.md (2500+ lines with equations)
- Glossary section (§21)
- **NEW:** "❓ Metric Help" button in GUI → scrollable help dialog with:
  - Joint angle definitions (elbow, knee, shoulder) with landmark triplets
  - Formula: $\theta = \arccos\left(\frac{A \cdot B}{|A| \cdot |B|}\right)$
  - Velocity and acceleration equations
  - Bone length normalization explanation
  - Confidence/visibility scoring formula
  - 3D reconstruction method glossary (triangulated, monocular, world_landmarks, predicted)
  - Uncertainty (disagreement) definition
  - Occlusion state descriptions
  - 1-Euro filter parameter explanation
- Help dialog uses color-coded tags: cyan headers, orange subheadings, green formulas

---

### 22.11 Latency Visibility

**Status: ✅ Implemented**

**What exists:**
- FPS counter in GUI status bar
- FPS state variable in frontend
- **NEW:** Pipeline latency instrumentation in `main_gui.py` video_loop:
  - `capture`: time to read frame from camera
  - `detection`: time for MediaPipe inference (pose + face + hand)
  - `sync`: time for `get_synchronized_batch()` frame matching
  - `triangulation`: time for `get_synced_3d_pose()` (DLT + fallbacks + kinematics)
  - `total`: end-to-end frame processing time
- **NEW:** `record_latency()` and `get_latency_stats()` in `MasterCoordinator` — computes mean, p95, max per stage
- **NEW:** `_log_latency_summary()` prints console summary every 100 frames
- **NEW:** Latency panel in GUI (master mode) showing 6 stages with real-time mean values
- **NEW:** `_update_latency_panel()` refreshes GUI labels every ~1 second
- Config: `ENABLE_LATENCY_TRACKING`, `LATENCY_LOG_INTERVAL`

---

### 22.12 Camera Placement Visualization

**Status: ✅ Implemented**

**What exists:**
- **NEW:** `src/camera_setup_viewer.py` — standalone module for 3D camera visualization
- `CameraSetupViewer.show()` renders:
  - Camera positions as scatter points with labels
  - Optical axes as quiver arrows
  - Field-of-view pyramids computed from intrinsic matrix ($f_x$, $f_y$, image size)
  - World origin with XYZ coordinate axes
  - Subject standing zone (dashed circle at z=2m)
  - Auto-scaling based on camera positions
- `get_camera_info()` returns per-camera summary: position, Euler angles, FOV, baselines
- **NEW:** "📷 Camera Setup Viewer" button in GUI (master mode)
- Prints camera positions, FOV, and inter-camera baselines to console

---

### 22.13 Recalibration

**Status: ✅ Implemented**

**What exists:**
- **NEW:** "🔧 Recalibrate" button in GUI (master mode) → launches guided workflow
- **NEW:** Interactive chessboard capture: SPACE to capture pairs, ESC to finish
- **NEW:** Real-time chessboard corner detection and display on both camera feeds
- **NEW:** Full stereo calibration pipeline: `cv2.calibrateCamera` (both) → `cv2.stereoCalibrate`
- **NEW:** Saves calibration to timestamped JSON file
- **NEW:** Auto-reloads calibration into live triangulation pipeline without restart
- Requires minimum 5 captures; 15-20 recommended for quality
- Displays stereo RMS error on completion
- Pose corrector auto-calibrates bone lengths from first 30 frames (complementary)

---

### 22.14 GPU Scheduling

**Status: ✅ Implemented (profiling + scheduling hints)**

**What exists:**
- CUDA / MPS / CPU device auto-detection
- `PREFER_GPU_DELEGATE = True` for MediaPipe Metal delegate
- DLT triangulation uses PyTorch on GPU when `CUDA_ENABLED`
- **NEW:** Per-model inference timing in `detector.py`: pose, face, and hand models individually timed
- **NEW:** `ENABLE_GPU_PROFILING` config flag — when True, accumulates timing stats
- **NEW:** Periodic GPU profile logging every 300 frames (pose/face/hand ms + delegate type)
- **NEW:** `get_gpu_profile()` API returns recent timing averages and delegate info
- **NEW:** `inference_ms` dict included in detector `process()` return value
- **NEW:** `camera_server.py` includes per-frame `gpu_compute` payload in network frame messages
- **NEW:** `master_coordinator.py` stores per-camera GPU reports and computes `compute_hints` from latency bottlenecks
- **NEW:** Master feedback packets now include `compute_hints` (action + target stage + per-camera timing summary)
- **NEW:** Camera feedback loop ingests `compute_hints` for adaptive scheduling decisions/logging

**Remaining for future:**
- Multi-GPU load balancing (requires N>1 GPUs)
- Memory-aware batching with dynamic CPU fallback under memory pressure

---

### Implementation Priority Guide

All 14 feature areas are now implemented. Remaining enhancement opportunities:

| Area | Potential Enhancement |
|------|----------------------|
| §22.3 Sync | NTP pre-check at startup, PTP support, hardware trigger/genlock |
| §22.8 Kinematics | Jerk computation, inverse kinematics, segment velocities |
| §22.9 Uncertainty | Full covariance propagation 2D→3D, confidence intervals in frontend |
| §22.14 GPU | Multi-GPU load balancing, memory-aware dynamic CPU fallback |

### 22.15 Standalone Module Wiring (Runtime Integration)

**Status: ✅ Integrated into runtime pipeline**

**What is now wired in `MasterCoordinator`:**
- `OcclusionFusionEngine` executes each synced frame and returns fused method/state counts.
- `UncertaintyEstimator` computes per-landmark uncertainty breakdowns.
- `ErrorMetricsCalculator` accumulates frame-level uncertainty summaries and quality grade.
- `AdvancedKinematics` computes extended anatomical kinematics and exports flat metrics.
- `DashboardMonitor` receives live latency-stage samples and exposes consolidated status.

**New fields now available in synced pose output payload:**
- `fusion`
- `uncertainty_detailed`
- `uncertainty_summary`
- `advanced_kinematics`
- `dashboard_status`

These are additive outputs; existing keys (`pose_3d`, `kinematics_3d`, `uncertainty`, `occlusion_states`) remain unchanged for backward compatibility.

---

## 23. Calibration, Quality, and Metric Policy (Normative)

This section is the implementation-aligned source of truth for calibration, reprojection filtering, confidence, and live metric semantics. It is intentionally explicit to remove ambiguity.

### 23.1 Camera Intrinsic Calibration Procedure

- Implementation: `src/stereo_calibration.py::calibrate_intrinsic`.
- Checkerboard defaults: `9x6` internal corners, `0.025 m` square size.
- Corner pipeline: `cv2.findChessboardCorners` → `cv2.cornerSubPix`.
- Solve intrinsics: `cv2.calibrateCamera`.
- Persist per-camera reprojection RMS in calibration data.
- Practical minimum in code path: at least 10 valid checkerboard images.

### 23.2 Intrinsic Parameter Storage Format

Calibration JSON (per camera) includes:
- `intrinsic_matrix` (3x3)
- `distortion_coeffs` (OpenCV vector, typically `[k1, k2, p1, p2, k3]`)
- `rotation` (3x3)
- `translation` (3x1)
- `image_size` (`[width, height]`)
- `reprojection_error` (pixels)

### 23.3 Distortion Compensation (Radial/Tangential)

- Distortion coefficients are actively used in triangulation preprocessing.
- Default triangulation uses `undistort=True`.
- Undistortion is applied with `cv2.undistortPoints(...)` using `intrinsic_matrix` + `distortion_coeffs`.
- This compensates both radial and tangential lens distortion terms represented in the OpenCV coefficient model.

### 23.4 Stereo Extrinsic Calibration

- Implementation: `src/stereo_calibration.py::calibrate_stereo`.
- Intrinsics are fixed during stereo solve (`cv2.CALIB_FIX_INTRINSIC`).
- Camera-1 (`local_cam`) is the world anchor (`R=I`, `t=0`).
- Camera-2 (`cam_0`) extrinsics are solved relative to camera-1.
- Projection matrix used by triangulation is $P = K[R|t]$.

### 23.5 Defined World Coordinate Frame

- Triangulation world frame is camera-1 anchored.
- Downstream, coordinator applies a world-axis convention transform for standardized application coordinates.

### 23.6 Sync Tolerance Specification (Numerical)

From `config.py`:
- `SYNC_TIME_THRESHOLD_MS = 100.0`
- `FRAME_BUFFER_SIZE = 10`
- `STALE_FRAME_TIMEOUT_MS = 2000`

Interpretation:
- Frames are considered sync-compatible if timestamp spread is within 100 ms.
- Stale frames older than 2 s against newest global frame are evicted.

### 23.7 Reprojection Threshold Policy (Unified)

From `config.py` and `src/triangulation.py`:
- `REPROJECTION_ERROR_THRESHOLD = 15.0 px`

When threshold is exceeded:
1. Triangulation candidate is discarded (`None`).
2. Coordinator applies fallback tiers in order:
       - Tier 2: monocular fallback (if enabled and best-camera visibility > 0.5)
       - Tier 3: averaged world landmarks fallback (penalized confidence)
3. If still missing: hold/predict last known landmark up to 15 frames, then mark occluded.

So policy is explicit: **discard triangulated point first, then fallback, then temporary freeze/predict**.

### 23.8 One-Euro Filter Parameters and Placement

From `config.py`:
- `FILTER_MIN_CUTOFF = 1.0 Hz`
- `FILTER_BETA = 0.005`
- `FILTER_D_CUTOFF = 1.0 Hz`

Filter placement in pipeline:
1. **Before triangulation:** per-camera landmark smoothing in `PoseCorrector`.
2. Triangulation + 3-tier fusion.
3. **After triangulation/fusion:** optional 3D One-Euro filtering in `MasterCoordinator` (`_filter_pose_3d`) when enabled.

### 23.9 Bone Length Policy

- Bone lengths are learned during calibration window (`CALIBRATION_FRAMES`, default 30).
- After calibration, reference lengths are fixed and enforced with visibility-weighted blending.
- They are not recomputed every frame during normal tracking.

### 23.10 Depth Resolution Derivation

First-order stereo depth sensitivity:

$$
\Delta Z \approx \frac{Z^2}{fB}\Delta d
$$

With approximate defaults ($f=1280$, $B=1.0$, $Z=2.0$, $\Delta d=1.0$):

$$
\Delta Z \approx \frac{4}{1280} = 0.003125\,m \approx 3.1\,mm
$$

### 23.11 Expected Measurement Error Range

- Dual-camera metric 3D target under good setup: approximately ±1–2 cm.
- Validation checks include distance-error target ≤ 0.05 m and static jitter target ≤ 0.005 m standard deviation.
- Errors increase with poor calibration, low light, motion blur, narrow baseline, and heavy occlusion.

### 23.12 Dashboard Metric Definition Appendix

#### Reprojection Error

$$
e_{reproj} = \frac{1}{N}\sum_{i=1}^{N}\|\hat{u}_i-u_i\|_2
$$

Hard reject threshold: `15.0 px`.

#### Confidence

Implemented confidence model:

$$
C = \left(w_r e^{-e_{reproj}/5} + w_v\bar{v}\right)\cdot\min(1, N/4)
$$

With `w_r = 0.4`, `w_v = 0.6`.

#### Residual Error

In runtime payloads, residual error is represented as the same reprojection residual field (`reproj_error`).

#### Angle Definition

$$
	heta = \arccos\left(\frac{\vec{BA}\cdot\vec{BC}}{\|\vec{BA}\|\|\vec{BC}\|}\right)
$$

Core reported joint angles: elbows, knees, shoulders, hips (left/right).

### 23.13 Live Display Clarification (Computed vs Surfaced)

- GUI currently displays FPS, latency, and kinematic metrics.
- Reprojection/confidence/residual are computed and propagated in pipeline structures and quality feedback messages.
- In master mode, GUI now includes a live **Reconstruction Quality** panel with:
       - Mean triangulated reprojection error (`Reproj`, px)
       - Mean triangulated confidence (`Confidence`, unitless)
       - Residual error (`Residual`, equivalent to reprojection residual, px)
       - Mean inter-camera uncertainty (`Uncertainty`, m)

For a concise architecture-only version of the same policies, see `SYSTEM_ARCHITECTURE.md`.
