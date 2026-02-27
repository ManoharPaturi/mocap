# Motion Capture System Architecture (Normative Spec)

**Version:** VS5  
**Last Updated:** 27 February 2026

---

## 1) Camera Intrinsic Calibration Procedure

Implementation source: `src/stereo_calibration.py`.

1. Capture checkerboard images per camera (default internal corners: `9x6`, square size: `0.025 m`).
2. Run corner detection: `cv2.findChessboardCorners`.
3. Refine corners: `cv2.cornerSubPix` with criteria `(EPS + MAX_ITER, 30, 0.001)`.
4. Solve intrinsics: `cv2.calibrateCamera`.
5. Compute per-camera reprojection RMS and store in calibration object.

Minimum accepted intrinsic set in code path: at least 10 valid checkerboard detections per camera.

---

## 2) Intrinsic Parameter Storage Format

Calibration is persisted as JSON using `StereoCalibration.save_calibration(...)`.

Per camera JSON fields:
- `intrinsic_matrix`: `3x3`, OpenCV camera matrix `K`
- `distortion_coeffs`: OpenCV distortion vector (typically `[k1, k2, p1, p2, k3]`)
- `rotation`: `3x3` extrinsic rotation matrix
- `translation`: `3x1` extrinsic translation vector
- `image_size`: `[width, height]`
- `reprojection_error`: scalar RMS error in pixels

---

## 3) Distortion Model and Compensation

Yes, distortion coefficients are used in the triangulation path.

- Triangulation default behavior is `undistort=True`.
- For each 2D observation, points are passed through `cv2.undistortPoints(...)` using camera `K` and `distortion_coeffs`.
- This compensates radial and tangential distortion terms represented in the OpenCV coefficient vector.

---

## 4) Stereo Extrinsic Calibration

Implementation source: `src/stereo_calibration.py::calibrate_stereo`.

1. Ensure intrinsics exist for both cameras.
2. Re-detect checkerboard pairs visible in both views.
3. Run `cv2.stereoCalibrate` with `cv2.CALIB_FIX_INTRINSIC`.
4. Set world frame anchor:
   - camera 1 (`local_cam`) is world origin with `R = I`, `t = 0`
   - camera 2 (`cam_0`) uses solved `R`, `T` relative to camera 1

Projection matrix per camera is `P = K [R|t]`.

---

## 5) Defined World Coordinate Frame

World frame for triangulation is camera-1-anchored:
- Origin at `local_cam` optical center
- Axes defined by `local_cam` extrinsic identity frame

After fusion, an additional axis-convention transform is applied in coordinator (`_apply_world_axis_convention`) so downstream consumers get a consistent project convention.

---

## 6) Sync Tolerance Specification (Numerical)

From `config.py`:
- `SYNC_TIME_THRESHOLD_MS = 100.0`
- `FRAME_BUFFER_SIZE = 10`
- `STALE_FRAME_TIMEOUT_MS = 2000`

Interpretation:
- Synchronized batches accept frame timestamp spreads up to 100 ms.
- Frames older than 2 s relative to newest multi-camera data are evicted.

---

## 7) Reprojection Threshold Policy

Unified policy in code:
- Threshold: `REPROJECTION_ERROR_THRESHOLD = 15.0 px`.
- Enforced in `Triangulator.triangulate_point(...)`.

If threshold is exceeded:
1. **Discard** triangulated candidate (`return None`).
2. Coordinator fallback sequence:
   - Tier 2: monocular fallback (if enabled and best camera visibility > 0.5)
   - Tier 3: averaged MediaPipe world landmarks (visibility penalty applied)
   - If no observation: temporary hold/prediction from last valid pose (up to 15 frames), then mark occluded.

So behavior is explicit: **discard triangulated sample first, then fallback, then temporary freeze/predict**.

---

## 8) Filtering Order and Parameters

### One-Euro numeric values

From `config.py`:
- `FILTER_MIN_CUTOFF = 1.0 Hz`
- `FILTER_BETA = 0.005`
- `FILTER_D_CUTOFF = 1.0 Hz`

### Execution order

1. **Before triangulation**: per-camera landmark smoothing in `PoseCorrector` (normalized and world landmarks).
2. Triangulation + fusion.
3. **After triangulation/fusion**: optional 3D One-Euro smoothing in coordinator (`_filter_pose_3d`) when enabled.

---

## 9) Bone Length Policy

Implementation source: `src/pose_corrector.py`.

- Bone lengths are learned during an initial calibration window (`CALIBRATION_FRAMES`, default 30).
- After calibration completes, reference lengths are fixed and enforced per frame via visibility-weighted constraints.
- They are **not recomputed every frame** during normal tracking.

---

## 10) Depth Resolution Derivation

For stereo triangulation, first-order depth sensitivity is:

$$
\Delta Z \approx \frac{Z^2}{fB} \Delta d
$$

Where:
- $Z$: depth (m)
- $f$: focal length in pixels
- $B$: baseline (m)
- $\Delta d$: disparity error (px)

Example with approximate defaults ($f=1280$, $B=1.0$, $Z=2.0$, $\Delta d=1.0$):

$$
\Delta Z \approx \frac{4}{1280} = 0.003125\,m \approx 3.1\,mm
$$

---

## 11) Expected Measurement Error Range

Operational expectations (already reflected in project docs and tests):
- Dual-camera metric 3D: typically around `±1–2 cm` in good calibration/lighting.
- Validation target: distance error within `0.05 m` in `tests/test_known_distance.py`.
- Static jitter target: standard deviation `<= 0.005 m` in `tests/test_static_jitter.py`.

Error increases with poor calibration, motion blur, low visibility, weak baseline geometry, and occlusion.

---

## 12) Dashboard Metric Definition Appendix

### 12.1 Reprojection Error

Mean pixel reprojection residual of triangulated 3D point across used cameras.

$$
e_{reproj} = \frac{1}{N}\sum_{i=1}^{N}\|\hat{u}_i - u_i\|_2
$$

Policy threshold: `15.0 px` (hard reject for triangulation result).

### 12.2 Confidence

Triangulation confidence combines reprojection quality and landmark visibility:

$$
C = \left(w_r e^{-e_{reproj}/5} + w_v \bar{v}\right)\cdot \min(1, N/4)
$$

With code-configured weights:
- `CONFIDENCE_WEIGHT_REPROJ = 0.4`
- `CONFIDENCE_WEIGHT_VISIBILITY = 0.6`

### 12.3 Residual Error

Residual error in this project is the same quantity reported as reprojection error (`reproj_error`) per landmark.

### 12.4 Angle Definitions

Joint angle definition (three-point angle at vertex):

$$
\theta = \arccos\left(\frac{\vec{BA}\cdot\vec{BC}}{\|\vec{BA}\|\,\|\vec{BC}\|}\right)
$$

Core angles in runtime metrics:
- left/right elbow
- left/right knee
- left/right shoulder
- left/right hip

---

## 13) Live Display Coverage (Current State)

Current GUI panels:
- Shows FPS and pipeline latency stages.
- Shows joint-angle and selected length metrics.

Current quality diagnostics availability:
- Reprojection error and confidence exist in pipeline data structures and quality feedback messages.
- Dedicated live GUI widgets for reprojection/confidence/residual are not yet wired in main dashboard.

This section is intentionally explicit so operational users know what is computed vs what is currently surfaced on-screen.
