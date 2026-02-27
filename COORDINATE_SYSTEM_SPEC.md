# Coordinate System Specification

Date: 2026-02-26

## 1) Global World Frame (W)

The system uses one locked right-handed world frame:

- +X: Right
- +Y: Up
- +Z: Forward (away from the front camera)

World origin is Camera A optical center (front camera).

## 2) Camera Frame (OpenCV)

Per-camera coordinates follow OpenCV:

- +X: right in image
- +Y: down in image
- +Z: forward from lens

Stereo calibration computes `R` and `T` between cameras. Camera A is treated as world reference.

## 3) Body Segment Conventions

Joint points: `P_joint = (X, Y, Z)` in meters.

Canonical segment vectors:

- Upper Arm (Right): `Elbow - Shoulder`
- Forearm (Right): `Wrist - Elbow`
- Trunk Axis: `MidShoulder - MidHip`
- Pelvic Axis: `RightHip - LeftHip`

MediaPipe pose indices are used as canonical joint IDs.

## 4) Angle Conventions

- Unit: Degrees
- Default range: 0° to 180° (unsigned)

Angle formula:

`theta = arccos( (v1·v2) / (|v1||v2|) )`

Right elbow (flexion) convention:

- `v1 = Shoulder - Elbow`
- `v2 = Wrist - Elbow`

Interpretation:

- 180° = fully extended
- smaller angle = increased flexion

## 5) Pipeline (Frame to Dashboard)

1. Capture RGB on each camera
2. Run MediaPipe pose and output 2D + confidence
3. Send packet: `{camera_id, timestamp, landmarks[33] = (x, y, conf)}`
4. Master syncs frames by timestamp tolerance (default ±20 ms)
5. Convert normalized coordinates to pixel coordinates and undistort
6. Triangulate each joint using projection matrices `P1`, `P2`
7. Convert homogeneous to Euclidean coordinates
8. Apply confidence gating and mark low-reliability joints
9. Filter 3D positions (1-Euro) before derivative calculations
10. Compute kinematics (velocity, acceleration, joint angles, angular velocity)
11. Render dashboard and log synchronized outputs

## 6) Data Logging Requirements

Per frame, log at minimum:

- timestamp
- joint_3d[33]
- velocity / acceleration
- joint angles
- confidence and reliability flags

## 7) Validation Protocol

1. Static jitter test: 20 s standing still, target low 3D variance
2. Known angle test: compare to goniometer, target low angle error
3. Known distance test: compare triangulated distance vs known metric ground truth

## 8) Known Error Sources

Primary controllable error sources:

1. Calibration error
2. Timestamp mismatch
3. Low-confidence landmarks
4. Motion blur
5. Small camera baseline
