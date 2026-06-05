# Quest Teleop Parity Contract

This document defines what "Reachy-style Quest teleop parity" means for
XLeRobot. The Reachy2 Unity project in `reference/Reachy2Teleoperation` is the
operator-experience reference. XLeRobot keeps its own backend authority for
SO-101 IK, safety limits, calibration persistence, recording blockers, and
dataset writes.

## Current Parity Target

The production path is the native Quest 3 Unity/OpenXR app in
`apps/quest-xlerobot`.

Full parity requires all of these to work together:

- Quest app launches as a complete Unity Android/OpenXR project with scenes,
  prefabs, XR origin, controller tracking, operator UI, and video display.
- Operator flow matches Reachy's workflow: connection, mirror/ready, teleop,
  and suspension/error handling.
- Controller controls are available in headset: grip anchors/drives, trigger
  controls gripper, A/X engages or switches active arm, Y toggles dual mode, and
  B requests recording.
- Unity/OpenXR controller positions and rotations are converted into the backend
  VR frame before calibration and control.
- Head and wrist camera feeds render in the Quest app at usable latency.
- Dashboard remains the setup and safety authority for robot connection,
  pairing, video health, recording blockers, and emergency stop.
- Recording remains blocked until calibration and robot verification are good
  for every connected arm.

## Acceptance Criteria

### Movement Translation Contract

Reachy2 converts headset-relative controller pose inside Unity and sends an
absolute `ArmCartesianGoal` protobuf to the robot. XLeRobot intentionally does
not send that command type. The Quest app sends absolute Unity/OpenXR controller
poses and button states; the workstation converts them into reset-relative VR
goals, applies calibration, and owns SO-101 IK, limits, robot verification, and
LeRobot recording.

The required equivalence is behavioral:

- Reachy-style controller forward/left/up intent must become robot
  forward/left/up end-effector intent.
- Dropped or invalid Quest packets must not corrupt the reset-relative
  displacement.
- Rotation basis conversion must apply to controller quaternions before wrist
  calibration and controller-to-EE alignment.

### Quest App

- The app opens from `apps/quest-xlerobot` in Unity `6000.4.10f1` or the Unity
  6 version recorded in the project.
- Meta Quest build profile or Android/OpenXR fallback settings are present, and
  Quest 3 can run the APK.
- At least one production scene contains an XR origin, controller tracking,
  operator panels, status displays, and video surfaces.
- Headset UI exposes connection, mirror/ready, teleop, suspension, calibration
  readiness, and recording readiness.

### Control

- A paired Quest app connects to `/api/vr/quest/ws`.
- Both controllers stream poses and buttons at a stable rate.
- Grip press anchors the corresponding arm without moving the robot until
  teleop is engaged.
- Grip-held translation maps forward/back, left/right, and up/down correctly for
  both arms.
- Wrist pitch and roll follow controller orientation after anchoring.
- A/X, Y, and B button edges behave the same from headset and dashboard status.

### Calibration And Verification

- Unity/OpenXR position and rotation frames are converted consistently.
- Robot verification captures expected VR and robot deltas for each connected
  arm.
- Low-scale tests must pass before recording is allowed.

### Video

- The selected transport is explicit: Reachy-compatible GStreamer/WebRTC, or a
  documented fallback with a Unity receiver.
- `head`, `left_wrist`, and `right_wrist` camera roles can stream to the
  headset.
- The Quest app renders video surfaces, and dashboard status reflects actual
  running/receiving state rather than descriptors alone.
- Video start/stop and reconnect failures are visible and safe.

### Security And Safety

- A random LAN client cannot fetch the pairing token and control the robot.
- Control and video mutation endpoints require pairing.
- Public endpoint generation does not assume hardcoded port `5000`.
- Emergency stop immediately disables torque on both arms.
- Watchdog behavior stops motor writes when Quest packets are stale.

### Dataset Quality

- Recording cannot start while calibration, robot verification, low-scale test,
  cameras, or task text are incomplete.
- Recorded episodes include joint actions, joint observations, and configured
  camera observations at the expected dataset FPS.
- A short manually inspected episode has controller intent matching robot motion.

## Verification Matrix

Run these before calling parity complete:

- `uv run pytest tests`
- `npm --prefix dashboard/frontend run build`
- Unity compile check for `apps/quest-xlerobot`
- Android APK build for Quest 3
- Quest connects using pairing and shows live backend status
- Quest displays all configured camera streams
- Robot direction verification for both arms
- Wrist pitch/roll verification for both arms
- Emergency stop test
- Recording blocker and successful recording test

## Known Non-Parity Areas

Reachy-specific features are intentionally not required unless needed for app
stability: mobile base, antennas, emotions, lidar, neck control, and full Reachy
protobuf state emulation.
