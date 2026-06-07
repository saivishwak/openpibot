# VR Teleop

Drive one or both SO-101 arms with a Meta Quest 3. The production headset path
is the native Unity/OpenXR Quest app in `apps/quest-xlerobot`.

For the full end-to-end runtime path, see [architecture.md](architecture.md).

## Controllers

| Button | What it does |
|---|---|
| **Grip** (side, middle finger) | **Hold to drive**, **first press = anchor for teleop**. Releasing stops motion. |
| **Trigger** (front, index finger) | Close gripper while held. Released = open. |
| **Both triggers together** | Hold to temporarily show Quest passthrough. Release either trigger to return to the robot-camera scene. Trigger/gripper values are suppressed while this chord is held. |
| **A** (right) / **X** (left) | Confirm the headset is facing the workspace, then engage/switch the active arm. |
| **Y** (left) | Toggle dual mode. Press Y by itself; X+Y chords are ignored so engage and dual mode do not fight. |
| **B** (right) | Toggle dataset recording. |
| Thumbstick / menu | Unused. |

## Per-session flow

1. **Set a pairing token** before starting the backend:
   `XLE_QUEST_PAIRING_TOKEN=<shared-secret> uv run openpibot run --log-file .openpibot/logs/server.log`.
2. **Open the dashboard**: `http://<workstation>:5000`. Click *Connect* on each arm you want to use.
3. **Launch the OpenPiBot Quest app** on the headset. In the app, connect to the
   tokenized `ws://<workstation>:5000/api/vr/quest/ws?token=...` endpoint shown
   on the *VR Operator* page. Enter the same pairing token in the dashboard and
   Unity app; the public status API does not expose it.
4. **Calibrate** if you haven't or if you're standing somewhere new — see
   [calibration.md](calibration.md). Once-per-setup; the calibration is saved.
5. In the headset, use the operator panel like Reachy's flow:
   **Connection** -> **Mirror/ready** -> **Teleop** -> **Suspension**.
   Hold **A** while facing the workspace to confirm Ready/recenter.
6. Press **A**/**X** or toggle Engage in the dashboard. The headset panel shows whether each arm is connected, torqued, anchored, wrist-aligned, and recording-ready.
7. **Squeeze grip** on a controller to anchor that arm's EE pose. The panel shows "anchored"; releasing grip stops motion.
8. **Hold grip + move your hand**. The arm follows. Pull trigger to close the gripper.
9. Press **B** to request dataset recording. The backend still enforces calibration/verification blockers, so the headset can request recording but cannot bypass safety or dataset-quality checks.

## Headset Video

The native path uses backend-managed GStreamer H.264/RTP pipelines for low-latency
camera delivery. Install `gst-launch-1.0` on the workstation and configure the
`head`, `left_wrist`, and `right_wrist` camera roles before expecting video to be
ready. Quest video, dashboard preview, LeRobot recording, and inference all read
from the backend `CameraService`; the Quest video bridge must not open `/dev/video*`
directly or suspend the recording cameras.

The headset default view is robot-camera-only on a dark operator scene. Quest
passthrough is disabled so dataset operators focus on what the robot sees. Hold
both index triggers together only when you need to check the real surroundings;
release either trigger to return to the previous robot-camera screen.

On the *VR Operator* page, enter the Quest headset IP and click **Start video**.
The backend starts one GStreamer process per camera role and shows the UDP ports,
PIDs, and any process errors. Click **Stop video** before changing cameras or
shutting down teleop.

## Bimanual

Both arms can be connected and torqued simultaneously.

### Single-arm mode

The *Active arm* segmented control on the Engagement card switches which arm VR drives. The controller buttons do the same from inside VR:

- **A** on the right controller engages/switches to the right arm.
- **X** on the left controller engages/switches to the left arm.
- Pressing the active arm's engage button again disengages.

### Dual mode

Press **Y** on the **left** controller to toggle dual mode. In dual mode, the backend runs the same per-arm VR path for both sides:

- Hold **left grip** to drive the left arm.
- Hold **right grip** to drive the right arm.
- Hold **both grips** to drive both arms together.

Each arm still needs its own calibration and grip-anchor. If one side is not anchored yet, squeeze that side's grip once before expecting it to move. Press **Y** again to turn dual mode off/disengage. Press **Y alone**; if **X** is held at the same time, the chord is ignored to avoid accidentally switching single-arm engage while toggling dual mode.

## Speed slider

Default is **0.5** for fine control on the SO-101's small workspace. Raise toward 1.0 for faster true-scale hand-to-EE motion. The per-tick joint caps are the hard safety limit underneath.

## Motion mapping

Grip press anchors two things for that arm:

- The current robot gripper pose.
- The current Quest controller pose and controller-to-gripper rotation.

While grip is held, the native Quest app streams Unity/OpenXR controller poses to
the backend. The backend converts those packets into reset-relative controller
displacements, maps them through the saved VR-to-robot frame, and caps the
end-effector target step before IK. This keeps slow hand motion smooth and
rejects one-frame tracking spikes. The controller's rotation is also mapped
through the reset-time controller-to-EE alignment so wrist intent does not
depend on the exact grip angle at anchor time.

The native adapter converts Unity/OpenXR controller coordinates into the
backend operator frame before calibration. The default frame is
`quest_operator_frame`:

```text
operator = (unity.z, -unity.x, unity.y)
```

The same basis is applied to controller rotations by matrix conjugation. Older
calibration files that say `unity_reachy` are treated as the same
`quest_operator_frame`; stale WebXR/legacy frame calibrations must be redone.

Useful tuning keys live in `config/xlerobot.yaml`:

```yaml
vr:
  kp: 0.75
  pos_ema_alpha: 0.22
  ori_ema_alpha: 0.25
  pos_deadzone_m: 0.002
  rot_deadzone_deg: 0.7
  max_ee_step_m: 0.003
  wrist_delta_limit_deg: 8.0
  joint_deg_caps:
    shoulder_pan: 5.0
    shoulder_lift: 5.0
    elbow_flex: 5.0
    wrist_flex: 6.0
    wrist_roll: 10.0
    gripper: 15.0
  joint_command_deadband_deg:
    shoulder_pan: 0.18
    shoulder_lift: 0.18
    elbow_flex: 0.18
    wrist_flex: 0.25
    wrist_roll: 0.25
    gripper: 0.0
  joint_command_filter_weights: [0.4, 0.3, 0.2, 0.1]
```

`wrist_flex`, `wrist_roll`, and `gripper` bypass the final command EMA/filter so
wrist and gripper intent remains responsive. Translation joints use the command
filter to reduce jitter. Recording stores the final commanded joint target after
these same caps/deadbands/filters, so dataset labels match what the robot was
asked to do.

## Safety

- **EMERGENCY STOP** button (top of page) — instantly disables torque on both arms. The robot freezes wherever it is.
- **Watchdog** — if VR goals stop arriving (controller down, Wi-Fi blip), the drive loop stops within 0.3 s.
- **Per-tick joint caps** — max joint speeds capped (e.g. shoulder_pan 5°/tick at 30 Hz = 150°/s). Independent of the speed slider.
- **No autonomous motion**, ever. Disconnect = torque off, no homing. The only motion the app initiates is the user-clicked *Go to Home*.

## Native Quest App

The app source lives in `apps/quest-xlerobot`. It is a first-party Quest 3 app
that adapts the Reachy2 Unity mirror/ready flow to XLeRobot:

- Controller snapshots stream to `/api/vr/quest/ws`.
- `XLeRobotStateClient` polls backend operator status for in-headset UI.
- `XLeRobotOperatorFlowManager` maps backend stages to connection, mirror,
  teleop, and suspension UI.
- `XLeRobotVideoBridgeClient` starts/stops backend camera pipelines, while
  `XLeRobotVideoSurfaceBinder` and `XLeRobotGStreamerTextureAdapter` bind
  received textures to headset surfaces.
- Bridge status is available at `/api/vr/quest/status`.
- GStreamer video runtime status is available at `/api/vr/quest/video/status`.
  Start/stop endpoints require the Quest pairing token.
- The backend remains responsible for safety, IK, calibration persistence,
  recording blockers, and dataset writes.

See [quest3_build_and_sideload.md](quest3_build_and_sideload.md) for Unity
build, APK, and Quest 3 sideload instructions.

See [quest_parity.md](quest_parity.md) for the full parity acceptance checklist.

See [troubleshooting.md](troubleshooting.md) if motion feels wrong or doesn't happen.
