# VR Teleop

Drive one or both SO-101 arms with a Meta Quest 3.

## Controllers

| Button | What it does |
|---|---|
| **Grip** (side, middle finger) | **Hold to drive**, **first press = anchor for teleop**. Releasing stops motion. |
| **Trigger** (front, index finger) | Close gripper while held. Released = open. |
| **A** (right) / **X** (left) | Confirm the headset is facing the workspace, then engage/switch the active arm. |
| **Y** (left) | Toggle dual mode. Press Y by itself; X+Y chords are ignored so engage and dual mode do not fight. |
| **B** (right) | Toggle dataset recording. |
| Thumbstick / menu | Unused. |

## Per-session flow

1. **Open the dashboard**: `http://<workstation>:5000`. Click *Connect* on each arm you want to use.
2. **Open the VR operator URL on the Quest browser** (shown on the *VR Operator* page). Accept the self-signed cert, enter VR.
3. **Calibrate** if you haven't, or if you're standing somewhere new — see [calibration.md](calibration.md). Once-per-setup; the calibration is saved.
4. In the headset, use the operator panel like Reachy's flow:
   **Connection** -> **Mirror/ready** -> **Teleop** -> **Suspension**.
   Hold **A** while facing the workspace to confirm Ready/recenter.
5. Press **A**/**X** or toggle Engage in the dashboard. The headset panel shows whether each arm is connected, torqued, anchored, wrist-aligned, and recording-ready.
6. **Squeeze grip** on a controller to anchor that arm's EE pose. The panel shows "anchored"; releasing grip stops motion.
7. **Hold grip + move your hand**. The arm follows. Pull trigger to close the gripper.
8. Press **B** to request dataset recording. The backend still enforces calibration/verification blockers, so the headset can request recording but cannot bypass safety or dataset-quality checks.

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

While grip is held, the backend integrates XLeVR's per-frame `relative_position`
packets, maps that reset-relative controller displacement through the saved
VR-to-robot frame, and caps the end-effector target step before IK. This keeps
slow hand motion smooth and rejects one-frame tracking spikes. The controller's
rotation is also mapped through the reset-time controller-to-EE alignment so
wrist intent does not depend on the exact grip angle at anchor time.

Useful tuning keys live in `config/xlerobot.yaml`:

```yaml
vr:
  kp: 0.75
  pos_ema_alpha: 0.3
  ori_ema_alpha: 0.35
  pos_deadzone_m: 0.001
  rot_deadzone_deg: 0.3
  max_ee_step_m: 0.004
```

## Safety

- **EMERGENCY STOP** button (top of page) — instantly disables torque on both arms. The robot freezes wherever it is.
- **Watchdog** — if VR goals stop arriving (controller down, Wi-Fi blip), the drive loop stops within 0.3 s.
- **Per-tick joint caps** — max joint speeds capped (e.g. shoulder_pan 5°/tick at 30 Hz = 150°/s). Independent of the speed slider.
- **No autonomous motion**, ever. Disconnect = torque off, no homing. The only motion the app initiates is the user-clicked *Go to Home*.

## Future OpenXR App TODO

The current implementation intentionally keeps the existing WebXR/XLeVR stack while adopting the Reachy-style operator flow. A future native OpenXR app should preserve the same backend contract and add:

- Lower-latency controller and headset pose streaming with explicit tracking-quality flags.
- Native multi-panel video rendering with head-camera and optional wrist/scene camera feeds.
- Persistent in-headset connection, mirror/ready, teleop, and suspension scenes.
- Haptic warnings for stale tracking, IK/reachability rejects, and recording blockers.
- Signed/packaged deployment so the Quest app does not depend on browser certificate acceptance.

See [troubleshooting.md](troubleshooting.md) if motion feels wrong or doesn't happen.
