# Calibration

Two independent setup steps, both done once per robot+user. Both persist to YAML.

## Home pose

The pose every recorded episode starts from. Critical for VLA training: every demonstration must start from the same proprioception state, or the policy can't generalise.

**Capture procedure** (per arm, on the Home pose card):

1. Click **Release for posing** → torque on that arm goes off. Support the arm by hand — it'll sag under gravity, especially the shoulder lift.
2. **Hand-pose** the arm to your desired starting position. Recommended: "robot waving hello" — upper arm raised ~45°, forearm forward, gripper open at chest height. Avoid extremes (fully extended or fully folded — IK gets sloppy).
3. Click **Capture** → reads joint positions, writes them to `robot.home_pose.<side>_arm_*` in `config/xlerobot.yaml` (comments preserved).
4. Click **Lock at current** → re-enables torque, holding at the just-captured pose (no snap-back to a stale goal).

**Go to Home**: from any subsequent pose, click *Go to Home* and the arm slowly interpolates back via the same drive loop teleop uses (same per-tick caps + KP). Slow enough to abort with EMERGENCY STOP if needed.

**Auto-home before each recorded episode** (recommended for VLA):
```yaml
dataset:
  home_before_episode: true
```
Then every B-press to start recording first homes all connected arms, waits for them to settle, then opens the episode.

## Calibration profiles

Use the **Calibration Profile** card on the VR Teleop page when multiple people,
headset guardian setups, or robot table positions share the same machine.

- **Create** makes a new active profile. Leave **copy active** enabled when the
  new user/setup starts from the current calibration and will refine it.
- **Select** switches the active profile loaded by teleop and recording.
- **Delete active** removes the selected profile; the UI prevents deleting the
  only remaining profile.

Profile switches are blocked while recording, while VR teleop is engaged, or
while any calibration/robot-verification flow is active. Switching reloads the
selected calibration and clears the current VR grip anchors, so press grip again
before driving.

`config/vr_calibration.yaml` stores profiles like this:

```yaml
active_profile: default
profiles:
  default:
    left:
      session_vr_to_robot: [...]
    right:
      session_vr_to_robot: [...]
  another_user:
    left:
      session_vr_to_robot: [...]
```

Older top-level `left:` / `right:` files are still accepted and are migrated into
the `default` profile the next time the calibration file is written.

## VR frame calibration (stage 1)

Tells the system which direction in VR-world space is "user-forward" and "user-up" for your body. Without this, motion direction is wrong unless you happen to stand exactly facing the room's VR-default direction.

**Procedure** (per arm, on the Calibration card):

1. Click **Calibrate** for that arm. Card switches to wizard mode.
2. **Step 1 — Forward axis**: put on headset. Squeeze grip on that controller, **keep it held**, move your hand straight forward (toward the robot, away from your body) by ~10 cm. Release grip. The card shows live motion magnitude.
3. **Step 2 — Up axis**: squeeze grip again, **keep it held**, move your hand straight up by ~10 cm. Release grip.
4. **Step 3 — Left axis**: squeeze grip again, **keep it held**, move your hand to your left by ~10 cm. Release grip.
5. **Step 4 — Wrist pitch**: squeeze grip, keep it held, pitch your wrist **UP** by ~20-45°, then release. For both controllers, use the same physical intent: thumb side rises toward the ceiling.
6. **Step 5 — Wrist roll**: squeeze grip, keep it held, roll your wrist **RIGHT** by ~20-45°, then release. For both controllers, use the same physical intent: thumb side rolls to your right. You can skip wrist steps for translation-only testing, but wrist teleop will then use WebXR defaults.
7. Calibration finalises. The 3×3 VR→robot rotation matrix plus empirical wrist pitch/roll axes are saved to **`config/vr_calibration.yaml`**.

Subsequent dashboard restarts load the saved matrix automatically. You only need to re-run the wizard if you change where you stand or how you orient yourself relative to the robot.

**Why multiple motions, not one**: a single forward motion only solves for yaw. Capturing forward, up, and left lets the system estimate the full operator frame and catch lateral sign mistakes.

## Robot-verified calibration (stage 2)

Stage 1 is still useful for quick manual teleop, but training data should use the robot-verified refinement. This matches top VR teleop systems: VR control is reset-relative, but the VR tracking frame is checked against the robot end-effector frame before recording demonstrations.

**Procedure** (per arm, on the Robot verification card):

1. Finish stage 1 VR calibration first.
2. Click **Start verification**. Torque is released for hand-posing; support the arm.
3. For each sample, capture a paired motion:
   - Place the robot end effector at the neutral pose and click **Robot start**.
   - Move the robot end effector to the target pose and click **Robot end**. The arm locks there.
   - Hold the VR controller comfortably and click **VR start**. This is only a temporary anchor for this sample.
   - Move the VR controller through the same relative motion. Watch the live target/mapped/error feedback.
   - When the live match is good, click **VR end**. The sample is saved and torque releases again for the next hand-posed sample.
   - If torque is locked and you need to redo or start the next robot pose, click **Release for posing** in the Robot verification card.
4. Capture at least six samples near the grasp workspace: forward, back, left, right, up, down.
5. Click **Solve verification**. The backend fits a robot-verified translation matrix, keeps the stage-1 rotation for wrist/orientation, computes RMS residual error, and saves the result. If the solve fails, the card lists per-sample residuals so you can recapture the bad directions instead of guessing.
6. Click **Start low-scale test**. Keep the controller still while starting; then hold the selected controller grip and move slowly while watching the real robot/camera. Click **Stop test** before recording.

The runtime uses the verified translation matrix when available. The learned
scale is already part of that 3×3 matrix at runtime; `translation_scale` is
saved for diagnostics/UI only:

```text
robot_delta = translation_vr_to_robot_matrix @ vr_delta
```

If the residual is too high, the solve is rejected. Re-capture samples with cleaner, more distinct motions near the object pickup area.

The VR start pose is not a remembered physical pose. Normal teleop and the low-scale test both use reset-relative control:

```text
vr_anchor = current controller pose
robot_anchor = current robot EE pose
robot_target = robot_anchor + translation_vr_to_robot_matrix @ integrated_relative_vr_delta
```

**Re-anchor vs re-calibrate**: every grip-press re-anchors the EE position (where the gripper sits at the moment of grip-press). That's different from the VR-frame calibration above, which only changes if you click *Calibrate*. Anchor = "where is my hand starting from now"; calibration = "what does forward/up mean".

**Stage 1 vs stage 2 in config**: before you run Robot verification, the YAML
will show `calibration_mode: vr_direction`. That is expected. It changes to
`robot_verified` after you capture all six directions and solve successfully.
The low-scale test is a per-session safety gate before recording; it is not
persisted as part of the calibration file.

## Files written

| File | Written by | When |
|---|---|---|
| `config/xlerobot.yaml` (`robot.home_pose`) | *Capture* button | On click |
| `config/vr_calibration.yaml` | Calibration wizard | When stage 1 or stage 2 finalises |

Both are auto-managed. To re-do, use the UI; don't edit by hand.
