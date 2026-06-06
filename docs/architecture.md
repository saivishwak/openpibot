# Runtime Architecture

This repo has one production path for VR data collection and policy deployment:
native Quest/OpenXR teleop, LeRobot dataset recording, PI0.5 finetuning, and
on-robot inference all share the same robot config, camera roles, joint order,
and command-shaping semantics.

## Source Of Truth

`config/xlerobot.yaml` is the shared runtime config:

- `robot.*` defines serial ports, SO-101 IDs, home pose, and optional present
  clamp limits.
- `cameras.head`, `cameras.left_wrist`, and `cameras.right_wrist` define the
  camera role-to-device mapping used by recording and inference.
- `dataset.*` defines LeRobot repo id, FPS, home-before-episode behavior,
  optional `dataset.root`, and Hub upload behavior.
- `pi05.*` defines inference defaults such as action horizon and control FPS.
- `vr.*` defines teleop command shaping: controller smoothing, EE step limit,
  wrist polarity, per-joint caps, deadbands, and command filter weights.

The dashboard Cameras page is the preferred way to assign camera roles. It lists
the configured devices first and then every visible V4L video device so a camera
can be remapped without editing YAML by hand. The Recording page lets you edit
the dataset repo id and storage root while idle; saving writes
`dataset.repo_id` and/or `dataset.root` back into `config/xlerobot.yaml`.

## Teleop Control Path

The shipped headset path is the Unity/OpenXR Quest app in `apps/quest-xlerobot`.
It streams absolute controller poses and buttons to:

```text
/api/vr/quest/ws
```

The backend owns robot authority. It converts Quest packets into reset-relative
VR controller deltas, applies the saved calibration, solves SO-101 IK against
the calibrated XLeRobot URDF, applies per-tick joint caps/deadbands/filtering,
and sends motor commands.

The default Quest coordinate frame is `quest_operator_frame`. The native adapter
maps Unity/OpenXR controller coordinates into the backend operator frame as:

```text
operator = (unity.z, -unity.x, unity.y)
```

Rotations use the same basis by matrix conjugation. The legacy
`unity_reachy` name is accepted only as an alias for `quest_operator_frame`.

## Calibration And Verification

Calibration is split into two stages:

1. **VR frame calibration** captures operator forward, up, left, wrist pitch,
   and wrist roll intent. This persists the operator-frame rotation and wrist
   axes to `config/vr_calibration.yaml`.
2. **Robot verification** captures paired VR and robot end-effector deltas near
   the task workspace. Solving this produces the robot-verified translation
   matrix used for dataset-quality teleop.

Normal teleop is reset-relative. Every grip press anchors the current controller
pose and the current robot end-effector pose; calibration tells the system how
controller deltas map into robot deltas after that anchor.

Recording is blocked for connected arms until robot verification and the
low-scale test are complete. Quick manual teleop can still use stage-1
calibration, but data intended for finetuning should be robot-verified.

## Dataset Contract

Dashboard recording writes a LeRobot dataset with:

- `observation.state`: present joint positions in the 12-joint `JOINT_ORDER`.
- `action`: the absolute motor command sent on that tick after VR IK,
  per-joint caps, deadbands, and command filtering.
- `observation.images.head`, `observation.images.left_wrist`, and
  `observation.images.right_wrist`: RGB camera frames from the role-mapped
  dashboard camera streams.

`action` is not the raw VR target and not a future end-effector pose. It is the
same-tick commanded joint target. This is why inference also shapes policy
outputs like recording before sending commands to the robot.

## Training Contract

`scripts/finetune_pi05.py` is a wrapper around `lerobot-train`. It reads the
default dataset repo id from `config/xlerobot.yaml` and applies the image-key
rename map expected by PI0.5:

```text
observation.images.head        -> observation.images.base_0_rgb
observation.images.left_wrist  -> observation.images.left_wrist_0_rgb
observation.images.right_wrist -> observation.images.right_wrist_0_rgb
```

The dataset action/state joint order remains the project `JOINT_ORDER`.

## Inference Contract

`scripts/infer_pi05_finetuned.py` uses the same config and robot path as the
runtime:

- `scripts/_xlerobot_loader.py` builds the `XLerobotConfig` from
  `config/xlerobot.yaml`.
- The LeRobot `XLerobot` class is used for motor reads/writes and calibration.
- The default camera backend is `dashboard`, which reuses the same
  `CameraStream` role registry as recording.
- Policy observations use the same 12-joint state vector and three camera
  roles as the dataset.
- Policy actions are reshaped with the same VR per-joint caps/deadbands loaded
  from `config/xlerobot.yaml` before `send_action`.

The inference command order is:

```text
raw policy target
-> optional policy EMA
-> optional first-action replan blend
-> VR-style per-joint cap/KP shaping vs previous command
-> optional present clamp when --clamp-to-present is enabled
-> joint deadband from vr.joint_command_deadband_deg
-> optional final command EMA
-> robot.send_action
```

The inference EMA and replan flags are deployment-time filters around the
recording-style command shaping. Set them to `1` to disable extra smoothing and
first-action replan blending:

```bash
uv run python scripts/infer_pi05_finetuned.py \
  --policy-path outputs/pi05_finetune/checkpoints/last/pretrained_model \
  --task "Pick up the medicine and place it in the bowl" \
  --episodes 1 \
  --episode-time 60 \
  --fps 30 \
  --policy-ema-alpha=1 \
  --command-ema-alpha=1 \
  --replan-blend=1
```

Use that preset when you want the closest policy-to-recording behavior for
debugging. Use the default EMA values when you want smoother deployment.
