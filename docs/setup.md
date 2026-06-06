# Setup

One-time install + hardware config for a bimanual SO-101 XLeRobot.

## Install

```bash
uv sync
```

This prepares the OpenPIBot CLI, backend, dashboard runtime, LeRobot dataset tooling, and PI0.5 training/inference dependencies. The optional OpenPI WebSocket policy server is started through `uv` from a package-managed OpenPI runtime instead of a vendored checkout.

See [architecture.md](architecture.md) for the current Quest/OpenXR teleop,
recording, finetuning, and inference data path.

## Configure `config/xlerobot.yaml`

Set five things:

1. **Motor bus ports** — both arms. Find them with `uv run lerobot-find-port`.
   ```yaml
   robot:
     port_left_base:  /dev/ttyACM1
     port_right_head: /dev/ttyACM0
   ```
2. **Camera roles** — assign `head`, `left_wrist`, and `right_wrist` from the
   dashboard Cameras page. The page lists configured devices first and then all
   visible V4L video devices. Manual YAML editing is still supported with
   `/dev/v4l/by-path/...` paths.
3. **Dataset repo and optional storage root** — for LeRobot recording.
   `dataset.root` is optional. If omitted, recording uses the LeRobot/Hugging
   Face default cache path; the Recording page can persist both `dataset.repo_id`
   and `dataset.root` into this YAML while recording is idle.
4. **Gripper convention** — if pulling the trigger opens (instead of closes), swap:
   ```yaml
   gripper:
     open_value: 0
     closed_value: 100
   ```
5. **Quest pairing token and video settings** — set
   `XLE_QUEST_PAIRING_TOKEN` before running the backend, and tune
   `vr.quest_video` if headset camera streaming needs different ports,
   bitrate, flip, or color settings.

## Motor calibration

The `config/calibration/so_follower/{left,right}_follower_arm.json` files come from lerobot's calibration tool. If you've already calibrated, copy them in. If not:

```bash
uv run lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=$PORT \
  --robot.id=left_follower_arm \
  --robot.calibration_dir=$(pwd)/config/calibration/so_follower
```

Repeat for the right arm.

**One known gotcha**: lerobot's calibration sometimes captures a `range_min == range_max` for the gripper if you don't move it through its full open↔close cycle. If that happens, re-run `lerobot-calibrate` for that arm and move the gripper through its full open/close range during calibration.

## Add yourself to `dialout`

```bash
sudo usermod -aG dialout $USER
newgrp dialout         # picks up the group in this shell
```

Required for `/dev/ttyACM*` access. Without it the dashboard can't open the motor bus.

## Run the dashboard

```bash
uv run openpibot run --host 0.0.0.0
```

The run command installs frontend dependencies with `pnpm`, builds the dashboard, and starts the backend. Use `--no-build-dashboard` only when you are running the Vite dev server separately.

That's it. Open the page, connect an arm, follow [teleop.md](teleop.md).
