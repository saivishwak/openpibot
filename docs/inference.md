# PI0.5 inference on XLeRobot (bimanual SO-101)

This repo supports local finetuned PI0.5 inference:

| Path | Script | Policy runtime | When to use |
|------|--------|----------------|-------------|
| **Finetuned (recommended)** | `scripts/infer_pi05_finetuned.py` | Local LeRobot checkpoint on GPU | After finetuning on your VR dataset |

This document focuses on **finetuned local inference**, which matches the dataset layout recorded via the dashboard (`head`, `left_wrist`, `right_wrist` cameras and 12 arm joints).

For the full Quest/OpenXR teleop -> LeRobot dataset -> finetuning -> inference
contract, see [architecture.md](architecture.md).

## Prerequisites

1. **Initialized `vendor/lerobot` submodule** on the configured `main` branch. The root `uv.lock` uses this submodule as the editable LeRobot workspace so the XLeRobot driver overlay is available.

2. **Hardware**: bimanual SO-101 arms on the ports in `config/xlerobot.yaml` (`port_left_base`, `port_right_head`), three USB cameras, and motor calibration files under `config/calibration/so_follower/`.

3. **Home pose** in `config/xlerobot.yaml` (`robot.home_pose`). Capture it from the dashboard (VR Teleop → Capture home) or edit the YAML directly.

4. **Finetuned checkpoint** (see [Finetuning](#finetuning)). Checkpoints are saved under `outputs/pi05_finetune/checkpoints/<step>/pretrained_model/`.

5. **Dependencies**: root `pyproject.toml` pins `transformers>=5.4.0,<5.6.0` for PI0.5. After pulling changes, run `git submodule update --init vendor/lerobot` and `uv sync`. If inference fails with `create_causal_mask() ... cache_position`, you likely have transformers 5.6+ installed — re-sync the venv.

## Finetuning

Train on your LeRobot dataset (defaults from `config/xlerobot.yaml` → `dataset.repo_id`):

```bash
uv run python scripts/finetune_pi05.py
```

Useful flags:

- `--dry-run` — print the underlying `lerobot-train` command without running
- `--steps`, `--save-freq`, `--output-dir` — training length and checkpoint layout
- `--no-oom-safe` — disable batch cap / frozen vision (needs more VRAM)

By default, `--oom-safe` caps batch size at 2, freezes the vision encoder, and trains the expert only. Camera keys are renamed for PI0.5 (`head` → `base_0_rgb`, etc.) via `--rename-map-json`.

Example checkpoint path after 5k steps:

```text
outputs/pi05_finetune/checkpoints/005000/pretrained_model/
```

## Shared Runtime Contract

Inference intentionally uses the same runtime assumptions as recording:

- `config/xlerobot.yaml` is the shared source of truth for serial ports, camera
  role mappings, home pose, dataset FPS, PI0.5 horizon, and VR joint
  caps/deadbands.
- `scripts/_xlerobot_loader.py` builds the LeRobot `XLerobotConfig`; inference
  uses the LeRobot `XLerobot` class for motor reads/writes instead of a separate
  robot implementation.
- The default `--camera-backend dashboard` reuses the same dashboard
  `CameraStream` role registry as VR recording, avoiding a second V4L capture
  stack.
- Policy observations use the same 12-joint `JOINT_ORDER` and the same three
  camera roles as recorded datasets.
- Policy actions are shaped like recording labels before `send_action`. The
  order is: optional policy EMA, optional first-action replan blend, VR-style
  per-joint cap/KP shaping vs previous command, optional present clamp,
  deadband, then optional final command EMA.

## Bimanual SO-101 vs full XLerobot driver

The LeRobot `XLerobot` class lists base wheels and head motors that are **not** on a typical dual SO-101 desk setup. Inference handles this automatically:

1. **Lenient bus connect** (default): missing motors are dropped at connect time with a warning.
2. **Calibration merge**: `scripts/_xlerobot_loader.py` builds `config/calibration/xlerobot/xlerobot.json` from `so_follower/{left,right}_follower_arm.json`.
3. **Prune uncalibrated motors**: after restore, motors without calibration entries (base, head) are removed so reads/writes do not fail.

Use `--strict-motors` only if you have the full XLerobot hardware and calibration for base/head.

Connect uses **`calibrate=False`** and restores calibration from disk (no interactive wizard).

## Test homing only (no policy)

Verify ports, calibration, and `robot.home_pose` before loading the model:

```bash
uv run python scripts/infer_pi05_finetuned.py \
  --dry-run-home \
  --home-timeout 90 \
  --fps 30
```

Homing reads **joint positions only** (no cameras). Expect warnings about pruned base/head motors on SO-101 setups.

## Run inference (balanced, recommended)

Validated on-robot for marker→cup with the 80-episode / 25k-step checkpoint. This keeps the arm joints on the configured present-position safety clamp while allowing the gripper to close at the same 15°/tick cap used in the VR recording labels.

```bash
uv run python scripts/infer_pi05_finetuned.py \
  --policy-path outputs/pi05_finetune_80ep/checkpoints/last/pretrained_model \
  --task "Pick up the marker from the table and place it inside the cup" \
  --camera-backend dashboard \
  --episodes 2 \
  --episode-time 180 \
  --fps 30 \
  --settle-steps 30 \
  --gripper-max-relative-target 15
```

## Run inference with extra smoothing disabled

Use this when you want to evaluate the policy with the closest behavior to the
recorded action labels. These flags disable policy-target EMA, final-command
EMA, and first-action blend across replans. VR-style per-joint caps/KP shaping
and deadbands from `config/xlerobot.yaml` still apply because those are part of
the dataset label contract.

```bash
uv run python scripts/infer_pi05_finetuned.py \
  --policy-path outputs/pi05_finetune_80ep/checkpoints/last/pretrained_model \
  --task "Pick up the marker from the table and place it inside the cup" \
  --camera-backend dashboard \
  --episodes 2 \
  --episode-time 180 \
  --fps 30 \
  --settle-steps 30 \
  --gripper-max-relative-target 15 \
  --policy-ema-alpha=1 \
  --command-ema-alpha=1 \
  --replan-blend=1
```

If this preset reaches better but jitters, return to the balanced defaults or
lower `--command-ema-alpha` gradually. If the balanced defaults stall near home,
raise `--command-ema-alpha` or use this preset as a diagnostic.

### Tuning reach vs jitter

| Symptom | Try |
|---------|-----|
| Stuck at home, only small bobbing | Raise `--command-ema-alpha` (e.g. `0.26`) and/or lower `--joint-deadband-deg` (e.g. `0.65`) |
| Reaches target but **jittery** | Lower `--command-ema-alpha` (e.g. `0.18`), raise `--joint-deadband-deg` (e.g. `0.85`), raise `--policy-ema-alpha` (e.g. `0.38`), or `--open-loop-steps 50` |
| Snappy reach, still noisy at replans | `--replan-blend 0.15` and/or longer `--open-loop-steps` |
| Misses grasp / closes late | Keep `--action-shape-kp 1.0`; raise `--gripper-max-relative-target` from `15` toward `20–25` |
| Stuck mid-chunk | `--replan-on-miss-deg 18` (execution lag vs last command; off by default) |

Known snappy preset (reaches well, may jitter): `--command-ema-alpha 0.28 --joint-deadband-deg 0.6`.

Optional: reach the marker before the full place-in-cup task (first ~20s):

```bash
uv run python scripts/infer_pi05_finetuned.py \
  --policy-path outputs/pi05_finetune_80ep/checkpoints/last/pretrained_model \
  --camera-backend dashboard \
  --phase1-task "Reach the marker on the table. Do not go to the cup yet." \
  --phase1-sec 20 \
  --task "Pick up the marker from the table and place it inside the cup" \
  --episodes 2 \
  --episode-time 180 \
  --fps 30 \
  --settle-steps 30 \
  --gripper-max-relative-target 15
```

### Control loop

- At `fps` (default from `dataset.fps` in yaml), the script pops one action per tick from a chunk predicted by the policy.
- Every `--open-loop-steps` (default 35), it grabs a new observation (motors + three cameras), runs preprocessors, calls `predict_action_chunk`, and postprocesses.
- **Policy reset** at each episode start, after settle, and at episode end (clears action queue + preprocessor/postprocessor state).
- **Homing**: unless `--skip-home`, the robot moves to `robot.home_pose` before the run and/or at each episode start (`--home-before-episode`, default from `dataset.home_before_episode`).

### Dry-run config (no robot)

```bash
uv run python scripts/infer_pi05_finetuned.py \
  --dry-run \
  --policy-path outputs/pi05_finetune_80ep/checkpoints/last/pretrained_model \
  --task "Pick up the marker from the table and place it inside the cup"
```

## CLI reference (`infer_pi05_finetuned.py`)

Inference runs **one observation → one action** per control tick (batch size 1). There is no `--batch-size` flag here; training batch size is set in [`scripts/finetune_pi05.py`](scripts/finetune_pi05.py) (`--batch-size`, default `8`).

| Flag | Default | Description |
|------|---------|-------------|
| `--policy-path` | — | Path to `.../pretrained_model` (required for inference) |
| `--task` | — | Language instruction for the policy (required for inference) |
| `--episodes` | `2` | Number of episodes |
| `--episode-time` | `120` | Max wall-clock seconds per episode (pre-home, loop, post-home share this budget) |
| `--episode-steps` | — | Max control-loop steps per episode (includes settle); ends when this **or** `--episode-time` is hit |
| `--stop-on-episode-error` | off | Abort remaining episodes after a failed episode |
| `--device` | `cuda` | `cuda`, `cpu`, or `mps` |
| `--fps` | yaml `dataset.fps` (usually `30`) | Control loop rate (match training) |
| `--action-horizon` | yaml `pi05.action_horizon` (often `50`) | Max policy chunk size (upper bound) |
| `--open-loop-steps` | `35` | Steps per scheduled chunk before re-inferring (higher = smoother) |
| `--replan-on-miss-deg` | `0` | Early re-infer if present lags **last sent command** (`18` typical when enabled); `0` = off |
| `--replan-miss-steps` | `2` | Consecutive ticks over threshold before early replan |
| `--replan-blend` | `0.2` | Blend first action after each new chunk (`1.0` = no blend) |
| `--settle-steps` | `60` | Hold pose after homing (~2s @ 30Hz) before policy runs |
| `--policy-ema-alpha` | `0.36` | EMA on raw policy targets before VR shaping; `1.0` disables |
| `--command-ema-alpha` | `0.2` | EMA on final motor command; lower = smoother, `1.0` disables |
| `--joint-deadband-deg` | yaml `vr.joint_command_deadband_deg` | Override final command deadband for every joint; otherwise uses per-joint config |
| `--clamp-to-present` | on | Clamp vs measured pose using `robot.max_relative_target`; use `--no-clamp-to-present` only for controlled debugging |
| `--phase1-task` | — | Shorter prompt for the first segment (e.g. reach marker only) |
| `--phase1-sec` | `0` | Seconds to use `--phase1-task` before `--task` |
| `--camera-backend` | `dashboard` | `dashboard` (shared V4L streams) or `lerobot` (robot OpenCVCamera) |
| `--show-cameras` / `--no-show-cameras` | on if `DISPLAY` set | Resizable pygame camera mosaic (background thread) |
| `--preview-fps` | `15` | Max refresh rate for camera preview |
| `--skip-home` | off | Skip homing entirely |
| `--home-before-episode` / `--no-home-before-episode` | yaml `dataset.home_before_episode` | Home at start of each episode |
| `--skip-home-after-episode` | off | Skip post-episode homing |
| `--home-timeout` | `60` | Seconds before homing gives up and continues |
| `--max-relative-target` | yaml `robot.max_relative_target` | Per-command joint cap (degrees); optional override |
| `--gripper-max-relative-target` | — | Override present clamp for gripper joints only; useful for late/weak grasps |
| `--action-shape-kp` | `1.0` | P blend while converting policy targets into VR-style commands; `1.0` matches recorded action labels |
| `--strict-motors` | off | Fail if base/head motors are missing |
| `--dry-run` | off | Print settings and exit |
| `--dry-run-home` | off | Connect, home, disconnect; no policy |

## How this matches VR recording (`openpibot/server/runtime/dataset.py`)

Training data from the dashboard uses:

| Field | Meaning |
|-------|---------|
| `observation.state` | Present joint positions (degrees), 12-vector in `JOINT_ORDER` |
| `action` | **Command sent to motors that tick** (absolute degrees), not the raw VR IK goal |
| `observation.images.{head,left_wrist,right_wrist}` | RGB 640×480 @ 30 Hz |

`action` is built in `vr_teleop.py` the same way as teleop:

1. Per-joint cap vs **previous command**: `cmd = last_sent + clip(target - last_sent, ±cap)` (caps 5–15°/tick).
2. Normal live teleop stores that rate-limited command as the dataset label.

So each training frame’s `|action − state|` is usually **small** (≤ per-tick cap) while moving, not a 40°+ jump.

`infer_pi05_finetuned.py` applies the same VR rate limits to policy outputs before `send_action`: optional policy EMA, optional replan blend, per-joint cap/KP shaping vs previous command, optional present clamp, deadband, then final command EMA. `wrist_flex`, `wrist_roll`, and `gripper` bypass the final command EMA so wrist/gripper response matches the recording path more closely. **Present-based** `max_relative_target` clamp is **on by default** for safety; use `--gripper-max-relative-target 15` when the arm reaches correctly but the gripper closes late or weakly. Use **`--fps 30`** to match `dataset.fps`.

## Configuration (`config/xlerobot.yaml`)

| Section | Used for |
|---------|----------|
| `robot.port_*` | Serial ports for left (arm) and right (arm) buses |
| `robot.home_pose` | Homing targets (12 joint names, degrees) |
| `cameras.*` | OpenCV device paths → `head`, `left_wrist`, `right_wrist` |
| `dataset.repo_id` | Finetuning dataset id |
| `dataset.root` | Optional local LeRobot dataset root used by recording/push tooling |
| `dataset.home_before_episode` | Default per-episode homing |
| `pi05.control_fps`, `pi05.action_horizon` | Inference timing defaults |
| `vr.joint_deg_caps`, `vr.joint_command_deadband_deg` | Recording-style action shaping used by inference |

Observation keys sent to the policy match finetuning rename map:

- `observation.images.head`, `left_wrist`, `right_wrist`
- `observation.state` — 12 arm joint positions (`.pos` keys)

## Troubleshooting

| Symptom | Likely cause | What to do |
|---------|----------------|------------|
| `lerobot.robots.xlerobot is not installed` | The `vendor/lerobot` submodule overlay is missing or not synced | Run `git submodule update --init vendor/lerobot`, then `uv sync` |
| `robot.home_pose is empty` | No saved pose | Capture it from the dashboard or edit `config/xlerobot.yaml` |
| `Missing motor IDs` with `--strict-motors` | SO-101 without base/head | Drop `--strict-motors` (default lenient mode) |
| `KeyError` on `head_pan` / `base_*` | Old driver without prune fix | Use current `xlerobot.py` + merged calibration |
| `missing camera observations` | Camera path wrong or unplugged | Fix `cameras.*.path` in yaml; check `/dev/v4l/...` |
| CUDA OOM during **training** | Full PI0.5 finetune | Keep default `--oom-safe`; reduce batch or steps |
| Policy moves wrong / no task following | Wrong checkpoint or no finetune | Use a finetuned `pretrained_model`, not only `pi05_base` |
| Arms **oscillate** / jitter in place | FPS != dataset, replan too often, present clamp too tight, or command EMA too high | Use `--fps 30`, defaults; if still jittery after reaching, lower `--command-ema-alpha` and raise `--policy-ema-alpha`; use `--no-clamp-to-present` only for controlled debugging |
| **Never leaves home** for a long time | Command EMA too low or deadband too high | Raise `--command-ema-alpha` (e.g. `0.26–0.28`) and/or lower `--joint-deadband-deg` (e.g. `0.6–0.65`) |
| Robot goes to **cup before marker** | Single task string for whole episode; head cam may bias toward cup; demos pause at home first | Use `--settle-steps 30–60`, `--phase1-task` / `--phase1-sec`; align scene with training; finetune longer |
| `create_causal_mask() got an unexpected keyword argument 'cache_position'` | transformers 5.6+ in venv | `uv sync` (root pins `transformers<5.6`) |

## File map

```text
scripts/finetune_pi05.py          # wrapper → lerobot-train
scripts/infer_pi05_finetuned.py   # local finetuned inference + homing
scripts/_xlerobot_loader.py       # yaml → XLerobotConfig, lenient motors, calib merge
config/xlerobot.yaml              # ports, cameras, home pose, dataset id
config/calibration/so_follower/   # per-arm calibration (source of truth)
config/calibration/xlerobot/      # merged xlerobot.json (generated at connect)
outputs/pi05_finetune/            # training outputs and checkpoints
```
