# PI0.5 inference on XLeRobot (bimanual SO-101)

This repo supports two inference paths:

| Path | Script | Policy runtime | When to use |
|------|--------|----------------|-------------|
| **Finetuned (recommended)** | `scripts/infer_pi05_finetuned.py` | Local LeRobot checkpoint on GPU | After finetuning on your VR dataset |
| **OpenPI server (baseline)** | `scripts/run_pi05_inference.py` | Remote WebSocket (`scripts/run_openpi_server.sh`) | Zero-shot experiments with `pi05_base` only |

The rest of this document focuses on **finetuned local inference**, which matches the dataset layout recorded via the webapp (`head`, `left_wrist`, `right_wrist` cameras and 12 arm joints).

## Prerequisites

1. **Copy the XLerobot robot driver into the LeRobot submodule** (once per clone):

   ```bash
   bash scripts/setup_xlerobot.sh
   ```

2. **Hardware**: bimanual SO-101 arms on the ports in `config/xlerobot.yaml` (`port_left_base`, `port_right_head`), three USB cameras, and motor calibration files under `config/calibration/so_follower/`.

3. **Home pose** in `config/xlerobot.yaml` (`robot.home_pose`). Capture it from the webapp (VR Teleop → Capture home) or:

   ```bash
   uv run python scripts/save_home_pose.py
   ```

4. **Finetuned checkpoint** (see [Finetuning](#finetuning)). Checkpoints are saved under `outputs/pi05_finetune/checkpoints/<step>/pretrained_model/`.

5. **Dependencies**: root `pyproject.toml` pins `transformers>=5.4.0,<5.6.0` for PI0.5 (LeRobot submodule). After pulling changes, run `uv sync`. If inference fails with `create_causal_mask() ... cache_position`, you likely have transformers 5.6+ installed — re-sync the venv.

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

## Run inference

```bash
uv run python scripts/infer_pi05_finetuned.py \
  --policy-path outputs/pi05_finetune/checkpoints/last/pretrained_model \
  --task "Pick up the medicine and place it in the bowl" \
  --episodes 1 \
  --episode-time 120 \
  --device cuda
```

### Control loop

- At `fps` (default from `pi05.control_fps` or `dataset.fps` in yaml), the script pops one action per tick from a chunk predicted by the policy.
- Every `--action-horizon` steps (default `pi05.action_horizon`, often 50), it grabs a new observation (motors + three cameras), runs preprocessors, calls `predict_action_chunk`, and postprocesses.
- **Homing**: unless `--skip-home`, the robot moves to `robot.home_pose` before the run and/or at each episode start (`--home-before-episode`, default from `dataset.home_before_episode`).

### Dry-run config (no robot)

```bash
uv run python scripts/infer_pi05_finetuned.py \
  --dry-run \
  --policy-path outputs/pi05_finetune/checkpoints/005000/pretrained_model \
  --task "..."
```

## CLI reference (`infer_pi05_finetuned.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--policy-path` | — | Path to `.../pretrained_model` (required for inference) |
| `--task` | — | Language instruction for the policy (required for inference) |
| `--episodes` | `2` | Number of episodes |
| `--episode-time` | `120` | Max seconds per episode |
| `--device` | `cuda` | `cuda`, `cpu`, or `mps` |
| `--fps` | yaml | Control loop rate |
| `--action-horizon` | yaml `pi05.action_horizon` | Re-infer every N control ticks |
| `--skip-home` | off | Skip homing entirely |
| `--home-before-episode` | yaml `dataset.home_before_episode` | Home at start of each episode |
| `--home-timeout` | `60` | Seconds before homing gives up and continues |
| `--strict-motors` | off | Fail if base/head motors are missing |
| `--dry-run` | off | Print settings and exit |
| `--dry-run-home` | off | Connect, home, disconnect; no policy |

## Configuration (`config/xlerobot.yaml`)

| Section | Used for |
|---------|----------|
| `robot.port_*` | Serial ports for left (arm) and right (arm) buses |
| `robot.home_pose` | Homing targets (12 joint names, degrees) |
| `cameras.*` | OpenCV device paths → `head`, `left_wrist`, `right_wrist` |
| `dataset.repo_id` | Finetuning dataset id |
| `dataset.home_before_episode` | Default per-episode homing |
| `pi05.control_fps`, `pi05.action_horizon` | Inference timing defaults |

Observation keys sent to the policy match finetuning rename map:

- `observation.images.head`, `left_wrist`, `right_wrist`
- `observation.state` — 12 arm joint positions (`.pos` keys)

## OpenPI server path (optional)

For the upstream dual-arm example with a **generic** `pi05_base` checkpoint over WebSocket:

```bash
# Terminal 1
bash scripts/run_openpi_server.sh

# Terminal 2
uv run python scripts/run_pi05_inference.py \
  --task "Pick the red block and place it in the bin" \
  --episodes 2 --episode-time 120
```

Expect weak zero-shot behavior until you finetune and use `infer_pi05_finetuned.py` instead.

## Troubleshooting

| Symptom | Likely cause | What to do |
|---------|----------------|------------|
| `lerobot.robots.xlerobot is not installed` | Submodule copy missing | `bash scripts/setup_xlerobot.sh` |
| `robot.home_pose is empty` | No saved pose | Webapp capture or `scripts/save_home_pose.py` |
| `Missing motor IDs` with `--strict-motors` | SO-101 without base/head | Drop `--strict-motors` (default lenient mode) |
| `KeyError` on `head_pan` / `base_*` | Old driver without prune fix | Use current `xlerobot.py` + merged calibration |
| `missing camera observations` | Camera path wrong or unplugged | Fix `cameras.*.path` in yaml; check `/dev/v4l/...` |
| CUDA OOM during **training** | Full PI0.5 finetune | Keep default `--oom-safe`; reduce batch or steps |
| Policy moves wrong / no task following | Wrong checkpoint or no finetune | Use a finetuned `pretrained_model`, not only `pi05_base` |
| Arms **oscillate** / jitter in place | FPS ≠ dataset (e.g. 50 vs 30), huge per-step joint jumps, weak finetune | Use `--fps 30`; watch startup `|cmd-present|` log; set `robot.max_relative_target` in yaml; use a later checkpoint |
| `create_causal_mask() got an unexpected keyword argument 'cache_position'` | transformers 5.6+ in venv | `uv sync` (root pins `transformers<5.6`) |

## File map

```text
scripts/finetune_pi05.py          # wrapper → lerobot-train
scripts/infer_pi05_finetuned.py   # local finetuned inference + homing
scripts/_xlerobot_loader.py       # yaml → XLerobotConfig, lenient motors, calib merge
scripts/run_pi05_inference.py     # OpenPI WebSocket wrapper (baseline)
config/xlerobot.yaml              # ports, cameras, home pose, dataset id
config/calibration/so_follower/   # per-arm calibration (source of truth)
config/calibration/xlerobot/      # merged xlerobot.json (generated at connect)
outputs/pi05_finetune/            # training outputs and checkpoints
```
