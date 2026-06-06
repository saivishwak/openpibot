# PI0.5 Training Guide (XLerobot)

This document covers the recommended training workflow for your VR teleop dataset.

The training path assumes datasets recorded through the current OpenXR teleop
architecture. For the full runtime contract, see [architecture.md](architecture.md).

## Fresh finetune from base model (recommended after new data)

Use this when you recorded new episodes and want a clean run from `lerobot/pi05_base` (not resume).

```bash
uv run python scripts/finetune_pi05.py \
  --dataset-repo-id saivishwak/xlerobot-vr-teleop-medicine-bowl \
  --pretrained-path lerobot/pi05_base \
  --output-dir outputs/pi05_finetune_v2 \
  --steps 50000 \
  --save-freq 5000
```

Notes:
- Do not pass `--resume` for a fresh run.
- Checkpoints are written under `outputs/pi05_finetune_v2/checkpoints/`.
- Latest checkpoint path is `outputs/pi05_finetune_v2/checkpoints/last/pretrained_model`.

## Resume an existing run (finetune)

Resume only if you intentionally want to continue the same run.

```bash
# Prefer lerobot-train to set total steps when resuming (wrapper does not forward --steps):
uv run lerobot-train \
  --config_path=outputs/pi05_finetune_v2/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --steps=70000 \
  --save_freq=5000
```

Resume from a specific checkpoint via the wrapper (loads that run’s config only):

```bash
uv run python scripts/finetune_pi05.py \
  --resume \
  --resume-from outputs/pi05_finetune_v2/checkpoints/050000
```

Important:
- `--steps` is the total target step count, not "add N more".
- Example: resuming from step 50000 with `--steps 70000` runs ~20000 more steps.

## Quick checks after training

### 1) Offline policy sanity check

```bash
uv run python scripts/eval_pi05_offline.py \
  --policy-path outputs/pi05_finetune_v2/checkpoints/last/pretrained_model \
  --episodes 0 \
  --max-samples 200 \
  --stride 2
```

### 2) On-robot inference

```bash
uv run python scripts/infer_pi05_finetuned.py \
  --policy-path outputs/pi05_finetune_v2/checkpoints/last/pretrained_model \
  --task "Pick up the medicine and place it in the bowl" \
  --episodes 1 \
  --episode-time 60 \
  --fps 30
```

For a deployment check with extra inference smoothing disabled:

```bash
uv run python scripts/infer_pi05_finetuned.py \
  --policy-path outputs/pi05_finetune_v2/checkpoints/last/pretrained_model \
  --task "Pick up the medicine and place it in the bowl" \
  --episodes 1 \
  --episode-time 60 \
  --fps 30 \
  --policy-ema-alpha=1 \
  --command-ema-alpha=1 \
  --replan-blend=1
```

Those flags disable policy EMA, final command EMA, and first-action replan
blend. The VR-style per-joint caps/deadbands from `config/xlerobot.yaml` still
apply, matching the recorded action-label contract.

## Warm-start from a checkpoint (new run, not resume)

Use this when you have new data (e.g. grasp-focused episodes) but want to keep weights from an earlier finetune. Pass your checkpoint as `--pretrained-path`, use a **new** `--output-dir`, and do **not** pass `--resume` (fresh optimizer + LR schedule).

```bash
uv run python scripts/finetune_pi05.py \
  --dataset-repo-id saivishwak/xlerobot-vr-teleop-medicine-bowl \
  --pretrained-path outputs/pi05_finetune/checkpoints/050000/pretrained_model \
  --output-dir outputs/pi05_finetune_grasp \
  --steps 40000 \
  --save-freq 5000
```

Compare several checkpoints on the robot; you often do not need another full 50k steps if 35k–45k already grasps well.

## How many steps?

| Dataset scale | Suggested `--steps` |
|---------------|---------------------|
| ~50 episodes | `50000` (sweep checkpoints every `5000`) |
| ~100 episodes | `50000` first; extend only if on-robot still improves past 50k |
| Resume same run | Add only `15000`–`25000` **total** steps if LR is already low; prefer warm-start instead |

`--steps` is **total** optimizer steps for the run, not “extra on top of resume” unless you override via `lerobot-train` (see resume section).

## CLI reference (`finetune_pi05.py`)

Wrapper around `lerobot-train`. Defaults for `--dataset-repo-id` come from `config/xlerobot.yaml` → `dataset.repo_id`.

The default rename map keeps the dataset layout compatible with PI0.5 while
leaving `observation.state` and `action` in the project 12-joint order:

```text
observation.images.head        -> observation.images.base_0_rgb
observation.images.left_wrist  -> observation.images.left_wrist_0_rgb
observation.images.right_wrist -> observation.images.right_wrist_0_rgb
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-repo-id` | yaml `dataset.repo_id` | Hugging Face / local LeRobot dataset id |
| `--pretrained-path` | `lerobot/pi05_base` | Base or checkpoint to finetune from (HF id or local `.../pretrained_model`) |
| `--output-dir` | `outputs/pi05_finetune` | Run directory; checkpoints under `<output-dir>/checkpoints/` |
| `--job-name` | `pi05_finetune_xlerobot` | Training job name (logs / W&B) |
| `--steps` | `20000` | **Total** training steps (use `50000` for full finetunes) |
| `--batch-size` | `8` | Training batch size (see **OOM-safe** below) |
| `--num-workers` | `4` | DataLoader workers |
| `--device` | `cuda` | `cuda`, `cpu`, or `mps` |
| `--dtype` | `bfloat16` | `float32`, `float16`, or `bfloat16` |
| `--log-freq` | `100` | Log every N steps |
| `--save-freq` | `5000` | Save checkpoint every N steps |
| `--eval-freq` | `5000` | Run eval every N steps |
| `--rename-map-json` | head/wrist → `*_rgb` | JSON map from dataset image keys to PI0.5 policy keys |
| `--oom-safe` / `--no-oom-safe` | on | Cap `batch_size` at 2 (VRAM); does not change which layers train |
| `--train-expert-only` / `--no-train-expert-only` | on | Freeze vision, train action head only (see below) |
| `--cuda-alloc-conf` | `expandable_segments:True` | Sets `PYTORCH_CUDA_ALLOC_CONF` for the train subprocess |
| `--wandb-enable` | off | Enable Weights & Biases |
| `--push-to-hub` | off | Push policy checkpoints to Hugging Face Hub |
| `--policy-repo-id` | — | HF repo id when `--push-to-hub` (optional) |
| `--resume` | off | Continue **same** run (optimizer + scheduler state from checkpoint) |
| `--resume-from` | — | Checkpoint dir or `train_config.json` (default: `<output-dir>/checkpoints/last/...`) |
| `--dry-run` | off | Print `lerobot-train` command without running |

### Training scope vs batch size

These are **separate** flags:

| Flag | Effect |
|------|--------|
| `--train-expert-only` (default) | `freeze_vision_encoder=true`, `train_expert_only=true` — **action head / expert only** |
| `--no-train-expert-only` | Full finetune (vision + expert) |
| `--oom-safe` (default) | Caps `--batch-size` at **2** |
| `--no-oom-safe` | Uses your `--batch-size` as-is (e.g. 4 or 8) |

**Expert-only with larger batch** (24GB+ GPU):

```bash
uv run python scripts/finetune_pi05.py \
  --no-oom-safe \
  --batch-size 4 \
  --train-expert-only \
  --steps 50000 \
  --save-freq 5000
```

With dataset
```bash
uv run python scripts/finetune_pi05.py \
  --no-oom-safe \
  --batch-size 4 \
  --train-expert-only \
  --dataset-repo-id saivishwak/xlerobot-vr-teleop-medicine-bowl \
  --pretrained-path lerobot/pi05_base \
  --output-dir outputs/pi05_finetune_100ep \
  --steps 50000 \
  --save-freq 5000
```

**Full finetune** (needs more VRAM):

```bash
uv run python scripts/finetune_pi05.py \
  --no-oom-safe \
  --no-train-expert-only \
  --batch-size 2 \
  --steps 50000
```

`gradient_checkpointing` stays enabled in all modes.

### Resume vs `--steps` on the wrapper

`--resume` only passes `--config_path` and `--resume=true` to `lerobot-train`. To change **total** steps, `save_freq`, etc. while resuming, use `lerobot-train` directly (as in the resume example above) or edit the saved `train_config.json`.

### Learning rate

The wrapper does **not** expose LR flags. PI0.5 uses the LeRobot config defaults (cosine schedule, peak `optimizer_lr≈2.5e-5` in `lerobot/pi05` config). Override via `lerobot-train` if needed, e.g. `--policy.optimizer_lr=1e-5`.

### Checkpoint layout

```text
<output-dir>/checkpoints/
  005000/pretrained_model/
  010000/pretrained_model/
  ...
  last/pretrained_model/          # symlink/copy of latest
```

Use `.../last/pretrained_model` for inference and offline eval.

## CLI reference (`eval_pi05_offline.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--policy-path` | — | Path to `.../pretrained_model` (required) |
| `--dataset-repo-id` | `saivishwak/xlerobot-vr-teleop` | Dataset to score against |
| `--episodes` | `0` | Comma-separated episode indices (e.g. `0,1,5`) |
| `--max-samples` | `200` | Max frames to evaluate |
| `--stride` | `1` | Evaluate every N-th frame |
| `--task` | — | Override language prompt (default: frame `task` from dataset) |
| `--device` | `cuda` | `cuda`, `cpu`, or `mps` |
| `--print-worst` | `5` | Print top-K worst frames by max joint error |

## Related docs

- [recording.md](recording.md) — collect data and push to Hub
- [inference.md](inference.md) — on-robot `infer_pi05_finetuned.py` flags (no `--batch-size`; inference is batch 1)
