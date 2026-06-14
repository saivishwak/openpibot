# VLA Training Guide (XLerobot)

This document covers the recommended training workflow for your VR teleop dataset.

The training path assumes datasets recorded through the current OpenXR teleop
architecture. For the full runtime contract, see [architecture.md](architecture.md).

## Model choices

| Model | Script | Default output | Default profile |
|-------|--------|----------------|-----------------|
| PI0.5 | `scripts/finetune_pi05.py` | `outputs/pi05_finetune` | vision frozen, expert/action head only |
| MolmoAct2 | `scripts/finetune_molmoact2.py` | `outputs/molmoact2_finetune` | frozen VLM, continuous 12-joint actions |

PI0.5 uses the editable `vendor/lerobot` workspace. MolmoAct2 is available only
in `vendor/allenai-lerobot`; the MolmoAct2 wrapper runs that vendor through an
isolated `PYTHONPATH` and does not modify either vendor tree.

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
- The wrapper defaults `--dataset-root` from `config/xlerobot.yaml` and uses
  `--video-backend pyav`, avoiding local TorchCodec/FFmpeg compatibility issues.
- Before a long run, validate dataset loading and video decode:

```bash
uv run python scripts/finetune_pi05.py \
  --dataset-repo-id saivishwak/xlerobot-vr-teleop-medicine-bowl \
  --check-only
```

- Checkpoints are written under `outputs/pi05_finetune_v2/checkpoints/`.
- Latest checkpoint path is `outputs/pi05_finetune_v2/checkpoints/last/pretrained_model`.

## MolmoAct2 fresh finetune

Use this when you want to fine-tune the regular MolmoAct2 LeRobot policy on the
same XLerobot dataset. The production default is conservative for this robot:
`action_mode=continuous` and `train_mode_vlm=freeze`, so training updates the
action expert for the 12-joint absolute command space while keeping the VLM
frozen.

Before a long run, validate the dataset and print the exact AllenAI LeRobot
training command. `HF_HUB_DISABLE_XET=1` forces the regular Hugging Face
download path, which has been more reliable for the large MolmoAct2 checkpoint
on this setup:

```bash
HF_HUB_DISABLE_XET=1 uv run python scripts/finetune_molmoact2.py \
  --dataset-repo-id saivishwak/xlerobot-desk-cleanup-phase1 \
  --checkpoint-path allenai/MolmoAct2 \
  --check-only
```

Run the full finetune:

```bash
HF_HUB_DISABLE_XET=1 uv run python scripts/finetune_molmoact2.py \
  --dataset-repo-id saivishwak/xlerobot-desk-cleanup-phase1 \
  --checkpoint-path allenai/MolmoAct2 \
  --output-dir outputs/train/molmoact2_xlerobot \
  --job-name molmoact2_xlerobot_phase1 \
  --steps 50000 \
  --batch-size 2 \
  --device cuda \
  --model-dtype bfloat16 \
  --action-mode continuous \
  --train-mode-vlm freeze \
  --chunk-size 50 \
  --n-action-steps 30 \
  --num-flow-timesteps 8 \
  --normalize-gripper \
  --save-freq 5000 \
  --eval-freq 5000 \
  --log-freq 100
```

Notes:
- MolmoAct2 requires the `vendor/allenai-lerobot` submodule and the root
  MolmoAct2 dependencies (`peft`, `scipy`, and `transformers>=5.4,<5.6`).
- `--dry-run` prints the command without requiring CUDA, a dataset, or model
  download. Real training with `--device=cuda` fails fast if CUDA is not
  available.
- The wrapper validates 12-dim `action` and `observation.state`, canonical
  XLerobot joint order, required camera keys, and action-label continuity before
  training.
- XLerobot gripper labels are degree-valued, so MolmoAct2 defaults
  `normalize_gripper=true`. Use `--no-normalize-gripper` only for datasets whose
  gripper channels are already constrained to `[-1, 1]`.
- The released base checkpoint generates 50-step action chunks; the repo
  defaults `chunk_size/action_horizon=50` and `n_action_steps/open_loop_steps=30`
  so inference replans before consuming the full chunk.
- Defaults come from `config/xlerobot.yaml` → `dataset.*` and `molmoact2.*`.
- Latest checkpoint path is
  `outputs/train/molmoact2_xlerobot/checkpoints/last/pretrained_model`.

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

### 3) MolmoAct2 base baseline

Before comparing finetuned checkpoints, run the released base checkpoint against
the same dataset metadata. This gives you a baseline with the exact XLerobot
feature names and normalization stats, without requiring a local
`pretrained_model` directory.

```bash
uv run python scripts/infer_molmoact2_finetuned.py \
  --checkpoint-path allenai/MolmoAct2 \
  --dataset-repo-id saivishwak/xlerobot-vr-teleop-medicine-bowl \
  --task "Pick up the medicine and place it in the bowl" \
  --episodes 1 \
  --episode-time 60 \
  --fps 30
```

### 4) MolmoAct2 on-robot inference

```bash
uv run python scripts/infer_molmoact2_finetuned.py \
  --policy-path outputs/molmoact2_finetune_v2/checkpoints/last/pretrained_model \
  --task "Pick up the medicine and place it in the bowl" \
  --episodes 1 \
  --episode-time 60 \
  --fps 30
```

MolmoAct2 inference reuses the same robot loop, homing, camera backend,
command shaping, deadbands, and safety clamps as PI0.5. It only swaps the policy
loader and processor setup.

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

## MolmoAct2 warm-start from a LeRobot checkpoint

Use `--policy-path` when starting a new run from a MolmoAct2 checkpoint already
saved by LeRobot. Use a new `--output-dir`; do not use `--resume` unless you
want to continue the same optimizer/scheduler state.

```bash
uv run python scripts/finetune_molmoact2.py \
  --dataset-repo-id saivishwak/xlerobot-vr-teleop-medicine-bowl \
  --policy-path outputs/molmoact2_finetune/checkpoints/050000/pretrained_model \
  --output-dir outputs/molmoact2_finetune_grasp \
  --steps 40000 \
  --save-freq 5000
```

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
| `--dataset-root` | yaml `dataset.root` | Local LeRobot dataset root; expanded before launching `lerobot-train` |
| `--video-backend` | `pyav` | Dataset video decoder backend passed to training |
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
| `--check-only` | off | Validate dataset loading/decoding, then exit before training |
| `--skip-dataset-check` | off | Skip the preflight dataset check before training |

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

## CLI reference (`finetune_molmoact2.py`)

Wrapper around AllenAI LeRobot's `python -m lerobot.scripts.lerobot_train`.
The wrapper prepends `vendor/allenai-lerobot/src` to `PYTHONPATH` for the train
subprocess so MolmoAct2 imports cannot accidentally resolve to the PI0.5
workspace vendor.

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-repo-id` | yaml `dataset.repo_id` | Hugging Face / local LeRobot dataset id |
| `--dataset-root` | yaml `dataset.root` | Local LeRobot dataset root |
| `--video-backend` | yaml `molmoact2.video_backend`, `pyav` | Dataset video decoder backend |
| `--checkpoint-path` | yaml `molmoact2.checkpoint_path`, `allenai/MolmoAct2` | Original MolmoAct2 HF checkpoint |
| `--policy-path` | — | LeRobot MolmoAct2 checkpoint to warm-start from |
| `--output-dir` | yaml `molmoact2.output_dir` | Run directory |
| `--job-name` | yaml `molmoact2.job_name` | Training job name |
| `--steps` | yaml `molmoact2.steps`, `20000` | Total training steps |
| `--batch-size` | yaml `molmoact2.batch_size`, `2` | Per-process batch size |
| `--num-workers` | yaml `molmoact2.num_workers`, `4` | DataLoader workers |
| `--device` | yaml `molmoact2.device`, `cuda` | `cuda`, `cpu`, or `mps` |
| `--model-dtype` | yaml `molmoact2.model_dtype`, `bfloat16` | Model load / forward dtype |
| `--action-mode` | yaml `molmoact2.action_mode`, `continuous` | `continuous`, `discrete`, or `both` |
| `--train-mode-vlm` | yaml `molmoact2.train_mode_vlm`, `freeze` | `freeze`, `lora`, or `fft`; `freeze` requires continuous actions |
| `--chunk-size` | yaml `molmoact2.chunk_size`, `50` | MolmoAct2 action chunk horizon |
| `--n-action-steps` | yaml `molmoact2.n_action_steps`, `30` | Actions consumed before re-query |
| `--num-flow-timesteps` | yaml `molmoact2.num_flow_timesteps`, `8` | Flow-matching timesteps |
| `--setup-type` | yaml `molmoact2.setup_type` | Text inserted into MolmoAct2 prompt |
| `--control-mode` | yaml `molmoact2.control_mode` | Text describing action space |
| `--normalize-gripper` / `--no-normalize-gripper` | yaml `molmoact2.normalize_gripper`, `true` | Whether gripper action/state channels are normalized instead of passed through |
| `--image-keys-json` | yaml `molmoact2.image_keys` | JSON list of camera observation keys |
| `--normalization-mapping-json` | yaml `molmoact2.normalization_mapping` | JSON normalization mapping |
| `--wandb-enable` | off | Enable Weights & Biases |
| `--push-to-hub` | off | Push policy checkpoints |
| `--policy-repo-id` | — | HF repo id for pushed policy |
| `--resume` | off | Continue same run from saved train config |
| `--resume-from` | — | Run/checkpoint/train_config path |
| `--dry-run` | off | Print command without validation or training |
| `--check-only` | off | Validate dataset, then exit before training |
| `--skip-dataset-check` | off | Skip preflight before training |

The wrapper intentionally does not expose a loose passthrough. Add new flags to
the wrapper when a production run needs them so defaults remain auditable.

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
- [inference.md](inference.md) — on-robot PI0.5 and MolmoAct2 inference flags (no `--batch-size`; inference is batch 1)
