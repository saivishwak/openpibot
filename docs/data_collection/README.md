# Data Collection Runbook

This directory is the operational guide for collecting the next VLA datasets.
Use it with the dashboard/VR recorder described in [recording.md](../recording.md).

The current baseline dataset must stay frozen:

```text
saivishwak/xlerobot-vr-pick-place-pen
```

Record new data into versioned datasets:

```text
saivishwak/xlerobot-desk-cleanup-phase1
saivishwak/xlerobot-desk-cleanup-phase2
saivishwak/xlerobot-desk-cleanup-phase3
```

## Collection Order

1. [Phase 1: Robust Marker/Pen Pickup](phase1.md)
2. [Phase 2: Desk Cleanup](phase2.md)
3. [Phase 3: Cold Toaster/Bread Skills](phase3.md)

Do not start Phase 2 until Phase 1 passes the marker pickup gate. Do not start
powered toaster work until Phase 3 cold/unplugged tests pass.

## Industry-Aligned Collection Strategy

Do not collect a large fixed quota blindly. Modern VLA work is moving toward a
data engine:

```text
small targeted batch -> train -> real robot eval -> collect failure corrections -> train again
```

The batch counts in the phase files are collection budgets, not mandatory
one-shot targets. Stop early when the acceptance gate is met, and only add data
for the failure mode that still dominates.

For this robot, the first practical cycle should be small:

```text
20 pilot demos
40 grasp-focused demos
20 recovery/regrasp demos
train
evaluate 10-20 held-out trials
```

Then add 20-40 more demos only for the dominant failure mode. This matches the
direction of foundation VLA post-training: use broad pretrained capability,
then spend robot time on targeted in-domain corrections.

## Global Rules

- Record at 30 FPS with the three camera roles: `head`, `left_wrist`, `right_wrist`.
- Keep the robot, cameras, table, cup, and calibration stable within a collection session.
- Keep electronics, keyboard, monitor, cables, and water bottle visible as distractors but outside the robot path.
- Reject unsafe episodes immediately.
- Keep failed autonomous attempts only when the human teleop correction is part of the intended recovery demo.
- Interleave left/right arm usage instead of collecting all demos for one arm in one block.
- Keep held-out physical layouts separate from training data.

## Task String Rule

For continuation-style demos that match the original behavior, use the same
syntax but name the receptacle explicitly:

```text
Pick up the marker from the table and place it inside the white mesh cup
```

For object-specific demos, keep the same syntax and change only the object:

```text
Pick up the blue marker from the table and place it inside the white mesh cup
Pick up the black pen from the table and place it inside the white mesh cup
Pick up the green pen from the table and place it inside the white mesh cup
Pick up the pencil from the table and place it inside the white mesh cup
```

Avoid vague System 1 training prompts such as `Clean the desk` until item-level
skills are reliable. System 2 can later decompose `Clean the desk` into these
bounded prompts.

## Dataset Config

Before recording each phase, update `config/xlerobot.yaml` from the dashboard
Recording page or edit it directly:

```yaml
dataset:
  repo_id: saivishwak/xlerobot-desk-cleanup-phase1
  root: ~/.cache/huggingface/lerobot/saivishwak/xlerobot-desk-cleanup-phase1
  fps: 30
  push_to_hub: false
  home_before_episode: true
```

After each batch:

```bash
uv run python scripts/finetune_pi05.py \
  --dataset-repo-id saivishwak/xlerobot-desk-cleanup-phase1 \
  --dataset-root ~/.cache/huggingface/lerobot/saivishwak/xlerobot-desk-cleanup-phase1 \
  --check-only
```

Upload only after the dataset passes inspection:

```bash
uv run python scripts/push_dataset.py \
  --repo-id saivishwak/xlerobot-desk-cleanup-phase1 \
  --root ~/.cache/huggingface/lerobot/saivishwak/xlerobot-desk-cleanup-phase1
```
