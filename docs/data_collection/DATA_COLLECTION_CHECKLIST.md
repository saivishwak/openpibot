# Data Collection Checklist

Use the phase runbooks in this directory as the source of truth:

- [README](README.md)
- [Phase 1: Robust Marker/Pen Pickup](phase1.md)
- [Phase 2: Desk Cleanup](phase2.md)
- [Phase 3: Cold Toaster/Bread Skills](phase3.md)

## Current Strategy

Do not collect large fixed quotas upfront. Use an iterative data engine:

```text
small targeted batch -> train -> real robot eval -> collect failure corrections -> train again
```

The phase batch counts are upper budgets, not mandatory one-shot targets.

## Immediate Phase 1 Cycle

Start with:

```text
20 pilot demos
40 grasp-focused pose demos
20 recovery/regrasp demos
train
evaluate 10-20 held-out marker poses
```

Only add more data for the dominant failure mode.

## Current Bottleneck

Observed behavior:

- Arm choice is mostly correct.
- Transport after a successful grasp is mostly okay.
- The weak point is marker/pen grasp acquisition from imperfect poses.

Therefore Phase 1 should prioritize:

- pickup-only demos,
- pre-grasp alignment,
- decisive close,
- vertical lift,
- recovery/regrasp after a bad first attempt.

Use `white mesh cup` in future task strings:

```text
Pick up the marker from the table and place it inside the white mesh cup
```

Keep the historical baseline dataset frozen:

```text
saivishwak/xlerobot-vr-pick-place-pen
```

## Data Collection Records

### Phase 1 (150 episodes)

Dataset:

```text
saivishwak/xlerobot-desk-cleanup-phase1
```

Local root:

```text
~/.cache/huggingface/lerobot/saivishwak/xlerobot-desk-cleanup-phase1
```

#### Batch 1: 20 Pilot demos 

Status: Complete

Total 20 Demos, 10 with marker in normal position, 5 with slightly rotated, 5 with displacement left/right/front/back 

Task string:

```text
Pick up the marker from the table and place it inside the white mesh cup
```

#### Batch 2: 80 Grasp-Focused Pose Demos

Status: Complete 

Task strings (First 50 episodes):

```text
Pick up the marker from the table (20 episodes)
Pick up the blue marker from the table (10 episodes)
Pick up the black marker from the table (10 episodes)
Pick up the pencil from the table (10 episodes)
```

Task string (second 30 episodes):

```text
Pick up the marker from the table and place it inside the white mesh cup
```

#### Batch 3: 30 Recovery/Regrasp Demos

Status: Complete 

Task strings (30 episodes):

```text
Pick up the marker from the table (15 episodes)
Pick up the marker from the table and place it inside the white mesh cup (15 episodes)
```

#### Batch 4: 20 Object And Clutter Demos

Status: Complete 

Task strings (20 episodes):

```text
Pick up the blue marker from the table and place it inside the white mesh cup (10 episodes)
Pick up the black marker from the table and place it inside the white mesh cup (10 episodes)
```
