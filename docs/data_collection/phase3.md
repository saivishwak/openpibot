# Phase 3: Cold Toaster/Bread Skills

Target: add bread and toaster manipulation in the same desk environment. This
phase is for a cold, unplugged toaster only.

Dataset:

```text
saivishwak/xlerobot-vr-cold-toaster-phase3
```

Local root:

```text
~/.cache/huggingface/lerobot/saivishwak/xlerobot-vr-cold-toaster-phase3
```

## Safety Gate

Do not use a powered or hot toaster in this phase.

Required setup:

- Toaster unplugged.
- Toaster cold.
- No loose cable in the robot workspace.
- Toaster stable and not sliding.
- Bread is room-temperature and safe to handle.
- Operator has a physical emergency stop path.

Reject any episode where the robot tips the toaster, traps the gripper, contacts
a cable, or uses excessive force on the lever.

## Phase 3 Task Strategy

Collect atomic skills before full bread-to-toaster sequences.

Good task strings:

```text
Pick up the bread slice from the table
Move the bread slice above the toaster slot
Insert the bread slice into the cold toaster slot
Press the cold toaster lever down
Pick up the bread slice from the table and place it into the cold toaster slot
```

Avoid:

```text
Make toast
Toast the bread
Use the toaster
```

Those are System 2 goals, not initial System 1 training prompts.

## Batch 1: 20 Cold Toaster Pilot Demos

Purpose: validate reachability, camera visibility, and safety with the toaster
present.

Scene:

- Cold unplugged toaster placed in a stable position.
- Slot visible from the head camera.
- At least one wrist camera sees bread alignment near the slot.
- No cup/pen clutter in the immediate toaster path.

Collect:

- 10 bread pickup demos.
- 5 bread-above-slot demos.
- 5 insert-bread demos.

Task strings:

```text
Pick up the bread slice from the table
Move the bread slice above the toaster slot
Insert the bread slice into the cold toaster slot
```

Accept if:

- Bread is grasped without tearing badly.
- Toaster does not move significantly.
- Gripper does not enter the slot deeply.

Reject if:

- Toaster slides or tips.
- Bread folds around the gripper and blocks visibility.
- Robot contacts the toaster with high force.

## Batch 2: 80 Bread Pickup And Handling Demos

Purpose: learn robust bread grasping before aiming at the toaster.

Task string:

```text
Pick up the bread slice from the table
```

Collect roughly:

- 20 bread horizontal/front demos.
- 20 bread rotated demos.
- 20 bread shifted left/right demos.
- 20 bread with mild nearby distractors.

Bread variation:

- Full slice.
- Half slice if safe.
- Slightly thicker slice.
- Slightly different crust orientation.

Accept if:

- Bread is lifted clearly off the table.
- Bread remains held for at least a short pause.
- Grasp does not crush the bread excessively.

Reject if:

- Bread tears or folds so much that the skill no longer matches the goal.
- Gripper closes on empty space.
- Robot drags bread across unsafe items.

## Batch 3: 80 Bread Alignment Over Slot Demos

Purpose: learn positioning above the toaster before insertion.

Task string:

```text
Move the bread slice above the toaster slot
```

Collect roughly:

- 40 demos from bread already grasped or immediately after pickup.
- 40 demos starting with bread on the table, then pick and align.

Toaster variation:

- Slot front-back.
- Slot slightly diagonal.
- Toaster slightly left/right within safe reach.
- Different bread orientation relative to slot.

Accept if:

- Bread is held above the slot with visible clearance.
- Bread orientation roughly matches the slot.
- Robot pauses above the slot before ending the episode.

Reject if:

- Bread hits the toaster side hard.
- Bread is not aligned enough to imply insertion would be possible.
- The toaster moves.

## Batch 4: 80 Cold Insertion Demos

Purpose: learn the contact-rich final insertion skill.

Task string:

```text
Insert the bread slice into the cold toaster slot
```

Collect roughly:

- 40 insertions from an already aligned position.
- 40 pickup-and-insert demos.

Rules:

- Move slowly near the slot.
- Keep gripper shallow. Do not drive fingers deep into the slot.
- Release only after the bread is stable enough in the slot.
- Stop if the toaster shifts.

Accept if:

- Bread enters the slot.
- Toaster remains stable.
- Gripper releases cleanly.

Reject if:

- Bread jams and requires force.
- Toaster tips, slides, or rotates significantly.
- Gripper collides with the slot edges in a way that could damage hardware.

## Batch 5: 40 Cold Lever Demos

Purpose: learn the lever skill separately, only if the cold toaster mechanics
are safe and reachable.

Task string:

```text
Press the cold toaster lever down
```

Collect:

- 20 approach-and-press demos.
- 20 varied toaster pose demos.

Accept if:

- The robot presses the lever with controlled motion.
- The toaster remains stable.
- There is no need for excessive force.

Reject if:

- Lever requires too much force.
- The robot pushes the toaster instead of the lever.
- The gripper or wrist gets trapped.

If lever pressing is unreliable, remove it from Phase 3 and keep the toaster
task limited to bread insertion.

## Batch 6: 40 Full Cold Bread-To-Toaster Demos

Purpose: combine pickup, alignment, insertion, and release after the atomic
skills pass.

Task string:

```text
Pick up the bread slice from the table and place it into the cold toaster slot
```

Collect:

- 20 normal layout demos.
- 10 shifted toaster demos.
- 10 shifted bread demos.

Accept if:

- Bread is picked up.
- Bread is aligned above the slot.
- Bread is inserted and released.
- Toaster remains stable.

Reject if:

- The sequence works only because the operator manually adjusts the toaster or bread.
- Bread is forced into the slot.
- Any powered-appliance risk appears.

## Phase 3 Training Check

Train and evaluate only on cold/unplugged tasks first:

```bash
uv run python scripts/finetune_pi05.py \
  --dataset-repo-id saivishwak/xlerobot-vr-cold-toaster-phase3 \
  --dataset-root ~/.cache/huggingface/lerobot/saivishwak/xlerobot-vr-cold-toaster-phase3 \
  --output-dir outputs/pi05_cold_toaster_phase3 \
  --steps 40000
```

Promotion gate:

- 80% cold bread insertion success on held-out toaster/bread poses.
- 0 unsafe contacts in the held-out test set.
- Separate written safety plan before any powered toaster experiment.
