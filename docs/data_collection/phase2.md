# Phase 2: Desk Cleanup

Target: clean the desk by composing reliable item-level skills. Phase 2 should
not start until Phase 1 passes the marker/pen pickup gate.

Dataset:

```text
saivishwak/xlerobot-vr-desk-cleanup-phase2
```

Local root:

```text
~/.cache/huggingface/lerobot/saivishwak/xlerobot-vr-desk-cleanup-phase2
```

## Acceptance Gate

Before using `Clean the desk` as an autonomous System 2 goal:

- Pen/marker item-level cleanup is at least 80% successful.
- Cloth move is at least 70% successful.
- Full staged cleanup of 3-4 items is at least 70% successful on held-out layouts.
- No unsafe contact with keyboard, monitor, electronics, water bottle, or cables.

## Phase 2 Task Strategy

Do not begin with:

```text
Clean the desk
```

Also do not re-collect Phase 1 under a new dataset name. Phase 1 owns basic
pen/marker-to-cup robustness. Phase 2 should only include a small bridge set for
pen/marker cleanup inside a denser desk scene, then focus on new cleanup targets:
cloth, safe tray objects, and staged multi-step cleanup.

Use the same data-engine loop as Phase 1. Do not record the whole Phase 2 budget
in one pass. Start with:

```text
30 pilot cleanup demos
20 pen/marker bridge demos
40 cloth demos
train combined Phase 1 + Phase 2
evaluate staged cleanup
```

Only add safe-object/tray and longer staged cleanup data after the item-level
skills pass held-out evaluation.

Use these Phase 2 task families:

```text
Put the leftmost blue marker into the white mesh cup
Put the front black pen into the white mesh cup
Put the rightmost pencil into the white mesh cup
Move the cloth from the table to the clear area on the right side of the desk
Pick up the cloth from the table and place it in the tray
Move the small safe object from the table into the tray
```

The System 2 agent can later convert `Clean the desk` into a sequence of these
prompts.

## Batch 1: 30 Desk Cleanup Pilot Demos

Purpose: validate the new Phase 2 dataset and make sure Phase 1 pickup behavior
still works when the desk has more objects. This is a bridge batch, not another
marker pickup dataset.

Scene:

- White mesh cup remains the target for pens/markers.
- 2-3 safe movable objects visible.
- Keyboard, monitor, water bottle, and electronics visible as no-touch distractors.
- No powered appliances.

Collect:

- 5 pen/marker-to-white-mesh-cup bridge demos in clutter.
- 10 cloth-to-clear-area demos.
- 10 cloth-to-tray demos.
- 5 small safe-object-to-tray demos.

Bridge task strings:

```text
Put the leftmost blue marker into the white mesh cup
Put the front black pen into the white mesh cup
Put the rightmost pencil into the white mesh cup
```

Accept if:

- The named object is manipulated.
- The target location is unambiguous.
- Other desk items are not touched.

Reject if:

- The operator improvises a different cleanup target.
- The robot sweeps multiple objects accidentally.
- The task string does not match the manipulated object.

## Batch 2: 20-40 Pen/Marker Cleanup Bridge Demos

Purpose: confirm Phase 1 pen/marker pickup transfers into cleanup scenes with
multiple visible objects. Keep this small; do not duplicate Phase 1.

Use location-disambiguated cleanup strings instead of the Phase 1 canonical
pickup strings:

```text
Put the leftmost blue marker into the white mesh cup
Put the front black pen into the white mesh cup
Put the green pen nearest the keyboard into the white mesh cup
Put the rightmost pencil into the white mesh cup
```

First cycle: collect 20 demos. Continue up to 40 only if Phase 1 pickup fails
specifically because multiple similar objects are visible.

Full budget mix:

- 10 leftmost/rightmost target demos.
- 10 front/back target demos.
- 10 near-object target demos, such as `nearest the keyboard` or `nearest the white mesh cup`.
- 10 mixed visible pen/marker demos with 3-4 pen-like objects on the table.

Scene variation:

- Put 2-4 pen-like objects on the desk at once.
- Keep one target clearly named by color, type, and location.
- Vary object order: front, back, left, right, near white mesh cup, far from white mesh cup.
- Keep distractors visible but not in the grasp path.

Accept if:

- The correct object is chosen even when similar objects are visible.
- The robot places the object inside the white mesh cup.
- The remaining objects are not displaced significantly.

Reject if:

- The wrong object is picked.
- The task string is ambiguous, such as two black pens with no disambiguation.
- The robot uses a sweeping motion instead of pick-and-place.

Use disambiguated task strings when needed. Avoid the generic Phase 1 form unless
you are intentionally collecting a small compatibility example.

```text
Put the front black pen into the white mesh cup
Put the leftmost pencil into the white mesh cup
```

## Batch 3: 40-120 Cloth Demos

Purpose: add a non-rigid desk object without mixing it into long-horizon cleanup
too early.

Task strings:

```text
Move the cloth from the table to the clear area on the right side of the desk
Pick up the cloth from the table and place it in the tray
Move the folded cloth from the table to the tray
```

First cycle: collect 40 demos. Continue up to 120 only if cloth handling is the
dominant blocker for cleanup.

Full budget mix:

- 40 move-cloth-to-clear-area demos.
- 40 cloth-to-tray demos.
- 20 folded/partly folded cloth demos.
- 20 cloth-near-pens demos where the cloth is moved without dragging pens.

Scene variation:

- Cloth flat.
- Cloth bunched.
- Cloth partly folded.
- Cloth near pens but not covering them.
- Cloth near white mesh cup but not blocking the cup opening.

Accept if:

- Cloth moves to the requested target.
- Robot avoids dragging pens or cup with the cloth.
- Cloth remains visible from head camera at start.

Reject if:

- Cloth is wrapped around the gripper.
- The cloth drags electronics/cables.
- The target area is not visible or not clearly defined.

## Batch 4: 30-100 Safe Object To Tray Demos

Purpose: broaden cleanup beyond pens/cloth while staying in the desk domain.

Only use safe lightweight objects:

- Eraser.
- Sticky note pad.
- Small empty box.
- Safe plastic cap.
- Small notebook only if the robot can move it reliably.

Task strings:

```text
Move the eraser from the table into the tray
Move the sticky note pad from the table into the tray
Move the small box from the table into the tray
```

Start with 30 demos across 2-3 safe objects. Continue up to 100 only after cloth
and pen/marker cleanup work reliably.

Full budget mix:

- 25 eraser demos.
- 25 sticky note pad demos.
- 25 small box demos.
- 25 mixed safe-object demos.

Accept if:

- Object is safe, light, and graspable.
- The object is moved to the tray without contacting no-touch items.

Reject if:

- Object is heavy, sharp, breakable, powered, or connected by cable.
- The object has no stable grasp for the current gripper.

## Batch 5: 20-80 Staged Multi-Step Cleanup Demos

Purpose: prepare System 2 execution where the agent runs several item-level VLA
prompts until the desk is clean.

Do not record this as one vague prompt at first. Record one episode per step.

Example staged layout:

```text
1. Put the leftmost blue marker into the white mesh cup
2. Put the front black pen into the white mesh cup
3. Move the cloth from the table to the clear area on the right side of the desk
4. Move the eraser from the table into the tray
```

Start with 20 short two-item sequences. Continue up to 80 only after individual
item-level skills work and System 2 execution exposes sequencing failures.

Full budget mix:

- 20 two-item cleanup sequences, recorded as separate episodes per item.
- 25 three-item cleanup sequences, recorded as separate episodes per item.
- 25 four-item cleanup sequences, recorded as separate episodes per item.
- 10 mixed recovery sequences where one item is nudged but the next step remains safe.

Metadata rule:

- Add `sequence_id`, `sequence_step`, and `sequence_total` to the JSONL sidecar.

Example:

```json
{"episode_index": 144, "sequence_id": "cleanup_layout_012", "sequence_step": 2, "sequence_total": 4, "task": "Put the front black pen into the white mesh cup", "object_type": "pen", "object_color": "black", "success": true}
```

Accept if:

- Each step leaves the scene ready for the next step.
- The target object remains clear from the task string.
- The sequence can be executed by System 2 as separate VLA calls.

Reject if:

- The step depends on hidden operator knowledge.
- A previous step accidentally changes the next target too much.
- The cleanup meaning is ambiguous.

## Later Phase 2: Long-Horizon Cleanup Prompts

Only after item-level and staged cleanup work, collect a smaller set of
long-horizon demos:

```text
Clean the desk by placing all visible pens and markers inside the white mesh cup
Clear the cloth from the workspace and place all visible pens in the white mesh cup
```

These should be used cautiously. The main production path should still be System
2 planning plus bounded System 1 execution.

## Phase 2 Training Check

Train after the first Phase 2 cycle with Phase 1 + Phase 2 data or with a
curated combined dataset. Keep a held-out physical layout set for real robot
evaluation.

```bash
uv run python scripts/finetune_pi05.py \
  --dataset-repo-id saivishwak/xlerobot-vr-desk-cleanup-phase2 \
  --dataset-root ~/.cache/huggingface/lerobot/saivishwak/xlerobot-vr-desk-cleanup-phase2 \
  --output-dir outputs/pi05_desk_cleanup_phase2 \
  --steps 40000
```

If one task dominates the dataset, rebalance collection before training more.
