# Phase 1: Robust Marker/Pen Pickup

Target: fix marker/pen grasp acquisition. Current behavior suggests arm choice
and transport after a successful grasp are mostly okay; the weak point is
spending too long around the marker and failing when the marker is not in a
convenient grasp pose.

Phase 1 should produce about 220 new demonstrations. Start with a pilot batch and
only continue if the dataset checks and visual inspection look correct. Bias the
data toward pre-grasp alignment, decisive close, lift, and recovery/regrasp.

Dataset:

```text
saivishwak/xlerobot-vr-pick-place-pen-phase1
```

Local root:

```text
~/.cache/huggingface/lerobot/saivishwak/xlerobot-vr-pick-place-pen-phase1
```

## Acceptance Gate

Before moving to Phase 2:

- At least 80% success on 30 held-out marker/pen poses.
- No object repositioning after episode start.
- No unsafe contact with electronics, keyboard, monitor, bottle, or cables.
- Failures are not dominated by one repeated issue such as bad grasp alignment.

## Task Strings

Use the original full-task string for about half of this phase:

```text
Pick up the marker from the table and place it inside the white mesh cup
```

Add pickup-only strings for grasp-focused episodes:

```text
Pick up the marker from the table
Pick up the blue marker from the table
Pick up the black pen from the table
Pick up the pencil from the table
```

Use object-specific full-task strings for object variation:

```text
Pick up the blue marker from the table and place it inside the white mesh cup
Pick up the black pen from the table and place it inside the white mesh cup
Pick up the green pen from the table and place it inside the white mesh cup
Pick up the pencil from the table and place it inside the white mesh cup
Pick up the thick marker from the table and place it inside the white mesh cup
Pick up the thin marker from the table and place it inside the white mesh cup
```

Do not use:

```text
Clean the desk
Pick up the object
Move it
```

## Batch 1: 20 Pilot Demos

Purpose: validate the new dataset root, task strings, camera roles, calibration,
and operator quality before collecting the grasp-focused set.

Setup:

- Same marker/white-mesh-cup arrangement style as the original 80 episodes.
- Normal lighting.
- Minimal clutter.
- White mesh cup fully visible from head camera.
- Marker fully visible from head and at least one wrist camera during grasp.

Collect:

- 10 demos with the marker in the normal expected position.
- 5 demos with the marker slightly rotated.
- 5 demos with the marker shifted left/right/front/back.

Task string:

```text
Pick up the marker from the table and place it inside the white mesh cup
```

Accept if:

- Gripper aligns around the marker body before closing.
- Marker leaves the table cleanly.
- Marker is carried above the white mesh cup and released inside the white mesh cup.
- Episode has stable video from all three cameras.

Reject and re-record if:

- Marker is pushed into the white mesh cup instead of lifted.
- The gripper closes beside the marker.
- The marker is manually repositioned during recording.
- Any camera frame stream is unstable.
- The robot contacts a distractor object.

After the batch:

```bash
uv run python scripts/finetune_pi05.py \
  --dataset-repo-id saivishwak/xlerobot-vr-pick-place-pen-phase1 \
  --dataset-root ~/.cache/huggingface/lerobot/saivishwak/xlerobot-vr-pick-place-pen-phase1 \
  --check-only
```

Also visualize the first, middle, and last pilot episodes.

## Batch 2: 100 Grasp-Focused Pose Demos

Purpose: teach the hard part directly: approach, open gripper, align around the
marker body, close decisively, and lift. These episodes should make the grasp
phase visible and high quality.

Use pickup-only task strings for 40-60 demos:

```text
Pick up the marker from the table
Pick up the blue marker from the table
Pick up the black pen from the table
Pick up the pencil from the table
```

Use the original full-task string for the remaining demos:

```text
Pick up the marker from the table and place it inside the white mesh cup
```

Collect roughly 100 demos:

- 25 horizontal marker/pen demos.
- 25 vertical marker/pen demos.
- 25 diagonal marker/pen demos, lower-left to upper-right.
- 25 diagonal marker/pen demos, upper-left to lower-right.

Within those 100 demos, distribute object positions:

- Near white mesh cup.
- Far from white mesh cup.
- Front-center.
- Left side of reachable workspace.
- Right side of reachable workspace.
- Slightly behind the white mesh cup but still visible and reachable.
- Slightly off-center for the gripper.
- Awkward but reachable angles where the operator must choose the approach side.

Important collection rules:

- Include awkward but reachable poses.
- Keep the marker fully visible at episode start.
- Do not use extreme positions outside the normal desk workspace.
- Interleave left-arm and right-arm demos.
- Use the arm that is most natural for the object pose, but keep both arms represented.
- Open the gripper clearly before final approach.
- Center the gripper on the marker body, not the cap or tip.
- Close once decisively after alignment.
- Lift mostly vertically after closing.
- For pickup-only demos, end after a visible stable lift/hold. Do not rush into cup placement.

Accept if:

- The operator deliberately approaches the marker from a useful side.
- Gripper closes on the body, not the tip.
- Marker is lifted cleanly and held briefly.
- For full-task demos, marker is lifted before translation to the white mesh cup.

Reject if:

- Marker rolls out of the original reachable area and the episode becomes a chase.
- The gripper scrapes the table for a long time before grasp.
- The marker is only dragged.
- The pickup-only episode ends before a stable lift is visible.

## Batch 3: 60 Recovery/Regrasp Demos

Purpose: teach the policy what to do when the first grasp attempt is imperfect.
These episodes should still end in success, but they should include one visible
correction before success.

Use pickup-only and full-task strings:

```text
Pick up the marker from the table
Pick up the blue marker from the table
Pick up the black pen from the table
Pick up the marker from the table and place it inside the white mesh cup
```

Collect roughly:

- 15 close-beside-marker demos: reopen, realign, close correctly.
- 15 marker-rolls-slightly demos: follow the new pose, then grasp.
- 15 weak-grip demos: lower, regrip, lift.
- 15 bad-approach-angle demos: back off, approach from a better side, grasp.

Rules:

- The recovery must be produced by teleop, not by manually moving the object.
- Do not record unsafe recovery attempts.
- Keep recovery attempts short and purposeful.
- Label `retry_count` in the metadata.
- Set `failure_mode` to the initial issue, even if the episode succeeds after recovery.
- Do not intentionally create violent failures. The goal is corrective servoing, not chaos.

Accept if:

- The correction is visible and physically plausible.
- The final grasp succeeds.
- The marker is lifted and held briefly.
- There is no unsafe contact.

Reject if:

- The object is moved by hand.
- Recovery becomes a long uncontrolled chase.
- The robot contacts the white mesh cup or distractors hard enough to move or tip them.

## Batch 4: 40 Full Pick-And-Place Demos

Purpose: keep the complete skill connected after the grasp-focused batches:
grasp, lift, transport, release into white mesh cup.

Use the original full-task string for most demos:

```text
Pick up the marker from the table and place it inside the white mesh cup
```

Use object-specific full-task strings for a minority:

```text
Pick up the blue marker from the table and place it inside the white mesh cup
Pick up the black pen from the table and place it inside the white mesh cup
Pick up the pencil from the table and place it inside the white mesh cup
```

Collect roughly:

- 20 normal full pick-and-place demos.
- 10 awkward-pose full pick-and-place demos.
- 10 object-variation full pick-and-place demos.

Accept if:

- The grasp follows the same quality standard as Batch 2.
- Marker is lifted before moving to the white mesh cup.
- Transport and release are clean.
- The white mesh cup target is not hit hard.

Reject if:

- The marker is pushed/dragged into the white mesh cup.
- Placement succeeds only because the marker was already in a convenient pose.
- The gripper never clearly closes on the marker.

## Optional Batch 5: 40 Object And Clutter Demos

Purpose: add more object identity and desk robustness only after grasp acquisition
improves. Skip this batch initially if Batch 2/3 are not yet enough.

Use a mix of original and object-specific strings.

Add safe distractors:

- Keyboard visible.
- Monitor visible.
- White mesh cup with other markers inside.
- Cloth nearby.
- Papers or notebook nearby.
- Water bottle visible but outside path.
- Electronics visible but outside path.

Collect roughly:

- 10 blue marker / black pen / green pen / pencil object-variation demos.
- 10 thick marker / thin marker demos.
- 10 light clutter demos.
- 5 partial occlusion demos, such as marker near cloth/paper.
- 5 different lighting or monitor-glare demos.

Accept if:

- The target object remains identifiable.
- The robot path avoids distractors.
- The object is still picked and placed into the white mesh cup.

Reject if:

- The target object is mostly hidden.
- The robot contacts electronics, keyboard, monitor, water bottle, or cables.
- Lighting makes the camera image unusable.

## Phase 1 Training Check

Train after Batch 2 + Batch 3 if the dataset passes inspection. Do not wait for
Phase 2; the current bottleneck is grasp acquisition.

```bash
uv run python scripts/finetune_pi05.py \
  --dataset-repo-id saivishwak/xlerobot-vr-pick-place-pen-phase1 \
  --dataset-root ~/.cache/huggingface/lerobot/saivishwak/xlerobot-vr-pick-place-pen-phase1 \
  --output-dir outputs/pi05_marker_phase1 \
  --steps 30000
```

If the failure is still visual grounding, train a full fine-tune only after the
dataset is diverse enough:

```bash
uv run python scripts/finetune_pi05.py \
  --dataset-repo-id saivishwak/xlerobot-vr-pick-place-pen-phase1 \
  --dataset-root ~/.cache/huggingface/lerobot/saivishwak/xlerobot-vr-pick-place-pen-phase1 \
  --output-dir outputs/pi05_marker_phase1_full \
  --steps 30000 \
  --no-train-expert-only \
  --no-oom-safe \
  --batch-size 4
```
