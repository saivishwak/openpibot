# Data Collection Checklist: Pen -> Jar

Dataset: `saivishwak/xlerobot-vr-pick-place-pen`

Primary eval task:

`Pick up the pen and place it in the jar`

Goal: improve grasp reliability. Keep only clean successful demonstrations.

## 1) Target Data

- [ ] Collect 80-120 successful episodes for the first fine-tune.
- [ ] Use the exact task label above for the primary dataset.
- [ ] Record at 30 Hz with the same 3 cameras: head, left_wrist, right_wrist.
- [ ] Reject failed grasps, pushed pens, dropped pens, camera glitches, and collisions.

## 2) Scene Setup

- [ ] Same desk, robot base position, camera mounts, lighting, pen type, and jar type.
- [ ] Pen and jar fully visible from head camera at episode start.
- [ ] Wrist camera can see the gripper and pen during grasp.
- [ ] Start every episode from verified home.
- [ ] Complete Quest calibration, robot verification, and low-scale test before recording.

## 3) Episode Structure

Every accepted episode must contain:

- [ ] 1-2 s still at home.
- [ ] Move to pen without touching it early.
- [ ] Align gripper around the pen before closing.
- [ ] Close gripper decisively on the pen.
- [ ] Lift pen clear of desk before moving to jar.
- [ ] Move above jar.
- [ ] Lower pen into jar.
- [ ] Open gripper clearly.
- [ ] Retreat after release.

## 4) Collection Mix

Use this mix for 100 episodes:

- [ ] 50 full task episodes: pen starts in normal expected location.
- [ ] 25 grasp-focused full task episodes: slower approach, pause before close, visible lift.
- [ ] 15 pose variation episodes: pen rotated across the desk plane.
- [ ] 10 position variation episodes: pen shifted within the expected workspace.

Do not collect:

- [ ] Failed grasp attempts.
- [ ] Episodes where the pen is only pushed or dragged.
- [ ] Extreme pen/jar positions outside the test setup.
- [ ] Fast jerky motions that saturate joint caps.
- [ ] Episodes with no meaningful gripper motion.

## 5) Grasp Requirements

Accept only if:

- [ ] Gripper fingers are centered near the pen body, not the tip.
- [ ] Gripper closes before lift.
- [ ] Pen visibly leaves the desk.
- [ ] Pen remains held during transfer.
- [ ] Release happens over or inside the jar.

Re-record if:

- [ ] Pen rolls away before grasp.
- [ ] Gripper closes beside the pen.
- [ ] Pen slips during lift.
- [ ] Pen hits jar rim and falls outside.
- [ ] The episode is shorter than a complete task.

## 6) After Recording

- [ ] Confirm `meta/info.json` episode count increased by the expected number.
- [ ] Check every episode has matching parquet rows and video frame counts.
- [ ] Spot-check first, middle, and last episodes visually.
- [ ] Check gripper action spans; reject episodes with near-zero gripper movement.
- [ ] Run dataset preflight before fine-tuning.

## Current Progress

| Task Label | Episodes |
|---|---:|
| Pick up the pen and place it in the jar | 0-1 |
