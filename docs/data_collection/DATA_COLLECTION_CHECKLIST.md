# Data Collection Checklist

Dataset: `saivishwak/xlerobot-vr-pick-place-pen`

Local root:
`~/.cache/huggingface/lerobot/saivishwak/xlerobot-vr-pick-place-pen`

Current policy goal: make the desk-domain PI0.5 VLA robust for item cleanup first, then expand to cold toaster/bread manipulation. The System 2 agent should decompose long tasks, but System 1 must learn reliable short physical skills from diverse data.

## Current Dataset Audit

- Format: LeRobot v3.0.
- Robot: `xlerobot-bimanual-so101`.
- Cameras: `observation.images.head`, `observation.images.left_wrist`, `observation.images.right_wrist`.
- State/action: 12 joints, left arm then right arm.
- FPS: 30.
- Episodes: 80.
- Frames: 42,439.
- Task labels: 1 unique task, `Pick up the marker from the table and place it inside the cup`.
- Episode duration: mean 17.68s, median 16.98s, p90 22.62s, max 34.87s.
- Active gripper split: 40 right-arm episodes, then 40 left-arm episodes. The arms are balanced in count, but collected in two contiguous blocks rather than interleaved.
- First gripper close: mean 5.29s, median 4.92s, p90 7.61s.
- Closed-gripper duration: mean 3.60s, median 3.43s, p90 4.60s.
- Start-pose variation is very low: mean joint std 0.30 deg, max joint std 0.87 deg.
- Grasp/release variation is much larger, so the model sees varied mid-task poses but not enough varied starting scenes.
- Max one-frame action deltas are within the expected recording caps: shoulder/elbow about 2.5-2.9 deg, wrist flex 6 deg, wrist roll about 8.8 deg, gripper 15 deg.

## Diagnosis

The current data is enough to learn a narrow pick-and-place routine, but not enough for reliable autonomous pickup from arbitrary desk placements.

- The dataset has one language label, so the policy cannot learn object-level grounding such as `blue marker`, `black pen`, `cloth`, `cup`, `tray`, or `toaster slot`.
- The dataset mostly teaches a single long behavior under one prompt. The VLA has to infer hidden stages: approach marker, open gripper, align around pen, close, lift, move to cup, release.
- The policy succeeds when the operator places the marker appropriately, which points to missing coverage around pen pose, angle, distance, occlusion, gripper approach angle, and recovery after a bad grasp.
- The start distribution is too tight. The robot starts from nearly the same arm configuration and likely sees a limited set of object layouts.
- The right-arm and left-arm data blocks are contiguous. Future data should interleave arms and scene variants to avoid run-order bias.
- Current fine-tuning defaults train the PI0.5 expert/action path only. That is useful for low VRAM, but new object/scene distributions may require unfreezing more visual/language layers after enough diverse data is collected.

## Research Notes

- PI0.5/OpenPI and similar VLA systems rely on broad co-training mixtures, not just one narrow task. Their useful ingredients are heterogeneous robot tasks, language commands, semantic/stage prediction, object detections, and low-level actions.
- Figure Helix uses a System 2/System 1 split: a slower VLM handles scene/language understanding and a fast visuomotor policy handles continuous control. Their data recipe emphasizes high-quality diverse teleoperation and hindsight language labels from video clips.
- DROID, Open X-Embodiment, Octo, and OpenVLA all point in the same direction: generalization comes from many scenes, objects, tasks, operators, and held-out evaluations, then fine-tuning into the target robot setup.
- LeRobot now supports subtask annotations. Use them for the exact hidden stages in the marker task so we can train or evaluate stage-aware behavior instead of relying only on a single episode-level task string.
- Human-in-the-loop collection is important here. Record failures, corrections, and retries because the current policy fails from states that are not present in clean demonstrations.

## Phase 0: Baseline Before New Data

Record an intervention-free benchmark before adding data.

- Run 30-50 real robot trials for `Pick up the blue marker from the desk and place it inside the white mesh cup`.
- Vary marker angle, distance from cup, horizontal/vertical/diagonal orientation, left/right side of workspace, and mild clutter.
- Do not reposition the marker after the episode starts.
- Log every trial with:
  - `success`: yes/no
  - `failure_mode`: missed_object, bad_grasp, pushed_object, dropped_object, wrong_receptacle, collision_risk, timeout, other
  - `object_pose_bin`: near_cup, center, far, left, right, angled, partly_occluded
  - `arm`: left/right/bimanual
  - `lighting`: normal/dim/bright
  - `clutter_level`: none/light/medium
  - `operator_intervention`: none/positioned_object/stopped_robot/recovered
- Keep 10-15 held-out physical layouts for later evaluation. Do not train on these layouts.

## Phase 1: Robust Marker/Pen Pickup

Target: fix marker/pen grasp acquisition. Current behavior suggests arm choice and transport after a successful grasp are mostly okay; the weak point is pre-grasp alignment, decisive close, lift, and recovery/regrasp.

- Collect about 220 additional demonstrations for pens/markers.
- Prioritize grasp-focused and pickup-only demos before broad cleanup data:
  - 20 pilot demos.
  - 100 grasp-focused pose demos.
  - 60 recovery/regrasp demos.
  - 40 full pick-and-place demos.
  - Optional 40 object/clutter demos after grasp improves.
- Use pickup-only strings for some grasp-focused episodes:
  - `Pick up the marker from the table.`
  - `Pick up the blue marker from the table.`
  - `Pick up the black pen from the table.`
  - `Pick up the pencil from the table.`
- Keep the original full-task string for most full pick-and-place demos:
  - `Pick up the marker from the table and place it inside the white mesh cup.`
- Vary object pose heavily during grasp-focused demos:
  - angle: 0/30/60/90/120/150 degrees
  - location: left, right, center, near cup, far from cup
  - distance from cup: close, medium, far
  - approach side: left arm, right arm, whichever is nearest
  - clutter: keyboard/monitor/bottle/electronics present as distractors but not touched
  - visibility: fully visible, partly occluded by another safe object
  - lighting: normal, dim, brighter monitor glare
- Interleave left/right arm episodes instead of collecting all right-arm then all left-arm data.
- Add recovery demonstrations:
  - gripper misses the marker, reopens, re-aligns, closes again
  - marker is nudged, robot re-approaches from the new pose
  - weak grasp, robot lowers, regrips, then lifts
  - bad approach angle, robot backs off and approaches from a better side
- Keep successful clean demos and successful recovery demos. Do not keep unsafe contacts.
- Do not manually reposition the object during the recorded autonomous part unless the episode is explicitly labeled as intervention/recovery data.

## Phase 2: Desk Cleanup Skills

Target: clean a desk with several pens, a cloth, and safe movable objects.

- Collect about 350-450 cleanup demonstrations after Phase 1 reaches the marker acceptance gate.
- Do not re-collect Phase 1 under a new dataset name. Phase 2 should include only a small pen/marker bridge set in clutter, then focus on cloth, tray objects, and staged cleanup.
- Break cleanup into item-level episodes first:
  - `Put the leftmost blue marker into the white mesh cup.`
  - `Put the front black pen into the white mesh cup.`
  - `Put the rightmost pencil into the white mesh cup.`
  - `Move the cloth from the table to the clear area on the right side of the desk.`
  - `Pick up the cloth from the table and place it in the tray.`
  - `Move the eraser from the table into the tray.`
  - `Move the sticky note pad from the table into the tray.`
- Add long-horizon cleanup episodes only after the item-level skills work:
  - `Clean the desk by placing all visible pens and markers inside the white mesh cup.`
  - `Clear the cloth from the workspace and place all visible pens in the white mesh cup.`
- Keep electronics, keyboard, monitor, cables, and water bottle as visible no-touch distractors.
- Do not collect ambiguous cleanup episodes where the operator would not know what `clean` means from the image alone. The agent can plan `clean the desk`, but System 1 prompts should stay object-specific.

## Phase 3: Cold Toaster/Bread Skills

Target: add bread/toaster manipulation in the same desk environment without unsafe appliance behavior.

- Use a cold, unplugged toaster for all initial data.
- Keep the toaster stable and positioned so the slot is visible to the head and wrist cameras.
- Collect atomic skills before full sequence:
  - `Pick up the bread slice from the desk.`
  - `Move the bread slice above the toaster slot.`
  - `Insert the bread slice into the toaster slot.`
  - `Press the toaster lever down.` only if the cold toaster mechanics are safe for the robot.
- Vary bread pose, toaster pose, slot orientation, lighting, and nearby distractors.
- Add full cold sequence only after atomic skills pass evaluation:
  - `Pick up the bread and place it into the cold toaster slot.`
- Do not train or test with a powered or hot toaster until cold/unplugged success is stable and a separate safety procedure exists.

## Labeling And Metadata

Add per-episode metadata next to the LeRobot dataset. At minimum, keep a JSONL sidecar keyed by `episode_index`.

Recommended fields:

- `task`
- `subtask_sequence`
- `object_type`
- `object_color`
- `object_instance`
- `receptacle`
- `object_pose_bin`
- `target_pose_bin`
- `arm`
- `clutter_level`
- `lighting`
- `success`
- `failure_mode`
- `retry_count`
- `operator_intervention`
- `safety_exclusions`
- `notes`

Add LeRobot subtask annotations when possible:

- `approach object`
- `open gripper`
- `align gripper around object`
- `close gripper`
- `lift object`
- `move to receptacle`
- `release object`
- `recover after failed grasp`

For long-horizon desk cleanup, keep System 2 responsible for choosing the next object and System 1 responsible for bounded prompts like `Put the leftmost blue marker into the white mesh cup`.

## Training Plan

Start simple, then widen only when data supports it.

1. Validate the current dataset before training:

   ```bash
   uv run python scripts/finetune_pi05.py --check-only
   ```

2. Train a marker-only improvement checkpoint using Phase 1 data.

   ```bash
   uv run python scripts/finetune_pi05.py \
     --dataset-repo-id <new-marker-dataset> \
     --dataset-root <new-marker-dataset-root> \
     --output-dir outputs/pi05_marker_robust \
     --steps 30000
   ```

3. If marker failures are visual grounding failures, run a full fine-tune after enough diverse data exists:

   ```bash
   uv run python scripts/finetune_pi05.py \
     --dataset-repo-id <new-marker-dataset> \
     --dataset-root <new-marker-dataset-root> \
     --output-dir outputs/pi05_marker_full \
     --steps 30000 \
     --no-train-expert-only \
     --no-oom-safe \
     --batch-size 4
   ```

4. Use offline eval only as a regression check, not as proof of real robot success:

   ```bash
   uv run python scripts/eval_pi05_offline.py \
     --policy-path outputs/pi05_marker_robust/checkpoints/last/pretrained_model \
     --episodes 0,1,2 \
     --max-samples 200
   ```

5. Train desk cleanup as a multi-task dataset only after marker pickup passes the real robot gate. Balance sampling so the cloth/toaster/rare skills are not drowned by marker examples.

6. Add human-in-the-loop rounds:
   - Deploy current checkpoint.
   - Intervene only when it is about to fail or become unsafe.
   - Record autonomous segment plus correction.
   - Fine-tune.
   - Re-test on held-out layouts.
   - Repeat until failures are no longer dominated by the same mode.

## Acceptance Gates

- Marker-to-cup: at least 80% success on 30 held-out marker poses with no object repositioning and no unsafe contact.
- Marker recovery: at least 70% success when the first grasp attempt misses or nudges the marker but remains safe.
- Desk cleanup: at least 70% success on held-out layouts with 3-4 pens/markers and one cloth, with no contact against electronics, keyboard, monitor, bottle, or cables.
- Cloth move: at least 70% success moving the cloth to the requested safe target area.
- Cold toaster insert: at least 80% success on a cold, unplugged toaster before any powered-appliance testing is considered.

## Do Not Collect Yet

- Powered toaster or hot toaster data.
- Any episode that requires contacting exposed electronics, keyboard, monitor, cables, or water bottle.
- Ambiguous `clean the desk` episodes without clear item-level interpretation.
- Demos where the object is manually moved into a convenient pose without labeling the intervention.
- Long-horizon multi-object episodes before item-level pickup succeeds reliably.
