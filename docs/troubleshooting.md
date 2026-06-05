# Troubleshooting

Quick diagnostics for the most common issues. Run from the repo root in a `newgrp dialout` shell.

## Gripper doesn't move

Most likely cause: motor calibration JSON has `range_min == range_max` for the gripper. Lerobot writes this to the motor's `Min_Position_Limit`/`Max_Position_Limit` registers on every connect, and the motor refuses any goal outside that ~1-tick window.

Diagnose:
Check the arm calibration JSON under `config/calibration/so_follower/` and the motor registers from the dashboard Diagnostics page.

If `Min_Position_Limit` and `Max_Position_Limit` are effectively the same, re-run `lerobot-calibrate` for that arm and move the gripper through its full open/close range during calibration.

## Robot doesn't move during teleop

Check on the page, in order:
1. `arms.<side>.connected` true?
2. `engaged` toggle on, `active_arm` set?
3. The arm is **anchored** (badge says "anchored" — i.e., you've squeezed grip at least once this session)?
4. Calibration card shows `session_yaw_deg` close to the angle you're standing at?
5. Controller card's "age ms" badge green (data flowing)?

If all green and still no motion, check the backend log — every grip-press logs `gripper:` lines, and motor send failures log warnings.

## Motion feels wrong (direction)

If moving your hand right makes the EE move sideways/back/down, the VR→robot frame calibration is off. Either:
- Stand differently and re-press grip (re-anchor),
- Or re-run the **Calibrate** wizard (see [calibration.md](calibration.md)).

Look at `session_yaw_deg` on the Calibration card — it should match (or be close to) the angle you're standing at relative to the robot.

## Wrist drifts when controller is still

If you're seeing slow wrist drift even with your hand still, restart the OpenPIBot server and the native Quest app so both sides clear their controller anchors.

If drift persists, the Quest controller may need re-calibration in the Quest system menu.

## Arm jitters while hand is still

The backend ignores very small VR `relative_position` and `relative_rotvec`
packets before integrating them. If the arm still jitters:

- Lower `vr.pos_ema_alpha` slightly (for example `0.25`).
- Lower `vr.max_ee_step_m` slightly (for example `0.003`).
- Check the Control page's per-arm **EE speed EMA** and **IK rejects** cards.
  A nonzero speed while your hand is still points to controller tracking noise;
  frequent IK rejects usually means the grip anchor or target is near the edge
  of the SO101 workspace.

## "Release for posing" doesn't release torque

Check the server log for the `release_torque_for_posing` call. If you see no log line, the API didn't reach the OpenPIBot server. If you see the log but the arm still holds, the bus write failed silently — power-cycle the robot and try again.

## Quest app connects but motion doesn't reach the backend

Check the *VR Operator* page for the tokenized `ws://<workstation>:5000/api/vr/quest/ws` URL and make sure the Quest app uses the same pairing token as `XLE_QUEST_PAIRING_TOKEN`.

## EE stops at the workspace boundary

The robot's reach is ~25 cm. If `offset_robot` keeps growing but `target_ee_pos` saturates, your hand is past the arm's reach. Walk closer to the robot OR re-grip (re-anchor) with the arm at a more central pose.

## "HOMING…" doesn't clear in the UI after the arm reaches home

The arm physically arrived, but the present-position check is stricter than the motor's mechanical resolution. The drive loop now declares "arrived" once the software target has converged, not the present position — restart the backend if you're still hitting this.

## Bus opens, then `Missing motor IDs` error

A motor on that arm isn't responding. Common causes: power not connected, USB cable loose, motor ID mismatch (each arm should have IDs 1–6). Run:
Use the dashboard Diagnostics page to inspect serial ports and motor status. A `Status: 0x20 (Overload)` means the motor latched a fault — power-cycle the robot to clear.
