# Full Reachy-Style Quest Teleop Parity TODO

This checklist is the working execution list for closing the parity gaps found
after reviewing the current XLeRobot Quest implementation against
`reference/Reachy2Teleoperation`.

## Current Parity Assessment

- Overall parity is not complete.
- Backend control bridge is the strongest area, but still needs contract cleanup
  and hardware validation.
- Unity app parity is the largest gap: the app has scripts, but still lacks
  production scenes, prefabs, XR origin wiring, and tested APK builds.
- Video is not end-to-end verified in Quest. The backend can start RTP/H.264
  pipelines, but Unity still needs an actual receiver path or a Reachy-compatible
  WebRTC plugin integration that produces textures.

## P0: Blockers For "Fully Working"

- [x] Add real Unity scenes under `apps/quest-xlerobot/Assets/Scenes`.
- [x] Add/preserve the XR origin prefab and scene references needed to run on
  Quest 3.
- [x] Wire `XLeRobotQuestBootstrap`, `XLeRobotQuestClient`,
  `XLeRobotStateClient`, `XLeRobotOperatorFlowManager`, and video surface
  binders into the Unity scene.
- [x] Implement a real Quest video receiver:
  - Preferred: Reachy/Pollen `GstreamerWebRTC` texture path adapted to XLeRobot.
  - Acceptable fallback: documented UDP/RTP receiver that decodes backend streams
    and emits Unity `Texture` objects.
- [x] Replace fragile Unity string-search status parsing with a stable flat
  Quest operator API and DTO.
- [x] Fix `NativeQuestAdapter` button parsing so string values like `"false"` do
  not become pressed buttons.
- [x] Ignore or idle controllers with `valid: false`; invalid controller packets
  must not update anchors, deltas, or button edges.
- [x] Fix Quest video host inference. The app must never default to
  `127.0.0.1` for headset video streams.
- [x] Keep the pairing token out of public status endpoints.
- [x] Ensure generated Quest URLs derive host/scheme from request/config and do
  not rely on hardcoded `:5000`.

## P1: Reachy-Style UX Parity

- [x] Port/adapt the minimal Reachy scene flow:
  - `BaseScene`
  - `MirrorScene`
  - `TeleoperationScene`
- [x] Port/adapt minimal Reachy managers:
  - `EventManager`
  - `ScenesManager`
  - `XRManager`
  - `UserTrackerManager`
  - `MirrorSceneManager`
  - `TeleoperationSceneManager`
  - suspension/presence UI managers
- [x] Keep Reachy unsupported features out unless needed for scene stability:
  mobile base, antennas, emotion, lidar, neck, full Reachy protobuf state.
- [x] Implement mirror/ready confirmation in headset and optionally gate headset
  engage until ready is confirmed.
- [x] Add a headset-visible suspension/error overlay for stale tracking, backend
  errors, missing robot readiness, and emergency stop.
- [x] Add headset e-stop chord or clear headset emergency-stop control.

## P1: Backend Contract Parity

- [x] Add token-gated `/api/vr/quest/operator` with a stable flat DTO:
  - stage
  - guidance
  - ready blockers
  - recording blockers
  - native connection state
  - recording active/ready
  - video summary
  - endpoint bundle
- [x] Optionally push operator status over `/api/vr/quest/ws` after ingest to
  reduce Unity polling latency.
- [x] Add Quest video receive-health reporting from headset to backend.
- [x] Add video state into the operator status contract.
- [x] Add GStreamer diagnostics to doctor checks.
- [x] Move Quest video base port/bitrate/roles into config.
- [x] Prevent camera contention between OpenCV MJPEG and GStreamer `v4l2src`.

## P1: Calibration And Dataset Quality

- [x] Document and test the movement translation contract:
  - Reachy Unity converts headset-relative controller pose into robot-frame
    absolute `ArmCartesianGoal` using `(z, -x, y)` plus a rotation basis change.
  - XLeRobot Quest must send absolute Unity/OpenXR controller poses, while the
    workstation converts them into reset-relative backend goals and owns
    SO-101 IK/safety/recording.
  - Add regression tests that accumulated native Quest deltas equal
    reset-relative absolute displacement after dropped/invalid packets.
- [x] Add a Reachy-reference parity test for pose basis conversion:
  - position basis for `unity_reachy`
  - rotation basis using `B @ R @ B.T`
  - Unity/OpenXR `unity_openxr` basis used by production Quest app
- [x] Add end-to-end mapping tests that start at a grip anchor, move the Quest
  controller in forward/left/up directions, and assert the backend target EE
  offset moves in the calibrated robot directions before IK.
- [x] Persist `teleop_source` with calibration profiles.
- [x] Mark old WebXR profiles stale or require re-verification when native Quest
  packets are used.
- [x] Add more rotation-frame tests for pitch/roll/yaw and both coordinate
  frames.
- [ ] Verify both arms with forward/left/up and wrist pitch/roll low-scale tests.
- [x] Ensure recording remains blocked until every connected arm passes robot
  verification and low-scale checks.

## P2: Validation

- [x] Run `uv run pytest tests`.
- [x] Run `npm --prefix dashboard/frontend run build`.
- [ ] Run Unity compile check for `apps/quest-xlerobot`.
- [ ] Build Android APK with Unity 2022.3.55f1.
- [ ] Sideload and run on Quest 3.
- [ ] Verify Quest connects with pairing token.
- [ ] Verify live video appears in headset at usable latency.
- [ ] Verify all controller directions and wrist motions on real XLeRobot.
- [ ] Verify emergency stop disables torque immediately.
- [ ] Record a short test episode and inspect actions/observations.

## Notes From Research

- The smallest useful Reachy Unity surface is roughly 15-20 scripts plus 3
  scenes, 1 XR origin prefab, video shaders/materials, and XLeRobot-specific
  bridge replacements.
- Do not port `reachy2-sdk-api`, `DataMessageManager`, or Reachy protobuf
  command generation as the XLeRobot backend owns IK/control.
- Reachy’s app architecture is valuable for scene flow and video display, not
  for robot command transport.
