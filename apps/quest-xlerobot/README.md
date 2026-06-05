# XLeRobot Quest Teleop

Standalone Quest 3/OpenXR Unity client for XLeRobot teleoperation.

This app is the first-party port target for the Reachy2 teleoperation reference
in `reference/Reachy2Teleoperation`. The production backend remains
`openpibot.server.runtime.vr_teleop.VRTeleopSession`; this Unity app only
streams operator intent and renders operator/video state.

## What Was Adapted From Reachy2

- Mirror/ready flow: use headset yaw to establish the operator origin before arm
  teleop.
- Controller tracking: sample Quest left/right controller poses and buttons each
  frame.
- Transport shape: stream controller snapshots continuously to the workstation
  over a tokenized WebSocket.
- Video target: consume backend-managed GStreamer H.264/RTP camera streams for
  low-latency headset video.

## What Is Different For XLeRobot

- The Quest app does not send Reachy `ArmCartesianGoal` protobuf commands.
- The workstation backend owns SO101 IK, safety gates, robot verification, and
  LeRobot recording.
- The app streams absolute controller snapshots to `/api/vr/quest/ws`; the
  backend converts them to reset/position/idle goals.
- Unsupported Reachy features such as mobile base, antennas, emotions, and neck
  control are intentionally omitted.

## Unity Setup

- Unity: 2022.3.55f1, matching the reference project.
- Build target: Android.
- XR provider: OpenXR.
- Device: Meta Quest 3.
- Scripting backend: IL2CPP.
- Target architecture: ARM64.
- Internet permission is required for workstation WebSocket/WebRTC traffic.

Open this directory as a Unity project, switch to Android, open
`Assets/Scenes/BaseScene.unity`, and build/sideload the APK.
`XLeRobotRuntimeSceneBootstrap` creates the runtime XR origin, operator status
panel, video surfaces, and backend clients when the scene loads. The bootstrap
uses PlayerPrefs keys `xlerobot.quest.host`, `xlerobot.quest.port`,
`xlerobot.quest.tls`, and `xlerobot.quest.token`; if they are unset, edit
`XLeRobotRuntimeSceneBootstrap` defaults or add a small settings scene before
shipping. Use the same token as `XLE_QUEST_PAIRING_TOKEN` on the backend. The
public dashboard status does not expose the token.

## App Components

- `XLeRobotQuestBootstrap` configures endpoint settings for the scene.
- `XLeRobotRuntimeSceneBootstrap` creates and wires the Quest scene graph.
- `XLeRobotQuestClient` streams controller snapshots over WebSocket.
- `XLeRobotStateClient` polls backend operator status for headset UI.
- `XLeRobotOperatorFlowManager` maps backend stages onto Reachy-style flow
  events.
- `XLeRobotSceneFlow` contains the minimal Reachy-style event, scene, user
  tracker, suspension, and emergency-stop managers used by the runtime scene.
- `XLeRobotVideoBridgeClient` starts/stops backend video pipelines.
- `XLeRobotAndroidRtpVideoReceiver` receives backend UDP/RTP H.264 streams on
  Quest, decodes them with Android `MediaCodec`, and uploads Unity textures.
- `XLeRobotVideoSurfaceBinder` and `XLeRobotGStreamerTextureAdapter` bind
  received textures to headset UI or renderer surfaces.
- `XLeRobotGStreamerWebRtcReceiver` optionally binds Pollen/Reachy
  `GstreamerWebRTC` texture events into the same video surfaces.

Hold the primary and secondary buttons together for about one second on either
controller to trigger backend emergency stop.

## Runtime Contract

The app sends packets like:

```json
{
  "timestamp": 1717580000.0,
  "controllers": {
    "left": {
      "position": [0.0, 1.0, 0.0],
      "rotation": [0.0, 0.0, 0.0, 1.0],
      "grip": false,
      "trigger": 0.0,
      "thumbstick": { "x": 0.0, "y": 0.0 },
      "buttons": { "X": false, "Y": false }
    },
    "right": {
      "position": [0.0, 1.0, 0.0],
      "rotation": [0.0, 0.0, 0.0, 1.0],
      "grip": false,
      "trigger": 0.0,
      "thumbstick": { "x": 0.0, "y": 0.0 },
      "buttons": { "A": false, "B": false }
    }
  }
}
```

Grip down becomes a backend reset/anchor, grip held becomes position deltas, and
grip release becomes idle/hold.
