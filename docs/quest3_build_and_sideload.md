# Quest 3 Build And Sideload Guide

This document explains how to compile the XLeRobot Quest teleop Unity app and
load it onto a Meta Quest 3.

## Prerequisites

- Meta Quest 3 with Developer Mode enabled.
- USB-C cable that supports data transfer.
- Unity Hub.
- Unity `6000.4.10f1` or the Unity 6 version recorded in
  `apps/quest-xlerobot/ProjectSettings/ProjectVersion.txt`.
- Unity Android Build Support, including:
  - Android SDK and NDK Tools
  - OpenJDK
- Android platform tools for `adb`.
- The workstation and Quest on the same network.
- Backend pairing token set with `XLE_QUEST_PAIRING_TOKEN`.

## Enable Quest Developer Mode

1. Create or use a Meta developer organization at <https://developer.oculus.com/>.
2. In the Meta Horizon mobile app, enable Developer Mode for the Quest 3.
3. Reboot the headset.
4. Connect the headset to the workstation over USB-C.
5. In the headset, accept the USB debugging prompt.
6. Verify `adb` can see the headset:

```bash
adb devices
```

Expected output should include one device in `device` state.

## Start The Backend

From the repository root:

```bash
export XLE_QUEST_PAIRING_TOKEN="dev-quest-token"
uv run openpibot run --no-build-dashboard
```

Open the dashboard from another browser tab:

```text
http://<workstation-ip>:5000
```

If port `5000` is busy, the CLI may choose another port. Use the port printed by
the backend.

## Open The Unity Project

1. Open Unity Hub.
2. Select **Add project from disk**.
3. Choose:

```text
apps/quest-xlerobot
```

4. Open it with Unity `6000.4.10f1`.
5. Let Unity resolve packages. The project depends on OpenXR, XR Interaction
   Toolkit, UGUI, URP, and the optional Pollen GStreamer WebRTC package.

## Configure Meta Quest Build

In Unity:

1. Wait for scripts and packages to finish importing.
2. Run **XLeRobot > Configure Quest Project**. This project utility configures
   Android player settings, adds the production scenes, registers the OpenXR
   Android loader, and enables the Quest OpenXR features.
3. Go to **File > Build Profiles**.
4. If **Meta Quest** is already listed, select it and click **Switch Profile**.
5. If it is not listed, click **Add Build Profile**, select **Meta Quest**,
   click **Enable Platform**, then **Add Build Profile**, and finally
   **Switch Profile**.
6. Confirm the build scenes include:
   - `Assets/Scenes/BaseScene.unity`
   - `Assets/Scenes/MirrorScene.unity`
   - `Assets/Scenes/TeleoperationScene.unity`
7. Go to **Edit > Project Settings > XR Plug-in Management**.
8. On the **Android** tab, verify **OpenXR** is enabled.
9. Go to **XR Plug-in Management > OpenXR** and verify Quest support is enabled:
   - **Meta Quest Support**
   - **Oculus Touch Controller Profile**
   - **Meta Quest Touch Plus Controller Profile** for Quest 3/3S controllers
10. Go to **Project Settings > Player > Android** and verify:
   - Scripting Backend: `IL2CPP`
   - Target Architectures: `ARM64`
   - Minimum API Level: Android 10 or newer
   - Internet Access: enabled or auto
   - Graphics API: `Vulkan`

If the **Meta Quest** platform button does not enable in Unity Hub or Build
Profiles, use **Android** as the active build profile, run
**XLeRobot > Configure Quest Project**, then verify the same Android OpenXR
settings above. Quest standalone builds are still Android/OpenXR APKs.

## Configure App Endpoint

The runtime bootstrap reads these PlayerPrefs keys:

- `xlerobot.quest.host`
- `xlerobot.quest.port`
- `xlerobot.quest.tls`
- `xlerobot.quest.token`

The current development build defaults to this workstation on the local network:

- host: `192.168.0.113`
- port: `5000`
- tls: `false` for local HTTP
- token: `dev-quest-token`

For a physical Quest headset, the host must be the workstation LAN IP because
`127.0.0.1` points at the headset itself. If the workstation IP changes, update
`xlerobot.quest.host` or the defaults in
`XLeRobotRuntimeSceneBootstrap.cs`, and keep the token equal to
`XLE_QUEST_PAIRING_TOKEN`.

## Build The APK

In Unity:

1. Open **File > Build Profiles**.
2. Select the active **Meta Quest** profile.
3. If using the fallback path, select the active **Android** profile instead.
4. Confirm the three XLeRobot scenes are checked in the profile.
5. Click **Build**.
6. Save the APK as:

```text
builds/xlerobot-quest-teleop.apk
```

Unity will compile scripts, build the Android player, and write the APK.

## Sideload To Quest 3

With the Quest connected over USB:

```bash
adb install -r apps/quest-xlerobot/builds/xlerobot-quest-teleop.apk
```

If Android reports a signing conflict, uninstall the old build first:

```bash
adb uninstall com.xlerobot.questteleop
adb install apps/quest-xlerobot/builds/xlerobot-quest-teleop.apk
```

The package name may differ if changed in Unity Player Settings.

## Launch The App

In the headset:

1. Open the app library.
2. Filter to **Unknown Sources**.
3. Launch the XLeRobot Quest teleop app.

The app should show the runtime operator panel, connect to the backend WebSocket,
and poll `/api/vr/quest/operator`.

## Start Headset Video

From the dashboard VR Teleop page:

1. Enter the same Quest pairing token.
2. Enter the Quest IP address.
3. Click **Start video**.

The backend starts GStreamer RTP/H.264 streams using the roles and ports in:

```yaml
vr:
  quest_video:
    roles:
    - head
    - left_wrist
    - right_wrist
    base_port: 5600
    bitrate_kbps: 2500
    flip_method: rotate-180
    brightness: 0.0
    contrast: 1.0
    saturation: 1.0
```

The Quest app includes `XLeRobotAndroidRtpVideoReceiver`, which listens on the
configured UDP ports in Android Quest builds, depacketizes RTP/H.264, decodes
with Android `MediaCodec`, and uploads decoded RGBA frames into Unity
`Texture2D` surfaces. The optional Pollen/Reachy `GstreamerWebRTC` event adapter
is still available for future WebRTC service integration.

If the camera feed is too dark in-headset, tune `brightness`, `contrast`, and
`saturation` under `vr.quest_video`, then restart the backend or stop/start Quest
video. These values are applied in the backend GStreamer pipeline, so changing
them does not require rebuilding the Quest APK.

If the camera feed is upside down or mirrored, tune `flip_method`. Supported
values include `none`, `rotate-180`, `horizontal-flip`, and `vertical-flip`.

## First-Run Verification Checklist

1. Dashboard shows one native Quest client connected.
2. Pairing token is accepted.
3. Operator status updates in the headset.
4. Grip down creates a backend reset/anchor.
5. Grip held streams position goals.
6. Releasing grip returns that arm to idle/hold.
7. Forward, left, and up controller motions move the robot EE in the calibrated
   robot directions.
8. Wrist pitch/roll directions are correct at low scale.
9. Holding primary + secondary buttons together triggers emergency stop.
10. Headset video appears with usable latency.
11. A short recording episode has synchronized actions and observations.

## Troubleshooting

- `adb devices` shows `unauthorized`: accept the USB debugging prompt inside the
  headset, then run `adb kill-server && adb start-server`.
- Unity cannot switch to Meta Quest or Android: install Android Build Support,
  Android SDK & NDK Tools, and OpenJDK for the exact Unity version opening the
  project.
- The **Meta Quest** Enable Platform button does nothing: close/reopen Unity,
  confirm the project is opened with `6000.4.10f1`, then use
  **XLeRobot > Configure Quest Project** and the **Android** profile fallback.
- App cannot connect: verify workstation IP, backend port, firewall, and
  `XLE_QUEST_PAIRING_TOKEN`.
- Backend rejects Quest packets: make sure the app token exactly matches the
  backend token.
- Video does not start: run `uv run openpibot doctor` and confirm
  `gst-launch-1.0` is available.
- Video start fails with camera contention: stop dashboard MJPEG viewers or any
  process currently holding the same `/dev/video*` devices, then start Quest
  video again.
- App launches but shows no controllers: verify OpenXR is enabled for Android
  and the Quest controllers are awake/tracked.
