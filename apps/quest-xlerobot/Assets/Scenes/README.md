These scenes provide the Reachy-style Quest app entry points:

- `BaseScene`
- `MirrorScene`
- `TeleoperationScene`

`XLeRobotRuntimeSceneBootstrap` builds the runnable Quest rig, status UI, video
surfaces, and backend clients at runtime. This avoids brittle hand-authored
Unity YAML references before Unity generates `.meta` GUIDs for the scripts.
