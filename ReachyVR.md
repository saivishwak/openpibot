# Reachy 2 VR Teleoperation Coordinate And Command Flow

This document explains how `reference/Reachy2Teleoperation` converts VR headset/controller tracking into Reachy 2 robot commands. The main focus is the path from Unity XR coordinates to Reachy Cartesian goals and, where visible in this repository, to joint angles.

## Short Version

The application does not directly command arm joint angles for the real robot. It sends Cartesian end-effector targets:

```text
VR tracked hand pose
-> Unity world-space Transform
-> user/body-relative pose
-> Unity-to-Reachy coordinate conversion
-> optional human-to-robot arm-length scaling
-> ArmCartesianGoal.GoalPose, a 4x4 matrix
-> WebRTC protobuf command
-> Reachy backend computes IK and motor commands
```

For the head, it sends a neck orientation quaternion:

```text
VR headset rotation
-> user/body-relative head rotation
-> Unity-to-Reachy quaternion component remapping
-> NeckJointGoal
-> WebRTC protobuf command
-> Reachy backend applies neck control
```

For the grippers and mobile base:

```text
controller trigger -> gripper opening percentage
controller joysticks -> mobile-base velocity direction
```

The relevant runtime send loop is in:

- `reference/Reachy2Teleoperation/Assets/Scripts/Manager/ScenesManager/TeleoperationManager.cs`
- `reference/Reachy2Teleoperation/Assets/Scripts/Tracking/HandsTracker.cs`
- `reference/Reachy2Teleoperation/Assets/Scripts/Tracking/HeadTracker.cs`
- `reference/Reachy2Teleoperation/Assets/Scripts/UserInputs/UserMovementsInput.cs`
- `reference/Reachy2Teleoperation/Assets/Scripts/Robot/RobotCommands.cs`
- `reference/Reachy2Teleoperation/Assets/Scripts/Robot/RobotJointCommands.cs`
- `reference/Reachy2Teleoperation/Assets/Scripts/WebRTC/DataMessageManager.cs`

## Code Navigation

| Topic | Code |
| --- | --- |
| Main per-frame teleop loop | [`TeleoperationManager.cs`](reference/Reachy2Teleoperation/Assets/Scripts/Manager/ScenesManager/TeleoperationManager.cs#L204-L227) |
| User/body origin from headset yaw | [`UserTrackerManager.cs`](reference/Reachy2Teleoperation/Assets/Scripts/Tracking/UserTrackerManager.cs#L25-L37) |
| VR hand pose to Reachy 4x4 matrix | [`HandsTracker.cs`](reference/Reachy2Teleoperation/Assets/Scripts/Tracking/HandsTracker.cs#L102-L129) |
| User-size arm scaling | [`UserMovementsInput.cs`](reference/Reachy2Teleoperation/Assets/Scripts/UserInputs/UserMovementsInput.cs#L43-L101) |
| Headset rotation to neck quaternion | [`HeadTracker.cs`](reference/Reachy2Teleoperation/Assets/Scripts/Tracking/HeadTracker.cs#L13-L37) |
| Gripper trigger mapping | [`UserMovementsInput.cs`](reference/Reachy2Teleoperation/Assets/Scripts/UserInputs/UserMovementsInput.cs#L103-L164) |
| Mobile-base joystick mapping | [`UserMobilityInput.cs`](reference/Reachy2Teleoperation/Assets/Scripts/UserInputs/UserMobilityInput.cs#L46-L87) |
| Attach robot part ids and IK mode | [`RobotCommands.cs`](reference/Reachy2Teleoperation/Assets/Scripts/Robot/RobotCommands.cs#L52-L83) |
| Send/gate robot commands | [`RobotJointCommands.cs`](reference/Reachy2Teleoperation/Assets/Scripts/Robot/RobotJointCommands.cs#L91-L110) |
| WebRTC command batching | [`DataMessageManager.cs`](reference/Reachy2Teleoperation/Assets/Scripts/WebRTC/DataMessageManager.cs#L45-L52) |
| Arm command protobuf wrapping | [`DataMessageManager.cs`](reference/Reachy2Teleoperation/Assets/Scripts/WebRTC/DataMessageManager.cs#L243-L252) |
| Simulated Cartesian IK to seven joints | [`ReachySimulatedServer.cs`](reference/Reachy2Teleoperation/Assets/Scripts/Robot/ReachySimulatedServer.cs#L61-L151) |

## 1. Where The VR Pose Comes From

Unity XR provides the tracking. The prefab `reference/Reachy2Teleoperation/Assets/Prefabs/XR origin.prefab` contains:

- `XR origin`
- `RightHand Controller`
- `LeftHand Controller`
- `Main Camera`

The controller objects have XR controller/tracking components enabled. The headset camera uses a tracked-pose driver. The application also queries Unity XR `InputDevice`s directly in `ControllersManager`.

Important code:

- `ControllersManager.UpdateDevicesList()` gets devices at `XRNode.RightHand`, `XRNode.LeftHand`, and `XRNode.Head`.
- `ControllersManager.Update()` reads `CommonUsages.isTracked` and emits tracking-lost/retrieved events.

Code reference:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Manager/ControllersManager.cs
  lines 44-51: get right/left/head XR devices
  lines 84-128: update isTracked state for each controller
```

The scene also contains application-level tracked proxy objects:

```text
TrackedLeftHand
TrackedRightHand
HeadTracker
HandsTracker
```

`HandsTracker` looks up `TrackedRightHand` and `TrackedLeftHand` by name and uses those `Transform`s as the VR hand poses.

Code reference:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Tracking/HandsTracker.cs
  lines 44-57: map "right"/"left" to TrackedRightHand/TrackedLeftHand Transforms
```

## 2. User Origin: The Body Frame

Before teleoperation, the app defines a user-centered coordinate frame. This is the parent frame used to interpret the headset and hand positions as body-relative motion.

`UserTrackerManager.FixUserOrigin()` does three important things:

1. Reads the headset local rotation.
2. Keeps only yaw, ignoring headset pitch and roll.
3. Places the origin slightly behind and below the headset, approximating the user's shoulder/body frame.

Code reference:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Tracking/UserTrackerManager.cs
  lines 25-37: define user origin from headset yaw and position
```

The code:

```csharp
Quaternion rotation = headset.localRotation;
Vector3 eulerAngles = rotation.eulerAngles;
Quaternion systemRotation = Quaternion.Euler(0, eulerAngles.y, 0);

transform.rotation = systemRotation;

Vector3 headPosition = headset.position - headset.forward * 0.1f;
transform.position = new Vector3(
    headPosition.x,
    headPosition.y - UserSize.Instance.UserShoulderHeadDistance,
    headPosition.z
);
```

So:

- User frame rotation = headset yaw only.
- User frame position = headset position, shifted 10 cm backward along headset forward, then shifted down by the estimated shoulder-head distance.

This matters because hand/head tracking is converted relative to this origin, not directly from Unity world coordinates.

The user-size estimates are in:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Tracking/UserSize.cs
  lines 17-21: estimate shoulder-head distance, arm size, and shoulder width from user height
```

## 3. Hand Position Conversion: Unity Coordinates To Reachy Coordinates

The key code is `HandsTracker.GetTransforms()`.

Code reference:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Tracking/HandsTracker.cs
  lines 102-129: convert each VR hand Transform into a Reachy Matrix4x4 target
```

### 3.1 Convert World Position Into User-Origin Coordinates

Unity gives a world-space hand position:

```text
p_hand_world
```

The application converts it to the user-origin frame:

```text
p_user = inverse(R_user_origin) * (p_hand_world - p_user_origin)
```

Code:

```csharp
Vector3 positionHeadset =
    Quaternion.Inverse(transform.parent.rotation)
    * (hand.GetVRHand().position - transform.parent.position);
```

Here, `transform.parent` is the user tracker/body origin. The variable name `positionHeadset` is slightly misleading; it is really the hand position expressed in the user-origin frame.

### 3.2 Axis Remapping

Unity uses the usual Unity frame:

```text
Unity +x = right
Unity +y = up
Unity +z = forward
```

The code maps this into Reachy's expected arm target frame:

```csharp
Vector3 positionReachy = new Vector3(
    positionHeadset.z,
    -positionHeadset.x,
    positionHeadset.y
);
```

So the position transform is:

```text
Reachy x =  Unity z
Reachy y = -Unity x
Reachy z =  Unity y
```

In matrix form:

```text
[x_r]   [ 0  0  1 ] [x_u]
[y_r] = [-1  0  0 ] [y_u]
[z_r]   [ 0  1  0 ] [z_u]
```

This same axis basis is used for rotation conversion.

### 3.3 Build The Homogeneous Translation Column

The converted position is put into a homogeneous vector:

```csharp
Vector4 positionVect = new Vector4(
    positionReachy.x,
    positionReachy.y,
    positionReachy.z,
    1
);
```

Later this becomes column 3 of the final 4x4 pose matrix:

```csharp
hand.handPose.SetColumn(3, positionVect);
```

## 4. Hand Rotation Conversion

Again, the key code is:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Tracking/HandsTracker.cs
  lines 109-120
```

### 4.1 Convert World Rotation Into User-Origin Rotation

The app first gets hand rotation relative to the user origin:

```csharp
Quaternion rotation =
    Quaternion.Inverse(transform.parent.rotation)
    * hand.GetVRHand().rotation;
```

Mathematically:

```text
R_user = inverse(R_user_origin) * R_hand_world
```

### 4.2 Convert Quaternion To A Unity Matrix

The rotation is placed into a Unity `Matrix4x4`:

```csharp
hand.handPose.SetTRS(
    new Vector3(0, 0, 0),
    rotation,
    new Vector3(1, 1, 1)
);
```

At this point, `hand.handPose` contains only rotation. Translation is added later.

### 4.3 Change Of Basis Matrix

The app defines this passage/change-of-basis matrix:

```csharp
Matrix4x4 mP = new Matrix4x4(
    new Vector4(0, -1, 0, 0),
    new Vector4(0, 0, 1, 0),
    new Vector4(1, 0, 0, 0),
    new Vector4(0, 0, 0, 1)
);
```

Unity's `Matrix4x4` constructor takes columns, not rows. Therefore the effective 4x4 matrix is:

```text
P =
[  0   0   1   0 ]
[ -1   0   0   0 ]
[  0   1   0   0 ]
[  0   0   0   1 ]
```

The top-left 3x3 block is exactly the axis mapping:

```text
x_r =  z_u
y_r = -x_u
z_r =  y_u
```

### 4.4 Rotation Change Of Basis

The rotation conversion is:

```csharp
hand.handPose = (mP * hand.handPose) * mP.inverse;
```

Mathematically:

```text
R_reachy = P * R_user * P^-1
```

This is the correct pattern for expressing a rotation matrix in a new coordinate basis.

### 4.5 Final Hand Pose Matrix

Finally, the converted translation column is inserted:

```csharp
hand.handPose.SetColumn(3, positionVect);
```

The final matrix has this structure:

```text
T_reachy_hand =
[ R00 R01 R02 x ]
[ R10 R11 R12 y ]
[ R20 R21 R22 z ]
[  0   0   0  1 ]
```

It is copied into the generated protobuf type:

```csharp
hand.target_pos = new Reachy.Kinematics.Matrix4x4
{
    Data = {
        hand.handPose[0,0], hand.handPose[0,1], ...
        ...
        hand.handPose[3,3]
    }
};
```

This `target_pos` is what eventually becomes the robot arm's Cartesian goal.

## 5. Human-To-Robot Arm Scaling

After `HandsTracker` produces a target matrix, `UserMovementsInput` may rescale the position based on user size.

Code reference:

```text
reference/Reachy2Teleoperation/Assets/Scripts/UserInputs/UserMovementsInput.cs
  lines 19-20: Reachy arm size and shoulder width constants
  lines 43-71: right arm target calibration
  lines 73-101: left arm target calibration
```

Constants:

```csharp
private float reachyArmSize = 0.6375f;
private float reachyShoulderWidth = 0.19f;
```

If `UserSize.Instance.UserArmSize == 0`, the app sends the raw converted target:

```csharp
GoalPose = handsTracker.rightHand.target_pos
```

Otherwise it scales the translation entries in the 4x4 matrix. The matrix is stored row-major in protobuf `Data`, so:

```text
Data[3]  = x translation
Data[7]  = y translation
Data[11] = z translation
```

### 5.1 Right Arm Scaling

Code:

```csharp
right_target_pos_calibrated.Data[3] =
    right_target_pos_calibrated.Data[3]
    * reachyArmSize / UserSize.Instance.UserArmSize;

right_target_pos_calibrated.Data[7] =
    (right_target_pos_calibrated.Data[7] + UserSize.Instance.UserShoulderWidth)
    * reachyArmSize / UserSize.Instance.UserArmSize
    - reachyShoulderWidth;

right_target_pos_calibrated.Data[11] =
    right_target_pos_calibrated.Data[11]
    * reachyArmSize / UserSize.Instance.UserArmSize;
```

The right arm y-axis is offset by the user's shoulder width before scaling, then shifted into Reachy's shoulder width. This handles the lateral difference between the human shoulder and the robot shoulder.

### 5.2 Left Arm Scaling

Code:

```csharp
left_target_pos_calibrated.Data[3] =
    left_target_pos_calibrated.Data[3]
    * reachyArmSize / UserSize.Instance.UserArmSize;

left_target_pos_calibrated.Data[7] =
    (left_target_pos_calibrated.Data[7] - UserSize.Instance.UserShoulderWidth)
    * reachyArmSize / UserSize.Instance.UserArmSize
    + reachyShoulderWidth;

left_target_pos_calibrated.Data[11] =
    left_target_pos_calibrated.Data[11]
    * reachyArmSize / UserSize.Instance.UserArmSize;
```

The left side mirrors the shoulder-width offset.

Important detail: only the translation is scaled. The rotation part of the hand pose matrix is not scaled.

## 6. Cartesian Arm Goal Sent To Reachy

`UserMovementsInput.GetRightEndEffectorTarget()` and `GetLeftEndEffectorTarget()` return:

```csharp
ArmCartesianGoal { GoalPose = target_matrix }
```

Then `TeleoperationManager.Update()` sends those targets every frame while arm teleoperation is active.

Code reference:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Manager/ScenesManager/TeleoperationManager.cs
  lines 204-227: main per-frame teleoperation send loop
```

Important loop:

```csharp
if (IsArmTeleoperationActive)
{
    ArmCartesianGoal leftEndEffector =
        userMovementsInput.GetLeftEndEffectorTarget();
    ArmCartesianGoal rightEndEffector =
        userMovementsInput.GetRightEndEffectorTarget();

    jointsCommands.SendArmsCommands(leftEndEffector, rightEndEffector);
    jointsCommands.SendGrippersCommands(pos_left_gripper, pos_right_gripper);
}
```

`RobotCommands.SendArmsCommands()` adds:

- left/right robot part id
- IK constrained mode

Code reference:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Robot/RobotCommands.cs
  lines 52-64: attach part ids and IK mode, then send
```

Code:

```csharp
leftArmRequest.Id = robotConfig.partsId["l_arm"];
leftArmRequest.ConstrainedMode = robotStatus.GetIKMode();

rightArmRequest.Id = robotConfig.partsId["r_arm"];
rightArmRequest.ConstrainedMode = robotStatus.GetIKMode();
```

The default IK mode is `LowElbow`.

Code reference:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Robot/RobotStatus.cs
  line 30: default arm IK mode is IKConstrainedMode.LowElbow
  lines 214-221: setter/getter for IK mode
```

The generated protobuf type confirms that `ArmCartesianGoal` carries the target matrix and IK metadata:

```text
reference/Reachy2Teleoperation/Assets/Scripts/reachy2-sdk-api/csharp/Arm.cs
  lines 2187-2351: ArmCartesianGoal type
  lines 2253-2263: GoalPose field
  lines 2301-2311: ConstrainedMode field
  lines 2341-2351: ContinuousMode field
```

## 7. Does Unity Compute Arm Joint Angles?

For the real robot command path: no, not in the teleoperation code shown here.

The real robot path sends `ArmCartesianGoal` to Reachy:

```text
RobotJointCommands
-> DataMessageManager.SendArmCommand()
-> Bridge.ArmCommand.ArmCartesianGoal
-> WebRTC lossy command channel
-> Reachy-side backend
```

Code reference:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Robot/RobotJointCommands.cs
  lines 97-100: send arm commands only if controller is tracked and arm is enabled

reference/Reachy2Teleoperation/Assets/Scripts/WebRTC/DataMessageManager.cs
  lines 243-252: wrap ArmCartesianGoal into Bridge.ArmCommand
```

The generated API includes an `ArmIKRequest`/`ArmIKSolution`, but the real teleop path here is not calling local IK. It sends Cartesian goals and receives reachability feedback in robot state.

Reachability feedback is parsed in `DataMessageManager.StreamReachyState()` and consumed by `RobotReachabilityManager`.

Code reference:

```text
reference/Reachy2Teleoperation/Assets/Scripts/WebRTC/DataMessageManager.cs
  lines 100-114: extract arm reachability answers from ReachyState

reference/Reachy2Teleoperation/Assets/Scripts/Manager/RobotReachabilityManager.cs
  lines 45-54: process reachability for left/right arms
  lines 71-90: emit unreachable or IK freeze events
```

### Simulated Robot Path

The local simulated robot path does compute joint angles using a native library:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Robot/ReachySimulatedServer.cs
```

Important code:

```text
lines 27-44: import Arm_kinematics native library
lines 61-91: receive ArmCartesianGoal and write simulated joint positions
lines 127-151: ComputeArmIK() copies the 4x4 matrix into double[16] and calls inverse()
```

In `ReachySimulatedServer.SendArmCommand()`:

```csharp
ArmIKRequest ikRequest = new ArmIKRequest {
    Id = armGoal.Id,
    Target = new ArmEndEffector {
        Pose = armGoal.GoalPose,
    }
};

List<double> armSolution = ComputeArmIK(ikRequest);
```

Then `ComputeArmIK()` does:

```csharp
for (int i = 0; i < 16; i++)
{
    M[i] = ikRequest.Target.Pose.Data[i];
}

inverse(side, M, q);
```

The resulting `q` is interpreted as seven arm joint angles:

```text
q[0] shoulder axis 1
q[1] shoulder axis 2
q[2] elbow axis 1
q[3] elbow axis 2
q[4] wrist roll
q[5] wrist pitch
q[6] wrist yaw
```

Those are converted from radians to degrees for the Unity simulated model:

```csharp
present_position["l_arm_shoulder_axis_1"] = Mathf.Rad2Deg * (float)armSolution[0];
...
present_position["l_arm_wrist_yaw"] = Mathf.Rad2Deg * (float)armSolution[6];
```

This simulated path is useful because it shows the intended IK boundary: the 4x4 Cartesian matrix is the input to IK, and the IK output is seven arm joint values.

## 8. Head/Neck Conversion

The head conversion is in:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Tracking/HeadTracker.cs
  lines 13-37: convert headset rotation into NeckJointGoal
```

The code starts from the head tracker's local rotation:

```csharp
Quaternion headQuat = transform.localRotation;
Quaternion RotZeroQuat = transform.parent.rotation;
headQuat = Quaternion.Inverse(RotZeroQuat) * headQuat;
```

This removes the user-origin yaw/rotation:

```text
q_user_head = inverse(q_user_origin) * q_head
```

Then the app amplifies head movement by 20%:

```csharp
headQuat = Quaternion.LerpUnclamped(
    Quaternion.identity,
    headQuat,
    1.2f
);
```

Finally it remaps Unity quaternion components into Reachy's neck quaternion convention:

```csharp
Q = new Reachy.Kinematics.Quaternion
{
    W = headQuat.w,
    X = -headQuat.z,
    Y = headQuat.x,
    Z = -headQuat.y,
}
```

So:

```text
Reachy W =  Unity w
Reachy X = -Unity z
Reachy Y =  Unity x
Reachy Z = -Unity y
```

`TeleoperationManager.Update()` sends this neck target every frame while robot teleoperation is active:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Manager/ScenesManager/TeleoperationManager.cs
  lines 208-209: get and send head target
```

`RobotCommands.SendNeckCommands()` adds the robot head id:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Robot/RobotCommands.cs
  lines 67-70: add head part id and send neck command
```

`DataMessageManager.SendNeckCommand()` wraps it in `Bridge.NeckCommand`:

```text
reference/Reachy2Teleoperation/Assets/Scripts/WebRTC/DataMessageManager.cs
  lines 255-264: wrap NeckJointGoal into Bridge.NeckCommand
```

### Simulated Neck Joint Conversion

The simulated server converts the neck quaternion back into Unity Euler angles and writes three simulated neck joints:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Robot/ReachySimulatedServer.cs
  lines 93-115: convert NeckJointGoal into simulated neck roll/pitch/yaw
```

The mapping there is:

```csharp
present_position["head_neck_roll"] = -neck_commands[2];
present_position["head_neck_pitch"] = neck_commands[0];
present_position["head_neck_yaw"] = -neck_commands[1];
```

Again, this is the virtual robot display path. The real robot command path sends `NeckJointGoal`.

## 9. Gripper Conversion

Gripper control uses controller trigger values, not hand pose.

Code reference:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Tracking/HandsTracker.cs
  lines 131-139: read CommonUsages.trigger

reference/Reachy2Teleoperation/Assets/Scripts/UserInputs/UserMovementsInput.cs
  lines 103-134: toggle state in grasp-lock mode
  lines 137-164: convert trigger to gripper opening target
```

The raw trigger value is:

```text
trigger = 0.0 open hand on controller
trigger = 1.0 trigger fully pressed
```

Normal mode:

```csharp
pos_right_gripper = 1 - handsTracker.rightHand.trigger;
pos_left_gripper = 1 - handsTracker.leftHand.trigger;
```

So:

```text
opening percentage = 1 - trigger
```

This is sent as:

```csharp
HandPositionRequest.Position.ParallelGripper.OpeningPercentage
```

Code reference:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Robot/RobotCommands.cs
  lines 73-83: create HandPositionRequest for left/right grippers

reference/Reachy2Teleoperation/Assets/Scripts/WebRTC/DataMessageManager.cs
  lines 231-240: wrap hand command into Bridge.HandCommand
```

In the simulated robot path, the opening percentage is converted to a model angle:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Robot/ReachySimulatedServer.cs
  lines 117-124: map opening percentage to simulated gripper position
```

Code:

```csharp
float open_gripper = 135;
float closed_gripper = -3;
float targetPosition = (1 - opening) * closed_gripper + opening * open_gripper;
```

## 10. Mobile Base Conversion

The mobile base is controlled by joystick axes, not by headset/body displacement.

Code reference:

```text
reference/Reachy2Teleoperation/Assets/Scripts/UserInputs/UserMobilityInput.cs
  lines 46-87: read joystick axes and create targetDirectionCommand
```

Inputs:

```csharp
rightHandDevice.primary2DAxis -> mobileBaseRotation
leftHandDevice.primary2DAxis  -> mobileBaseTranslation
```

The left joystick translation vector is snapped near cardinal directions by zeroing one component when the joystick angle is close to an axis:

```csharp
if (Mathf.Abs(phi) < (Mathf.PI / 8)) mobileBaseTranslation[1] = 0;
...
```

The final command vector is:

```csharp
targetDirectionCommand = new Vector3(
    direction[1] * translationSpeed,
    -direction[0] * translationSpeed,
    -mobileBaseRotation[0] * 1.5f
);
```

So:

```text
base X     =  leftJoystickY * translationSpeed
base Y     = -leftJoystickX * translationSpeed
base theta = -rightJoystickX * 1.5
```

`translationSpeed` is:

```text
0.5 normally
1.0 when right secondary button is pressed
```

The vector is wrapped into a Reachy mobile-base command:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Robot/RobotMobilityCommands.cs
  lines 43-58: create TargetDirectionCommand with X, Y, Theta
```

Code:

```csharp
TargetDirectionCommand command = new TargetDirectionCommand
{
    Id = robotConfig.partsId["mobile_base"],
    Direction = new DirectionVector
    {
        X = direction[0],
        Y = direction[1],
        Theta = direction[2],
    }
};
```

## 11. Command Transport

The command batching layer is:

```text
reference/Reachy2Teleoperation/Assets/Scripts/WebRTC/DataMessageManager.cs
```

Each Unity frame:

```csharp
if (commands.Commands.Count != 0)
{
    webRTCController.SendCommandMessageLossy(commands);
}
commands = new AnyCommands { };
```

Code reference:

```text
reference/Reachy2Teleoperation/Assets/Scripts/WebRTC/DataMessageManager.cs
  lines 45-52: send queued realtime commands over lossy channel
```

Continuous motion commands are batched as `Bridge.AnyCommands` and sent over the lossy command channel because old realtime commands should be dropped rather than delayed.

Reliable WebRTC messages are used for state/mode operations such as:

- turn arm/head/hand/mobile base on/off
- set speed limit
- set torque limit

Code reference:

```text
reference/Reachy2Teleoperation/Assets/Scripts/WebRTC/GStreamer/Scripts/GstreamerPluginCustom.cs
  lines 158-165: serialize AnyCommands to reliable or lossy WebRTC command channel
```

## 12. End-To-End Per-Frame Runtime Flow

The main loop is:

```text
reference/Reachy2Teleoperation/Assets/Scripts/Manager/ScenesManager/TeleoperationManager.cs
  lines 204-227
```

Per frame, if robot teleoperation is active and not suspended:

1. Get head target from `UserMovementsInput`.
2. Send neck command.
3. If arm teleoperation is active:
   - get left/right end-effector Cartesian targets;
   - get left/right gripper targets;
   - send arm Cartesian commands;
   - send gripper commands.
4. If mobile-base teleoperation is active:
   - get joystick-derived base direction;
   - send mobile-base command.

Pseudo-flow:

```text
TeleoperationManager.Update()
  -> UserMovementsInput.GetHeadTarget()
     -> HeadTracker.GetHeadTarget()
     -> RobotCommands.SendNeckCommands()
     -> DataMessageManager.SendNeckCommand()

  -> UserMovementsInput.GetLeftEndEffectorTarget()
  -> UserMovementsInput.GetRightEndEffectorTarget()
     -> HandsTracker target_pos matrices
     -> optional user-size scaling
     -> RobotCommands.SendArmsCommands()
     -> RobotJointCommands.ActualSendArmsCommands()
     -> DataMessageManager.SendArmCommand()

  -> UserMovementsInput.GetLeftGripperTarget()
  -> UserMovementsInput.GetRightGripperTarget()
     -> RobotCommands.SendGrippersCommands()
     -> DataMessageManager.SetHandPosition()

  -> UserMobilityInput.GetTargetDirectionCommand()
     -> RobotMobilityCommands.SendMobileBaseDirection()
     -> DataMessageManager.SendMobileBaseCommand()

DataMessageManager.Update()
  -> SendCommandMessageLossy(AnyCommands)
```

## 13. Key Transform Summary

### Position

```text
p_user = inverse(R_user_origin) * (p_hand_world - p_user_origin)

[x_r]   [ 0  0  1 ] [x_user]
[y_r] = [-1  0  0 ] [y_user]
[z_r]   [ 0  1  0 ] [z_user]
```

Equivalent scalar mapping:

```text
x_reachy =  z_user
y_reachy = -x_user
z_reachy =  y_user
```

### Rotation

```text
R_user = inverse(R_user_origin) * R_hand_world

P =
[  0   0   1   0 ]
[ -1   0   0   0 ]
[  0   1   0   0 ]
[  0   0   0   1 ]

R_reachy = P * R_user * P^-1
```

### Hand Pose

```text
T_reachy_hand =
[ R_reachy  p_reachy ]
[ 0 0 0     1        ]
```

### Head Quaternion

```text
q_user_head = inverse(q_user_origin) * q_head
q_user_head = amplify(q_user_head, 1.2)

Reachy W =  Unity w
Reachy X = -Unity z
Reachy Y =  Unity x
Reachy Z = -Unity y
```

### Mobile Base

```text
base_x     =  left_joystick_y * translation_speed
base_y     = -left_joystick_x * translation_speed
base_theta = -right_joystick_x * 1.5
```

### Gripper

```text
opening_percentage = 1 - trigger
```

## 14. Main Files To Navigate

Start with these files in this order:

1. `reference/Reachy2Teleoperation/Assets/Scripts/Manager/ScenesManager/TeleoperationManager.cs`
   - Main per-frame teleoperation loop.

2. `reference/Reachy2Teleoperation/Assets/Scripts/Tracking/UserTrackerManager.cs`
   - Defines the user/body frame from headset pose.

3. `reference/Reachy2Teleoperation/Assets/Scripts/Tracking/HandsTracker.cs`
   - Converts VR hand poses to Reachy 4x4 Cartesian target matrices.

4. `reference/Reachy2Teleoperation/Assets/Scripts/UserInputs/UserMovementsInput.cs`
   - Applies user/robot arm scaling and exposes arm/head/gripper targets.

5. `reference/Reachy2Teleoperation/Assets/Scripts/Tracking/HeadTracker.cs`
   - Converts headset orientation to Reachy neck quaternion.

6. `reference/Reachy2Teleoperation/Assets/Scripts/UserInputs/UserMobilityInput.cs`
   - Converts joystick input to mobile base velocity direction.

7. `reference/Reachy2Teleoperation/Assets/Scripts/Robot/RobotCommands.cs`
   - Adds robot part ids and IK mode.

8. `reference/Reachy2Teleoperation/Assets/Scripts/Robot/RobotJointCommands.cs`
   - Gates commands based on robot configuration/status/controller tracking.

9. `reference/Reachy2Teleoperation/Assets/Scripts/WebRTC/DataMessageManager.cs`
   - Wraps commands into protobuf messages and sends them each frame.

10. `reference/Reachy2Teleoperation/Assets/Scripts/Robot/ReachySimulatedServer.cs`
    - Shows how a Cartesian arm matrix can be converted to seven simulated arm joint angles with the native IK library.
