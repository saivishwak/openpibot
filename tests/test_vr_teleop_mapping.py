import math
from types import SimpleNamespace

import numpy as np
import pytest

from openpibot.server.runtime import vr_teleop as vr


def _sample(label, vr_delta, robot_delta):
    vr_delta = np.asarray(vr_delta, dtype=float)
    robot_delta = np.asarray(robot_delta, dtype=float)
    return {
        "label": label,
        "vr_delta": [float(v) for v in vr_delta],
        "robot_delta": [float(v) for v in robot_delta],
        "vr_motion_m": float(np.linalg.norm(vr_delta)),
        "robot_motion_m": float(np.linalg.norm(robot_delta)),
    }


def _six_axis_samples(matrix):
    axes = {
        "forward": np.array([0.10, 0.00, 0.00]),
        "back": np.array([-0.10, 0.00, 0.00]),
        "left": np.array([0.00, 0.10, 0.00]),
        "right": np.array([0.00, -0.10, 0.00]),
        "up": np.array([0.00, 0.00, 0.10]),
        "down": np.array([0.00, 0.00, -0.10]),
    }
    return [_sample(label, delta, matrix @ delta) for label, delta in axes.items()]


class _FakeKinematics:
    def __init__(self):
        self._analytical = vr._SO101Kin()

    def forward_kinematics(self, q_deg):
        q = np.asarray(q_deg, dtype=float)
        reach, z = self._analytical.forward(float(q[1]), float(q[2]))
        pan_rad = math.radians(float(q[0]))
        T = np.eye(4)
        T[:3, 3] = (
            reach * math.cos(pan_rad),
            reach * math.sin(pan_rad),
            z,
        )
        pitch = float(q[1] + q[2] + q[3])
        T[:3, :3] = vr._R.from_euler(
            "zyx",
            [float(q[0]), pitch, float(q[4])],
            degrees=True,
        ).as_matrix()
        return T

    def inverse_kinematics(
        self,
        q_seed,
        target_T,
        *,
        position_weight=1.0,
        orientation_weight=0.0,
    ):
        q = np.asarray(q_seed, dtype=float).copy()
        tx, ty, tz = np.asarray(target_T[:3, 3], dtype=float)
        q[0] = math.degrees(math.atan2(float(ty), float(tx)))
        q[1], q[2] = self._analytical.inverse(math.hypot(float(tx), float(ty)), float(tz))
        return q


def _mark_urdf_available(arm):
    arm.kinematics = _FakeKinematics()
    arm.using_analytical_fallback = False


class _BoundsOnlyMotors:
    bounds = {
        f"right_arm_{joint}": (-180.0, 180.0)
        for joint in vr._IK_JOINT_ORDER + ("gripper",)
    } | {
        f"left_arm_{joint}": (-180.0, 180.0)
        for joint in vr._IK_JOINT_ORDER + ("gripper",)
    }
    connected_sides = ["right"]

    def __init__(self):
        self._positions = {}
        for side in ("left", "right"):
            prefix = f"{side}_arm_"
            self._positions.update({
                f"{prefix}shoulder_pan": 0.0,
                f"{prefix}shoulder_lift": -60.0,
                f"{prefix}elbow_flex": 45.0,
                f"{prefix}wrist_flex": 20.0,
                f"{prefix}wrist_roll": 0.0,
                f"{prefix}gripper": 0.0,
            })

    def is_connected(self, side):
        return side in self.connected_sides

    def is_torque_enabled(self, side):
        return True

    def read_positions(self, side=None):
        if side is None:
            return dict(self._positions)
        return {k: v for k, v in self._positions.items() if k.startswith(f"{side}_arm_")}


class _BothArmBoundsOnlyMotors(_BoundsOnlyMotors):
    connected_sides = ["left", "right"]


class _FakeCamera:
    name = "head"
    role = "head"


class _FakeRoleCamera:
    def __init__(self, role):
        self.name = role
        self.role = role


def _mark_recording_ready(session, monkeypatch):
    monkeypatch.setattr(vr, "MOTORS", _BothArmBoundsOnlyMotors())
    monkeypatch.setattr(
        vr._dataset,
        "role_camera_list",
        lambda: (["head", "left_wrist", "right_wrist"], (2, 2, 3)),
    )
    monkeypatch.setattr(vr._cameras, "suspended_capture_roles", lambda: {})
    monkeypatch.setattr(vr._home, "home_pose_status", lambda: {
        "left": {"captured": True, "joints": {}},
        "right": {"captured": True, "joints": {}},
    })
    session._native_quest_clients = 1
    for side in ("left", "right"):
        arm = session._arms[side]
        _mark_urdf_available(arm)
        arm.cal_confidence = "good"
        arm.robot_verify_quality = "good"
        arm.robot_verify_fit_error_cm = 0.0
        arm.robot_verify_test_completed = True
        arm.controller_anchor_T = vr._pose_matrix_from_vr((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        arm.vr_ctrl_to_ee = vr._R.identity()
        arm.calibrated = True
        arm.anchor_generation = arm.pose_generation
        arm.anchor_invalid_reason = ""


def _configure_compute_target_arm(session, monkeypatch, *, side="right", matrix=None):
    monkeypatch.setattr(vr, "MOTORS", _BoundsOnlyMotors())
    monkeypatch.setattr(vr, "POS_EMA_ALPHA", 1.0)
    monkeypatch.setattr(vr, "MAX_EE_STEP_M", 1.0)
    monkeypatch.setattr(vr, "POS_DEADZONE_M", 0.0)
    arm = session._arms[side]
    _mark_urdf_available(arm)
    arm.session_vr_to_robot = np.array(matrix if matrix is not None else np.eye(3), dtype=float)
    arm.translation_scale = 1.0
    arm.controller_anchor_T = vr._pose_matrix_from_vr((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    arm.anchor_ee_pos = (0.20, 0.0, 0.05)
    arm.anchor_R_robot = np.eye(3)
    arm.smoothed_R_target = np.eye(3)
    arm.target_R_robot = np.eye(3)
    arm.anchor = vr._AnchorPose(
        shoulder_lift_deg=-60.0,
        elbow_flex_deg=45.0,
        wrist_flex_deg=20.0,
        wrist_roll_deg=0.0,
        captured=True,
    )
    return arm


def _queue_rel(arm, rel):
    arm.pending_rel_position = tuple(float(v) for v in rel)


def test_analytical_so101_fk_ik_roundtrip():
    kin = vr._SO101Kin()
    for shoulder_lift, elbow_flex in [(-80.0, 60.0), (-35.0, 25.0), (20.0, -10.0)]:
        x, z = kin.forward(shoulder_lift, elbow_flex)
        sl2, ef2 = kin.inverse(x, z)
        assert sl2 == pytest.approx(shoulder_lift, abs=1e-6)
        assert ef2 == pytest.approx(elbow_flex, abs=1e-6)


def test_three_motion_calibration_solves_operator_frame_axes():
    matrix, confidence = vr._compute_session_frame_from_three_motions(
        (0.10, 0.00, 0.00),
        (0.00, 0.00, 0.10),
        (0.00, 0.10, 0.00),
    )

    assert confidence == "good"
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-9)


def test_three_motion_calibration_rejects_parallel_motions():
    with pytest.raises(ValueError, match="too parallel"):
        vr._compute_session_frame_from_three_motions(
            (0.10, 0.00, 0.00),
            (0.08, 0.01, 0.00),
            (0.00, 0.10, 0.00),
        )


def test_finalize_translation_calibration_resets_on_invalid_motions():
    session = vr.VRTeleopSession()
    arm = session._arms["right"]
    arm.cal_state = "motioning_left"
    arm.cal_captured_fwd = (0.10, 0.00, 0.00)
    arm.cal_captured_up = (0.08, 0.01, 0.00)
    arm.cal_captured_left = (0.00, 0.10, 0.00)
    arm.session_vr_to_robot = np.full((3, 3), 2.0)

    session._finalize_translation_calibration("right")

    assert arm.cal_state == "awaiting_anchor_fwd"
    assert arm.cal_confidence == "poor"
    np.testing.assert_allclose(arm.session_vr_to_robot, vr._VR_TO_ROBOT)
    assert arm.cal_captured_fwd is None
    assert "rejected" in (session._last_error or "")


def test_real_teleop_uses_calibrated_so101_urdf_when_available(monkeypatch):
    fake_kinematics = object()
    monkeypatch.setattr(vr, "_load_urdf_kinematics", lambda: fake_kinematics)

    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")

    assert session._ensure_kinematics(arm) is True
    assert arm.kinematics is fake_kinematics
    assert arm.using_analytical_fallback is False


def test_real_teleop_refuses_analytical_without_urdf(monkeypatch):
    monkeypatch.setattr(vr, "_load_urdf_kinematics", lambda: None)

    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")

    assert session._ensure_kinematics(arm) is False
    assert arm.kinematics is None
    assert arm.using_analytical_fallback is False


def test_live_joint_deadband_holds_subdegree_noise():
    previous = {
        "right_arm_shoulder_pan": 10.0,
        "right_arm_wrist_roll": -5.0,
        "right_arm_gripper": 0.0,
    }
    targets = {
        "right_arm_shoulder_pan": 10.05,
        "right_arm_wrist_roll": -4.90,
        "right_arm_gripper": 0.1,
    }

    filtered = vr._apply_live_joint_deadband("right_arm_", targets, previous)

    assert filtered["right_arm_shoulder_pan"] == pytest.approx(10.0)
    assert filtered["right_arm_wrist_roll"] == pytest.approx(-5.0)
    assert filtered["right_arm_gripper"] == pytest.approx(0.1)


def test_live_joint_deadband_allows_intentional_motion():
    previous = {
        "right_arm_shoulder_pan": 10.0,
        "right_arm_wrist_roll": -5.0,
    }
    targets = {
        "right_arm_shoulder_pan": 10.5,
        "right_arm_wrist_roll": -4.6,
    }

    filtered = vr._apply_live_joint_deadband("right_arm_", targets, previous)

    assert filtered == targets


def test_live_joint_command_filter_matches_reference_weight_order(monkeypatch):
    monkeypatch.setattr(vr, "JOINT_COMMAND_FILTER_WEIGHTS", (0.4, 0.3, 0.2, 0.1))
    filt = vr._JointCommandFilter()
    prefix = "right_arm_"
    joint = f"{prefix}shoulder_pan"
    wrist = f"{prefix}wrist_roll"
    gripper = f"{prefix}gripper"

    outputs = [
        filt.apply(prefix, {joint: value, wrist: value, gripper: value})
        for value in (0.0, 10.0, 20.0, 30.0)
    ]

    assert outputs[0][joint] == pytest.approx(0.0)
    assert outputs[1][joint] == pytest.approx(10.0)
    assert outputs[2][joint] == pytest.approx(20.0)
    # Same newest-heavy weighting as reference/xr_teleoperate:
    # 30*0.4 + 20*0.3 + 10*0.2 + 0*0.1.
    assert outputs[3][joint] == pytest.approx(20.0)
    assert outputs[3][wrist] == pytest.approx(30.0)
    assert outputs[3][gripper] == pytest.approx(30.0)


def test_live_joint_command_filter_reset_seeds_from_present(monkeypatch):
    monkeypatch.setattr(vr, "JOINT_COMMAND_FILTER_WEIGHTS", (0.4, 0.3, 0.2, 0.1))
    filt = vr._JointCommandFilter()
    prefix = "left_arm_"
    joint = f"{prefix}elbow_flex"

    filt.reset({joint: 42.0})
    out = filt.apply(prefix, {joint: 42.2})

    assert out[joint] == pytest.approx(42.08)


def test_wrist_axes_apply_left_handedness_once():
    right_pitch, right_roll = vr._effective_wrist_axes("right")
    left_pitch, left_roll = vr._effective_wrist_axes("left")

    assert right_pitch.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert right_roll.tolist() == pytest.approx([0.0, 0.0, -1.0])
    assert left_pitch.tolist() == pytest.approx([-1.0, 0.0, 0.0])
    assert left_roll.tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_runtime_translation_uses_valid_robot_verified_matrix():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    arm.session_vr_to_robot = vr._VR_TO_ROBOT.copy()
    arm.translation_scale = 0.5
    verified = 0.5 * vr._VR_TO_ROBOT
    verified[0, 1] += 0.02
    arm.translation_vr_to_robot = verified
    arm.robot_verify_quality = "good"

    assert session._runtime_translation_matrix(arm) is not None
    np.testing.assert_allclose(session._runtime_translation_matrix(arm), verified)
    assert session._runtime_translation_source(arm) == "robot_verified_3d"


@pytest.mark.parametrize("side", ["left", "right"])
def test_arm_movement_mapper_is_shared_for_left_and_right(side):
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side=side)
    arm.session_vr_to_robot = vr._VR_TO_ROBOT.copy()
    arm.translation_scale = 0.75
    verified = 0.75 * vr._VR_TO_ROBOT
    verified[2, 0] += 0.015
    arm.translation_vr_to_robot = verified
    arm.robot_verify_quality = "good"

    np.testing.assert_allclose(session._runtime_translation_matrix(arm), verified)
    assert session._runtime_translation_source(arm) == "robot_verified_3d"


def test_runtime_translation_applies_lateral_invert_after_robot_verification():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    arm.session_vr_to_robot = np.eye(3)
    arm.invert_lateral = True
    verified = np.array([[0.0, -0.4, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.4]])
    arm.translation_vr_to_robot = verified
    arm.robot_verify_quality = "good"

    np.testing.assert_allclose(session._runtime_translation_matrix(arm), np.diag([1.0, -1.0, 1.0]))
    assert session._runtime_translation_source(arm) == "stage1_scaled_verified_matrix_invalid"


def test_left_verified_runtime_keeps_forward_axis_when_solved_matrix_is_degenerate(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr, "MOTORS", _BoundsOnlyMotors())
    monkeypatch.setattr(vr, "POS_EMA_ALPHA", 1.0)
    monkeypatch.setattr(vr, "MAX_EE_STEP_M", 1.0)
    monkeypatch.setattr(vr, "POS_DEADZONE_M", 0.0)

    arm = session._arms["left"]
    _mark_urdf_available(arm)
    arm.session_vr_to_robot = vr._VR_TO_ROBOT.copy()
    arm.translation_scale = 1.0
    arm.robot_verify_quality = "good"
    # This is the class of bad solve that broke live teleop: the verified matrix
    # can pass persistence/quality checks yet map VR forward to no robot X.
    arm.translation_vr_to_robot = np.array([
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    arm.controller_anchor_T = vr._pose_matrix_from_vr((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    arm.anchor_ee_pos = (0.20, 0.0, 0.05)
    arm.anchor_R_robot = np.eye(3)
    arm.smoothed_R_target = np.eye(3)
    arm.target_R_robot = np.eye(3)
    arm.anchor = vr._AnchorPose(
        shoulder_lift_deg=-60.0,
        elbow_flex_deg=45.0,
        wrist_flex_deg=20.0,
        wrist_roll_deg=0.0,
        captured=True,
    )
    _queue_rel(arm, (0.03, 0.0, 0.0))

    session._compute_targets_from_vr(
        "left",
        vr._LatestGoal(
            received_at=1.0,
            has_data=True,
            mode="position",
            rel_position=(0.03, 0.0, 0.0),
            controller_position=(0.03, 0.0, 0.0),
            rotation_quat=(0.0, 0.0, 0.0, 1.0),
            trigger=False,
        ),
        scale=1.0,
    )

    assert arm.offset_robot[0] > 0.0
    np.testing.assert_allclose(arm.last_diag["dp_robot"], [0.03, 0.0, 0.0], atol=1e-9)
    assert arm.last_diag["translation_source"] == "stage1_scaled_verified_matrix_invalid"


def test_runtime_translation_applies_lateral_invert_to_vr_only_matrix():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    arm.session_vr_to_robot = np.eye(3)
    arm.invert_lateral = True
    arm.robot_verify_quality = "unverified"

    np.testing.assert_allclose(
        session._runtime_translation_matrix(arm),
        np.diag([1.0, -1.0, 1.0]),
    )


def test_runtime_translation_ignores_non_good_robot_verification():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    arm.session_vr_to_robot = np.eye(3)
    arm.translation_scale = 0.25
    arm.translation_vr_to_robot = np.eye(3) * 3.0
    arm.robot_verify_quality = "needs_recapture"

    np.testing.assert_allclose(session._runtime_translation_matrix(arm), np.eye(3))


def test_restore_non_good_robot_verification_uses_base_frame(monkeypatch):
    session = vr.VRTeleopSession()
    base = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    stale_verified = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    stale_translation = np.eye(3) * 4.0
    data = {
        "calibration_mode": "robot_verified",
        "teleop_source": "native_quest",
        "coordinate_frame": "quest_operator_frame",
        "confidence": "good",
        "calibration_quality": "needs_recapture",
        "base_vr_direction_matrix": base.tolist(),
        "translation_vr_to_robot_matrix": stale_translation.tolist(),
        "translation_scale": 0.25,
        "fit_error_cm": 4.0,
    }

    monkeypatch.setattr(vr._vrcal, "read_invert_lateral_flags", lambda: {"right": False})
    monkeypatch.setattr(vr._vrcal, "read_invert_lateral_overrides", lambda: {"right": False})
    monkeypatch.setattr(vr._vrcal, "read_for_arm", lambda side: data if side == "right" else None)
    monkeypatch.setattr(vr._vrcal, "matrix_for_arm", lambda side: stale_verified if side == "right" else None)
    monkeypatch.setattr(vr._vrcal, "translation_scale_for_arm", lambda side: 0.25)

    session._restore_persisted_arm_config("right")
    arm = session._arms["right"]

    np.testing.assert_allclose(arm.session_vr_to_robot, base)
    assert arm.translation_vr_to_robot is None
    assert arm.translation_scale == pytest.approx(1.0)
    np.testing.assert_allclose(session._runtime_translation_matrix(arm), base)


def test_restore_good_robot_verification_keeps_scale_and_test_completion(monkeypatch):
    session = vr.VRTeleopSession()
    base = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    full_translation = np.array([[0.0, -0.8, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.8]])
    data = {
        "calibration_mode": "vr_direction",
        "teleop_source": "native_quest",
        "coordinate_frame": "quest_operator_frame",
        "confidence": "good",
        "session_vr_to_robot": base.tolist(),
        "robot_verification": {
            "calibration_mode": "robot_verified",
            "teleop_source": "native_quest",
            "coordinate_frame": "quest_operator_frame",
            "calibration_quality": "good",
            "base_vr_direction_matrix": base.tolist(),
            "translation_vr_to_robot_matrix": full_translation.tolist(),
            "translation_scale": 0.5,
            "fit_error_cm": 1.0,
            "low_scale_test_completed": True,
        },
    }

    monkeypatch.setattr(vr._vrcal, "read_invert_lateral_flags", lambda: {"right": False})
    monkeypatch.setattr(vr._vrcal, "read_invert_lateral_overrides", lambda: {"right": False})
    monkeypatch.setattr(vr._vrcal, "read_for_arm", lambda side: data if side == "right" else None)
    monkeypatch.setattr(vr._vrcal, "matrix_for_arm", lambda side: base if side == "right" else None)
    monkeypatch.setattr(vr._vrcal, "translation_scale_for_arm", lambda side: 0.5)

    session._restore_persisted_arm_config("right")
    arm = session._arms["right"]

    np.testing.assert_allclose(arm.session_vr_to_robot, base)
    np.testing.assert_allclose(arm.translation_vr_to_robot, full_translation)
    np.testing.assert_allclose(session._runtime_translation_matrix(arm), full_translation)
    assert session._runtime_translation_source(arm) == "robot_verified_3d"
    assert arm.robot_verify_quality == "good"
    assert arm.robot_verify_fit_error_cm == pytest.approx(1.0)
    assert arm.robot_verify_test_completed is True


def test_restore_ignores_native_quest_calibration_from_wrong_coordinate_frame(monkeypatch):
    session = vr.VRTeleopSession()
    stale_matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    data = {
        "calibration_mode": "vr_direction",
        "teleop_source": "native_quest",
        "coordinate_frame": "unity_openxr",
        "confidence": "good",
        "session_vr_to_robot": stale_matrix.tolist(),
        "robot_verification": {
            "calibration_mode": "robot_verified",
            "teleop_source": "native_quest",
            "coordinate_frame": "unity_openxr",
            "calibration_quality": "good",
            "translation_vr_to_robot_matrix": stale_matrix.tolist(),
            "translation_scale": 0.25,
            "fit_error_cm": 0.0,
        },
    }

    monkeypatch.setattr(vr._vrcal, "read_invert_lateral_flags", lambda: {"right": False})
    monkeypatch.setattr(vr._vrcal, "read_invert_lateral_overrides", lambda: {"right": False})
    monkeypatch.setattr(vr._vrcal, "read_for_arm", lambda side: data if side == "right" else None)
    monkeypatch.setattr(vr._vrcal, "matrix_for_arm", lambda side: stale_matrix if side == "right" else None)
    monkeypatch.setattr(vr._vrcal, "translation_scale_for_arm", lambda side: 0.25)

    session._restore_persisted_arm_config("right")
    arm = session._arms["right"]

    np.testing.assert_allclose(arm.session_vr_to_robot, vr._VR_TO_ROBOT)
    assert arm.robot_verify_quality == "unverified"
    assert arm.translation_vr_to_robot is None
    assert arm.translation_scale == pytest.approx(1.0)


def test_restore_ignores_legacy_webxr_calibration_for_native_quest(monkeypatch):
    session = vr.VRTeleopSession()
    stale_matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    data = {
        "calibration_mode": "robot_verified",
        "teleop_source": "webxr",
        "confidence": "good",
        "calibration_quality": "good",
        "session_vr_to_robot": stale_matrix.tolist(),
        "base_vr_direction_matrix": stale_matrix.tolist(),
        "translation_vr_to_robot_matrix": stale_matrix.tolist(),
        "translation_scale": 0.25,
        "fit_error_cm": 0.0,
    }

    monkeypatch.setattr(vr._vrcal, "read_invert_lateral_flags", lambda: {"right": False})
    monkeypatch.setattr(vr._vrcal, "read_invert_lateral_overrides", lambda: {"right": False})
    monkeypatch.setattr(vr._vrcal, "read_for_arm", lambda side: data if side == "right" else None)
    monkeypatch.setattr(vr._vrcal, "matrix_for_arm", lambda side: stale_matrix if side == "right" else None)
    monkeypatch.setattr(vr._vrcal, "translation_scale_for_arm", lambda side: 0.25)

    session._restore_persisted_arm_config("right")
    arm = session._arms["right"]

    np.testing.assert_allclose(arm.session_vr_to_robot, vr._VR_TO_ROBOT)
    assert arm.robot_verify_quality == "unverified"
    assert arm.translation_vr_to_robot is None
    assert arm.translation_scale == pytest.approx(1.0)


def test_verified_anchor_refuses_analytical_fallback_after_restart(monkeypatch):
    session = vr.VRTeleopSession()
    fake_motors = _BoundsOnlyMotors()
    monkeypatch.setattr(vr, "MOTORS", fake_motors)
    monkeypatch.setattr(vr, "_load_urdf_kinematics", lambda: None)

    arm = session._arms["right"]
    arm.robot_verify_quality = "good"
    arm.latest = vr._LatestGoal(
        received_at=1.0,
        has_data=True,
        mode="reset",
        controller_position=(0.1, 0.2, 0.3),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
    )

    with pytest.raises(RuntimeError, match="requires calibrated SO101 URDF"):
        session._capture_anchor("right")


def test_compute_targets_reconciles_clamped_offset(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr, "MOTORS", _BoundsOnlyMotors())

    arm = session._arms["right"]
    _mark_urdf_available(arm)
    arm.session_vr_to_robot = np.eye(3)
    arm.translation_scale = 1.0
    arm.controller_anchor_T = vr._pose_matrix_from_vr((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    arm.anchor_ee_pos = (0.20, 0.0, 0.05)
    arm.anchor_R_robot = np.eye(3)
    arm.smoothed_R_target = np.eye(3)
    arm.target_R_robot = np.eye(3)
    arm.anchor = vr._AnchorPose(
        shoulder_lift_deg=-60.0,
        elbow_flex_deg=45.0,
        wrist_flex_deg=20.0,
        wrist_roll_deg=0.0,
        captured=True,
    )

    goal = vr._LatestGoal(
        received_at=1.0,
        has_data=True,
        mode="position",
        controller_position=(2.0, 0.0, 0.0),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
        trigger=False,
    )
    session._compute_targets_from_vr("right", goal, scale=1.0)

    target = np.asarray(arm.target_T[:3, 3], dtype=float)
    offset = np.asarray(arm.offset_robot, dtype=float)
    np.testing.assert_allclose(offset, target - np.asarray(arm.anchor_ee_pos), atol=1e-9)
    assert np.linalg.norm(target) <= vr.WORKSPACE_REACH_M + 1e-9


def test_compute_targets_ignores_packet_delta_when_world_pose_is_static(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr, "MOTORS", _BoundsOnlyMotors())
    monkeypatch.setattr(vr, "POS_EMA_ALPHA", 1.0)
    monkeypatch.setattr(vr, "MAX_EE_STEP_M", 1.0)
    monkeypatch.setattr(vr, "POS_DEADZONE_M", 0.0)

    arm = session._arms["right"]
    _mark_urdf_available(arm)
    arm.session_vr_to_robot = np.eye(3)
    arm.controller_anchor_T = vr._pose_matrix_from_vr((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    arm.anchor_ee_pos = (0.20, 0.0, 0.05)
    arm.anchor_R_robot = np.eye(3)
    arm.smoothed_R_target = np.eye(3)
    arm.target_R_robot = np.eye(3)
    arm.anchor = vr._AnchorPose(
        shoulder_lift_deg=-60.0,
        elbow_flex_deg=45.0,
        wrist_flex_deg=20.0,
        wrist_roll_deg=0.0,
        captured=True,
    )

    goal = vr._LatestGoal(
        received_at=1.0,
        has_data=True,
        mode="position",
        rel_position=(0.01, 0.0, 0.0),
        controller_position=(0.0, 0.0, 0.0),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
        trigger=False,
    )
    _queue_rel(arm, (0.01, 0.0, 0.0))
    session._compute_targets_from_vr("right", goal, scale=1.0)

    np.testing.assert_allclose(arm.pending_rel_position, (0.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(arm.vr_offset_accum, (0.0, 0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(arm.offset_robot, (0.0, 0.0, 0.0), atol=1e-9)


def test_compute_targets_does_not_replay_stale_goal_delta(monkeypatch):
    session = vr.VRTeleopSession()
    arm = _configure_compute_target_arm(session, monkeypatch, matrix=np.eye(3))
    goal = vr._LatestGoal(
        received_at=1.0,
        has_data=True,
        mode="position",
        rel_position=(0.01, 0.0, 0.0),
        controller_position=(0.01, 0.0, 0.0),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
        trigger=False,
    )

    _queue_rel(arm, goal.rel_position)
    session._compute_targets_from_vr("right", goal, scale=1.0)
    offset_after_packet = tuple(arm.offset_robot)

    session._compute_targets_from_vr("right", goal, scale=1.0)

    np.testing.assert_allclose(arm.pending_rel_position, (0.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(arm.vr_offset_accum, (0.01, 0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(arm.offset_robot, offset_after_packet, atol=1e-9)


@pytest.mark.parametrize(
        ("label", "vr_delta", "expected_robot_delta"),
        [
            ("forward", (0.02, 0.0, 0.0), (0.02, 0.0, 0.0)),
            ("left", (0.0, 0.02, 0.0), (0.0, 0.02, 0.0)),
            ("up", (0.0, 0.0, 0.02), (0.0, 0.0, 0.02)),
    ],
)
def test_compute_targets_maps_anchored_quest_motion_to_robot_directions(
    monkeypatch,
    label,
    vr_delta,
    expected_robot_delta,
):
    session = vr.VRTeleopSession()
    arm = _configure_compute_target_arm(session, monkeypatch, matrix=vr._VR_TO_ROBOT)
    _queue_rel(arm, vr_delta)

    session._compute_targets_from_vr(
        "right",
        vr._LatestGoal(
            received_at=1.0,
            has_data=True,
            mode="position",
            rel_position=vr_delta,
            controller_position=vr_delta,
            rotation_quat=(0.0, 0.0, 0.0, 1.0),
            trigger=False,
        ),
        scale=1.0,
    )

    np.testing.assert_allclose(
        arm.last_diag["dp_robot"],
        expected_robot_delta,
        atol=1e-9,
        err_msg=f"{label} VR motion should map to calibrated robot direction",
    )
    np.testing.assert_allclose(arm.offset_robot, expected_robot_delta, atol=1e-9)


def test_compute_targets_uses_absolute_controller_displacement_over_packet_batch(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr, "MOTORS", _BoundsOnlyMotors())
    monkeypatch.setattr(vr, "POS_EMA_ALPHA", 1.0)
    monkeypatch.setattr(vr, "MAX_EE_STEP_M", 1.0)
    monkeypatch.setattr(vr, "POS_DEADZONE_M", 0.001)

    arm = session._arms["right"]
    _mark_urdf_available(arm)
    arm.session_vr_to_robot = np.eye(3)
    arm.controller_anchor_T = vr._pose_matrix_from_vr((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    arm.anchor_ee_pos = (0.20, 0.0, 0.05)
    arm.anchor_R_robot = np.eye(3)
    arm.smoothed_R_target = np.eye(3)
    arm.target_R_robot = np.eye(3)
    arm.pending_rel_position = (0.0012, 0.0, 0.0)
    arm.anchor = vr._AnchorPose(
        shoulder_lift_deg=-60.0,
        elbow_flex_deg=45.0,
        wrist_flex_deg=20.0,
        wrist_roll_deg=0.0,
        captured=True,
    )

    goal = vr._LatestGoal(
        received_at=1.0,
        has_data=True,
        mode="position",
        rel_position=(0.0, 0.0, 0.0),
        controller_position=(0.0012, 0.0, 0.0),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
        trigger=False,
    )
    session._compute_targets_from_vr("right", goal, scale=1.0)

    np.testing.assert_allclose(arm.vr_offset_accum, (0.0012, 0.0, 0.0), atol=1e-9)


def test_compute_targets_caps_large_relative_position_spike(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr, "MOTORS", _BoundsOnlyMotors())
    monkeypatch.setattr(vr, "POS_EMA_ALPHA", 1.0)
    monkeypatch.setattr(vr, "MAX_EE_STEP_M", 0.004)
    monkeypatch.setattr(vr, "POS_DEADZONE_M", 0.0)

    arm = session._arms["right"]
    _mark_urdf_available(arm)
    arm.session_vr_to_robot = np.eye(3)
    arm.controller_anchor_T = vr._pose_matrix_from_vr((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    arm.anchor_ee_pos = (0.20, 0.0, 0.05)
    arm.anchor_R_robot = np.eye(3)
    arm.smoothed_R_target = np.eye(3)
    arm.target_R_robot = np.eye(3)
    arm.anchor = vr._AnchorPose(
        shoulder_lift_deg=-60.0,
        elbow_flex_deg=45.0,
        wrist_flex_deg=20.0,
        wrist_roll_deg=0.0,
        captured=True,
    )

    goal = vr._LatestGoal(
        received_at=1.0,
        has_data=True,
        mode="position",
        rel_position=(0.10, 0.0, 0.0),
        controller_position=(0.10, 0.0, 0.0),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
        trigger=False,
    )
    _queue_rel(arm, (0.10, 0.0, 0.0))
    session._compute_targets_from_vr("right", goal, scale=1.0)

    assert np.linalg.norm(np.asarray(arm.offset_robot, dtype=float)) == pytest.approx(0.004)
    assert arm.quality_last_offset_step_m == pytest.approx(0.004)


def test_wrist_roll_position_jitter_does_not_move_arm_joints(monkeypatch):
    session = vr.VRTeleopSession()
    arm = _configure_compute_target_arm(session, monkeypatch, matrix=np.eye(3))
    monkeypatch.setattr(vr, "POS_EMA_ALPHA", 1.0)
    monkeypatch.setattr(vr, "MAX_EE_STEP_M", 1.0)
    monkeypatch.setattr(vr, "POS_DEADZONE_M", 0.002)
    monkeypatch.setattr(vr, "ORI_EMA_ALPHA", 1.0)
    monkeypatch.setattr(vr, "ROT_DEADZONE_RAD", 0.0)
    monkeypatch.setattr(vr, "WRIST_RAD_DELTA_LIMIT", math.radians(45.0))
    monkeypatch.setitem(
        vr._WRIST_MOTOR_POLARITY,
        "right",
        {"flex": 1.0, "roll": 1.0},
    )

    session._compute_targets_from_vr(
        "right",
        vr._LatestGoal(
            received_at=1.0,
            has_data=True,
            mode="position",
            controller_position=(0.05, 0.0, 0.0),
            rotation_quat=(0.0, 0.0, 0.0, 1.0),
            trigger=False,
        ),
        scale=1.0,
    )
    arm_joints_before = (
        arm.targets.shoulder_pan,
        arm.targets.shoulder_lift,
        arm.targets.elbow_flex,
    )
    offset_before = tuple(arm.offset_robot)

    roll_quat = vr._R.from_rotvec(np.array((0.0, 0.0, -0.25), dtype=float)).as_quat()
    session._compute_targets_from_vr(
        "right",
        vr._LatestGoal(
            received_at=1.0,
            has_data=True,
            mode="position",
            controller_position=(0.0506, 0.0004, 0.0),
            rotation_quat=tuple(float(v) for v in roll_quat),
            trigger=False,
        ),
        scale=1.0,
    )

    np.testing.assert_allclose(arm.offset_robot, offset_before, atol=1e-12)
    assert (
        arm.targets.shoulder_pan,
        arm.targets.shoulder_lift,
        arm.targets.elbow_flex,
    ) == pytest.approx(arm_joints_before)
    assert arm.targets.wrist_roll > 5.0


def test_wrist_roll_cap_is_not_reduced_by_translation_scale(monkeypatch):
    session = vr.VRTeleopSession()
    arm = _configure_compute_target_arm(session, monkeypatch, matrix=np.eye(3))
    arm.vr_ctrl_to_ee = vr._R.identity()
    monkeypatch.setattr(vr, "POS_DEADZONE_M", 0.0)
    monkeypatch.setattr(vr, "ORI_EMA_ALPHA", 1.0)
    monkeypatch.setattr(vr, "ROT_DEADZONE_RAD", 0.0)
    monkeypatch.setattr(vr, "WRIST_RAD_DELTA_LIMIT", math.radians(8.0))
    monkeypatch.setitem(
        vr._WRIST_MOTOR_POLARITY,
        "right",
        {"flex": 1.0, "roll": 1.0},
    )

    roll_quat = vr._R.from_rotvec(np.array((0.0, 0.0, -0.5), dtype=float)).as_quat()
    session._compute_targets_from_vr(
        "right",
        vr._LatestGoal(
            received_at=1.0,
            has_data=True,
            mode="position",
            controller_position=(0.0, 0.0, 0.0),
            rotation_quat=tuple(float(v) for v in roll_quat),
            trigger=False,
        ),
        scale=0.2,
    )

    assert arm.last_diag["wrist_delta_deg"][1] == pytest.approx(8.0, abs=0.2)


def test_capture_anchor_sets_controller_to_ee_frame(monkeypatch):
    session = vr.VRTeleopSession()
    fake_motors = _BoundsOnlyMotors()
    monkeypatch.setattr(vr, "MOTORS", fake_motors)
    monkeypatch.setattr(vr, "_load_urdf_kinematics", lambda: _FakeKinematics())

    arm = session._arms["right"]
    arm.latest = vr._LatestGoal(
        received_at=1.0,
        has_data=True,
        mode="reset",
        controller_position=(0.1, 0.2, 0.3),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
    )

    session._capture_anchor("right")

    assert arm.controller_anchor_T is not None
    assert arm.vr_ctrl_to_ee is not None
    np.testing.assert_allclose(arm.vr_offset_accum, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(arm.pending_rel_position, (0.0, 0.0, 0.0))


def test_capture_anchor_resets_quality_metrics(monkeypatch):
    session = vr.VRTeleopSession()
    fake_motors = _BoundsOnlyMotors()
    monkeypatch.setattr(vr, "MOTORS", fake_motors)
    monkeypatch.setattr(vr, "_load_urdf_kinematics", lambda: _FakeKinematics())

    arm = session._arms["right"]
    arm.quality_ticks = 12
    arm.quality_ik_rejects = 3
    arm.quality_offset_speed_ema_mps = 0.42
    arm.quality_last_offset_step_m = 0.01
    arm.latest = vr._LatestGoal(
        received_at=1.0,
        has_data=True,
        mode="reset",
        controller_position=(0.1, 0.2, 0.3),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
    )

    session._capture_anchor("right")

    assert arm.quality_ticks == 0
    assert arm.quality_ik_rejects == 0
    assert arm.quality_offset_speed_ema_mps == pytest.approx(0.0)
    assert arm.quality_last_offset_step_m == pytest.approx(0.0)


def test_dual_arm_relative_offsets_are_independent(monkeypatch):
    session = vr.VRTeleopSession()
    fake_motors = _BoundsOnlyMotors()
    fake_motors.connected_sides = ["left", "right"]
    monkeypatch.setattr(vr, "MOTORS", fake_motors)
    monkeypatch.setattr(vr, "POS_EMA_ALPHA", 1.0)
    monkeypatch.setattr(vr, "MAX_EE_STEP_M", 1.0)
    monkeypatch.setattr(vr, "POS_DEADZONE_M", 0.0)

    for side, rel in (("left", (0.0, 0.02, 0.0)), ("right", (0.01, 0.0, 0.0))):
        arm = session._arms[side]
        _mark_urdf_available(arm)
        arm.session_vr_to_robot = np.eye(3)
        arm.controller_anchor_T = vr._pose_matrix_from_vr((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        arm.anchor_ee_pos = (0.20, 0.0, 0.05)
        arm.anchor_R_robot = np.eye(3)
        arm.smoothed_R_target = np.eye(3)
        arm.target_R_robot = np.eye(3)
        arm.anchor = vr._AnchorPose(
            shoulder_lift_deg=-60.0,
            elbow_flex_deg=45.0,
            wrist_flex_deg=20.0,
            wrist_roll_deg=0.0,
            captured=True,
        )
        _queue_rel(arm, rel)
        session._compute_targets_from_vr(
            side,
            vr._LatestGoal(
                received_at=1.0,
                has_data=True,
                mode="position",
                rel_position=rel,
                controller_position=rel,
                rotation_quat=(0.0, 0.0, 0.0, 1.0),
                trigger=False,
            ),
            scale=1.0,
        )

    np.testing.assert_allclose(session._arms["left"].vr_offset_accum, (0.0, 0.02, 0.0))
    np.testing.assert_allclose(session._arms["right"].vr_offset_accum, (0.01, 0.0, 0.0))


def test_invert_lateral_keeps_controller_to_ee_wrist_alignment(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr, "MOTORS", _BoundsOnlyMotors())
    monkeypatch.setattr(vr, "POS_EMA_ALPHA", 1.0)
    monkeypatch.setattr(vr, "MAX_EE_STEP_M", 1.0)
    monkeypatch.setattr(vr, "POS_DEADZONE_M", 0.0)

    arm = session._arms["right"]
    _mark_urdf_available(arm)
    arm.session_vr_to_robot = np.eye(3)
    arm.invert_lateral = True
    arm.controller_anchor_T = vr._pose_matrix_from_vr((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    arm.vr_ctrl_to_ee = vr._R.from_euler("z", 90, degrees=True)
    arm.anchor_ee_pos = (0.20, 0.0, 0.05)
    arm.anchor_R_robot = np.eye(3)
    arm.smoothed_R_target = np.eye(3)
    arm.target_R_robot = np.eye(3)
    arm.anchor = vr._AnchorPose(
        shoulder_lift_deg=-60.0,
        elbow_flex_deg=45.0,
        wrist_flex_deg=20.0,
        wrist_roll_deg=0.0,
        captured=True,
    )

    session._compute_targets_from_vr(
        "right",
        vr._LatestGoal(
            received_at=1.0,
            has_data=True,
            mode="position",
            rel_position=(0.0, 0.0, 0.0),
            controller_position=(0.0, 0.0, 0.0),
            rotation_quat=(0.0, 0.0, 0.0, 1.0),
            trigger=False,
        ),
        scale=1.0,
    )

    pitch_axis, _ = vr._effective_wrist_axes(
        "right",
        pitch_canonical=arm.wrist_pitch_canonical,
        roll_canonical=arm.wrist_roll_canonical,
    )
    expected_pitch = arm.vr_ctrl_to_ee.apply(pitch_axis)
    np.testing.assert_allclose(arm.last_diag["wrist_axes"]["pitch"], expected_pitch, atol=1e-6)


@pytest.mark.parametrize(
    ("rotvec", "delta_index"),
    [
        ((0.10, 0.00, 0.00), 0),
        ((0.00, 0.00, -0.10), 1),
    ],
)
def test_positive_wrist_motor_polarity_preserves_calibrated_direction(monkeypatch, rotvec, delta_index):
    session = vr.VRTeleopSession()
    arm = _configure_compute_target_arm(session, monkeypatch, matrix=np.eye(3))
    monkeypatch.setitem(
        vr._WRIST_MOTOR_POLARITY,
        "right",
        {"flex": 1.0, "roll": 1.0},
    )

    quat = vr._R.from_rotvec(np.asarray(rotvec, dtype=float)).as_quat()
    session._compute_targets_from_vr(
        "right",
        vr._LatestGoal(
            received_at=1.0,
            has_data=True,
            mode="position",
            rel_position=(0.0, 0.0, 0.0),
            controller_position=(0.0, 0.0, 0.0),
            rotation_quat=tuple(float(v) for v in quat),
            trigger=False,
        ),
        scale=1.0,
    )

    assert arm.last_diag["wrist_delta_deg"][delta_index] > 0.0


def test_robot_verification_capture_uses_absolute_controller_delta(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr, "MOTORS", _BoundsOnlyMotors())
    arm = session._arms["right"]
    arm.robot_verify_state = "vr_start_captured"
    arm.robot_verify_robot_start = (0.20, 0.00, 0.05)
    arm.robot_verify_robot_end = (0.25, 0.00, 0.05)
    arm.robot_verify_vr_start = (0.00, 0.00, 0.00)
    arm.robot_verify_vr_delta_accum = (0.0, 0.0, 0.0)
    arm.latest = vr._LatestGoal(
        received_at=1.0,
        has_data=True,
        mode="position",
        controller_position=(0.10, 0.00, 0.00),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
    )
    monkeypatch.setattr(vr.time, "time", lambda: 1.0)

    session.capture_robot_verification_vr("right", "end", "forward")

    assert arm.robot_verify_samples[-1]["vr_delta"] == pytest.approx([0.10, 0.00, 0.00])
    assert arm.robot_verify_samples[-1]["vr_delta_source"] == "absolute_controller_pose"


def test_robot_verification_live_status_uses_absolute_controller_delta(monkeypatch):
    session = vr.VRTeleopSession()
    arm = session._arms["right"]
    arm.robot_verify_state = "vr_start_captured"
    arm.robot_verify_robot_start = (0.20, 0.00, 0.05)
    arm.robot_verify_robot_end = (0.25, 0.00, 0.05)
    arm.robot_verify_vr_start = (0.00, 0.00, 0.00)
    arm.robot_verify_vr_delta_accum = (0.0, 0.0, 0.0)
    arm.latest = vr._LatestGoal(
        received_at=1.0,
        has_data=True,
        mode="position",
        controller_position=(0.10, 0.00, 0.00),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
    )

    live = session._robot_verification_live_status(arm, now=1.0)

    assert live["vr_delta"] == pytest.approx([0.10, 0.00, 0.00])
    assert live["vr_delta_source"] == "absolute_controller_pose"


def test_robot_verification_live_status_uses_solve_runtime_model_not_provisional_lstsq():
    session = vr.VRTeleopSession()
    arm = session._arms["right"]
    arm.session_vr_to_robot = vr._VR_TO_ROBOT.copy()
    arm.base_vr_direction_matrix = vr._VR_TO_ROBOT.copy()
    arm.robot_verify_state = "vr_start_captured"
    arm.robot_verify_robot_start = (0.0, 0.0, 0.0)
    arm.robot_verify_robot_end = (0.0, 0.0, 0.10)
    arm.robot_verify_vr_start = (0.0, 0.0, 0.0)
    # This is a twisted "up" movement: mostly up, but with lateral drift. The
    # old provisional least-squares preview could learn/cancel that drift and
    # mark it green, while final solve/runtime still used the stage-1 frame.
    arm.robot_verify_vr_delta_accum = (0.05, 0.10, 0.0)
    provisional = np.array([
        [0.0, 0.0, -1.0],
        [-1.0, 0.5, 0.0],
        [0.0, 1.0, 0.0],
    ])
    arm.robot_verify_samples = [
        _sample("forward", [1.0, 0.0, 0.0], provisional @ np.array([1.0, 0.0, 0.0])),
        _sample("left", [0.0, 1.0, 0.0], provisional @ np.array([0.0, 1.0, 0.0])),
        _sample("up", [0.0, 0.0, 1.0], provisional @ np.array([0.0, 0.0, 1.0])),
    ]
    arm.latest = vr._LatestGoal(
        received_at=1.0,
        has_data=True,
        mode="position",
        controller_position=(0.05, 0.10, 0.0),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
    )

    live = session._robot_verification_live_status(arm, now=1.0)

    assert live["preview_source"] == "stage1_scaled"
    assert live["state"] == "adjust"
    assert live["position_error_cm"] > vr.ROBOT_VERIFY_PASS_ERROR_CM


def test_robot_verification_live_good_requires_solve_position_error_threshold():
    session = vr.VRTeleopSession()
    arm = session._arms["right"]
    arm.session_vr_to_robot = vr._VR_TO_ROBOT.copy()
    arm.base_vr_direction_matrix = vr._VR_TO_ROBOT.copy()
    arm.robot_verify_state = "vr_start_captured"
    arm.robot_verify_robot_start = (0.0, 0.0, 0.0)
    arm.robot_verify_robot_end = (0.0, 0.0, 0.20)
    arm.robot_verify_vr_start = (0.0, 0.0, 0.0)
    # Same direction and within the old ratio band, but the endpoint is 7 cm
    # short. It must not show green because solve scores RMS position residual.
    arm.robot_verify_vr_delta_accum = (0.0, 0.0, 0.13)
    arm.latest = vr._LatestGoal(
        received_at=1.0,
        has_data=True,
        mode="position",
        controller_position=(0.0, 0.0, 0.13),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
    )

    live = session._robot_verification_live_status(arm, now=1.0)

    assert live["direction_error_deg"] == pytest.approx(0.0, abs=1e-6)
    assert live["magnitude_ratio"] == pytest.approx(0.65)
    assert live["position_error_cm"] == pytest.approx(7.0)
    assert live["state"] == "adjust"


def test_robot_verification_vr_start_requires_active_grip(monkeypatch):
    session = vr.VRTeleopSession()
    arm = session._arms["right"]
    arm.robot_verify_state = "robot_end_captured"
    arm.robot_verify_robot_start = (0.0, 0.0, 0.0)
    arm.robot_verify_robot_end = (0.10, 0.0, 0.0)
    arm.latest = vr._LatestGoal(
        received_at=vr.time.time(),
        has_data=True,
        mode="idle",
        controller_position=(0.0, 0.0, 0.0),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
    )

    with pytest.raises(RuntimeError, match="hold grip"):
        session.capture_robot_verification_vr("right", "start", "forward")


def test_robot_verification_grip_release_resets_in_progress_vr_capture():
    session = vr.VRTeleopSession()
    arm = session._arms["right"]
    arm.robot_verify_state = "vr_start_captured"
    arm.robot_verify_label = "forward"
    arm.robot_verify_robot_start = (0.0, 0.0, 0.0)
    arm.robot_verify_robot_end = (0.10, 0.0, 0.0)
    arm.robot_verify_vr_start = (0.0, 1.0, 0.0)
    arm.robot_verify_vr_delta_accum = (0.04, 0.01, 0.0)

    session._apply_control_goal(SimpleNamespace(
        arm="right",
        mode="idle",
        relative_position=(0.0, 0.0, 0.0),
        relative_rotvec=(0.0, 0.0, 0.0),
        vr_ctrl_position=(0.0, 1.0, 0.0),
        vr_ctrl_rotation=None,
        trigger=False,
        thumbstick={},
        buttons={},
    ))

    assert arm.robot_verify_state == "robot_end_captured"
    assert arm.robot_verify_robot_start == (0.0, 0.0, 0.0)
    assert arm.robot_verify_robot_end == (0.10, 0.0, 0.0)
    assert arm.robot_verify_vr_start is None
    assert arm.robot_verify_vr_delta_accum == (0.0, 0.0, 0.0)
    assert arm.robot_verify_label == "forward"


def test_robot_verification_robot_pose_capture_sets_selected_label(monkeypatch):
    session = vr.VRTeleopSession()
    arm = session._arms["right"]
    arm.robot_verify_state = "collecting"

    T = np.eye(4)
    T[:3, 3] = (0.20, 0.0, 0.05)
    monkeypatch.setattr(session, "_current_ee_transform", lambda side: T)

    session.capture_robot_verification_pose("right", "start", label="left")

    assert arm.robot_verify_label == "left"
    assert arm.robot_verify_state == "robot_start_captured"


def test_robot_verification_vr_end_does_not_release_torque(monkeypatch):
    class Motors:
        connected_sides = ["right"]
        bounds = {}

        def is_connected(self, side):
            return True

        def is_torque_enabled(self, side):
            return True

        def read_positions(self, side=None):
            return {}

        def release_torque_for_posing(self, side):
            raise AssertionError("VR end capture must not release torque")

    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr, "MOTORS", Motors())
    arm = session._arms["right"]
    arm.robot_verify_state = "vr_start_captured"
    arm.robot_verify_label = "forward"
    arm.robot_verify_robot_start = (0.0, 0.0, 0.0)
    arm.robot_verify_robot_end = (0.10, 0.0, 0.0)
    arm.robot_verify_vr_start = (0.0, 1.0, 0.0)
    arm.robot_verify_vr_delta_accum = (0.10, 0.0, 0.0)
    arm.latest = vr._LatestGoal(
        received_at=vr.time.time(),
        has_data=True,
        mode="position",
        controller_position=(0.10, 1.0, 0.0),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
        trigger=False,
    )

    session.capture_robot_verification_vr("right", "end", "forward")

    assert arm.robot_verify_state == "collecting"
    assert len(arm.robot_verify_samples) == 1


def test_robot_verification_solve_accepts_clean_six_axis_fit():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    expected = 0.5 * vr._VR_TO_ROBOT
    arm.session_vr_to_robot = vr._VR_TO_ROBOT.copy()
    arm.robot_verify_samples = _six_axis_samples(expected)

    final_m, translation_m, scale, fit_error_cm, quality, residuals = (
        session._solve_robot_verified_calibration(arm)
    )

    np.testing.assert_allclose(translation_m, expected, atol=1e-6)
    assert fit_error_cm == pytest.approx(0.0, abs=1e-5)
    assert quality == "good"
    assert scale == pytest.approx(0.5, abs=1e-6)
    assert len(residuals) == 6
    assert final_m.shape == (3, 3)


def test_robot_verification_solve_scores_against_lateral_inverted_live_frame():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    arm.invert_lateral = True
    arm.session_vr_to_robot = vr._VR_TO_ROBOT.copy()
    expected = 0.5 * (np.diag([1.0, -1.0, 1.0]) @ vr._VR_TO_ROBOT)
    arm.robot_verify_samples = _six_axis_samples(expected)

    _final_m, translation_m, scale, fit_error_cm, quality, _residuals = (
        session._solve_robot_verified_calibration(arm)
    )

    np.testing.assert_allclose(translation_m, expected, atol=1e-6)
    assert scale == pytest.approx(0.5, abs=1e-6)
    assert fit_error_cm == pytest.approx(0.0, abs=1e-5)
    assert quality == "good"


def test_robot_verification_solve_marks_inconsistent_fit_for_recapture():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    arm.session_vr_to_robot = np.eye(3)
    expected = np.eye(3) * 0.5
    samples = _six_axis_samples(expected)
    samples[0] = _sample("forward", [0.10, 0.0, 0.0], [0.15, 0.10, 0.0])
    arm.robot_verify_samples = samples

    *_unused, fit_error_cm, quality, _residuals = session._solve_robot_verified_calibration(arm)

    assert fit_error_cm > vr.ROBOT_VERIFY_PASS_ERROR_CM
    assert quality in {"needs_recapture", "poor"}


def test_robot_verification_residuals_report_per_sample_vr_motion():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    arm.session_vr_to_robot = np.eye(3)
    expected = np.eye(3) * 0.5
    arm.robot_verify_samples = [
        _sample("forward", [0.12, 0.00, 0.00], expected @ np.array([0.12, 0.00, 0.00])),
        _sample("back", [-0.08, 0.00, 0.00], expected @ np.array([-0.08, 0.00, 0.00])),
        _sample("left", [0.00, 0.10, 0.00], expected @ np.array([0.00, 0.10, 0.00])),
        _sample("right", [0.00, -0.11, 0.00], expected @ np.array([0.00, -0.11, 0.00])),
        _sample("up", [0.00, 0.00, 0.09], expected @ np.array([0.00, 0.00, 0.09])),
        _sample("down", [0.00, 0.00, -0.07], expected @ np.array([0.00, 0.00, -0.07])),
    ]

    *_unused, residuals = session._solve_robot_verified_calibration(arm)

    motions = {item["label"]: item["vr_motion_cm"] for item in residuals}
    assert motions["forward"] == pytest.approx(12.0)
    assert motions["back"] == pytest.approx(8.0)
    assert motions["down"] == pytest.approx(7.0)


def test_robot_verification_failed_solve_reports_worst_residual(monkeypatch):
    session = vr.VRTeleopSession()
    arm = session._arms["right"]
    arm.session_vr_to_robot = np.eye(3)
    expected = np.eye(3) * 0.5
    samples = _six_axis_samples(expected)
    samples[0] = _sample("forward", [0.10, 0.0, 0.0], [0.15, 0.10, 0.0])
    arm.robot_verify_samples = samples
    monkeypatch.setattr(session, "_require_urdf_kinematics", lambda side, context: None)

    with pytest.raises(RuntimeError) as exc:
        session.solve_robot_verification("right")

    message = str(exc.value)
    assert "Worst residuals:" in message
    assert "Recapture forward" in message
    assert arm.robot_verify_sample_residuals
    assert arm.robot_verify_quality in {"needs_recapture", "poor"}


def test_robot_verification_solve_rejects_raw_fit_that_breaks_live_forward_axis():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="left")
    arm.session_vr_to_robot = vr._VR_TO_ROBOT.copy()
    raw_fit_only = np.array([
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5],
    ])
    arm.robot_verify_samples = _six_axis_samples(raw_fit_only)

    _final_m, translation_m, _scale, fit_error_cm, quality, residuals = (
        session._solve_robot_verified_calibration(arm)
    )

    np.testing.assert_allclose(translation_m, raw_fit_only, atol=1e-6)
    assert fit_error_cm > vr.ROBOT_VERIFY_PASS_ERROR_CM
    assert quality in {"needs_recapture", "poor"}
    assert any(item["label"] == "forward" for item in residuals)


def test_robot_verification_requires_named_six_directions():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    arm.robot_verify_samples = [
        _sample(f"sample-{idx}", [0.03 + idx * 0.01, 0.04, 0.05], [0.02, 0.03 + idx * 0.01, 0.04])
        for idx in range(6)
    ]

    with pytest.raises(RuntimeError, match="missing required directions"):
        session._solve_robot_verified_calibration(arm)


def test_robot_verification_missing_labels_ignores_too_small_direction_samples():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    arm.session_vr_to_robot = vr._VR_TO_ROBOT.copy()
    expected = 0.5 * vr._VR_TO_ROBOT
    arm.robot_verify_samples = [
        _sample("forward", [0.005, 0.0, 0.0], expected @ np.array([0.005, 0.0, 0.0])),
        _sample("back", [-0.10, 0.00, 0.00], expected @ np.array([-0.10, 0.00, 0.00])),
        _sample("left", [0.00, 0.10, 0.00], expected @ np.array([0.00, 0.10, 0.00])),
        _sample("right", [0.00, -0.10, 0.00], expected @ np.array([0.00, -0.10, 0.00])),
        _sample("up", [0.00, 0.00, 0.10], expected @ np.array([0.00, 0.00, 0.10])),
        _sample("down", [0.00, 0.00, -0.10], expected @ np.array([0.00, 0.00, -0.10])),
        _sample("left", [0.00, 0.08, 0.00], expected @ np.array([0.00, 0.08, 0.00])),
    ]

    assert session._missing_robot_verification_labels(arm) == ["forward"]

    with pytest.raises(RuntimeError, match="verification samples missing required directions: forward"):
        session._solve_robot_verified_calibration(arm)


def test_robot_verification_accepts_legacy_ordered_sample_labels():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    expected = 0.5 * vr._VR_TO_ROBOT
    arm.session_vr_to_robot = vr._VR_TO_ROBOT.copy()
    arm.robot_verify_samples = _six_axis_samples(expected)
    for idx, sample in enumerate(arm.robot_verify_samples, start=1):
        sample["label"] = f"sample-{idx}"

    assert session._missing_robot_verification_labels(arm) == []

    _final_m, _translation_m, _scale, fit_error_cm, quality, residuals = (
        session._solve_robot_verified_calibration(arm)
    )

    assert fit_error_cm == pytest.approx(0.0, abs=1e-5)
    assert quality == "good"
    assert [item["label"] for item in residuals[:6]] == list(vr.ROBOT_VERIFY_REQUIRED_LABELS)


def test_robot_verification_start_requires_urdf_kinematics(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr, "MOTORS", _BoundsOnlyMotors())
    monkeypatch.setattr(vr, "_load_urdf_kinematics", lambda: None)

    with pytest.raises(RuntimeError, match="URDF kinematics unavailable"):
        session.start_robot_verification("right")


def test_recording_blocker_requires_urdf_kinematics(monkeypatch):
    session = vr.VRTeleopSession()
    _mark_recording_ready(session, monkeypatch)
    session._arms["right"].kinematics = None
    session._arms["right"].using_analytical_fallback = False
    monkeypatch.setattr(vr, "_load_urdf_kinematics", lambda: None)

    blockers = session._recording_calibration_blockers()

    assert any("URDF kinematics missing" in blocker for blocker in blockers)


def test_status_exposes_reachy_style_operator_summary(monkeypatch):
    session = vr.VRTeleopSession()
    motors = _BothArmBoundsOnlyMotors()
    monkeypatch.setattr(vr, "MOTORS", motors)
    monkeypatch.setattr(
        vr._cameras,
        "enumerate_cameras",
        lambda: [_FakeRoleCamera("head"), _FakeRoleCamera("left_wrist"), _FakeRoleCamera("right_wrist")],
    )
    monkeypatch.setattr(
        vr._dataset,
        "role_camera_list",
        lambda: (["head", "left_wrist", "right_wrist"], (2, 2, 3)),
    )
    monkeypatch.setattr(vr._cameras, "suspended_capture_roles", lambda: {})
    monkeypatch.setattr(vr._home, "home_pose_status", lambda: {
        "left": {"captured": True, "joints": {}},
        "right": {"captured": True, "joints": {}},
    })
    monkeypatch.setattr(vr._vrcal, "status", lambda: {
        "left": {"saved": False},
        "right": {"saved": True},
    })
    monkeypatch.setattr(vr._vrcal, "profile_status", lambda: {"active_profile": "default", "profiles": []})
    session._native_quest_clients = 1
    for side in ("left", "right"):
        arm = session._arms[side]
        _mark_urdf_available(arm)
        arm.cal_confidence = "good"
        arm.robot_verify_quality = "good"
        arm.robot_verify_fit_error_cm = 0.0
        arm.robot_verify_test_completed = True
        arm.calibrated = True
        arm.anchor_generation = arm.pose_generation
        arm.controller_anchor_T = vr._pose_matrix_from_vr((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        arm.vr_ctrl_to_ee = vr._R.identity()
        arm.latest = vr._LatestGoal(
            received_at=vr.time.time(),
            has_data=True,
            mode="position",
            controller_position=(0.0, 0.0, 0.0),
            rotation_quat=(0.0, 0.0, 0.0, 1.0),
        )

    status = session.status()

    assert status["operator"]["stage"] == "mirror_ready"
    assert status["operator"]["head_camera_url"] == "/camera/head/stream"
    assert status["operator"]["connection"]["websocket_clients"] == 1
    assert status["operator"]["arm_panels"]["right"]["wrist_aligned"] is True
    assert status["operator"]["recording"]["ready"] is True


def test_recording_blocker_requires_good_robot_verification(monkeypatch):
    session = vr.VRTeleopSession()
    _mark_recording_ready(session, monkeypatch)
    arm = session._arms["right"]
    arm.robot_verify_quality = "needs_recapture"
    arm.robot_verify_fit_error_cm = vr.ROBOT_VERIFY_PASS_ERROR_CM + 0.1

    blockers = session._recording_calibration_blockers()

    assert any("robot verification" in blocker for blocker in blockers)


def test_recording_blocker_requires_low_scale_test_after_good_verification(monkeypatch):
    session = vr.VRTeleopSession()
    _mark_recording_ready(session, monkeypatch)
    arm = session._arms["right"]
    arm.robot_verify_test_completed = False

    blockers = session._recording_calibration_blockers()

    assert any("low-scale calibration test not completed" in blocker for blocker in blockers)


def test_recording_blocker_clears_after_low_scale_test_completed(monkeypatch):
    session = vr.VRTeleopSession()
    _mark_recording_ready(session, monkeypatch)

    assert session._recording_calibration_blockers() == []
