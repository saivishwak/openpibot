import math

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


def test_analytical_so101_fk_ik_roundtrip():
    kin = vr._SO101Kin()
    for shoulder_lift, elbow_flex in [(-80.0, 60.0), (-35.0, 25.0), (20.0, -10.0)]:
        x, z = kin.forward(shoulder_lift, elbow_flex)
        sl2, ef2 = kin.inverse(x, z)
        assert sl2 == pytest.approx(shoulder_lift, abs=1e-6)
        assert ef2 == pytest.approx(elbow_flex, abs=1e-6)


def test_real_teleop_uses_calibrated_so101_urdf_when_available(monkeypatch):
    fake_kinematics = object()
    monkeypatch.setattr(vr, "_load_urdf_kinematics", lambda: fake_kinematics)

    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")

    assert session._ensure_kinematics(arm) is True
    assert arm.kinematics is fake_kinematics
    assert arm.using_analytical_fallback is False


def test_real_teleop_falls_back_to_analytical_without_urdf(monkeypatch):
    monkeypatch.setattr(vr, "_load_urdf_kinematics", lambda: None)

    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")

    assert session._ensure_kinematics(arm) is False
    assert arm.kinematics is None
    assert arm.using_analytical_fallback is True


def test_wrist_axes_apply_left_handedness_once():
    right_pitch, right_roll = vr._effective_wrist_axes("right")
    left_pitch, left_roll = vr._effective_wrist_axes("left")

    assert right_pitch.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert right_roll.tolist() == pytest.approx([0.0, 0.0, -1.0])
    assert left_pitch.tolist() == pytest.approx([-1.0, 0.0, 0.0])
    assert left_roll.tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_runtime_translation_prefers_robot_verified_matrix():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    arm.session_vr_to_robot = np.eye(3)
    arm.translation_scale = 0.5
    verified = np.array([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    arm.translation_vr_to_robot = verified
    arm.robot_verify_quality = "good"

    assert session._runtime_translation_matrix(arm) is not None
    np.testing.assert_allclose(session._runtime_translation_matrix(arm), verified)


def test_runtime_translation_does_not_double_invert_robot_verified_matrix():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    arm.session_vr_to_robot = np.eye(3)
    arm.invert_lateral = True
    verified = np.array([[0.0, -0.4, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.4]])
    arm.translation_vr_to_robot = verified
    arm.robot_verify_quality = "good"

    np.testing.assert_allclose(session._runtime_translation_matrix(arm), verified)


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


def test_compute_targets_reconciles_clamped_offset(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr, "MOTORS", _BoundsOnlyMotors())

    arm = session._arms["right"]
    arm.using_analytical_fallback = True
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


def test_compute_targets_integrates_relative_position_when_world_pose_is_static(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr, "MOTORS", _BoundsOnlyMotors())
    monkeypatch.setattr(vr, "POS_EMA_ALPHA", 1.0)
    monkeypatch.setattr(vr, "MAX_EE_STEP_M", 1.0)
    monkeypatch.setattr(vr, "POS_DEADZONE_M", 0.0)

    arm = session._arms["right"]
    arm.using_analytical_fallback = True
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
    session._compute_targets_from_vr("right", goal, scale=1.0)

    np.testing.assert_allclose(arm.vr_offset_accum, (0.01, 0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(arm.offset_robot, (0.01, 0.0, 0.0), atol=1e-9)


def test_compute_targets_keeps_sub_deadzone_packets_after_batching(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr, "MOTORS", _BoundsOnlyMotors())
    monkeypatch.setattr(vr, "POS_EMA_ALPHA", 1.0)
    monkeypatch.setattr(vr, "MAX_EE_STEP_M", 1.0)
    monkeypatch.setattr(vr, "POS_DEADZONE_M", 0.001)

    arm = session._arms["right"]
    arm.using_analytical_fallback = True
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
        controller_position=(0.0, 0.0, 0.0),
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
    arm.using_analytical_fallback = True
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
        controller_position=(0.0, 0.0, 0.0),
        rotation_quat=(0.0, 0.0, 0.0, 1.0),
        trigger=False,
    )
    session._compute_targets_from_vr("right", goal, scale=1.0)

    assert np.linalg.norm(np.asarray(arm.offset_robot, dtype=float)) == pytest.approx(0.004)
    assert arm.quality_last_offset_step_m == pytest.approx(0.004)


def test_capture_anchor_sets_controller_to_ee_frame(monkeypatch):
    session = vr.VRTeleopSession()
    fake_motors = _BoundsOnlyMotors()
    monkeypatch.setattr(vr, "MOTORS", fake_motors)
    monkeypatch.setattr(vr, "_load_urdf_kinematics", lambda: None)

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
    monkeypatch.setattr(vr, "_load_urdf_kinematics", lambda: None)

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
        arm.using_analytical_fallback = True
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
        session._compute_targets_from_vr(
            side,
            vr._LatestGoal(
                received_at=1.0,
                has_data=True,
                mode="position",
                rel_position=rel,
                controller_position=(0.0, 0.0, 0.0),
                rotation_quat=(0.0, 0.0, 0.0, 1.0),
                trigger=False,
            ),
            scale=1.0,
        )

    np.testing.assert_allclose(session._arms["left"].vr_offset_accum, (0.0, 0.02, 0.0))
    np.testing.assert_allclose(session._arms["right"].vr_offset_accum, (0.01, 0.0, 0.0))


def test_robot_verification_rejects_absolute_pose_fallback_without_relative_motion(monkeypatch):
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

    with pytest.raises(RuntimeError, match="hold grip"):
        session.capture_robot_verification_vr("right", "end", "forward")


def test_robot_verification_live_status_requires_relative_motion(monkeypatch):
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

    assert live["state"] == "move_vr"
    assert "Hold grip" in live["message"]


def test_robot_verification_solve_accepts_clean_six_axis_fit():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    expected = np.array([[0.0, -0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]])
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


def test_robot_verification_solve_does_not_apply_lateral_invert_to_samples():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    arm.invert_lateral = True
    expected = np.array([[0.0, -0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]])
    arm.robot_verify_samples = _six_axis_samples(expected)

    _final_m, translation_m, _scale, fit_error_cm, quality, _residuals = (
        session._solve_robot_verified_calibration(arm)
    )

    np.testing.assert_allclose(translation_m, expected, atol=1e-6)
    assert fit_error_cm == pytest.approx(0.0, abs=1e-5)
    assert quality == "good"


def test_robot_verification_solve_marks_inconsistent_fit_for_recapture():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    expected = np.eye(3) * 0.5
    samples = _six_axis_samples(expected)
    samples[0] = _sample("forward", [0.10, 0.0, 0.0], [0.15, 0.10, 0.0])
    arm.robot_verify_samples = samples

    *_unused, fit_error_cm, quality, _residuals = session._solve_robot_verified_calibration(arm)

    assert fit_error_cm > vr.ROBOT_VERIFY_PASS_ERROR_CM
    assert quality in {"needs_recapture", "poor"}


def test_robot_verification_requires_named_six_directions():
    session = vr.VRTeleopSession()
    arm = vr._PerArm(side="right")
    arm.robot_verify_samples = [
        _sample(f"sample-{idx}", [0.03 + idx * 0.01, 0.04, 0.05], [0.02, 0.03 + idx * 0.01, 0.04])
        for idx in range(6)
    ]

    with pytest.raises(RuntimeError, match="missing required directions"):
        session._solve_robot_verified_calibration(arm)


def test_recording_blocker_requires_good_robot_verification(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr, "MOTORS", _BoundsOnlyMotors())
    arm = session._arms["right"]
    arm.cal_confidence = "good"
    arm.robot_verify_quality = "needs_recapture"
    arm.robot_verify_fit_error_cm = vr.ROBOT_VERIFY_PASS_ERROR_CM + 0.1

    blockers = session._recording_calibration_blockers()

    assert any("robot verification" in blocker for blocker in blockers)


def test_recording_blocker_requires_low_scale_test_after_good_verification(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr, "MOTORS", _BoundsOnlyMotors())
    arm = session._arms["right"]
    arm.cal_confidence = "good"
    arm.robot_verify_quality = "good"
    arm.robot_verify_fit_error_cm = 0.0

    blockers = session._recording_calibration_blockers()

    assert any("low-scale calibration test not completed" in blocker for blocker in blockers)


def test_recording_blocker_clears_after_low_scale_test_completed(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr, "MOTORS", _BoundsOnlyMotors())
    arm = session._arms["right"]
    arm.cal_confidence = "good"
    arm.robot_verify_quality = "good"
    arm.robot_verify_fit_error_cm = 0.0
    arm.robot_verify_test_completed = True

    assert session._recording_calibration_blockers() == []
