"""Safe single-arm VR teleop session.

The dashboard's only control surface. Drives **one** arm at a time via the
Meta Quest 3 controller stream, with three independent safety guards:

  1. Engagement gate — UI toggle. Motors stay still until the user flips it.
  2. Calibration gate — even while engaged, motors stay still until the user
     issues a Quest-controller RESET. The RESET anchors VR poses to the
     robot's current EE pose, so motion is always *relative* and bounded.
  3. Watchdog — if VR goals stop arriving (controller put down, browser
     closed, network blip), the drive loop stops sending within 0.3 s and
     auto-disengages within 1 s.

Plus per-tick joint clamps and the degree-space calibration bounds already
in motors.SESSION.bounds.

The live controller path is reset-relative, not packet-delta-integrated:
Quest poses are normalized to `quest_operator_frame`, each grip RESET anchors
the current controller pose to the current robot EE pose, and every drive tick
maps the absolute controller displacement since that anchor through the shared
per-arm calibration mapper.

Use `RobotKinematics` with SO-ARM100's `so101_new_calib.urdf` for teleop.
Calibration, robot verification, live control, and recording refuse to proceed
without that calibrated URDF. The analytical 2-link model remains only as an
isolated math helper for round-trip tests.
"""
from __future__ import annotations

import json
import logging
import hashlib
import math
import os
import pathlib
import secrets
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

from . import motors as _motors
from .motors import SESSION as MOTORS, ArmSide
from . import cameras as _cameras
from . import dataset as _dataset
from . import home as _home
from .homing import HomingStep, JointHomingController, normalized_joint_name
from . import quest_video_bridge as _quest_video_bridge
from . import vr_calibration as _vrcal
from .native_quest import MAX_NATIVE_QUEST_PACKET_BYTES, NativeQuestAdapter, NativeQuestProtocolError
from openpibot.server.config import REPO_ROOT

log = logging.getLogger(__name__)

# --- control / safety constants -----------------------------------------------

LOOP_HZ = 30.0
LOOP_PERIOD_S = 1.0 / LOOP_HZ

GOAL_SKIP_AGE_S = 0.30     # skip motor write if VR goal older than this
# Homing-only software P-control. Live VR bypasses present-position blending so
# bus read noise cannot perturb controller-driven actions.
KP = 0.75

# Per-tick caps. At LOOP_HZ=30 these are effectively doubled in deg/sec vs the
# old 15 Hz tuning, putting max joint speeds in the natural-hand-motion range:
#   shoulder_pan 4°/tick × 30Hz = 120°/s
#   shoulder_lift 2°/tick × 30Hz = 60°/s (gravity-loaded, kept tighter)
#   elbow_flex 3°/tick × 30Hz = 90°/s
#   wrist 4°/tick × 30Hz = 120°/s
PER_TICK_DEG_CAPS: dict[str, float] = {
    # At 30 Hz: 5°/tick = 150°/s, 6°/tick = 180°/s. These are deliberately
    # below the recent 8-12° caps but still responsive enough for hand teleop.
    "shoulder_pan":  5.0,
    "shoulder_lift": 5.0,
    "elbow_flex":    5.0,
    "wrist_flex":    6.0,
    "wrist_roll":    6.0,
    "gripper":       15.0,   # ~0.25s open/close
}

# Do not send tiny live-VR target changes to the servos. Quest pose noise and
# URDF IK can create sub-degree target movement even when the operator is
# holding still; this deadband stops that noise without changing large motions.
JOINT_COMMAND_DEADBAND_DEG: dict[str, float] = {
    "shoulder_pan":  0.18,
    "shoulder_lift": 0.18,
    "elbow_flex":    0.18,
    "wrist_flex":    0.25,
    "wrist_roll":    0.25,
    "gripper":       0.0,
}
JOINT_COMMAND_FILTER_WEIGHTS: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)
JOINT_COMMAND_FILTER_BYPASS: set[str] = {"wrist_flex", "wrist_roll", "gripper"}

SAFE_REAR_X_M = 0.035
IK_JUMP_REJECT_DEG = 25.0

# Full 3D EE position is tracked (not just planar). At each tick the calibrated
# controller displacement yields a robot-frame EE target; URDF IK solves the arm
# joints, and wrist_flex/wrist_roll carry the reset-relative controller rotation.

# EE-position safety box in robot base frame (metres). Tuned for the SO101's
# actual reach (L1+L2 = 0.251m). Includes the full hemisphere in front + sides,
# with small clearance behind the base (in case shoulder_pan rotates >90°).
# The IK and motor-calibration clamps downstream are the final safety net.
EE_BOUNDS = {
    # Conservative sanity box outside which we do not try IK. Analytical IK
    # then softly saturates against the actual planar linkage reach.
    "x": (-0.45, 0.45),
    "y": (-0.45, 0.45),
    "z": (-0.30, 0.45),
}

# Gripper convention. Default 100 = open, 0 = closed. If your SO101 calibration
# is the opposite (calibrated with the gripper open at the "min" tick instead of
# the "max" tick), override `gripper.open_value` / `gripper.closed_value` in
# config/xlerobot.yaml.
DEFAULT_GRIPPER_OPEN = 100.0
DEFAULT_GRIPPER_CLOSED = 0.0

# Quest face-button mapping per controller side. The lower face button on each
# controller (A on right, X on left) toggles engage for THAT controller's arm.
# Only the right controller has a B button — it's the global recording toggle.
ENGAGE_BUTTON_BY_SIDE: dict[str, str] = {"right": "A", "left": "X"}
RECORD_BUTTON_BY_SIDE: dict[str, str] = {"right": "B"}
DUAL_MODE_BUTTON_BY_SIDE: dict[str, str] = {"left": "Y"}

# Minimum motion magnitude (in metres) required to accept a calibration. Anything
# smaller than this is too noisy to reliably determine "user-forward" direction.
CALIBRATION_MIN_MOTION_M: float = 0.05    # 5 cm
CALIBRATION_TARGET_MOTION_M: float = 0.10  # the wizard says "move ~10 cm"
WRIST_VERIFY_MIN_DEG: float = 8.0
WRIST_VERIFY_TARGET_DEG: float = 20.0
ROBOT_VERIFY_MIN_SAMPLES: int = 6
ROBOT_VERIFY_REQUIRED_LABELS: tuple[str, ...] = ("forward", "back", "left", "right", "up", "down")
ROBOT_VERIFY_MIN_MOTION_M: float = 0.025
ROBOT_VERIFY_PASS_ERROR_CM: float = 3.0
ROBOT_VERIFY_WARN_ERROR_CM: float = 5.0
ROBOT_VERIFY_LIVE_GOOD_ANGLE_DEG: float = 25.0
ROBOT_VERIFY_LIVE_GOOD_RATIO_MIN: float = 0.60
ROBOT_VERIFY_LIVE_GOOD_RATIO_MAX: float = 1.60
ROBOT_VERIFY_TEST_SCALE: float = 0.20
RECORDING_REQUIRED_SIDES: tuple[ArmSide, ArmSide] = ("left", "right")
VERIFIED_TRANSLATION_MIN_SINGULAR: float = 0.05
VERIFIED_TRANSLATION_MAX_SINGULAR: float = 5.0
VERIFIED_TRANSLATION_MAX_COND: float = 8.0
VERIFIED_TRANSLATION_MAX_AXIS_ANGLE_DEG: float = 60.0
QUEST_OPERATOR_FRAME: str = "quest_operator_frame"

# Translation smoothing factor. 1.0 = raw input, 0.0 = frozen.
POS_EMA_ALPHA: float = 0.30
# Input filtering for the Quest controller stream. The native Quest app sends
# per-frame relative_position while grip is held; ignore tiny controller
# translation noise before integrating it into the reset-relative target.
POS_DEADZONE_M: float = 0.001
# Cartesian target rate cap in robot base frame. This mirrors LeRobot's
# EEBoundsAndSafety max_ee_step_m and prevents tracking spikes from entering IK.
MAX_EE_STEP_M: float = 0.004
# Hardware polarity of the wrist motors, per arm. Loaded from
# `config/xlerobot.yaml` (`vr.wrist_motor_polarity`) at startup. This is the
# only invariant wrist sign: it depends on motor mounting/wiring, not on where
# the user stood when calibrating. Runtime wrist deltas are projected in the
# controller-anchor-local frame and multiplied by this polarity directly.
_WRIST_MOTOR_POLARITY: dict[str, dict[str, float]] = {
    "left":  {"flex": -1.0, "roll": 1.0},
    "right": {"flex": -1.0, "roll": 1.0},
}


# Homing: command trajectory is deterministic, but completion must be based on
# measured joint feedback. Otherwise a slow/lagging arm can be reported homed
# while still several ticks away from the saved pose.
HOMING_TOL_DEG: float = 0.5
HOMING_PRESENT_TOL_DEG: float = 4.0
HOMING_SETTLE_TICKS: int = 5
HOMING_TIMEOUT_S: float = 45.0   # hard cap; if not physically settled by then, give up
RECORDING_HOME_TIMEOUT_BUFFER_S: float = 2.0
RECORDING_ANCHOR_INPUT_WAIT_S: float = 2.0
RECORDING_MAX_CONSECUTIVE_CAMERA_SKIPS: int = 3
RECORDING_MAX_CAMERA_SKIP_RATIO: float = 0.02
RECORDING_MIN_SAMPLES_FOR_SKIP_RATIO: int = 150


# --- SO101 analytical kinematics ---------------------------------------------
#
# Inverse kinematics formula is transcribed verbatim from
# `lerobot.model.SO101Robot.SO101Kinematics.inverse_kinematics` (the math used by
# XLerobot keyboard teleop and the upstream VR script). Reproduced here because
# importing that module fails: SO101Robot.py has `from lerobot.robots.so101_follower...`
# which doesn't exist in the current lerobot layout.
#
# Forward kinematics is derived by inverting the IK chain, including the same
# theta1_offset / theta2_offset for the SO101's joint-zero geometry and the final
# 90°-transform. FK ↔ IK round-trip is exact within numerical precision for joint
# angles inside the IK's clamp range; for poses outside the IK envelope the FK
# still returns sensible values but the IK→FK roundtrip won't recover them.

class _SO101Kin:
    """Analytical 2-link kinematics for SO101 with joint-zero offsets baked in."""

    # Constants from lerobot.model.SO101Robot.SO101Kinematics
    THETA1_OFFSET = math.atan2(0.028, 0.11257)                              # ≈ 0.244 rad / 14°
    THETA2_OFFSET = math.atan2(0.0052, 0.1349) + THETA1_OFFSET              # ≈ 0.282 rad / 16°
    # Internal joint pre-clamps. Widened beyond the upstream's [-0.1, 3.45] /
    # [-0.2, π] so the IK can output the full range the SO101's URDF + motor
    # calibration actually allows (shoulder_lift ±100°, elbow_flex ±96.8°).
    # The motor calibration clamp downstream is the actual safety guard.
    JOINT2_PRE_MIN = -0.20    # shoulder_lift output max: 90 - degrees(-0.20) = +101.5°
    JOINT2_PRE_MAX = 3.65     # shoulder_lift output min: 90 - degrees(3.65) = -119.2°
    JOINT3_PRE_MIN = -0.30    # elbow_flex output min: degrees(-0.30) - 90 = -107.2°
    JOINT3_PRE_MAX = 3.55     # elbow_flex output max: degrees(3.55) - 90 = +113.4°

    def __init__(self, l1: float = 0.1159, l2: float = 0.1350):
        self.l1, self.l2 = l1, l2

    def inverse(self, x: float, y: float) -> tuple[float, float]:
        """(x, y) in IK plane (metres, base frame) → (shoulder_lift_deg, elbow_flex_deg).
        Output is in lerobot's degree convention (matches what the motor expects)."""
        l1, l2 = self.l1, self.l2
        # Workspace scaling — if target is beyond reach, scale onto the boundary.
        r = math.hypot(x, y)
        r_max = l1 + l2
        if r > r_max:
            scale_factor = r_max / r
            x *= scale_factor; y *= scale_factor; r = r_max
        r_min = abs(l1 - l2)
        if 0 < r < r_min:
            scale_factor = r_min / r
            x *= scale_factor; y *= scale_factor; r = r_min

        # Law of cosines (note the leading minus — upstream's convention).
        cos_theta2 = -(r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
        cos_theta2 = max(-1.0, min(1.0, cos_theta2))
        theta2 = math.pi - math.acos(cos_theta2)

        beta = math.atan2(y, x)
        gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
        theta1 = beta + gamma

        joint2 = theta1 + self.THETA1_OFFSET
        joint3 = theta2 + self.THETA2_OFFSET
        # Pre-transform clamp (the upstream's safety net for URDF joint limits).
        joint2 = max(self.JOINT2_PRE_MIN, min(self.JOINT2_PRE_MAX, joint2))
        joint3 = max(self.JOINT3_PRE_MIN, min(self.JOINT3_PRE_MAX, joint3))

        # Final coordinate transform to match SO101 motor convention.
        sl_deg = 90 - math.degrees(joint2)
        ef_deg = math.degrees(joint3) - 90
        return sl_deg, ef_deg

    def forward(self, sl_deg: float, ef_deg: float) -> tuple[float, float]:
        """(shoulder_lift_deg, elbow_flex_deg) → (x, y) in IK plane (metres).
        Exact inverse of the IK formula above (assuming joints inside the clamp range)."""
        # Reverse the final coordinate transform.
        joint2 = math.radians(90 - sl_deg)
        joint3 = math.radians(ef_deg + 90)
        theta1 = joint2 - self.THETA1_OFFSET
        theta2 = joint3 - self.THETA2_OFFSET
        gamma = math.atan2(self.l2 * math.sin(theta2),
                           self.l1 + self.l2 * math.cos(theta2))
        beta = theta1 - gamma
        # r² = l1² + l2² + 2·l1·l2·cos(theta2), derived from the IK's cos_theta2 formula.
        r_sq = self.l1 ** 2 + self.l2 ** 2 + 2 * self.l1 * self.l2 * math.cos(theta2)
        r = math.sqrt(max(0.0, r_sq))
        return r * math.cos(beta), r * math.sin(beta)

    @classmethod
    def sl_deg_in_ik_envelope(cls, sl_deg: float) -> bool:
        """True if `sl_deg` is in a region the IK can actually generate (i.e. its FK
        output round-trips through IK back to (sl_deg, _)). Used to warn at RESET
        when the user's arm is sitting outside the IK envelope."""
        joint2 = math.radians(90 - sl_deg)
        return cls.JOINT2_PRE_MIN <= joint2 <= cls.JOINT2_PRE_MAX


# Joint name -> array index for the live 5-DOF arm target. Values are in
# LeRobot/XLeRobot calibrated motor degrees and match `so101_new_calib.urdf`.
_SO101_URDF = REPO_ROOT / "reference" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
_BODY_IK_JOINT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex")
_IK_JOINT_ORDER = _BODY_IK_JOINT_ORDER + ("wrist_flex", "wrist_roll")

# SO101 reach. Used to clamp the running EE target inside the actual workspace.
WORKSPACE_REACH_M = 0.45         # stay inside mechanical reach to avoid IK edge flips


def _load_urdf_kinematics():
    """Construct LeRobot's calibrated SO101 URDF FK/IK solver.

    The local LeRobot examples and `robot_kinematic_processor.py` use
    `so101_new_calib.urdf` with observed motor `.pos` values directly. Keep the
    same convention here so RESET anchors and target IK stay in the per-arm
    SO101 motor frame. The dashboard viewer uses the full XLeRobot URDF only
    for visualization and maps these motor-degree targets into that model.
    """
    try:
        from lerobot.model.kinematics import RobotKinematics
        if not _SO101_URDF.is_file():
            log.warning("calibrated SO101 URDF not found at %s", _SO101_URDF)
            return None
        return RobotKinematics(
            urdf_path=str(_SO101_URDF),
            target_frame_name="wrist_link",
            joint_names=list(_BODY_IK_JOINT_ORDER),
        )
    except Exception as e:
        log.warning("failed to load calibrated SO101 URDF kinematics: %s", e)
        return None


# Default operator-frame → robot base-frame rotation. The native Quest adapter
# converts Unity/OpenXR poses into `quest_operator_frame` first:
#   operator.x = user forward
#   operator.y = user left
#   operator.z = user up
# That matches XLeRobot's calibrated robot convention directly:
#   robot.x = forward, robot.y = left, robot.z = up.
# Grip RESET does not rebuild this frame. It only captures a controller pose
# anchor; calibration and persisted profiles own the translation/orientation
# frame so hand movement stays stable across resets.
import numpy as _np
from scipy.spatial.transform import Rotation as _R

_VR_TO_ROBOT = _np.eye(3, dtype=float)


def _slerp_rotation_matrix(
    previous: _np.ndarray,
    target: _np.ndarray,
    alpha: float,
    max_step_rad: float | None = None,
) -> _np.ndarray:
    """EMA-style smoothing for rotation matrices using shortest-path SLERP."""
    if alpha <= 0.0:
        return previous.copy()
    if alpha >= 1.0 and max_step_rad is None:
        return target.copy()
    from scipy.spatial.transform import Rotation as _R, Slerp as _Slerp

    prev_r = _R.from_matrix(previous)
    target_r = _R.from_matrix(target)
    candidate = _Slerp([0.0, 1.0], _R.concatenate([prev_r, target_r]))([alpha])[0]

    if max_step_rad is not None:
        step_angle = float((candidate * prev_r.inv()).magnitude())
        if step_angle > max_step_rad > 0.0:
            capped_alpha = max_step_rad / step_angle
            candidate = _Slerp([0.0, 1.0], _R.concatenate([prev_r, candidate]))([capped_alpha])[0]
    return candidate.as_matrix()


def _project_to_rotation_matrix(matrix: _np.ndarray) -> _np.ndarray:
    """Project a near-rotation matrix to SO(3), matching BEAVR's safety step."""
    try:
        u, _, vt = _np.linalg.svd(_np.array(matrix, dtype=float))
        rot = u @ vt
        if _np.linalg.det(rot) < 0:
            vt[-1, :] *= -1.0
            rot = u @ vt
        return rot
    except Exception:
        return _np.eye(3)


def _positive_quat_xyzw(quat: _np.ndarray) -> _np.ndarray:
    q = _np.array(quat, dtype=float)
    norm = float(_np.linalg.norm(q))
    if norm < 1e-9:
        return _np.array([0.0, 0.0, 0.0, 1.0])
    q = q / norm
    if q[3] < 0:
        q = -q
    return q


def _normalize_quest_coordinate_frame(frame: Any) -> str:
    if frame == "unity_reachy":
        return QUEST_OPERATOR_FRAME
    return str(frame or "")


def _pose_matrix_from_vr(position: tuple[float, float, float],
                         quat_xyzw: tuple[float, float, float, float]) -> _np.ndarray:
    from scipy.spatial.transform import Rotation as _R

    T = _np.eye(4)
    T[:3, 3] = _np.array(position, dtype=float)
    T[:3, :3] = _R.from_quat(_positive_quat_xyzw(_np.array(quat_xyzw, dtype=float))).as_matrix()
    return T


def _controller_rotation_delta_for_side(side: ArmSide, rotation_delta_vr: _np.ndarray) -> _np.ndarray:
    """Normalize controller rotation handedness before VR->robot mapping.

    Quest controller positions share the same room frame, but the left and right
    controller local orientation frames are mirrored. The right-hand controller
    rotation matches the robot mapping; the left-hand controller rotation must be
    inverted so wrist/tool rotation follows the same user intent.
    """
    rot = _project_to_rotation_matrix(rotation_delta_vr)
    if side == "left":
        return rot.T
    return rot


def _wrist_rotation_deg_since_anchor(
    anchor_q: tuple[float, float, float, float],
    release_q: tuple[float, float, float, float],
) -> float:
    """Total rotation (degrees) from anchor quaternion to release quaternion."""
    R_anchor = _R.from_quat(anchor_q)
    R_release = _R.from_quat(release_q)
    R_rel = R_anchor.inv() * R_release
    return math.degrees(float(_np.linalg.norm(_np.asarray(R_rel.as_rotvec(), dtype=float))))


def _wrist_rotvec_since_anchor(
    anchor_q: tuple[float, float, float, float],
    release_q: tuple[float, float, float, float],
) -> tuple[_np.ndarray, float]:
    """Raw controller anchor-local rotation vector.

    Do not apply the left-controller transpose/handedness correction here.
    Runtime already applies `_controller_rotation_delta_for_side()` before
    converting rotation deltas to rotvecs. Storing the raw empirical controller
    axis avoids double-flipping the left wrist calibration.
    """
    R_anchor = _R.from_quat(anchor_q)
    R_release = _R.from_quat(release_q)
    R_rel = R_anchor.inv() * R_release
    rotvec = _np.asarray(R_rel.as_rotvec(), dtype=float)
    mag = float(_np.linalg.norm(rotvec))
    if mag > 1e-9:
        rotvec = rotvec / mag
    return rotvec, mag


def _canonicalize_empirical_roll_axis(axis_local: _np.ndarray) -> tuple[_np.ndarray, bool]:
    """Normalize a captured roll axis without letting it invert roll direction.

    Main-branch roll control used the Quest reference "roll right" canonical
    of -Z in controller-anchor-local coordinates, then applied the left-hand
    runtime correction separately. The empirical roll step should refine the
    physical axis, not replace that known-good sign convention. If the user
    rolls the wrist in the opposite mathematical direction during capture, keep
    the axis but flip it back to the main-branch roll-right sign.
    """
    vec = _np.asarray(axis_local, dtype=float)
    norm = float(_np.linalg.norm(vec))
    if norm <= 1e-6:
        return _np.array([0.0, 0.0, -1.0], dtype=float), False
    vec = vec / norm
    default_roll_right = _np.array([0.0, 0.0, -1.0], dtype=float)
    if float(_np.dot(vec, default_roll_right)) < 0.0:
        return -vec, True
    return vec, False


def _valid_wrist_axis_tuple(axis: Any) -> Optional[tuple[float, float, float]]:
    try:
        arr = _np.asarray(axis, dtype=float)
    except Exception:
        return None
    if arr.shape != (3,) or not _np.all(_np.isfinite(arr)):
        return None
    if float(_np.linalg.norm(arr)) <= 1e-6:
        return None
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _wrist_axes_ready(arm: Any) -> bool:
    return (
        _valid_wrist_axis_tuple(getattr(arm, "wrist_pitch_canonical", None)) is not None
        and _valid_wrist_axis_tuple(getattr(arm, "wrist_roll_canonical", None)) is not None
    )


def _effective_wrist_axes(
    side: ArmSide,
    pitch_canonical: Optional[tuple[float, float, float]] = None,
    roll_canonical: Optional[tuple[float, float, float]] = None,
) -> tuple[_np.ndarray, _np.ndarray]:
    pitch_axis = _np.asarray(
        pitch_canonical if pitch_canonical is not None else (1.0, 0.0, 0.0),
        dtype=float,
    )
    pitch_norm = float(_np.linalg.norm(pitch_axis))
    if pitch_norm <= 1e-6:
        pitch_axis = _np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        pitch_axis = pitch_axis / pitch_norm
    if side == "left":
        pitch_axis = -pitch_axis

    if roll_canonical is not None:
        roll_axis, _ = _canonicalize_empirical_roll_axis(
            _np.asarray(roll_canonical, dtype=float)
        )
    else:
        roll_axis = _np.array([0.0, 0.0, -1.0], dtype=float)
    if side == "left":
        roll_axis = -roll_axis
    return pitch_axis, roll_axis


def _clamp_to_workspace_reach(position: _np.ndarray) -> _np.ndarray:
    """Keep the requested EE target inside the robot-base reach sphere."""
    radius = float(_np.linalg.norm(position))
    if radius <= WORKSPACE_REACH_M or radius <= 1e-9:
        return position
    return position * (WORKSPACE_REACH_M / radius)


def _clamp_target_position(position: _np.ndarray) -> _np.ndarray:
    """Clamp EE target before IK, including a rear guard for base singularities."""
    target = _np.array(position, dtype=float).copy()
    target[0] = max(EE_BOUNDS["x"][0], min(EE_BOUNDS["x"][1], float(target[0])))
    target[1] = max(EE_BOUNDS["y"][0], min(EE_BOUNDS["y"][1], float(target[1])))
    target[2] = max(EE_BOUNDS["z"][0], min(EE_BOUNDS["z"][1], float(target[2])))
    target[0] = max(SAFE_REAR_X_M, float(target[0]))
    return _clamp_to_workspace_reach(target)


def _apply_live_joint_deadband(
    prefix: str,
    targets: dict[str, float],
    previous: dict[str, float],
) -> dict[str, float]:
    """Hold last live command when the next target only moves by noise."""
    out: dict[str, float] = {}
    for joint_name, target in targets.items():
        joint = joint_name.removeprefix(prefix)
        deadband = max(0.0, float(JOINT_COMMAND_DEADBAND_DEG.get(joint, 0.0)))
        prev = previous.get(joint_name)
        if prev is not None and abs(float(target) - float(prev)) < deadband:
            out[joint_name] = float(prev)
        else:
            out[joint_name] = float(target)
    return out


class _JointCommandFilter:
    """Newest-heavy weighted moving filter for live arm joint commands."""

    def __init__(self) -> None:
        self._history: dict[str, list[float]] = {}

    def reset(self, seed: Optional[dict[str, float]] = None) -> None:
        self._history = {}
        if not seed:
            return
        window = max(1, len(JOINT_COMMAND_FILTER_WEIGHTS))
        for joint_name, value in seed.items():
            self._history[joint_name] = [float(value)] * window

    def apply(self, prefix: str, targets: dict[str, float]) -> dict[str, float]:
        weights = tuple(float(w) for w in JOINT_COMMAND_FILTER_WEIGHTS)
        if len(weights) <= 1:
            return {k: float(v) for k, v in targets.items()}
        total = sum(weights)
        if total <= 0.0 or not all(math.isfinite(w) and w >= 0.0 for w in weights):
            return {k: float(v) for k, v in targets.items()}
        norm_weights = tuple(w / total for w in weights)
        window = len(norm_weights)
        out: dict[str, float] = {}
        for joint_name, target in targets.items():
            joint = joint_name.removeprefix(prefix)
            value = float(target)
            if joint in JOINT_COMMAND_FILTER_BYPASS:
                self._history[joint_name] = [value] * window
                out[joint_name] = value
                continue
            hist = self._history.setdefault(joint_name, [])
            if hist and abs(hist[-1] - value) <= 1e-12:
                out[joint_name] = value if len(hist) < window else sum(
                    sample * weight for sample, weight in zip(reversed(hist[-window:]), norm_weights)
                )
                continue
            hist.append(value)
            if len(hist) > window:
                del hist[:-window]
            if len(hist) < window:
                out[joint_name] = value
            else:
                out[joint_name] = sum(
                    sample * weight for sample, weight in zip(reversed(hist[-window:]), norm_weights)
                )
        return out


def _compute_session_frame_from_three_motions(
    motion_fwd_vr: tuple[float, float, float],
    motion_up_vr: tuple[float, float, float],
    motion_left_vr: tuple[float, float, float],
) -> tuple[_np.ndarray, str]:
    """Build the session VR→robot rotation matrix from all three USER-MOTION
    vectors via constrained least-squares (Kabsch / Procrustes).

    The user moved their hand FORWARD, UP, and to-their-LEFT (in their body
    frame). We want a rotation matrix M ∈ SO(3) such that:

        M @ motion_fwd_vr  ≈ +robot_x  (forward away from arm)
        M @ motion_up_vr   ≈ +robot_z  (vertical up)
        M @ motion_left_vr ≈ +robot_y  (robot's left)

    Three motion vectors give 9 equations for SO(3)'s 3 DoFs — overdetermined.
    The Kabsch solution finds the closest rotation matrix in least-squares
    sense, averaging out any single-motion noise (off-axis drift, jitter in a
    hand-held motion). Much more robust than the 2-motion Gram-Schmidt path,
    which can be perturbed by any small noise on the up-motion vector.

    This is the final calibration solve. It deliberately does not fall back to
    a weaker two-motion/yaw-only matrix: if the three motions are degenerate,
    too parallel, or mutually inconsistent, the caller must ask the user to
    recapture the calibration.
    """
    f = _np.array(motion_fwd_vr,  dtype=float)
    u = _np.array(motion_up_vr,   dtype=float)
    l = _np.array(motion_left_vr, dtype=float)
    fn = float(_np.linalg.norm(f))
    un = float(_np.linalg.norm(u))
    ln = float(_np.linalg.norm(l))
    if fn < 1e-3 or un < 1e-3 or ln < 1e-3:
        raise ValueError(
            "3-motion calibration has a degenerate vector "
            f"(|fwd|={fn:.3f}, |up|={un:.3f}, |left|={ln:.3f}); recapture all three motions"
        )

    f_hat = f / fn
    u_hat = u / un
    l_hat = l / ln

    # Pairwise separation check (degrees apart).
    cos_fu = abs(float(_np.dot(f_hat, u_hat)))
    cos_fl = abs(float(_np.dot(f_hat, l_hat)))
    cos_ul = abs(float(_np.dot(u_hat, l_hat)))
    cos_max = max(cos_fu, cos_fl, cos_ul)
    if cos_max > 0.6:
        raise ValueError(
            "3-motion calibration motions are too parallel "
            f"(max cos={cos_max:.2f}, about {math.degrees(math.acos(min(1.0, cos_max))):.1f} deg apart); "
            "recapture with clearly orthogonal forward/up/left motions"
        )

    # Procrustes: minimize ||M A - B||_F over M ∈ SO(3).
    #   A's columns are the normalized VR motions.
    #   B's columns are the target robot-frame unit axes.
    A = _np.stack([f_hat, u_hat, l_hat], axis=1)
    B = _np.stack([
        _np.array([1.0, 0.0, 0.0]),  # +x_robot ← fwd
        _np.array([0.0, 0.0, 1.0]),  # +z_robot ← up
        _np.array([0.0, 1.0, 0.0]),  # +y_robot ← left
    ], axis=1)

    H = B @ A.T
    U_, _S, Vt = _np.linalg.svd(H)
    D = _np.eye(3)
    if _np.linalg.det(U_ @ Vt) < 0:
        D[-1, -1] = -1.0
    M = U_ @ D @ Vt

    # Residual fit error: how far is M from a perfect mapping of all three?
    residual = float(_np.linalg.norm(M @ A - B, ord="fro"))
    if residual > 0.5:
        raise ValueError(
            f"3-motion calibration residual is too large ({residual:.3f}); "
            "forward/up/left motions disagree, recapture calibration"
        )
    log.info(
        "3-motion calibration: residual fit error %.3f (max cos %.2f); confidence good",
        residual, cos_max,
    )

    return M, "good"


# --- VR data snapshots --------------------------------------------------------

@dataclass
class _LatestGoal:
    """Snapshot of the most recent VR goal — used for status display, the
    calibration wizard (which reads per-frame `rel_position`/`rel_rotvec`), and
    the SE(3) controller-pose mapping (which reads `controller_position` +
    `rotation_quat`).

    `buttons` carries the Quest face-button pressed-state, keyed by Meta's labels:
        right controller → {"A": bool, "B": bool}
        left  controller → {"X": bool, "Y": bool}
    """
    received_at: float = 0.0
    has_data: bool = False
    mode: str = "idle"            # "idle" | "position" | "reset"
    rel_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rel_rotvec: tuple[float, float, float] = (0.0, 0.0, 0.0)
    controller_position: Optional[tuple[float, float, float]] = None
    rotation_quat: Optional[tuple[float, float, float, float]] = None
    trigger: bool = False
    thumbstick: tuple[float, float] = (0.0, 0.0)
    buttons: dict[str, bool] = field(default_factory=dict)


@dataclass
class _AnchorPose:
    """Robot + controller state snapshotted at the most recent RESET.

    Per-tick wrist targets are derived as (anchor_wrist + rotation_delta) where
    rotation_delta is the absolute current-vs-anchor controller quaternion.
    Position uses the absolute controller pose carried in every goal too: the
    VR-world displacement from the controller anchor maps through
    `session_vr_to_robot` directly to the robot-frame cartesian offset, with no
    per-frame delta integration."""
    ee_x: float = 0.0
    ee_y: float = 0.0
    pan_deg: float = 0.0
    shoulder_lift_deg: float = 0.0
    elbow_flex_deg: float = 0.0
    wrist_flex_deg: float = 0.0
    wrist_roll_deg: float = 0.0
    gripper_pct: float = 50.0
    captured: bool = False
    # Controller orientation in VR world frame at RESET (quaternion x,y,z,w).
    # Used to compute absolute wrist mapping with zero drift across the session.
    ctrl_quat: Optional[tuple[float, float, float, float]] = None


@dataclass
class _LiveTargets:
    """Current commanded joint targets, in degrees. UI reads this."""
    shoulder_pan: float = 0.0
    shoulder_lift: float = 0.0
    elbow_flex: float = 0.0
    wrist_flex: float = 0.0
    wrist_roll: float = 0.0
    gripper: float = 100.0   # open by default; trigger held → closes

    def to_dict_with_prefix(self, side: ArmSide) -> dict[str, float]:
        prefix = f"{side}_arm_"
        return {
            f"{prefix}shoulder_pan":  self.shoulder_pan,
            f"{prefix}shoulder_lift": self.shoulder_lift,
            f"{prefix}elbow_flex":    self.elbow_flex,
            f"{prefix}wrist_flex":    self.wrist_flex,
            f"{prefix}wrist_roll":    self.wrist_roll,
            f"{prefix}gripper":       self.gripper,
        }


@dataclass(frozen=True)
class _DirectWristTargets:
    """Wrist command derived only from reset-relative controller rotation."""

    wrist_flex: float
    wrist_roll: float
    wrist_flex_delta_deg: float
    wrist_roll_delta_deg: float
    pitch_axis: _np.ndarray
    roll_axis: _np.ndarray
    polarity: dict[str, float]


# --- the session ---------------------------------------------------------------

@dataclass
class _PerArm:
    """All per-arm runtime state for VR teleop. One instance per side, always
    created — populated when the arm is connected/calibrated, otherwise idle.

    Lives inside `VRTeleopSession._arms`. The drive loop iterates over connected
    arms and only acts on the one that is `_active_arm`.
    """
    side: ArmSide
    calibrated: bool = False
    # Clamped IK target (4x4 homogeneous) passed to URDF IK each tick.
    # Derived as `anchor_ee_pos + offset_robot`, clamped to EE_BOUNDS + workspace
    # radius. NOT the integrator; see `offset_robot` below.
    target_T: _np.ndarray = field(default_factory=lambda: _np.eye(4))
    # Anchor EE position in robot base frame, captured at RESET via URDF FK when available.
    anchor_ee_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Cumulative offset from `anchor_ee_pos` in robot base frame, integrated from
    # VR position deltas and reconciled to the reachable target each tick. Do not
    # allow hidden offset to grow past workspace limits; that stored "debt" can
    # release later as a sudden jump.
    offset_robot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Latest reset-relative VR-controller displacement used by live control.
    # Packet-relative deltas are still queued/cleared for calibration and
    # diagnostics, but live teleop derives motion from absolute controller pose
    # relative to `controller_anchor_T` so packet cadence cannot create drift.
    vr_offset_accum: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pending_rel_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    anchor: _AnchorPose = field(default_factory=_AnchorPose)
    # Incremented whenever robot pose can change outside live VR control. A
    # teleop anchor is fresh only when captured at the current generation.
    pose_generation: int = 0
    anchor_generation: int = -1
    anchor_invalid_reason: str = "not anchored"
    targets: _LiveTargets = field(default_factory=_LiveTargets)
    last_sent_targets: dict[str, float] = field(default_factory=dict)
    # VR->robot orientation matrix from the calibration wizard. Grip RESET keeps
    # this frame stable and only captures a controller pose anchor.
    session_vr_to_robot: _np.ndarray = field(default_factory=lambda: _VR_TO_ROBOT.copy())
    latest: _LatestGoal = field(default_factory=_LatestGoal)
    reset_pending: bool = False
    # Last-tick button state for edge detection (A/B/X/Y face buttons).
    prev_buttons: dict[str, bool] = field(default_factory=dict)
    # Guided-calibration wizard state. See `_advance_calibration`:
    #   idle → awaiting_anchor_fwd → motioning_fwd → awaiting_anchor_up →
    #   motioning_up -> awaiting_anchor_left -> motioning_left -> wrist checks -> idle
    cal_state: str = "idle"
    cal_motion_acc: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cal_captured_fwd:  Optional[tuple[float, float, float]] = None
    cal_captured_up:   Optional[tuple[float, float, float]] = None
    cal_captured_left: Optional[tuple[float, float, float]] = None
    # Wrist-verify step 4: controller quaternion captured at grip-press, used
    # to compute the delta rotation at grip-release.
    cal_anchor_quat_for_wrist: Optional[tuple[float, float, float, float]] = None
    # Latest release quaternion latched from the last POSITION goal while in
    # motioning_wrist_verify; IDLE goals do not include rotation.
    cal_wrist_release_quat: Optional[tuple[float, float, float, float]] = None
    # Live wrist rotation since the step-4 anchor (degrees) — for wizard UI.
    cal_wrist_verify_deg: float = 0.0
    cal_wrist_pitch_verify_deg: float = 0.0
    cal_wrist_roll_verify_deg: float = 0.0
    # Per-arm raw controller-anchor-local rotvec (unit vector) the user's wrist
    # rotates around when pitching UP. Captured by the wrist-verify wizard step;
    # persisted in `vr_calibration.yaml` as `wrist_pitch_anchor_local`. None
    # means wrist calibration is incomplete and live teleop must not drive.
    wrist_pitch_canonical: Optional[tuple[float, float, float]] = None
    # Empirical anchor-local rotvec for user's roll-right motion. None means
    # wrist calibration is incomplete and live teleop must not drive.
    wrist_roll_canonical: Optional[tuple[float, float, float]] = None
    # Last completion-time motion magnitudes (m) — for UI to show "calibrated to N cm"
    cal_last_fwd_m:  float = 0.0
    cal_last_up_m:   float = 0.0
    cal_last_left_m: float = 0.0
    cal_validation: dict[str, Any] = field(default_factory=dict)
    # Optional robot-verified refinement layered on top of the VR-only
    # direction calibration. `session_vr_to_robot` remains the orientation/wrist
    # frame; `translation_vr_to_robot` is the optional full position map learned
    # from six robot-verification samples.
    base_vr_direction_matrix: Optional[_np.ndarray] = None
    translation_vr_to_robot: Optional[_np.ndarray] = None
    translation_scale: float = 1.0
    robot_verify_state: str = "idle"
    robot_verify_samples: list[dict[str, Any]] = field(default_factory=list)
    robot_verify_robot_start: Optional[tuple[float, float, float]] = None
    robot_verify_robot_end: Optional[tuple[float, float, float]] = None
    robot_verify_vr_start: Optional[tuple[float, float, float]] = None
    robot_verify_label: str = ""
    robot_verify_fit_error_cm: Optional[float] = None
    robot_verify_sample_residuals: list[dict[str, Any]] = field(default_factory=list)
    robot_verify_quality: str = "unverified"
    robot_verified_at: Optional[str] = None
    robot_verify_test_active: bool = False
    robot_verify_test_completed: bool = False
    robot_verify_test_scale: float = ROBOT_VERIFY_TEST_SCALE
    # Homing state. While True, the drive loop drives this arm toward
    # `home_target` (a per-joint absolute target, in degrees), instead of the
    # VR-driven target. Cleared after measured joint feedback settles at home.
    homing: bool = False
    home_target: dict[str, float] = field(default_factory=dict)
    home_start_t: float = 0.0   # monotonic seconds when homing began (timeout safety)
    home_controller: JointHomingController | None = None
    home_last_command_error_deg: float = 0.0
    home_last_present_error_deg: float = 0.0
    home_last_worst_joint: str = ""
    home_next_progress_log_t: float = 0.0
    # User-facing knob: when True, mirror the LATERAL translation axis
    # (left/right). Wrist signs are direct controller-rotation mapping and are
    # controlled only by empirical wrist axes plus vr.wrist_motor_polarity.
    # Read from config/xlerobot.yaml's `vr:` block per arm.
    invert_lateral: bool = False
    # When True, the YAML setting is EXPLICITLY set by the user (override mode):
    # the calibration wizard's auto-detection at step 3 must not touch
    # `invert_lateral`. Lets users with physically mirror-mounted motors keep
    # their fix in place across recalibrations.
    invert_lateral_override: bool = False
    # Required calibrated URDF kinematics adapter plus last-good IK solutions.
    # Body IK is solved only over shoulder_pan/shoulder_lift/elbow_flex, seeded
    # from `last_body_q_sol`; wrist targets are direct VR commands and must not
    # feed the IK seed.
    kinematics: Any = None
    last_body_q_sol: _np.ndarray = field(default_factory=lambda: _np.zeros(3, dtype=float))
    # Full 5-joint command state kept for status, filtering, and compatibility.
    last_q_sol: _np.ndarray = field(default_factory=lambda: _np.zeros(5, dtype=float))
    # Legacy diagnostic field kept for API/test compatibility. Production live
    # teleop refuses to use analytical IK when calibrated URDF is unavailable.
    using_analytical_fallback: bool = False
    # Per-arm filtered target state. The cartesian offset is LERP-smoothed in
    # `_compute_targets_from_vr`. Wrist targets bypass body IK and final command
    # filtering; the final live command still goes through per-joint caps and a
    # small deadband before send_action.
    last_q_filtered: Optional[_np.ndarray] = None
    command_filter: _JointCommandFilter = field(default_factory=_JointCommandFilter)
    # Anchor orientation matrix (3×3) captured at RESET. Body IK ignores target
    # orientation; this is kept for FK/status context only.
    anchor_R_robot: _np.ndarray = field(default_factory=lambda: _np.eye(3))
    controller_anchor_T: Optional[_np.ndarray] = None
    # Rotation from controller-anchor local axes to EE-anchor local axes,
    # captured on grip RESET. Kept only as a diagnostic signal; production wrist
    # commands are direct reset-relative controller rotations.
    vr_ctrl_to_ee: Optional[Any] = None
    robot_anchor_T: _np.ndarray = field(default_factory=lambda: _np.eye(4))
    # Calibration confidence: "good" if the wizard's captured motion vectors
    # were well-separated, "poor" if too parallel (and the matrix is shaky).
    cal_confidence: str = "good"
    stale_since: Optional[float] = None
    last_commanded_targets: dict[str, float] = field(default_factory=dict)
    last_diag: dict[str, Any] = field(default_factory=dict)
    robot_verify_vr_delta_accum: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quality_ticks: int = 0
    quality_ik_rejects: int = 0
    quality_offset_speed_ema_mps: float = 0.0
    quality_last_offset_step_m: float = 0.0


class _ArmMovementMapper:
    """Calibrated VR-controller translation mapper used identically by both arms."""

    @staticmethod
    def effective_translation_matrix(arm: _PerArm, matrix: _np.ndarray) -> _np.ndarray:
        M = _np.asarray(matrix, dtype=float)
        if arm.invert_lateral:
            return _np.diag([1.0, -1.0, 1.0]) @ M
        return M

    @classmethod
    def usable_verified_translation_matrix(cls, arm: _PerArm) -> _np.ndarray | None:
        """Return the robot-verified translation matrix when it is safe to drive.

        The six-direction verification solve is allowed to learn cross-axis
        coupling and per-axis scale. Reject only mathematically collapsed or
        wildly inconsistent fits; those must not silently delete a live axis.
        """
        if arm.robot_verify_quality != "good" or arm.translation_vr_to_robot is None:
            return None
        M = _np.asarray(arm.translation_vr_to_robot, dtype=float)
        if M.shape != (3, 3) or not _np.all(_np.isfinite(M)):
            return None
        try:
            singular = _np.linalg.svd(M, compute_uv=False)
        except Exception:
            return None
        if singular.shape != (3,) or not _np.all(_np.isfinite(singular)):
            return None
        s_min = float(_np.min(singular))
        s_max = float(_np.max(singular))
        if (
            s_min < VERIFIED_TRANSLATION_MIN_SINGULAR
            or s_max > VERIFIED_TRANSLATION_MAX_SINGULAR
            or (s_max / s_min) > VERIFIED_TRANSLATION_MAX_COND
        ):
            return None

        reference = arm.translation_scale * cls.effective_translation_matrix(
            arm,
            arm.session_vr_to_robot,
        )
        for axis in _np.eye(3):
            learned = M @ axis
            expected = reference @ axis
            learned_norm = float(_np.linalg.norm(learned))
            expected_norm = float(_np.linalg.norm(expected))
            if learned_norm <= 1e-9 or expected_norm <= 1e-9:
                return None
            cos = float(_np.dot(learned, expected) / (learned_norm * expected_norm))
            cos = max(-1.0, min(1.0, cos))
            angle_deg = math.degrees(math.acos(cos))
            if angle_deg > VERIFIED_TRANSLATION_MAX_AXIS_ANGLE_DEG:
                return None
        return M

    @classmethod
    def runtime_translation_matrix(cls, arm: _PerArm) -> _np.ndarray:
        """Effective VR-delta to robot-delta matrix for live translation."""
        verified = cls.usable_verified_translation_matrix(arm)
        if verified is not None:
            return verified
        scale = arm.translation_scale if arm.robot_verify_quality == "good" else 1.0
        return scale * cls.effective_translation_matrix(
            arm,
            arm.session_vr_to_robot,
        )

    @classmethod
    def runtime_translation_source(cls, arm: _PerArm) -> str:
        if cls.usable_verified_translation_matrix(arm) is not None:
            return "robot_verified_3d"
        if arm.robot_verify_quality == "good":
            return "stage1_scaled_verified_matrix_invalid"
        return "stage1_unverified"


class VRTeleopSession:
    def __init__(self):
        self._lock = threading.RLock()

        # Per-arm state — always created for both sides; populated when connected.
        self._arms: dict[ArmSide, _PerArm] = {
            "left":  _PerArm(side="left"),
            "right": _PerArm(side="right"),
        }
        # The arm that VR is currently driving in single-arm mode. In dual mode,
        # both connected arms are driven and this remains the preferred selected
        # arm when dual mode is turned off.
        self._active_arm: Optional[ArmSide] = None
        self._dual_mode: bool = False

        # VR pipeline (process-global, persists across motor reconnects).
        # The Quest app sends controller poses in headset-yaw-relative Unity
        # coordinates. Convert them to our operator frame
        # (x=Unity z, y=-Unity x, z=Unity y) before calibration/verification so
        # "move hand forward" previews as robot-forward instead of sideways.
        self._native_quest = NativeQuestAdapter(coordinate_frame=QUEST_OPERATOR_FRAME)
        self._native_quest_clients = 0
        self._native_quest_last_seen: float | None = None
        self._native_quest_pairing_token = (
            os.environ.get("XLE_QUEST_PAIRING_TOKEN") or secrets.token_urlsafe(18)
        )

        # Drive loop (process-global).
        self._drive_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        # Global teleop state
        self._engaged = False
        # Standalone calibration scripts use the VR stream as a pose sensor, not
        # as a controller. While disabled, Quest face buttons and grip/reset
        # anchors are ignored so B/A/X/Y cannot start recording or engage teleop.
        self._controller_buttons_enabled = True
        self._teleop_reset_anchors_enabled = True
        # 0.5 = half hand-to-EE mapping (30 cm/s peak). Conservative default
        # because the SO-101's ~45 cm reach is small and 1:1 mapping can feel
        # fast/jerky. Slider goes 0.1..1.0; users bump up for fast tasks.
        self._scale = 0.5
        self._scale_before_robot_verify_test: Optional[float] = None
        self._gripper_open = DEFAULT_GRIPPER_OPEN
        self._gripper_closed = DEFAULT_GRIPPER_CLOSED
        self._last_drive_tick: float = 0.0
        self._last_error: Optional[str] = None
        self._recording_notice: str = ""
        # Recording state. B button on right controller OR UI toggle flip this.
        # `_recorder` is lazily created on first start (so a session that never
        # records pays no dataset cost).
        self._recording: bool = False
        # Compatibility/status bit for an older "armed until grip reset" flow.
        # Normal recording start now refreshes anchors immediately from the
        # latest Quest controller poses and either opens the episode or reports
        # a hard blocker. No production path should leave this true.
        self._recording_armed: bool = False
        self._recording_pending_task: str = ""
        self._recording_pending_root: Optional[str] = None
        self._recorder: Optional[_dataset.DatasetRecorder] = None
        self._recording_transition_lock = threading.Lock()
        self._recording_transition_active: bool = False
        self._recording_transition_started_at: float = 0.0
        self._recording_transition_source: str = ""
        self._recording_transition_target: Optional[bool] = None
        self._recording_button_thread: Optional[threading.Thread] = None
        self._episodes_saved: int = 0
        self._last_saved_episode_index: Optional[int] = None
        self._last_saved_episode_frames: int = 0
        self._recording_camera_frame_skips: int = 0
        self._recording_consecutive_camera_frame_skips: int = 0
        self._recording_last_camera_skip_reason: str = ""
        # Last task string synced from the UI. Cached here so the Quest B button
        # can start an episode with the task the user typed on the web page.
        # Empty text clears the cache and recording start is rejected.
        self._last_task: str = ""
        # Resolved (absolute, ~-expanded) dataset storage root from most recent
        # recorder init. Shown on the UI's Recording card.
        self._last_dataset_root: str = ""
        self._recording_repo_id: Optional[str] = None

        # Kinematics: live teleop requires calibrated SO101 URDF FK/IK.
        # `_analytical_kin` remains only for isolated math tests/helpers.
        self._analytical_kin = _SO101Kin()

        # Restore previously-saved VR calibrations from config/vr_calibration.yaml
        # so the user doesn't have to re-run the wizard every session. New
        # calibrations overwrite the file via `_finalize_calibration`.
        self._load_persisted_calibrations()
        # Populate UI recording counters from existing dataset (if present)
        # before the first "Start recording" click.
        self._bootstrap_recording_status_from_disk()

    def _bootstrap_recording_status_from_disk(self) -> None:
        """Load existing dataset episode summary for UI status on startup."""
        try:
            cfg = _dataset.load_dataset_config()
            repo_id = str(cfg["repo_id"])
            configured_root = cfg.get("root")
            resolved_root = _dataset.resolve_root(configured_root, repo_id)
            idx, frames = _dataset.last_episode_summary(repo_id=repo_id, root=configured_root)
            self._recording_repo_id = repo_id
            self._last_dataset_root = resolved_root
            self._last_saved_episode_index = idx
            self._last_saved_episode_frames = frames
            self._episodes_saved = (idx + 1) if idx is not None else 0
        except Exception as e:
            # Keep status operational even if dataset metadata is absent/corrupt.
            log.info("recording bootstrap skipped: %s", e)

    def _load_persisted_calibrations(self) -> None:
        """Restore per-arm session_vr_to_robot from disk. Silent no-op if no
        file exists or the file is malformed. Also reads per-arm invert_lateral
        flags from config/xlerobot.yaml's `vr:` section, both the value AND
        whether it's explicitly set (override mode). Also reads the global
        translation smoothing/rate-limit factors, plus the hardware
        `vr.wrist_motor_polarity` block (per-arm motor polarity applied
        directly to wrist pitch/roll deltas)."""
        import yaml
        global KP, HOMING_PRESENT_TOL_DEG
        global POS_EMA_ALPHA, POS_DEADZONE_M, MAX_EE_STEP_M
        global JOINT_COMMAND_FILTER_WEIGHTS
        try:
            cfg = yaml.safe_load((REPO_ROOT / "config" / "xlerobot.yaml").read_text()) or {}
            vr_section = cfg.get("vr") or {}
            def _float_key(name: str, default: float, lo: float, hi: float) -> float:
                value = vr_section.get(name)
                if value is None:
                    return default
                return max(lo, min(hi, float(value)))

            KP = _float_key("kp", KP, 0.0, 1.0)
            HOMING_PRESENT_TOL_DEG = _float_key(
                "homing_present_tolerance_deg",
                HOMING_PRESENT_TOL_DEG,
                0.5,
                8.0,
            )
            POS_EMA_ALPHA = _float_key("pos_ema_alpha", POS_EMA_ALPHA, 0.0, 1.0)
            POS_DEADZONE_M = _float_key("pos_deadzone_m", POS_DEADZONE_M, 0.0, 0.02)
            MAX_EE_STEP_M = _float_key("max_ee_step_m", MAX_EE_STEP_M, 0.001, 0.05)
            joint_caps = vr_section.get("joint_deg_caps") or {}
            if isinstance(joint_caps, dict):
                for joint, cap in joint_caps.items():
                    if joint in PER_TICK_DEG_CAPS:
                        PER_TICK_DEG_CAPS[joint] = max(0.1, min(30.0, float(cap)))
            joint_deadbands = vr_section.get("joint_command_deadband_deg") or {}
            if isinstance(joint_deadbands, (int, float)):
                value = max(0.0, min(5.0, float(joint_deadbands)))
                for joint in JOINT_COMMAND_DEADBAND_DEG:
                    if joint != "gripper":
                        JOINT_COMMAND_DEADBAND_DEG[joint] = value
            elif isinstance(joint_deadbands, dict):
                for joint, deadband in joint_deadbands.items():
                    if joint in JOINT_COMMAND_DEADBAND_DEG:
                        JOINT_COMMAND_DEADBAND_DEG[joint] = max(0.0, min(5.0, float(deadband)))
            filter_weights = vr_section.get("joint_command_filter_weights")
            if filter_weights is not None:
                try:
                    parsed_weights = tuple(float(w) for w in filter_weights)
                    weight_sum = sum(parsed_weights)
                    if (
                        len(parsed_weights) < 1
                        or len(parsed_weights) > 8
                        or weight_sum <= 0.0
                        or any((not math.isfinite(w) or w < 0.0) for w in parsed_weights)
                    ):
                        raise ValueError("weights must be 1..8 non-negative finite values with positive sum")
                    JOINT_COMMAND_FILTER_WEIGHTS = tuple(w / weight_sum for w in parsed_weights)
                except Exception as e:
                    log.warning(
                        "vr.joint_command_filter_weights invalid (%r): %s; keeping %s",
                        filter_weights,
                        e,
                        JOINT_COMMAND_FILTER_WEIGHTS,
                    )
            # Per-arm hardware motor polarity. Anything missing falls back to
            # the defaults in _WRIST_MOTOR_POLARITY.
            polarity_block = vr_section.get("wrist_motor_polarity") or {}
            if isinstance(polarity_block, dict):
                for side in ("left", "right"):
                    arm_block = polarity_block.get(side) or {}
                    if not isinstance(arm_block, dict):
                        continue
                    for axis in ("flex", "roll"):
                        if axis not in arm_block:
                            continue
                        try:
                            raw = float(arm_block[axis])
                        except (TypeError, ValueError):
                            log.warning(
                                "vr.wrist_motor_polarity.%s.%s is not numeric (%r); "
                                "keeping default %+.0f",
                                side, axis, arm_block[axis],
                                _WRIST_MOTOR_POLARITY[side][axis],
                            )
                            continue
                        _WRIST_MOTOR_POLARITY[side][axis] = 1.0 if raw >= 0 else -1.0
            log.info(
                "VR control loaded: homing_kp=%.2f pos_ema=%.2f "
                "pos_deadzone=%.1fmm max_ee_step=%.1fmm "
                "homing_feedback_tol=%.1f° joint_filter=%s "
                "wrist_motor_polarity(left)=(flex %+.0f, roll %+.0f) "
                "wrist_motor_polarity(right)=(flex %+.0f, roll %+.0f)",
                KP, POS_EMA_ALPHA,
                POS_DEADZONE_M * 1000.0,
                MAX_EE_STEP_M * 1000.0,
                HOMING_PRESENT_TOL_DEG,
                tuple(round(w, 3) for w in JOINT_COMMAND_FILTER_WEIGHTS),
                _WRIST_MOTOR_POLARITY["left"]["flex"], _WRIST_MOTOR_POLARITY["left"]["roll"],
                _WRIST_MOTOR_POLARITY["right"]["flex"], _WRIST_MOTOR_POLARITY["right"]["roll"],
            )
        except Exception as e:
            log.warning("could not read VR smoothing config from YAML: %s", e)
        for side in ("left", "right"):
            self._restore_persisted_arm_config(side)

    def _restore_persisted_arm_config(self, side: ArmSide) -> None:
        """Restore saved calibration and lateral mapping for a freshly reset arm.

        Wrist motor polarity is hardware configuration and is loaded from
        `config/xlerobot.yaml`; empirical wrist axes are restored from the
        calibration profile if they were captured.
        """
        invert_flags = _vrcal.read_invert_lateral_flags()
        overrides = _vrcal.read_invert_lateral_overrides()
        arm = self._arms[side]
        arm.invert_lateral_override = overrides.get(side, False)
        data = _vrcal.read_for_arm(side) or {}
        if data and data.get("teleop_source") != "native_quest":
            log.warning(
                "[%s] ignoring stale VR calibration profile from teleop_source=%r",
                side,
                data.get("teleop_source", "legacy"),
            )
            data = {}
        expected_frame = _normalize_quest_coordinate_frame(self._native_quest.coordinate_frame)
        if data and _normalize_quest_coordinate_frame(data.get("coordinate_frame")) != expected_frame:
            log.warning(
                "[%s] ignoring VR calibration profile from coordinate_frame=%r; expected %r",
                side,
                data.get("coordinate_frame", "legacy"),
                expected_frame,
            )
            data = {}
        arm.session_vr_to_robot = _VR_TO_ROBOT.copy()
        arm.base_vr_direction_matrix = None
        arm.translation_vr_to_robot = None
        arm.translation_scale = 1.0
        arm.robot_verify_samples = []
        arm.robot_verify_fit_error_cm = None
        arm.robot_verify_quality = "unverified"
        arm.robot_verified_at = None
        arm.robot_verify_sample_residuals = []
        arm.robot_verify_test_completed = False
        arm.wrist_pitch_canonical = None
        arm.wrist_roll_canonical = None
        if arm.invert_lateral_override:
            arm.invert_lateral = invert_flags.get(side, False)
        else:
            arm.invert_lateral = bool(data.get("invert_lateral", invert_flags.get(side, False)))
        arm.cal_confidence = str(data.get("confidence", "unknown") or "unknown")
        arm.cal_last_fwd_m = float(data.get("forward_motion_m", 0.0))
        arm.cal_last_up_m = float(data.get("up_motion_m", 0.0))
        arm.cal_last_left_m = float(data.get("left_motion_m", 0.0))
        robot_data = _vrcal.robot_verification_entry(data)
        if robot_data and _normalize_quest_coordinate_frame(robot_data.get("coordinate_frame")) != expected_frame:
            log.warning(
                "[%s] ignoring robot verification from coordinate_frame=%r; expected %r",
                side,
                robot_data.get("coordinate_frame", "legacy"),
                expected_frame,
            )
            robot_data = {}
        arm.translation_scale = _vrcal.translation_scale_for_arm(side)
        arm.robot_verify_samples = list(robot_data.get("robot_verified_samples") or [])
        fit_error = robot_data.get("fit_error_cm")
        try:
            arm.robot_verify_fit_error_cm = float(fit_error) if fit_error is not None else None
        except (TypeError, ValueError):
            arm.robot_verify_fit_error_cm = None
        arm.robot_verify_quality = str(robot_data.get("calibration_quality") or (
            "good" if robot_data.get("calibration_mode") == "robot_verified" else "unverified"
        ))
        arm.robot_verified_at = robot_data.get("verified_at")
        arm.robot_verify_sample_residuals = list(robot_data.get("robot_verified_sample_residuals") or [])
        robot_verification_good = arm.robot_verify_quality == "good"
        if not robot_verification_good:
            arm.translation_scale = 1.0
        arm.robot_verify_test_completed = (
            robot_verification_good
            and bool(robot_data.get("low_scale_test_completed", False))
        )
        base_raw = robot_data.get("base_vr_direction_matrix")
        if base_raw is not None:
            try:
                base = _np.array(base_raw, dtype=float)
                if base.shape == (3, 3) and _np.all(_np.isfinite(base)):
                    arm.base_vr_direction_matrix = _project_to_rotation_matrix(base)
            except Exception:
                arm.base_vr_direction_matrix = None
        translation_raw = robot_data.get("translation_vr_to_robot_matrix")
        if robot_verification_good and translation_raw is not None:
            try:
                translation = _np.array(translation_raw, dtype=float)
                if translation.shape == (3, 3) and _np.all(_np.isfinite(translation)):
                    arm.translation_vr_to_robot = translation
            except Exception:
                arm.translation_vr_to_robot = None
        # Empirical wrist pitch/roll canonical vectors are stored as raw
        # controller-anchor-local rotvecs. Older left-arm files omitted the
        # frame marker and are converted below for backward compatibility.
        arm.wrist_pitch_canonical = _valid_wrist_axis_tuple(data.get("wrist_pitch_anchor_local"))
        arm.wrist_roll_canonical = _valid_wrist_axis_tuple(data.get("wrist_roll_anchor_local"))
        if side == "left" and data.get("wrist_canonical_frame") != "raw_controller_anchor_local":
            if arm.wrist_pitch_canonical is not None:
                arm.wrist_pitch_canonical = tuple(float(-v) for v in arm.wrist_pitch_canonical)
            if arm.wrist_roll_canonical is not None:
                arm.wrist_roll_canonical = tuple(float(-v) for v in arm.wrist_roll_canonical)
        if not data:
            return
        saved_M = _vrcal.matrix_for_arm(side)
        if saved_M is None:
            return

        # Robot verification learns a full linear translation map. That matrix
        # can include scale/shear/workspace compensation and its nearest
        # rotation is not necessarily a good controller-orientation frame.
        # Keep wrist/orientation on the stage-1 VR direction matrix; use the
        # verified matrix only for position via `translation_vr_to_robot`.
        using_base_frame = (
            robot_data.get("calibration_mode") == "robot_verified"
            and arm.base_vr_direction_matrix is not None
        )
        if using_base_frame:
            arm.session_vr_to_robot = arm.base_vr_direction_matrix.copy()
        else:
            arm.session_vr_to_robot = saved_M
        pitch_label = "empirical" if arm.wrist_pitch_canonical is not None else "missing"
        roll_label = "empirical" if arm.wrist_roll_canonical is not None else "missing"
        polarity = _WRIST_MOTOR_POLARITY.get(side, {"flex": -1.0, "roll": -1.0})
        frame_label = (
            "base_vr_direction"
            if using_base_frame
            else "session"
        )
        log.info(
            "[%s] restored saved VR calibration (invert_lateral=%s, override=%s, "
            "confidence=%s, wrist_motor_polarity=(flex %+.0f, roll %+.0f), "
            "wrist canonical=(pitch %s, roll %s), "
            "orientation_frame=%s)",
            side, arm.invert_lateral, arm.invert_lateral_override,
            arm.cal_confidence, polarity["flex"], polarity["roll"],
            pitch_label, roll_label, frame_label,
        )

    # ── public API ────────────────────────────────────────────────────────────
    @property
    def any_connected(self) -> bool:
        return MOTORS.any_connected

    @property
    def connected_sides(self) -> list[ArmSide]:
        return MOTORS.connected_sides

    @property
    def active_arm(self) -> Optional[ArmSide]:
        return self._active_arm

    def set_vr_control_inputs_enabled(self, enabled: bool) -> None:
        """Enable/disable Quest button and grip side effects.

        Disabling keeps the VR pose stream alive but prevents normal teleop
        actions: face-button engage/recording/dual-mode and grip-triggered
        teleop anchors. Used by standalone calibration scripts that only need
        controller positions.
        """
        with self._lock:
            self._controller_buttons_enabled = bool(enabled)
            self._teleop_reset_anchors_enabled = bool(enabled)
            if not enabled:
                self._engaged = False
                self._active_arm = None
                self._dual_mode = False
                for arm in self._arms.values():
                    arm.reset_pending = False
                    arm.prev_buttons = {}

    def _robot_verification_or_test_active(self) -> bool:
        return any(
            arm.robot_verify_state != "idle" or arm.robot_verify_test_active
            for arm in self._arms.values()
        )

    def _restore_vr_control_inputs_if_idle(self) -> None:
        """Re-enable Quest button/grip side effects only after all verification
        and test modes are idle."""
        if not self._robot_verification_or_test_active():
            self.set_vr_control_inputs_enabled(True)

    def _restore_robot_verify_test_scale_if_idle(self) -> None:
        if (not any(arm.robot_verify_test_active for arm in self._arms.values())
                and self._scale_before_robot_verify_test is not None):
            self._scale = max(0.1, min(1.0, float(self._scale_before_robot_verify_test)))
            self._scale_before_robot_verify_test = None

    def connect(self, side: ArmSide) -> dict:
        """Connect ONE arm. The other arm (if connected) stays untouched."""
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        with self._lock:
            if MOTORS.is_connected(side):
                return self.status()
            # Reset per-arm state for this side.
            self._arms[side] = _PerArm(side=side)
            self._restore_persisted_arm_config(side)
            self._last_error = None
            self._load_gripper_config()
            try:
                MOTORS.connect(side)
                self._seed_targets_from_present(side)
                if self._drive_thread is None or not self._drive_thread.is_alive():
                    self._stop_evt.clear()
                    self._start_drive_loop()
            except Exception as e:
                self._last_error = f"{type(e).__name__}: {e}"
                log.exception("VR session connect failed for %s", side)
                try: MOTORS.disconnect(side)
                except Exception: pass
                raise
            return self.status()

    def disconnect(self, side: Optional[ArmSide] = None) -> dict:
        """Disconnect ONE arm, or both if side is None. KEEPS the VR pipeline
        running so the Quest browser stays connected. Use `emergency_stop()` to
        also tear down the VR servers."""
        with self._lock:
            if self._recording:
                raise RuntimeError("stop dataset recording before disconnecting arms")
            if self._recording_armed:
                self._recording_armed = False
                self._recording_pending_task = ""
                self._recording_pending_root = None
            sides = list(MOTORS.connected_sides) if side is None else [side]
            for s in sides:
                # Reset per-arm state on disconnect.
                self._arms[s] = _PerArm(side=s)
                if self._active_arm == s:
                    self._active_arm = None
                    self._engaged = False
                if self._dual_mode:
                    self._dual_mode = False
                    self._engaged = False
                try: MOTORS.disconnect(s)
                except Exception as e:
                    self._last_error = f"disconnect {s}: {e}"
                    log.warning("disconnect %s: %s", s, e)
            self._restore_robot_verify_test_scale_if_idle()
            self._restore_vr_control_inputs_if_idle()
            return self.status()

    def emergency_stop(self) -> dict:
        """Release torque on every connected arm immediately and tear down the
        VR servers. Flush any in-flight dataset episode. No motion. No homing."""
        with self._lock:
            self._engaged = False
            self._active_arm = None
            self._dual_mode = False
            was_recording = self._recording
            self._recording = False
            self._recording_armed = False
            self._recording_pending_task = ""
            self._recording_pending_root = None
            rec = self._recorder
            self._stop_evt.set()
            try:
                MOTORS.emergency_release_torque()
            except Exception as e:
                self._last_error = f"e-stop: {e}"
                log.warning("e-stop: %s", e)
            self._stop_threads_and_servers()
            for s in ("left", "right"):
                self._arms[s] = _PerArm(side=s)
        # Flush the recorder OUTSIDE the lock — finalize may encode video.
        if was_recording and rec is not None:
            try: rec.end_episode()
            except Exception as e: log.warning("e-stop: end_episode: %s", e)
            else:
                with self._lock:
                    self._episodes_saved = rec.episode_count
        if rec is not None:
            try: rec.finalize()
            except Exception as e: log.warning("e-stop: finalize: %s", e)
        with self._lock:
            self._recorder = None
        return self.status()

    def _load_gripper_config(self) -> None:
        """Read gripper.open_value / gripper.closed_value from config/xlerobot.yaml.
        Some SO101 calibrations have 0=open and 100=closed; others have the reverse.
        Lets the user flip the convention without touching code."""
        import yaml
        try:
            cfg = yaml.safe_load((REPO_ROOT / "config" / "xlerobot.yaml").read_text()) or {}
            g = cfg.get("gripper") or {}
            self._gripper_open = float(g.get("open_value", DEFAULT_GRIPPER_OPEN))
            self._gripper_closed = float(g.get("closed_value", DEFAULT_GRIPPER_CLOSED))
        except Exception as e:
            log.warning("could not read gripper config: %s", e)
            self._gripper_open = DEFAULT_GRIPPER_OPEN
            self._gripper_closed = DEFAULT_GRIPPER_CLOSED
        log.info("gripper config: open=%s closed=%s", self._gripper_open, self._gripper_closed)

    def engage(self, engaged: bool, scale: Optional[float] = None,
               active_arm: Optional[ArmSide] = None) -> dict:
        """Set the global engage state.

        - If `active_arm` is provided and `engaged=True`, that arm becomes the
          one that VR drives. The arm must be connected first.
        - If `active_arm` is omitted and `engaged=True`, the system picks the
          arm: if exactly one is connected, that one; if both, leaves the
          previous `_active_arm` if still valid; otherwise raises.
        - `engaged=False` clears the active arm and dual mode.
        - This API is single-arm selection; the left-controller Y button toggles
          dual mode without changing the target/IK/control code.
        """
        with self._lock:
            if self._robot_verification_or_test_active():
                if engaged:
                    raise RuntimeError(
                        "finish robot verification or stop the low-scale test before engaging VR teleop"
                    )
                self._engaged = False
                self._active_arm = None
                self._dual_mode = False
                return self.status()
            old_active = self._active_arm
            if scale is not None:
                self._scale = max(0.1, min(1.0, float(scale)))
            for arm in self._arms.values():
                arm.robot_verify_test_active = False
            self._controller_buttons_enabled = True
            self._teleop_reset_anchors_enabled = True
            if engaged:
                if not MOTORS.any_connected:
                    raise RuntimeError("connect an arm before engaging")
                if active_arm is not None:
                    if active_arm not in ("left", "right"):
                        raise ValueError(
                            f"active_arm must be 'left' or 'right', got {active_arm!r}"
                        )
                    if not MOTORS.is_connected(active_arm):
                        raise RuntimeError(f"{active_arm} arm not connected")
                    self._active_arm = active_arm
                elif self._active_arm is None or not MOTORS.is_connected(self._active_arm):
                    connected = MOTORS.connected_sides
                    if len(connected) == 1:
                        self._active_arm = connected[0]
                    else:
                        raise RuntimeError(
                            "both arms are connected; pass active_arm=left|right "
                            "to choose which arm to engage"
                        )
                arm = self._arms[self._active_arm]
                if arm.cal_confidence in ("poor", "legacy"):
                    raise RuntimeError(
                        f"{self._active_arm} VR calibration confidence is {arm.cal_confidence}; rerun calibration before engaging"
                    )
                if not arm.calibrated:
                    log.info(
                        "engage on %s but vr_calibrated=False — motors stay still"
                        " until the Quest controller's RESET (grip) is pressed",
                        self._active_arm,
                    )
            else:
                self._active_arm = None
                self._dual_mode = False
            self._engaged = bool(engaged)
            if self._engaged and self._active_arm is not None:
                self._dual_mode = False
            if self._active_arm is not None and self._active_arm != old_active:
                arm = self._arms[self._active_arm]
                arm.stale_since = None
            return self.status()

    def _ensure_profile_switch_allowed(self) -> None:
        if self._recording or self._recording_armed:
            raise RuntimeError("stop dataset recording before switching calibration profiles")
        if self._engaged:
            raise RuntimeError("disengage VR teleop before switching calibration profiles")
        for side, arm in self._arms.items():
            if arm.cal_state != "idle":
                raise RuntimeError(f"finish or cancel {side} VR calibration before switching profiles")
            if arm.robot_verify_state != "idle" or arm.robot_verify_test_active:
                raise RuntimeError(f"finish or cancel {side} robot verification before switching profiles")

    def _reload_active_calibration_profile(self) -> None:
        for side in ("left", "right"):
            self._restore_persisted_arm_config(side)
            self._invalidate_teleop_anchor(side, "calibration profile changed")

    def select_calibration_profile(self, name: str) -> dict:
        with self._lock:
            self._ensure_profile_switch_allowed()
            selected = _vrcal.set_active_profile(name)
            self._reload_active_calibration_profile()
            log.info("selected VR calibration profile %s", selected)
            return self.status()

    def create_calibration_profile(self, name: str, copy_from_active: bool = True) -> dict:
        with self._lock:
            self._ensure_profile_switch_allowed()
            current = _vrcal.profile_status().get("active_profile")
            created = _vrcal.create_profile(name, copy_from=current if copy_from_active else None)
            self._reload_active_calibration_profile()
            log.info("created and selected VR calibration profile %s", created)
            return self.status()

    def delete_calibration_profile(self, name: str) -> dict:
        with self._lock:
            self._ensure_profile_switch_allowed()
            active = _vrcal.delete_profile(name)
            self._reload_active_calibration_profile()
            log.info("deleted VR calibration profile %s; active profile is %s", name, active)
            return self.status()

    def _invalidate_teleop_anchor(self, side: ArmSide, reason: str) -> None:
        """Mark a teleop reset anchor stale after robot pose or calibration changes.

        This does not delete persisted VR/robot-verification calibration; it only
        requires the operator to grip-reset before live teleop or dataset capture.
        Caller must hold ``self._lock``.
        """
        arm = self._arms[side]
        arm.pose_generation += 1
        arm.anchor_generation = -1
        arm.anchor_invalid_reason = str(reason or "anchor invalidated")
        arm.calibrated = False
        arm.controller_anchor_T = None
        arm.vr_ctrl_to_ee = None
        arm.offset_robot = (0.0, 0.0, 0.0)
        arm.vr_offset_accum = (0.0, 0.0, 0.0)
        arm.pending_rel_position = (0.0, 0.0, 0.0)
        arm.reset_pending = False
        arm.prev_buttons = {}
        arm.quality_ticks = 0
        arm.quality_ik_rejects = 0

    def _teleop_anchor_fresh(self, side: ArmSide, *, include_wrist_axes: bool = True) -> bool:
        arm = self._arms[side]
        return (
            bool(arm.calibrated)
            and arm.anchor_generation == arm.pose_generation
            and arm.controller_anchor_T is not None
            and (not include_wrist_axes or _wrist_axes_ready(arm))
        )

    def _fresh_anchor_blockers(
        self,
        sides: list[ArmSide],
        *,
        include_wrist_axes: bool = True,
    ) -> list[str]:
        blockers: list[str] = []
        for side in sides:
            if not self._teleop_anchor_fresh(side, include_wrist_axes=include_wrist_axes):
                arm = self._arms[side]
                if include_wrist_axes and not _wrist_axes_ready(arm):
                    blockers.append(
                        f"{side} wrist pitch/roll calibration missing; rerun VR calibration wrist steps"
                    )
                elif arm.controller_anchor_T is None:
                    blockers.append(f"{side} teleop anchor not fresh (missing grip-reset controller anchor); squeeze grip to reset")
                else:
                    reason = arm.anchor_invalid_reason or "not anchored"
                    blockers.append(f"{side} teleop anchor not fresh ({reason}); squeeze grip to reset after homing")
        return blockers

    def _recording_anchor_refresh_blockers_locked(self, *, now: float) -> list[str]:
        blockers: list[str] = []
        for side in RECORDING_REQUIRED_SIDES:
            arm = self._arms[side]
            if not MOTORS.is_connected(side):
                blockers.append(f"{side} arm is not connected")
                continue
            if not MOTORS.is_torque_enabled(side):
                blockers.append(f"{side} torque is off")
                continue
            if arm.homing:
                blockers.append(f"{side} arm is still homing")
                continue
            if arm.cal_state != "idle":
                blockers.append(f"{side} VR calibration is still active")
                continue
            if not self._teleop_reset_anchors_enabled:
                blockers.append(f"{side} teleop anchoring is disabled by verification/test mode")
                continue
            goal = arm.latest
            if not goal.has_data:
                blockers.append(f"{side} Quest controller pose missing")
                continue
            age_s = max(0.0, now - goal.received_at)
            if age_s > GOAL_SKIP_AGE_S:
                blockers.append(f"{side} Quest controller pose stale ({age_s * 1000:.0f} ms)")
                continue
            if goal.controller_position is None or goal.rotation_quat is None:
                blockers.append(f"{side} Quest controller pose incomplete")
        return blockers

    def _refresh_recording_anchors_for_start(self) -> list[str]:
        """Capture fresh recording anchors from current robot and Quest poses.

        Pressing B should open an episode immediately after verification. The
        old `armed` state required the operator to know to grip-reset both arms
        after B. This method performs that deterministic reset itself from the
        latest Quest controller pose and current URDF FK pose. If the required
        pose data is not available, start is blocked with explicit reasons.
        """
        with self._lock:
            blockers = self._recording_anchor_refresh_blockers_locked(now=time.time())
            if blockers:
                return blockers
            refreshed: list[str] = []
            for side in RECORDING_REQUIRED_SIDES:
                try:
                    self._capture_anchor(side)
                    refreshed.append(side)
                except Exception as exc:
                    blockers.append(f"{side} recording anchor refresh failed: {exc}")
            if blockers:
                for side in refreshed:
                    self._invalidate_teleop_anchor(side, "recording anchor refresh failed")
        return blockers

    def _operator_status(
        self,
        arms_status: dict[str, Any],
        recording_info: dict[str, Any],
        now: float,
    ) -> dict[str, Any]:
        """Condensed Reachy-style operator state for the in-headset UI."""
        connected = list(MOTORS.connected_sides)
        camera_roles: dict[str, dict[str, Any]] = {}
        try:
            for cam in _cameras.enumerate_cameras():
                role = cam.role or cam.name
                camera_roles[role] = {
                    "configured": bool(cam.role),
                    "name": cam.name,
                    "stream_url": f"/camera/{cam.name}/stream",
                }
        except Exception as e:
            camera_roles["error"] = {"configured": False, "error": str(e)}

        head_camera = camera_roles.get("head")
        ready_blockers: list[str] = []
        if not connected:
            ready_blockers.append("connect at least one arm")
        native_ready = self._native_quest_clients > 0
        if not native_ready:
            ready_blockers.append("Quest app is not connected")
        if not head_camera:
            ready_blockers.append("head camera role is not configured")
        for side in connected:
            arm_status = arms_status.get(side, {})
            arm = self._arms[side]
            if not arm_status.get("torque_enabled", False):
                ready_blockers.append(f"{side} torque is off")
            age_ms = arm_status.get("controller", {}).get("age_ms")
            if age_ms is None:
                ready_blockers.append(f"{side} controller has not sent a pose")
            elif age_ms > int(GOAL_SKIP_AGE_S * 1000):
                ready_blockers.append(f"{side} controller pose is stale")

        if not connected:
            stage = "connect_required"
            guidance = "Connect at least one SO101 arm before entering VR teleop."
        elif self._engaged:
            active_sides = connected if self._dual_mode else ([self._active_arm] if self._active_arm in connected else [])
            anchored = any(self._arms[s].calibrated for s in active_sides)
            stage = "teleop_arms" if anchored else "teleop_head_only"
            guidance = (
                "Hold grip on each active controller to anchor and drive the arm."
                if anchored
                else "Teleop is engaged. Squeeze grip once on the active controller to anchor the arm."
            )
        elif ready_blockers:
            stage = "mirror_waiting_robot"
            guidance = ready_blockers[0]
        else:
            stage = "mirror_ready"
            guidance = "Face the workspace, press Ready in VR, then hold A/X or use Engage to start teleop."
        if self._last_error and self._engaged and self._last_error_suspends_teleop():
            stage = "suspended"
            guidance = self._last_error

        arm_panels: dict[str, Any] = {}
        for side in ("left", "right"):
            arm_status = arms_status.get(side, {})
            cal = arm_status.get("calibration", {})
            quality = cal.get("quality") or {}
            robot_verify = cal.get("robot_verification") or {}
            wrist_calibrated = bool(cal.get("wrist_axes_ready", False))
            arm_panels[side] = {
                "connected": bool(arm_status.get("connected", False)),
                "torque_enabled": bool(arm_status.get("torque_enabled", False)),
                "anchored": bool(arm_status.get("calibrated", False)),
                "wrist_aligned": wrist_calibrated,
                "wrist_calibrated": wrist_calibrated,
                "active": bool(self._dual_mode or self._active_arm == side),
                "controller_age_ms": arm_status.get("controller", {}).get("age_ms"),
                "ee_speed_cm_s": float(quality.get("offset_speed_ema_mps", 0.0)) * 100.0,
                "ik_reject_fraction": float(quality.get("ik_reject_fraction", 0.0)),
                "recording_readiness": robot_verify.get("readiness", "stage1_only"),
            }

        native_age_ms = (
            int(1000 * (now - self._native_quest_last_seen))
            if self._native_quest_last_seen is not None else None
        )
        return {
            "stage": stage,
            "guidance": guidance,
            "ready_blockers": ready_blockers,
            "recording_blockers": list(recording_info.get("calibration_blockers") or []),
            "recording_start_blockers": list(recording_info.get("start_blockers") or []),
            "recording_anchor_blockers": list(recording_info.get("anchor_blockers") or []),
            "connection": {
                "backend_ready": True,
                "https_ready": False,
                "websocket_ready": native_ready,
                "websocket_clients": self._native_quest_clients,
                "native_quest_ready": native_ready,
                "native_quest_clients": self._native_quest_clients,
                "native_quest_last_seen_ms": native_age_ms,
                "connected_arms": connected,
            },
            "camera_roles": camera_roles,
            "head_camera_url": head_camera.get("stream_url") if head_camera else None,
            "arm_panels": arm_panels,
            "recording": {
                "active": bool(self._recording),
                "armed": bool(recording_info.get("armed", False)),
                "frames": int(recording_info.get("frames_in_current_episode") or 0),
                "episodes_saved": int(recording_info.get("episodes_saved") or 0),
                "task": str(recording_info.get("last_task") or ""),
                "notice": str(recording_info.get("notice") or ""),
                "ready": bool(recording_info.get("calibration_ready", False)),
                "start_allowed": bool(recording_info.get("start_allowed", False)),
                "anchor_pending": bool(recording_info.get("anchor_pending", False)),
                "transition_active": bool(recording_info.get("transition_active", False)),
                "transition_source": str(recording_info.get("transition_source") or ""),
                "transition_target": recording_info.get("transition_target"),
                "transition_age_s": recording_info.get("transition_age_s"),
            },
            "updated_at": now,
        }

    def _recording_transition_status_locked(self) -> dict[str, Any]:
        age_s = (
            max(0.0, time.monotonic() - self._recording_transition_started_at)
            if self._recording_transition_active and self._recording_transition_started_at
            else 0.0
        )
        return {
            "transition_active": bool(self._recording_transition_active),
            "transition_source": self._recording_transition_source,
            "transition_target": self._recording_transition_target,
            "transition_age_s": age_s,
        }

    def _last_error_suspends_teleop(self) -> bool:
        """Only control-path failures should put the operator flow in suspended.

        Recording readiness/save messages must be visible diagnostics without
        disabling the Quest operator surface or blocking A/X engagement.
        """
        err = (self._last_error or "").strip().lower()
        if not err:
            return False
        non_suspending_prefixes = (
            "recording ",
            "record:",
            "recorder init:",
            "episode ",
            "save_episode failed:",
            "camera ",
            "task description ",
            "delete last episode ",
            "stop recording ",
        )
        return not err.startswith(non_suspending_prefixes)

    def _any_homing_locked(self) -> bool:
        return any(arm.homing for arm in self._arms.values())

    def quest_bridge_status(self, *, public_base_url: str | None = None) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            age_ms = (
                int(1000 * (now - self._native_quest_last_seen))
                if self._native_quest_last_seen is not None else None
            )
            return {
                "clients": self._native_quest_clients,
                "last_seen_ms": age_ms,
                "endpoint": "/api/vr/quest/ws",
                "ws_url": self._native_quest_ws_url(public_base_url=public_base_url),
                "coordinate_frame": self._native_quest.coordinate_frame,
                "pairing_required": True,
                "max_packet_bytes": MAX_NATIVE_QUEST_PACKET_BYTES,
            }

    def quest_operator_status(self, *, public_base_url: str | None = None) -> dict[str, Any]:
        operator = self._quest_operator_status_snapshot()
        connection = operator.get("connection") or {}
        recording = operator.get("recording") or {}
        calibration = operator.get("calibration") or {}
        robot_verification = operator.get("robot_verification") or {}
        video = _quest_video_bridge.bridge_status()
        with self._lock:
            last_error = self._last_error
            active_arm = self._active_arm
            dual_mode = self._dual_mode
            engaged = self._engaged
        return {
            "stage": str(operator.get("stage") or "connect_required"),
            "guidance": str(operator.get("guidance") or ""),
            "last_error": str(last_error or ""),
            "ready_blockers": list(operator.get("ready_blockers") or []),
            "recording_blockers": list(operator.get("recording_blockers") or []),
            "recording_start_blockers": list(operator.get("recording_start_blockers") or []),
            "recording_anchor_blockers": list(operator.get("recording_anchor_blockers") or []),
            "native_quest_ready": bool(connection.get("native_quest_ready", False)),
            "native_quest_clients": int(connection.get("native_quest_clients") or 0),
            "native_quest_last_seen_ms": connection.get("native_quest_last_seen_ms"),
            "connected_arms": list(connection.get("connected_arms") or []),
            "recording_active": bool(recording.get("active", False)),
            "recording_armed": bool(recording.get("armed", False)),
            "recording_ready": bool(recording.get("ready", False)),
            "recording_start_allowed": bool(recording.get("start_allowed", False)),
            "recording_anchor_pending": bool(recording.get("anchor_pending", False)),
            "recording_notice": str(recording.get("notice") or ""),
            "recording_frames": int(recording.get("frames") or 0),
            "recording_episodes_saved": int(recording.get("episodes_saved") or 0),
            "active_arm": active_arm,
            "dual_mode": bool(dual_mode),
            "engaged": bool(engaged),
            "calibration_active": bool(calibration.get("active", False)),
            "calibration_side": str(calibration.get("side") or ""),
            "calibration_state": str(calibration.get("state") or "idle"),
            "calibration_motion_m": float(calibration.get("motion_m") or 0.0),
            "calibration_target_m": float(calibration.get("target_m") or CALIBRATION_TARGET_MOTION_M),
            "calibration_min_m": float(calibration.get("min_m") or CALIBRATION_MIN_MOTION_M),
            "calibration_wrist_pitch_deg": float(calibration.get("wrist_pitch_deg") or 0.0),
            "calibration_wrist_roll_deg": float(calibration.get("wrist_roll_deg") or 0.0),
            "calibration_wrist_target_deg": float(calibration.get("wrist_target_deg") or WRIST_VERIFY_TARGET_DEG),
            "calibration_wrist_min_deg": float(calibration.get("wrist_min_deg") or WRIST_VERIFY_MIN_DEG),
            "calibration_confidence": str(calibration.get("confidence") or ""),
            "robot_verification_active": bool(robot_verification.get("active", False)),
            "robot_verification_side": str(robot_verification.get("side") or ""),
            "robot_verification_state": str(robot_verification.get("state") or "idle"),
            "robot_verification_label": str(robot_verification.get("label") or ""),
            "robot_verification_sample_count": int(robot_verification.get("sample_count") or 0),
            "robot_verification_min_samples": int(robot_verification.get("min_samples") or ROBOT_VERIFY_MIN_SAMPLES),
            "robot_verification_quality": str(robot_verification.get("quality") or "unverified"),
            "robot_verification_message": str(robot_verification.get("message") or ""),
            "robot_verification_live_state": str(robot_verification.get("live_state") or ""),
            "robot_verification_ready": bool(robot_verification.get("ready", False)),
            "robot_verification_direction_error_deg": robot_verification.get("direction_error_deg"),
            "robot_verification_magnitude_ratio": robot_verification.get("magnitude_ratio"),
            "robot_verification_position_error_cm": robot_verification.get("position_error_cm"),
            "robot_verification_vr_motion_m": robot_verification.get("vr_motion_m"),
            "robot_verification_target_motion_m": robot_verification.get("target_motion_m"),
            "robot_verification_target_robot_delta": robot_verification.get("target_robot_delta"),
            "robot_verification_predicted_robot_delta": robot_verification.get("predicted_robot_delta"),
            "robot_verification_vr_delta": robot_verification.get("vr_delta"),
            "robot_verification_fit_error_cm": robot_verification.get("fit_error_cm"),
            "robot_verification_residual_hint": str(robot_verification.get("residual_hint") or ""),
            "robot_verification_controls": str(robot_verification.get("controls") or ""),
            "ws_url": self._native_quest_ws_url(public_base_url=public_base_url),
            "coordinate_frame": self._native_quest.coordinate_frame,
            "video_ready": bool(video.get("ready", False)),
            "video_running": bool(video.get("running", False)),
            "video_transport": str(video.get("transport") or ""),
            "video_base_port": int(video.get("base_port") or 5600),
            "video_bitrate_kbps": int(video.get("bitrate_kbps") or 0),
            "video_running_roles": list(video.get("running_roles") or []),
            "video_receive_health": dict(video.get("receive_health") or {}),
        }

    def _quest_operator_status_snapshot(self) -> dict[str, Any]:
        """Readless status for the in-headset UI.

        The Quest app asks for this while streaming controller poses. Avoid
        motor bus reads here so headset traffic cannot starve dashboard actions
        such as robot connect, calibration, and recording controls.
        """
        now = time.time()
        with self._lock:
            arms_status: dict[str, Any] = {}
            calibration_summary: dict[str, Any] = {
                "active": False,
                "side": "",
                "state": "idle",
                "motion_m": 0.0,
                "target_m": CALIBRATION_TARGET_MOTION_M,
                "min_m": CALIBRATION_MIN_MOTION_M,
                "wrist_pitch_deg": 0.0,
                "wrist_roll_deg": 0.0,
                "wrist_target_deg": WRIST_VERIFY_TARGET_DEG,
                "wrist_min_deg": WRIST_VERIFY_MIN_DEG,
                "confidence": "",
            }
            robot_verification_summary: dict[str, Any] = {
                "active": False,
                "side": "",
                "state": "idle",
                "label": "",
                "sample_count": 0,
                "min_samples": ROBOT_VERIFY_MIN_SAMPLES,
                "quality": "unverified",
                "message": "",
                "live_state": "",
                "ready": False,
                "direction_error_deg": None,
                "magnitude_ratio": None,
                "position_error_cm": None,
                "vr_motion_m": None,
                "target_motion_m": None,
                "target_robot_delta": None,
                "predicted_robot_delta": None,
                "vr_delta": None,
                "fit_error_cm": None,
                "residual_hint": "",
                "controls": "",
            }
            for side in ("left", "right"):
                arm = self._arms[side]
                controller_age_ms = (
                    int(1000 * (now - arm.latest.received_at))
                    if arm.latest.has_data else None
                )
                ik_reject_fraction = (
                    float(arm.quality_ik_rejects) / float(arm.quality_ticks)
                    if arm.quality_ticks else 0.0
                )
                arms_status[side] = {
                    "connected": MOTORS.is_connected(side),
                    "torque_enabled": MOTORS.is_torque_enabled(side),
                    "calibrated": arm.calibrated,
                    "controller": {
                        "age_ms": controller_age_ms,
                    },
                    "calibration": {
                        "wrist_axes_ready": _wrist_axes_ready(arm),
                        "vr_ctrl_to_ee_ready": arm.vr_ctrl_to_ee is not None,
                        "quality": {
                            "offset_speed_ema_mps": float(arm.quality_offset_speed_ema_mps),
                            "ik_reject_fraction": ik_reject_fraction,
                        },
                        "robot_verification": {
                            "readiness": (
                                "ready_to_record"
                                if (
                                    arm.robot_verify_quality == "good"
                                    and arm.robot_verify_fit_error_cm is not None
                                    and arm.robot_verify_fit_error_cm <= ROBOT_VERIFY_PASS_ERROR_CM
                                    and arm.robot_verify_test_completed
                                )
                                else "verified_test_pending"
                                if arm.robot_verify_quality == "good"
                                else "stage1_only"
                            ),
                        },
                    },
                }
                if not calibration_summary["active"] and arm.cal_state != "idle":
                    calibration_summary = {
                        "active": True,
                        "side": side,
                        "state": arm.cal_state,
                        "motion_m": math.sqrt(sum(v * v for v in arm.cal_motion_acc)),
                        "target_m": CALIBRATION_TARGET_MOTION_M,
                        "min_m": CALIBRATION_MIN_MOTION_M,
                        "wrist_pitch_deg": float(arm.cal_wrist_pitch_verify_deg),
                        "wrist_roll_deg": float(arm.cal_wrist_roll_verify_deg),
                        "wrist_target_deg": WRIST_VERIFY_TARGET_DEG,
                        "wrist_min_deg": WRIST_VERIFY_MIN_DEG,
                        "confidence": arm.cal_confidence,
                    }
                if not robot_verification_summary["active"] and arm.robot_verify_state != "idle":
                    live = self._robot_verification_live_status(arm, now)
                    engage_button = ENGAGE_BUTTON_BY_SIDE.get(side, "A" if side == "right" else "X")
                    controls = (
                        f"Keep grip held; press {engage_button} to capture VR end"
                        if arm.robot_verify_vr_start is not None
                        else f"Hold grip; press {engage_button} to capture VR start"
                    )
                    robot_verification_summary = {
                        "active": True,
                        "side": side,
                        "state": arm.robot_verify_state,
                        "label": arm.robot_verify_label,
                        "sample_count": len(arm.robot_verify_samples),
                        "min_samples": ROBOT_VERIFY_MIN_SAMPLES,
                        "quality": arm.robot_verify_quality,
                        "message": str(live.get("message") or ""),
                        "live_state": str(live.get("state") or ""),
                        "ready": bool(live.get("ready", False)),
                        "direction_error_deg": live.get("direction_error_deg"),
                        "magnitude_ratio": live.get("magnitude_ratio"),
                        "position_error_cm": live.get("position_error_cm"),
                        "vr_motion_m": live.get("vr_motion_m"),
                        "target_motion_m": live.get("target_motion_m"),
                        "target_robot_delta": live.get("target_robot_delta"),
                        "predicted_robot_delta": live.get("predicted_robot_delta"),
                        "vr_delta": live.get("vr_delta"),
                        "fit_error_cm": arm.robot_verify_fit_error_cm,
                        "residual_hint": self._robot_verification_residual_hint(
                            arm.robot_verify_sample_residuals
                        ),
                        "controls": controls,
                    }

            rec = self._recorder
            recording_readiness = self._recording_readiness()
            recording_transition = self._recording_transition_status_locked()
            recording_info = {
                "active": self._recording,
                "armed": self._recording_armed,
                "episodes_saved": (rec.episode_count if rec else self._episodes_saved),
                "frames_in_current_episode": rec.frame_count_in_episode if rec else 0,
                "last_task": self._last_task,
                "notice": self._recording_notice,
                **recording_readiness,
                **recording_transition,
            }
            operator = self._operator_status(arms_status, recording_info, now)
            operator["calibration"] = calibration_summary
            operator["robot_verification"] = robot_verification_summary
            return operator

    def verify_native_quest_pairing_token(self, token: str | None) -> bool:
        expected = self._native_quest_pairing_token
        return bool(token) and secrets.compare_digest(str(token), expected)

    def _native_quest_ws_url(self, *, public_base_url: str | None = None) -> str:
        if public_base_url:
            parsed = urlparse(public_base_url)
            scheme = "wss" if parsed.scheme == "https" else "ws"
            if parsed.netloc:
                return f"{scheme}://{parsed.netloc}/api/vr/quest/ws"
        host = os.environ.get("OPENPIBOT_PUBLIC_HOST") or self._local_ip()
        port = int(os.environ.get("OPENPIBOT_PORT") or "5000")
        scheme = "wss" if os.environ.get("OPENPIBOT_TLS", "").lower() in {"1", "true", "yes"} else "ws"
        return f"{scheme}://{host}:{port}/api/vr/quest/ws"

    def status(self) -> dict:
        now = time.time()
        with self._lock:
            # Single bus read for both arms (each arm only has its own joints in
            # the result; MOTORS.read_positions(None) merges connected sides).
            joint_present: dict[str, float] = {}
            try:
                joint_present = MOTORS.read_positions()
            except Exception as e:
                self._last_error = f"read: {e}"

            arms_status: dict[str, Any] = {}
            for s in ("left", "right"):
                arm = self._arms[s]
                arms_status[s] = {
                    "connected": MOTORS.is_connected(s),
                    "torque_enabled": MOTORS.is_torque_enabled(s),
                    "calibrated": arm.calibrated,
                    "joint_target": arm.targets.to_dict_with_prefix(s)
                                     if MOTORS.is_connected(s) else {},
                    "controller": {
                        "position": (list(arm.latest.controller_position)
                                     if arm.latest.controller_position else None),
                        "rotation": (list(arm.latest.rotation_quat)
                                     if arm.latest.rotation_quat else None),
                        "trigger": arm.latest.trigger,
                        "thumbstick": {"x": arm.latest.thumbstick[0],
                                       "y": arm.latest.thumbstick[1]},
                        "age_ms": (int(1000 * (now - arm.latest.received_at))
                                    if arm.latest.has_data else None),
                        "mode": arm.latest.mode,
                    },
                    # Calibration diagnostics. After a RESET (grip-press), these
                    # let the user see the mapping in action:
                    #   anchor_ee_pos    = robot EE at the moment of grip-press
                    #   offset_robot     = cumulative offset since RESET (unclamped)
                    #   target_ee_pos    = anchor + offset, clamped to workspace
                    #   session_yaw_deg  = the yaw the user's "forward" was at RESET
                    "calibration": {
                        "anchor_ee_pos": list(arm.anchor_ee_pos),
                        "offset_robot": list(arm.offset_robot),
                        "target_ee_pos": [float(arm.target_T[0, 3]),
                                           float(arm.target_T[1, 3]),
                                           float(arm.target_T[2, 3])],
                        # Yaw of the calibrated operator-forward row in the
                        # horizontal operator frame. 0° means +operator.x maps
                        # to robot forward.
                        "session_yaw_deg": float(math.degrees(math.atan2(
                            arm.session_vr_to_robot[0, 1],
                            arm.session_vr_to_robot[0, 0],
                        ))),
                        # Guided-calibration wizard state.
                        "wizard_state": arm.cal_state,
                        "wizard_motion_m": math.sqrt(
                            arm.cal_motion_acc[0]**2 +
                            arm.cal_motion_acc[1]**2 +
                            arm.cal_motion_acc[2]**2
                        ),
                        "wizard_target_m": CALIBRATION_TARGET_MOTION_M,
                        "wizard_min_m": CALIBRATION_MIN_MOTION_M,
                        "wizard_last_fwd_m":  arm.cal_last_fwd_m,
                        "wizard_last_up_m":   arm.cal_last_up_m,
                        "wizard_last_left_m": arm.cal_last_left_m,
                        "validation":          dict(arm.cal_validation),
                        "wizard_fwd_captured":  arm.cal_captured_fwd  is not None,
                        "wizard_up_captured":   arm.cal_captured_up   is not None,
                        "wizard_left_captured": arm.cal_captured_left is not None,
                        "wizard_wrist_verify_deg": arm.cal_wrist_verify_deg,
                        "wizard_wrist_pitch_verify_deg": arm.cal_wrist_pitch_verify_deg,
                        "wizard_wrist_roll_verify_deg": arm.cal_wrist_roll_verify_deg,
                        "wizard_wrist_verify_target_deg": WRIST_VERIFY_TARGET_DEG,
                        "wizard_wrist_verify_min_deg": WRIST_VERIFY_MIN_DEG,
                        "wizard_wrist_captured": (
                            arm.wrist_pitch_canonical is not None
                            and arm.wrist_roll_canonical is not None
                        ),
                        "wizard_wrist_pitch_captured": arm.wrist_pitch_canonical is not None,
                        "wizard_wrist_roll_captured": arm.wrist_roll_canonical is not None,
                        "wrist_pitch_canonical": (
                            list(arm.wrist_pitch_canonical)
                            if arm.wrist_pitch_canonical is not None else None
                        ),
                        "wrist_roll_canonical": (
                            list(arm.wrist_roll_canonical)
                            if arm.wrist_roll_canonical is not None else None
                        ),
                        "wrist_axes_ready": _wrist_axes_ready(arm),
                        "direct_wrist_ready": _wrist_axes_ready(arm),
                        "invert_lateral":       arm.invert_lateral,
                        "confidence":           arm.cal_confidence,
                        "wrist_motor_polarity": dict(_WRIST_MOTOR_POLARITY.get(
                            s, {"flex": -1.0, "roll": -1.0}
                        )),
                        "controller_anchor_T": (
                            arm.controller_anchor_T.round(4).tolist()
                            if arm.controller_anchor_T is not None else None
                        ),
                        "vr_ctrl_to_ee_ready": arm.vr_ctrl_to_ee is not None,
                        "robot_anchor_T": arm.robot_anchor_T.round(4).tolist(),
                        "mapping_dry_run": self._mapping_dry_run_status(arm),
                        "diagnostics":          dict(arm.last_diag),
                        "quality": {
                            "offset_step_m": float(arm.quality_last_offset_step_m),
                            "offset_speed_ema_mps": float(arm.quality_offset_speed_ema_mps),
                            "ik_reject_fraction": (
                                float(arm.quality_ik_rejects) / float(arm.quality_ticks)
                                if arm.quality_ticks else 0.0
                            ),
                            "samples": arm.quality_ticks,
                        },
                        "robot_verification": {
                            "state": arm.robot_verify_state,
                            "sample_count": len(arm.robot_verify_samples),
                            "samples": list(arm.robot_verify_samples),
                            "current_label": arm.robot_verify_label,
                            "robot_start": list(arm.robot_verify_robot_start)
                                           if arm.robot_verify_robot_start else None,
                            "robot_end": list(arm.robot_verify_robot_end)
                                         if arm.robot_verify_robot_end else None,
                            "vr_start": list(arm.robot_verify_vr_start)
                                        if arm.robot_verify_vr_start else None,
                            "translation_scale": arm.translation_scale,
                            "fit_error_cm": arm.robot_verify_fit_error_cm,
                            "sample_residuals": list(arm.robot_verify_sample_residuals),
                            "worst_residuals": self._robot_verification_worst_residuals(
                                arm.robot_verify_sample_residuals
                            ),
                            "residual_hint": self._robot_verification_residual_hint(
                                arm.robot_verify_sample_residuals
                            ),
                            "quality": arm.robot_verify_quality,
                            "verified_at": arm.robot_verified_at,
                            "min_samples": ROBOT_VERIFY_MIN_SAMPLES,
                            "min_motion_m": ROBOT_VERIFY_MIN_MOTION_M,
                            "pass_error_cm": ROBOT_VERIFY_PASS_ERROR_CM,
                            "warn_error_cm": ROBOT_VERIFY_WARN_ERROR_CM,
                            "required_labels": list(ROBOT_VERIFY_REQUIRED_LABELS),
                            "missing_labels": self._missing_robot_verification_labels(arm),
                            "needs_recapture": arm.robot_verify_quality in ("warn", "poor", "needs_recapture"),
                            "has_verified_matrix": arm.robot_verify_quality == "good",
                            "readiness": (
                                "ready_to_record"
                                if (
                                    arm.robot_verify_quality == "good"
                                    and arm.robot_verify_fit_error_cm is not None
                                    and arm.robot_verify_fit_error_cm <= ROBOT_VERIFY_PASS_ERROR_CM
                                    and arm.robot_verify_test_completed
                                )
                                else "verified_test_pending"
                                if arm.robot_verify_quality == "good"
                                else "stage1_only"
                            ),
                            "live": self._robot_verification_live_status(arm, now),
                            "test_active": arm.robot_verify_test_active,
                            "test_completed": arm.robot_verify_test_completed,
                            "test_scale": arm.robot_verify_test_scale,
                        },
                    },
                }

            rec = self._recorder
            # If no recorder yet, compute what root WOULD be used (so the UI's
            # placeholder shows the actual default before first Start).
            task_default = ""
            if self._last_dataset_root:
                shown_root = self._last_dataset_root
                try:
                    task_default = str(_dataset.load_dataset_config().get("task_default") or "").strip()
                except Exception:
                    task_default = ""
            else:
                try:
                    cfg_now = _dataset.load_dataset_config()
                    task_default = str(cfg_now.get("task_default") or "").strip()
                    shown_root = _dataset.resolve_root(
                        cfg_now.get("root"), str(cfg_now["repo_id"]),
                    )
                except Exception:
                    shown_root = ""
            recording_readiness = self._recording_readiness()
            recording_transition = self._recording_transition_status_locked()
            recording_info = {
                "active": self._recording,
                "armed": self._recording_armed,
                "episodes_saved": (rec.episode_count if rec else self._episodes_saved),
                "frames_in_current_episode": rec.frame_count_in_episode if rec else 0,
                "last_episode_index": (
                    rec.last_saved_episode_index if rec else self._last_saved_episode_index
                ),
                "last_episode_frames": (
                    rec.last_saved_episode_frames if rec else self._last_saved_episode_frames
                ),
                "repo_id": rec.repo_id if rec else self._recording_repo_id,
                "last_task": self._last_task,
                "task_default": task_default,
                "root": shown_root,
                "notice": self._recording_notice,
                **recording_readiness,
                **recording_transition,
            }
            # Per-arm home pose status (from YAML + live homing flag).
            try:
                hp_status = _home.home_pose_status()
            except Exception as e:
                log.warning("home_pose_status failed: %s", e)
                hp_status = {"left": {"captured": False, "joints": {}},
                             "right": {"captured": False, "joints": {}}}
            for s in ("left", "right"):
                hp_status[s]["homing"] = self._arms[s].homing
                arms_status[s]["home"] = hp_status[s]

            # Per-arm persisted VR calibration status (config/vr_calibration.yaml).
            try:
                vr_cal_status = _vrcal.status()
            except Exception as e:
                log.warning("vr_calibration.status failed: %s", e)
                vr_cal_status = {"left": {"saved": False}, "right": {"saved": False}}
            for s in ("left", "right"):
                arms_status[s]["calibration"]["persisted"] = vr_cal_status[s]

            out = {
                "arms": arms_status,
                "operator": self._operator_status(arms_status, recording_info, now),
                "calibration_profiles": _vrcal.profile_status(),
                "connected_sides": list(MOTORS.connected_sides),
                "active_arm": self._active_arm,
                "dual_mode": self._dual_mode,
                "engaged": self._engaged,
                "scale": self._scale,
                "recording": self._recording,
                "recording_armed": self._recording_armed,
                "recording_info": recording_info,
                "last_tick_age_ms": (int(1000 * (now - self._last_drive_tick))
                                       if self._last_drive_tick else None),
                "last_error": self._last_error,
                "joint_present": joint_present,
                "joint_bounds": {j: list(MOTORS.bounds[j]) for j in MOTORS.bounds},
                "vr_endpoint": self._vr_endpoint_url(),
            }
            return out

    def _mapping_dry_run_status(self, arm: _PerArm) -> dict[str, Any]:
        """Synthetic, non-driving mapping checks shown in status for calibration review."""
        M = arm.session_vr_to_robot

        def map_vec(vec: tuple[float, float, float]) -> list[float]:
            out = self._runtime_translation_matrix(arm) @ _np.array(vec, dtype=float)
            return [float(v) for v in out]

        pitch_axis, roll_axis = _effective_wrist_axes(
            arm.side,
            pitch_canonical=arm.wrist_pitch_canonical,
            roll_canonical=arm.wrist_roll_canonical,
        )

        polarity = _WRIST_MOTOR_POLARITY.get(arm.side, {"flex": -1.0, "roll": -1.0})
        flex_pol = 1.0 if float(polarity.get("flex", -1.0)) >= 0.0 else -1.0
        roll_pol = 1.0 if float(polarity.get("roll", -1.0)) >= 0.0 else -1.0
        backward_target = _clamp_target_position(
            _np.array(arm.anchor_ee_pos) + _np.array(map_vec((0.0, 0.0, 0.10)))
        )
        return {
            "vr_forward_to_robot": map_vec((0.0, 0.0, -0.10)),
            "vr_backward_target_clamped": [float(v) for v in backward_target],
            "vr_up_to_robot": map_vec((0.0, 0.10, 0.0)),
            "vr_left_to_robot": map_vec((-0.10, 0.0, 0.0)),
            "wrist_pitch_probe_deg": float(flex_pol * math.degrees(0.1 * float(_np.dot(pitch_axis, pitch_axis)))),
            "wrist_roll_probe_deg": float(roll_pol * math.degrees(0.1 * float(_np.dot(roll_axis, roll_axis)))),
        }

    def _robot_verification_preview_scale(self, arm: _PerArm) -> float:
        ratios: list[float] = []
        for sample in arm.robot_verify_samples:
            try:
                robot_motion = float(sample.get("robot_motion_m", 0.0))
                vr_motion = float(sample.get("vr_motion_m", 0.0))
            except (TypeError, ValueError):
                continue
            if robot_motion >= ROBOT_VERIFY_MIN_MOTION_M and vr_motion >= ROBOT_VERIFY_MIN_MOTION_M:
                ratios.append(robot_motion / vr_motion)
        if ratios:
            return max(0.05, min(5.0, float(_np.median(_np.array(ratios, dtype=float)))))
        return max(0.05, min(5.0, float(arm.translation_scale or 1.0)))

    def _robot_verification_preview_matrix(self, arm: _PerArm) -> tuple[_np.ndarray, str]:
        """Translation preview used by robot verification UI.

        This must match the model used by final solve and live teleop:
        calibrated stage-1 direction frame plus one scalar reach scale. Do not
        guide the user with a provisional 3x3 least-squares matrix here. That
        matrix can cross-couple axes from partial/noisy samples, causing prompts
        like "up" to require a twist, while solve later rejects the same data
        against the stable runtime model.
        """
        matrix = (
            arm.base_vr_direction_matrix
            if arm.robot_verify_state != "idle" and arm.base_vr_direction_matrix is not None
            else arm.session_vr_to_robot
        )
        return self._robot_verification_preview_scale(arm) * self._effective_translation_matrix(arm, matrix), "stage1_scaled"

    @staticmethod
    def _normalize_robot_verification_label(label: str) -> str:
        return str(label or "").strip().lower().replace("_", "-").replace(" ", "-")

    def _effective_robot_verification_label(
        self,
        samples: list[dict[str, Any]],
        index: int,
    ) -> str:
        raw = self._normalize_robot_verification_label(samples[index].get("label", ""))
        if raw in ROBOT_VERIFY_REQUIRED_LABELS:
            return raw
        # Compatibility for samples captured by the Quest A/X button before the
        # dashboard sent the selected direction into backend state. Those samples
        # were stored as sample-1..sample-6 even though the UI directed the user
        # through the required directions in order.
        if (
            0 <= index < len(ROBOT_VERIFY_REQUIRED_LABELS)
            and raw == f"sample-{index + 1}"
            and len(samples) >= len(ROBOT_VERIFY_REQUIRED_LABELS)
        ):
            first_labels = [
                self._normalize_robot_verification_label(sample.get("label", ""))
                for sample in samples[:len(ROBOT_VERIFY_REQUIRED_LABELS)]
            ]
            if all(label == f"sample-{i + 1}" for i, label in enumerate(first_labels)):
                return ROBOT_VERIFY_REQUIRED_LABELS[index]
        return raw

    def _robot_verification_sample_motion_valid(self, sample: dict[str, Any]) -> bool:
        try:
            vr = _np.array(sample.get("vr_delta"), dtype=float)
            robot = _np.array(sample.get("robot_delta"), dtype=float)
        except Exception:
            return False
        if vr.shape != (3,) or robot.shape != (3,):
            return False
        if not _np.all(_np.isfinite(vr)) or not _np.all(_np.isfinite(robot)):
            return False
        return (
            float(_np.linalg.norm(vr)) >= ROBOT_VERIFY_MIN_MOTION_M
            and float(_np.linalg.norm(robot)) >= ROBOT_VERIFY_MIN_MOTION_M
        )

    def _missing_robot_verification_labels(self, arm: _PerArm) -> list[str]:
        samples = list(arm.robot_verify_samples)
        captured = {
            self._effective_robot_verification_label(samples, idx)
            for idx in range(len(samples))
            if self._robot_verification_sample_motion_valid(samples[idx])
        }
        return [label for label in ROBOT_VERIFY_REQUIRED_LABELS if label not in captured]

    @staticmethod
    def _robot_verification_worst_residuals(
        residuals: list[dict[str, Any]],
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for item in residuals or []:
            if not isinstance(item, dict):
                continue
            try:
                residual_cm = float(item.get("residual_cm"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(residual_cm):
                continue
            normalized = dict(item)
            normalized["residual_cm"] = residual_cm
            cleaned.append(normalized)
        cleaned.sort(key=lambda item: float(item["residual_cm"]), reverse=True)
        return cleaned[:max(0, limit)]

    def _robot_verification_residual_hint(self, residuals: list[dict[str, Any]]) -> str:
        worst = self._robot_verification_worst_residuals(residuals, limit=6)
        if not worst:
            return ""

        def label_for(item: dict[str, Any]) -> str:
            return str(item.get("label") or f"sample-{int(item.get('index', 0)) + 1}")

        worst_text = ", ".join(
            f"{label_for(item)} {float(item['residual_cm']):.1f} cm"
            for item in worst[:3]
        )
        recapture = [item for item in worst if float(item["residual_cm"]) > ROBOT_VERIFY_PASS_ERROR_CM]
        if not recapture:
            recapture = worst[:1]
        recapture_labels: list[str] = []
        for item in recapture:
            label = label_for(item)
            if label not in recapture_labels:
                recapture_labels.append(label)
            if len(recapture_labels) >= 3:
                break
        return (
            f"Worst residuals: {worst_text}. "
            f"Recapture {', '.join(recapture_labels)} first near the grasp workspace."
        )

    def _effective_translation_matrix(self, arm: _PerArm, matrix: _np.ndarray) -> _np.ndarray:
        return _ArmMovementMapper.effective_translation_matrix(arm, matrix)

    def _usable_verified_translation_matrix(self, arm: _PerArm) -> _np.ndarray | None:
        return _ArmMovementMapper.usable_verified_translation_matrix(arm)

    def _runtime_translation_matrix(self, arm: _PerArm) -> _np.ndarray:
        """Effective VR-delta → robot-delta matrix for translation.

        Prefer a good six-direction robot-verified matrix so the physical arm's
        measured scale and cross-axis behavior are represented in live teleop.
        Fall back to the stage-1 frame only when the verified matrix is absent,
        unverified, or mathematically unsafe.
        """
        return _ArmMovementMapper.runtime_translation_matrix(arm)

    def _runtime_translation_source(self, arm: _PerArm) -> str:
        return _ArmMovementMapper.runtime_translation_source(arm)

    def _robot_verification_live_status(self, arm: _PerArm, now: float) -> dict[str, Any]:
        """Live, non-driving check for the current robot-verification sample.

        The VR "start" here is only a neutral controller anchor. We compare the
        user's current controller delta from that anchor with the demonstrated
        robot EE delta for the selected direction.
        """
        out: dict[str, Any] = {
            "ready": False,
            "state": "waiting",
            "message": "Capture robot start and robot end for the selected direction.",
            "sample_label": arm.robot_verify_label,
            "controller_age_ms": (
                int(1000 * (now - arm.latest.received_at))
                if arm.latest.has_data else None
            ),
            "target_robot_delta": None,
            "predicted_robot_delta": None,
            "vr_delta": None,
            "target_motion_m": None,
            "predicted_motion_m": None,
            "vr_motion_m": None,
            "vr_delta_source": "absolute_controller_pose",
            "direction_error_deg": None,
            "direction_match": None,
            "magnitude_ratio": None,
            "position_error_cm": None,
            "scale_estimate": self._robot_verification_preview_scale(arm),
            "preview_source": "stage1_scaled",
        }
        if arm.robot_verify_state == "idle":
            out["state"] = "idle"
            if arm.robot_verify_quality == "good":
                out["message"] = "Verified calibration is saved. Use low-scale test before recording."
            elif arm.robot_verify_quality in ("warn", "needs_recapture", "poor"):
                hint = self._robot_verification_residual_hint(arm.robot_verify_sample_residuals)
                out["message"] = hint or "Robot verification residual is too high. Recapture all six directions."
            else:
                out["message"] = "Start robot verification to refine the VR-only mapping."
            return out
        if arm.robot_verify_robot_start is None or arm.robot_verify_robot_end is None:
            return out

        robot_start = _np.array(arm.robot_verify_robot_start, dtype=float)
        robot_end = _np.array(arm.robot_verify_robot_end, dtype=float)
        robot_delta = robot_end - robot_start
        robot_motion = float(_np.linalg.norm(robot_delta))
        out["target_robot_delta"] = [float(v) for v in robot_delta]
        out["target_motion_m"] = robot_motion
        if robot_motion < ROBOT_VERIFY_MIN_MOTION_M:
            out["state"] = "robot_motion_too_small"
            out["message"] = (
                f"Robot target move is only {robot_motion * 100:.1f} cm; "
                f"use at least {ROBOT_VERIFY_MIN_MOTION_M * 100:.1f} cm."
            )
            return out
        if arm.robot_verify_vr_start is None:
            out["state"] = "need_vr_neutral"
            out["message"] = (
                "Set VR neutral with the controller held naturally. This becomes "
                "the temporary VR start pose for this sample."
            )
            return out
        if not arm.latest.has_data or arm.latest.controller_position is None:
            out["state"] = "no_controller"
            out["message"] = "Waiting for controller pose from the Quest page."
            return out
        age_s = now - arm.latest.received_at
        if age_s > 2.0:
            out["state"] = "stale_controller"
            out["message"] = (
                f"Controller pose is stale ({age_s:.1f}s). Move the controller "
                "or briefly hold its grip to refresh pose packets."
            )
            return out

        vr_start = _np.array(arm.robot_verify_vr_start, dtype=float)
        vr_current = _np.array(arm.latest.controller_position, dtype=float)
        vr_delta = vr_current - vr_start
        if not _np.all(_np.isfinite(vr_delta)):
            out["state"] = "no_controller"
            out["message"] = "Controller pose is invalid; move the controller and retry this sample."
            return out
        vr_motion = float(_np.linalg.norm(vr_delta))
        out["vr_delta"] = [float(v) for v in vr_delta]
        out["vr_motion_m"] = vr_motion
        effective_M, preview_source = self._robot_verification_preview_matrix(arm)
        predicted = effective_M @ vr_delta
        predicted_motion = float(_np.linalg.norm(predicted))
        out["predicted_robot_delta"] = [float(v) for v in predicted]
        out["predicted_motion_m"] = predicted_motion
        out["scale_estimate"] = self._robot_verification_preview_scale(arm)
        out["preview_source"] = preview_source
        if vr_motion < ROBOT_VERIFY_MIN_MOTION_M:
            out["state"] = "move_vr"
            out["message"] = (
                f"Hold grip while moving the controller in the same direction as "
                f"the robot target ({ROBOT_VERIFY_MIN_MOTION_M * 100:.1f}+ cm)."
            )
            return out

        if predicted_motion < 1e-9:
            out["state"] = "move_vr"
            out["message"] = "Controller movement is too small to evaluate."
            return out

        cos = float(_np.dot(robot_delta, predicted) / (robot_motion * predicted_motion))
        cos = max(-1.0, min(1.0, cos))
        direction_error_deg = math.degrees(math.acos(cos))
        magnitude_ratio = predicted_motion / robot_motion
        err_cm = float(_np.linalg.norm(predicted - robot_delta) * 100.0)
        ready = (
            direction_error_deg <= ROBOT_VERIFY_LIVE_GOOD_ANGLE_DEG
            and ROBOT_VERIFY_LIVE_GOOD_RATIO_MIN <= magnitude_ratio <= ROBOT_VERIFY_LIVE_GOOD_RATIO_MAX
            and err_cm <= ROBOT_VERIFY_PASS_ERROR_CM
        )
        out.update({
            "ready": ready,
            "state": "good" if ready else "adjust",
            "message": (
                "Good match. Capture VR end now."
                if ready
                else "Adjust the controller move until direction and scale are closer."
            ),
            "direction_error_deg": direction_error_deg,
            "direction_match": max(0.0, min(1.0, (cos + 1.0) / 2.0)),
            "magnitude_ratio": magnitude_ratio,
            "position_error_cm": err_cm,
        })
        return out

    # ── Native Quest packet pipeline ────────────────────────────────────────
    def _vr_endpoint_url(self) -> Optional[str]:
        return None

    @staticmethod
    def _local_ip() -> str:
        """Pick the most likely LAN IP for the Quest to connect to.

        Order of preference:
          1. 192.168.x.x  (typical home LAN)
          2. 10.x.x.x     (corporate LAN / second-tier)
          3. 172.16-31.x  (also private; exclude Docker bridges 172.17-19)
          4. Whatever the default-route trick returns

        Filtered out:
          - 127.x.x.x (loopback)
          - 169.254.x.x (link-local — won't be reachable)
          - 100.64-127.x.x (Tailscale CGNAT range)
          - 172.17-19.x.x (default Docker bridges)
        """
        def _classify(ip: str) -> int:
            try:
                a, b, *_ = (int(p) for p in ip.split("."))
            except ValueError:
                return 99
            if ip.startswith("127.") or ip.startswith("169.254."):
                return 99
            if a == 100 and 64 <= b <= 127:                 # Tailscale
                return 90
            if a == 172 and 17 <= b <= 19:                  # Docker bridges
                return 90
            if ip.startswith("192.168."):
                return 0
            if ip.startswith("10."):
                return 1
            if a == 172 and 16 <= b <= 31:                  # other private 172.x
                return 2
            return 50  # public IPs and everything else

        # Collect every IPv4 we can find via hostname -I.
        candidates: list[str] = []
        try:
            out = subprocess.check_output(["hostname", "-I"], timeout=2).decode().strip()
            candidates = [ip for ip in out.split() if "." in ip]
        except Exception:
            pass

        # Add the default-route IP too (in case `hostname -I` is missing on the host).
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                if ip not in candidates:
                    candidates.append(ip)
        except OSError:
            pass

        if not candidates:
            return "localhost"
        candidates.sort(key=_classify)
        return candidates[0] if _classify(candidates[0]) < 90 else candidates[0]

    def ingest_native_quest_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        """Ingest one native Unity/OpenXR packet and update per-arm goals.

        This is intentionally synchronous so REST handlers, WebSocket handlers,
        and tests can share the exact same adapter path.
        """
        if not isinstance(packet, dict):
            raise NativeQuestProtocolError("packet must be a JSON object")
        try:
            packet_bytes = len(json.dumps(packet, separators=(",", ":")).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise NativeQuestProtocolError("packet must be JSON serializable") from exc
        if packet_bytes > MAX_NATIVE_QUEST_PACKET_BYTES:
            raise NativeQuestProtocolError(
                f"packet exceeds {MAX_NATIVE_QUEST_PACKET_BYTES} byte limit"
            )
        goals = self._native_quest.process_packet(packet)
        with self._lock:
            self._native_quest_last_seen = time.time()
        for goal in goals:
            self._apply_control_goal(goal)
        return {"ok": True, "goals": len(goals)}

    def note_native_quest_client(self, connected: bool) -> None:
        with self._lock:
            if connected:
                self._native_quest_clients += 1
                self._native_quest_last_seen = time.time()
            else:
                self._native_quest_clients = max(0, self._native_quest_clients - 1)

    def _apply_control_goal(self, goal: Any) -> None:
        side = getattr(goal, "arm", None)
        if side not in ("left", "right"):
            return  # headset / unknown
        arm = self._arms[side]
        mode_obj = getattr(goal, "mode", None)
        mode = getattr(mode_obj, "value", mode_obj) or "idle"
        rp = getattr(goal, "relative_position", None)
        rr = getattr(goal, "relative_rotvec", None)
        cp = getattr(goal, "vr_ctrl_position", None)
        rot = getattr(goal, "vr_ctrl_rotation", None)
        trig = bool(getattr(goal, "trigger", False))
        thumb = getattr(goal, "thumbstick", None) or {}
        btn = getattr(goal, "buttons", None) or {}
        cur_buttons: dict[str, bool] = {}
        prev_buttons: dict[str, bool] = {}
        handle_buttons = False
        with self._lock:
            cur_buttons = {str(k): bool(v) for k, v in btn.items()}
            prev_buttons = arm.prev_buttons
            arm.prev_buttons = dict(cur_buttons)
            if str(mode) == "reset":
                arm.reset_pending = True
                arm.pending_rel_position = (0.0, 0.0, 0.0)
            arm.latest = _LatestGoal(
                received_at=time.time(),
                has_data=True,
                mode=str(mode),
                rel_position=tuple(float(v) for v in (rp if rp is not None else (0, 0, 0))),
                rel_rotvec=tuple(float(v) for v in (rr if rr is not None else (0, 0, 0))),
                controller_position=(tuple(float(v) for v in cp) if cp is not None else None),
                rotation_quat=(tuple(float(v) for v in rot.as_quat()) if rot is not None and hasattr(rot, "as_quat") else None),
                trigger=trig,
                thumbstick=(float(thumb.get("x", 0)), float(thumb.get("y", 0))),
                buttons=cur_buttons,
            )
            if str(mode) == "position":
                rel = _np.array(rp if rp is not None else (0.0, 0.0, 0.0), dtype=float)
                if _np.all(_np.isfinite(rel)):
                    driving_this_arm = self._engaged and (
                        self._dual_mode or self._active_arm == side
                    )
                    if driving_this_arm:
                        pending = _np.array(arm.pending_rel_position, dtype=float) + rel
                        arm.pending_rel_position = (
                            float(pending[0]), float(pending[1]), float(pending[2])
                        )
                if (
                    arm.robot_verify_state == "vr_start_captured"
                    and arm.robot_verify_vr_start is not None
                    and cp is not None
                ):
                    verify_delta = _np.array(cp, dtype=float) - _np.array(
                        arm.robot_verify_vr_start,
                        dtype=float,
                    )
                    if _np.all(_np.isfinite(verify_delta)):
                        arm.robot_verify_vr_delta_accum = (
                            float(verify_delta[0]),
                            float(verify_delta[1]),
                            float(verify_delta[2]),
                        )
            elif str(mode) == "idle":
                if (
                    arm.robot_verify_state == "vr_start_captured"
                    and arm.robot_verify_vr_start is not None
                ):
                    self._reset_robot_verification_vr_capture(
                        arm,
                        reason="grip released before VR end capture",
                    )
            if arm.cal_state != "idle":
                self._advance_calibration(side, arm.latest)
            handle_buttons = self._controller_buttons_enabled or arm.robot_verify_state != "idle"
        if handle_buttons:
            self._handle_button_edges(side, cur_buttons, prev_buttons)

    # ── drive loop ──────────────────────────────────────────────────────────
    def _start_drive_loop(self) -> None:
        self._drive_thread = threading.Thread(
            target=self._drive_loop, daemon=True, name="vr-drive"
        )
        self._drive_thread.start()

    def _seed_targets_from_present(self, side: ArmSide) -> None:
        """Initialise live targets for ONE arm from its present joint positions
        so the first command doesn't try to swing toward zero."""
        if not MOTORS.is_connected(side):
            return
        pres = MOTORS.read_positions(side)
        prefix = f"{side}_arm_"
        get = lambda j: float(pres.get(f"{prefix}{j}", 0.0))
        arm = self._arms[side]
        arm.targets = _LiveTargets(
            shoulder_pan=get("shoulder_pan"),
            shoulder_lift=get("shoulder_lift"),
            elbow_flex=get("elbow_flex"),
            wrist_flex=get("wrist_flex"),
            wrist_roll=get("wrist_roll"),
            gripper=get("gripper"),
        )
        arm.last_sent_targets = {f"{prefix}{j}": getattr(arm.targets, j)
                                  for j in _motors.JOINTS_PER_ARM}
        arm.last_commanded_targets = dict(arm.last_sent_targets)
        arm.command_filter.reset(arm.last_sent_targets)
        q_now_deg = _np.array([get(j) for j in _IK_JOINT_ORDER], dtype=float)
        arm.last_q_sol = q_now_deg.copy()
        arm.last_body_q_sol = q_now_deg[:len(_BODY_IK_JOINT_ORDER)].copy()
        arm.last_q_filtered = q_now_deg.copy()

    def _ensure_kinematics(self, arm: _PerArm) -> bool:
        """Return whether calibrated SO101 URDF kinematics should be used."""
        if arm.kinematics is None:
            arm.kinematics = _load_urdf_kinematics()
        arm.using_analytical_fallback = False
        return arm.kinematics is not None

    def _require_urdf_kinematics(self, side: ArmSide, context: str) -> None:
        """Dataset-quality calibration/recording must use calibrated URDF FK/IK."""
        arm = self._arms[side]
        if not self._ensure_kinematics(arm):
            raise RuntimeError(
                f"{side} calibrated SO101 URDF kinematics unavailable; cannot {context}. "
                "Install/load the calibrated URDF path before robot verification or recording."
            )

    def _homing_cap_for_key(self, key: str) -> float:
        return PER_TICK_DEG_CAPS.get(normalized_joint_name(key), 1.0)

    def _new_homing_controller(
        self,
        arm: _PerArm,
        present: dict[str, float],
    ) -> JointHomingController:
        return JointHomingController(
            targets=arm.home_target,
            present=present,
            cap_for_key=self._homing_cap_for_key,
            kp=KP,
            command_tolerance_deg=HOMING_TOL_DEG,
            present_tolerance_deg=HOMING_PRESENT_TOL_DEG,
            settle_ticks=HOMING_SETTLE_TICKS,
        )

    def _homing_step(self, arm: _PerArm, present: dict[str, float]) -> HomingStep:
        if arm.home_controller is None:
            arm.home_controller = self._new_homing_controller(arm, present)
        return arm.home_controller.step(present)

    def _homing_step_command(
        self,
        side: ArmSide,
        arm: _PerArm,
        present: dict[str, float],
    ) -> tuple[dict[str, float], bool]:
        """Compatibility wrapper for tests and older callers."""
        del side
        step = self._homing_step(arm, present)
        return step.command, step.settled

    def _current_ee_transform(self, side: ArmSide) -> _np.ndarray:
        """Read current body joints and return the estimated wrist-link pose."""
        if not MOTORS.is_connected(side):
            raise RuntimeError(f"{side} arm not connected")
        arm = self._arms[side]
        use_urdf = self._ensure_kinematics(arm)
        if not use_urdf:
            raise RuntimeError(
                f"{side} calibrated SO101 URDF kinematics unavailable; cannot estimate EE pose"
            )
        present = MOTORS.read_positions(side)
        prefix = f"{side}_arm_"
        get = lambda j: float(present.get(f"{prefix}{j}", 0.0))
        q_now_deg = _np.array([get(j) for j in _BODY_IK_JOINT_ORDER], dtype=float)
        return arm.kinematics.forward_kinematics(q_now_deg)

    def _capture_anchor(self, side: ArmSide) -> None:
        """RESET handler for ONE arm.

        Snapshot the current EE pose; this becomes the anchor for
        reset-relative teleop. Resets the target filters and delta accumulator
        while keeping the calibrated VR->robot translation frame stable.
        """
        if not MOTORS.is_connected(side):
            return
        arm = self._arms[side]
        use_urdf = self._ensure_kinematics(arm)
        if not use_urdf:
            self._last_error = (
                f"{side} teleop requires calibrated SO101 URDF kinematics; "
                "refusing analytical IK fallback"
            )
            raise RuntimeError(self._last_error)
        if not _wrist_axes_ready(arm):
            self._last_error = (
                f"{side} wrist pitch/roll calibration missing; rerun VR calibration wrist steps"
            )
            arm.calibrated = False
            arm.anchor_invalid_reason = self._last_error
            raise RuntimeError(self._last_error)
        ctrl_quat = arm.latest.rotation_quat if arm.latest.has_data else None
        ctrl_pos = arm.latest.controller_position if arm.latest.has_data else None
        if ctrl_quat is None or ctrl_pos is None:
            self._last_error = (
                f"{side} controller pose missing during reset; hold grip with full Quest pose data"
            )
            arm.calibrated = False
            arm.controller_anchor_T = None
            arm.vr_ctrl_to_ee = None
            arm.anchor_invalid_reason = self._last_error
            raise RuntimeError(self._last_error)
        try:
            controller_anchor_T = _pose_matrix_from_vr(ctrl_pos, ctrl_quat)
        except Exception as e:
            self._last_error = f"{side} controller pose invalid during reset: {e}"
            arm.calibrated = False
            arm.controller_anchor_T = None
            arm.vr_ctrl_to_ee = None
            arm.anchor_invalid_reason = self._last_error
            raise RuntimeError(self._last_error)

        present = MOTORS.read_positions(side)
        prefix = f"{side}_arm_"
        get = lambda j: float(present.get(f"{prefix}{j}", 0.0))

        q_now_deg = _np.array([get(j) for j in _IK_JOINT_ORDER], dtype=float)
        q_body_now_deg = q_now_deg[:len(_BODY_IK_JOINT_ORDER)].copy()
        # Calibrated SO101 URDF FK at current body joints -> 4x4 wrist-link pose.
        T_now = arm.kinematics.forward_kinematics(q_body_now_deg)

        arm.target_T = T_now.copy()
        arm.robot_anchor_T = T_now.copy()
        arm.anchor_ee_pos = (float(T_now[0, 3]), float(T_now[1, 3]), float(T_now[2, 3]))
        arm.anchor_R_robot = T_now[:3, :3].copy()
        arm.offset_robot = (0.0, 0.0, 0.0)
        arm.vr_offset_accum = (0.0, 0.0, 0.0)
        arm.pending_rel_position = (0.0, 0.0, 0.0)
        arm.last_q_sol = q_now_deg.copy()
        arm.last_body_q_sol = q_body_now_deg.copy()
        arm.last_q_filtered = q_now_deg.copy()
        arm.vr_ctrl_to_ee = None
        arm.quality_ticks = 0
        arm.quality_ik_rejects = 0
        arm.quality_offset_speed_ema_mps = 0.0
        arm.quality_last_offset_step_m = 0.0

        # Keep the translation frame stable. Rebuilding `session_vr_to_robot`
        # from the controller quaternion on every RESET makes hand translation
        # depend on wrist pose at anchor time, so arm motion feels random. Use the
        # saved/wizard/default frame for position; the quaternion is still stored
        # below as the wrist-orientation anchor.
        arm.controller_anchor_T = controller_anchor_T
        try:
            R_ee = _R.from_matrix(_project_to_rotation_matrix(T_now[:3, :3]))
            R_vr = _R.from_quat(_positive_quat_xyzw(_np.array(ctrl_quat, dtype=float)))
            R_ctrl_in_robot = _R.from_matrix(_project_to_rotation_matrix(arm.session_vr_to_robot)) * R_vr
            arm.vr_ctrl_to_ee = R_ee.inv() * R_ctrl_in_robot
        except Exception as e:
            arm.vr_ctrl_to_ee = None
            log.warning("[%s] could not capture controller→EE wrist frame: %s", side, e)
        log.info("[%s] keeping VR→robot translation frame:\n%s", side, arm.session_vr_to_robot.round(3))

        arm.anchor = _AnchorPose(
            ee_x=float(T_now[0, 3]),
            ee_y=float(T_now[1, 3]),
            pan_deg=get("shoulder_pan"),
            shoulder_lift_deg=get("shoulder_lift"),
            elbow_flex_deg=get("elbow_flex"),
            wrist_flex_deg=get("wrist_flex"),
            wrist_roll_deg=get("wrist_roll"),
            gripper_pct=get("gripper"),
            captured=True,
            ctrl_quat=ctrl_quat,
        )
        arm.targets = _LiveTargets(
            shoulder_pan=get("shoulder_pan"),
            shoulder_lift=get("shoulder_lift"),
            elbow_flex=get("elbow_flex"),
            wrist_flex=get("wrist_flex"),
            wrist_roll=get("wrist_roll"),
            gripper=get("gripper"),
        )
        arm.last_sent_targets = arm.targets.to_dict_with_prefix(side)
        arm.last_commanded_targets = dict(arm.last_sent_targets)
        arm.command_filter.reset(arm.last_sent_targets)
        arm.calibrated = True
        arm.anchor_generation = arm.pose_generation
        arm.anchor_invalid_reason = ""
        log.info("[%s] VR anchor: body wrist-link=(%.3f, %.3f, %.3f) m (URDF FK)",
                 side, T_now[0, 3], T_now[1, 3], T_now[2, 3])

    def _drive_loop(self) -> None:
        """Per-tick: process RESETs for ALL connected arms (so each arm's anchor
        is up-to-date). Command motion on the active arm, or both arms when
        dual mode is enabled. The target/IK/control math remains per-arm."""
        next_tick = time.monotonic()
        while not self._stop_evt.is_set():
            now = time.time()
            try:
                with self._lock:
                    engaged = self._engaged
                    active = self._active_arm
                    dual_mode = self._dual_mode
                    scale = self._scale
                    connected = list(MOTORS.connected_sides)
                self._last_drive_tick = now

                if not connected:
                    next_tick = self._sleep_until(next_tick)
                    continue

                # Phase 1: handle RESET / IDLE / drain accumulator
                # for EVERY connected arm. We do this regardless of which one is
                # "active" so that switching active_arm mid-session doesn't pick up
                # stale accumulator data or skip a fresh anchor.
                for side in connected:
                    arm = self._arms[side]
                    with self._lock:
                        goal = arm.latest
                        reset_now = arm.reset_pending
                        if reset_now:
                            arm.reset_pending = False

                    # Note: we do NOT clear `arm.calibrated` on IDLE goals.
                    # Releasing grip just stops motion; the anchor stays valid so
                    # the UI keeps showing the captured pose. The next grip-press
                    # sends a fresh RESET goal which re-anchors via `_capture_anchor`
                    # below. Without this, the UI flipped to "not calibrated" every
                    # time the user released grip, which looked like a bug.

                    # RESET captures the anchor for THIS arm — but ONLY if we're
                    # not in the middle of a guided calibration. During calibration,
                    # the grip-press is consumed by the calibration state machine
                    # (via _advance_calibration in _drain_goals); we don't want to
                    # also anchor for teleop until calibration is done.
                    should_capture_anchor = (
                        (reset_now or (goal.has_data and goal.mode == "reset"))
                        and arm.cal_state == "idle"
                        and self._teleop_reset_anchors_enabled
                    )
                    if should_capture_anchor:
                        with self._lock:
                            self._capture_anchor(side)

                # Phase 1.5: drive any HOMING arms toward their home targets.
                # Runs regardless of engage/active state — homing is its own mode.
                # Uses the same per-tick caps as VR teleop plus homing KP, so
                # motion is slow and bus-safe.
                for side in connected:
                    arm = self._arms[side]
                    if not arm.homing or not arm.home_target:
                        continue
                    if not MOTORS.is_torque_enabled(side):
                        # User released torque mid-homing — abort homing.
                        with self._lock:
                            arm.homing = False
                            arm.home_target = {}
                            arm.home_controller = None
                            self._invalidate_teleop_anchor(side, "homing aborted: torque off")
                        continue
                    present = MOTORS.read_positions(side)
                    step = self._homing_step(arm, present)
                    arm.home_last_command_error_deg = step.max_command_error_deg
                    arm.home_last_present_error_deg = step.max_present_error_deg
                    arm.home_last_worst_joint = step.worst_present_joint
                    try:
                        sent = MOTORS.send_action(side, step.command)
                        arm.last_sent_targets = dict(sent)
                        arm.last_commanded_targets = dict(sent)
                    except Exception as e:
                        log.warning("[%s] homing send failed: %s", side, e)
                    elapsed = time.monotonic() - arm.home_start_t
                    if time.monotonic() >= arm.home_next_progress_log_t:
                        log.info(
                            "[%s] homing progress %.1fs: command_error=%.2f deg "
                            "feedback_error=%.2f deg worst=%s reached=%s settled=%s",
                            side,
                            elapsed,
                            step.max_command_error_deg,
                            step.max_present_error_deg,
                            step.worst_present_joint or "none",
                            step.present_reached,
                            step.settled,
                        )
                        arm.home_next_progress_log_t = time.monotonic() + 2.0
                    if step.settled or elapsed > HOMING_TIMEOUT_S:
                        with self._lock:
                            arm.homing = False
                            arm.home_target = {}
                            arm.home_controller = None
                            if not step.settled:
                                self._invalidate_teleop_anchor(
                                    side,
                                    f"homing timed out (max error {step.max_present_error_deg:.1f} deg"
                                    + (f" at {step.worst_present_joint}" if step.worst_present_joint else "")
                                    + ")",
                                )
                        if step.settled:
                            log.info(
                                "[%s] homing complete in %.1fs; max feedback error %.2f deg",
                                side,
                                elapsed,
                                step.max_present_error_deg,
                            )
                        else:
                            log.warning(
                                "[%s] homing TIMED OUT after %.1fs; max feedback error %.2f deg at %s",
                                side,
                                elapsed,
                                step.max_present_error_deg,
                                step.worst_present_joint or "unknown joint",
                            )

                # Commands sent during this tick become the dataset action for
                # the same tick. Passive no-op arms use their present state.
                commanded_this_tick: dict[ArmSide, dict[str, float]] = {}
                expected_driven_this_tick: list[ArmSide] = []
                expected_held_this_tick: list[ArmSide] = []

                # Phase 2: command the active arm if engaged + calibrated. In
                # dual mode, run the same per-arm path for both connected arms.
                if dual_mode:
                    drive_sides = [s for s in ("left", "right") if s in connected]
                elif active is not None and active in connected:
                    drive_sides = [active]
                else:
                    drive_sides = []
                if not engaged or not drive_sides:
                    # Still record idle/passive ticks; they become no-op
                    # frames with action equal to present state.
                    self._record_frame_if_active(
                        commanded_this_tick=commanded_this_tick,
                        expected_driven_sides=expected_driven_this_tick,
                        expected_held_sides=expected_held_this_tick,
                    )
                    next_tick = self._sleep_until(next_tick)
                    continue
                for drive_side in drive_sides:
                    # Don't VR-drive an arm that's currently homing — homing already
                    # owns send_action above.
                    if self._arms[drive_side].homing:
                        continue
                    # Don't VR-drive an arm whose torque is released (user is
                    # hand-posing it).
                    if not MOTORS.is_torque_enabled(drive_side):
                        continue

                    arm = self._arms[drive_side]
                    with self._lock:
                        goal = arm.latest
                    controls_arm = arm.calibrated

                    # Watchdog: skip if last goal too stale (controller put down).
                    goal_age = now - goal.received_at if goal.has_data else 1e9
                    if not goal.has_data or goal_age > GOAL_SKIP_AGE_S:
                        disengaged = False
                        with self._lock:
                            if arm.stale_since is None:
                                arm.stale_since = now
                            if now - arm.stale_since > 1.0:
                                self._engaged = False
                                self._active_arm = None
                                self._dual_mode = False
                                disengaged = True
                                arm.robot_verify_test_active = False
                                self._restore_robot_verify_test_scale_if_idle()
                                self._restore_vr_control_inputs_if_idle()
                                log.warning("[%s] VR goals stale for >1s; auto-disengaged", drive_side)
                        if controls_arm and not disengaged:
                            expected_held_this_tick.append(drive_side)
                        continue
                    arm.stale_since = None
                    if goal.mode != "position":
                        # Grip release sends IDLE: hold last commanded targets; the
                        # per-tick joint cap below already prevents drift.
                        if controls_arm:
                            expected_held_this_tick.append(drive_side)
                        continue
                    if not controls_arm:
                        continue

                    # Build joint targets from the latest VR controller pose.
                    self._compute_targets_from_vr(drive_side, goal, scale)

                    # Per-tick joint clamp vs last sent (caps max joint velocity).
                    prefix = f"{drive_side}_arm_"
                    raw = arm.targets.to_dict_with_prefix(drive_side)
                    clamped: dict[str, float] = {}
                    for pj, val in raw.items():
                        cap = PER_TICK_DEG_CAPS.get(pj.removeprefix(prefix), 1.0)
                        prev = arm.last_sent_targets.get(pj, val)
                        delta = max(-cap, min(cap, val - prev))
                        clamped[pj] = prev + delta

                    # Live VR sends the deterministic cartesian-smoothed,
                    # joint-rate-capped target through the same style of
                    # command filter used by the references (XR Teleoperate
                    # filters solved joints; Open-Teach filters final poses).
                    # A present-position P-blend here feeds servo/bus noise
                    # back into every action and makes LeRobot recordings less
                    # faithful to user intent.
                    filtered = arm.command_filter.apply(prefix, clamped)
                    final = _apply_live_joint_deadband(
                        prefix,
                        filtered,
                        arm.last_sent_targets,
                    )
                    present_full: dict[str, float] = {}     # for the debug log

                    sent = MOTORS.send_action(drive_side, final)
                    arm.last_sent_targets = dict(sent)
                    arm.last_commanded_targets = dict(sent)
                    commanded_this_tick[drive_side] = dict(sent)
                    expected_driven_this_tick.append(drive_side)

                    # Debug: per-arm gripper trigger/target/sent/present log (1Hz).
                    self._debug_log_gripper(drive_side, goal, arm.targets,
                                             final, present_full, now)

                # Dataset capture: one frame per drive tick when recording is on.
                # Capture after motor writes so `action` is the command from this
                # tick, not the previous tick.
                self._record_frame_if_active(
                    commanded_this_tick=commanded_this_tick,
                    expected_driven_sides=expected_driven_this_tick,
                    expected_held_sides=expected_held_this_tick,
                )

            except Exception as e:
                log.exception("drive loop error: %s", e)
                with self._lock:
                    self._engaged = False
                    self._last_error = f"drive: {e}"

            next_tick = self._sleep_until(next_tick)

        log.info("drive loop exited")

    # ── guided calibration wizard ──────────────────────────────────────────
    def start_calibration(self, side: ArmSide) -> dict:
        """Begin a 3-vector motion-based calibration for one arm.

        State machine:
          idle → awaiting_anchor_fwd  → motioning_fwd
               → awaiting_anchor_up   → motioning_up
               → awaiting_anchor_left → motioning_left
               → idle (matrix + lateral verified)

        Steps:
          1 (forward): user moves hand in their forward direction → captures
            user-forward axis in VR world frame.
          2 (up):      user moves hand up → captures user-up axis.
            After steps 1+2, the 3×3 session matrix is built via Gram-Schmidt.
          3 (left):    user moves hand to THEIR left → captures a verification
            vector. We transform it through M; if the resulting robot-frame y
            is NEGATIVE (i.e., motion ended up on robot's right despite user
            moving left), `invert_lateral` gets set to True. Catches motor
            sign-convention mismatches that the forward+up math alone misses.

        Wrist motor polarity is NOT part of this wizard. It is hardware
        configuration loaded from
        `config/xlerobot.yaml` under `vr.wrist_motor_polarity.{left,right}.{flex,roll}`.
        Flip polarity there only if you rewire/remount a wrist motor.

        While calibration is active, the arm is force-unengaged so the robot
        doesn't drive during motion capture.
        """
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        with self._lock:
            if self._recording or self._recording_armed:
                raise RuntimeError("stop dataset recording before starting VR calibration")
            if not MOTORS.is_connected(side):
                raise RuntimeError(f"{side} arm not connected")
            if self._engaged and self._active_arm == side:
                self._engaged = False
                self._active_arm = None
            arm = self._arms[side]
            arm.cal_state = "awaiting_anchor_fwd"
            arm.cal_motion_acc = (0.0, 0.0, 0.0)
            arm.cal_captured_fwd  = None
            arm.cal_captured_up   = None
            arm.cal_captured_left = None
            arm.cal_anchor_quat_for_wrist = None
            arm.cal_wrist_release_quat = None
            arm.cal_wrist_verify_deg = 0.0
            arm.cal_wrist_pitch_verify_deg = 0.0
            arm.cal_wrist_roll_verify_deg = 0.0
            # Discard any prior empirical wrist axes; the user is re-running
            # the wizard and must capture both direct wrist-control axes again.
            arm.wrist_pitch_canonical = None
            arm.wrist_roll_canonical = None
            arm.cal_validation = {}
            arm.cal_last_fwd_m  = 0.0
            arm.cal_last_up_m   = 0.0
            arm.cal_last_left_m = 0.0
            arm.calibrated = False
            log.info("[%s] calibration started; awaiting grip-press for forward axis", side)
            return self.status()

    def cancel_calibration(self, side: ArmSide) -> dict:
        with self._lock:
            arm = self._arms[side]
            if arm.cal_state == "idle":
                return self.status()
            log.info("[%s] calibration cancelled", side)
            arm.cal_state = "idle"
            arm.cal_motion_acc = (0.0, 0.0, 0.0)
            arm.cal_captured_fwd = None
            arm.cal_captured_up = None
            arm.cal_captured_left = None
            arm.cal_anchor_quat_for_wrist = None
            arm.cal_wrist_release_quat = None
            arm.cal_wrist_verify_deg = 0.0
            arm.cal_wrist_pitch_verify_deg = 0.0
            arm.cal_wrist_roll_verify_deg = 0.0
            arm.cal_validation = {}
            # Restore matrix, lateral override, motor polarity, and any
            # persisted empirical wrist canonical from disk.
            self._restore_persisted_arm_config(side)
            return self.status()

    # ── home pose capture + go-to-home ─────────────────────────────────────
    def capture_home(self, side: Optional[ArmSide] = None) -> dict:
        """Read present joint positions for the connected arm(s) and write them
        to `config/xlerobot.yaml`'s `robot.home_pose:` block.

        - `side="left"` or `"right"`: only that arm's joints are written.
        - `side=None`: writes for every connected arm (existing values for
          disconnected arms in the YAML are preserved).
        """
        with self._lock:
            if self._recording:
                raise RuntimeError("stop dataset recording before capturing home")
            sides = [side] if side else list(MOTORS.connected_sides)
            if not sides:
                raise RuntimeError("connect an arm before capturing home")
            for s in sides:
                if not MOTORS.is_connected(s):
                    raise RuntimeError(f"{s} arm not connected")
            pose: dict[str, float] = {}
            for s in sides:
                pres = MOTORS.read_positions(s)
                prefix = f"{s}_arm_"
                for j in _motors.JOINTS_PER_ARM:
                    key = f"{prefix}{j}"
                    if key in pres:
                        pose[key] = float(pres[key])
            try:
                _home.write_home_pose(pose)
            except Exception as e:
                self._last_error = f"capture_home: {e}"
                log.exception("capture_home failed")
                raise
            log.info("home pose captured for sides=%s: %d joints written",
                     sides, len(pose))
        return self.status()

    def go_home(self, side: Optional[ArmSide] = None) -> dict:
        """Begin a slow, per-tick-clamped interpolation from the current pose to
        the saved home pose. Uses the same drive loop as VR teleop (same
        per-tick caps, same bus.send_action path) plus homing KP, so it's
        protected by all the existing safety guards. Forces the arm out of
        engage so the homing motion and VR teleop don't fight each other.
        """
        with self._lock:
            if self._recording:
                raise RuntimeError("stop dataset recording before homing")
            sides = [side] if side else list(MOTORS.connected_sides)
            if not sides:
                raise RuntimeError("connect an arm before homing")
            full_home = _home.read_home_pose()
            if not full_home:
                raise RuntimeError(
                    "no home pose saved — click 'Capture home' first while the "
                    "arm is in the desired starting pose."
                )
            for s in sides:
                if not MOTORS.is_connected(s):
                    raise RuntimeError(f"{s} arm not connected")
                target = {k: v for k, v in full_home.items() if k.startswith(f"{s}_arm_")}
                if not target:
                    raise RuntimeError(
                        f"no home pose saved for {s} arm — capture one first"
                    )
                arm = self._arms[s]
                present = MOTORS.read_positions(s)
                if not present:
                    raise RuntimeError(f"{s} arm position read failed")

                # Clamp saved home targets to this arm's calibrated bounds so we
                # don't queue unreachable goals that keep triggering low-level
                # safety clamps forever.
                bounded_target: dict[str, float] = {}
                bounds = MOTORS.bounds
                for pj, goal in target.items():
                    lo, hi = bounds.get(pj, (-180.0, 180.0))
                    safe_goal = max(lo, min(hi, float(goal)))
                    if abs(safe_goal - float(goal)) > 1e-6:
                        log.warning(
                            "[%s] home target %s out of calibrated bounds: %.3f -> %.3f (bounds %.3f..%.3f)",
                            s,
                            pj,
                            float(goal),
                            safe_goal,
                            lo,
                            hi,
                        )
                    bounded_target[pj] = safe_goal

                arm.home_target = bounded_target
                self._invalidate_teleop_anchor(s, "homing changed robot pose")
                arm.pending_rel_position = (0.0, 0.0, 0.0)
                # IMPORTANT: seed from present pose. If stale last_sent_targets
                # are reused (e.g. from an earlier VR drive), homing can
                # incorrectly mark itself converged immediately even though the
                # arm is far from home.
                arm.last_sent_targets = {pj: present.get(pj, v) for pj, v in bounded_target.items()}
                arm.last_commanded_targets = dict(arm.last_sent_targets)
                arm.command_filter.reset(arm.last_sent_targets)
                arm.home_controller = self._new_homing_controller(arm, present)
                arm.homing = True
                arm.home_start_t = time.monotonic()
                arm.home_last_command_error_deg = 0.0
                arm.home_last_present_error_deg = 0.0
                arm.home_last_worst_joint = ""
                arm.home_next_progress_log_t = arm.home_start_t + 1.0
                arm.stale_since = None
                # Homing owns the robot until it finishes. Clear all live-control
                # ownership, including dual mode, so a connected Quest client can
                # keep streaming without fighting the home trajectory.
                self._engaged = False
                self._active_arm = None
                self._dual_mode = False
                log.info("[%s] go_home started; %d joint targets queued", s, len(bounded_target))
        return self.status()

    def release_torque_for_posing(self, side: ArmSide) -> dict:
        """Disable torque on one arm so the user can hand-pose it. Forces the
        arm out of engage so VR drive won't fight the user. The drive loop
        skips arms with `torque_enabled=False`."""
        with self._lock:
            if self._recording:
                raise RuntimeError("stop dataset recording before releasing torque")
            if not MOTORS.is_connected(side):
                raise RuntimeError(f"{side} arm not connected")
            if self._active_arm == side:
                self._engaged = False
                self._active_arm = None
            arm = self._arms[side]
            if arm.homing:
                arm.homing = False
                arm.home_target = {}
                arm.home_controller = None
            MOTORS.release_torque_for_posing(side)
            # Invalidate the anchor — joint pose just changed unpredictably.
            self._invalidate_teleop_anchor(side, "torque released for hand posing")
            return self.status()

    def lock_torque(self, side: ArmSide) -> dict:
        """Re-enable torque on one arm at its current position (no snap-back).
        Caller should typically pair this with `capture_home(side)` if the
        intent was to pose-then-capture, but they're independent operations."""
        with self._lock:
            if self._recording:
                raise RuntimeError("stop dataset recording before locking torque")
            if not MOTORS.is_connected(side):
                raise RuntimeError(f"{side} arm not connected")
            MOTORS.lock_at_current(side)
            # Seed targets from the new pose so VR drive starts cleanly.
            self._seed_targets_from_present(side)
            self._invalidate_teleop_anchor(side, "torque locked at a new pose")
            return self.status()

    def cancel_homing(self, side: Optional[ArmSide] = None) -> dict:
        """Abort an in-progress homing motion. The arm freezes at its current
        pose (motor PID holds it)."""
        with self._lock:
            if self._recording:
                raise RuntimeError("stop dataset recording before cancelling homing")
            sides = [side] if side else ("left", "right")
            for s in sides:
                arm = self._arms[s]
                if arm.homing:
                    arm.homing = False
                    arm.home_target = {}
                    arm.home_controller = None
                    self._invalidate_teleop_anchor(s, "homing cancelled")
                    log.info("[%s] homing cancelled", s)
        return self.status()

    def wait_for_homing(self, sides: list[ArmSide], timeout_s: float = 10.0) -> bool:
        """Block until all `sides` finish homing (or timeout). Returns True if
        all finished, False if timeout hit. Caller MUST NOT hold `self._lock`."""
        deadline = time.monotonic() + timeout_s
        next_log = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self._lock:
                if all(not self._arms[s].homing for s in sides):
                    log.info("wait_for_homing complete for %s", sides)
                    return True
                if time.monotonic() >= next_log:
                    pending = []
                    for s in sides:
                        arm = self._arms[s]
                        if arm.homing:
                            pending.append(
                                f"{s}: feedback_error={arm.home_last_present_error_deg:.2f} deg "
                                f"worst={arm.home_last_worst_joint or 'unknown'}"
                            )
                    if pending:
                        log.info("wait_for_homing pending %.1fs remaining: %s",
                                 max(0.0, deadline - time.monotonic()), "; ".join(pending))
                    next_log = time.monotonic() + 2.0
            time.sleep(0.05)
        log.warning("wait_for_homing timeout after %.1fs; arms=%s", timeout_s, sides)
        return False

    def _advance_calibration(self, side: ArmSide, goal: _LatestGoal) -> None:
        """Three-vector wizard state machine. Called WITH `self._lock` held."""
        arm = self._arms[side]
        state = arm.cal_state
        mode = str(goal.mode)
        if state == "idle":
            return
        # Grip-press transitions (RESET goal)
        if mode == "reset":
            if state == "awaiting_anchor_fwd":
                arm.cal_motion_acc = (0.0, 0.0, 0.0)
                arm.cal_state = "motioning_fwd"
                log.info("[%s] cal: anchor for forward captured; "
                         "move hand FORWARD ~10 cm, then release grip", side)
            elif state == "awaiting_anchor_up":
                arm.cal_motion_acc = (0.0, 0.0, 0.0)
                arm.cal_state = "motioning_up"
                log.info("[%s] cal: anchor for up captured; "
                         "move hand UP ~10 cm, then release grip", side)
            elif state == "awaiting_anchor_left":
                arm.cal_motion_acc = (0.0, 0.0, 0.0)
                arm.cal_state = "motioning_left"
                log.info("[%s] cal: anchor for left captured; "
                         "move hand LEFT ~10 cm, then release grip", side)
            elif state in ("awaiting_anchor_wrist_verify", "awaiting_anchor_wrist_pitch"):
                anchor_q = goal.rotation_quat
                if anchor_q is None:
                    log.warning("[%s] cal: wrist-pitch reset has no controller quaternion; release grip and try again", side)
                    return
                arm.cal_anchor_quat_for_wrist = anchor_q
                arm.cal_wrist_release_quat = None
                arm.cal_wrist_verify_deg = 0.0
                arm.cal_wrist_pitch_verify_deg = 0.0
                arm.cal_state = "motioning_wrist_pitch"
                log.info(
                    "[%s] cal: wrist-pitch anchor captured. KEEP GRIP HELD and pitch "
                    "your wrist clearly UP ~20-45°, then release.",
                    side,
                )
            elif state == "awaiting_anchor_wrist_roll":
                anchor_q = goal.rotation_quat
                if anchor_q is None:
                    log.warning("[%s] cal: wrist-roll reset has no controller quaternion; release grip and try again", side)
                    return
                arm.cal_anchor_quat_for_wrist = anchor_q
                arm.cal_wrist_release_quat = None
                arm.cal_wrist_verify_deg = 0.0
                arm.cal_wrist_roll_verify_deg = 0.0
                arm.cal_state = "motioning_wrist_roll"
                log.info(
                    "[%s] cal: wrist-roll anchor captured. KEEP GRIP HELD and roll "
                    "your wrist clearly RIGHT ~20-45°, then release.",
                    side,
                )
            return
        # Accumulate per-frame position deltas while moving.
        if mode == "position" and state in (
                "motioning_fwd", "motioning_up", "motioning_left"):
            dp = goal.rel_position
            arm.cal_motion_acc = (
                arm.cal_motion_acc[0] + float(dp[0]),
                arm.cal_motion_acc[1] + float(dp[1]),
                arm.cal_motion_acc[2] + float(dp[2]),
            )
            return
        if mode == "position" and state in ("motioning_wrist_verify", "motioning_wrist_pitch", "motioning_wrist_roll"):
            release_q = goal.rotation_quat
            anchor_q = arm.cal_anchor_quat_for_wrist
            if release_q is not None:
                arm.cal_wrist_release_quat = release_q
            if anchor_q is not None and release_q is not None:
                arm.cal_wrist_verify_deg = _wrist_rotation_deg_since_anchor(
                    anchor_q, release_q,
                )
                if state in ("motioning_wrist_verify", "motioning_wrist_pitch"):
                    arm.cal_wrist_pitch_verify_deg = arm.cal_wrist_verify_deg
                else:
                    arm.cal_wrist_roll_verify_deg = arm.cal_wrist_verify_deg
            return
        # Grip-release transitions (IDLE goal)
        if mode == "idle":
            if state == "motioning_fwd":
                mag = math.sqrt(sum(v * v for v in arm.cal_motion_acc))
                if mag < CALIBRATION_MIN_MOTION_M:
                    log.warning("[%s] cal: forward motion too small (%.1f cm); "
                                "still awaiting forward — re-grip and move further",
                                side, mag * 100)
                    arm.cal_state = "awaiting_anchor_fwd"
                    return
                arm.cal_captured_fwd = arm.cal_motion_acc
                arm.cal_last_fwd_m = mag
                arm.cal_state = "awaiting_anchor_up"
                log.info("[%s] cal: forward axis captured (%.1f cm); "
                         "press grip and move hand UP ~10 cm to capture vertical axis",
                         side, mag * 100)
            elif state == "motioning_up":
                mag = math.sqrt(sum(v * v for v in arm.cal_motion_acc))
                if mag < CALIBRATION_MIN_MOTION_M:
                    log.warning("[%s] cal: up motion too small (%.1f cm); "
                                "re-grip and move further", side, mag * 100)
                    arm.cal_state = "awaiting_anchor_up"
                    return
                arm.cal_captured_up = arm.cal_motion_acc
                arm.cal_last_up_m = mag
                arm.cal_state = "awaiting_anchor_left"
                log.info("[%s] cal: up axis captured (%.1f cm). "
                         "Now press grip and move hand LEFT ~10 cm; the movement frame "
                         "will be built only after all three motions pass validation.",
                         side, mag * 100)
            elif state == "motioning_left":
                mag = math.sqrt(sum(v * v for v in arm.cal_motion_acc))
                if mag < CALIBRATION_MIN_MOTION_M:
                    log.warning("[%s] cal: left motion too small (%.1f cm); "
                                "re-grip and move further", side, mag * 100)
                    arm.cal_state = "awaiting_anchor_left"
                    return
                arm.cal_captured_left = arm.cal_motion_acc
                arm.cal_last_left_m = mag
                self._finalize_translation_calibration(side)
            elif state in ("motioning_wrist_verify", "motioning_wrist_pitch"):
                self._finalize_wrist_axis(side, goal, axis="pitch")
            elif state == "motioning_wrist_roll":
                self._finalize_wrist_axis(side, goal, axis="roll")

    def _finalize_translation_calibration(self, side: ArmSide) -> None:
        """Rebuild the session matrix from all three captured motions via
        Procrustes/Kabsch, verify the lateral axis sign, then transition to the
        wrist-verify step. Called WITH `self._lock` held."""
        arm = self._arms[side]
        f = arm.cal_captured_fwd
        u = arm.cal_captured_up
        l = arm.cal_captured_left
        if f is None or u is None or l is None:
            log.warning("[%s] cal: finalize called without all three vectors", side)
            arm.cal_state = "idle"
            return

        # Rebuild M using ALL three captured motions. This averages out noise
        # in any single motion (small off-axis drift, hand jitter) far better
        # than the preliminary 2-motion matrix. The final solve is strict: it
        # must not silently fall back to a weaker frame.
        try:
            arm.session_vr_to_robot, arm.cal_confidence = (
                _compute_session_frame_from_three_motions(f, u, l)
            )
        except ValueError as exc:
            arm.session_vr_to_robot = _VR_TO_ROBOT.copy()
            arm.cal_confidence = "poor"
            arm.cal_validation = {"error": str(exc)}
            arm.cal_captured_fwd = None
            arm.cal_captured_up = None
            arm.cal_captured_left = None
            arm.cal_motion_acc = (0.0, 0.0, 0.0)
            arm.cal_state = "awaiting_anchor_fwd"
            self._last_error = f"{side} VR calibration rejected: {exc}"
            log.warning("[%s] %s", side, self._last_error)
            return

        # Verify lateral: transform the captured left-motion through M to robot
        # frame. y > 0 = "user-left → robot-left" (correct, no invert).
        # y < 0 = mirrored (set invert_lateral). BUT if the user has manually
        # set `invert_lateral_<side>` in config/xlerobot.yaml, that's an
        # OVERRIDE — typically for physically mirror-mounted motors that the
        # math can't see — and we skip the auto-decision.
        l_vec = _np.array(l, dtype=float)
        l_robot = arm.session_vr_to_robot @ l_vec
        if arm.invert_lateral_override:
            verdict = (f"OVERRIDDEN by YAML (invert_lateral_{side} explicitly set "
                       f"to {arm.invert_lateral}) — wizard's auto-decision skipped")
        else:
            arm.invert_lateral = bool(l_robot[1] < 0)
            verdict = ("INVERTED (set invert_lateral=True)" if arm.invert_lateral
                       else "OK (invert_lateral=False)")
        f_robot = arm.session_vr_to_robot @ _np.array(f, dtype=float)
        u_robot = arm.session_vr_to_robot @ _np.array(u, dtype=float)
        arm.cal_validation = {
            "forward_robot_delta": [float(v) for v in f_robot],
            "up_robot_delta": [float(v) for v in u_robot],
            "left_robot_delta": [float(v) for v in l_robot],
            "lateral_verdict": verdict,
        }
        log.info(
            "[%s] translation calibration finalized (Procrustes M, conf=%s) — "
            "forward=%s up=%s left=%s → robot deltas fwd=%s up=%s left=%s → lateral %s\n"
            "session matrix:\n%s",
            side, arm.cal_confidence,
            tuple(f"{v:.3f}" for v in f),
            tuple(f"{v:.3f}" for v in u),
            tuple(f"{v:.3f}" for v in l),
            tuple(f"{v:.3f}" for v in f_robot),
            tuple(f"{v:.3f}" for v in u_robot),
            tuple(f"{v:.3f}" for v in l_robot),
            verdict,
            arm.session_vr_to_robot.round(3),
        )

        # Hand off to required wrist calibration steps. Live wrist control uses
        # these captured controller-local axes directly.
        arm.cal_anchor_quat_for_wrist = None
        arm.cal_state = "awaiting_anchor_wrist_pitch"
        log.info(
            "[%s] Translation done. Next: squeeze grip, pitch wrist UP, release, "
            "then capture roll RIGHT for direct wrist control.",
            side,
        )

    def _finalize_wrist_axis(self, side: ArmSide, goal: _LatestGoal, *, axis: str) -> None:
        """Capture one empirical wrist axis in controller anchor-local frame."""
        if axis not in ("pitch", "roll"):
            raise ValueError(f"axis must be pitch|roll, got {axis!r}")
        arm = self._arms[side]
        anchor_q = arm.cal_anchor_quat_for_wrist
        release_q = goal.rotation_quat or arm.cal_wrist_release_quat
        if anchor_q is None or release_q is None:
            log.warning(
                "[%s] cal: wrist-%s missing controller quaternion (anchor=%s, "
                "release=%s); re-grip and rotate wrist again.",
                side, axis,
                anchor_q is not None, release_q is not None,
            )
            arm.cal_state = f"awaiting_anchor_wrist_{axis}"
            return

        canonical, mag = _wrist_rotvec_since_anchor(anchor_q, release_q)
        deg = math.degrees(mag)
        arm.cal_wrist_verify_deg = deg
        if axis == "pitch":
            arm.cal_wrist_pitch_verify_deg = deg
        else:
            arm.cal_wrist_roll_verify_deg = deg
        if mag < math.radians(WRIST_VERIFY_MIN_DEG):
            log.warning(
                "[%s] cal: wrist-%s motion too small (%.1f°, need ≥%.0f°). "
                "Re-grip and rotate wrist further.",
                side, axis, deg, WRIST_VERIFY_MIN_DEG,
            )
            arm.cal_state = f"awaiting_anchor_wrist_{axis}"
            arm.cal_anchor_quat_for_wrist = None
            arm.cal_wrist_release_quat = None
            arm.cal_wrist_verify_deg = 0.0
            if axis == "pitch":
                arm.cal_wrist_pitch_verify_deg = 0.0
            else:
                arm.cal_wrist_roll_verify_deg = 0.0
            return

        if axis == "pitch":
            arm.wrist_pitch_canonical = tuple(float(v) for v in canonical)
            arm.cal_validation["wrist_pitch_anchor_local"] = list(arm.wrist_pitch_canonical)
            arm.cal_validation["wrist_pitch_magnitude_deg"] = deg
            analytical = _np.array([1.0, 0.0, 0.0])
            flipped_to_default = False
        else:
            raw_canonical = _np.asarray(canonical, dtype=float)
            canonical, flipped_to_default = _canonicalize_empirical_roll_axis(raw_canonical)
            arm.cal_validation["wrist_roll_raw_anchor_local"] = [float(v) for v in raw_canonical]
            arm.cal_validation["wrist_roll_flipped_to_main_sign"] = bool(flipped_to_default)
            arm.wrist_roll_canonical = tuple(float(v) for v in canonical)
            arm.cal_validation["wrist_roll_anchor_local"] = list(arm.wrist_roll_canonical)
            arm.cal_validation["wrist_roll_magnitude_deg"] = deg
            analytical = _np.array([0.0, 0.0, -1.0])

        cos = float(_np.dot(_np.asarray(canonical), analytical))
        cos_clamped = max(-1.0, min(1.0, cos))
        off_deg = math.degrees(math.acos(cos_clamped))
        arm.cal_validation[f"wrist_{axis}_off_from_quest_default_deg"] = off_deg
        msg = "matches" if cos >= 0.85 else "differs from"
        log.info(
            "[%s] wrist-%s captured: canonical=%s, %.1f° motion, %s Quest default by %.0f°%s",
            side, axis, tuple(f"{v:+.3f}" for v in canonical), deg, msg, off_deg,
            " (flipped to main roll-right sign)" if flipped_to_default else "",
        )

        arm.cal_anchor_quat_for_wrist = None
        arm.cal_wrist_release_quat = None
        arm.cal_wrist_verify_deg = 0.0
        if axis == "pitch":
            arm.cal_state = "awaiting_anchor_wrist_roll"
            log.info(
                "[%s] wrist pitch captured. Now squeeze grip and roll wrist RIGHT "
                "~20-45°, then release.",
                side,
            )
        else:
            self._persist_final_calibration(side)

    def skip_wrist_verify(self, side: ArmSide) -> dict:
        """Reject wrist-calibration skipping; direct wrist axes are required."""
        with self._lock:
            arm = self._arms[side]
            if arm.cal_state not in (
                "awaiting_anchor_wrist_verify", "motioning_wrist_verify",
                "awaiting_anchor_wrist_pitch", "motioning_wrist_pitch",
                "awaiting_anchor_wrist_roll", "motioning_wrist_roll",
            ):
                return self.status()
            self._last_error = (
                f"{side} wrist pitch and roll calibration are required for direct VR wrist control"
            )
            log.warning("[%s] %s", side, self._last_error)
        return self.status()

    # ── robot-verified calibration refinement ─────────────────────────────
    def start_robot_verification(self, side: ArmSide, release_torque: bool = False) -> dict:
        """Start the additive robot-verified calibration refinement.

        This keeps the existing VR-only matrix as the base prior. The user then
        captures paired robot EE deltas and VR controller deltas. `solve` fits a
        small correction + translation scale on top of that base.
        """
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        with self._lock:
            if not MOTORS.is_connected(side):
                raise RuntimeError(f"{side} arm not connected")
            if self._recording or self._recording_armed:
                raise RuntimeError("stop dataset recording before robot verification")
            arm = self._arms[side]
            self._require_urdf_kinematics(side, "start robot verification")
            if arm.cal_state != "idle":
                raise RuntimeError("finish or cancel the VR calibration wizard before robot verification")
            for other in self._arms.values():
                other.robot_verify_test_active = False
            self._restore_robot_verify_test_scale_if_idle()
            arm.robot_verify_state = "collecting"
            arm.robot_verify_samples = []
            arm.robot_verify_robot_start = None
            arm.robot_verify_robot_end = None
            arm.robot_verify_vr_start = None
            arm.robot_verify_vr_delta_accum = (0.0, 0.0, 0.0)
            arm.robot_verify_label = ""
            arm.robot_verify_fit_error_cm = None
            arm.robot_verify_sample_residuals = []
            arm.robot_verify_quality = "collecting"
            arm.robot_verified_at = None
            arm.robot_verify_test_active = False
            arm.robot_verify_test_completed = False
            arm.base_vr_direction_matrix = (
                arm.base_vr_direction_matrix.copy()
                if arm.base_vr_direction_matrix is not None
                else arm.session_vr_to_robot.copy()
            )
            arm.session_vr_to_robot = arm.base_vr_direction_matrix.copy()
            arm.translation_vr_to_robot = None
            arm.translation_scale = 1.0
            self.set_vr_control_inputs_enabled(False)
            log.info("[%s] robot verification started; base matrix:\n%s",
                     side, arm.base_vr_direction_matrix.round(3))
            if release_torque:
                # Keep the verification flow self-contained in the dashboard:
                # start = release for hand-posing; robot-end capture locks.
                self.release_torque_for_posing(side)
        return self.status()

    def cancel_robot_verification(self, side: ArmSide) -> dict:
        with self._lock:
            arm = self._arms[side]
            arm.robot_verify_state = "idle"
            arm.robot_verify_robot_start = None
            arm.robot_verify_robot_end = None
            arm.robot_verify_vr_start = None
            arm.robot_verify_vr_delta_accum = (0.0, 0.0, 0.0)
            arm.robot_verify_label = ""
            arm.robot_verify_test_active = False
            self._restore_persisted_arm_config(side)
            arm.robot_verify_state = "idle"
            arm.robot_verify_robot_start = None
            arm.robot_verify_robot_end = None
            arm.robot_verify_vr_start = None
            arm.robot_verify_vr_delta_accum = (0.0, 0.0, 0.0)
            arm.robot_verify_label = ""
            arm.robot_verify_test_active = False
            self._restore_vr_control_inputs_if_idle()
            log.info("[%s] robot verification cancelled", side)
        return self.status()

    def capture_robot_verification_pose(self, side: ArmSide, point: str, label: str = "") -> dict:
        """Capture robot EE start/end pose for the current verification sample."""
        if point not in ("start", "end"):
            raise ValueError("point must be 'start' or 'end'")
        with self._lock:
            arm = self._arms[side]
            if arm.robot_verify_state == "idle":
                raise RuntimeError("start robot verification first")
            label_text = self._normalize_robot_verification_label(label or arm.robot_verify_label)
            if label_text:
                arm.robot_verify_label = label_text
            T = self._current_ee_transform(side)
            pos = (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
            if point == "start":
                arm.robot_verify_robot_start = pos
                arm.robot_verify_robot_end = None
                arm.robot_verify_vr_start = None
                arm.robot_verify_vr_delta_accum = (0.0, 0.0, 0.0)
                arm.robot_verify_state = "robot_start_captured"
            else:
                if arm.robot_verify_robot_start is None:
                    raise RuntimeError("capture robot start before robot end")
                arm.robot_verify_robot_end = pos
                arm.robot_verify_vr_start = None
                arm.robot_verify_vr_delta_accum = (0.0, 0.0, 0.0)
                arm.robot_verify_state = "robot_end_captured"
                if not MOTORS.is_torque_enabled(side):
                    # The user has just placed the robot at the end pose. Lock
                    # there so the arm does not sag while they perform the VR
                    # start/end motion for this sample.
                    MOTORS.lock_at_current(side)
                    self._seed_targets_from_present(side)
            log.info("[%s] robot verification %s pose (%s): %s", side, point, arm.robot_verify_label, pos)
        return self.status()

    def capture_robot_verification_vr(self, side: ArmSide, point: str, label: str = "") -> dict:
        """Capture VR controller start/end pose. End capture creates one sample."""
        if point not in ("start", "end"):
            raise ValueError("point must be 'start' or 'end'")
        with self._lock:
            arm = self._arms[side]
            if arm.robot_verify_state == "idle":
                raise RuntimeError("start robot verification first")
            if arm.robot_verify_robot_start is None or arm.robot_verify_robot_end is None:
                raise RuntimeError("capture robot start and end before VR motion")
            if not arm.latest.has_data or arm.latest.controller_position is None:
                raise RuntimeError(
                    "no VR controller pose available; open the Quest page, move the selected controller, "
                    "and hold its grip briefly so pose packets stream"
                )
            age_s = time.time() - arm.latest.received_at
            if age_s > 2.0:
                raise RuntimeError(
                    f"VR controller pose is stale ({age_s:.1f}s old); move the selected controller "
                    "and hold its grip briefly, then click Capture VR again"
                )
            if arm.latest.mode not in ("reset", "position"):
                raise RuntimeError("hold grip while capturing VR start/end for one continuous sample")
            pos = tuple(float(v) for v in arm.latest.controller_position)
            if point == "start":
                arm.robot_verify_vr_start = pos
                arm.robot_verify_vr_delta_accum = (0.0, 0.0, 0.0)
                arm.robot_verify_label = (
                    self._normalize_robot_verification_label(label or arm.robot_verify_label)
                    or f"sample-{len(arm.robot_verify_samples)+1}"
                )
                arm.robot_verify_state = "vr_start_captured"
                log.info("[%s] robot verification VR start: %s", side, pos)
                return self.status()

            if arm.robot_verify_vr_start is None:
                raise RuntimeError("capture VR start before VR end")
            if arm.latest.mode != "position":
                raise RuntimeError("keep grip held while moving, then capture VR end before releasing")
            robot_start = _np.array(arm.robot_verify_robot_start, dtype=float)
            robot_end = _np.array(arm.robot_verify_robot_end, dtype=float)
            vr_start = _np.array(arm.robot_verify_vr_start, dtype=float)
            vr_end = _np.array(pos, dtype=float)
            robot_delta = robot_end - robot_start
            vr_delta = vr_end - vr_start
            if not _np.all(_np.isfinite(vr_delta)):
                raise RuntimeError("VR controller pose is invalid; retry this verification sample")
            robot_mag = float(_np.linalg.norm(robot_delta))
            vr_mag = float(_np.linalg.norm(vr_delta))
            if robot_mag < ROBOT_VERIFY_MIN_MOTION_M:
                raise RuntimeError(
                    f"robot motion too small ({robot_mag*100:.1f} cm); move at least "
                    f"{ROBOT_VERIFY_MIN_MOTION_M*100:.1f} cm"
                )
            if vr_mag < ROBOT_VERIFY_MIN_MOTION_M:
                raise RuntimeError(
                    f"VR controller motion too small ({vr_mag*100:.1f} cm); hold grip while "
                    f"moving at least {ROBOT_VERIFY_MIN_MOTION_M*100:.1f} cm"
                )
            label_text = (
                self._normalize_robot_verification_label(label or arm.robot_verify_label)
                or f"sample-{len(arm.robot_verify_samples)+1}"
            )
            sample = {
                "label": label_text,
                "robot_start": [float(v) for v in robot_start],
                "robot_end": [float(v) for v in robot_end],
                "robot_delta": [float(v) for v in robot_delta],
                "vr_start": [float(v) for v in vr_start],
                "vr_end": [float(v) for v in vr_end],
                "vr_delta": [float(v) for v in vr_delta],
                "vr_delta_source": "absolute_controller_pose",
                "robot_motion_m": robot_mag,
                "vr_motion_m": vr_mag,
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            arm.robot_verify_samples.append(sample)
            arm.robot_verify_robot_start = None
            arm.robot_verify_robot_end = None
            arm.robot_verify_vr_start = None
            arm.robot_verify_vr_delta_accum = (0.0, 0.0, 0.0)
            arm.robot_verify_label = ""
            arm.robot_verify_state = "collecting"
            log.info(
                "[%s] robot verification sample %d captured (%s): robot %.1f cm, VR %.1f cm; torque held",
                side, len(arm.robot_verify_samples), label_text, robot_mag * 100, vr_mag * 100,
            )
        return self.status()

    def _reset_robot_verification_vr_capture(self, arm: _PerArm, *, reason: str) -> None:
        """Clear an in-progress VR sample but keep robot start/end for retry.

        Robot verification samples must be one continuous grip-held motion:
        A/X captures VR start, the user moves while holding grip, then A/X
        captures VR end before grip release. Releasing grip cancels only the VR
        half of the current sample so stale deltas cannot affect the next try.
        Caller must hold `self._lock`.
        """
        arm.robot_verify_vr_start = None
        arm.robot_verify_vr_delta_accum = (0.0, 0.0, 0.0)
        if arm.robot_verify_robot_start is not None and arm.robot_verify_robot_end is not None:
            arm.robot_verify_state = "robot_end_captured"
        elif arm.robot_verify_robot_start is not None:
            arm.robot_verify_state = "robot_start_captured"
        elif arm.robot_verify_state != "idle":
            arm.robot_verify_state = "collecting"
        log.info("[%s] robot verification VR capture reset: %s", arm.side, reason)

    def discard_last_robot_verification_sample(self, side: ArmSide) -> dict:
        with self._lock:
            arm = self._arms[side]
            if arm.robot_verify_samples:
                removed = arm.robot_verify_samples.pop()
                log.info("[%s] discarded robot verification sample %s", side, removed.get("label"))
            arm.robot_verify_fit_error_cm = None
            arm.robot_verify_sample_residuals = []
            arm.robot_verify_quality = "collecting" if arm.robot_verify_state != "idle" else "unverified"
        return self.status()

    def solve_robot_verification(self, side: ArmSide) -> dict:
        with self._lock:
            arm = self._arms[side]
            self._require_urdf_kinematics(side, "solve robot verification")
            missing_labels = self._missing_robot_verification_labels(arm)
            if missing_labels:
                raise RuntimeError(
                    f"capture required verification directions before solving: {', '.join(missing_labels)}"
                )
            final_M, translation_M, scale, fit_error_cm, quality, residuals = (
                self._solve_robot_verified_calibration(arm)
            )
            if quality != "good":
                arm.robot_verify_fit_error_cm = fit_error_cm
                arm.robot_verify_sample_residuals = residuals
                arm.robot_verify_quality = quality
                arm.translation_vr_to_robot = None
                arm.translation_scale = 1.0
                arm.robot_verify_test_completed = False
                hint = self._robot_verification_residual_hint(residuals)
                self._last_error = (
                    f"{side} robot verification residual {fit_error_cm:.1f} cm is too high "
                    f"(must be <= {ROBOT_VERIFY_PASS_ERROR_CM:.1f} cm RMS); "
                    f"{hint or 'recapture all six directions near the grasp workspace'}"
                )
                raise RuntimeError(self._last_error)

            base = (
                arm.base_vr_direction_matrix.copy()
                if arm.base_vr_direction_matrix is not None
                else arm.session_vr_to_robot.copy()
            )
            arm.session_vr_to_robot = base.copy()
            arm.base_vr_direction_matrix = base.copy()
            arm.translation_vr_to_robot = translation_M
            arm.translation_scale = scale
            arm.robot_verify_fit_error_cm = fit_error_cm
            arm.robot_verify_sample_residuals = residuals
            arm.robot_verify_quality = quality
            arm.robot_verified_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            arm.robot_verify_state = "idle"
            arm.robot_verify_test_active = False
            arm.robot_verify_test_completed = False
            self._restore_vr_control_inputs_if_idle()
            _vrcal.write_robot_verification_for_arm(
                side,
                base_matrix=base,
                verified_matrix=final_M,
                translation_matrix=translation_M,
                translation_scale=scale,
                fit_error_cm=fit_error_cm,
                sample_residuals=residuals,
                samples=arm.robot_verify_samples,
                quality=quality,
                coordinate_frame=self._native_quest.coordinate_frame,
            )
            log.info(
                "[%s] robot verification solved: quality=%s fit=%.2f cm scale=%.3f "
                "translation_rotation=\n%s\norientation_frame_kept=\n%s",
                side, quality, fit_error_cm, scale, final_M.round(3), base.round(3),
            )
        return self.status()

    def start_robot_verification_test(
        self,
        side: ArmSide,
        scale: float = ROBOT_VERIFY_TEST_SCALE,
    ) -> dict:
        """Start a low-scale live test of the verified mapping.

        This mirrors normal teleop: current robot EE pose and current controller
        pose become a temporary neutral origin. Quest face buttons and automatic
        grip-reset anchoring are disabled so the test cannot start recording or
        unexpectedly re-anchor while the user is checking motion.
        """
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        with self._lock:
            if not MOTORS.is_connected(side):
                raise RuntimeError(f"{side} arm not connected")
            if self._recording or self._recording_armed:
                raise RuntimeError("stop dataset recording before calibration test mode")
            arm = self._arms[side]
            if arm.robot_verify_quality != "good":
                raise RuntimeError("solve robot verification before low-scale test")
            if arm.latest.controller_position is None or arm.latest.rotation_quat is None:
                raise RuntimeError(
                    "no fresh controller pose; open the Quest page and move the selected controller"
                )
            age_s = time.time() - arm.latest.received_at if arm.latest.has_data else 1e9
            if age_s > 2.0:
                raise RuntimeError(
                    f"controller pose is stale ({age_s:.1f}s); move the selected controller first"
                )
            if not MOTORS.is_torque_enabled(side):
                MOTORS.lock_at_current(side)
                self._seed_targets_from_present(side)

            if not any(other.robot_verify_test_active for other in self._arms.values()):
                self._scale_before_robot_verify_test = self._scale
            for other in self._arms.values():
                other.robot_verify_test_active = False
            self._recording = False
            self._active_arm = side
            self._dual_mode = False
            self._scale = max(0.05, min(0.35, float(scale)))
            arm.robot_verify_test_scale = self._scale
            arm.robot_verify_test_active = True
            arm.robot_verify_test_completed = False
            try:
                _vrcal.set_robot_verification_test_completed(side, False)
            except Exception as e:
                log.warning("[%s] could not persist low-scale test reset: %s", side, e)
            self._controller_buttons_enabled = False
            self._teleop_reset_anchors_enabled = False
            for each in self._arms.values():
                each.prev_buttons = {}
                each.reset_pending = False
            self._capture_anchor(side)
            self._engaged = True
            log.info("[%s] robot verification low-scale test started (scale=%.2f)", side, self._scale)
        return self.status()

    def stop_robot_verification_test(self, side: ArmSide) -> dict:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        with self._lock:
            arm = self._arms[side]
            was_active = arm.robot_verify_test_active
            arm.robot_verify_test_active = False
            if was_active and arm.robot_verify_quality == "good":
                arm.robot_verify_test_completed = True
                try:
                    _vrcal.set_robot_verification_test_completed(side, True)
                except Exception as e:
                    log.warning("[%s] could not persist low-scale test completion: %s", side, e)
            if self._active_arm == side:
                self._engaged = False
                self._active_arm = None
                self._dual_mode = False
            self._restore_robot_verify_test_scale_if_idle()
            self._restore_vr_control_inputs_if_idle()
            log.info("[%s] robot verification low-scale test stopped", side)
        return self.status()

    def _solve_robot_verified_calibration(
        self, arm: _PerArm,
    ) -> tuple[_np.ndarray, _np.ndarray, float, float, str, list[dict[str, Any]]]:
        samples = list(arm.robot_verify_samples)
        if len(samples) < ROBOT_VERIFY_MIN_SAMPLES:
            raise RuntimeError(
                f"need at least {ROBOT_VERIFY_MIN_SAMPLES} robot/VR samples; "
                f"only {len(samples)} captured"
            )
        missing_labels = self._missing_robot_verification_labels(arm)
        if missing_labels:
            raise RuntimeError(
                f"verification samples missing required directions: {', '.join(missing_labels)}"
            )
        vr_deltas: list[_np.ndarray] = []
        robot_deltas: list[_np.ndarray] = []
        valid_samples: list[dict[str, Any]] = []
        valid_labels: list[str] = []
        for idx, sample in enumerate(samples):
            vr = _np.array(sample.get("vr_delta"), dtype=float)
            robot = _np.array(sample.get("robot_delta"), dtype=float)
            if vr.shape != (3,) or robot.shape != (3,):
                continue
            if not _np.all(_np.isfinite(vr)) or not _np.all(_np.isfinite(robot)):
                continue
            vn = float(_np.linalg.norm(vr))
            rn = float(_np.linalg.norm(robot))
            if vn < ROBOT_VERIFY_MIN_MOTION_M or rn < ROBOT_VERIFY_MIN_MOTION_M:
                continue
            vr_deltas.append(vr)
            robot_deltas.append(robot)
            valid_samples.append(sample)
            valid_labels.append(self._effective_robot_verification_label(samples, idx))
        if len(vr_deltas) < ROBOT_VERIFY_MIN_SAMPLES:
            raise RuntimeError(
                f"need {ROBOT_VERIFY_MIN_SAMPLES} valid samples after filtering; got {len(vr_deltas)}"
            )
        captured_valid = set(valid_labels)
        missing_valid = [
            label for label in ROBOT_VERIFY_REQUIRED_LABELS
            if label not in captured_valid
        ]
        if missing_valid:
            raise RuntimeError(
                "valid verification samples missing required directions: "
                + ", ".join(missing_valid)
            )

        V = _np.stack(vr_deltas, axis=0)
        R = _np.stack(robot_deltas, axis=0)
        if _np.linalg.matrix_rank(V, tol=0.015) < 3 or _np.linalg.matrix_rank(R, tol=0.015) < 3:
            raise RuntimeError(
                "verification samples do not span 3D motion; capture forward/back, left/right, and up/down"
            )

        # Solve the full VR-delta -> robot-delta linear map for position. Runtime
        # uses this matrix only after safety checks in _ArmMovementMapper; the
        # scalar stage-1 prediction below remains the quality gate so a noisy
        # 3D solve cannot hide a bad calibration sample.
        translation_effective_T, *_ = _np.linalg.lstsq(V, R, rcond=None)
        translation_effective = translation_effective_T.T
        if not _np.all(_np.isfinite(translation_effective)):
            raise RuntimeError("verification solve produced non-finite translation matrix")

        # Extract the nearest rotation for diagnostics/backward compatibility in
        # the YAML. Runtime wrist/orientation must keep using the stage-1 VR
        # direction frame; the full linear matrix below is position-only.
        final_M = _project_to_rotation_matrix(translation_effective)
        translation_M = translation_effective

        base_M = self._effective_translation_matrix(arm, arm.session_vr_to_robot)
        base_predictions = [base_M @ vr for vr in vr_deltas]
        denom = float(sum(_np.dot(pred, pred) for pred in base_predictions))
        if denom <= 1e-12:
            raise RuntimeError("verification solve has no usable calibrated-frame motion")
        scale_raw = float(sum(_np.dot(pred, robot) for pred, robot in zip(base_predictions, robot_deltas)) / denom)
        scale = scale_raw
        scale = max(0.05, min(5.0, scale))

        residual_norms = []
        residual_details: list[dict[str, Any]] = []
        for idx, (base_pred, robot, vr_delta, sample, label) in enumerate(
            zip(base_predictions, robot_deltas, vr_deltas, valid_samples, valid_labels)
        ):
            pred = scale * base_pred
            residual_m = float(_np.linalg.norm(pred - robot))
            residual_norms.append(residual_m)
            pred_norm = float(_np.linalg.norm(pred))
            robot_norm = float(_np.linalg.norm(robot))
            if pred_norm > 1e-9 and robot_norm > 1e-9:
                cos = float(_np.dot(pred, robot) / (pred_norm * robot_norm))
                cos = max(-1.0, min(1.0, cos))
                direction_error_deg = math.degrees(math.acos(cos))
            else:
                direction_error_deg = None
            residual_details.append({
                "index": idx,
                "label": label or f"sample-{idx + 1}",
                "residual_cm": residual_m * 100.0,
                "direction_error_deg": direction_error_deg,
                "robot_motion_cm": robot_norm * 100.0,
                "vr_motion_cm": float(_np.linalg.norm(vr_delta)) * 100.0,
                "target_robot_delta": [float(v) for v in robot],
                "predicted_robot_delta": [float(v) for v in pred],
                "error_vector_cm": [float(v) * 100.0 for v in (pred - robot)],
            })
        fit_error_cm = 100.0 * math.sqrt(sum(r * r for r in residual_norms) / len(residual_norms))
        if fit_error_cm <= ROBOT_VERIFY_PASS_ERROR_CM:
            quality = "good"
        elif fit_error_cm <= ROBOT_VERIFY_WARN_ERROR_CM:
            quality = "needs_recapture"
        else:
            quality = "poor"
        return final_M, translation_M, scale, fit_error_cm, quality, residual_details

    def _persist_final_calibration(self, side: ArmSide) -> None:
        arm = self._arms[side]
        if not _wrist_axes_ready(arm):
            arm.cal_state = (
                "awaiting_anchor_wrist_pitch"
                if _valid_wrist_axis_tuple(arm.wrist_pitch_canonical) is None
                else "awaiting_anchor_wrist_roll"
            )
            self._last_error = (
                f"{side} wrist pitch and roll calibration are required before saving VR calibration"
            )
            log.warning("[%s] %s", side, self._last_error)
            return

        # Persist to disk so subsequent sessions don't need to re-run the wizard.
        # Runtime wrist polarity is not persisted here. Motor polarity lives in
        # config/xlerobot.yaml; empirical wrist axes are persisted when captured.
        try:
            _vrcal.write_for_arm(
                side, arm.session_vr_to_robot,
                forward_motion_m=arm.cal_last_fwd_m,
                up_motion_m=arm.cal_last_up_m,
                left_motion_m=arm.cal_last_left_m,
                invert_lateral=arm.invert_lateral,
                confidence=arm.cal_confidence,
                wrist_pitch_anchor_local=arm.wrist_pitch_canonical,
                wrist_roll_anchor_local=arm.wrist_roll_canonical,
                coordinate_frame=self._native_quest.coordinate_frame,
            )
        except Exception as e:
            arm.cal_state = "awaiting_anchor_wrist_roll"
            self._last_error = f"{side} VR calibration save failed: {e}"
            log.warning("[%s] %s", side, self._last_error)
            return
        # A fresh VR-only calibration changes the base frame. Any older
        # robot-verified refinement was solved against a different base, so it
        # must be explicitly re-collected.
        arm.base_vr_direction_matrix = None
        arm.translation_vr_to_robot = None
        arm.translation_scale = 1.0
        arm.robot_verify_state = "idle"
        arm.robot_verify_samples = []
        arm.robot_verify_fit_error_cm = None
        arm.robot_verify_sample_residuals = []
        arm.robot_verify_quality = "unverified"
        arm.robot_verified_at = None
        arm.robot_verify_test_completed = False
        arm.cal_state = "idle"
        arm.cal_motion_acc = (0.0, 0.0, 0.0)
        self._last_error = None
        self._invalidate_teleop_anchor(side, "VR calibration changed")
        pitch_label = "empirical" if arm.wrist_pitch_canonical is not None else "missing"
        roll_label = "empirical" if arm.wrist_roll_canonical is not None else "missing"
        polarity = _WRIST_MOTOR_POLARITY.get(side, {"flex": -1.0, "roll": -1.0})
        log.info(
            "[%s] calibration COMPLETE — confidence=%s invert_lateral=%s "
            "wrist_motor_polarity=(flex %+.0f, roll %+.0f), "
            "wrist canonical=(pitch %s, roll %s). "
            "Squeeze grip again to anchor for teleop.",
            side, arm.cal_confidence, arm.invert_lateral,
            polarity["flex"], polarity["roll"], pitch_label, roll_label,
        )
        # Note: arm.calibrated stays False — user must grip-press once more
        # to anchor for real teleop. The new session matrix will be applied
        # to subsequent VR deltas via `_compute_targets_from_vr`.

    def _handle_engage_button(self, side: ArmSide) -> None:
        """A button on right controller (or X on left) was just pressed.

        Toggle the engage state with this controller's arm as active:
          - Not engaged → engage on this side.
          - Engaged on this side → disengage.
          - Engaged on the OTHER side → switch to this side (keep engaged).
        Equivalent to clicking the UI Engage switch + picking active_arm.
        """
        with self._lock:
            if not MOTORS.is_connected(side):
                log.warning("[%s] engage button pressed but arm not connected", side)
                return
            if self._engaged and self._active_arm == side:
                self._engaged = False
                self._active_arm = None
                log.info("[%s] engage button → DISENGAGED", side)
            elif self._engaged and self._active_arm != side:
                if self._arms[side].cal_confidence in ("poor", "legacy"):
                    log.warning("[%s] engage switch ignored: VR calibration confidence is %s; rerun calibration",
                                side, self._arms[side].cal_confidence)
                    return
                self._active_arm = side
                arm = self._arms[side]
                arm.stale_since = None
                log.info("[%s] engage button → SWITCHED active arm to %s", side, side)
            else:
                if self._arms[side].cal_confidence in ("poor", "legacy"):
                    log.warning("[%s] engage button ignored: VR calibration confidence is %s; rerun calibration",
                                side, self._arms[side].cal_confidence)
                    return
                self._engaged = True
                self._active_arm = side
                arm = self._arms[side]
                arm.stale_since = None
                log.info("[%s] engage button → ENGAGED on %s arm", side, side)

    def _handle_button_edges(
        self,
        side: ArmSide,
        cur_btn: dict[str, bool],
        prev_btn: dict[str, bool],
    ) -> None:
        """Handle Quest face-button edges immediately as goals arrive.

        This avoids sampling races from the 30 Hz drive loop where a quick tap
        could be overwritten by a later goal before the drive loop saw it. Y is
        intentionally a clean, single-button left-controller press: X+Y chords
        are ignored so dual mode cannot accidentally fight single-arm engage.
        """
        pressed_edges = {
            name for name, pressed in cur_btn.items()
            if pressed and not prev_btn.get(name, False)
        }
        if not pressed_edges:
            return

        verify_capture_btn = ENGAGE_BUTTON_BY_SIDE.get(side)
        if verify_capture_btn and verify_capture_btn in pressed_edges:
            arm = self._arms[side]
            if arm.robot_verify_state != "idle":
                self._handle_robot_verification_vr_button(side)
                return

        if not self._controller_buttons_enabled:
            return

        with self._lock:
            any_homing = self._any_homing_locked()

        dual_btn = DUAL_MODE_BUTTON_BY_SIDE.get(side)
        if dual_btn and dual_btn in pressed_edges:
            if any_homing:
                log.info("[%s] %s ignored for dual mode while homing", side, dual_btn)
                return
            other_face_held = any(
                cur_btn.get(name, False)
                for name in ("X", "Y")
                if name != dual_btn
            )
            if other_face_held:
                log.info("[%s] %s ignored for dual mode because another face button is held", side, dual_btn)
                return
            if self._arms[side].cal_state != "idle":
                log.info("[%s] %s ignored for dual mode during calibration", side, dual_btn)
                return
            self._handle_dual_mode_button(side)
            return

        # B edge (right controller only) → recording toggle.
        record_btn = RECORD_BUTTON_BY_SIDE.get(side)
        if record_btn and record_btn in pressed_edges:
            self._handle_record_button(side)

        # A/X edge → single-arm engage toggle.
        engage_btn = ENGAGE_BUTTON_BY_SIDE.get(side)
        if engage_btn and engage_btn in pressed_edges:
            if any_homing:
                log.info("[%s] %s ignored for engage while homing", side, engage_btn)
                return
            if self._arms[side].cal_state != "idle":
                log.info("[%s] %s ignored for engage during calibration", side, engage_btn)
                return
            self._handle_engage_button(side)

    def _handle_robot_verification_vr_button(self, side: ArmSide) -> None:
        arm = self._arms[side]
        if arm.robot_verify_robot_start is None or arm.robot_verify_robot_end is None:
            log.info("[%s] robot verification button ignored until robot start/end are captured", side)
            return
        try:
            point = "end" if arm.robot_verify_vr_start is not None else "start"
            label = arm.robot_verify_label or ""
            self.capture_robot_verification_vr(side, point, label)
            log.info("[%s] robot verification button captured VR %s", side, point)
        except Exception as exc:
            self._last_error = f"{side} robot verification VR button: {exc}"
            log.warning("[%s] robot verification VR button failed: %s", side, exc)

    def _handle_dual_mode_button(self, side: ArmSide) -> None:
        """Y button on the left controller toggles dual-arm drive mode.

        This only changes enablement/selection: when dual mode is on, the drive
        loop runs the existing per-arm target/IK/control path once for left and
        once for right. It does not alter the IK, smoothing, wrist mapping, or
        per-tick motor control.
        """
        with self._lock:
            if side != "left":
                return
            connected = set(MOTORS.connected_sides)
            if {"left", "right"} - connected:
                log.warning("[left] Y dual-mode toggle ignored: connect both arms first")
                return
            if self._dual_mode:
                self._dual_mode = False
                self._engaged = False
                log.info("[left] Y button → DUAL MODE OFF (disengaged)")
                return
            for s in ("left", "right"):
                arm = self._arms[s]
                if arm.cal_confidence in ("poor", "legacy"):
                    log.warning(
                        "[left] Y dual-mode toggle ignored: %s calibration confidence is %s; rerun calibration",
                        s, arm.cal_confidence,
                    )
                    return
                arm.stale_since = None
            self._dual_mode = True
            self._engaged = True
            self._active_arm = None
            log.info("[left] Y button → DUAL MODE ON (left + right arms)")

    def _handle_record_button(self, side: ArmSide) -> None:
        """B button on right controller was just pressed → toggle dataset recording."""
        with self._lock:
            armed = self._recording_armed
            active = self._recording
            transitioning = self._recording_transition_active
        log.info(
            "[%s] B button -> toggle recording (active=%s armed=%s transitioning=%s)",
            side,
            active,
            armed,
            transitioning,
        )
        if armed and not active:
            target = False
        else:
            target = not active
        self._request_recording_transition_async(target, source=f"quest-{side}-button")

    def _request_recording_transition_async(self, enabled: bool, *, source: str) -> bool:
        """Run a controller-requested recording transition off the ingest thread."""
        with self._lock:
            if self._recording_transition_active or self._recording_transition_lock.locked():
                age_s = (
                    time.monotonic() - self._recording_transition_started_at
                    if self._recording_transition_started_at else 0.0
                )
                self._recording_notice = (
                    "recording transition already in progress"
                    + (f" from {self._recording_transition_source}" if self._recording_transition_source else "")
                )
                log.info(
                    "recording %s request from %s ignored; transition already active "
                    "(source=%s target=%s age=%.1fs)",
                    "start" if enabled else "stop",
                    source,
                    self._recording_transition_source or "unknown",
                    self._recording_transition_target,
                    age_s,
                )
                return False
            thread = self._recording_button_thread
            if thread is not None and thread.is_alive():
                log.info("recording %s request from %s ignored; prior button worker still running",
                         "start" if enabled else "stop", source)
                return False
            self._recording_notice = (
                f"recording {'start' if enabled else 'stop'} requested from {source}"
            )

        def worker() -> None:
            try:
                ok = self.set_recording(enabled, source=source)
                log.info(
                    "recording %s request from %s completed ok=%s",
                    "start" if enabled else "stop",
                    source,
                    ok,
                )
            except Exception as exc:
                with self._lock:
                    self._last_error = f"recording {source}: {exc}"
                    self._recording_notice = self._last_error
                log.exception("recording transition worker failed from %s", source)

        thread = threading.Thread(
            target=worker,
            daemon=True,
            name=f"recording-{source}",
        )
        with self._lock:
            self._recording_button_thread = thread
        thread.start()
        return True

    def set_recording_task(self, task: str) -> dict:
        """Cache the UI task text for future B-button recording starts."""
        with self._lock:
            self._last_task = (task or "").strip()
        return self.status()

    def set_recording_root(self, root: Optional[str], repo_id: Optional[str] = None) -> dict:
        """Persist the dataset storage root/repo id in config/xlerobot.yaml."""
        if not self._recording_transition_lock.acquire(blocking=False):
            raise RuntimeError(
                "wait for the recording transition to finish before changing dataset storage root"
            )
        try:
            with self._lock:
                if self._recording_transition_active:
                    raise RuntimeError(
                        "wait for the recording transition to finish before changing dataset storage root"
                    )
                if self._recording or self._recording_armed:
                    raise RuntimeError("stop recording before changing dataset storage root")
            cfg = _dataset.write_dataset_config(root=root, repo_id=repo_id)
            resolved_root = _dataset.resolve_root(cfg.get("root"), str(cfg["repo_id"]))
            with self._lock:
                self._last_dataset_root = resolved_root
                self._recording_repo_id = str(cfg["repo_id"])
                self._last_error = None
            return self.status()
        finally:
            self._recording_transition_lock.release()

    def delete_last_recorded_episode(self) -> dict:
        """Delete the most recently saved episode so operators can retry."""
        with self._recording_transition_lock:
            with self._lock:
                if self._recording:
                    self._last_error = "stop recording before deleting the last episode"
                    return self.status()
                rec = self._recorder
                self._recorder = None
            if rec is not None:
                try:
                    rec.finalize()
                except Exception as e:
                    log.warning("delete-last: finalize before delete failed: %s", e)
            try:
                cfg = _dataset.load_dataset_config()
                effective_root = self._last_dataset_root or _dataset.resolve_root(
                    cfg.get("root"),
                    str(cfg["repo_id"]),
                )
                new_total, resolved_root = _dataset.delete_last_episode(
                    repo_id=str(cfg["repo_id"]),
                    root=effective_root,
                )
                with self._lock:
                    self._episodes_saved = new_total
                    self._last_dataset_root = resolved_root
                    idx, frames = _dataset.last_episode_summary(
                        repo_id=str(cfg["repo_id"]),
                        root=resolved_root,
                    )
                    self._last_saved_episode_index = idx
                    self._last_saved_episode_frames = frames
                    self._last_error = None
            except Exception as e:
                with self._lock:
                    self._last_error = f"delete last episode failed: {e}"
            return self.status()

    def _git_revision(self) -> str:
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO_ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                check=False,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
        except Exception:
            pass
        return ""

    def _recording_context_metadata(self, task: str) -> dict[str, Any]:
        home_pose = _home.read_home_pose()
        home_blob = json.dumps(home_pose, sort_keys=True).encode("utf-8")
        arms: dict[str, Any] = {}
        for side in RECORDING_REQUIRED_SIDES:
            arm = self._arms[side]
            pitch_axis = _valid_wrist_axis_tuple(arm.wrist_pitch_canonical)
            roll_axis = _valid_wrist_axis_tuple(arm.wrist_roll_canonical)
            arms[side] = {
                "teleop_source": "native_quest",
                "coordinate_frame": self._native_quest.coordinate_frame,
                "session_vr_to_robot": arm.session_vr_to_robot.tolist(),
                "translation_matrix_runtime": self._runtime_translation_matrix(arm).tolist(),
                "calibration_confidence": arm.cal_confidence,
                "anchor_generation": arm.anchor_generation,
                "pose_generation": arm.pose_generation,
                "robot_verification": {
                    "quality": arm.robot_verify_quality,
                    "fit_error_cm": arm.robot_verify_fit_error_cm,
                    "translation_scale": arm.translation_scale,
                    "verified_at": arm.robot_verified_at,
                    "low_scale_test_completed": arm.robot_verify_test_completed,
                    "sample_residuals": list(arm.robot_verify_sample_residuals),
                },
                "wrist_mapping": {
                    "source": "direct_controller_rotation",
                    "ready": pitch_axis is not None and roll_axis is not None,
                    "pitch_axis": list(pitch_axis) if pitch_axis is not None else None,
                    "roll_axis": list(roll_axis) if roll_axis is not None else None,
                    "motor_polarity": dict(_WRIST_MOTOR_POLARITY.get(
                        side, {"flex": -1.0, "roll": -1.0}
                    )),
                },
            }
        return {
            "created_at_unix_s": time.time(),
            "task": task,
            "required_sides": list(RECORDING_REQUIRED_SIDES),
            "calibration_profile": _vrcal.profile_status().get("active_profile"),
            "software_revision": self._git_revision(),
            "home_pose_sha256": hashlib.sha256(home_blob).hexdigest(),
            "home_pose_joint_count": len(home_pose),
            "arms": arms,
        }

    def _recording_effective_task(self, task: str = "") -> str:
        effective = (task or "").strip() or self._last_task
        if effective:
            return effective
        try:
            cfg = _dataset.load_dataset_config()
            return str(cfg.get("task_default") or "").strip()
        except Exception:
            return ""

    def _recording_start_failed(self, message: str) -> bool:
        with self._lock:
            self._last_error = message
            self._recording = False
            self._recording_armed = False
            self._recording_pending_task = ""
            self._recording_pending_root = None
            self._recording_notice = message
        log.warning(message)
        return False

    def _wait_for_recording_anchor_inputs(self, timeout_s: float) -> list[str]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        blockers: list[str] = []
        while True:
            with self._lock:
                blockers = self._recording_anchor_refresh_blockers_locked(now=time.time())
            if not blockers:
                return []
            pose_only_blockers = all("Quest controller pose" in b for b in blockers)
            if not pose_only_blockers or time.monotonic() >= deadline:
                return blockers
            time.sleep(0.05)

    def _start_recording_episode_locked(self, task: str, root: Optional[str] = None) -> bool:
        """Open a strict LeRobot episode.

        The recording transition lock is held by the caller, but the session
        status lock is intentionally not held while creating the dataset and
        warming cameras. Camera open/warm-up can take seconds; holding the
        status lock there makes the dashboard appear stuck.
        """
        rec = None
        created_rec = False
        try:
            cfg = _dataset.load_dataset_config()
            roles, shape = _dataset.role_camera_list()
            required_roles = set(_dataset.REQUIRED_CAMERA_ROLES)
            if set(roles) != required_roles:
                missing = sorted(required_roles - set(roles))
                extra = sorted(set(roles) - required_roles)
                detail = []
                if missing:
                    detail.append("missing " + ", ".join(missing))
                if extra:
                    detail.append("unexpected " + ", ".join(extra))
                suffix = "; ".join(detail) if detail else "role mismatch"
                raise RuntimeError(
                    "camera roles must be exactly head, left_wrist, right_wrist (" + suffix + ")"
                )
            effective_root = (root or "").strip() or cfg.get("root") or None
            self._last_dataset_root = _dataset.resolve_root(
                effective_root, str(cfg["repo_id"]),
            )
            self._recording_repo_id = str(cfg["repo_id"])
            log.info(
                "recording recorder init: repo_id=%s root=%s fps=%s roles=%s shape=%s push_to_hub=%s",
                cfg.get("repo_id"),
                self._last_dataset_root,
                cfg.get("fps"),
                roles,
                shape,
                cfg.get("push_to_hub"),
            )
            with self._lock:
                rec = self._recorder
            if rec is None:
                rec = _dataset.DatasetRecorder(
                    repo_id=str(cfg["repo_id"]),
                    fps=int(cfg["fps"]),
                    camera_roles=roles,
                    camera_shape=shape,
                    root=effective_root,
                    push_to_hub=bool(cfg["push_to_hub"]),
                )
                created_rec = True
            if hasattr(rec, "write_recording_context"):
                rec.write_recording_context(self._recording_context_metadata(task))
            rec.start_episode(task=task)
            with self._lock:
                self._recorder = rec
                self._episodes_saved = rec.episode_count
                self._last_saved_episode_index = getattr(rec, "last_saved_episode_index", None)
                self._last_saved_episode_frames = int(getattr(rec, "last_saved_episode_frames", 0))
                self._recording = True
                self._recording_armed = False
                self._recording_pending_task = ""
                self._recording_pending_root = None
                self._last_error = None
                self._recording_notice = ""
                self._recording_camera_frame_skips = 0
                self._recording_consecutive_camera_frame_skips = 0
                self._recording_last_camera_skip_reason = ""
            log.info("strict recording episode opened (task=%r)", task)
            return True
        except Exception as e:
            if created_rec and rec is not None:
                try:
                    rec.finalize()
                except Exception as finalize_exc:
                    log.warning("finalize failed after recorder init error: %s", finalize_exc)
            with self._lock:
                if created_rec and self._recorder is rec:
                    self._recorder = None
                self._last_error = f"recorder init: {e}"
                self._recording = False
                self._recording_armed = False
                self._recording_pending_task = ""
                self._recording_pending_root = None
                self._recording_notice = self._last_error
                log.exception("could not start dataset recorder for task=%r", task)
                return False

    def set_recording(
        self,
        enabled: bool,
        task: str = "",
        home_first: Optional[bool] = None,
        root: Optional[str] = None,
        *,
        source: str = "api",
    ) -> bool:
        started = time.monotonic()
        with self._recording_transition_lock:
            with self._lock:
                self._recording_transition_active = True
                self._recording_transition_started_at = started
                self._recording_transition_source = source
                self._recording_transition_target = bool(enabled)
            log.info(
                "recording transition begin source=%s enabled=%s task_present=%s root_override=%s",
                source,
                bool(enabled),
                bool((task or "").strip()),
                bool((root or "").strip()),
            )
            try:
                result = self._set_recording_locked(
                    enabled,
                    task=task,
                    home_first=home_first,
                    root=root,
                )
                with self._lock:
                    active = self._recording
                    notice = self._recording_notice
                    last_error = self._last_error
                log.info(
                    "recording transition end source=%s enabled=%s result=%s active=%s "
                    "elapsed=%.2fs notice=%r error=%r",
                    source,
                    bool(enabled),
                    result,
                    active,
                    time.monotonic() - started,
                    notice,
                    last_error,
                )
                return result
            finally:
                with self._lock:
                    self._recording_transition_active = False
                    self._recording_transition_source = ""
                    self._recording_transition_target = None
                    self._recording_transition_started_at = 0.0

    def _set_recording_locked(self, enabled: bool, task: str = "",
                              home_first: Optional[bool] = None,
                              root: Optional[str] = None) -> bool:
        """Idempotent recording toggle. Lazy-creates the LeRobotDataset on first
        start, opens a new episode each ON transition, saves the episode on the
        OFF transition. Returns the new recording state.

        `home_first`: if True (or None and `dataset.home_before_episode: true`
        in config/xlerobot.yaml), move every connected arm to its saved home
        pose before opening the new episode. Ensures consistent training data.
        """
        if enabled:
            with self._lock:
                if self._recording:
                    return True

        effective_task = self._recording_effective_task(task)
        if enabled and not effective_task:
            return self._recording_start_failed(
                "task description required before starting an episode"
            )

        if enabled:
            readiness = self._recording_readiness()
            if readiness["start_blockers"]:
                log.warning(
                    "recording start blocked before homing: %s",
                    "; ".join(readiness["start_blockers"]),
                )
                return self._recording_start_failed(
                    "recording blocked: " + "; ".join(readiness["start_blockers"])
                )

        # Resolve home_first from config if not explicitly set.
        if enabled and home_first is None:
            try:
                cfg = _dataset.load_dataset_config()
                home_first = bool(cfg.get("home_before_episode", False))
                log.info("recording start config: home_before_episode=%s repo_id=%s root=%s",
                         home_first, cfg.get("repo_id"), cfg.get("root"))
            except Exception as exc:
                log.exception("recording start config read failed")
                return self._recording_start_failed(
                    f"recording config read failed: {exc}"
                )
        elif home_first is None:
            home_first = False

        # If starting recording AND home_first AND have home pose AND arms
        # connected: home them BEFORE opening the episode. Block until done.
        auto_homed = False
        if enabled and home_first:
            with self._lock:
                sides_to_home = list(MOTORS.connected_sides)
                have_home = bool(_home.read_home_pose()) if sides_to_home else False
            if sides_to_home and have_home:
                log.info("recording start: homing %s before opening episode", sides_to_home)
                try:
                    self.go_home(side=None)  # all connected
                except Exception as e:
                    return self._recording_start_failed(
                        f"recording auto-home failed: {e}"
                    )
                auto_homed = True
                # Wait outside the lock; the drive loop owns homing motion.
                homed = self.wait_for_homing(
                    sides_to_home,
                    timeout_s=HOMING_TIMEOUT_S + RECORDING_HOME_TIMEOUT_BUFFER_S,
                )
                if not homed:
                    with self._lock:
                        still_homing = [
                            side for side in sides_to_home if self._arms[side].homing
                        ]
                    detail = ", ".join(still_homing or sides_to_home)
                    log.warning("recording auto-home did not finish: %s", detail)
                    return self._recording_start_failed(
                        "recording blocked: homing did not finish before opening episode "
                        f"({detail})"
                    )
                with self._lock:
                    home_failures = [
                        f"{side}: {self._arms[side].anchor_invalid_reason}"
                        for side in sides_to_home
                        if self._arms[side].anchor_invalid_reason.startswith(
                            ("homing timed out", "homing aborted")
                        )
                    ]
                if home_failures:
                    log.warning("recording auto-home reported failures: %s", "; ".join(home_failures))
                    return self._recording_start_failed(
                        "recording blocked: auto-home failed ("
                        + "; ".join(home_failures)
                        + ")"
                    )

            readiness = self._recording_readiness()
            if readiness["start_blockers"]:
                log.warning(
                    "recording start blocked after auto-home: %s",
                    "; ".join(readiness["start_blockers"]),
                )
                return self._recording_start_failed(
                    "recording blocked: " + "; ".join(readiness["start_blockers"])
                )

        if enabled:
            readiness = self._recording_readiness()
            if readiness["start_blockers"]:
                log.warning(
                    "recording start blocked before anchor refresh: %s",
                    "; ".join(readiness["start_blockers"]),
                )
                return self._recording_start_failed(
                    "recording blocked: " + "; ".join(readiness["start_blockers"])
                )
            if auto_homed:
                log.info("recording start: waiting for fresh Quest controller poses after auto-home")
                anchor_input_blockers = self._wait_for_recording_anchor_inputs(
                    RECORDING_ANCHOR_INPUT_WAIT_S
                )
                if anchor_input_blockers:
                    log.warning(
                        "recording start blocked waiting for anchor inputs: %s",
                        "; ".join(anchor_input_blockers),
                    )
                    return self._recording_start_failed(
                        "recording blocked: " + "; ".join(anchor_input_blockers)
                    )
            log.info("recording start: refreshing teleop anchors")
            anchor_refresh_blockers = self._refresh_recording_anchors_for_start()
            if anchor_refresh_blockers:
                log.warning(
                    "recording start blocked refreshing anchors: %s",
                    "; ".join(anchor_refresh_blockers),
                )
                return self._recording_start_failed(
                    "recording blocked: " + "; ".join(anchor_refresh_blockers)
                )
            readiness = self._recording_readiness()
            if readiness["calibration_blockers"]:
                log.warning(
                    "recording start blocked after anchor refresh: %s",
                    "; ".join(readiness["calibration_blockers"]),
                )
                return self._recording_start_failed(
                    "recording blocked: " + "; ".join(readiness["calibration_blockers"])
                )

        if enabled:
            with self._lock:
                self._last_task = effective_task
            return self._start_recording_episode_locked(
                effective_task,
                root=(root or "").strip() or None,
            )

        rec = None
        with self._lock:
            if self._recording_armed and not self._recording:
                self._recording_armed = False
                self._recording_pending_task = ""
                self._recording_pending_root = None
                self._last_error = None
                self._recording_notice = ""
                return False
            if not self._recording:
                return self._recording
            # End the in-flight episode. Capture writes finish on the
            # recorder's internal lock; we don't hold ours during the actual
            # save (which can take seconds for video encoding).
            self._recording = False
            self._recording_armed = False
            self._recording_pending_task = ""
            self._recording_pending_root = None
            rec = self._recorder
        # Save the episode OUTSIDE the session lock — `end_episode` flushes
        # frames + may invoke video encoding which can take a while.
        if not enabled and rec is not None:
            saved = rec.end_episode()
            # LeRobot buffers episode metadata until finalize(); without this,
            # the viewer sees data/video files but no meta/episodes parquet.
            if saved:
                with self._lock:
                    self._episodes_saved = rec.episode_count
                    self._last_saved_episode_index = getattr(rec, "last_saved_episode_index", None)
                    self._last_saved_episode_frames = int(getattr(rec, "last_saved_episode_frames", 0))
                rec.finalize()
                with self._lock:
                    if self._recorder is rec:
                        self._recorder = None
                    self._last_error = None
                    self._recording_notice = "episode saved"
            else:
                reason = getattr(rec, "last_end_reason", "") or "episode was not saved"
                with self._lock:
                    self._last_error = reason
                    self._recording_notice = reason
        return self._recording

    def _recording_readiness(self) -> dict[str, Any]:
        """Split hard recording blockers from auto-refreshable anchor blockers.

        Fresh post-home VR anchors are not a hard start blocker anymore:
        pressing Start/B refreshes both anchors from the latest Quest controller
        poses immediately before the episode opens. These blockers are kept in
        status as operator diagnostics. Missing/stale Quest controller poses are
        checked by `_refresh_recording_anchors_for_start` and become hard
        start errors at the moment of recording start.
        """
        blockers = self._recording_calibration_blockers()
        anchor_blockers = self._recording_anchor_blockers()
        anchor_blocker_set = set(anchor_blockers)
        start_blockers = [b for b in blockers if b not in anchor_blocker_set]
        verification_blockers = self._recording_verification_blockers()
        return {
            "calibration_ready": not blockers,
            "calibration_blockers": blockers,
            "start_allowed": not start_blockers,
            "start_blockers": start_blockers,
            "anchor_pending": bool(anchor_blockers),
            "anchor_blockers": anchor_blockers,
            "verification_ready": not verification_blockers,
            "verification_blockers": verification_blockers,
        }

    def _recording_calibration_blockers(self) -> list[str]:
        """Return reasons dataset recording should not start yet."""
        blockers: list[str] = []
        connected = list(MOTORS.connected_sides)
        required = list(RECORDING_REQUIRED_SIDES)
        if set(connected) != set(required):
            blockers.append("connect both left and right arms before recording")
        if self._native_quest_clients <= 0:
            blockers.append("Quest app is not connected")
        try:
            roles, _shape = _dataset.role_camera_list()
            role_set = set(roles)
            required_roles = set(_dataset.REQUIRED_CAMERA_ROLES)
            missing_roles = sorted(required_roles - role_set)
            extra_roles = sorted(role_set - required_roles)
            if missing_roles:
                blockers.append("camera roles missing: " + ", ".join(missing_roles))
            if extra_roles:
                blockers.append("unexpected camera roles configured: " + ", ".join(extra_roles))
            suspended = _cameras.suspended_capture_roles()
            suspended_required = [role for role in _dataset.REQUIRED_CAMERA_ROLES if role in suspended]
            if suspended_required:
                blockers.append("camera capture suspended for recording: " + ", ".join(suspended_required))
        except Exception as e:
            blockers.append(f"camera readiness check failed: {e}")
        try:
            home_status = _home.home_pose_status()
        except Exception as e:
            home_status = {}
            blockers.append(f"home pose status failed: {e}")
        for side in required:
            arm = self._arms[side]
            if not MOTORS.is_connected(side):
                continue
            if not MOTORS.is_torque_enabled(side):
                blockers.append(f"{side} torque is off")
            home = home_status.get(side) or {}
            if not home.get("captured", False):
                blockers.append(f"{side} home pose not captured")
            if not self._ensure_kinematics(arm):
                blockers.append(f"{side} calibrated URDF kinematics missing")
            if not _wrist_axes_ready(arm):
                blockers.append(
                    f"{side} wrist pitch/roll calibration missing; rerun VR calibration wrist steps"
                )
            if arm.robot_verify_test_active:
                blockers.append(f"{side} low-scale calibration test is still active")
            if arm.cal_confidence != "good":
                blockers.append(f"{side} VR-only calibration confidence is {arm.cal_confidence}")
            verification_blocker = self._recording_robot_verification_blocker(
                side, arm, include_active_test=False
            )
            if verification_blocker:
                blockers.append(verification_blocker)
        blockers.extend(self._recording_anchor_blockers())
        return blockers

    def _recording_verification_blockers(self) -> list[str]:
        blockers: list[str] = []
        for side in RECORDING_REQUIRED_SIDES:
            blocker = self._recording_robot_verification_blocker(side, self._arms[side])
            if blocker:
                blockers.append(blocker)
        return blockers

    def _recording_robot_verification_blocker(
        self,
        side: ArmSide,
        arm: _PerArm,
        *,
        include_active_test: bool = True,
    ) -> str | None:
        if include_active_test and arm.robot_verify_test_active:
            return f"{side} low-scale calibration test is still active"
        if arm.robot_verify_quality != "good":
            if arm.robot_verify_quality in ("warn", "needs_recapture", "poor"):
                detail = (
                    f" residual {arm.robot_verify_fit_error_cm:.1f} cm"
                    if arm.robot_verify_fit_error_cm is not None else ""
                )
                return f"{side} robot verification needs recapture{detail}"
            if arm.robot_verify_samples:
                missing = self._missing_robot_verification_labels(arm)
                suffix = f"; missing {', '.join(missing)}" if missing else "; solve verification"
                return f"{side} robot verification incomplete{suffix}"
            return f"{side} robot verification missing"
        if arm.robot_verify_fit_error_cm is None:
            return f"{side} robot verification has no residual"
        if arm.robot_verify_fit_error_cm > ROBOT_VERIFY_PASS_ERROR_CM:
            return f"{side} robot verification residual {arm.robot_verify_fit_error_cm:.1f} cm"
        if not arm.robot_verify_test_completed:
            return f"{side} low-scale calibration test not completed"
        return None

    def _recording_anchor_blockers(self) -> list[str]:
        connected = list(MOTORS.connected_sides)
        if set(connected) != set(RECORDING_REQUIRED_SIDES):
            return []
        return self._fresh_anchor_blockers(
            list(RECORDING_REQUIRED_SIDES),
            include_wrist_axes=False,
        )

    @staticmethod
    def _is_skippable_recording_camera_error(message: str) -> bool:
        lowered = message.lower()
        return (
            "camera frames invalid" in lowered
            or "camera frames missing roles" in lowered
        )

    def _skip_recording_camera_tick_or_abort(self, reason: str) -> None:
        with self._lock:
            now = time.time()
            rec = self._recorder
            if not (self._recording and rec is not None and rec.in_episode):
                return
            self._recording_camera_frame_skips += 1
            self._recording_consecutive_camera_frame_skips += 1
            self._recording_last_camera_skip_reason = reason
            saved_frames = int(getattr(rec, "frame_count_in_episode", 0) or 0)
            total_samples = saved_frames + self._recording_camera_frame_skips
            skip_ratio = (
                self._recording_camera_frame_skips / total_samples
                if total_samples > 0 else 1.0
            )
            over_consecutive = (
                self._recording_consecutive_camera_frame_skips
                > RECORDING_MAX_CONSECUTIVE_CAMERA_SKIPS
            )
            over_ratio = (
                total_samples >= RECORDING_MIN_SAMPLES_FOR_SKIP_RATIO
                and skip_ratio > RECORDING_MAX_CAMERA_SKIP_RATIO
            )
            if over_consecutive or over_ratio:
                detail = (
                    "recording camera unstable: "
                    f"skipped {self._recording_camera_frame_skips}/{total_samples} frame ticks "
                    f"({skip_ratio * 100:.1f}%), "
                    f"consecutive={self._recording_consecutive_camera_frame_skips}; "
                    f"last={reason}"
                )
                self._abort_active_recording_locked(detail)
                return
            self._recording_notice = (
                "camera frame skipped "
                f"({self._recording_camera_frame_skips}/{total_samples}); "
                f"{reason}"
            )
        log.warning("recording skipped incomplete camera tick: %s", reason)

    def _record_frame_if_active(
        self,
        commanded_this_tick: Optional[dict[ArmSide, dict[str, float]]] = None,
        expected_driven_sides: Optional[list[ArmSide]] = None,
        expected_held_sides: Optional[list[ArmSide]] = None,
    ) -> None:
        """If recording is on and an episode is active, append one frame.

        Action = same-tick commanded joint positions for arms moved this tick.
        Engaged arms that are holding due to IDLE/stale controller input reuse
        their last command; passive arms use present joint positions as no-op.
        Observation.state = present joint positions (both arms).
        Observation.images.<role> = latest snapshot from each configured camera.
        Missing arm/motor data aborts the active strict recording episode.
        Isolated camera-frame misses are skipped within a strict per-episode
        budget; repeated camera misses abort and discard the episode.
        """
        with self._lock:
            now = time.time()
            rec = self._recorder
            if not (self._recording and rec is not None and rec.in_episode):
                return
            # Snapshot dictionaries while holding the lock; release before doing
            # camera capture (which is slow).
            connected = list(MOTORS.connected_sides)
            commanded_by_side = {
                s: dict(commanded)
                for s, commanded in (commanded_this_tick or {}).items()
            }
            if set(connected) != set(RECORDING_REQUIRED_SIDES):
                self._abort_active_recording_locked("connected arms changed during recording")
                return
            expected_driven: list[ArmSide] = list(expected_driven_sides or [])
            expected_held: list[ArmSide] = list(expected_held_sides or [])
            held_command_by_side: dict[ArmSide, dict[str, float]] = {}
            if expected_driven_sides is None and expected_held_sides is None and self._engaged:
                candidates = connected if self._dual_mode else (
                    [self._active_arm] if self._active_arm in connected else []
                )
                for s in candidates:
                    arm = self._arms[s]
                    goal = arm.latest
                    goal_age = now - goal.received_at if goal.has_data else 1e9
                    controls_arm = (
                        arm.calibrated
                        and MOTORS.is_torque_enabled(s)
                        and not arm.homing
                    )
                    if controls_arm:
                        prefix = f"{s}_arm_"
                        held = {
                            f"{prefix}{j}": float(arm.last_commanded_targets[f"{prefix}{j}"])
                            for j in _motors.JOINTS_PER_ARM
                            if f"{prefix}{j}" in arm.last_commanded_targets
                        }
                        if len(held) == len(_motors.JOINTS_PER_ARM):
                            held_command_by_side[s] = held
                    if (
                        controls_arm
                        and goal.has_data
                        and goal_age <= GOAL_SKIP_AGE_S
                        and goal.mode == "position"
                    ):
                        expected_driven.append(s)
                    elif controls_arm:
                        expected_held.append(s)
            for s in expected_held:
                arm = self._arms[s]
                if not (arm.calibrated and MOTORS.is_torque_enabled(s) and not arm.homing):
                    continue
                prefix = f"{s}_arm_"
                held = {
                    f"{prefix}{j}": float(arm.last_commanded_targets[f"{prefix}{j}"])
                    for j in _motors.JOINTS_PER_ARM
                    if f"{prefix}{j}" in arm.last_commanded_targets
                }
                if len(held) == len(_motors.JOINTS_PER_ARM):
                    held_command_by_side[s] = held
            missing_driven = [s for s in expected_driven if not commanded_by_side.get(s)]
            if missing_driven:
                self._abort_active_recording_locked(
                    "missing same-tick motor command for driven arm(s): " + ", ".join(missing_driven)
                )
                return
            missing_held = [s for s in expected_held if not held_command_by_side.get(s)]
            if missing_held:
                self._abort_active_recording_locked(
                    "missing held motor command for controlled arm(s): " + ", ".join(missing_held)
                )
                return
        # Outside lock: read present positions (bus I/O) + camera snapshots.
        try:
            present_dict = MOTORS.read_positions()
        except Exception as e:
            self._abort_recording_due_to_error(f"record: read_positions failed: {e}")
            return
        action_dict: dict[str, float] = {}
        for s in connected:
            prefix = f"{s}_arm_"
            if commanded_by_side.get(s):
                action_dict.update(commanded_by_side[s])
            elif held_command_by_side.get(s):
                action_dict.update(held_command_by_side[s])
            else:
                for j in _motors.JOINTS_PER_ARM:
                    key = f"{prefix}{j}"
                    if key in present_dict:
                        action_dict[key] = float(present_dict[key])
        try:
            if hasattr(rec, "grab_camera_frames"):
                cam_frames = rec.grab_camera_frames()
            else:
                cam_frames = _dataset.grab_camera_frames()
        except Exception as e:
            self._abort_recording_due_to_error(f"record: grab_camera_frames failed: {e}")
            return
        try:
            rec.add_frame(action_dict, present_dict, cam_frames)
        except Exception as e:
            reason = f"record: strict frame rejected: {e}"
            if self._is_skippable_recording_camera_error(str(e)):
                self._skip_recording_camera_tick_or_abort(reason)
                return
            self._abort_recording_due_to_error(reason)
            return
        with self._lock:
            self._recording_consecutive_camera_frame_skips = 0

    def _abort_active_recording_locked(self, reason: str) -> None:
        """Abort the current episode without saving. Caller holds self._lock."""
        self._recording = False
        self._recording_armed = False
        self._recording_pending_task = ""
        self._recording_pending_root = None
        self._last_error = reason
        self._recording_notice = reason
        rec = self._recorder
        if rec is not None and hasattr(rec, "discard_episode"):
            try:
                rec.discard_episode()
            except Exception as e:
                log.warning("discard failed after recording abort: %s", e)
        log.warning("recording aborted: %s", reason)

    def _abort_recording_due_to_error(self, reason: str) -> None:
        with self._lock:
            if not self._recording:
                return
            self._abort_active_recording_locked(reason)

    def _debug_log_gripper(self, side: ArmSide, goal: _LatestGoal,
                            targets: _LiveTargets, final: dict[str, float],
                            present: dict[str, float], now: float) -> None:
        """1Hz per-arm log showing trigger value vs gripper target vs sent vs present.
        Lets you bisect 'gripper not moving' between VR/IK/motor sides at a glance."""
        if not hasattr(self, "_dbg_gripper_state"):
            self._dbg_gripper_state: dict[ArmSide, dict[str, Any]] = {}
        state = self._dbg_gripper_state.setdefault(side, {"t": 0.0, "trig": None})
        trigger_now = bool(goal.trigger)
        if trigger_now != state["trig"] or (now - state["t"]) > 1.0:
            prefix = f"{side}_arm_"
            log.info(
                "[%s] gripper: trigger=%s target=%.1f sent=%.1f present=%.1f "
                "(open=%.1f closed=%.1f)",
                side, trigger_now, targets.gripper,
                final.get(f"{prefix}gripper", float("nan")),
                present.get(f"{prefix}gripper", float("nan")),
                self._gripper_open, self._gripper_closed,
            )
            state["t"] = now
            state["trig"] = trigger_now

    def _direct_wrist_targets_from_vr(
        self,
        side: ArmSide,
        arm: _PerArm,
        controller_rel_T: _np.ndarray,
    ) -> _DirectWristTargets:
        """Map reset-relative controller rotation directly to SO101 wrist joints.

        This is deliberately independent from body IK. The SO101 wrist has two
        controllable DOFs, so the calibrated controller-anchor-local rotation
        vector is projected onto the calibrated pitch/roll axes and added to the
        wrist joint values captured at grip RESET.
        """
        R_delta_vr = _controller_rotation_delta_for_side(side, controller_rel_T[:3, :3])
        wrist_rotvec_local = _np.asarray(_R.from_matrix(R_delta_vr).as_rotvec(), dtype=float)
        if wrist_rotvec_local.shape != (3,) or not _np.all(_np.isfinite(wrist_rotvec_local)):
            raise ValueError("controller wrist rotation is invalid")

        pitch_canonical = _valid_wrist_axis_tuple(arm.wrist_pitch_canonical)
        roll_canonical = _valid_wrist_axis_tuple(arm.wrist_roll_canonical)
        if pitch_canonical is None or roll_canonical is None:
            raise RuntimeError("calibrated wrist pitch/roll axes are required")

        pitch_axis, roll_axis = _effective_wrist_axes(
            side,
            pitch_canonical=pitch_canonical,
            roll_canonical=roll_canonical,
        )
        polarity = dict(_WRIST_MOTOR_POLARITY.get(side, {"flex": -1.0, "roll": -1.0}))
        flex_pol = 1.0 if float(polarity.get("flex", -1.0)) >= 0.0 else -1.0
        roll_pol = 1.0 if float(polarity.get("roll", -1.0)) >= 0.0 else -1.0
        wrist_flex_delta_deg = flex_pol * math.degrees(float(_np.dot(wrist_rotvec_local, pitch_axis)))
        wrist_roll_delta_deg = roll_pol * math.degrees(float(_np.dot(wrist_rotvec_local, roll_axis)))

        wrist_flex = arm.anchor.wrist_flex_deg + wrist_flex_delta_deg
        wrist_roll = arm.anchor.wrist_roll_deg + wrist_roll_delta_deg
        bounds = MOTORS.bounds
        wrist_flex_lo, wrist_flex_hi = bounds.get(f"{side}_arm_wrist_flex", (-180.0, 180.0))
        wrist_roll_lo, wrist_roll_hi = bounds.get(f"{side}_arm_wrist_roll", (-180.0, 180.0))
        wrist_flex = max(wrist_flex_lo, min(wrist_flex_hi, float(wrist_flex)))
        wrist_roll = max(wrist_roll_lo, min(wrist_roll_hi, float(wrist_roll)))

        return _DirectWristTargets(
            wrist_flex=float(wrist_flex),
            wrist_roll=float(wrist_roll),
            wrist_flex_delta_deg=float(wrist_flex_delta_deg),
            wrist_roll_delta_deg=float(wrist_roll_delta_deg),
            pitch_axis=pitch_axis,
            roll_axis=roll_axis,
            polarity=polarity,
        )

    def _compute_targets_from_vr(self, side: ArmSide, goal: _LatestGoal,
                                  scale: float) -> None:
        """Convert the latest VR controller pose -> SO101 joint targets.

        Pipeline (per-tick):
          1. Read the controller's absolute reset-relative displacement from
             the current pose and RESET anchor. Queued packet deltas are cleared
             for diagnostics only; they do not drive live motion.
          2. Map the integrated reset-relative VR displacement through the
             runtime translation matrix, then LERP-smooth and cap the Cartesian
             target step before IK.
          3. Map controller-relative rotation directly to wrist_flex/wrist_roll
             using calibrated wrist axes and motor polarity. This path does not
             read IK output or smoothed EE orientation.
          4. Clamp the resulting EE position to the workspace box + reach sphere
             + rear-singularity guard, then reconcile `arm.offset_robot` so a
             clamped step doesn't accumulate hidden motion debt.
          5. Calibrated SO101 URDF IK for pan/lift/elbow in LeRobot motor
             degrees; large jumps are rejected and the previous solution is
             reused. Missing URDF kinematics holds the target instead of
             issuing uncalibrated commands.
          6. Wrist flex/roll are added from the direct controller-rotation path.
          7. The drive loop rate-caps every joint, applies weighted filtering
             to IK arm joints, and deliberately bypasses that filter for wrist
             pitch/roll and gripper.

        Rotation and translation both require `arm.controller_anchor_T`
        (captured on RESET), the current controller position, and the current
        controller quaternion.
        """
        arm = self._arms[side]
        current_pos = goal.controller_position
        current_q = goal.rotation_quat

        if arm.controller_anchor_T is None or current_pos is None or current_q is None:
            log.warning(
                "[%s] SE(3) inputs missing (anchor=%s current_pos=%s current_q=%s); holding targets",
                side,
                arm.controller_anchor_T is not None,
                current_pos is not None,
                current_q is not None,
            )
            return

        try:
            controller_current_T = _pose_matrix_from_vr(current_pos, current_q)
            controller_rel_T = _np.linalg.solve(arm.controller_anchor_T, controller_current_T)
        except Exception as e:
            log.warning("[%s] SE(3) controller mapping failed (%s); holding targets", side, e)
            return

        try:
            wrist_targets = self._direct_wrist_targets_from_vr(side, arm, controller_rel_T)
        except Exception as e:
            log.warning("[%s] direct wrist mapping failed (%s); holding targets", side, e)
            return

        with self._lock:
            pending_rel = _np.array(arm.pending_rel_position, dtype=float)
            arm.pending_rel_position = (0.0, 0.0, 0.0)
        if not _np.all(_np.isfinite(pending_rel)):
            pending_rel = _np.zeros(3, dtype=float)
        controller_world_delta = (
            _np.array(current_pos, dtype=float) - _np.array(arm.controller_anchor_T[:3, 3], dtype=float)
        )
        previous_accum_vr = _np.array(arm.vr_offset_accum, dtype=float)
        if not _np.all(_np.isfinite(previous_accum_vr)):
            previous_accum_vr = _np.zeros(3, dtype=float)
        if not _np.all(_np.isfinite(controller_world_delta)):
            accum_vr = previous_accum_vr
        elif float(_np.linalg.norm(controller_world_delta)) < POS_DEADZONE_M:
            accum_vr = _np.zeros(3, dtype=float)
        elif float(_np.linalg.norm(controller_world_delta - previous_accum_vr)) < POS_DEADZONE_M:
            accum_vr = previous_accum_vr
        else:
            accum_vr = controller_world_delta
        arm.vr_offset_accum = (float(accum_vr[0]), float(accum_vr[1]), float(accum_vr[2]))

        translation_M = self._runtime_translation_matrix(arm)
        dp_robot = translation_M @ accum_vr
        desired_offset = dp_robot * scale
        current_offset = _np.array(arm.offset_robot, dtype=float)
        # BEAVR-style LERP smoothing plus a LeRobot-style Cartesian step cap.
        pos_alpha = max(0.05, min(1.0, POS_EMA_ALPHA))
        new_offset = (1.0 - pos_alpha) * current_offset + pos_alpha * desired_offset
        step = new_offset - current_offset
        step_norm = float(_np.linalg.norm(step))
        max_step = max(0.001, float(MAX_EE_STEP_M))
        if step_norm > max_step:
            new_offset = current_offset + step * (max_step / step_norm)
        arm.offset_robot = (float(new_offset[0]), float(new_offset[1]), float(new_offset[2]))
        arm.quality_last_offset_step_m = min(step_norm, max_step)
        inst_speed = arm.quality_last_offset_step_m / LOOP_PERIOD_S
        arm.quality_offset_speed_ema_mps = (
            0.85 * arm.quality_offset_speed_ema_mps + 0.15 * inst_speed
        )

        # Target position from anchor + offset, axis-clamped to EE_BOUNDS
        # (sanity box), then radially clamped before IK near the edge of the
        # SO101 reach envelope. We reconcile `arm.offset_robot` to the clamped
        # position so a clamp doesn't leave the LERP integrator chasing an
        # unreachable target.
        target_pos = _clamp_target_position(_np.array([
            arm.anchor_ee_pos[0] + arm.offset_robot[0],
            arm.anchor_ee_pos[1] + arm.offset_robot[1],
            arm.anchor_ee_pos[2] + arm.offset_robot[2],
        ], dtype=float))
        tx, ty, tz = (float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
        arm.offset_robot = (
            tx - arm.anchor_ee_pos[0],
            ty - arm.anchor_ee_pos[1],
            tz - arm.anchor_ee_pos[2],
        )
        arm.target_T[:3, 3] = (tx, ty, tz)
        arm.target_T[:3, :3] = arm.anchor_R_robot

        # Position IK in calibrated SO101 joint space. Prefer the per-arm URDF
        # solver so RESET anchors and target pose use the same motor-degree
        # convention sent to the SOFollower.
        if arm.kinematics is None:
            self._last_error = f"{side} calibrated SO101 URDF kinematics unavailable; holding VR target"
            log.warning("[%s] %s", side, self._last_error)
            return
        ik_mode = "urdf"
        ik_rejected = False
        try:
            q_ik = arm.kinematics.inverse_kinematics(
                arm.last_body_q_sol,
                arm.target_T,
                position_weight=1.0,
                orientation_weight=0.0,
            )
            q_body = _np.asarray(q_ik, dtype=float).copy()
            if q_body.ndim != 1 or q_body.shape[0] < len(_BODY_IK_JOINT_ORDER):
                raise RuntimeError(
                    f"body IK returned shape {q_body.shape}; expected at least {len(_BODY_IK_JOINT_ORDER)} joints"
                )
            q_body = q_body[:len(_BODY_IK_JOINT_ORDER)].copy()
            if not _np.all(_np.isfinite(q_body)):
                log.warning("[%s] position IK output NaN/Inf; reusing previous body q_sol", side)
                q_body = arm.last_body_q_sol.copy()
            else:
                raw_jump = _np.abs(q_body - arm.last_body_q_sol)
                if _np.any(raw_jump > IK_JUMP_REJECT_DEG):
                    ik_rejected = True
                    log.warning(
                        "[%s] position IK jump rejected: target_pos=(%.3f, %.3f, %.3f), dq=%s",
                        side, tx, ty, tz, tuple(f"{v:.1f}" for v in raw_jump),
                    )
                    q_body = arm.last_body_q_sol.copy()
                # The cartesian offset is already LERP-smoothed; the drive loop
                # applies output filtering only to IK arm joints. Wrist joints
                # are direct VR commands and bypass that final filter.
                bounds = MOTORS.bounds
                for idx, joint in enumerate(_BODY_IK_JOINT_ORDER):
                    lo, hi = bounds.get(f"{side}_arm_{joint}", (-180.0, 180.0))
                    q_body[idx] = max(lo, min(hi, float(q_body[idx])))
                arm.last_body_q_sol = q_body.copy()
        except Exception as e:
            ik_rejected = True
            log.warning("[%s] position IK failed (%s); reusing previous body q_sol", side, e)
            q_body = arm.last_body_q_sol.copy()

        # Build the live joint targets. The first three joints come from body
        # position IK; wrist joints come only from direct controller rotation.
        q_full = _np.asarray(arm.last_q_sol, dtype=float).copy()
        if q_full.shape != (len(_IK_JOINT_ORDER),) or not _np.all(_np.isfinite(q_full)):
            q_full = _np.zeros(len(_IK_JOINT_ORDER), dtype=float)
        q_full[:len(_BODY_IK_JOINT_ORDER)] = q_body
        q_full[3] = wrist_targets.wrist_flex
        q_full[4] = wrist_targets.wrist_roll
        arm.last_q_sol = q_full.copy()
        arm.last_q_filtered = q_full.copy()
        gripper_target = self._gripper_closed if goal.trigger else self._gripper_open
        arm.targets = _LiveTargets(
            shoulder_pan=float(q_body[0]),
            shoulder_lift=float(q_body[1]),
            elbow_flex=float(q_body[2]),
            wrist_flex=float(wrist_targets.wrist_flex),
            wrist_roll=float(wrist_targets.wrist_roll),
            gripper=gripper_target,
        )
        arm.quality_ticks += 1
        if ik_rejected:
            arm.quality_ik_rejects += 1
        current_pos_for_diag = _np.array(current_pos, dtype=float)
        arm.last_diag = {
            "controller_rotation_handedness": "inverted_for_left" if side == "left" else "normal",
            "controller_position": [float(v) for v in current_pos_for_diag],
            "controller_rel_translation": [float(v) for v in controller_rel_T[:3, 3]],
            "controller_world_delta": [
                float(v) for v in (current_pos_for_diag - arm.controller_anchor_T[:3, 3])
            ],
            "pending_rel_step": [float(v) for v in pending_rel],
            "vr_offset_accum": [float(v) for v in arm.vr_offset_accum],
            "dp_robot": [float(v) for v in dp_robot],
            "translation_scale": float(arm.translation_scale),
            "translation_source": self._runtime_translation_source(arm),
            "offset_robot": [float(v) for v in arm.offset_robot],
            "target_ee_pos": [tx, ty, tz],
            "body_target_quat_xyzw": [
                float(v) for v in _positive_quat_xyzw(_R.from_matrix(arm.anchor_R_robot).as_quat())
            ],
            "ik_mode": ik_mode,
            "q_arm": [float(v) for v in q_body],
            "wrist_mapping_source": "direct_controller_rotation",
            "wrist_delta_deg": [
                wrist_targets.wrist_flex_delta_deg,
                wrist_targets.wrist_roll_delta_deg,
            ],
            "wrist_motor_polarity": dict(wrist_targets.polarity),
            "wrist_axes": {
                "pitch": [float(v) for v in wrist_targets.pitch_axis],
                "roll": [float(v) for v in wrist_targets.roll_axis],
            },
            "using_analytical_fallback": bool(arm.using_analytical_fallback),
            "quality": {
                "offset_step_m": float(arm.quality_last_offset_step_m),
                "offset_speed_ema_mps": float(arm.quality_offset_speed_ema_mps),
                "ik_reject_fraction": (
                    float(arm.quality_ik_rejects) / float(arm.quality_ticks)
                    if arm.quality_ticks else 0.0
                ),
            },
        }

    @staticmethod
    def _sleep_until(next_tick: float) -> float:
        next_tick += LOOP_PERIOD_S
        wait = next_tick - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        elif wait < -0.2:
            # We're behind by >200 ms — likely the bus stalled; resync rather than spin.
            next_tick = time.monotonic()
        return next_tick

    # ── teardown ────────────────────────────────────────────────────────────
    def _stop_threads_and_servers(self) -> None:
        self._stop_evt.set()
        if self._drive_thread is not None and self._drive_thread.is_alive():
            self._drive_thread.join(timeout=2)
        self._drive_thread = None


SESSION = VRTeleopSession()
