"""Per-arm VR→robot calibration persistence.

The calibration wizard captures a 3×3 rotation matrix per arm that maps VR-world
coordinates to robot base coordinates. Re-running the wizard every session is
tedious — so we save the result to `config/vr_calibration.yaml` and reload it
on startup. Users can keep using the same calibration as long as their VR setup
(headset position, where they stand) hasn't changed.

File format (`config/vr_calibration.yaml`):

    active_profile: default
    profiles:
      default:
        left:
          calibration_mode: vr_direction
          coordinate_frame: quest_operator_frame
          session_vr_to_robot:
            - [m00, m01, m02]
            - [m10, m11, m12]
            - [m20, m21, m22]
          calibrated_at: '2026-05-24T12:34:56'
          forward_motion_m: 0.103
          up_motion_m: 0.092
          left_motion_m: 0.088
          invert_lateral: false
          confidence: good
          robot_verification:
            calibration_mode: robot_verified
            translation_scale: 0.51
            fit_error_cm: 1.2
            calibration_quality: good
            low_scale_test_completed: true
            robot_verified_samples: [...]
        right: { ... same shape ... }

Older files with top-level `left:` / `right:` are still accepted and are
treated as the `default` profile until the file is next written.

Note that wrist-motor polarity is NOT a calibration output — it is a hardware
property of how each arm's wrist motors are mounted, and lives in
`config/xlerobot.yaml` under `vr.wrist_motor_polarity.{left,right}.{flex,roll}`.
Older calibration files may still contain `wrist_flex_sign` / `wrist_roll_sign`
keys; they are now ignored by the backend and will be dropped the next time
this arm is recalibrated.

This file is auto-managed by the calibration wizard. Sides that haven't been
calibrated yet are simply absent. Edit by re-running the wizard, not by hand.
"""
from __future__ import annotations

import datetime
import logging
import re
from typing import Any

import numpy as np
import yaml

from openpibot.server.config import REPO_ROOT

log = logging.getLogger(__name__)
CFG_PATH = REPO_ROOT / "config" / "vr_calibration.yaml"
SIDES = ("left", "right")
DEFAULT_PROFILE = "default"


def _header() -> str:
    return (
        "# VR→robot calibration data. Auto-managed by the calibration wizard in\n"
        "# the dashboard — edit by re-running the wizard, not by hand.\n"
        "# Multiple user/setup profiles live under profiles.<name>.\n"
        "# Wrist-motor polarity lives in config/xlerobot.yaml (vr.wrist_motor_polarity).\n"
    )


def _normalize_profile_name(name: str | None) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "").strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("profile name cannot be empty")
    if cleaned in SIDES or cleaned in {"profiles", "active_profile"}:
        raise ValueError(f"{cleaned!r} is reserved")
    return cleaned[:64]


def _read_raw() -> dict[str, Any]:
    if not CFG_PATH.exists():
        return {}
    try:
        data = yaml.safe_load(CFG_PATH.read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        log.warning("could not read %s: %s", CFG_PATH, e)
        return {}


def _normalize_profile_entry(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    normalized = dict(entry)
    # Wrist-roll calibration was added with the raw-controller-anchor wrist
    # convention. If a file was written during that transition but lacks the
    # explicit marker, treat it as the new format instead of applying the older
    # left-arm legacy sign conversion on load.
    for side in SIDES:
        arm = normalized.get(side)
        if not isinstance(arm, dict):
            continue
        if (
            "wrist_canonical_frame" not in arm
            and "wrist_roll_anchor_local" in arm
        ):
            arm = dict(arm)
            arm["wrist_canonical_frame"] = "raw_controller_anchor_local"
            normalized[side] = arm
        roll = arm.get("wrist_roll_anchor_local")
        if isinstance(roll, (list, tuple)) and len(roll) == 3:
            try:
                if float(roll[2]) > 0.0:
                    arm = dict(arm)
                    arm["wrist_roll_anchor_local"] = [-float(v) for v in roll]
                    normalized[side] = arm
            except (TypeError, ValueError):
                pass
    return normalized


def _normalized_doc(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return canonical profile-shaped config without writing it."""
    data = raw if raw is not None else _read_raw()
    if isinstance(data.get("profiles"), dict):
        profiles: dict[str, dict[str, Any]] = {}
        for name, value in data["profiles"].items():
            try:
                profile = _normalize_profile_name(str(name))
            except ValueError:
                log.warning("ignoring invalid VR calibration profile name %r", name)
                continue
            if profile in profiles:
                log.warning("duplicate normalized VR calibration profile name %r; keeping first", profile)
                continue
            profiles[profile] = _normalize_profile_entry(value)
        if not profiles:
            profiles = {DEFAULT_PROFILE: {}}
        active_raw = data.get("active_profile") or DEFAULT_PROFILE
        try:
            active = _normalize_profile_name(active_raw)
        except ValueError:
            active = DEFAULT_PROFILE
        if active not in profiles:
            active = next(iter(profiles.keys()))
        return {"active_profile": active, "profiles": profiles}

    # Backward compatibility: old files had top-level left/right entries.
    legacy_profile: dict[str, Any] = {}
    for side in SIDES:
        if isinstance(data.get(side), dict):
            legacy_profile[side] = dict(data[side])
    return {
        "active_profile": DEFAULT_PROFILE,
        "profiles": {DEFAULT_PROFILE: _normalize_profile_entry(legacy_profile)},
    }


def _write_doc(doc: dict[str, Any]) -> None:
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=None)
    CFG_PATH.write_text(_header() + body)


def _active_profile_entry(doc: dict[str, Any]) -> dict[str, Any]:
    active = str(doc.get("active_profile") or DEFAULT_PROFILE)
    profiles = doc.setdefault("profiles", {})
    if active not in profiles or not isinstance(profiles[active], dict):
        profiles[active] = {}
    return profiles[active]


def read_all() -> dict[str, dict[str, Any]]:
    """Load active profile per-arm calibrations."""
    return dict(_active_profile_entry(_normalized_doc()))


def read_for_arm(side: str) -> dict[str, Any] | None:
    """Calibration data for one arm, or None if not yet saved."""
    return (read_all() or {}).get(side)


def _valid_matrix(raw: Any, *, side: str, label: str) -> np.ndarray | None:
    """Parse, validate, and orthonormalize a persisted 3x3 rotation matrix."""
    if not raw:
        return None
    try:
        M = np.array(raw, dtype=float)
        if M.shape != (3, 3):
            log.warning("[%s] saved %s has wrong shape %s; ignoring", side, label, M.shape)
            return None
        if not np.all(np.isfinite(M)):
            log.warning("[%s] saved %s has non-finite values; ignoring", side, label)
            return None
        ortho = _orthonormalize_matrix(M)
        error = float(np.linalg.norm(M.T @ M - np.eye(3)))
        if error > 0.05:
            log.warning(
                "[%s] saved %s is too skewed (orthogonality error %.3f); ignoring",
                side, label, error,
            )
            return None
        if error > 1e-3:
            log.warning(
                "[%s] saved %s was slightly non-orthonormal (%.4f); using closest rotation",
                side, label, error,
            )
        return ortho
    except Exception as e:
        log.warning("[%s] saved %s is malformed: %s; ignoring", side, label, e)
        return None


def _valid_wrist_axis(raw: Any) -> bool:
    try:
        axis = np.asarray(raw, dtype=float)
    except Exception:
        return False
    return (
        axis.shape == (3,)
        and np.all(np.isfinite(axis))
        and float(np.linalg.norm(axis)) > 1e-6
    )


def matrix_for_arm(side: str) -> np.ndarray | None:
    """The 3×3 session_vr_to_robot matrix for one arm, or None if not saved
    OR if the saved data is malformed (wrong shape, bad values)."""
    data = read_for_arm(side)
    if not data:
        return None
    return _valid_matrix(data.get("session_vr_to_robot"), side=side, label="session_vr_to_robot")


def robot_verification_entry(data: dict[str, Any] | None) -> dict[str, Any]:
    """Return the robot-verification block from an arm entry.

    New files store this separately under `robot_verification`. Older files
    flattened these keys into the arm entry with `calibration_mode:
    robot_verified`; keep accepting that shape so existing profiles keep loading.
    """
    if not isinstance(data, dict) or not data:
        return {}
    nested = data.get("robot_verification")
    if isinstance(nested, dict):
        return dict(nested)
    if data.get("calibration_mode") != "robot_verified":
        return {}
    return {
        "calibration_mode": "robot_verified",
        "teleop_source": data.get("teleop_source"),
        "coordinate_frame": data.get("coordinate_frame"),
        "base_vr_direction_matrix": data.get(
            "base_vr_direction_matrix",
            data.get("session_vr_to_robot"),
        ),
        "verified_vr_to_robot_matrix": data.get("verified_vr_to_robot_matrix"),
        "translation_vr_to_robot_matrix": data.get("translation_vr_to_robot_matrix"),
        "translation_scale": data.get("translation_scale", 1.0),
        "fit_error_cm": data.get("fit_error_cm"),
        "calibration_quality": data.get("calibration_quality"),
        "verified_at": data.get("verified_at"),
        "low_scale_test_completed": data.get("low_scale_test_completed", False),
        "robot_verified_samples": data.get("robot_verified_samples") or [],
        "robot_verified_sample_residuals": data.get("robot_verified_sample_residuals") or [],
    }


def robot_verification_for_arm(side: str) -> dict[str, Any] | None:
    """Robot-verification data for one arm, separate from VR direction data."""
    entry = robot_verification_entry(read_for_arm(side))
    return entry or None


def verified_matrix_for_arm(side: str) -> np.ndarray | None:
    """Return the explicitly verified matrix if the arm has passed robot verification."""
    data = robot_verification_for_arm(side)
    if not data or data.get("calibration_mode") != "robot_verified":
        return None
    return _valid_matrix(
        data.get("verified_vr_to_robot_matrix"),
        side=side,
        label="verified_vr_to_robot_matrix",
    )


def translation_scale_for_arm(side: str) -> float:
    """Robot/VR translation scale learned by robot verification. Defaults to 1."""
    data = robot_verification_for_arm(side) or {}
    try:
        scale = float(data.get("translation_scale", 1.0))
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(scale) or scale <= 0:
        return 1.0
    return max(0.05, min(5.0, scale))


def write_for_arm(side: str, matrix: np.ndarray,
                   forward_motion_m: float = 0.0,
                   up_motion_m: float = 0.0,
                   left_motion_m: float = 0.0,
                   invert_lateral: bool | None = None,
                   confidence: str = "good",
                   wrist_pitch_anchor_local: tuple[float, float, float] | None = None,
                   wrist_roll_anchor_local: tuple[float, float, float] | None = None,
                   coordinate_frame: str = "quest_operator_frame",
                   ) -> None:
    """Persist one arm's calibration. Preserves other arms' entries by reading
    the file first, mutating, and writing back.

    Notes:
      - The `vr.invert_lateral_<side>` override toggle lives in
        config/xlerobot.yaml's `vr:` section, not here.
      - Wrist-motor polarity (`vr.wrist_motor_polarity` in xlerobot.yaml) is
        hardware configuration — NOT a wizard output — so we deliberately do
        not write old runtime `wrist_flex_sign` / `wrist_roll_sign` keys. Any
        stale entries left over from older versions are dropped on re-write.
      - `wrist_pitch_anchor_local` / `wrist_roll_anchor_local` are
        empirical raw controller-anchor unit rotvecs captured by the wrist
        wizard. Missing or invalid wrist axes block live direct-wrist teleop.
        Older left-arm files without `wrist_canonical_frame` are converted on
        load for backward compatibility.
    """
    doc = _normalized_doc()
    existing = dict(_active_profile_entry(doc))
    M = _orthonormalize_matrix(np.array(matrix, dtype=float))
    def _validated_wrist_axis(axis: tuple[float, float, float] | None, name: str) -> list[float]:
        if axis is None:
            raise ValueError(f"{name} is required for direct wrist teleop")
        if not _valid_wrist_axis(axis):
            raise ValueError(f"{name} must be a finite non-zero 3-vector")
        arr = np.asarray(axis, dtype=float)
        return [float(v) for v in arr]

    wrist_pitch_axis = _validated_wrist_axis(wrist_pitch_anchor_local, "wrist_pitch_anchor_local")
    wrist_roll_axis = _validated_wrist_axis(wrist_roll_anchor_local, "wrist_roll_anchor_local")
    entry: dict[str, Any] = {
        "calibration_mode": "vr_direction",
        "teleop_source": "native_quest",
        "coordinate_frame": str(coordinate_frame or "quest_operator_frame"),
        "wrist_canonical_frame": "raw_controller_anchor_local",
        "session_vr_to_robot": [[float(v) for v in row] for row in M],
        "calibrated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "forward_motion_m": float(forward_motion_m),
        "up_motion_m": float(up_motion_m),
        "left_motion_m": float(left_motion_m),
        "confidence": str(confidence or "good"),
        "wrist_pitch_anchor_local": wrist_pitch_axis,
        "wrist_roll_anchor_local": wrist_roll_axis,
    }
    if invert_lateral is not None:
        entry["invert_lateral"] = bool(invert_lateral)
    existing[side] = entry
    doc["profiles"][doc["active_profile"]] = existing
    _write_doc(doc)
    log.info("[%s] VR calibration saved to %s profile=%s", side, CFG_PATH, doc["active_profile"])


def write_robot_verification_for_arm(
    side: str,
    base_matrix: np.ndarray,
    verified_matrix: np.ndarray,
    translation_matrix: np.ndarray,
    translation_scale: float,
    fit_error_cm: float,
    sample_residuals: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    quality: str,
    low_scale_test_completed: bool = False,
    coordinate_frame: str = "quest_operator_frame",
) -> None:
    """Persist the robot-verified refinement layer for one arm.

    The top-level arm entry remains the VR-direction calibration. The
    robot-verified solve is stored as a separate `robot_verification` block so
    VR direction capture and robot verification are not conflated.
    """
    doc = _normalized_doc()
    existing = dict(_active_profile_entry(doc))
    entry = dict(existing.get(side) or {})
    base = _orthonormalize_matrix(np.array(base_matrix, dtype=float))
    verified = _orthonormalize_matrix(np.array(verified_matrix, dtype=float))
    translation = np.array(translation_matrix, dtype=float)
    if translation.shape != (3, 3) or not np.all(np.isfinite(translation)):
        raise ValueError("translation_matrix must be a finite 3x3 matrix")
    for key in (
        "base_vr_direction_matrix",
        "verified_vr_to_robot_matrix",
        "translation_vr_to_robot_matrix",
        "translation_scale",
        "fit_error_cm",
        "calibration_quality",
        "verified_at",
        "low_scale_test_completed",
        "robot_verified_samples",
        "robot_verified_sample_residuals",
    ):
        entry.pop(key, None)
    entry["calibration_mode"] = "vr_direction"
    entry["teleop_source"] = "native_quest"
    entry["coordinate_frame"] = str(coordinate_frame or "quest_operator_frame")
    entry["session_vr_to_robot"] = [[float(v) for v in row] for row in base]
    entry["robot_verification"] = {
        "calibration_mode": "robot_verified",
        "teleop_source": "native_quest",
        "coordinate_frame": str(coordinate_frame or "quest_operator_frame"),
        "base_vr_direction_matrix": [[float(v) for v in row] for row in base],
        "verified_vr_to_robot_matrix": [[float(v) for v in row] for row in verified],
        "translation_vr_to_robot_matrix": [[float(v) for v in row] for row in translation],
        "translation_scale": float(translation_scale),
        "fit_error_cm": float(fit_error_cm),
        "calibration_quality": str(quality or "unknown"),
        "verified_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "low_scale_test_completed": bool(low_scale_test_completed),
        "robot_verified_samples": samples,
        "robot_verified_sample_residuals": sample_residuals,
    }
    existing[side] = entry
    doc["profiles"][doc["active_profile"]] = existing
    _write_doc(doc)
    log.info("[%s] robot-verified VR calibration saved to %s profile=%s", side, CFG_PATH, doc["active_profile"])


def set_robot_verification_test_completed(side: str, completed: bool) -> None:
    """Persist whether the operator accepted the low-scale verified teleop test."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    doc = _normalized_doc()
    existing = dict(_active_profile_entry(doc))
    entry = dict(existing.get(side) or {})
    robot = robot_verification_entry(entry)
    if not robot or robot.get("calibration_mode") != "robot_verified":
        raise ValueError(f"{side} has no robot verification to update")
    robot["low_scale_test_completed"] = bool(completed)
    entry["robot_verification"] = robot
    existing[side] = entry
    doc["profiles"][doc["active_profile"]] = existing
    _write_doc(doc)


def status() -> dict[str, dict[str, Any]]:
    """Status dict for the UI — per side, indicates whether saved + when.

    Wrist-sign keys are deliberately omitted — they live in xlerobot.yaml and
    are surfaced separately via the per-arm runtime state (`arm.wrist_*_sign`).
    """
    saved = read_all()
    out: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        data = saved.get(side) or {}
        robot_data = robot_verification_entry(data)
        robot_quality = robot_data.get("calibration_quality")
        robot_verified = robot_data.get("calibration_mode") == "robot_verified"
        has_wrist_pitch = _valid_wrist_axis(data.get("wrist_pitch_anchor_local"))
        has_wrist_roll = _valid_wrist_axis(data.get("wrist_roll_anchor_local"))
        wrist_axes_ready = has_wrist_pitch and has_wrist_roll
        out[side] = {
            "saved": "session_vr_to_robot" in data,
            "calibration_mode": data.get("calibration_mode", "legacy" if data else None),
            "teleop_source": data.get("teleop_source", "legacy" if data else None),
            "coordinate_frame": data.get("coordinate_frame"),
            "calibrated_at": data.get("calibrated_at"),
            "forward_motion_m": float(data.get("forward_motion_m", 0.0)),
            "up_motion_m": float(data.get("up_motion_m", 0.0)),
            "left_motion_m": float(data.get("left_motion_m", 0.0)),
            "invert_lateral": data.get("invert_lateral"),
            "confidence": data.get("confidence", "unknown"),
            "has_empirical_wrist_canonical": wrist_axes_ready,
            "has_empirical_wrist_pitch_canonical": has_wrist_pitch,
            "has_empirical_wrist_roll_canonical": has_wrist_roll,
            "wrist_axes_ready": wrist_axes_ready,
            "robot_verified": robot_verified,
            "verified_at": robot_data.get("verified_at"),
            "verified_coordinate_frame": robot_data.get("coordinate_frame"),
            "fit_error_cm": robot_data.get("fit_error_cm"),
            "translation_scale": float(robot_data.get("translation_scale", 1.0)),
            "calibration_quality": robot_quality,
            "needs_recapture": robot_quality in {"warn", "poor", "needs_recapture"},
            "verified_sample_count": len(robot_data.get("robot_verified_samples") or []),
        }
    return out


def profile_status() -> dict[str, Any]:
    doc = _normalized_doc()
    profiles = doc.get("profiles") or {}
    rows = []
    for name, profile in profiles.items():
        profile = profile if isinstance(profile, dict) else {}
        left = profile.get("left") or {}
        right = profile.get("right") or {}
        left_robot = robot_verification_entry(left)
        right_robot = robot_verification_entry(right)
        timestamps = [
            v for v in (
                left_robot.get("verified_at"), left.get("calibrated_at"),
                right_robot.get("verified_at"), right.get("calibrated_at"),
            )
            if v
        ]
        rows.append({
            "name": name,
            "left_saved": "session_vr_to_robot" in left,
            "right_saved": "session_vr_to_robot" in right,
            "left_robot_verified": left_robot.get("calibration_mode") == "robot_verified",
            "right_robot_verified": right_robot.get("calibration_mode") == "robot_verified",
            "updated_at": max(timestamps) if timestamps else None,
        })
    rows.sort(key=lambda p: (p["name"] != doc["active_profile"], p["name"]))
    return {
        "active_profile": doc["active_profile"],
        "profiles": rows,
    }


def set_active_profile(name: str) -> str:
    profile = _normalize_profile_name(name)
    doc = _normalized_doc()
    if profile not in doc["profiles"]:
        raise ValueError(f"unknown calibration profile: {profile}")
    doc["active_profile"] = profile
    _write_doc(doc)
    log.info("active VR calibration profile set to %s", profile)
    return profile


def create_profile(name: str, *, copy_from: str | None = None) -> str:
    profile = _normalize_profile_name(name)
    doc = _normalized_doc()
    if profile in doc["profiles"]:
        raise ValueError(f"calibration profile already exists: {profile}")
    if copy_from is not None:
        src = _normalize_profile_name(copy_from)
        if src not in doc["profiles"]:
            raise ValueError(f"unknown source calibration profile: {src}")
        import copy
        doc["profiles"][profile] = copy.deepcopy(doc["profiles"].get(src) or {})
    else:
        doc["profiles"][profile] = {}
    doc["active_profile"] = profile
    _write_doc(doc)
    log.info("created VR calibration profile %s copy_from=%s", profile, copy_from)
    return profile


def delete_profile(name: str) -> str:
    profile = _normalize_profile_name(name)
    doc = _normalized_doc()
    if profile not in doc["profiles"]:
        raise ValueError(f"unknown calibration profile: {profile}")
    if len(doc["profiles"]) <= 1:
        raise ValueError("cannot delete the only calibration profile")
    del doc["profiles"][profile]
    if doc["active_profile"] == profile:
        doc["active_profile"] = next(iter(doc["profiles"].keys()))
    _write_doc(doc)
    log.info("deleted VR calibration profile %s; active=%s", profile, doc["active_profile"])
    return doc["active_profile"]


def _orthonormalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the nearest proper 3D rotation matrix."""
    u, _, vt = np.linalg.svd(matrix)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return rot


def read_invert_lateral_flags() -> dict[str, bool]:
    """Read per-arm `vr.invert_lateral_<side>` flags from config/xlerobot.yaml.
    Returns {'left': bool, 'right': bool}. Missing keys default to False."""
    return {s: bool(_yaml_invert_raw().get(s)) for s in ("left", "right")}


def read_invert_lateral_overrides() -> dict[str, bool]:
    """For each side, is the YAML flag EXPLICITLY set (so it should override
    the calibration wizard's auto-decision)?

    Distinguishes 'key absent / null' (wizard decides) from 'key present with
    bool value' (manual override; wizard skips its decision). The wizard's
    step-3 lateral check catches matrix-math mirroring but NOT physical motor
    mirroring (mirror-mounted arm with reversed sign convention), so users
    need a manual escape hatch."""
    raw = _yaml_invert_raw()
    return {s: (raw.get(s) is not None) for s in ("left", "right")}


def _yaml_invert_raw() -> dict[str, Any]:
    """Internal: return the raw values (or None if absent) for invert flags."""
    cfg_path = REPO_ROOT / "config" / "xlerobot.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception as e:
        log.warning("could not read %s for invert flags: %s", cfg_path, e)
        return {"left": None, "right": None}
    vr = cfg.get("vr") or {}
    return {
        "left":  vr.get("invert_lateral_left"),    # None / True / False
        "right": vr.get("invert_lateral_right"),
    }
