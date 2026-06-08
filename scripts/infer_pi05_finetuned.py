#!/usr/bin/env python3
"""Run on-robot inference with a finetuned LeRobot PI0.5 checkpoint.

Loads the checkpoint locally (no OpenPI server) and drives the XLerobot
bimanual SO-101 stack — the same robot/dataset layout used during VR recording.

Prerequisites:
    The installed LeRobot package provides lerobot.robots.xlerobot.

Default camera backend is dashboard CameraStream (one V4L capture per role, same as VR recording).
Use --camera-backend lerobot only if you are not sharing USB with the dashboard and want
per-step async_read (opens a second VideoCapture per camera on the robot object).

Usage:
    uv run python scripts/infer_pi05_finetuned.py \\
        --policy-path outputs/pi05_finetune/checkpoints/005000/pretrained_model \\
        --task "Pick up the medicine and place it in the bowl" \\
        --episodes 2 --episode-time 120

    # Dry-run homing only (no policy load, no inference):
    uv run python scripts/infer_pi05_finetuned.py --dry-run-home
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openpibot.server.runtime.homing import JointHomingController

# Must match openpibot/server/runtime/dataset.py JOINT_ORDER (LeRobot action / observation.state).
_JOINTS_PER_ARM = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
JOINT_ORDER: list[str] = [
    f"{side}_arm_{j}" for side in ("left", "right") for j in _JOINTS_PER_ARM
]

# Per-tick joint caps (degrees) — match openpibot/server/runtime/vr_teleop.py PER_TICK_DEG_CAPS.
_PER_TICK_DEG_CAPS: dict[str, float] = {
    "shoulder_pan": 5.0,
    "shoulder_lift": 5.0,
    "elbow_flex": 5.0,
    "wrist_flex": 6.0,
    "wrist_roll": 10.0,
    "gripper": 15.0,
}
_JOINT_DEADBAND_DEG: dict[str, float] = {
    "shoulder_pan": 0.18,
    "shoulder_lift": 0.18,
    "elbow_flex": 0.18,
    "wrist_flex": 0.25,
    "wrist_roll": 0.25,
    "gripper": 0.0,
}
_FINAL_SMOOTHING_BYPASS = {"wrist_flex", "wrist_roll", "gripper"}
_HOMING_TOL_DEG = 0.5
_HOMING_PRESENT_TOL_DEG = 1.0
_HOMING_FINAL_DIRECT_TOL_DEG = 8.0
_HOMING_SOFT_STALL_TOL_DEG = 6.0
_HOMING_SETTLE_TICKS = 5
_HOMING_KP = 0.75
_HOMING_PROGRESS_LOG_S = 2.0
_HOMING_STALL_WINDOW_S = 3.0
_HOMING_STALL_MIN_COMMAND_DEG = 1.0
_HOMING_STALL_MAX_FEEDBACK_CHANGE_DEG = 0.25
# VR recording uses KP=1.0 → action label is rate-limited absolute command (see _shape_action_like_recording).
_DEFAULT_VR_KP = 1.0


def _normalize_joint_name(name: str) -> str:
    return str(name).removesuffix(".pos")


def _robot_pos_key(name: str) -> str:
    base = _normalize_joint_name(name)
    return f"{base}.pos"


def _load_yaml() -> dict[str, Any]:
    if not CONFIG_YAML.is_file():
        sys.exit(f"missing {CONFIG_YAML}")
    return yaml.safe_load(CONFIG_YAML.read_text()) or {}


def _configured_motor_ports(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    robot_cfg = cfg.get("robot") if isinstance(cfg.get("robot"), dict) else {}
    ports: list[tuple[str, str]] = []
    for label, key in (
        ("left/base bus", "port_left_base"),
        ("right/head bus", "port_right_head"),
    ):
        value = robot_cfg.get(key)
        if value:
            ports.append((label, str(value)))
    return ports


def _run_motor_port_preflight(cfg: dict[str, Any]) -> None:
    """Fail before model load if motor serial ports are missing or already owned."""
    ports = _configured_motor_ports(cfg)
    if not ports:
        return

    busy: list[str] = []
    missing: list[str] = []
    print("Motor port preflight:")
    for label, port in ports:
        path = pathlib.Path(port)
        if not path.exists():
            missing.append(f"{label}: {port}")
            print(f"  {label:14} MISSING\n    {port}")
            continue
        print(f"  {label:14} ok\n    {port}")
        try:
            proc = subprocess.run(
                ["fuser", "-v", port],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        combined = ((proc.stdout or "") + (proc.stderr or "")).strip()
        if proc.returncode == 0 and combined:
            busy.append(f"{label}: {port}\n{combined[:800]}")

    if missing:
        sys.exit(
            "Motor port preflight failed: configured serial port(s) do not exist.\n"
            + "\n".join(f"  {item}" for item in missing)
        )
    if busy:
        sys.exit(
            "Motor port preflight failed: another process is using the robot motor bus.\n"
            "Stop the dashboard backend / VR teleop / older inference run, then re-run inference.\n\n"
            + "\n\n".join(busy)
        )


def _load_vr_control_shaping(cfg: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """Load the command caps/deadbands used by VR recording.

    Inference should shape policy actions like the dataset labels: per-tick
    caps by joint, then tiny-command deadband by joint. Missing/invalid entries
    keep the compiled defaults above.
    """
    vr_cfg = cfg.get("vr") if isinstance(cfg.get("vr"), dict) else {}
    caps = dict(_PER_TICK_DEG_CAPS)
    raw_caps = vr_cfg.get("joint_deg_caps") if isinstance(vr_cfg, dict) else None
    if isinstance(raw_caps, dict):
        for joint, value in raw_caps.items():
            if joint in caps:
                try:
                    caps[joint] = max(0.1, min(30.0, float(value)))
                except (TypeError, ValueError):
                    pass

    deadbands = dict(_JOINT_DEADBAND_DEG)
    raw_deadbands = vr_cfg.get("joint_command_deadband_deg") if isinstance(vr_cfg, dict) else None
    if isinstance(raw_deadbands, (int, float)):
        value = max(0.0, min(5.0, float(raw_deadbands)))
        for joint in deadbands:
            if joint != "gripper":
                deadbands[joint] = value
    elif isinstance(raw_deadbands, dict):
        for joint, value in raw_deadbands.items():
            if joint in deadbands:
                try:
                    deadbands[joint] = max(0.0, min(5.0, float(value)))
                except (TypeError, ValueError):
                    pass
    return caps, deadbands


def _load_homing_tolerances(cfg: dict[str, Any]) -> tuple[float, float]:
    vr_cfg = cfg.get("vr") if isinstance(cfg.get("vr"), dict) else {}
    present_tol = _HOMING_PRESENT_TOL_DEG
    raw_present_tol = vr_cfg.get("homing_present_tolerance_deg")
    if raw_present_tol is not None:
        try:
            present_tol = max(0.25, min(10.0, float(raw_present_tol)))
        except (TypeError, ValueError):
            pass

    soft_stall_tol = max(_HOMING_SOFT_STALL_TOL_DEG, present_tol + 2.0)
    raw_soft_stall_tol = vr_cfg.get("homing_soft_stall_tolerance_deg")
    if raw_soft_stall_tol is not None:
        try:
            soft_stall_tol = max(present_tol, min(15.0, float(raw_soft_stall_tol)))
        except (TypeError, ValueError):
            pass
    return present_tol, soft_stall_tol


def _parse_args() -> argparse.Namespace:
    cfg = _load_yaml()
    ds = cfg.get("dataset") or {}
    pi05 = cfg.get("pi05") or {}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--policy-path",
        default=None,
        help="Path to finetuned checkpoint (.../pretrained_model). Not needed with --dry-run-home.",
    )
    p.add_argument(
        "--task",
        default=None,
        help="Natural-language task prompt. Not needed with --dry-run-home.",
    )
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument(
        "--episode-time",
        type=int,
        default=120,
        help=(
            "Wall-clock seconds for the whole episode (pre-home, control loop, and "
            "post-home share this budget). Whichever limit is hit first with --episode-steps."
        ),
    )
    p.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        help=(
            "Max control-loop steps per episode (includes settle steps). "
            "Use this for step budgets (e.g. 2000 steps @ 30 Hz ≈ 67 s). "
            "Ends when this OR --episode-time is reached."
        ),
    )
    p.add_argument(
        "--stop-on-episode-error",
        action="store_true",
        help="Abort remaining episodes after a failed episode (default: home and continue).",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument(
        "--fps",
        type=int,
        default=int(ds.get("fps", pi05.get("control_fps", 30))),
        help="Control loop Hz (default: dataset.fps, usually 30 — match training, not pi05.control_fps).",
    )
    p.add_argument(
        "--action-horizon",
        type=int,
        default=int(pi05.get("action_horizon", 50)),
        help="Max chunk size from policy (upper bound).",
    )
    p.add_argument(
        "--open-loop-steps",
        type=int,
        default=35,
        help=(
            "Policy chunk steps before scheduled re-infer (default: 35 @ 30Hz ≈ 1.2s). "
            "Higher = fewer replans / smoother; very low values (e.g. 5) jitter at boundaries."
        ),
    )
    p.add_argument(
        "--replan-on-miss-deg",
        type=float,
        default=0.0,
        help=(
            "Optional: when max |present - last sent command| exceeds this (degrees) for "
            "--replan-miss-steps ticks, drop the open-loop chunk (re-infer next tick). "
            "0 disables. Do not compare raw policy targets — that false-triggers during motion."
        ),
    )
    p.add_argument(
        "--replan-miss-steps",
        type=int,
        default=2,
        help="Consecutive ticks over --replan-on-miss-deg before an early replan.",
    )
    p.add_argument(
        "--settle-steps",
        type=int,
        default=60,
        help=(
            "Hold present pose for this many control ticks after homing (default: 60 @ 30Hz = 2s). "
            "Matches VR demos where the operator pauses at home before moving."
        ),
    )
    p.add_argument(
        "--replan-blend",
        type=float,
        default=0.2,
        help=(
            "Blend factor for the first action after each new chunk [0..1]. "
            "Lower = smoother across replans; 1.0 disables blending."
        ),
    )
    p.add_argument(
        "--phase1-task",
        default=None,
        help=(
            "Optional language prompt for the first segment (e.g. reach medicine only). "
            "Use with --phase1-sec."
        ),
    )
    p.add_argument(
        "--phase1-sec",
        type=float,
        default=0.0,
        help="Seconds to use --phase1-task before switching to --task (default: 0 = disabled).",
    )
    p.add_argument(
        "--strict-motors",
        action="store_true",
        help="Require all XLerobot motors on the bus (default: lenient prune).",
    )
    p.add_argument(
        "--skip-home",
        action="store_true",
        help="Do not move to saved home pose before inference.",
    )
    p.add_argument(
        "--home-before-episode",
        action=argparse.BooleanOptionalAction,
        default=bool(ds.get("home_before_episode", True)),
        help="Return to home pose at the start of each episode (default: from dataset config).",
    )
    p.add_argument(
        "--skip-home-after-episode",
        action="store_true",
        help="Skip post-episode homing (faster; next episode may still home at start).",
    )
    p.add_argument(
        "--home-timeout",
        type=float,
        default=60.0,
        help="Max seconds to spend homing before continuing anyway.",
    )
    p.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="Max joint change (deg) per policy command (default: robot.max_relative_target in yaml).",
    )
    p.add_argument(
        "--policy-ema-alpha",
        type=float,
        default=0.36,
        help=(
            "EMA on raw policy targets before VR shaping [0..1]. "
            "Lower is smoother; 1.0 disables."
        ),
    )
    p.add_argument(
        "--command-ema-alpha",
        type=float,
        default=0.2,
        help=(
            "EMA smoothing for final command [0..1]. Lower is smoother, higher is snappier. "
            "Too low (~0.14) can stall at home; too high (~0.28) reaches targets but jitters."
        ),
    )
    p.add_argument(
        "--joint-deadband-deg",
        type=float,
        default=None,
        help=(
            "Override final command deadband for every joint (deg). "
            "Default uses vr.joint_command_deadband_deg from config/xlerobot.yaml."
        ),
    )
    p.add_argument(
        "--clamp-to-present",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Clamp each command vs measured joint pose (robot.max_relative_target). "
            "Default on to match the SOFollower safety layer used during recording; "
            "use --no-clamp-to-present only for controlled debugging."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned settings and exit (no robot connection).",
    )
    p.add_argument(
        "--dry-run-home",
        action="store_true",
        help="Connect and move to saved home pose only; skip policy load and inference.",
    )
    from _camera_preview_window import display_available

    p.add_argument(
        "--show-cameras",
        action=argparse.BooleanOptionalAction,
        default=display_available(),
        help=(
            "Show a live 3-camera mosaic in a background thread (reads dashboard streams only; "
            "does not block inference). Default: on when DISPLAY/WAYLAND_DISPLAY is set."
        ),
    )
    p.add_argument(
        "--preview-fps",
        type=float,
        default=15.0,
        help="Max refresh rate for --show-cameras window (default: 15).",
    )
    p.add_argument(
        "--camera-backend",
        choices=("lerobot", "dashboard"),
        default="dashboard",
        help=(
            "dashboard: shared CameraStream registry — one capture per camera (default, stable USB). "
            "lerobot: extra OpenCVCamera on the robot object (can conflict with dashboard / hub)."
        ),
    )
    args = p.parse_args()
    if args.dry_run and args.dry_run_home:
        p.error("use only one of --dry-run or --dry-run-home")
    if not args.dry_run and not args.dry_run_home:
        if not args.policy_path or not args.task:
            p.error("--policy-path and --task are required for inference")
    return args


def _ensure_xlerobot_import() -> None:
    try:
        import lerobot.robots.xlerobot  # noqa: F401
    except ImportError as e:
        sys.exit(
            "lerobot.robots.xlerobot is not installed.\n"
            "Install a LeRobot package release that provides the XLeRobot driver module, then run uv sync.\n"
            f"Original error: {e}"
        )


def _build_state_vector(obs: dict[str, Any], joint_names: list[str]) -> np.ndarray:
    return np.array([float(obs[f"{name}.pos"]) for name in joint_names], dtype=np.float32)


def _to_hwc_uint8_image(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"camera image must be rank-3, got shape={arr.shape}")
    # Accept both CHW and HWC, normalize to HWC uint8 for prepare_observation_for_inference.
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0 + 1e-6:
                arr = np.clip(arr, 0.0, 1.0)
                arr = np.rint(arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _build_observation(
    obs: dict[str, Any],
    joint_names: list[str],
) -> dict[str, np.ndarray]:
    camera_obs = {k: v for k, v in obs.items() if k in ("head", "left_wrist", "right_wrist")}
    if set(camera_obs) != {"head", "left_wrist", "right_wrist"}:
        missing = {"head", "left_wrist", "right_wrist"} - set(camera_obs)
        sys.exit(f"missing camera observations: {sorted(missing)}")
    return {
        "observation.images.head": _to_hwc_uint8_image(camera_obs["head"]),
        "observation.images.left_wrist": _to_hwc_uint8_image(camera_obs["left_wrist"]),
        "observation.images.right_wrist": _to_hwc_uint8_image(camera_obs["right_wrist"]),
        "observation.state": _build_state_vector(obs, joint_names),
    }


def _read_home_pose(cfg: dict[str, Any]) -> dict[str, float]:
    hp = (cfg.get("robot") or {}).get("home_pose") or {}
    if not hp:
        sys.exit(
            "robot.home_pose is empty in config/xlerobot.yaml.\n"
            "Capture a home pose first (dashboard VR Teleop → Capture home, or "
            "edit robot.home_pose in config/xlerobot.yaml)."
        )
    return {str(k): float(v) for k, v in hp.items()}


def _cap_for_joint_key(key: str, caps: dict[str, float] | None = None) -> float:
    for suffix, cap in (caps or _PER_TICK_DEG_CAPS).items():
        if f"_{suffix}.pos" in key:
            return cap
    return 2.0


def _vr_kp(cfg: dict[str, Any]) -> float:
    return float((cfg.get("vr") or {}).get("kp", _DEFAULT_VR_KP))


def _shape_action_like_recording(
    policy_targets: dict[str, float],
    present: dict[str, float],
    last_sent: dict[str, float],
    *,
    kp: float,
    caps: dict[str, float],
) -> dict[str, float]:
    """Match VR dataset labels: per-tick cap vs previous command, optional P blend.

    Recording stores `action` = motor command after the same logic in vr_teleop.py
    (not the raw policy/VR target). With kp>=0.999 this is:
        cmd = last_sent + clip(target - last_sent, -cap, cap)
    """
    shaped: dict[str, float] = {}
    for key, target in policy_targets.items():
        cap = _cap_for_joint_key(key, caps)
        prev = last_sent.get(key, present.get(key, target))
        delta = max(-cap, min(cap, target - prev))
        clamped = prev + delta
        if kp >= 0.999:
            shaped[key] = clamped
        else:
            here = present.get(key, clamped)
            shaped[key] = here + kp * (clamped - here)
    return shaped


def _clamp_max_relative(
    command: dict[str, float],
    present: dict[str, float],
    max_rel: float,
) -> dict[str, float]:
    """Clamp absolute goals vs present (same semantics as XLerobot max_relative_target)."""
    out: dict[str, float] = {}
    for key, goal in command.items():
        here = present.get(key, goal)
        delta = max(-max_rel, min(max_rel, goal - here))
        out[key] = here + delta
    return out


def _top_joint_deltas(command: dict[str, float], present: dict[str, float], top_k: int = 4) -> list[tuple[str, float]]:
    deltas = []
    for key, goal in command.items():
        here = present.get(key, goal)
        deltas.append((key, float(goal - here)))
    deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    return deltas[:top_k]


def _apply_joint_deadband(
    command: dict[str, float],
    reference: dict[str, float],
    deadbands_deg: dict[str, float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, goal in command.items():
        ref = reference.get(key, goal)
        deadband_deg = 0.0
        for suffix, value in deadbands_deg.items():
            if f"_{suffix}.pos" in key:
                deadband_deg = float(value)
                break
        out[key] = ref if abs(goal - ref) < deadband_deg else goal
    return out


def _max_tracking_error_deg(
    present: dict[str, float],
    target: dict[str, float],
) -> float:
    errs = [
        abs(float(present[f"{name}.pos"]) - float(target[f"{name}.pos"]))
        for name in JOINT_ORDER
        if f"{name}.pos" in present and f"{name}.pos" in target
    ]
    return max(errs) if errs else 0.0


def _flush_policy_action_queue(policy: Any) -> int:
    queue = getattr(policy, "_action_queue", None)
    if queue is None:
        return 0
    n = len(queue)
    if n:
        queue.clear()
    return n


def _blend_action_dict(
    new_cmd: dict[str, float],
    prev_cmd: dict[str, float],
    alpha: float,
    *,
    bypass_suffixes: set[str] | None = None,
) -> dict[str, float]:
    if alpha >= 0.999 or not prev_cmd:
        return dict(new_cmd)
    if alpha <= 0:
        return dict(prev_cmd)
    out: dict[str, float] = {}
    for key, goal in new_cmd.items():
        if bypass_suffixes and any(f"_{suffix}.pos" in key for suffix in bypass_suffixes):
            out[key] = goal
            continue
        prev = prev_cmd.get(key, goal)
        out[key] = (1.0 - alpha) * prev + alpha * goal
    return out


def _task_for_step(
    args: argparse.Namespace,
    *,
    step: int,
    settle_steps: int,
    fps: float,
) -> str:
    if args.phase1_task and args.phase1_sec > 0:
        phase1_end = settle_steps + int(args.phase1_sec * fps)
        if step < phase1_end:
            return args.phase1_task
    return args.task


def _ema_command(
    command: dict[str, float],
    prev_command: dict[str, float],
    alpha: float,
    *,
    bypass_suffixes: set[str] | None = None,
) -> dict[str, float]:
    if alpha >= 0.999:
        return dict(command)
    if alpha <= 0:
        return dict(prev_command) if prev_command else dict(command)
    out: dict[str, float] = {}
    for key, goal in command.items():
        if bypass_suffixes and any(f"_{suffix}.pos" in key for suffix in bypass_suffixes):
            out[key] = goal
            continue
        prev = prev_command.get(key, goal)
        out[key] = (1.0 - alpha) * prev + alpha * goal
    return out


def _send_positions(
    robot: Any,
    command: dict[str, float],
    *,
    present: dict[str, float] | None = None,
) -> dict[str, float]:
    """Send joint goals; clamp here so older XLerobot send_action bugs are avoided."""
    max_rel = getattr(robot.config, "max_relative_target", None)
    if present is not None and max_rel is not None:
        command = _clamp_max_relative(command, present, float(max_rel))
    saved_max_rel = robot.config.max_relative_target
    robot.config.max_relative_target = None
    try:
        sent = robot.send_action(command)
    finally:
        robot.config.max_relative_target = saved_max_rel
    sent_command = {str(k): float(v) for k, v in (sent or command).items()}
    missing = sorted(set(command) - set(sent_command))
    if missing:
        raise RuntimeError(f"robot.send_action ignored {len(missing)} command key(s): {missing}")
    return sent_command


def _connect_robot_with_retries(
    robot: Any,
    *,
    attempts: int = 4,
    retry_sleep_s: float = 1.5,
) -> None:
    """Connect robot/cameras with retry for transient OpenCV warmup failures."""
    last_err: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            robot.connect(calibrate=False)
            if attempt > 1:
                print(f"Robot connected on retry {attempt}/{attempts}.")
            return
        except Exception as e:  # pragma: no cover - hardware path
            last_err = e
            msg = str(e)
            camera_hint = "OpenCVCamera" in msg or "Timed out waiting for frame from camera" in msg
            print(
                f"Connect attempt {attempt}/{attempts} failed: {e.__class__.__name__}: {msg}"
                + (" (camera warmup/read issue)" if camera_hint else "")
            )
            try:
                robot.disconnect()
            except Exception:
                pass
            if attempt < attempts:
                time.sleep(retry_sleep_s)
    assert last_err is not None
    raise last_err


def _read_motor_observation(robot: Any, *, attempts: int = 5) -> dict[str, Any]:
    """Motor-only observation with retries (USB contention during homing + cameras)."""
    last_err: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return robot.get_observation(include_cameras=False)
        except ConnectionError as e:
            last_err = e
            if attempt < attempts:
                time.sleep(0.05 * attempt)
    assert last_err is not None
    raise last_err


def go_to_home_pose(
    robot: Any,
    home_pose: dict[str, float],
    *,
    fps: float,
    timeout_s: float,
    precise_sleep: Any,
    caps: dict[str, float] | None = None,
    present_tolerance_deg: float = _HOMING_PRESENT_TOL_DEG,
    soft_stall_tolerance_deg: float = _HOMING_SOFT_STALL_TOL_DEG,
    episode_deadline: float | None = None,
) -> None:
    """Rate-limited move to `robot.home_pose` joint targets (degrees)."""
    targets = {f"{name}.pos": deg for name, deg in home_pose.items()}
    keys = list(targets.keys())
    dt = 1.0 / fps
    deadline = time.perf_counter() + timeout_s
    if episode_deadline is not None:
        deadline = min(deadline, episode_deadline)
    paused_lerobot_cams = False
    controller: JointHomingController | None = None
    last_step = None
    start_t = time.perf_counter()
    next_progress_log_t = start_t
    stall_window_t = start_t
    stall_window_present: dict[str, float] | None = None

    print(f"Homing to saved pose ({len(keys)} joints, timeout={timeout_s:.0f}s)...")
    if getattr(robot, "cameras", None):
        from _opencv_camera_patch import pause_robot_cameras

        pause_robot_cameras(robot)
        paused_lerobot_cams = True
    try:
        while time.perf_counter() < deadline:
            loop_start = time.perf_counter()
            obs = _read_motor_observation(robot)
            missing_obs = sorted(set(keys) - set(obs))
            if missing_obs:
                raise RuntimeError(
                    "homing cannot read motor feedback for "
                    f"{len(missing_obs)} joint(s): {missing_obs}"
                )
            present = {k: float(obs[k]) for k in keys}
            if stall_window_present is None:
                stall_window_present = dict(present)
                stall_window_t = loop_start
            if controller is None:
                controller = JointHomingController(
                    targets=targets,
                    present=present,
                    cap_for_key=lambda key: _cap_for_joint_key(key, caps),
                    kp=_HOMING_KP,
                    command_tolerance_deg=_HOMING_TOL_DEG,
                    present_tolerance_deg=present_tolerance_deg,
                    final_direct_tolerance_deg=max(
                        _HOMING_FINAL_DIRECT_TOL_DEG,
                        present_tolerance_deg,
                    ),
                    settle_ticks=_HOMING_SETTLE_TICKS,
                )

            step = controller.step(present)
            last_step = step
            sent_command = _send_positions(robot, step.command, present=present)
            max_sent_delta = max(
                abs(float(sent_command[k]) - float(present[k])) for k in keys
            )

            now = time.perf_counter()
            if now >= next_progress_log_t:
                joint = f" worst={step.worst_present_joint}" if step.worst_present_joint else ""
                print(
                    f"  Homing {now - start_t:.1f}s: command_error="
                    f"{step.max_command_error_deg:.2f} deg, feedback_error="
                    f"{step.max_present_error_deg:.2f} deg{joint}, "
                    f"max_sent_delta={max_sent_delta:.2f} deg, settled={step.settled}"
                )
                for name, delta in _top_joint_deltas(sent_command, present, top_k=3):
                    print(f"    homing sent delta {name}: {delta:+.2f} deg")
                next_progress_log_t = now + _HOMING_PROGRESS_LOG_S

            if stall_window_present is not None and now - stall_window_t >= _HOMING_STALL_WINDOW_S:
                max_feedback_change = max(
                    abs(float(present[k]) - float(stall_window_present.get(k, present[k])))
                    for k in keys
                )
                if (
                    max_sent_delta >= _HOMING_STALL_MIN_COMMAND_DEG
                    and step.max_present_error_deg > present_tolerance_deg
                    and max_feedback_change <= _HOMING_STALL_MAX_FEEDBACK_CHANGE_DEG
                ):
                    top_deltas = ", ".join(
                        f"{name} {delta:+.2f} deg"
                        for name, delta in _top_joint_deltas(sent_command, present, top_k=3)
                    )
                    if step.max_present_error_deg <= soft_stall_tolerance_deg:
                        print(
                            "Warning: homing stalled near saved home; continuing "
                            f"(max feedback error {step.max_present_error_deg:.2f} deg <= "
                            f"soft tolerance {soft_stall_tolerance_deg:.2f} deg; "
                            f"max feedback change {max_feedback_change:.2f} deg over "
                            f"{now - stall_window_t:.1f}s; {top_deltas})."
                        )
                        return
                    raise RuntimeError(
                        "homing is commanding motion but motor feedback is not changing "
                        f"(max feedback change {max_feedback_change:.2f} deg over "
                        f"{now - stall_window_t:.1f}s; {top_deltas}). "
                        "Stop the dashboard backend/VR teleop or any other process using "
                        "the motor serial ports, then retry."
                    )
                stall_window_present = dict(present)
                stall_window_t = now

            if step.settled:
                print(f"Home pose reached (max feedback error {step.max_present_error_deg:.2f} deg).")
                return

            remaining = dt - (time.perf_counter() - loop_start)
            if remaining > 0:
                precise_sleep(remaining)

        if episode_deadline is not None and _episode_timed_out(episode_deadline):
            print("Warning: homing stopped (episode wall-clock limit).")
        else:
            if last_step is None:
                print("Warning: homing timed out before a motor observation was available; continuing anyway.")
            else:
                joint = f" at {last_step.worst_present_joint}" if last_step.worst_present_joint else ""
                print(
                    "Warning: homing timed out; continuing anyway "
                    f"(max feedback error {last_step.max_present_error_deg:.2f} deg{joint})."
                )
    finally:
        if paused_lerobot_cams:
            from _opencv_camera_patch import resume_lerobot_cameras

            resume_lerobot_cameras(robot)


def _episode_seconds_left(episode_deadline: float) -> float:
    return episode_deadline - time.perf_counter()


def _episode_timed_out(episode_deadline: float) -> bool:
    return time.perf_counter() >= episode_deadline


def _episode_limit_reason(
    step: int,
    episode_deadline: float,
    args: argparse.Namespace,
) -> str | None:
    if _episode_timed_out(episode_deadline):
        return f"wall-clock limit ({args.episode_time}s)"
    if args.episode_steps is not None and step >= args.episode_steps:
        return f"step limit ({args.episode_steps} steps)"
    return None


def _reset_policy_episode_state(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    *,
    open_loop_steps: int,
    last_sent: dict[str, float],
    last_filtered: dict[str, float],
    last_policy_smoothed: dict[str, float],
    reason: str,
) -> None:
    """Clear PI0.5 action queue, processor state, and command smoothers."""
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    last_sent.clear()
    last_filtered.clear()
    last_policy_smoothed.clear()
    policy.config.n_action_steps = min(
        int(open_loop_steps),
        int(policy.config.chunk_size),
    )
    print(f"  Policy reset ({reason}).")


def _maybe_log_episode_progress(
    *,
    episode_deadline: float,
    episode_time_s: float,
    step: int,
    last_log_t: float,
    interval_s: float = 30.0,
) -> float:
    """Print elapsed/remaining every interval_s; return updated last_log_t."""
    now = time.perf_counter()
    if now - last_log_t < interval_s:
        return last_log_t
    elapsed = episode_time_s - _episode_seconds_left(episode_deadline)
    left = max(0.0, _episode_seconds_left(episode_deadline))
    print(
        f"  Episode progress: {elapsed:.0f}s / {episode_time_s}s "
        f"({left:.0f}s left), step {step}"
    )
    return now


def _brief_hold_pose(
    robot: Any,
    hold_cmd: dict[str, float],
    *,
    fps: float,
    hold_s: float,
    precise_sleep: Any,
) -> None:
    """Send the same pose for a short time so the arms decelerate before homing."""
    n = max(1, int(hold_s * fps))
    for _ in range(n):
        _send_positions(robot, hold_cmd, present=hold_cmd)
        precise_sleep(1.0 / fps)


def _safe_finish_episode(
    robot: Any,
    home_pose: dict[str, float],
    *,
    reason: str,
    fps: float,
    home_timeout_s: float,
    precise_sleep: Any,
    caps: dict[str, float],
    homing_present_tolerance_deg: float,
    homing_soft_stall_tolerance_deg: float,
    hold_cmd: dict[str, float] | None,
    hold_s: float = 0.5,
    episode_deadline: float | None = None,
    skip_home_after: bool = False,
) -> None:
    """Stop cleanly: brief hold, then home (if configured) before the next episode."""
    print(f"Episode stop: {reason}")
    if hold_cmd and (
        episode_deadline is None or _episode_seconds_left(episode_deadline) > hold_s + 1.0
    ):
        try:
            _brief_hold_pose(
                robot, hold_cmd, fps=fps, hold_s=hold_s, precise_sleep=precise_sleep
            )
        except Exception as exc:
            print(f"  Warning: hold pose failed: {exc}")
    if home_pose and not skip_home_after:
        if episode_deadline is not None and _episode_seconds_left(episode_deadline) < 3.0:
            print("  Skipping post-episode homing (episode wall-clock limit).")
        else:
            post_home_timeout = home_timeout_s
            if episode_deadline is not None:
                post_home_timeout = min(
                    home_timeout_s, max(1.0, _episode_seconds_left(episode_deadline) - 1.0)
                )
            try:
                go_to_home_pose(
                    robot,
                    home_pose,
                    fps=fps,
                    timeout_s=post_home_timeout,
                    precise_sleep=precise_sleep,
                    caps=caps,
                    present_tolerance_deg=homing_present_tolerance_deg,
                    soft_stall_tolerance_deg=homing_soft_stall_tolerance_deg,
                    episode_deadline=episode_deadline,
                )
            except Exception as exc:
                print(f"  Warning: post-episode homing failed: {exc}")


def _run_home_only(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """Connect, homing only, disconnect — no policy."""
    home_pose = _read_home_pose(cfg)
    caps, _deadbands = _load_vr_control_shaping(cfg)
    homing_present_tol, homing_soft_stall_tol = _load_homing_tolerances(cfg)
    _run_motor_port_preflight(cfg)
    _ensure_xlerobot_import()
    from _xlerobot_loader import make_config, patch_motors_bus_lenient
    from lerobot.robots.xlerobot import XLerobot
    from lerobot.utils.robot_utils import precise_sleep

    if not args.strict_motors:
        patch_motors_bus_lenient()

    robot = XLerobot(make_config(robot_id="xlerobot", use_cameras=False))
    _connect_robot_with_retries(robot)
    try:
        go_to_home_pose(
            robot,
            home_pose,
            fps=float(args.fps),
            timeout_s=float(args.home_timeout),
            precise_sleep=precise_sleep,
            caps=caps,
            present_tolerance_deg=homing_present_tol,
            soft_stall_tolerance_deg=homing_soft_stall_tol,
        )
        print("Dry-run home complete.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        sys.exit(f"Dry-run home failed: {exc}")
    finally:
        robot.disconnect()


def _verify_policy_checkpoint(policy_path: pathlib.Path, policy: Any) -> None:
    """Log that a local finetuned checkpoint (not only base weights) is loaded."""
    weights = policy_path / "model.safetensors"
    if not weights.is_file():
        sys.exit(f"missing finetuned weights: {weights}")
    train_step = policy_path.parent / "training_state" / "training_step.json"
    step = "unknown"
    if train_step.is_file():
        step = str(json.loads(train_step.read_text()).get("step", "?"))
    print(f"  Weights file      : {weights} ({weights.stat().st_size // 1_000_000} MB)")
    print(f"  Training step     : {step}")
    print(f"  Base pretrained   : {getattr(policy.config, 'pretrained_path', 'n/a')}")
    print(f"  Action joints     : {list(policy.config.action_feature_names or [])}")


def _load_pi05_policy_with_compat(policy_path: pathlib.Path, device: torch.device) -> Any:
    """Load PI05 strictly via root-level shared compatibility loader."""
    from _pi05_loader import load_pi05_policy_with_compat

    return load_pi05_policy_with_compat(policy_path, device)


def _actions_to_robot_dict(action_row: torch.Tensor, joint_names: list[str]) -> dict[str, float]:
    row = action_row.detach().cpu().numpy().reshape(-1)
    if row.shape[0] != len(joint_names):
        raise ValueError(f"expected {len(joint_names)} action dims, got {row.shape[0]}")
    return {_robot_pos_key(name): float(row[i]) for i, name in enumerate(joint_names)}


def main() -> None:
    cfg = _load_yaml()
    args = _parse_args()
    vr_joint_caps, vr_joint_deadbands = _load_vr_control_shaping(cfg)
    homing_present_tol, homing_soft_stall_tol = _load_homing_tolerances(cfg)
    if args.joint_deadband_deg is not None:
        override_deadband = max(0.0, min(5.0, float(args.joint_deadband_deg)))
        vr_joint_deadbands = {
            joint: (0.0 if joint == "gripper" else override_deadband)
            for joint in vr_joint_deadbands
        }

    if args.dry_run_home:
        print("=" * 60)
        print("  Mode              : dry-run-home (homing only)")
        print(f"  FPS               : {args.fps}")
        print(f"  Home timeout      : {args.home_timeout}s")
        print(
            f"  Home tolerance    : feedback {homing_present_tol:.2f} deg, "
            f"soft stall {homing_soft_stall_tol:.2f} deg"
        )
        print("=" * 60)
        _run_home_only(args, cfg)
        return

    policy_path = pathlib.Path(args.policy_path).resolve() if args.policy_path else None
    if policy_path is not None and not policy_path.is_dir():
        sys.exit(f"policy path not found: {policy_path}")

    print("=" * 60)
    print(f"  Policy checkpoint : {policy_path or '(not required for --dry-run)'}")
    print(f"  Task              : {args.task}")
    ep_lim = [f"<= {args.episode_time}s wall-clock"]
    if args.episode_steps is not None:
        ep_lim.append(f"<= {args.episode_steps} steps")
    print(f"  Episodes          : {args.episodes} x ({', '.join(ep_lim)}, first wins) @ {args.fps} fps")
    if args.episode_time >= 600 and args.episode_steps is None:
        print(
            f"  Warning           : --episode-time {args.episode_time}s is "
            f"{args.episode_time / 60:.0f} min; did you mean --episode-steps {args.episode_time}?"
        )
    print(f"  Action horizon    : {args.action_horizon} (chunk cap)")
    print(f"  Open-loop steps   : {args.open_loop_steps} (scheduled re-infer interval)")
    print(f"  Settle steps      : {args.settle_steps} (hold pose after homing)")
    print(f"  Replan blend      : {args.replan_blend}")
    if args.replan_on_miss_deg > 0:
        print(
            f"  Replan on miss    : >{args.replan_on_miss_deg:.1f} deg for "
            f"{args.replan_miss_steps} tick(s) → early re-infer"
        )
    else:
        print("  Replan on miss    : disabled")
    if args.phase1_task and args.phase1_sec > 0:
        print(f"  Phase-1 task      : {args.phase1_task!r} for {args.phase1_sec}s")
    print(f"  Device            : {args.device}")
    print(f"  Home before run   : {not args.skip_home}")
    print(f"  Home per episode  : {args.home_before_episode and not args.skip_home}")
    print(
        f"  Home tolerance    : feedback {homing_present_tol:.2f} deg, "
        f"soft stall {homing_soft_stall_tol:.2f} deg"
    )
    robot_cfg_yaml = cfg.get("robot") or {}
    max_rel = args.max_relative_target
    if max_rel is None:
        max_rel = robot_cfg_yaml.get("max_relative_target")
    print(
        f"  Clamp to present  : {args.clamp_to_present}"
        + (f" (max {max_rel} deg)" if args.clamp_to_present and max_rel is not None else "")
    )
    print(f"  Policy EMA alpha  : {args.policy_ema_alpha}")
    print(f"  Command EMA alpha : {args.command_ema_alpha}")
    print(f"  VR joint caps     : {vr_joint_caps}")
    print(f"  Joint deadbands   : {vr_joint_deadbands}")
    print(f"  EMA bypass joints : {sorted(_FINAL_SMOOTHING_BYPASS)}")
    print(f"  Camera backend    : {args.camera_backend}")
    print(
        f"  Camera preview    : {args.show_cameras} "
        "(resizable pygame window, Q/Esc closes preview only)"
    )
    print("=" * 60)
    if args.dry_run:
        return

    if policy_path is None:
        sys.exit("--policy-path is required for inference")

    _run_motor_port_preflight(cfg)
    home_pose = _read_home_pose(cfg) if not args.skip_home else {}

    _ensure_xlerobot_import()
    from _camera_preview_window import CameraPreviewWindow
    from _camera_utils import print_inference_camera_map
    from _dashboard_camera_session import attach_dashboard_cameras_to_robot, detach_dashboard_cameras_from_robot
    from _xlerobot_loader import (
        make_config,
        patch_motors_bus_lenient,
        patch_xlerobot_motors_only_connected,
    )
    from lerobot.common.control_utils import predict_action
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.robots.xlerobot import XLerobot
    from lerobot.utils.robot_utils import precise_sleep

    use_dashboard_cameras = args.camera_backend == "dashboard"
    if not use_dashboard_cameras:
        print(
            "Note: --camera-backend lerobot opens 3 V4L captures on the robot; "
            "this often destabilizes right_wrist on a shared USB hub. "
            "Prefer --camera-backend dashboard (default) unless required."
        )
    patch_xlerobot_motors_only_connected()
    if not use_dashboard_cameras:
        from _opencv_camera_patch import (
            patch_opencv_camera_resilient,
            patch_xlerobot_camera_observation,
        )

        patch_opencv_camera_resilient()
        patch_xlerobot_camera_observation()
    if not args.strict_motors:
        patch_motors_bus_lenient()

    if args.max_relative_target is not None:
        cfg.setdefault("robot", {})["max_relative_target"] = args.max_relative_target

    device = torch.device(args.device)
    policy = _load_pi05_policy_with_compat(policy_path, device)
    _verify_policy_checkpoint(policy_path, policy)

    policy_joint_names = list(policy.config.action_feature_names or [])
    normalized_policy_joint_names = [_normalize_joint_name(name) for name in policy_joint_names]
    if len(normalized_policy_joint_names) != 12:
        sys.exit(f"expected 12 action joints in checkpoint config, got {len(normalized_policy_joint_names)}")
    if normalized_policy_joint_names != JOINT_ORDER:
        print(
            "Warning: checkpoint action_feature_names order differs from dataset "
            f"JOINT_ORDER.\n  checkpoint: {policy_joint_names}\n  normalized:{normalized_policy_joint_names}\n  dataset:   {JOINT_ORDER}"
        )

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_path),
    )

    robot_cfg = make_config(robot_id="xlerobot", use_cameras=not use_dashboard_cameras)
    robot = XLerobot(robot_cfg)
    _connect_robot_with_retries(robot)
    cam_session = None
    if use_dashboard_cameras:
        from _camera_preflight import run_camera_preflight

        run_camera_preflight(CONFIG_YAML)
        cam_session = attach_dashboard_cameras_to_robot(robot, warmup_s=3.0)
    print_inference_camera_map(cfg, policy)
    camera_preview: CameraPreviewWindow | None = None
    if args.show_cameras:
        preview_src: Any = cam_session if use_dashboard_cameras else robot.cameras
        camera_preview = CameraPreviewWindow(
            preview_src, preview_fps=min(float(args.preview_fps), float(args.fps))
        )
        camera_preview.start()
    import lerobot.robots.xlerobot.xlerobot as xlerobot_module

    print(f"  XLerobot driver    : {xlerobot_module.__file__}")
    print(
        f"  Cameras            : {args.camera_backend} "
        + (
            "(CameraStream registry)"
            if use_dashboard_cameras
            else "(LeRobot OpenCVCamera, RGB async_read)"
        )
    )

    dt = 1.0 / float(args.fps)
    vr_kp = _vr_kp(cfg)
    print(f"  VR-style cmd shape : kp={vr_kp} per-tick caps (matches dataset action labels)")
    last_sent: dict[str, float] = {}
    last_filtered: dict[str, float] = {}
    last_policy_smoothed: dict[str, float] = {}
    try:
        if home_pose and not args.home_before_episode:
            try:
                go_to_home_pose(
                    robot,
                    home_pose,
                    fps=float(args.fps),
                    timeout_s=float(args.home_timeout),
                    precise_sleep=precise_sleep,
                    caps=vr_joint_caps,
                    present_tolerance_deg=homing_present_tol,
                    soft_stall_tolerance_deg=homing_soft_stall_tol,
                )
            except Exception as exc:
                sys.exit(f"Pre-run homing failed: {exc}")

        for ep in range(args.episodes):
            print(f"\n=== Episode {ep + 1}/{args.episodes} ===")
            episode_deadline = time.perf_counter() + float(args.episode_time)
            progress_log_t = 0.0
            if home_pose and args.home_before_episode:
                pre_home_timeout = float(args.home_timeout)
                if _episode_seconds_left(episode_deadline) < pre_home_timeout:
                    pre_home_timeout = max(1.0, _episode_seconds_left(episode_deadline) - 1.0)
                try:
                    go_to_home_pose(
                        robot,
                        home_pose,
                        fps=float(args.fps),
                        timeout_s=pre_home_timeout,
                        precise_sleep=precise_sleep,
                        caps=vr_joint_caps,
                        present_tolerance_deg=homing_present_tol,
                        soft_stall_tolerance_deg=homing_soft_stall_tol,
                        episode_deadline=episode_deadline,
                    )
                except Exception as exc:
                    sys.exit(f"Pre-episode homing failed: {exc}")
            _reset_policy_episode_state(
                policy,
                preprocessor,
                postprocessor,
                open_loop_steps=int(args.open_loop_steps),
                last_sent=last_sent,
                last_filtered=last_filtered,
                last_policy_smoothed=last_policy_smoothed,
                reason=f"episode {ep + 1} start",
            )

            t_start = time.perf_counter()
            step = 0
            logged_action_debug = False
            settle_steps = max(0, int(args.settle_steps))
            stop_reason: str | None = None
            abort_run = False
            miss_streak = 0
            miss_replans = 0

            try:
                while True:
                    stop_reason = _episode_limit_reason(step, episode_deadline, args)
                    if stop_reason:
                        break

                    loop_start = time.perf_counter()
                    try:
                        raw_obs = robot.get_observation()
                    except Exception as exc:
                        stop_reason = f"observation failed: {exc}"
                        break
                    if _episode_timed_out(episode_deadline):
                        stop_reason = f"wall-clock limit ({args.episode_time}s)"
                        break

                    present_dict = {
                        f"{n}.pos": float(raw_obs[f"{n}.pos"]) for n in JOINT_ORDER
                    }

                    if step < settle_steps:
                        hold_cmd = dict(present_dict)
                        sent_cmd = _send_positions(robot, hold_cmd, present=present_dict)
                        last_sent = dict(sent_cmd)
                        last_filtered = dict(sent_cmd)
                        step += 1
                        remaining = dt - (time.perf_counter() - loop_start)
                        if remaining > 0:
                            precise_sleep(remaining)
                        continue

                    if step == settle_steps:
                        _reset_policy_episode_state(
                            policy,
                            preprocessor,
                            postprocessor,
                            open_loop_steps=int(args.open_loop_steps),
                            last_sent=last_sent,
                            last_filtered=last_filtered,
                            last_policy_smoothed=last_policy_smoothed,
                            reason=f"episode {ep + 1} after settle",
                        )

                    try:
                        obs_frame = _build_observation(raw_obs, JOINT_ORDER)
                        task_prompt = _task_for_step(
                            args, step=step, settle_steps=settle_steps, fps=float(args.fps)
                        )
                        queue_empty_before = len(getattr(policy, "_action_queue", ())) == 0
                        action = predict_action(
                            obs_frame,
                            policy,
                            device,
                            preprocessor,
                            postprocessor,
                            use_amp=bool(getattr(policy.config, "use_amp", False)),
                            task=task_prompt,
                            robot_type=robot.name,
                        )
                    except Exception as exc:
                        stop_reason = f"policy step failed: {exc}"
                        break
                    if _episode_timed_out(episode_deadline):
                        stop_reason = f"wall-clock limit ({args.episode_time}s)"
                        break

                    action_dict = _actions_to_robot_dict(action, policy_joint_names)
                    if args.policy_ema_alpha < 0.999:
                        policy_ref = (
                            last_policy_smoothed if last_policy_smoothed else action_dict
                        )
                        action_dict = _ema_command(
                            action_dict,
                            policy_ref,
                            float(args.policy_ema_alpha),
                            bypass_suffixes=_FINAL_SMOOTHING_BYPASS,
                        )
                        last_policy_smoothed = dict(action_dict)
                    if (
                        queue_empty_before
                        and last_filtered
                        and args.replan_blend < 0.999
                    ):
                        action_dict = _blend_action_dict(
                            action_dict,
                            last_filtered,
                            float(args.replan_blend),
                            bypass_suffixes=_FINAL_SMOOTHING_BYPASS,
                        )
                    shaped = _shape_action_like_recording(
                        action_dict,
                        present_dict,
                        last_sent,
                        kp=vr_kp,
                        caps=vr_joint_caps,
                    )
                    final_cmd = shaped
                    if args.clamp_to_present:
                        max_rel_cfg = getattr(robot.config, "max_relative_target", None)
                        if max_rel_cfg is not None:
                            final_cmd = _clamp_max_relative(
                                shaped, present_dict, float(max_rel_cfg)
                            )
                    final_cmd = _apply_joint_deadband(
                        final_cmd, last_filtered, vr_joint_deadbands
                    )
                    final_cmd = _ema_command(
                        final_cmd,
                        last_filtered,
                        float(args.command_ema_alpha),
                        bypass_suffixes=_FINAL_SMOOTHING_BYPASS,
                    )
                    if args.replan_on_miss_deg > 0 and last_filtered:
                        exec_err = _max_tracking_error_deg(
                            present_dict, last_filtered
                        )
                        if exec_err >= float(args.replan_on_miss_deg):
                            miss_streak += 1
                        else:
                            miss_streak = 0
                        if (
                            miss_streak >= int(args.replan_miss_steps)
                            and _flush_policy_action_queue(policy) > 0
                        ):
                            miss_streak = 0
                            miss_replans += 1
                    if not logged_action_debug:
                        present = _build_state_vector(raw_obs, JOINT_ORDER)
                        debug_final_cmd = final_cmd
                        max_rel_cfg = getattr(robot.config, "max_relative_target", None)
                        if args.clamp_to_present and max_rel_cfg is not None:
                            debug_final_cmd = _clamp_max_relative(
                                final_cmd, present_dict, float(max_rel_cfg)
                            )
                        raw_cmd = np.array([action_dict[f"{n}.pos"] for n in JOINT_ORDER])
                        shaped_cmd = np.array([shaped[f"{n}.pos"] for n in JOINT_ORDER])
                        sent_cmd_arr = np.array([debug_final_cmd[f"{n}.pos"] for n in JOINT_ORDER])
                        raw_delta = raw_cmd - present
                        shaped_delta = shaped_cmd - present
                        final_delta = sent_cmd_arr - present
                        print(
                            f"  First step |policy-present| max={np.abs(raw_delta).max():.2f} deg "
                            f"(raw policy, before VR shaping)"
                        )
                        print(
                            f"  First step |shaped-present| max={np.abs(shaped_delta).max():.2f} deg "
                            f"(after VR shaping only)"
                        )
                        sent_note = (
                            "after present clamp"
                            if args.clamp_to_present
                            else "after EMA/deadband (VR caps only)"
                        )
                        print(
                            f"  First step |sent-present|   max={np.abs(final_delta).max():.2f} deg "
                            f"(final command {sent_note})"
                        )
                        for name, delta in _top_joint_deltas(debug_final_cmd, present_dict):
                            print(f"    sent delta {name}: {delta:+.2f} deg")
                        logged_action_debug = True

                    try:
                        send_present = present_dict if args.clamp_to_present else None
                        sent_cmd = _send_positions(robot, final_cmd, present=send_present)
                    except Exception as exc:
                        stop_reason = f"motor command failed: {exc}"
                        break

                    last_sent = dict(sent_cmd)
                    last_filtered = dict(sent_cmd)

                    step += 1
                    progress_log_t = _maybe_log_episode_progress(
                        episode_deadline=episode_deadline,
                        episode_time_s=float(args.episode_time),
                        step=step,
                        last_log_t=progress_log_t,
                    )
                    remaining = dt - (time.perf_counter() - loop_start)
                    if remaining > 0:
                        precise_sleep(remaining)

            except Exception as exc:
                stop_reason = f"unhandled error: {exc}"
                abort_run = bool(args.stop_on_episode_error)
                if abort_run:
                    raise
            finally:
                elapsed = time.perf_counter() - t_start
                hold = dict(last_filtered or last_sent)
                _reset_policy_episode_state(
                    policy,
                    preprocessor,
                    postprocessor,
                    open_loop_steps=int(args.open_loop_steps),
                    last_sent=last_sent,
                    last_filtered=last_filtered,
                    last_policy_smoothed=last_policy_smoothed,
                    reason=f"episode {ep + 1} end",
                )
                _safe_finish_episode(
                    robot,
                    home_pose,
                    reason=stop_reason or "episode loop ended",
                    fps=float(args.fps),
                    home_timeout_s=float(args.home_timeout),
                    precise_sleep=precise_sleep,
                    caps=vr_joint_caps,
                    homing_present_tolerance_deg=homing_present_tol,
                    homing_soft_stall_tolerance_deg=homing_soft_stall_tol,
                    hold_cmd=hold if hold else None,
                    episode_deadline=episode_deadline,
                    skip_home_after=bool(args.skip_home_after_episode),
                )
                miss_note = (
                    f", {miss_replans} early replan(s) on tracking miss"
                    if miss_replans
                    else ""
                )
                print(
                    f"Episode {ep + 1} done: {step} steps, {elapsed:.1f}s elapsed{miss_note}"
                    + (f" ({stop_reason})" if stop_reason else "")
                )

            if abort_run:
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if camera_preview is not None:
            try:
                camera_preview.stop()
            except Exception as exc:
                print(f"Warning: camera preview stop: {exc}")
        if cam_session is not None:
            try:
                detach_dashboard_cameras_from_robot(robot)
            except Exception as exc:
                print(f"Warning: camera session teardown: {exc}")
        try:
            robot.disconnect()
        except Exception as exc:
            print(f"Warning: robot disconnect: {exc}")


if __name__ == "__main__":
    main()
