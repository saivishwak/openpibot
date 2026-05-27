#!/usr/bin/env python3
"""Run on-robot inference with a finetuned LeRobot PI0.5 checkpoint.

Loads the checkpoint locally (no OpenPI server) and drives the XLerobot
bimanual SO-101 stack — the same robot/dataset layout used during VR recording.

Prerequisites:
    bash scripts/setup_xlerobot.sh   # copies xlerobot into lerobot submodule

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
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"

# Per-tick joint caps (degrees) — same spirit as webapp/backend/vr_teleop.py homing.
_PER_TICK_DEG_CAPS: dict[str, float] = {
    "shoulder_pan": 5.0,
    "shoulder_lift": 3.0,
    "elbow_flex": 3.0,
    "wrist_flex": 2.0,
    "wrist_roll": 5.0,
    "gripper": 2.0,
}
_HOMING_TOL_DEG = 0.5
_HOMING_KP = 0.75


def _load_yaml() -> dict[str, Any]:
    if not CONFIG_YAML.is_file():
        sys.exit(f"missing {CONFIG_YAML}")
    return yaml.safe_load(CONFIG_YAML.read_text()) or {}


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
    p.add_argument("--episode-time", type=int, default=120, help="Max seconds per episode.")
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
        help="Policy action-chunk length (re-infer every N control ticks).",
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
        "--home-timeout",
        type=float,
        default=60.0,
        help="Max seconds to spend homing before continuing anyway.",
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
            "Run: bash scripts/setup_xlerobot.sh\n"
            f"Original error: {e}"
        )


def _build_state_vector(obs: dict[str, Any], joint_names: list[str]) -> np.ndarray:
    return np.array([float(obs[f"{name}.pos"]) for name in joint_names], dtype=np.float32)


def _build_observation(
    obs: dict[str, Any],
    joint_names: list[str],
) -> dict[str, np.ndarray]:
    camera_obs = {k: v for k, v in obs.items() if k in ("head", "left_wrist", "right_wrist")}
    if set(camera_obs) != {"head", "left_wrist", "right_wrist"}:
        missing = {"head", "left_wrist", "right_wrist"} - set(camera_obs)
        sys.exit(f"missing camera observations: {sorted(missing)}")
    return {
        "observation.images.head": np.asarray(camera_obs["head"]),
        "observation.images.left_wrist": np.asarray(camera_obs["left_wrist"]),
        "observation.images.right_wrist": np.asarray(camera_obs["right_wrist"]),
        "observation.state": _build_state_vector(obs, joint_names),
    }


def _read_home_pose(cfg: dict[str, Any]) -> dict[str, float]:
    hp = (cfg.get("robot") or {}).get("home_pose") or {}
    if not hp:
        sys.exit(
            "robot.home_pose is empty in config/xlerobot.yaml.\n"
            "Capture a home pose first (webapp VR Teleop → Capture home, or "
            "scripts/save_home_pose.py)."
        )
    return {str(k): float(v) for k, v in hp.items()}


def _cap_for_joint_key(key: str) -> float:
    for suffix, cap in _PER_TICK_DEG_CAPS.items():
        if f"_{suffix}.pos" in key:
            return cap
    return 2.0


def go_to_home_pose(
    robot: Any,
    home_pose: dict[str, float],
    *,
    fps: float,
    timeout_s: float,
    precise_sleep: Any,
) -> None:
    """Rate-limited move to `robot.home_pose` joint targets (degrees)."""
    targets = {f"{name}.pos": deg for name, deg in home_pose.items()}
    keys = list(targets.keys())
    last_sent: dict[str, float] = {}
    dt = 1.0 / fps
    deadline = time.perf_counter() + timeout_s

    print(f"Homing to saved pose ({len(keys)} joints, timeout={timeout_s:.0f}s)...")
    while time.perf_counter() < deadline:
        loop_start = time.perf_counter()
        obs = robot.get_observation(include_cameras=False)
        present = {k: float(obs[k]) for k in keys if k in obs}

        clamped: dict[str, float] = {}
        converged = True
        for key, target in targets.items():
            cap = _cap_for_joint_key(key)
            prev = last_sent.get(key, present.get(key, target))
            step = max(-cap, min(cap, target - prev))
            clamped[key] = prev + step
            if abs(clamped[key] - target) > _HOMING_TOL_DEG:
                converged = False

        command = {
            key: present.get(key, tgt) + _HOMING_KP * (tgt - present.get(key, tgt))
            for key, tgt in clamped.items()
        }
        robot.send_action(command)
        last_sent = dict(command)

        if converged:
            print("Home pose reached.")
            return

        remaining = dt - (time.perf_counter() - loop_start)
        if remaining > 0:
            precise_sleep(remaining)

    print("Warning: homing timed out; continuing anyway.")


def _run_home_only(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """Connect, homing only, disconnect — no policy."""
    home_pose = _read_home_pose(cfg)
    _ensure_xlerobot_import()
    from _xlerobot_loader import make_config, patch_motors_bus_lenient
    from lerobot.robots.xlerobot import XLerobot
    from lerobot.utils.robot_utils import precise_sleep

    if not args.strict_motors:
        patch_motors_bus_lenient()

    robot = XLerobot(make_config(robot_id="xlerobot"))
    robot.connect(calibrate=False)
    try:
        go_to_home_pose(
            robot,
            home_pose,
            fps=float(args.fps),
            timeout_s=float(args.home_timeout),
            precise_sleep=precise_sleep,
        )
        print("Dry-run home complete.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
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


def _actions_to_robot_dict(action_row: torch.Tensor, joint_names: list[str]) -> dict[str, float]:
    row = action_row.detach().cpu().numpy().reshape(-1)
    if row.shape[0] != len(joint_names):
        raise ValueError(f"expected {len(joint_names)} action dims, got {row.shape[0]}")
    return {f"{name}.pos": float(row[i]) for i, name in enumerate(joint_names)}


def main() -> None:
    cfg = _load_yaml()
    args = _parse_args()

    if args.dry_run_home:
        print("=" * 60)
        print("  Mode              : dry-run-home (homing only)")
        print(f"  FPS               : {args.fps}")
        print(f"  Home timeout      : {args.home_timeout}s")
        print("=" * 60)
        _run_home_only(args, cfg)
        return

    policy_path = pathlib.Path(args.policy_path).resolve()
    if not policy_path.is_dir():
        sys.exit(f"policy path not found: {policy_path}")

    print("=" * 60)
    print(f"  Policy checkpoint : {policy_path}")
    print(f"  Task              : {args.task}")
    print(f"  Episodes          : {args.episodes} x <= {args.episode_time}s @ {args.fps} fps")
    print(f"  Action horizon    : {args.action_horizon}")
    print(f"  Device            : {args.device}")
    print(f"  Home before run   : {not args.skip_home}")
    print(f"  Home per episode  : {args.home_before_episode and not args.skip_home}")
    print("=" * 60)
    if args.dry_run:
        return

    home_pose = _read_home_pose(cfg) if not args.skip_home else {}

    _ensure_xlerobot_import()
    from _xlerobot_loader import make_config, patch_motors_bus_lenient
    from lerobot.common.control_utils import predict_action
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.robots.xlerobot import XLerobot
    from lerobot.utils.robot_utils import precise_sleep

    if not args.strict_motors:
        patch_motors_bus_lenient()

    device = torch.device(args.device)
    policy = PI05Policy.from_pretrained(str(policy_path))
    policy.eval()
    policy.to(device)
    _verify_policy_checkpoint(policy_path, policy)

    joint_names = list(policy.config.action_feature_names or [])
    if len(joint_names) != 12:
        sys.exit(f"expected 12 action joints in checkpoint config, got {len(joint_names)}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_path),
    )

    robot_cfg = make_config(robot_id="xlerobot")
    robot = XLerobot(robot_cfg)
    robot.connect(calibrate=False)

    dt = 1.0 / float(args.fps)
    try:
        if home_pose and not args.home_before_episode:
            go_to_home_pose(
                robot,
                home_pose,
                fps=float(args.fps),
                timeout_s=float(args.home_timeout),
                precise_sleep=precise_sleep,
            )

        for ep in range(args.episodes):
            print(f"\n=== Episode {ep + 1}/{args.episodes} ===")
            if home_pose and args.home_before_episode:
                go_to_home_pose(
                    robot,
                    home_pose,
                    fps=float(args.fps),
                    timeout_s=float(args.home_timeout),
                    precise_sleep=precise_sleep,
                )
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            t_end = time.perf_counter() + args.episode_time
            step = 0
            logged_action_debug = False

            while time.perf_counter() < t_end:
                loop_start = time.perf_counter()

                raw_obs = robot.get_observation()
                obs_frame = _build_observation(raw_obs, joint_names)
                # LeRobot pattern: select_action queues chunks; postprocessor runs per step.
                action = predict_action(
                    obs_frame,
                    policy,
                    device,
                    preprocessor,
                    postprocessor,
                    use_amp=bool(getattr(policy.config, "use_amp", False)),
                    task=args.task,
                    robot_type=robot.name,
                )
                if not logged_action_debug:
                    present = _build_state_vector(raw_obs, joint_names)
                    cmd = action.detach().cpu().numpy().reshape(-1)
                    delta = cmd - present
                    print(
                        f"  First action |cmd-present| max={np.abs(delta).max():.2f} deg "
                        f"mean={np.abs(delta).mean():.2f} deg"
                    )
                    logged_action_debug = True

                action_dict = _actions_to_robot_dict(action, joint_names)
                robot.send_action(action_dict)

                step += 1
                remaining = dt - (time.perf_counter() - loop_start)
                if remaining > 0:
                    precise_sleep(remaining)

            print(f"Episode {ep + 1} finished ({step} control steps).")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
