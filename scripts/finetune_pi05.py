#!/usr/bin/env python3
"""Finetune a PI0.5 policy on the current LeRobot dataset.

This script is a thin wrapper around LeRobot's package-managed native training entrypoint.

Defaults are loaded from `config/xlerobot.yaml`:
  - dataset.repo_id
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"
PRETRAINED_COMPAT_DIR = REPO_ROOT / ".cache" / "pretrained_compat"
XLEROBOT_JOINTS_PER_ARM = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
XLEROBOT_JOINT_ORDER = [
    f"{side}_arm_{joint}"
    for side in ("left", "right")
    for joint in XLEROBOT_JOINTS_PER_ARM
]
RECORDING_PER_TICK_DEG_CAPS = {
    "shoulder_pan": 5.0,
    "shoulder_lift": 5.0,
    "elbow_flex": 5.0,
    "wrist_flex": 6.0,
    "wrist_roll": 6.0,
    "gripper": 15.0,
}


def _load_yaml() -> dict[str, Any]:
    if not CONFIG_YAML.is_file():
        sys.exit(f"missing {CONFIG_YAML}")
    return yaml.safe_load(CONFIG_YAML.read_text()) or {}


def _load_recording_joint_caps(cfg: dict[str, Any]) -> dict[str, float]:
    """Load the same per-tick action caps used by VR recording."""
    caps = dict(RECORDING_PER_TICK_DEG_CAPS)
    vr_cfg = cfg.get("vr") if isinstance(cfg, dict) else None
    raw_caps = vr_cfg.get("joint_deg_caps") if isinstance(vr_cfg, dict) else None
    if isinstance(raw_caps, dict):
        for joint, value in raw_caps.items():
            if joint in caps:
                try:
                    caps[joint] = max(0.1, min(30.0, float(value)))
                except (TypeError, ValueError) as e:
                    raise SystemExit(f"invalid vr.joint_deg_caps.{joint}: {value!r}") from e
    return caps


def _load_robot_max_relative_target(cfg: dict[str, Any]) -> float | dict[str, float] | None:
    robot_cfg = cfg.get("robot") if isinstance(cfg, dict) else None
    raw_target = robot_cfg.get("max_relative_target") if isinstance(robot_cfg, dict) else None
    if raw_target is None:
        return None
    if isinstance(raw_target, (int, float)):
        return max(0.0, float(raw_target))
    if isinstance(raw_target, dict):
        out: dict[str, float] = {}
        for key, value in raw_target.items():
            suffix = _joint_suffix(str(key))
            if suffix in RECORDING_PER_TICK_DEG_CAPS:
                try:
                    out[suffix] = max(0.0, float(value))
                except (TypeError, ValueError) as e:
                    raise SystemExit(f"invalid robot.max_relative_target.{key}: {value!r}") from e
        return out
    sys.exit(f"invalid robot.max_relative_target: {raw_target!r}")


def _parse_args() -> argparse.Namespace:
    cfg = _load_yaml()
    ds = cfg.get("dataset") or {}
    default_repo = str(ds.get("repo_id") or "your-org/your-dataset")
    default_root = str(ds.get("root") or "")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-repo-id", default=default_repo, help="LeRobot dataset repo_id.")
    p.add_argument(
        "--dataset-root",
        default=default_root,
        help=(
            "Local LeRobot dataset root. Defaults to config/xlerobot.yaml dataset.root. "
            "Leave empty to use LeRobot/HF cache lookup."
        ),
    )
    p.add_argument(
        "--video-backend",
        default="pyav",
        choices=["pyav", "torchcodec", "video_reader"],
        help=(
            "Dataset video decoder backend passed to lerobot-train. Defaults to pyav "
            "because this environment's torchcodec install is incompatible with "
            "the current PyTorch/FFmpeg runtime."
        ),
    )
    p.add_argument("--pretrained-path", default="lerobot/pi05_base",
                   help="Base checkpoint to finetune from (HF id or local path).")
    p.add_argument("--output-dir", default="outputs/pi05_finetune", help="Training output directory.")
    p.add_argument("--job-name", default="pi05_finetune_xlerobot", help="Training job name.")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--log-freq", type=int, default=100)
    p.add_argument("--save-freq", type=int, default=5_000)
    p.add_argument("--eval-freq", type=int, default=5_000)
    p.add_argument("--policy-repo-id", default="",
                   help="Optional HF repo id for pushing policy checkpoints.")
    p.add_argument("--push-to-hub", action="store_true",
                   help="If set, push policy checkpoints to Hugging Face Hub.")
    p.add_argument("--wandb-enable", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument(
        "--oom-safe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cap batch size at 2 for low VRAM (default: on). "
            "Does not control which layers train — use --train-expert-only / --no-train-expert-only."
        ),
    )
    p.add_argument(
        "--train-expert-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Freeze vision encoder and train the PI0 expert / action head only (default: on). "
            "Use --no-train-expert-only for full finetune (vision + expert). "
            "Works with --no-oom-safe and your chosen --batch-size."
        ),
    )
    p.add_argument(
        "--cuda-alloc-conf",
        default="expandable_segments:True",
        help="Value for PYTORCH_CUDA_ALLOC_CONF passed to training subprocess.",
    )
    p.add_argument(
        "--rename-map-json",
        default='{"observation.images.head":"observation.images.base_0_rgb","observation.images.left_wrist":"observation.images.left_wrist_0_rgb","observation.images.right_wrist":"observation.images.right_wrist_0_rgb"}',
        help="JSON mapping from dataset observation keys to policy expected keys.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print command without executing.")
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Validate dataset loading/decoding and print the train command without executing training.",
    )
    p.add_argument(
        "--skip-dataset-check",
        action="store_true",
        help="Skip the preflight dataset load/sample check before launching training.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from an existing checkpoint (uses --output-dir/checkpoints/last by default).",
    )
    p.add_argument(
        "--resume-from",
        default="",
        help=(
            "Optional resume source path. Can be a run dir (with checkpoints/), "
            "a checkpoint dir (.../checkpoints/020000), or a train_config.json path."
        ),
    )
    return p.parse_args()


def _resolve_resume_config_path(args: argparse.Namespace) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    if args.resume_from:
        src = pathlib.Path(args.resume_from).expanduser().resolve()
        if src.is_file():
            candidates.append(src)
        else:
            candidates.append(src / "train_config.json")
            candidates.append(src / "pretrained_model" / "train_config.json")
            candidates.append(src / "checkpoints" / "last" / "pretrained_model" / "train_config.json")
    else:
        out = pathlib.Path(args.output_dir).expanduser().resolve()
        candidates.append(out / "checkpoints" / "last" / "pretrained_model" / "train_config.json")

    for p in candidates:
        if p.is_file():
            return p

    tried = "\n  - ".join(str(p) for p in candidates)
    sys.exit(
        "Could not find resume train_config.json. Tried:\n"
        f"  - {tried}\n"
        "Pass --resume-from with a valid checkpoint/run path."
    )


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _rewrite_processor_registry_names(path: pathlib.Path) -> bool:
    if not path.is_file():
        return False
    data = json.loads(path.read_text())
    changed = False
    for step in data.get("steps") or []:
        if step.get("registry_name") == "relative_actions_processor":
            step["registry_name"] = "delta_actions_processor"
            changed = True
    if changed:
        path.write_text(json.dumps(data, indent=4) + "\n")
    return changed


def _resolve_pretrained_snapshot(pretrained_path: str) -> pathlib.Path:
    local = pathlib.Path(pretrained_path).expanduser()
    if local.exists():
        return local.resolve()

    from huggingface_hub import snapshot_download

    return pathlib.Path(snapshot_download(repo_id=pretrained_path)).resolve()


def _link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _prepare_pretrained_for_lerobot(args: argparse.Namespace) -> str:
    """Return a local pretrained path compatible with this vendored LeRobot.

    Some cached `lerobot/pi05_base` snapshots serialize the old processor
    registry name `relative_actions_processor`. The vendored LeRobot code now
    registers the same implementation as `delta_actions_processor`. Because
    `lerobot-train` loads processor JSON from `policy.pretrained_path`, we make
    a local metadata-compatible view of the pretrained checkpoint instead of
    mutating Hugging Face cache files or the LeRobot submodule.
    """
    source = _resolve_pretrained_snapshot(args.pretrained_path)
    preprocessor = source / "policy_preprocessor.json"
    if not preprocessor.is_file() or "relative_actions_processor" not in preprocessor.read_text():
        return str(source if pathlib.Path(args.pretrained_path).expanduser().exists() else args.pretrained_path)

    digest = hashlib.sha256(f"{args.pretrained_path}|{source}".encode("utf-8")).hexdigest()[:12]
    compat = PRETRAINED_COMPAT_DIR / f"{_safe_name(args.pretrained_path)}-{digest}"
    compat.mkdir(parents=True, exist_ok=True)

    for child in source.iterdir():
        dst = compat / child.name
        if child.name in {"policy_preprocessor.json", "policy_postprocessor.json"}:
            if not dst.exists():
                shutil.copy2(child, dst)
        else:
            _link_or_copy(child, dst)

    changed = _rewrite_processor_registry_names(compat / "policy_preprocessor.json")
    changed = _rewrite_processor_registry_names(compat / "policy_postprocessor.json") or changed
    if changed:
        print(
            "Prepared LeRobot processor compatibility copy: "
            f"{compat} (relative_actions_processor -> delta_actions_processor)"
        )
    return str(compat)


def _build_cmd(args: argparse.Namespace) -> tuple[list[str], pathlib.Path | None, int]:
    if args.resume:
        config_path = _resolve_resume_config_path(args)
        cmd = [
            "uv",
            "run",
            "lerobot-train",
            f"--config_path={config_path}",
            "--resume=true",
        ]
        return cmd, config_path, args.batch_size

    try:
        rename_map = json.loads(args.rename_map_json)
    except json.JSONDecodeError as e:
        sys.exit(f"invalid --rename-map-json: {e}")
    rename_map_json = json.dumps(rename_map, separators=(",", ":"))

    effective_batch_size = args.batch_size
    if args.oom_safe:
        effective_batch_size = min(args.batch_size, 2)

    expert_only = bool(args.train_expert_only)
    freeze_vision = "true" if expert_only else "false"
    train_expert_only = "true" if expert_only else "false"

    cmd = [
        "uv",
        "run",
        "lerobot-train",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.video_backend={args.video_backend}",
        f"--policy.path={args.pretrained_path}",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--steps={args.steps}",
        f"--batch_size={effective_batch_size}",
        f"--num_workers={args.num_workers}",
        f"--policy.device={args.device}",
        f"--policy.dtype={args.dtype}",
        "--policy.gradient_checkpointing=true",
        f"--policy.freeze_vision_encoder={freeze_vision}",
        f"--policy.train_expert_only={train_expert_only}",
        "--policy.push_to_hub=false",
        f"--rename_map={rename_map_json}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={args.save_freq}",
        f"--eval_freq={args.eval_freq}",
        f"--wandb.enable={'true' if args.wandb_enable else 'false'}",
    ]
    dataset_root = str(args.dataset_root or "").strip()
    if dataset_root:
        dataset_root = str(pathlib.Path(dataset_root).expanduser())
        cmd.insert(5, f"--dataset.root={dataset_root}")
    if args.push_to_hub:
        cmd.append("--policy.push_to_hub=true")
        if args.policy_repo_id:
            cmd.append(f"--policy.repo_id={args.policy_repo_id}")
    return cmd, None, effective_batch_size


def _preflight_dataset(args: argparse.Namespace, rename_map: dict[str, str]) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = str(args.dataset_root or "").strip() or None
    if root is not None:
        root = str(pathlib.Path(root).expanduser())
    ds = LeRobotDataset(
        args.dataset_repo_id,
        root=root,
        video_backend=args.video_backend,
    )
    if len(ds) <= 0:
        sys.exit("dataset preflight failed: dataset has no frames")

    required = {
        "action",
        "observation.state",
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }
    missing = sorted(required - set(ds.meta.features))
    if missing:
        sys.exit("dataset preflight failed: missing features: " + ", ".join(missing))

    missing_rename = sorted(k for k in rename_map if k not in ds.meta.features)
    if missing_rename:
        sys.exit("dataset preflight failed: rename_map source keys missing: " + ", ".join(missing_rename))

    _preflight_xlerobot_joint_metadata(ds.meta.features)
    cfg = _load_yaml()
    _preflight_action_continuity(
        ds,
        _load_recording_joint_caps(cfg),
        _load_robot_max_relative_target(cfg),
    )

    sample_indices = sorted(set([0, len(ds) // 2, len(ds) - 1]))
    for idx in sample_indices:
        sample = ds[idx]
        action = sample["action"]
        state = sample["observation.state"]
        if tuple(action.shape) != (12,) or tuple(state.shape) != (12,):
            sys.exit(
                "dataset preflight failed: expected action/state shape (12,), "
                f"got action={tuple(action.shape)} state={tuple(state.shape)} at index {idx}"
            )
        for key in (
            "observation.images.head",
            "observation.images.left_wrist",
            "observation.images.right_wrist",
        ):
            img = sample[key]
            if tuple(img.shape) != (3, 480, 640):
                sys.exit(
                    "dataset preflight failed: expected image shape (3,480,640), "
                    f"got {key}={tuple(img.shape)} at index {idx}"
                )
    print(
        "Dataset preflight OK: "
        f"repo_id={args.dataset_repo_id} frames={len(ds)} episodes={ds.num_episodes} "
        f"video_backend={args.video_backend}"
    )


def _normalize_feature_name(name: str) -> str:
    return str(name).removesuffix(".pos")


def _joint_suffix(name: str) -> str:
    base = _normalize_feature_name(name)
    if "_arm_" in base:
        return base.split("_arm_", 1)[1]
    return base


def _preflight_xlerobot_joint_metadata(features: dict[str, Any]) -> None:
    action_names = list((features.get("action") or {}).get("names") or [])
    state_names = list((features.get("observation.state") or {}).get("names") or [])
    if len(action_names) != len(XLEROBOT_JOINT_ORDER) or len(state_names) != len(XLEROBOT_JOINT_ORDER):
        sys.exit(
            "dataset preflight failed: expected 12 action/state joint names, "
            f"got action={len(action_names)} state={len(state_names)}"
        )
    normalized_action = [_normalize_feature_name(name) for name in action_names]
    normalized_state = [_normalize_feature_name(name) for name in state_names]
    if normalized_action != XLEROBOT_JOINT_ORDER or normalized_state != XLEROBOT_JOINT_ORDER:
        sys.exit(
            "dataset preflight failed: action/state joint order is not XLerobot left-six/right-six order.\n"
            f"  action: {action_names}\n"
            f"  state:  {state_names}\n"
            f"  expected: {[f'{name}.pos' for name in XLEROBOT_JOINT_ORDER]}"
        )
    expected_pos_names = [f"{name}.pos" for name in XLEROBOT_JOINT_ORDER]
    if action_names != expected_pos_names or state_names != expected_pos_names:
        print(
            "WARNING: dataset joint metadata uses legacy names without '.pos'. "
            "New recordings will use canonical LeRobot XLerobot '.pos' names."
        )


def _preflight_action_continuity(
    ds: Any,
    recording_joint_caps: dict[str, float],
    max_relative_target: float | dict[str, float] | None = None,
) -> None:
    actions = np.asarray(ds.hf_dataset["action"], dtype=np.float32)
    states = np.asarray(ds.hf_dataset["observation.state"], dtype=np.float32)
    episode_index = np.asarray(ds.hf_dataset["episode_index"], dtype=np.int64)
    if actions.ndim != 2 or actions.shape[1] != len(XLEROBOT_JOINT_ORDER):
        sys.exit(f"dataset preflight failed: expected action matrix Nx12, got {actions.shape}")
    if states.shape != actions.shape:
        sys.exit(f"dataset preflight failed: expected observation.state matrix {actions.shape}, got {states.shape}")
    if len(actions) < 2:
        return

    names = list((ds.meta.features.get("action") or {}).get("names") or [])
    normalized_names = [_normalize_feature_name(name) for name in names]
    same_episode = episode_index[1:] == episode_index[:-1]
    deltas = np.abs(np.diff(actions, axis=0))
    caps = np.asarray(
        [recording_joint_caps.get(_joint_suffix(name), 1.0) for name in normalized_names],
        dtype=np.float32,
    )
    max_rel = _max_relative_target_array(normalized_names, max_relative_target)
    # Small margin avoids false positives from float encoding and older cap tweaks,
    # while still catching action/state fallback jumps.
    raw_over = (deltas > (caps[None, :] + 0.75)) & same_episode[:, None]
    if max_rel is not None:
        action_state_delta = np.abs(actions - states)
        safety_clip_active = (
            (action_state_delta[:-1] >= (max_rel[None, :] - 0.75))
            | (action_state_delta[1:] >= (max_rel[None, :] - 0.75))
        )
        over = raw_over & ~safety_clip_active
    else:
        over = raw_over
    if not over.any():
        return

    bad: list[tuple[float, int, int, str, float, float]] = []
    for frame_idx, joint_idx in np.argwhere(over):
        delta = float(deltas[frame_idx, joint_idx])
        bad.append((
            delta,
            int(frame_idx),
            int(frame_idx + 1),
            names[joint_idx] if joint_idx < len(names) else str(joint_idx),
            float(actions[frame_idx, joint_idx]),
            float(actions[frame_idx + 1, joint_idx]),
        ))
    bad.sort(reverse=True)
    details = "\n".join(
        f"  frame {a}->{b} {name}: {before:.3f}->{after:.3f} (delta {delta:.3f} deg)"
        for delta, a, b, name, before, after in bad[:12]
    )
    sys.exit(
        "dataset preflight failed: action labels contain per-frame jumps beyond the "
        "VR recorder's rate caps. This usually means active-arm hold ticks were recorded "
        "from observation.state instead of the held command. Re-record the affected episode "
        "with the fixed recorder before finetuning.\n"
        + details
    )


def _max_relative_target_array(
    normalized_names: list[str],
    max_relative_target: float | dict[str, float] | None,
) -> np.ndarray | None:
    if max_relative_target is None:
        return None
    if isinstance(max_relative_target, (int, float)):
        return np.full(len(normalized_names), float(max_relative_target), dtype=np.float32)
    values: list[float] = []
    for name in normalized_names:
        suffix = _joint_suffix(name)
        if suffix not in max_relative_target:
            return None
        values.append(float(max_relative_target[suffix]))
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    args = _parse_args()
    if not args.resume:
        args.pretrained_path = _prepare_pretrained_for_lerobot(args)
    cmd, resume_cfg, effective_batch_size = _build_cmd(args)
    print("Running:\n  " + " \\\n  ".join(shlex.quote(x) for x in cmd))
    if args.resume and resume_cfg is not None:
        print(f"Resume mode enabled from: {resume_cfg}")
    else:
        if args.oom_safe and effective_batch_size < args.batch_size:
            print(
                f"OOM-safe: batch_size capped at {effective_batch_size} "
                f"(requested {args.batch_size})."
            )
        elif not args.oom_safe:
            print(f"OOM-safe off: using batch_size={effective_batch_size}.")
        if args.train_expert_only:
            print("Training scope: vision frozen, expert/action head only.")
        else:
            print("Training scope: full finetune (vision encoder + expert).")
    if args.dry_run:
        return
    if not args.resume and not args.skip_dataset_check:
        try:
            rename_map = json.loads(args.rename_map_json)
        except json.JSONDecodeError as e:
            sys.exit(f"invalid --rename-map-json: {e}")
        _preflight_dataset(args, rename_map)
    if args.check_only:
        return
    env = dict(os.environ)
    if args.cuda_alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    main()
