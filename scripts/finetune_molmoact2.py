#!/usr/bin/env python3
"""Fine-tune MolmoAct2 on the current XLerobot LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
from typing import Any

import yaml

from _molmoact2_vendor import (
    REPO_ROOT,
    assert_allenai_lerobot_imported,
    molmoact2_subprocess_env,
    prepend_allenai_lerobot_src,
)
from finetune_pi05 import (
    CONFIG_YAML,
    XLEROBOT_JOINT_ORDER,
    _load_recording_joint_caps,
    _load_robot_max_relative_target,
    _preflight_action_continuity,
    _preflight_xlerobot_joint_metadata,
)

DEFAULT_IMAGE_KEYS = [
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
]
DEFAULT_NORMALIZATION_MAPPING = {
    "ACTION": "MEAN_STD",
    "STATE": "MEAN_STD",
    "VISUAL": "IDENTITY",
}


def _load_yaml() -> dict[str, Any]:
    if not CONFIG_YAML.is_file():
        sys.exit(f"missing {CONFIG_YAML}")
    return yaml.safe_load(CONFIG_YAML.read_text()) or {}


def _molmo_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("molmoact2")
    return block if isinstance(block, dict) else {}


def _json_list(value: str, *, flag: str) -> list[str]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        sys.exit(f"invalid {flag}: {exc}")
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        sys.exit(f"invalid {flag}: expected JSON list of strings")
    return payload


def _json_dict(value: str, *, flag: str) -> dict[str, str]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        sys.exit(f"invalid {flag}: {exc}")
    if not isinstance(payload, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in payload.items()):
        sys.exit(f"invalid {flag}: expected JSON object with string keys/values")
    return payload


def _parse_args() -> argparse.Namespace:
    cfg = _load_yaml()
    ds = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    molmo = _molmo_cfg(cfg)
    default_repo = str(ds.get("repo_id") or "your-org/your-dataset")
    default_root = str(ds.get("root") or "")
    default_image_keys = molmo.get("image_keys") if isinstance(molmo.get("image_keys"), list) else DEFAULT_IMAGE_KEYS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-repo-id", default=default_repo, help="LeRobot dataset repo_id.")
    p.add_argument("--dataset-root", default=default_root, help="Local LeRobot dataset root.")
    p.add_argument("--video-backend", default=str(molmo.get("video_backend") or "pyav"), choices=["pyav", "torchcodec", "video_reader"])
    p.add_argument("--checkpoint-path", default=str(molmo.get("checkpoint_path") or "allenai/MolmoAct2"), help="Original MolmoAct2 HF checkpoint.")
    p.add_argument("--policy-path", default="", help="LeRobot MolmoAct2 checkpoint (.../pretrained_model) to continue from.")
    p.add_argument("--output-dir", default=str(molmo.get("output_dir") or "outputs/molmoact2_finetune"))
    p.add_argument("--job-name", default=str(molmo.get("job_name") or "molmoact2_finetune_xlerobot"))
    p.add_argument("--steps", type=int, default=int(molmo.get("steps", 20_000)))
    p.add_argument("--batch-size", type=int, default=int(molmo.get("batch_size", 2)))
    p.add_argument("--num-workers", type=int, default=int(molmo.get("num_workers", 4)))
    p.add_argument("--device", default=str(molmo.get("device") or "cuda"), choices=["cuda", "cpu", "mps"])
    p.add_argument("--model-dtype", default=str(molmo.get("model_dtype") or "bfloat16"), choices=["float32", "float16", "bfloat16"])
    p.add_argument("--action-mode", default=str(molmo.get("action_mode") or "continuous"), choices=["continuous", "discrete", "both"])
    p.add_argument("--train-mode-vlm", default=str(molmo.get("train_mode_vlm") or "freeze"), choices=["freeze", "lora", "fft"])
    p.add_argument("--chunk-size", type=int, default=int(molmo.get("chunk_size", 30)))
    p.add_argument("--n-action-steps", type=int, default=int(molmo.get("n_action_steps", molmo.get("chunk_size", 30))))
    p.add_argument("--num-flow-timesteps", type=int, default=int(molmo.get("num_flow_timesteps", 8)))
    p.add_argument("--setup-type", default=str(molmo.get("setup_type") or "dual-arm XLerobot tabletop manipulation"))
    p.add_argument("--control-mode", default=str(molmo.get("control_mode") or "absolute joint position in degrees"))
    p.add_argument(
        "--normalize-gripper",
        action=argparse.BooleanOptionalAction,
        default=bool(molmo.get("normalize_gripper", True)),
        help="Normalize gripper channels with the rest of action/state. Required for degree-valued XLerobot grippers.",
    )
    p.add_argument("--image-keys-json", default=json.dumps(default_image_keys), help="JSON image observation keys for MolmoAct2.")
    p.add_argument(
        "--normalization-mapping-json",
        default=json.dumps(molmo.get("normalization_mapping") if isinstance(molmo.get("normalization_mapping"), dict) else DEFAULT_NORMALIZATION_MAPPING),
        help="JSON LeRobot normalization mapping. Defaults to MEAN_STD state/action for XLerobot.",
    )
    p.add_argument("--log-freq", type=int, default=int(molmo.get("log_freq", 100)))
    p.add_argument("--save-freq", type=int, default=int(molmo.get("save_freq", 5_000)))
    p.add_argument("--eval-freq", type=int, default=int(molmo.get("eval_freq", -1)))
    p.add_argument("--wandb-enable", action="store_true")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--policy-repo-id", default="")
    p.add_argument("--cuda-alloc-conf", default=str(molmo.get("cuda_alloc_conf") or "expandable_segments:True"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--skip-dataset-check", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume from output-dir/checkpoints/last train_config.json.")
    p.add_argument("--resume-from", default="", help="Run/checkpoint/train_config path for resume mode.")
    args = p.parse_args()
    if args.train_mode_vlm == "freeze" and args.action_mode != "continuous":
        sys.exit("--train-mode-vlm=freeze requires --action-mode=continuous for MolmoAct2")
    if args.chunk_size < 1 or args.n_action_steps < 1:
        sys.exit("--chunk-size and --n-action-steps must be positive")
    return args


def _resolve_resume_config_path(args: argparse.Namespace) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    if args.resume_from:
        src = pathlib.Path(args.resume_from).expanduser().resolve()
        if src.is_file():
            candidates.append(src)
        else:
            candidates.extend([
                src / "train_config.json",
                src / "pretrained_model" / "train_config.json",
                src / "checkpoints" / "last" / "pretrained_model" / "train_config.json",
            ])
    else:
        out = pathlib.Path(args.output_dir).expanduser().resolve()
        candidates.append(out / "checkpoints" / "last" / "pretrained_model" / "train_config.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = "\n  - ".join(str(p) for p in candidates)
    sys.exit(f"Could not find resume train_config.json. Tried:\n  - {tried}")


def _build_cmd(args: argparse.Namespace) -> tuple[list[str], list[str], dict[str, str]]:
    if args.resume:
        cfg = _resolve_resume_config_path(args)
        return (
            ["uv", "run", "python", "-m", "lerobot.scripts.lerobot_train", f"--config_path={cfg}", "--resume=true"],
            [],
            {},
        )

    image_keys = _json_list(args.image_keys_json, flag="--image-keys-json")
    normalization_mapping = _json_dict(args.normalization_mapping_json, flag="--normalization-mapping-json")
    policy_source: list[str]
    if str(args.policy_path).strip():
        policy_source = [f"--policy.path={pathlib.Path(args.policy_path).expanduser()}"]
    else:
        policy_source = [
            "--policy.type=molmoact2",
            f"--policy.checkpoint_path={args.checkpoint_path}",
        ]

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.video_backend={args.video_backend}",
        *policy_source,
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--policy.device={args.device}",
        f"--policy.model_dtype={args.model_dtype}",
        f"--policy.action_mode={args.action_mode}",
        f"--policy.inference_action_mode=continuous",
        f"--policy.train_mode_vlm={args.train_mode_vlm}",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.num_flow_timesteps={args.num_flow_timesteps}",
        "--policy.gradient_checkpointing=true",
        "--policy.freeze_embedding=true",
        f"--policy.normalize_gripper={'true' if args.normalize_gripper else 'false'}",
        "--policy.enable_knowledge_insulation=false",
        f"--policy.setup_type={args.setup_type}",
        f"--policy.control_mode={args.control_mode}",
        f"--policy.image_keys={json.dumps(image_keys, separators=(',', ':'))}",
        f"--policy.normalization_mapping={json.dumps(normalization_mapping, separators=(',', ':'))}",
        f"--policy.push_to_hub={'true' if args.push_to_hub else 'false'}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={args.save_freq}",
        f"--eval_freq={args.eval_freq}",
        f"--wandb.enable={'true' if args.wandb_enable else 'false'}",
    ]
    dataset_root = str(args.dataset_root or "").strip()
    if dataset_root:
        cmd.insert(7, f"--dataset.root={pathlib.Path(dataset_root).expanduser()}")
    if args.push_to_hub:
        if args.policy_repo_id:
            cmd.append(f"--policy.repo_id={args.policy_repo_id}")
    return cmd, image_keys, normalization_mapping


def _preflight_dataset(args: argparse.Namespace, image_keys: list[str]) -> None:
    prepend_allenai_lerobot_src()
    assert_allenai_lerobot_imported()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = str(args.dataset_root or "").strip() or None
    if root is not None:
        root = str(pathlib.Path(root).expanduser())
    ds = LeRobotDataset(args.dataset_repo_id, root=root, video_backend=args.video_backend)
    if len(ds) <= 0:
        sys.exit("dataset preflight failed: dataset has no frames")

    required = {"action", "observation.state", *image_keys}
    missing = sorted(required - set(ds.meta.features))
    if missing:
        sys.exit("dataset preflight failed: missing features: " + ", ".join(missing))

    _preflight_xlerobot_joint_metadata(ds.meta.features)
    cfg = _load_yaml()
    _preflight_action_continuity(
        ds,
        _load_recording_joint_caps(cfg),
        _load_robot_max_relative_target(cfg),
    )

    for idx in sorted(set([0, len(ds) // 2, len(ds) - 1])):
        sample = ds[idx]
        action = sample["action"]
        state = sample["observation.state"]
        if tuple(action.shape) != (len(XLEROBOT_JOINT_ORDER),) or tuple(state.shape) != (len(XLEROBOT_JOINT_ORDER),):
            sys.exit(
                "dataset preflight failed: expected action/state shape (12,), "
                f"got action={tuple(action.shape)} state={tuple(state.shape)} at index {idx}"
            )
        for key in image_keys:
            img = sample[key]
            if len(tuple(img.shape)) != 3:
                sys.exit(f"dataset preflight failed: expected rank-3 image for {key}, got {tuple(img.shape)}")
    print(
        "MolmoAct2 dataset preflight OK: "
        f"repo_id={args.dataset_repo_id} frames={len(ds)} episodes={ds.num_episodes} "
        f"video_backend={args.video_backend}"
    )


def main() -> None:
    args = _parse_args()
    cmd, image_keys, _normalization_mapping = _build_cmd(args)
    print("Running:\n  " + " \\\n  ".join(shlex.quote(x) for x in cmd))
    print(f"MolmoAct2 training profile: action_mode={args.action_mode}, train_mode_vlm={args.train_mode_vlm}")
    print(f"MolmoAct2 image keys: {image_keys if image_keys else '(from checkpoint config)'}")
    if args.dry_run:
        return
    if not args.resume and not args.skip_dataset_check:
        _preflight_dataset(args, image_keys)
    if args.check_only:
        return
    if args.device == "cuda":
        try:
            import torch
        except ImportError as exc:
            sys.exit(f"MolmoAct2 training requires torch for CUDA validation: {exc}")
        if not torch.cuda.is_available():
            sys.exit("MolmoAct2 training requested --device=cuda, but torch.cuda.is_available() is false")
    env = molmoact2_subprocess_env(extra_pythonpath=[str(REPO_ROOT)])
    if args.cuda_alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    main()
