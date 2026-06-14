#!/usr/bin/env python3
"""Run on-robot inference with a finetuned LeRobot MolmoAct2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

import torch

from _molmoact2_vendor import (
    assert_allenai_lerobot_imported,
    assert_molmoact2_and_xlerobot_sources,
    extend_xlerobot_hardware_paths,
    prepend_allenai_lerobot_src,
)

prepend_allenai_lerobot_src()
assert_allenai_lerobot_imported()
extend_xlerobot_hardware_paths()

from openpibot import pi05_inference_runtime as runtime


MODEL_NAME = "MolmoAct2"
_ORIGINAL_PARSE_ARGS = runtime._parse_args
BASELINE_CACHE_DIR = runtime.REPO_ROOT / ".cache" / "molmoact2_baseline_sources"
BASELINE_DESCRIPTOR = "molmoact2_baseline.json"
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


def _flatten_feature_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for key in sorted(value):
            out.extend(_flatten_feature_names(value[key]))
        return out
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_feature_names(item))
        return out
    return [str(value)]


def _strict_action_names(policy: Any) -> list[str]:
    names = list(getattr(policy.config, "action_feature_names", None) or [])
    if not names:
        dataset_feature_names = getattr(policy.config, "dataset_feature_names", {}) or {}
        names = _flatten_feature_names(dataset_feature_names.get("action"))
    if not names:
        raise RuntimeError(
            "MolmoAct2 checkpoint config is missing dataset action feature names. "
            "Run training with scripts/finetune_molmoact2.py so LeRobot saves dataset_feature_names."
        )
    normalized = [runtime._normalize_joint_name(name) for name in names]
    if normalized != runtime.JOINT_ORDER:
        raise RuntimeError(
            "MolmoAct2 checkpoint action features do not match XLerobot joint order.\n"
            f"  checkpoint: {names}\n"
            f"  normalized:{normalized}\n"
            f"  expected:  {[f'{name}.pos' for name in runtime.JOINT_ORDER]}"
        )
    return names


def _ensure_runtime_action_feature_names(policy: Any) -> list[str]:
    names = _strict_action_names(policy)
    if not getattr(policy.config, "action_feature_names", None):
        policy.config.action_feature_names = names
    return names


def _molmo_cfg() -> dict[str, Any]:
    cfg = runtime._load_yaml()
    block = cfg.get("molmoact2")
    return block if isinstance(block, dict) else {}


def _dataset_cfg() -> dict[str, Any]:
    cfg = runtime._load_yaml()
    block = cfg.get("dataset")
    return block if isinstance(block, dict) else {}


def _json_dict(value: str, *, flag: str) -> dict[str, str]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid {flag}: {exc}") from exc
    if not isinstance(payload, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in payload.items()):
        raise SystemExit(f"invalid {flag}: expected JSON object with string keys/values")
    return payload


def _json_list(value: str, *, flag: str) -> list[str]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid {flag}: {exc}") from exc
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise SystemExit(f"invalid {flag}: expected JSON list of strings")
    return payload


def _write_baseline_descriptor(args: argparse.Namespace) -> pathlib.Path:
    payload = {
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_revision": args.checkpoint_revision,
        "checkpoint_force_download": bool(args.checkpoint_force_download),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": args.dataset_root,
        "action_mode": args.action_mode,
        "train_mode_vlm": args.train_mode_vlm,
        "model_dtype": args.model_dtype,
        "chunk_size": int(args.action_horizon),
        "n_action_steps": int(args.open_loop_steps),
        "setup_type": args.setup_type,
        "control_mode": args.control_mode,
        "normalize_gripper": bool(args.normalize_gripper),
        "image_keys": _json_list(args.image_keys_json, flag="--image-keys-json"),
        "normalization_mapping": _json_dict(
            args.normalization_mapping_json,
            flag="--normalization-mapping-json",
        ),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    descriptor_dir = BASELINE_CACHE_DIR / digest
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    (descriptor_dir / BASELINE_DESCRIPTOR).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return descriptor_dir


def _load_baseline_descriptor(path: pathlib.Path) -> dict[str, Any] | None:
    descriptor_path = path / BASELINE_DESCRIPTOR
    if not descriptor_path.is_file():
        return None
    return json.loads(descriptor_path.read_text())


def _metadata_root(value: str) -> str | None:
    text = str(value or "").strip()
    return str(pathlib.Path(text).expanduser()) if text else None


def _validate_image_features(policy: Any) -> None:
    image_features = sorted(getattr(policy.config, "image_features", {}) or {})
    required = {
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }
    missing = sorted(required - set(image_features))
    if missing:
        raise RuntimeError(
            "MolmoAct2 checkpoint is missing required image features: "
            + ", ".join(missing)
        )


def _align_molmoact2_action_horizon(policy: Any) -> None:
    horizon = int(getattr(policy.config, "chunk_size", 0) or 0)
    if horizon < 1:
        raise RuntimeError(f"MolmoAct2 policy has invalid chunk_size={horizon}")

    override = getattr(policy, "_override_loaded_max_action_horizon", None)
    if callable(override):
        override(horizon)

    for model_getter in ("_hf_model", "_backbone"):
        getter = getattr(policy, model_getter, None)
        if not callable(getter):
            continue
        model = getter()
        config = getattr(model, "config", None)
        if config is not None:
            config.max_action_horizon = horizon
            action_expert_config = getattr(config, "action_expert_config", None)
            if action_expert_config is not None:
                action_expert_config.max_action_horizon = horizon
        action_expert = getattr(model, "action_expert", None)
        action_expert_config = getattr(action_expert, "config", None)
        if action_expert_config is not None:
            action_expert_config.max_action_horizon = horizon


def load_molmoact2_policy_with_checks(policy_path: pathlib.Path, device: torch.device) -> Any:
    baseline = _load_baseline_descriptor(policy_path)
    if baseline is not None:
        return load_molmoact2_baseline_policy_with_checks(baseline, device)

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.molmoact2 import MolmoAct2Policy

    config = PreTrainedConfig.from_pretrained(str(policy_path))
    if getattr(config, "type", None) != "molmoact2":
        raise RuntimeError(f"expected MolmoAct2 checkpoint config type, got {getattr(config, 'type', None)!r}")
    config.device = str(device)
    config.inference_action_mode = "continuous"
    policy = MolmoAct2Policy.from_pretrained(str(policy_path), config=config, strict=True)
    _align_molmoact2_action_horizon(policy)
    policy.eval()
    policy.to(device)
    _ensure_runtime_action_feature_names(policy)
    _validate_image_features(policy)
    return policy


def load_molmoact2_baseline_policy_with_checks(source: dict[str, Any], device: torch.device) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import dataset_to_policy_features
    from lerobot.policies.molmoact2 import MolmoAct2Config, MolmoAct2Policy
    from lerobot.utils.constants import ACTION

    dataset_repo_id = str(source["dataset_repo_id"])
    dataset_root = _metadata_root(str(source.get("dataset_root") or ""))
    dataset_meta = LeRobotDatasetMetadata(dataset_repo_id, pathlib.Path(dataset_root) if dataset_root else None)
    features = dataset_to_policy_features(dataset_meta.features)
    if ACTION not in features:
        raise RuntimeError(f"MolmoAct2 baseline dataset {dataset_repo_id!r} has no action feature")

    config = MolmoAct2Config(
        checkpoint_path=str(source["checkpoint_path"]),
        checkpoint_revision=source.get("checkpoint_revision") or None,
        checkpoint_force_download=bool(source.get("checkpoint_force_download", False)),
        action_mode=str(source.get("action_mode") or "continuous"),
        inference_action_mode="continuous",
        train_mode_vlm=str(source.get("train_mode_vlm") or "freeze"),
        model_dtype=str(source.get("model_dtype") or "bfloat16"),
        chunk_size=int(source.get("chunk_size") or 30),
        n_action_steps=int(source.get("n_action_steps") or source.get("chunk_size") or 30),
        setup_type=str(source.get("setup_type") or ""),
        control_mode=str(source.get("control_mode") or ""),
        image_keys=list(source.get("image_keys") or []),
        normalize_gripper=bool(source.get("normalize_gripper", True)),
        normalization_mapping=dict(source.get("normalization_mapping") or DEFAULT_NORMALIZATION_MAPPING),
        device=str(device),
    )
    config.output_features = {key: ft for key, ft in features.items() if key == ACTION}
    config.input_features = {key: ft for key, ft in features.items() if key != ACTION}
    config.set_dataset_feature_metadata(dataset_meta.features)
    config._xlerobot_baseline_dataset_stats = dataset_meta.stats
    config._xlerobot_baseline_dataset_meta = dataset_meta
    config._xlerobot_molmoact2_baseline = True

    policy = MolmoAct2Policy(config)
    _align_molmoact2_action_horizon(policy)
    policy.eval()
    policy.to(device)
    _ensure_runtime_action_feature_names(policy)
    _validate_image_features(policy)
    return policy


def verify_molmoact2_checkpoint(policy_path: pathlib.Path, policy: Any) -> None:
    baseline = _load_baseline_descriptor(policy_path)
    if baseline is not None:
        print(f"  Model             : {MODEL_NAME} baseline")
        print(f"  Base checkpoint   : {baseline['checkpoint_path']}")
        print(f"  Dataset stats     : {baseline['dataset_repo_id']}")
        print(f"  Action mode       : {getattr(policy.config, 'action_mode', 'n/a')}")
        print(f"  Inference mode    : {getattr(policy.config, 'inference_action_mode', 'n/a')}")
        print(f"  Action joints     : {_strict_action_names(policy)}")
        assert_molmoact2_and_xlerobot_sources()
        return

    weights = policy_path / "model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"missing finetuned weights: {weights}")
    train_step = policy_path.parent / "training_state" / "training_step.json"
    step = "unknown"
    if train_step.is_file():
        step = str(json.loads(train_step.read_text()).get("step", "?"))
    print(f"  Model             : {MODEL_NAME}")
    print(f"  Weights file      : {weights} ({weights.stat().st_size // 1_000_000} MB)")
    print(f"  Training step     : {step}")
    print(f"  Action mode       : {getattr(policy.config, 'action_mode', 'n/a')}")
    print(f"  Inference mode    : {getattr(policy.config, 'inference_action_mode', 'n/a')}")
    print(f"  Action joints     : {_strict_action_names(policy)}")
    assert_molmoact2_and_xlerobot_sources()


def _parse_args(
    argv: list[str] | None = None,
    *,
    require_task: bool = True,
) -> argparse.Namespace:
    molmo = _molmo_cfg()
    ds = _dataset_cfg()
    defaults = argparse.ArgumentParser(add_help=False)
    defaults.add_argument("--checkpoint-path", default=None)
    defaults.add_argument("--checkpoint-revision", default=None)
    defaults.add_argument("--checkpoint-force-download", action="store_true")
    defaults.add_argument("--dataset-repo-id", default=str(ds.get("repo_id") or ""))
    defaults.add_argument("--dataset-root", default=str(ds.get("root") or ""))
    defaults.add_argument("--action-mode", default=str(molmo.get("action_mode") or "continuous"), choices=["continuous", "both"])
    defaults.add_argument("--train-mode-vlm", default=str(molmo.get("train_mode_vlm") or "freeze"), choices=["freeze", "lora", "fft"])
    defaults.add_argument("--model-dtype", default=str(molmo.get("model_dtype") or "bfloat16"), choices=["float32", "float16", "bfloat16"])
    defaults.add_argument("--setup-type", default=str(molmo.get("setup_type") or "dual-arm XLerobot tabletop manipulation"))
    defaults.add_argument("--control-mode", default=str(molmo.get("control_mode") or "absolute joint position in degrees"))
    defaults.add_argument(
        "--normalize-gripper",
        action=argparse.BooleanOptionalAction,
        default=bool(molmo.get("normalize_gripper", True)),
    )
    defaults.add_argument(
        "--image-keys-json",
        default=json.dumps(molmo.get("image_keys") if isinstance(molmo.get("image_keys"), list) else DEFAULT_IMAGE_KEYS),
    )
    defaults.add_argument(
        "--normalization-mapping-json",
        default=json.dumps(
            molmo.get("normalization_mapping")
            if isinstance(molmo.get("normalization_mapping"), dict)
            else DEFAULT_NORMALIZATION_MAPPING
        ),
    )
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    baseline_args, remaining = defaults.parse_known_args(raw_argv)
    args = _ORIGINAL_PARSE_ARGS(remaining, require_task=False)
    for key, value in vars(baseline_args).items():
        setattr(args, key, value)
    if args.train_mode_vlm == "freeze" and args.action_mode != "continuous":
        raise SystemExit("--train-mode-vlm=freeze requires --action-mode=continuous")
    if args.checkpoint_path and args.policy_path:
        raise SystemExit("use only one of --policy-path or --checkpoint-path")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--action-horizon" not in raw_argv:
        args.action_horizon = int(molmo.get("action_horizon", molmo.get("n_action_steps", 30)))
    if "--open-loop-steps" not in raw_argv:
        args.open_loop_steps = int(molmo.get("open_loop_steps", molmo.get("n_action_steps", 30)))
    if args.checkpoint_path and not args.dry_run_home:
        if not args.dataset_repo_id:
            raise SystemExit("--dataset-repo-id is required with --checkpoint-path baseline inference")
        args.policy_path = str(_write_baseline_descriptor(args))
    elif args.policy_path is None and not args.dry_run and not args.dry_run_home:
        args.policy_path = str(
            molmo.get("policy_path")
            or "outputs/molmoact2_finetune/checkpoints/last/pretrained_model"
        )
    if require_task and not args.dry_run and not args.dry_run_home:
        if not args.policy_path or not args.task:
            raise SystemExit("--policy-path and --task are required for inference")
    return args


def main() -> None:
    from lerobot.policies import factory as policy_factory

    original_make_pre_post_processors = policy_factory.make_pre_post_processors

    def make_molmoact2_pre_post_processors(policy_cfg: Any, pretrained_path: str | None = None, **kwargs: Any):
        if getattr(policy_cfg, "_xlerobot_molmoact2_baseline", False):
            return original_make_pre_post_processors(
                policy_cfg,
                pretrained_path=None,
                dataset_stats=getattr(policy_cfg, "_xlerobot_baseline_dataset_stats"),
                dataset_meta=getattr(policy_cfg, "_xlerobot_baseline_dataset_meta"),
            )
        return original_make_pre_post_processors(policy_cfg, pretrained_path=pretrained_path, **kwargs)

    policy_factory.make_pre_post_processors = make_molmoact2_pre_post_processors
    runtime._load_pi05_policy_with_compat = load_molmoact2_policy_with_checks
    runtime._verify_policy_checkpoint = verify_molmoact2_checkpoint
    runtime._parse_args = _parse_args
    runtime.main()


for _name, _value in vars(runtime).items():
    if _name not in globals() and _name not in {"__name__", "__package__", "__loader__", "__spec__", "__file__", "__cached__"}:
        globals()[_name] = _value


if __name__ == "__main__":
    main()
