#!/usr/bin/env python3
"""Offline evaluation for finetuned PI0.5 on LeRobot dataset frames.

This script does NOT move hardware. It loads a finetuned PI0.5 checkpoint,
replays samples from a LeRobot dataset, runs policy inference, and compares:
  - predicted action vs dataset action
  - predicted action vs observation.state (sanity)

Use it to quickly validate whether the finetuned checkpoint behaves correctly
before debugging robot-side inference/control details.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
from typing import Any

import numpy as np
import torch

from lerobot.common.control_utils import predict_action
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from _pi05_loader import load_pi05_policy_with_compat


IMAGE_KEY_CANDIDATES: list[tuple[str, str]] = [
    ("observation.images.head", "observation.images.base_0_rgb"),
    ("observation.images.left_wrist", "observation.images.left_wrist_0_rgb"),
    ("observation.images.right_wrist", "observation.images.right_wrist_0_rgb"),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--policy-path",
        required=True,
        help="Path to finetuned checkpoint (.../pretrained_model).",
    )
    p.add_argument(
        "--dataset-repo-id",
        default="saivishwak/xlerobot-vr-teleop",
        help="LeRobot dataset repo id.",
    )
    p.add_argument(
        "--episodes",
        default="0",
        help="Comma-separated episode indices (e.g. 0,1,2).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max frames to evaluate.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Evaluate every N-th frame.",
    )
    p.add_argument(
        "--task",
        default=None,
        help="Override task text. Default uses frame['task'].",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Inference device.",
    )
    p.add_argument(
        "--print-worst",
        type=int,
        default=5,
        help="Print top-K worst frames by max joint error.",
    )
    return p.parse_args()


def _parse_episode_csv(text: str) -> list[int]:
    vals = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    if not vals:
        raise ValueError("No episode indices parsed from --episodes")
    return vals


def _pick_image(frame: dict[str, Any], a: str, b: str) -> np.ndarray:
    if a in frame:
        return frame[a].cpu().numpy()
    if b in frame:
        return frame[b].cpu().numpy()
    raise KeyError(f"missing both image keys: {a} and {b}")


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """Convert dataset image tensor/array to HWC uint8 expected by inference helper."""
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"expected image with 3 dims, got shape={arr.shape}")
    # Dataset often stores CHW tensors, while prepare_observation_for_inference expects HWC.
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        # LeRobot datasets commonly store float images in [0, 1].
        # convert to uint8 so `prepare_observation_for_inference` (which divides by 255)
        # matches real robot inference path.
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = np.rint(arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _build_obs(frame: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "observation.images.head": _to_hwc_uint8(
            _pick_image(frame, IMAGE_KEY_CANDIDATES[0][0], IMAGE_KEY_CANDIDATES[0][1])
        ),
        "observation.images.right_wrist": _to_hwc_uint8(
            _pick_image(frame, IMAGE_KEY_CANDIDATES[2][0], IMAGE_KEY_CANDIDATES[2][1])
        ),
        "observation.images.left_wrist": _to_hwc_uint8(
            _pick_image(frame, IMAGE_KEY_CANDIDATES[1][0], IMAGE_KEY_CANDIDATES[1][1])
        ),
        "observation.state": frame["observation.state"].cpu().numpy(),
    }


def _fmt_triplet(values: list[float]) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=np.float64)
    return f"mean={arr.mean():.2f} p95={np.percentile(arr, 95):.2f} max={arr.max():.2f}"


def main() -> None:
    args = _parse_args()
    episodes = _parse_episode_csv(args.episodes)

    policy_path = pathlib.Path(args.policy_path).resolve()
    if not policy_path.is_dir():
        raise SystemExit(f"policy path not found: {policy_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    print("=" * 72)
    print(f"Policy   : {policy_path}")
    print(f"Dataset  : {args.dataset_repo_id}")
    print(f"Episodes : {episodes}")
    print(f"Device   : {device}")
    print(f"Max samp : {args.max_samples} (stride={args.stride})")
    print("=" * 72)

    policy = load_pi05_policy_with_compat(policy_path, device)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(policy_path))

    ds = LeRobotDataset(args.dataset_repo_id, episodes=episodes)
    print(f"Loaded dataset split with {len(ds)} frames.")

    n = 0
    pred_vs_action_max: list[float] = []
    pred_vs_action_mae: list[float] = []
    pred_vs_state_max: list[float] = []
    gt_vs_state_max: list[float] = []
    worst: list[tuple[float, int, str]] = []

    for i in range(0, len(ds), max(1, args.stride)):
        frame = ds[i]
        obs = _build_obs(frame)
        task = args.task if args.task is not None else str(frame.get("task", ""))

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        pred = predict_action(
            obs,
            policy,
            device,
            preprocessor,
            postprocessor,
            use_amp=bool(getattr(policy.config, "use_amp", False)),
            task=task,
            robot_type="xlerobot",
        )
        pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
        gt = frame["action"].cpu().numpy().astype(np.float32)
        state = frame["observation.state"].cpu().numpy().astype(np.float32)

        e = np.abs(pred_np - gt)
        e_max = float(e.max())
        e_mae = float(e.mean())
        p_s_max = float(np.abs(pred_np - state).max())
        g_s_max = float(np.abs(gt - state).max())

        pred_vs_action_max.append(e_max)
        pred_vs_action_mae.append(e_mae)
        pred_vs_state_max.append(p_s_max)
        gt_vs_state_max.append(g_s_max)
        worst.append((e_max, i, task))

        n += 1
        if n >= args.max_samples:
            break

    worst.sort(key=lambda x: x[0], reverse=True)
    show = max(0, args.print_worst)

    print("\nResults")
    print("-" * 72)
    print(f"evaluated frames                     : {n}")
    print(f"|pred-action| max                    : {_fmt_triplet(pred_vs_action_max)}")
    print(f"|pred-action| mean(abs over joints)  : {_fmt_triplet(pred_vs_action_mae)}")
    print(f"|pred-state| max                     : {_fmt_triplet(pred_vs_state_max)}")
    print(f"|gt-state| max                       : {_fmt_triplet(gt_vs_state_max)}")
    if pred_vs_action_max and gt_vs_state_max:
        ratio = statistics.mean(pred_vs_action_max) / (statistics.mean(gt_vs_state_max) + 1e-6)
        print(f"mean(max|pred-action|)/mean(max|gt-state|): {ratio:.3f}")

    if show > 0:
        print("\nWorst frames (by max |pred-action|)")
        print("-" * 72)
        for e_max, idx, task in worst[:show]:
            print(f"frame={idx:6d}  max_err={e_max:7.2f} deg  task={task!r}")


if __name__ == "__main__":
    main()

