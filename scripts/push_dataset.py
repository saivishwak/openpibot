#!/usr/bin/env python3
"""Push a local LeRobot dataset directory to Hugging Face Hub.

For multi-GB datasets (many video files), use large-folder upload (default when
local size > 500 MB). It is resumable and avoids upload_folder stalling ~60%.

Examples:
  uv run python scripts/push_dataset.py
  uv run python scripts/push_dataset.py --repo-id saivishwak/xlerobot-vr-teleop-medicine-bowl
  uv run python scripts/push_dataset.py --root /custom/path/to/dataset
  uv run python scripts/push_dataset.py --large --num-workers 4
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

_LARGE_UPLOAD_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MiB


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"


def _load_dataset_cfg() -> dict:
    if not CONFIG_YAML.is_file():
        return {}
    try:
        cfg = yaml.safe_load(CONFIG_YAML.read_text()) or {}
    except Exception:
        return {}
    return cfg.get("dataset") or {}


def _resolve_root(repo_id: str, root: str | None) -> pathlib.Path:
    if root:
        return pathlib.Path(root).expanduser().resolve()
    import os

    env_home = os.environ.get("HF_LEROBOT_HOME")
    if env_home:
        hf_home = pathlib.Path(env_home).expanduser().resolve()
    else:
        hf_home = pathlib.Path("~/.cache/huggingface/lerobot").expanduser().resolve()
    return (hf_home / repo_id).resolve()


def _dir_size_bytes(path: pathlib.Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GiB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.1f} MiB"
    return f"{num_bytes / 1024:.0f} KiB"


def _parse_args() -> argparse.Namespace:
    ds_cfg = _load_dataset_cfg()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-id",
        default=ds_cfg.get("repo_id", "saivishwak/xlerobot-vr-teleop"),
        help="HF dataset repo id (e.g. user/name).",
    )
    p.add_argument(
        "--root",
        default=ds_cfg.get("root"),
        help=(
            "Local dataset root path. If omitted, uses dataset.root from config; "
            "otherwise falls back to $HF_LEROBOT_HOME/<repo_id> or ~/.cache/huggingface/lerobot/<repo_id>."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved settings and exit without uploading.",
    )
    p.add_argument(
        "--large",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use Hugging Face upload_large_folder (resumable, parallel). "
            "Default: auto-on when local dataset > 500 MiB."
        ),
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Parallel workers for --large upload (default: 4).",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create/update the Hub repo as private.",
    )
    p.add_argument(
        "--no-videos",
        action="store_true",
        help="Skip uploading videos/ (parquet data only; faster smoke test).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = _resolve_root(str(args.repo_id), args.root)

    print(f"Repo ID        : {args.repo_id}")
    print(f"Local root     : {root}")

    if not root.is_dir():
        sys.exit(f"dataset root not found: {root}")
    if not (root / "meta" / "info.json").is_file():
        sys.exit(f"not a LeRobot dataset root (missing meta/info.json): {root}")
    size_b = _dir_size_bytes(root)
    print(f"Local size       : {_format_size(size_b)}")
    use_large = (
        bool(args.large)
        if args.large is not None
        else size_b >= _LARGE_UPLOAD_THRESHOLD_BYTES
    )
    print(
        f"Upload mode      : {'upload_large_folder (resumable)' if use_large else 'upload_folder'}"
    )
    if use_large:
        print(f"Workers          : {args.num_workers}")
        print(
            "Tip: if a plain upload stalled, Ctrl+C and re-run the same command — "
            "large-folder upload skips files already on the Hub."
        )

    if args.dry_run:
        return

    import contextlib

    from huggingface_hub import HfApi
    from huggingface_hub.errors import RevisionNotFoundError

    from lerobot.datasets import CODEBASE_VERSION, create_lerobot_dataset_card
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=str(args.repo_id), root=root)
    push_videos = not args.no_videos
    ignore_patterns = ["images/"]
    if not push_videos:
        ignore_patterns.append("videos/")

    hub_api = HfApi()
    hub_api.create_repo(
        repo_id=str(args.repo_id),
        repo_type="dataset",
        private=bool(args.private),
        exist_ok=True,
    )

    if use_large:
        print("Uploading files (large-folder mode, resumable)...")
        hub_api.upload_large_folder(
            repo_id=str(args.repo_id),
            folder_path=str(root),
            repo_type="dataset",
            ignore_patterns=ignore_patterns,
            num_workers=max(1, int(args.num_workers)),
            print_report=True,
            print_report_every=30,
        )
    else:
        print("Uploading files...")
        hub_api.upload_folder(
            repo_id=str(args.repo_id),
            folder_path=str(root),
            repo_type="dataset",
            ignore_patterns=ignore_patterns,
        )

    card = create_lerobot_dataset_card(
        dataset_info=ds.meta.info,
        license="apache-2.0",
        repo_id=str(args.repo_id),
    )
    card.push_to_hub(repo_id=str(args.repo_id), repo_type="dataset")
    with contextlib.suppress(RevisionNotFoundError):
        hub_api.delete_tag(str(args.repo_id), tag=CODEBASE_VERSION, repo_type="dataset")
    hub_api.create_tag(str(args.repo_id), tag=CODEBASE_VERSION, repo_type="dataset")

    print(f"Uploaded dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
