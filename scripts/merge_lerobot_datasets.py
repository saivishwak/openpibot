#!/usr/bin/env python3
"""Merge multiple local LeRobot datasets into one new dataset root.

Examples:
  uv run python scripts/merge_lerobot_datasets.py \
    --dataset saivishwak/xlerobot-desk-cleanup-phase1=~/.cache/huggingface/lerobot/saivishwak/xlerobot-desk-cleanup-phase1 \
    --dataset saivishwak/xlerobot-vr-pick-place-pen=~/.cache/huggingface/lerobot/saivishwak/xlerobot-vr-pick-place-pen \
    --output-repo-id saivishwak/xlerobot-merged \
    --output-root ~/.cache/huggingface/lerobot/saivishwak/xlerobot-merged
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    root: pathlib.Path


def _parse_dataset_spec(value: str) -> DatasetSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "dataset must be in the form repo_id=/path/to/root, "
            "for example saivishwak/my-dataset=~/.cache/huggingface/lerobot/saivishwak/my-dataset"
        )

    repo_id, root = value.split("=", 1)
    repo_id = repo_id.strip()
    root = root.strip()
    if not repo_id:
        raise argparse.ArgumentTypeError("dataset repo_id is empty")
    if not root:
        raise argparse.ArgumentTypeError("dataset root is empty")

    return DatasetSpec(repo_id=repo_id, root=pathlib.Path(root).expanduser().resolve())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset_spec,
        required=True,
        help="Input dataset as repo_id=/path/to/root. Repeat for each input dataset.",
    )
    parser.add_argument(
        "--output-repo-id",
        required=True,
        help="Repo id to write into the merged dataset metadata, e.g. saivishwak/xlerobot-merged.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="New local root directory for the merged LeRobot dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove --output-root first if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved inputs and expected totals without writing the merged dataset.",
    )
    return parser.parse_args()


def _validate_root(root: pathlib.Path) -> None:
    if not root.is_dir():
        sys.exit(f"dataset root not found: {root}")
    if not (root / "meta" / "info.json").is_file():
        sys.exit(f"not a LeRobot dataset root (missing meta/info.json): {root}")


def main() -> None:
    args = _parse_args()
    specs: list[DatasetSpec] = args.dataset
    output_root = pathlib.Path(args.output_root).expanduser().resolve()

    if len(specs) < 2:
        sys.exit("provide at least two --dataset entries to merge")

    for spec in specs:
        _validate_root(spec.root)

    if output_root.exists() and not args.overwrite:
        sys.exit(f"output root already exists; pass --overwrite to replace it: {output_root}")
    if output_root.exists() and args.overwrite and not output_root.is_dir():
        sys.exit(f"output root exists but is not a directory: {output_root}")

    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    datasets = []
    expected_episodes = 0
    expected_frames = 0

    print("Inputs:")
    for spec in specs:
        ds = LeRobotDataset(repo_id=spec.repo_id, root=spec.root)
        datasets.append(ds)
        expected_episodes += int(ds.meta.total_episodes)
        expected_frames += int(ds.meta.total_frames)
        print(
            f"  {spec.repo_id} @ {spec.root} "
            f"episodes={ds.meta.total_episodes} frames={ds.meta.total_frames}"
        )

    print(f"Output repo id : {args.output_repo_id}")
    print(f"Output root    : {output_root}")
    print(f"Expected total : episodes={expected_episodes} frames={expected_frames}")

    if args.dry_run:
        return

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)

    merged = merge_datasets(
        datasets=datasets,
        output_repo_id=str(args.output_repo_id),
        output_dir=output_root,
    )

    # Re-open from disk so the validation covers the written folder, not just the in-memory object.
    reloaded = LeRobotDataset(repo_id=str(args.output_repo_id), root=output_root)
    if int(reloaded.meta.total_episodes) != expected_episodes:
        raise RuntimeError(
            f"episode count mismatch: expected {expected_episodes}, got {reloaded.meta.total_episodes}"
        )
    if int(reloaded.meta.total_frames) != expected_frames:
        raise RuntimeError(f"frame count mismatch: expected {expected_frames}, got {reloaded.meta.total_frames}")

    print(
        f"Merged dataset written: {merged.root} "
        f"episodes={reloaded.meta.total_episodes} frames={reloaded.meta.total_frames}"
    )


if __name__ == "__main__":
    main()
