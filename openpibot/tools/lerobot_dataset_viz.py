"""Project-owned LeRobot dataset visualization CLI.

The upstream LeRobot command does not expose ``video_backend``. This project
records videos that load correctly through PyAV in the current environment, so
the local console script constructs ``LeRobotDataset`` with ``video_backend``
explicitly and then delegates rendering to LeRobot's visualization helper.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lerobot.datasets import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import visualize_dataset
from lerobot.utils.utils import init_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True, help="Name of hugging face repository.")
    parser.add_argument("--root", type=Path, default=None, help="Root directory for the dataset stored locally.")
    parser.add_argument("--episode-index", type=int, required=True, help="Episode to visualize.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size loaded by the dataloader.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for the dataloader.")
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["local", "distant"],
        help=(
            "Use local to spawn a Rerun viewer, or distant to serve a viewer over gRPC/web "
            "for another machine to connect."
        ),
    )
    parser.add_argument("--web-port", type=int, default=9090, help="Web port when --mode distant is set.")
    parser.add_argument("--ws-port", type=int, help="Deprecated; use --grpc-port instead.")
    parser.add_argument("--grpc-port", type=int, default=9876, help="gRPC port when --mode distant is set.")
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help="Save a .rrd file under --output-dir instead of spawning a viewer.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory used with --save.")
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="Timestamp tolerance passed to LeRobotDataset.",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        choices=["pyav", "torchcodec", "video_reader"],
        help="Dataset video decoder backend. Defaults to pyav for this project.",
    )
    parser.add_argument(
        "--display-compressed-images",
        action="store_true",
        help="Display compressed images in Rerun instead of uncompressed ones.",
    )

    args = parser.parse_args()
    if args.ws_port is not None:
        logging.warning("--ws-port is deprecated and will be removed in future versions. Use --grpc-port.")
        args.grpc_port = args.ws_port

    init_logging()
    logging.info("Loading dataset with video_backend=%s", args.video_backend)
    dataset = LeRobotDataset(
        args.repo_id,
        episodes=[args.episode_index],
        root=args.root,
        tolerance_s=args.tolerance_s,
        video_backend=args.video_backend,
    )

    visualize_dataset(
        dataset,
        episode_index=args.episode_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.mode,
        web_port=args.web_port,
        grpc_port=args.grpc_port,
        save=bool(args.save),
        output_dir=args.output_dir,
        display_compressed_images=args.display_compressed_images,
    )
