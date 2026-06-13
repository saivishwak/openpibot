#!/usr/bin/env python3
"""CLI wrapper for the embeddable warm PI0.5 runner."""

from __future__ import annotations

from openpibot.pi05_runner import (
    parse_runner_cli_args,
    runtime_options_from_infer_args,
    serve_runner,
)


def main() -> None:
    args = parse_runner_cli_args()
    runtime_options = runtime_options_from_infer_args(args.infer_args)
    serve_runner(host=args.host, port=int(args.port), runtime_options=runtime_options)


if __name__ == "__main__":
    main()
