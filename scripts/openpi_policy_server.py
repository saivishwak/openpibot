#!/usr/bin/env python3
"""Start an OpenPI WebSocket policy server from the installed `openpi` package."""

from __future__ import annotations

import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Reserved for future default-policy serving modes."""


@dataclasses.dataclass
class Args:
    """Arguments for the OpenPI policy server."""

    port: int = 8000
    default_prompt: str | None = None
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(
        default_factory=lambda: Checkpoint(
            config="pi05_bimanual_so101_lora",
            dir="gs://openpi-assets/checkpoints/pi05_base",
        )
    )


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            raise ValueError("default OpenPI policy serving is not configured for OpenPIBot")


def main(args: Args) -> None:
    policy = create_policy(args)
    metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating OpenPI policy server (host=%s, ip=%s, port=%s)", hostname, local_ip, args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
