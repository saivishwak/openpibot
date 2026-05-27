#!/usr/bin/env python3
"""Strict PI05 checkpoint loader with local key-compat handling.

This module keeps PI05 weight-compat logic in the root repo so we don't need to
patch the `lerobot` submodule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from lerobot.configs import PreTrainedConfig


def load_pi05_policy_with_compat(policy_path: str | Path, device: torch.device) -> Any:
    """Load PI05 policy strictly, with vision namespace fallback remap.

    Handles checkpoints where vision keys are named either:
    - `vision_tower.*`
    - `vision_tower.vision_model.*`
    """
    from lerobot.policies.pi05 import PI05Policy

    policy_path = Path(policy_path).resolve()
    config = PreTrainedConfig.from_pretrained(str(policy_path))
    policy = PI05Policy(config)

    weights_path = policy_path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"missing checkpoint weights: {weights_path}")

    original_state = load_file(str(weights_path))
    fixed_state = policy._fix_pytorch_state_dict_keys(original_state, policy.config)

    remapped: dict[str, torch.Tensor] = {}
    for key, value in fixed_state.items():
        remapped[key if key.startswith("model.") else f"model.{key}"] = value

    try:
        missing, unexpected = policy.load_state_dict(remapped, strict=True)
    except RuntimeError as first_err:
        alt = dict(remapped)
        remapped_vision = 0
        for key, value in list(remapped.items()):
            if "vision_tower.vision_model." in key:
                alt_key = key.replace("vision_tower.vision_model.", "vision_tower.")
                alt.pop(key, None)
                alt[alt_key] = value
                remapped_vision += 1
            elif "vision_tower." in key and "vision_tower.vision_model." not in key:
                alt_key = key.replace("vision_tower.", "vision_tower.vision_model.")
                alt.pop(key, None)
                alt[alt_key] = value
                remapped_vision += 1
        if remapped_vision == 0:
            raise RuntimeError(f"PI05 strict load failed: {first_err}") from first_err
        print(
            "PI05 strict load retry with alternate vision namespace "
            f"({remapped_vision} keys)."
        )
        missing, unexpected = policy.load_state_dict(alt, strict=True)

    if missing or unexpected:
        raise RuntimeError(
            f"PI05 strict load incomplete (missing={len(missing)}, unexpected={len(unexpected)})."
        )

    policy.eval()
    policy.to(device)
    print("PI05 strict compatibility loader: all keys loaded successfully.")
    return policy
