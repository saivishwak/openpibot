"""Camera role / path validation for inference (root app only)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

_CAMERA_ROLES = ("head", "left_wrist", "right_wrist")

# Matches policy_preprocessor.json from finetune (rename_observations_processor step).
PI05_IMAGE_RENAME: dict[str, str] = {
    "observation.images.head": "observation.images.base_0_rgb",
    "observation.images.left_wrist": "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist": "observation.images.right_wrist_0_rgb",
}


def print_inference_camera_map(cfg: dict[str, Any], policy: Any | None = None) -> None:
    """Log YAML roles → observation keys → PI05 policy keys (after preprocessor rename)."""
    cams = cfg.get("cameras") or {}
    print("Camera pipeline (config role → infer obs key → policy key after preprocessor):")
    for role in _CAMERA_ROLES:
        block = cams.get(role) or {}
        path = block.get("path", "")
        exists = Path(path).exists() if path else False
        status = "ok" if exists else "MISSING" if path else "not configured"
        obs_key = f"observation.images.{role}"
        policy_key = PI05_IMAGE_RENAME.get(obs_key, obs_key)
        print(f"  {role:12} device {path or '(no path)'}  [{status}]")
        print(f"               {obs_key}  →  {policy_key}")
    if policy is not None:
        img_feats = sorted(getattr(policy.config, "image_features", {}) or {})
        if img_feats:
            print(f"Policy image_features ({len(img_feats)}): {', '.join(img_feats)}")
