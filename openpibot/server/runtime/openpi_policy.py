"""Command builder for the package-managed OpenPI policy server."""

from __future__ import annotations

import os

from openpibot.server.config import REPO_ROOT, load_project_config

DEFAULT_OPENPI_PACKAGE = "openpi @ git+https://github.com/Physical-Intelligence/openpi.git"
DEFAULT_OPENPI_CONFIG = "pi05_bimanual_so101_lora"
DEFAULT_OPENPI_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_base"


def build_openpi_policy_server_command() -> list[str]:
    cfg = load_project_config()
    pi05 = cfg.get("pi05") if isinstance(cfg.get("pi05"), dict) else {}

    package = str(os.environ.get("OPENPI_PACKAGE") or pi05.get("package") or DEFAULT_OPENPI_PACKAGE)
    config = str(os.environ.get("OPENPI_CONFIG") or pi05.get("openpi_config") or DEFAULT_OPENPI_CONFIG)
    checkpoint = str(
        os.environ.get("OPENPI_CHECKPOINT_DIR")
        or pi05.get("checkpoint_dir")
        or DEFAULT_OPENPI_CHECKPOINT
    )
    port = str(os.environ.get("OPENPI_PORT") or pi05.get("server_port") or 8000)

    return [
        "uv",
        "run",
        "--no-project",
        "--with",
        package,
        "python",
        str(REPO_ROOT / "scripts" / "openpi_policy_server.py"),
        "--port",
        port,
        "policy:checkpoint",
        f"--policy.config={config}",
        f"--policy.dir={checkpoint}",
    ]
