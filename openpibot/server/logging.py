"""Logging setup for OpenPiBot."""

from __future__ import annotations

import logging
import logging.handlers
import pathlib

from .config import REPO_ROOT

LOG_DIR = REPO_ROOT / ".openpibot" / "logs"
LOG_FILE = LOG_DIR / "server.log"


def configure_logging(level: int | str = logging.INFO) -> pathlib.Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    if not any(isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == str(LOG_FILE) for h in root.handlers):
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=2 * 1024 * 1024,
            backupCount=5,
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)

    return LOG_FILE
