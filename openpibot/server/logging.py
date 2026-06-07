"""Logging setup for OpenPiBot."""

from __future__ import annotations

import logging
import logging.handlers
import os
import pathlib

from .config import REPO_ROOT

LOG_DIR = REPO_ROOT / ".openpibot" / "logs"
LOG_FILE = LOG_DIR / "server.log"


def _resolve_log_file(log_file: str | pathlib.Path | None = None) -> pathlib.Path:
    raw = log_file or os.environ.get("OPENPIBOT_LOG_FILE") or LOG_FILE
    path = pathlib.Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _normalize_level(level: int | str) -> int | str:
    return level.upper() if isinstance(level, str) else level


def configure_logging(
    level: int | str | None = None,
    log_file: str | pathlib.Path | None = None,
) -> pathlib.Path:
    global LOG_FILE
    resolved_log_file = _resolve_log_file(log_file)
    LOG_FILE = resolved_log_file
    os.environ["OPENPIBOT_LOG_FILE"] = str(resolved_log_file)
    resolved_level = (
        level if level is not None else os.environ.get("OPENPIBOT_LOG_LEVEL", logging.INFO)
    )
    if level is not None:
        os.environ["OPENPIBOT_LOG_LEVEL"] = str(level)
    resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(_normalize_level(resolved_level))

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    if not any(
        isinstance(h, logging.handlers.RotatingFileHandler)
        and pathlib.Path(h.baseFilename) == resolved_log_file
        for h in root.handlers
    ):
        file_handler = logging.handlers.RotatingFileHandler(
            resolved_log_file,
            maxBytes=2 * 1024 * 1024,
            backupCount=5,
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)

    return resolved_log_file


def current_log_file() -> pathlib.Path:
    return LOG_FILE
