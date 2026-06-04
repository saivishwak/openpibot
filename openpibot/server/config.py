"""Project configuration helpers."""

from __future__ import annotations

import pathlib
from typing import Any

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"
VR_CALIBRATION_YAML = REPO_ROOT / "config" / "vr_calibration.yaml"
STATIC_ROOT = REPO_ROOT / "dashboard" / "dist"
REFERENCE_ROOT = REPO_ROOT / "reference"
UPSTREAM_OPENPI_ROOT = REFERENCE_ROOT / "openpi_upstream"


def load_yaml(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        return {}
    return data


def load_project_config() -> dict[str, Any]:
    return load_yaml(CONFIG_YAML)


def load_vr_calibration_config() -> dict[str, Any]:
    return load_yaml(VR_CALIBRATION_YAML)


def get_active_robot_id(config: dict[str, Any] | None = None) -> str:
    cfg = config if config is not None else load_project_config()
    if isinstance(cfg.get("active_robot"), str) and cfg["active_robot"].strip():
        return cfg["active_robot"].strip()
    robot = cfg.get("robot") if isinstance(cfg.get("robot"), dict) else {}
    return str(robot.get("id") or "xlerobot")


def get_robot_profiles(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return robot profiles from either the new `robots:` map or legacy `robot:` block."""
    cfg = config if config is not None else load_project_config()
    active = get_active_robot_id(cfg)
    profiles: list[dict[str, Any]] = []

    robots = cfg.get("robots")
    if isinstance(robots, dict):
        for robot_id, raw in robots.items():
            if not isinstance(raw, dict):
                continue
            profile = dict(raw)
            profile.setdefault("id", str(robot_id))
            profile["active"] = str(profile["id"]) == active
            profiles.append(profile)

    legacy = cfg.get("robot")
    if isinstance(legacy, dict):
        profile = dict(legacy)
        profile.setdefault("id", str(profile.get("id") or active))
        profile.setdefault("name", "XLeRobot")
        profile["active"] = str(profile["id"]) == active
        if not any(p["id"] == profile["id"] for p in profiles):
            profiles.insert(0, profile)

    return profiles
