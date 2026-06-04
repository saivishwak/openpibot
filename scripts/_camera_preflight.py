"""V4L device checks before inference (paths, concurrent access)."""
from __future__ import annotations

import pathlib
import subprocess
from typing import Any

import yaml


def _camera_paths(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    cams = cfg.get("cameras") or {}
    out: list[tuple[str, str]] = []
    for role in ("head", "left_wrist", "right_wrist"):
        entry = cams.get(role)
        if isinstance(entry, dict) and entry.get("path"):
            out.append((role, str(entry["path"])))
    return out


def run_camera_preflight(config_yaml: pathlib.Path) -> None:
    """Print warnings for missing devices or other processes holding V4L nodes."""
    if not config_yaml.is_file():
        return
    with config_yaml.open() as f:
        cfg = yaml.safe_load(f) or {}
    paths = _camera_paths(cfg)
    if not paths:
        print("Camera preflight: no camera paths in config.")
        return

    print("Camera preflight:")
    any_busy = False
    for role, path in paths:
        p = pathlib.Path(path)
        exists = p.exists()
        status = "ok" if exists else "MISSING (check USB cable / hub)"
        print(f"  {role:12} {status}\n    {path}")
        if not exists:
            continue
        try:
            proc = subprocess.run(
                ["fuser", "-v", path],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        combined = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0 and combined.strip():
            any_busy = True
            print(
                f"    WARNING: another process is using this device — stop dashboard "
                f"backend / other capture before inference:\n"
                f"    {combined.strip()[:400]}"
            )

    rw = next((p for r, p in paths if r == "right_wrist"), "")
    if "usb-0:4.1" in rw:
        print(
            "  Note: right_wrist is on USB hub port 4.1; it often drops under load "
            "(motors + 3×30fps). Use --camera-backend dashboard (default), stop duplicate "
            "captures, and prefer a powered hub or a different port if drops continue."
        )
    if any_busy:
        print(
            "  Stop conflicting processes, then re-run inference. "
            "Double-opens commonly cause errno=19 (No such device)."
        )
