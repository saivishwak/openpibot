"""Import-path isolation for the AllenAI MolmoAct2 LeRobot vendor.

The root project currently uses `vendor/lerobot` for the XLerobot hardware
driver and PI0.5. MolmoAct2 exists in `vendor/allenai-lerobot`, which exposes
the same top-level Python package name: `lerobot`.

These helpers keep that collision explicit and fail fast when the selected
runtime imports the wrong vendor.
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLENAI_LEROBOT_ROOT = REPO_ROOT / "vendor" / "allenai-lerobot"
ALLENAI_LEROBOT_SRC = ALLENAI_LEROBOT_ROOT / "src"
XLEROBOT_LEROBOT_ROOT = REPO_ROOT / "vendor" / "lerobot"
XLEROBOT_LEROBOT_SRC = XLEROBOT_LEROBOT_ROOT / "src"


def _ensure_dir(path: pathlib.Path, label: str) -> None:
    if not path.is_dir():
        raise RuntimeError(f"missing {label}: {path}")


def prepend_allenai_lerobot_src() -> pathlib.Path:
    """Make the current process import `lerobot` from AllenAI first."""
    _ensure_dir(ALLENAI_LEROBOT_SRC, "AllenAI LeRobot source tree")
    src = str(ALLENAI_LEROBOT_SRC)
    if src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)
    return ALLENAI_LEROBOT_SRC


def molmoact2_subprocess_env(extra_pythonpath: list[str] | None = None) -> dict[str, str]:
    """Return an environment where AllenAI LeRobot wins import resolution."""
    _ensure_dir(ALLENAI_LEROBOT_SRC, "AllenAI LeRobot source tree")
    env = dict(os.environ)
    parts = [str(ALLENAI_LEROBOT_SRC)]
    parts.extend(extra_pythonpath or [])
    current = env.get("PYTHONPATH")
    if current:
        parts.append(current)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def assert_allenai_lerobot_imported() -> Any:
    """Import `lerobot` and verify it came from the AllenAI vendor."""
    import lerobot

    loaded = pathlib.Path(getattr(lerobot, "__file__", "")).resolve()
    expected = ALLENAI_LEROBOT_SRC.resolve()
    if expected not in loaded.parents:
        raise RuntimeError(
            "MolmoAct2 requires the AllenAI LeRobot vendor, but imported "
            f"`lerobot` from {loaded}. Expected a path under {expected}."
        )
    return lerobot


def extend_xlerobot_hardware_paths() -> None:
    """Expose XLerobot hardware-only modules from `vendor/lerobot`.

    This is needed only for on-robot MolmoAct2 inference. The MolmoAct2 policy
    package is imported from AllenAI LeRobot, while the XLerobot driver exists
    only in the existing LeRobot vendor checked into this repo.
    """
    _ensure_dir(XLEROBOT_LEROBOT_SRC, "XLerobot LeRobot source tree")
    assert_allenai_lerobot_imported()

    import lerobot.robots
    import lerobot.teleoperators

    robots_path = str(XLEROBOT_LEROBOT_SRC / "lerobot" / "robots")
    teleoperators_path = str(XLEROBOT_LEROBOT_SRC / "lerobot" / "teleoperators")
    for package, path in (
        (lerobot.robots, robots_path),
        (lerobot.teleoperators, teleoperators_path),
    ):
        if path not in package.__path__:
            package.__path__.append(path)


def assert_molmoact2_and_xlerobot_sources() -> None:
    """Verify MolmoAct2 and XLerobot are loaded from their intended vendors."""
    assert_allenai_lerobot_imported()
    from lerobot.policies.molmoact2 import MolmoAct2Policy
    from lerobot.robots.xlerobot import XLerobot

    molmo_file = pathlib.Path(sys.modules[MolmoAct2Policy.__module__].__file__).resolve()
    xlerobot_file = pathlib.Path(sys.modules[XLerobot.__module__].__file__).resolve()
    allenai_root = ALLENAI_LEROBOT_SRC.resolve()
    xlerobot_root = XLEROBOT_LEROBOT_SRC.resolve()
    if allenai_root not in molmo_file.parents:
        raise RuntimeError(
            f"MolmoAct2 policy loaded from {molmo_file}, expected under {allenai_root}."
        )
    if xlerobot_root not in xlerobot_file.parents:
        raise RuntimeError(
            f"XLerobot driver loaded from {xlerobot_file}, expected under {xlerobot_root}."
        )
