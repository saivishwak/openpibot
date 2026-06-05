"""Build an XLerobotConfig from config/xlerobot.yaml.

Shared between the CLI runners (scripts/) and the dashboard (openpibot/server/runtime/motors.py).
Passes every override explicitly to the XLerobotConfig constructor — setting class-level
attributes does NOT change dataclass defaults, since `@dataclass` captures them into
the generated `__init__` signature at decoration time.
"""
from __future__ import annotations

import logging
import pathlib
import json
from typing import Any

import yaml

log = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"


def load_yaml() -> dict:
    if not CONFIG_YAML.is_file():
        return {}
    return yaml.safe_load(CONFIG_YAML.read_text()) or {}


def build_cameras(cfg: dict) -> dict[str, Any]:
    """Convert config/xlerobot.yaml's cameras section into lerobot CameraConfig objects."""
    from lerobot.cameras.configs import ColorMode, Cv2Backends, Cv2Rotation
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cams: dict[str, Any] = {}
    for name, c in (cfg.get("cameras") or {}).items():
        if c.get("type") != "opencv" or not c.get("path"):
            continue
        cams[name] = OpenCVCameraConfig(
            index_or_path=c["path"],
            fps=int(c.get("fps", 30)),
            width=int(c.get("width", 640)),
            height=int(c.get("height", 480)),
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION,
            fourcc=c.get("fourcc", "MJPG"),
            backend=Cv2Backends.V4L2,
        )
    return cams


def _refresh_motor_registry(bus: Any) -> None:
    """Rebuild id maps after pruning motors; clear cached `ids` / `models`."""
    bus._id_to_model_dict = {m.id: m.model for m in bus.motors.values()}
    bus._id_to_name_dict = {m.id: name for name, m in bus.motors.items()}
    for key in ("ids", "models", "_has_different_ctrl_tables"):
        bus.__dict__.pop(key, None)


def patch_motors_bus_lenient() -> None:
    """Make `MotorsBus._assert_motors_exist` PRUNE missing motors instead of raising.

    Reason: older reference scripts instantiated the full XLerobot driver,
    whose motor list includes 3 lekiwi base wheels (left bus) + 2 head motors (right bus).
    Bimanual SO101 setups have only the 6 arm motors per side, so the connect-time presence
    check fails. *But* both scripts only ever read/write arm motors at runtime — the base/head
    declarations are ceremonial. We drop the absent motors from the bus's registry so connect
    succeeds, then runtime calls that only touch arm motors work unchanged.

    Loud warnings are printed for every motor dropped, so the user knows what's not actually
    on the bus.
    """
    from lerobot.motors.motors_bus import MotorsBus

    if getattr(MotorsBus, "_xlerobot_lenient_patch", False):
        return  # idempotent

    original = MotorsBus._assert_motors_exist

    def lenient(self) -> None:  # type: ignore[no-untyped-def]
        # MotorsBus.ids is a @cached_property; must refresh after pruning motors.
        for _ in range(len(self.motors) + 1):
            try:
                original(self)
                return
            except RuntimeError as exc:
                msg = str(exc)
                if "Missing motor IDs" not in msg:
                    raise  # wrong model, protocol error, etc.

            found_ids: set[int] = set()
            for id_ in self.ids:
                if self.ping(id_) is not None:
                    found_ids.add(id_)
            to_drop = [
                name for name, motor in list(self.motors.items())
                if motor.id not in found_ids
            ]
            if not to_drop:
                raise

            log.warning(
                "[lenient-motors] Pruning %d absent motor(s) from %s on port %s: %s",
                len(to_drop),
                type(self).__name__,
                self.port,
                to_drop,
            )
            print(
                f"[lenient-motors] {type(self).__name__} on {self.port}: "
                f"dropping {to_drop} (not detected on bus)"
            )
            for name in to_drop:
                del self.motors[name]
            _refresh_motor_registry(self)

        raise RuntimeError(
            f"{type(self).__name__} motor check failed on port {self.port!r} "
            "after lenient pruning"
        )

    MotorsBus._assert_motors_exist = lenient  # type: ignore[assignment]
    MotorsBus._xlerobot_lenient_patch = True  # type: ignore[attr-defined]


def patch_xlerobot_motors_only_connected() -> None:
    """Treat robot as connected when motor buses are up, even if a camera dropped.

    Upstream XLerobot.is_connected also requires every OpenCVCamera.is_opened(). A single
    USB glitch on right_wrist then blocks send_action() despite healthy arms.
    """
    import lerobot.robots.xlerobot.xlerobot as xr

    if getattr(xr.XLerobot, "_xlerobot_motors_only_connected", False):
        return

    def _motors_only_connected(self: Any) -> bool:
        return bool(self.bus_left_base.is_connected and self.bus_right_head.is_connected)

    xr.XLerobot.is_connected = property(_motors_only_connected)  # type: ignore[method-assign]
    xr.XLerobot._xlerobot_motors_only_connected = True  # type: ignore[attr-defined]


def make_config(robot_id: str = "xlerobot", *, use_cameras: bool = True) -> Any:
    """Build an XLerobotConfig with all overrides from config/xlerobot.yaml applied.

    Set use_cameras=False when frames come from openpibot.server.runtime.cameras (inference/VR).
    """
    from lerobot.robots.xlerobot import XLerobotConfig

    cfg = load_yaml()
    r = cfg.get("robot") or {}
    cams = build_cameras(cfg)
    calib_dir = REPO_ROOT / "config" / "calibration" / "xlerobot"
    calib_dir.mkdir(parents=True, exist_ok=True)

    # Build XLerobot-prefixed calibration from per-arm SOFollower calibrations.
    # This lets XLerobot connect(calibrate=False) use the same calibration source as the
    # rest of this repo (config/calibration/so_follower/{left,right}_follower_arm.json).
    so_calib_dir = REPO_ROOT / "config" / "calibration" / "so_follower"
    left_id = str(r.get("left_arm_id", "left_follower_arm"))
    right_id = str(r.get("right_arm_id", "right_follower_arm"))
    left_path = so_calib_dir / f"{left_id}.json"
    right_path = so_calib_dir / f"{right_id}.json"
    merged: dict[str, Any] = {}
    try:
        if left_path.is_file():
            left = json.loads(left_path.read_text())
            merged.update({f"left_arm_{k}": v for k, v in left.items()})
        if right_path.is_file():
            right = json.loads(right_path.read_text())
            merged.update({f"right_arm_{k}": v for k, v in right.items()})
        if merged:
            merged_path = calib_dir / f"{robot_id}.json"
            merged_path.write_text(json.dumps(merged, indent=4))
    except Exception as e:
        log.warning("failed to build xlerobot calibration JSON: %s", e)

    kwargs: dict[str, Any] = {"id": robot_id}
    kwargs["calibration_dir"] = calib_dir
    if r.get("port_left_base"):
        kwargs["port_left_base"] = r["port_left_base"]
    if r.get("port_right_head"):
        kwargs["port_right_head"] = r["port_right_head"]
    if "max_relative_target" in r:
        kwargs["max_relative_target"] = r["max_relative_target"]
    kwargs["use_degrees"] = r.get("use_degrees", True)
    if use_cameras and cams:
        kwargs["cameras"] = cams
    else:
        kwargs["cameras"] = {}

    return XLerobotConfig(**kwargs)
