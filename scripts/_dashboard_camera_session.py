"""Use dashboard CameraStream singletons for inference (same as VR teleop recording).

The dashboard keeps long-lived V4L2 capture threads in a registry; VR recording reads
`last_rgb` without tearing down capture during motor moves. Inference uses the same
path so homing does not drop cameras like lerobot's OpenCVCamera connect/disconnect.
"""
from __future__ import annotations

import logging
import pathlib
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CAMERA_ROLES = ("head", "left_wrist", "right_wrist")

log = logging.getLogger(__name__)

# Stagger opens to reduce USB hub inrush (right_wrist on 4.1 is often the first to drop).
_ACQUIRE_STAGGER_S = 0.35
_GRAB_RETRIES = 4
_RECOVER_FRAME_TIMEOUT_S = 4.0


def _dashboard_cameras_module() -> Any:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from openpibot.server.runtime import cameras as dashboard_cameras

    return dashboard_cameras


def _frame_ready(stream: Any) -> bool:
    return stream.last_rgb is not None or stream.last_jpeg is not None


class DashboardCameraSession:
    """Hold acquire() on role camera streams for the whole inference run."""

    def __init__(self, *, warmup_s: float = 3.0) -> None:
        self._warmup_s = warmup_s
        self._streams: list[tuple[str, Any]] = []
        self._wc: Any | None = None
        self._last_good: dict[str, np.ndarray] = {}
        self._stale_warned: set[str] = set()

    @property
    def streams(self) -> dict[str, Any]:
        """Role name → dashboard CameraStream (for preview; do not release)."""
        return {role: stream for role, stream in self._streams}

    def _replace_stream(self, role: str) -> Any:
        assert self._wc is not None
        for i, (r, old) in enumerate(self._streams):
            if r == role:
                try:
                    old.release()
                except Exception:
                    pass
                break
        stream = self._wc.restart_stream(role)
        if stream is None:
            raise RuntimeError(f"failed to restart camera stream for {role}")
        stream.encode_jpeg = False
        stream.acquire()
        for i, (r, _s) in enumerate(self._streams):
            if r == role:
                self._streams[i] = (role, stream)
                break
        return stream

    def _wait_first_frame(self, role: str, stream: Any) -> None:
        deadline = time.monotonic() + self._warmup_s
        while time.monotonic() < deadline:
            if _frame_ready(stream):
                return
            if stream.last_error and (
                stream.thread is None or not stream.thread.is_alive()
            ):
                stream = self._replace_stream(role)
            time.sleep(0.05)
        raise RuntimeError(
            f"camera {role} produced no frames within {self._warmup_s:.0f}s "
            f"(path={stream.spec.path})"
        )

    def start(self) -> None:
        if self._streams:
            return
        self._wc = _dashboard_cameras_module()
        missing: list[str] = []
        for role in _CAMERA_ROLES:
            if self._wc.find_camera(role) is None:
                missing.append(role)
        if missing:
            raise RuntimeError(
                "missing camera role(s) in config/xlerobot.yaml: "
                f"{missing}. Assign them on the dashboard Cameras page."
            )

        print(
            "Tip: stop the dashboard backend before inference — a second process "
            "opening the same /dev/v4l nodes often drops right_wrist (USB 4.1)."
        )

        for role in _CAMERA_ROLES:
            stream = self._wc.get_stream(role)
            if stream is None:
                raise RuntimeError(f"failed to open camera stream for {role}")
            stream.encode_jpeg = False
            stream.acquire()
            self._streams.append((role, stream))
            if role != _CAMERA_ROLES[-1]:
                time.sleep(_ACQUIRE_STAGGER_S)

        for role, stream in self._streams:
            self._wait_first_frame(role, stream)
            rgb = stream.last_rgb
            if rgb is not None:
                self._last_good[role] = np.asarray(rgb)

        paths = ", ".join(f"{r}={s.spec.path}" for r, s in self._streams)
        print(
            f"Dashboard cameras ready ({len(self._streams)} streams, same path as VR teleop).\n"
            f"  {paths}"
        )

    def stop(self) -> None:
        for _role, stream in reversed(self._streams):
            try:
                stream.release()
            except Exception:
                pass
        self._streams.clear()
        self._last_good.clear()
        self._stale_warned.clear()

    def _read_role(self, role: str, stream: Any) -> np.ndarray | None:
        with stream.lock:
            img = stream.last_rgb
        if img is not None:
            return np.asarray(img)
        return stream.get_rgb(timeout=0.4)

    def _recover_role(self, role: str) -> Any:
        log.warning("restarting camera stream for %s", role)
        return self._replace_stream(role)

    def grab(self) -> dict[str, np.ndarray]:
        """Latest RGB frame per role; recover USB drops; stale last-good as last resort."""
        out: dict[str, np.ndarray] = {}
        for role, stream in self._streams:
            img: np.ndarray | None = None
            for attempt in range(_GRAB_RETRIES):
                img = self._read_role(role, stream)
                if img is not None:
                    break
                alive = stream.thread is not None and stream.thread.is_alive()
                err = stream.last_error or "no frame"
                if not alive or attempt >= 1:
                    stream = self._recover_role(role)
                    deadline = time.monotonic() + _RECOVER_FRAME_TIMEOUT_S
                    while time.monotonic() < deadline:
                        img = self._read_role(role, stream)
                        if img is not None:
                            break
                        time.sleep(0.05)
                    if img is not None:
                        break
                if attempt == _GRAB_RETRIES - 1:
                    if role in self._last_good:
                        if role not in self._stale_warned:
                            self._stale_warned.add(role)
                            log.warning(
                                "camera %s still unavailable (%s); using last good frame "
                                "(policy quality will suffer until USB recovers)",
                                role,
                                err,
                            )
                        img = self._last_good[role].copy()
                        break
                    raise RuntimeError(f"camera {role}: {err}")
            assert img is not None
            self._last_good[role] = np.asarray(img)
            out[role] = img
        return out

    def __enter__(self) -> DashboardCameraSession:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def attach_dashboard_cameras_to_robot(robot: Any, *, warmup_s: float = 3.0) -> DashboardCameraSession:
    """Patch robot.get_observation to merge dashboard camera frames with motor state."""
    session = DashboardCameraSession(warmup_s=warmup_s)
    session.start()
    orig_get_observation = robot.get_observation

    def get_observation(*, include_cameras: bool = True) -> dict[str, Any]:
        obs = orig_get_observation(include_cameras=False)
        if include_cameras:
            obs.update(session.grab())
        return obs

    robot.get_observation = get_observation  # type: ignore[method-assign]
    robot._dashboard_camera_session = session  # type: ignore[attr-defined]
    return session


def detach_dashboard_cameras_from_robot(robot: Any) -> None:
    session = getattr(robot, "_dashboard_camera_session", None)
    if session is not None:
        session.stop()
        del robot._dashboard_camera_session
