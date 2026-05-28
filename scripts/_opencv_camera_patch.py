"""Legacy lerobot OpenCVCamera monkeypatch (not used by infer_pi05_finetuned).

Inference uses webapp CameraStream via scripts/_webapp_camera_session.py instead.

Resilient OpenCVCamera behavior without editing the lerobot submodule.

Patches lerobot's OpenCVCamera at runtime (same pattern as patch_motors_bus_lenient):
  - Background read loop recovers from transient V4L/USB failures instead of exiting.
  - async_read reconnects on timeout (thread alive but no frames during USB reset).
  - XLerobot.get_camera_observation uses longer timeouts and per-camera retry.
  - pause/resume helpers stop cameras during motor homing (reduces USB contention).
"""
from __future__ import annotations

import logging
import platform
import time
from pathlib import Path
from typing import Any

import cv2

log = logging.getLogger(__name__)

_READ_FAILURES_BEFORE_RECOVER = 5
_RECOVER_DEVICE_WAIT_S = 12.0
_RECOVER_BACKOFF_S = 0.5
# LeRobot default async_read timeout is 200 ms — too short while a camera is recovering.
_INFER_ASYNC_READ_TIMEOUT_MS = 4000
_INFER_ASYNC_READ_ATTEMPTS = 4


def _opencv_backend() -> int:
    if platform.system() == "Linux":
        return int(cv2.CAP_V4L2)
    return int(cv2.CAP_ANY)


def _device_path(cam: Any) -> Path | None:
    path = cam.index_or_path
    if isinstance(path, Path):
        return path
    if isinstance(path, str) and path.startswith("/"):
        return Path(path)
    return None


def _wait_for_device(cam: Any, timeout_s: float = _RECOVER_DEVICE_WAIT_S) -> bool:
    dev = _device_path(cam)
    if dev is None:
        return True
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if dev.exists():
            return True
        time.sleep(0.2)
    return False


def _release_videocapture(cam: Any) -> None:
    if cam.videocapture is not None:
        try:
            cam.videocapture.release()
        except Exception:
            pass
        cam.videocapture = None


def _open_videocapture(cam: Any) -> None:
    cam.videocapture = cv2.VideoCapture(cam.index_or_path, _opencv_backend())
    if not cam.videocapture.isOpened():
        _release_videocapture(cam)
        raise ConnectionError(f"Failed to open {cam}")


def reopen_videocapture(cam: Any) -> None:
    """Release and reopen V4L after transient read failures."""
    cam._stop_read_thread()
    _release_videocapture(cam)
    if not _wait_for_device(cam):
        raise ConnectionError(f"{cam} device path not present: {cam.index_or_path}")
    time.sleep(_RECOVER_BACKOFF_S)
    _open_videocapture(cam)
    cam._configure_capture_settings()
    cam._start_read_thread()


def reconnect_camera(cam: Any, *, warmup: bool = True) -> None:
    """Re-open capture and the background read thread."""
    log.info("%s reconnecting capture", cam)
    cam._stop_read_thread()
    _release_videocapture(cam)
    with cam.frame_lock:
        cam.latest_frame = None
        cam.latest_timestamp = None
        cam.new_frame_event.clear()
    if not _wait_for_device(cam):
        raise ConnectionError(f"{cam} device path not present: {cam.index_or_path}")
    cv2.setNumThreads(1)
    time.sleep(_RECOVER_BACKOFF_S)
    _open_videocapture(cam)
    cam._configure_capture_settings()
    cam._start_read_thread()
    if warmup and cam.warmup_s > 0:
        read_camera_frame(
            cam,
            timeout_ms=max(float(cam.warmup_s) * 1000.0, _INFER_ASYNC_READ_TIMEOUT_MS),
            attempts=2,
        )


def read_camera_frame(
    cam: Any,
    *,
    name: str = "",
    timeout_ms: float = _INFER_ASYNC_READ_TIMEOUT_MS,
    attempts: int = _INFER_ASYNC_READ_ATTEMPTS,
) -> Any:
    """Read one frame with reconnect retries (for inference / observation)."""
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    orig_async_read = OpenCVCamera._xlerobot_orig_async_read  # type: ignore[attr-defined]

    label = name or str(cam)
    last_err: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            if cam.thread is None or not cam.thread.is_alive():
                reconnect_camera(cam, warmup=False)
            elif cam.videocapture is None or not cam.videocapture.isOpened():
                reconnect_camera(cam, warmup=False)
            return orig_async_read(cam, timeout_ms=timeout_ms)
        except TimeoutError as e:
            last_err = e
            log.warning(
                "%s timed out waiting for frame (%d/%d); reconnecting",
                label,
                attempt,
                attempts,
            )
            try:
                reconnect_camera(cam, warmup=False)
            except Exception as rec_err:
                log.warning("%s reconnect after timeout failed: %s", label, rec_err)
            if attempt < attempts:
                time.sleep(0.5)
        except RuntimeError as e:
            last_err = e
            if "read thread is not running" not in str(e):
                raise
            log.warning("%s read thread stopped (%d/%d); reconnecting", label, attempt, attempts)
            reconnect_camera(cam, warmup=False)
            if attempt < attempts:
                time.sleep(0.5)
    assert last_err is not None
    raise RuntimeError(f"camera {label} failed after {attempts} read attempts") from last_err


def warn_missing_camera_paths(cfg: dict[str, Any]) -> None:
    """Print a actionable warning when configured V4L paths are absent."""
    cams = cfg.get("cameras") or {}
    by_path = Path("/dev/v4l/by-path")
    available = sorted(p.name for p in by_path.iterdir()) if by_path.is_dir() else []
    missing: list[str] = []
    for role, c in cams.items():
        path = c.get("path")
        if path and not Path(path).exists():
            missing.append(f"  {role}: {path}")
    if not missing:
        return
    print("WARNING: configured camera device path(s) not found:")
    print("\n".join(missing))
    if available:
        print("Available /dev/v4l/by-path nodes:")
        for name in available:
            print(f"  {by_path / name}")
    print("Re-assign cameras in the webapp Cameras page or edit config/xlerobot.yaml.")


def patch_opencv_camera_resilient() -> None:
    """Monkeypatch lerobot OpenCVCamera for USB/V4L recovery (idempotent)."""
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.utils.errors import DeviceNotConnectedError

    if getattr(OpenCVCamera, "_xlerobot_resilient_patch", False):
        return

    orig_async_read = OpenCVCamera.async_read

    def resilient_read_loop(self: Any) -> None:
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not self.stop_event.is_set():
            if self.videocapture is None or not self.videocapture.isOpened():
                time.sleep(0.5)
                continue
            try:
                raw_frame = self._read_from_hardware()
                processed_frame = self._postprocess_image(raw_frame)
                capture_time = time.perf_counter()
                with self.frame_lock:
                    self.latest_frame = processed_frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0
            except DeviceNotConnectedError:
                break
            except Exception as e:
                failure_count += 1
                if failure_count <= 3:
                    log.warning("Error reading frame in background thread for %s: %s", self, e)
                elif failure_count == _READ_FAILURES_BEFORE_RECOVER:
                    log.warning(
                        "%s sustained read failures; releasing V4L handle "
                        "(main thread will reconnect on next read)",
                        self,
                    )
                if failure_count >= _READ_FAILURES_BEFORE_RECOVER:
                    # Release capture only — never join this thread from inside itself.
                    # read_camera_frame / reconnect_camera on the main thread reopens.
                    _release_videocapture(self)
                    with self.frame_lock:
                        self.latest_frame = None
                        self.latest_timestamp = None
                    failure_count = 0
                    time.sleep(1.0)
                else:
                    time.sleep(0.02)

    def resilient_async_read(self: Any, timeout_ms: float = 200) -> Any:
        try:
            if self.thread is None or not self.thread.is_alive():
                reconnect_camera(self, warmup=False)
            elif self.videocapture is None or not self.videocapture.isOpened():
                reconnect_camera(self, warmup=False)
            return orig_async_read(self, timeout_ms=timeout_ms)
        except TimeoutError:
            reconnect_camera(self, warmup=False)
            return orig_async_read(self, timeout_ms=max(timeout_ms, _INFER_ASYNC_READ_TIMEOUT_MS))

    OpenCVCamera._read_loop = resilient_read_loop  # type: ignore[method-assign]
    OpenCVCamera.async_read = resilient_async_read  # type: ignore[method-assign]
    OpenCVCamera.reconnect = reconnect_camera  # type: ignore[attr-defined]
    OpenCVCamera._xlerobot_orig_async_read = orig_async_read  # type: ignore[attr-defined]
    OpenCVCamera._xlerobot_resilient_patch = True  # type: ignore[attr-defined]


def patch_xlerobot_camera_observation() -> None:
    """Route XLerobot camera reads through read_camera_frame (idempotent)."""
    import lerobot.robots.xlerobot.xlerobot as xr

    if getattr(xr.XLerobot, "_xlerobot_camera_obs_patch", False):
        return

    def resilient_get_camera_observation(self: Any) -> dict[str, Any]:
        obs_dict: dict[str, Any] = {}
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = read_camera_frame(
                cam, name=cam_key, timeout_ms=_INFER_ASYNC_READ_TIMEOUT_MS
            )
        return obs_dict

    xr.XLerobot.get_camera_observation = resilient_get_camera_observation  # type: ignore[method-assign]
    xr.XLerobot._xlerobot_camera_obs_patch = True  # type: ignore[attr-defined]


def pause_robot_cameras(robot: Any) -> None:
    """Stop background camera reads and release V4L handles during homing."""
    for cam in robot.cameras.values():
        if cam.thread is not None and cam.thread.is_alive():
            cam._stop_read_thread()
        _release_videocapture(cam)


def resume_lerobot_cameras(robot: Any, *, label: str = "After homing") -> None:
    """Reconnect all LeRobot OpenCVCamera instances on the robot."""
    if not getattr(robot, "cameras", None):
        return
    ensure_robot_cameras_ready(robot, label=label)


def ensure_robot_cameras_ready(
    robot: Any,
    *,
    label: str = "",
    attempts: int = 4,
    warmup_timeout_ms: float = _INFER_ASYNC_READ_TIMEOUT_MS,
) -> None:
    """Verify every camera has a live read thread and a fresh frame."""
    prefix = f"{label}: " if label else ""
    for cam_name, cam in robot.cameras.items():
        last_err: Exception | None = None
        for attempt in range(1, max(1, attempts) + 1):
            try:
                read_camera_frame(
                    cam, name=cam_name, timeout_ms=warmup_timeout_ms, attempts=2
                )
                if attempt > 1:
                    print(f"{prefix}Camera {cam_name} OK (retry {attempt})")
                break
            except Exception as e:
                last_err = e
                print(
                    f"{prefix}Camera {cam_name} not ready ({attempt}/{attempts}): "
                    f"{e.__class__.__name__}: {e}"
                )
                try:
                    reconnect_camera(cam, warmup=False)
                except Exception:
                    pass
                if attempt < attempts:
                    time.sleep(1.0)
        else:
            assert last_err is not None
            raise RuntimeError(f"{prefix}camera {cam_name} failed to recover") from last_err
