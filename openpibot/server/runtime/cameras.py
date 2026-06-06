"""Camera enumeration and MJPEG streaming."""
from __future__ import annotations

import fcntl
import logging
import os
import pathlib
import struct
import threading
import time
from dataclasses import dataclass
from collections.abc import Iterable

import cv2
import numpy as np
import yaml

from openpibot.server.config import REPO_ROOT

CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"

log = logging.getLogger(__name__)

# After this many failed reads, release V4L and wait for the device node to reappear.
_READ_FAILS_BEFORE_RECOVER = 8
_DEVICE_WAIT_S = 15.0

# V4L2 — see linux/videodev2.h
_VIDIOC_QUERYCAP = 0x80685600                     # _IOR('V', 0, struct v4l2_capability)
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_DEVICE_CAPS = 0x80000000
# v4l2_capability layout: 16 driver + 32 card + 32 bus_info + 4 version + 4 caps + 4 device_caps + 12 reserved
_CAP_OFFSET = 16 + 32 + 32 + 4
_DEV_CAP_OFFSET = _CAP_OFFSET + 4
_STRUCT_SIZE = 16 + 32 + 32 + 4 + 4 + 4 + 12  # 104

_ROLES = ("head", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class CameraSpec:
    name: str        # logical name from config (e.g. "head"), or "raw:videoN" for unassigned
    path: str        # canonical device path (or by-path symlink — kept as configured)
    width: int
    height: int
    fps: int
    fourcc: str
    role: str | None = None      # one of _ROLES if assigned, else None
    by_path: str | None = None   # /dev/v4l/by-path/... if known
    card: str = ""               # human-readable card name from V4L2_QUERYCAP
    available: bool = True       # configured path/device node exists right now
    capture: bool | None = None  # V4L2 capture-capable if probed; None if unknown


def _query_v4l2_cap(path: str) -> tuple[bool, str] | None:
    """Returns (is_capture, card_name) or None if path can't be probed."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        buf = bytearray(_STRUCT_SIZE)
        try:
            fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
        except OSError:
            return None
        card = bytes(buf[16:16 + 32]).split(b"\x00", 1)[0].decode(errors="replace")
        caps = struct.unpack_from("I", buf, _CAP_OFFSET)[0]
        # If V4L2_CAP_DEVICE_CAPS set, prefer device_caps (per-node) over the union caps.
        if caps & _V4L2_CAP_DEVICE_CAPS:
            caps = struct.unpack_from("I", buf, _DEV_CAP_OFFSET)[0]
        return bool(caps & _V4L2_CAP_VIDEO_CAPTURE), card
    finally:
        os.close(fd)


def _canonical_video_node(path: str) -> str:
    """Resolve a /dev/v4l/by-path/... symlink down to /dev/videoN."""
    return str(pathlib.Path(path).resolve())


def _opencv_capture_path(path: str) -> str:
    """Path passed to OpenCV.

    Keep config on stable by-path/by-id strings, but OpenCV's V4L2 backend emits
    warnings and may fail when opened by symlink name. The resolved /dev/videoN
    node is the correct capture handle while the stable mapping remains in YAML.
    """
    try:
        resolved = _canonical_video_node(path)
    except Exception:
        return path
    return resolved if resolved.startswith("/dev/video") else path


def _by_path_for_node(node: str) -> str | None:
    """Find the first /dev/v4l/by-path/* symlink pointing to /dev/videoN. Prefer non-usbv2."""
    by_path_dir = pathlib.Path("/dev/v4l/by-path")
    if not by_path_dir.is_dir():
        return None
    target = pathlib.Path(node).resolve()
    matches = [p for p in by_path_dir.iterdir() if p.resolve() == target]
    if not matches:
        return None
    matches.sort(key=lambda p: ("usbv2" in p.name, p.name))
    return str(matches[0])


def _all_symlinks_for_node(node: str) -> list[str]:
    """Stable /dev/v4l symlinks pointing at a video node, by-path first."""
    target = pathlib.Path(node).resolve()
    out: list[str] = []
    for directory in (pathlib.Path("/dev/v4l/by-path"), pathlib.Path("/dev/v4l/by-id")):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            try:
                if path.resolve() == target:
                    out.append(str(path))
            except OSError:
                continue
    out.sort(key=lambda p: ("/by-path/" not in p, "usbv2" in p, p))
    deduped: list[str] = []
    for path in out:
        if path not in deduped:
            deduped.append(path)
    return deduped


def _video_device_candidates() -> list[str]:
    """All visible V4L video device paths useful for camera role assignment.

    Includes stable by-path/by-id symlinks and raw /dev/videoN nodes. Capability
    probing can fail on some hosts while a device is still assignable, so probing
    happens later and does not hide the candidate.
    """
    paths: list[str] = []
    for directory in (pathlib.Path("/dev/v4l/by-path"), pathlib.Path("/dev/v4l/by-id")):
        if directory.is_dir():
            paths.extend(str(path) for path in sorted(directory.iterdir()))
    paths.extend(str(path) for path in sorted(
        pathlib.Path("/dev").glob("video*"),
        key=lambda p: int(p.name[5:]) if p.name[5:].isdigit() else 999999,
    ))
    out: list[str] = []
    for path in paths:
        if path not in out:
            out.append(path)
    return out


def _candidate_name(path: str, canonical_node: str, index: int) -> str:
    if canonical_node.startswith("/dev/video"):
        return f"raw:{pathlib.Path(canonical_node).name}"
    leaf = pathlib.Path(path).name or f"video{index}"
    return f"raw:{leaf}"


def _read_config() -> dict:
    if not CONFIG_YAML.is_file():
        return {}
    try:
        return yaml.safe_load(CONFIG_YAML.read_text()) or {}
    except Exception:
        return {}


def _write_config(cfg: dict) -> None:
    CONFIG_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False))


def enumerate_cameras() -> list[CameraSpec]:
    """Configured cameras first, then every visible unassigned V4L video device."""
    cfg = _read_config()
    cams_cfg = (cfg.get("cameras") or {})

    # Build a set of paths the config already claims, plus the canonical video node behind each.
    configured_paths: set[str] = set()
    configured_nodes: set[str] = set()
    out: list[CameraSpec] = []
    for role in _ROLES + tuple(k for k in cams_cfg if k not in _ROLES):
        c = cams_cfg.get(role)
        if not c or c.get("type") != "opencv" or not c.get("path"):
            continue
        path = c["path"]
        configured_paths.add(path)
        available = pathlib.Path(path).exists()
        canonical = ""
        try:
            canonical = _canonical_video_node(path)
            if available:
                configured_nodes.add(canonical)
        except Exception:
            pass
        probed = _query_v4l2_cap(path) if available else None
        capture, card = probed if probed is not None else (None, "")
        out.append(CameraSpec(
            name=role,
            path=path,
            width=int(c.get("width", 640)),
            height=int(c.get("height", 480)),
            fps=int(c.get("fps", 30)),
            fourcc=c.get("fourcc", "MJPG"),
            role=role if role in _ROLES else None,
            by_path=path if "/by-path/" in path or "/by-id/" in path else (_by_path_for_node(path) or path),
            card=card,
            available=available,
            capture=capture,
        ))

    # Discover all visible video device candidates the config doesn't already use.
    seen_nodes: set[str] = set(configured_nodes)
    seen_paths: set[str] = set(configured_paths)
    raw_index = 0
    for candidate in _video_device_candidates():
        if candidate in seen_paths:
            continue
        try:
            node = _canonical_video_node(candidate)
        except Exception:
            node = candidate
        if node in seen_nodes:
            continue
        available = pathlib.Path(candidate).exists()
        probed = _query_v4l2_cap(candidate) if available else None
        capture, card = probed if probed is not None else (None, "")
        seen_nodes.add(node)
        seen_paths.add(candidate)
        symlinks = _all_symlinks_for_node(node) if node != candidate else []
        assign_path = symlinks[0] if symlinks else candidate
        out.append(CameraSpec(
            name=_candidate_name(candidate, node, raw_index),
            path=assign_path,
            width=640, height=480, fps=30, fourcc="MJPG",
            role=None,
            by_path=assign_path,
            card=card,
            available=available,
            capture=capture,
        ))
        raw_index += 1
    return out


def find_camera(cam_id: str) -> CameraSpec | None:
    for c in enumerate_cameras():
        if c.name == cam_id:
            return c
    return None


def assign_role(by_path: str, role: str | None) -> dict:
    """Set or clear a camera role in config/xlerobot.yaml. Returns the updated config."""
    if role is not None and role not in _ROLES:
        raise ValueError(f"role must be one of {_ROLES} or None, got {role!r}")

    cfg = _read_config()
    cfg.setdefault("cameras", {})
    cams = cfg["cameras"]

    # If another role currently owns this by_path, clear it first.
    for r in list(cams.keys()):
        existing = cams.get(r) or {}
        if existing.get("path") == by_path and r != role:
            del cams[r]

    if role is None:
        # Just removed (above); nothing else to do.
        _write_config(cfg)
        return cfg

    existing = cams.get(role) or {}
    cams[role] = {
        "type": "opencv",
        "path": by_path,
        "fps": existing.get("fps", 30),
        "width": existing.get("width", 640),
        "height": existing.get("height", 480),
        "fourcc": existing.get("fourcc", "MJPG"),
    }
    _write_config(cfg)
    return cfg


class CameraStream:
    """Captures from a single device. Reused across HTTP clients via a singleton registry."""

    def __init__(self, spec: CameraSpec):
        self.spec = spec
        self.cap: cv2.VideoCapture | None = None
        self.last_jpeg: bytes | None = None
        self.last_rgb: Any | None = None  # HWC uint8 RGB — policy path (no JPEG round-trip)
        self.last_error: str | None = None
        self.lock = threading.Lock()
        self.subscribers = 0
        self.thread: threading.Thread | None = None
        self.stop_evt = threading.Event()
        # Inference only needs RGB; skipping JPEG encode reduces CPU/USB load.
        self.encode_jpeg = True

    def _close_cap(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _wait_for_device(self, timeout: float = _DEVICE_WAIT_S) -> bool:
        path = pathlib.Path(self.spec.path)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.stop_evt.is_set():
                return False
            if path.exists():
                return True
            time.sleep(0.15)
        return False

    def _open(self) -> bool:
        self._close_cap()
        capture_path = _opencv_capture_path(self.spec.path)
        cap = cv2.VideoCapture(capture_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(capture_path)
        if not cap.isOpened():
            self.last_error = f"VideoCapture failed to open {self.spec.path} ({capture_path})"
            return False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.spec.fourcc:
            fcc = cv2.VideoWriter_fourcc(*self.spec.fourcc)
            cap.set(cv2.CAP_PROP_FOURCC, fcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
        cap.set(cv2.CAP_PROP_FPS, self.spec.fps)
        self.cap = cap
        self.last_error = None
        return True

    def _recover_capture(self) -> None:
        """Release V4L handle and wait for the sysfs node (USB replug / errno 19)."""
        log.warning(
            "camera %s: recovering capture (%s)",
            self.spec.name,
            self.spec.path,
        )
        self._close_cap()
        if not self._wait_for_device():
            self.last_error = f"device missing: {self.spec.path}"
            return
        if not self._open():
            self.last_error = f"reopen failed: {self.spec.path}"

    def _loop(self) -> None:
        period = 1.0 / max(self.spec.fps, 1)
        while not self.stop_evt.is_set():
            if not self._open():
                if not self._wait_for_device(timeout=5.0):
                    time.sleep(0.5)
                continue
            consecutive_failures = 0
            while not self.stop_evt.is_set():
                t0 = time.monotonic()
                ok, frame = self.cap.read()  # type: ignore[union-attr]
                if not ok:
                    consecutive_failures += 1
                    self.last_error = "read() returned False"
                    if consecutive_failures >= _READ_FAILS_BEFORE_RECOVER:
                        self._recover_capture()
                        break
                    time.sleep(0.05)
                    continue
                consecutive_failures = 0
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.last_rgb = rgb
                    if self.encode_jpeg:
                        ok2, buf = cv2.imencode(
                            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                        )
                        if ok2:
                            self.last_jpeg = bytes(buf)
                    self.last_error = None
                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)
        self._close_cap()

    def acquire(self) -> None:
        with self.lock:
            self.subscribers += 1
            if self.thread is None or not self.thread.is_alive():
                self.stop_evt.clear()
                self.thread = threading.Thread(target=self._loop, daemon=True,
                                               name=f"cam-{self.spec.name}")
                self.thread.start()

    def release(self) -> None:
        with self.lock:
            self.subscribers = max(0, self.subscribers - 1)
            if self.subscribers == 0:
                self.stop_evt.set()

    def get_rgb(self, timeout: float = 2.0) -> np.ndarray | None:
        """Latest RGB frame (640×480 HWC uint8) for LeRobot / policy — not JPEG-compressed."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.last_rgb is not None:
                    return self.last_rgb
                if self.last_error and (self.thread is None or not self.thread.is_alive()):
                    return None
            time.sleep(0.05)
        return None

    def get_jpeg(self, timeout: float = 2.0) -> bytes | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.last_jpeg is not None:
                    return self.last_jpeg
                if self.last_error and (self.thread is None or not self.thread.is_alive()):
                    return None
            time.sleep(0.05)
        return None

    def snapshot(self) -> tuple[bytes | None, str | None]:
        self.acquire()
        try:
            jpeg = self.get_jpeg(timeout=3.0)
            return jpeg, self.last_error
        finally:
            self.release()


_REGISTRY: dict[str, CameraStream] = {}
_REG_LOCK = threading.Lock()
_SUSPENDED_ROLES: dict[str, str] = {}


def get_stream(cam_id: str) -> CameraStream | None:
    spec = find_camera(cam_id)
    if spec is None:
        return None
    if spec.role and spec.role in _SUSPENDED_ROLES:
        return None
    with _REG_LOCK:
        cached = _REGISTRY.get(cam_id)
        # If the underlying spec changed (e.g. user reassigned), drop the cached stream.
        if cached is not None and cached.spec != spec:
            cached.stop_evt.set()
            del _REGISTRY[cam_id]
            cached = None
        if cached is None:
            _REGISTRY[cam_id] = CameraStream(spec)
        return _REGISTRY[cam_id]


def reset_streams() -> None:
    """Tear down all running captures (called after a config change)."""
    with _REG_LOCK:
        for s in _REGISTRY.values():
            s.stop_evt.set()
        _REGISTRY.clear()
        _SUSPENDED_ROLES.clear()


def suspend_capture_roles(roles: Iterable[str], *, reason: str = "") -> list[str]:
    """Stop and block OpenCV/MJPEG capture for roles owned by another transport.

    The Quest GStreamer bridge opens the same V4L2 devices directly. Browser
    dashboard previews are long-lived MJPEG clients, so merely checking for
    contention makes Quest video fragile. Suspending the role closes the OpenCV
    handle and prevents immediate reacquire until the bridge stops.
    """
    wanted = {str(role) for role in roles if str(role) in _ROLES}
    stopped: list[CameraStream] = []
    with _REG_LOCK:
        for role in wanted:
            _SUSPENDED_ROLES[role] = reason or "capture suspended"
        for cam_id, stream in list(_REGISTRY.items()):
            if stream.spec.role in wanted:
                stream.subscribers = 0
                stream.stop_evt.set()
                stopped.append(stream)
                del _REGISTRY[cam_id]

    for stream in stopped:
        if stream.thread is not None and stream.thread.is_alive():
            stream.thread.join(timeout=2.0)
        stream._close_cap()
    return sorted(wanted)


def resume_capture_roles(roles: Iterable[str] | None = None) -> list[str]:
    """Allow OpenCV/MJPEG capture for the given roles again."""
    with _REG_LOCK:
        if roles is None:
            resumed = sorted(_SUSPENDED_ROLES)
            _SUSPENDED_ROLES.clear()
            return resumed
        wanted = {str(role) for role in roles if str(role) in _ROLES}
        resumed = sorted(role for role in wanted if role in _SUSPENDED_ROLES)
        for role in resumed:
            _SUSPENDED_ROLES.pop(role, None)
        return resumed


def suspended_capture_roles() -> dict[str, str]:
    with _REG_LOCK:
        return dict(_SUSPENDED_ROLES)


def capture_suspension_reason(cam_id: str) -> str:
    spec = find_camera(cam_id)
    if spec is None or not spec.role:
        return ""
    return _SUSPENDED_ROLES.get(spec.role, "")


def active_capture_roles() -> list[str]:
    """Camera roles with a live OpenCV stream handle.

    GStreamer opens the same V4L2 devices directly for Quest video. Reporting
    active OpenCV consumers lets the Quest bridge avoid racing the MJPEG/policy
    capture path for exclusive USB camera handles.
    """
    with _REG_LOCK:
        out: list[str] = []
        for stream in _REGISTRY.values():
            if stream.spec.role and stream.subscribers > 0:
                out.append(stream.spec.role)
        return sorted(set(out))


def restart_stream(cam_id: str) -> CameraStream | None:
    """Stop and recreate one camera stream (e.g. after USB drop during inference)."""
    with _REG_LOCK:
        old = _REGISTRY.pop(cam_id, None)
    if old is not None:
        old.stop_evt.set()
        if old.thread is not None and old.thread.is_alive():
            old.thread.join(timeout=3.0)
        old._close_cap()
    return get_stream(cam_id)


def mjpeg_iter(cam_id: str, max_fps: int = 15):
    boundary = b"--frame"
    stream = get_stream(cam_id)
    if stream is None:
        reason = capture_suspension_reason(cam_id) or "camera unavailable"
        yield boundary + b"\r\nContent-Type: text/plain\r\n\r\n" + reason.encode() + b"\r\n"
        return
    stream.acquire()
    period = 1.0 / max_fps
    try:
        while True:
            if stream.stop_evt.is_set():
                yield boundary + b"\r\nContent-Type: text/plain\r\n\r\ncamera stream suspended\r\n"
                break
            t0 = time.monotonic()
            jpeg = stream.get_jpeg(timeout=1.5)
            if jpeg is None:
                err = (stream.last_error or "no frame").encode()
                yield boundary + b"\r\nContent-Type: text/plain\r\n\r\n" + err + b"\r\n"
                break
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\n"
                   + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                   + jpeg + b"\r\n")
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        stream.release()
