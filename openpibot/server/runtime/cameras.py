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
    """Configured cameras first (in role order), then any detected unassigned capture nodes."""
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
        try:
            configured_nodes.add(_canonical_video_node(path))
        except Exception:
            pass
        out.append(CameraSpec(
            name=role,
            path=path,
            width=int(c.get("width", 640)),
            height=int(c.get("height", 480)),
            fps=int(c.get("fps", 30)),
            fourcc=c.get("fourcc", "MJPG"),
            role=role if role in _ROLES else None,
            by_path=path if "/by-path/" in path else _by_path_for_node(path),
            card="",
        ))

    # Discover raw capture-capable /dev/videoN nodes the config doesn't already use.
    raw_nodes = sorted(pathlib.Path("/dev").glob("video*"), key=lambda p: int(p.name[5:]))
    seen_nodes: set[str] = set(configured_nodes)
    for vid in raw_nodes:
        node = str(vid)
        if node in seen_nodes:
            continue
        probed = _query_v4l2_cap(node)
        if probed is None:
            continue
        is_capture, card = probed
        if not is_capture:
            continue
        seen_nodes.add(node)
        by_path = _by_path_for_node(node)
        out.append(CameraSpec(
            name=f"raw:{vid.name}",
            path=by_path or node,
            width=640, height=480, fps=30, fourcc="MJPG",
            role=None,
            by_path=by_path,
            card=card,
        ))
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
        cap = cv2.VideoCapture(self.spec.path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.spec.path)
        if not cap.isOpened():
            self.last_error = f"VideoCapture failed to open {self.spec.path}"
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


def get_stream(cam_id: str) -> CameraStream | None:
    spec = find_camera(cam_id)
    if spec is None:
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
    stream = get_stream(cam_id)
    if stream is None:
        return
    stream.acquire()
    period = 1.0 / max_fps
    boundary = b"--frame"
    try:
        while True:
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
