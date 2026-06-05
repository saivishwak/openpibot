"""Quest video bridge descriptors for low-latency GStreamer/WebRTC streaming.

The camera capture owner remains `runtime.cameras`. This module only resolves
camera roles and builds explicit GStreamer launch descriptors that a Quest video
bridge process can use without changing dataset/inference camera semantics.
"""
from __future__ import annotations

import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import yaml

from openpibot.server.config import REPO_ROOT
from . import cameras

QUEST_VIDEO_ROLES = ("head", "left_wrist", "right_wrist")
QUEST_VIDEO_BASE_PORT = 5600
QUEST_VIDEO_BITRATE_KBPS = 2500
QUEST_VIDEO_BRIGHTNESS = 0.0
QUEST_VIDEO_CONTRAST = 1.0
QUEST_VIDEO_SATURATION = 1.0
QUEST_VIDEO_FLIP_METHOD = "none"
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"


@dataclass(frozen=True)
class QuestVideoStream:
    role: str
    camera_name: str
    device_path: str
    width: int
    height: int
    fps: int
    fourcc: str
    mount: str
    gst_launch: str
    udp_port: int
    receiver_pipeline: str
    active_gst_launch: str | None = None
    running: bool = False
    pid: int | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "camera_name": self.camera_name,
            "device_path": self.device_path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "fourcc": self.fourcc,
            "mount": self.mount,
            "gst_launch": self.gst_launch,
            "udp_port": self.udp_port,
            "receiver_pipeline": self.receiver_pipeline,
            "active_gst_launch": self.active_gst_launch,
            "running": self.running,
            "pid": self.pid,
            "last_error": self.last_error,
        }


def bridge_status() -> dict[str, Any]:
    return _MANAGER.status()


def start_bridge(quest_host: str, roles: list[str] | None = None) -> dict[str, Any]:
    return _MANAGER.start(quest_host=quest_host, roles=roles)


def stop_bridge() -> dict[str, Any]:
    return _MANAGER.stop()


def report_receive_health(payload: dict[str, Any] | None) -> dict[str, Any]:
    return _MANAGER.report_receive_health(payload or {})


def _video_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    try:
        if CONFIG_YAML.is_file():
            cfg = yaml.safe_load(CONFIG_YAML.read_text()) or {}
    except Exception:
        cfg = {}
    video = ((cfg.get("vr") or {}).get("quest_video") or {})
    roles = tuple(str(role) for role in (video.get("roles") or QUEST_VIDEO_ROLES))
    return {
        "roles": roles,
        "base_port": int(video.get("base_port", QUEST_VIDEO_BASE_PORT)),
        "bitrate_kbps": int(video.get("bitrate_kbps", QUEST_VIDEO_BITRATE_KBPS)),
        "brightness": _clamp_float(video.get("brightness", QUEST_VIDEO_BRIGHTNESS), QUEST_VIDEO_BRIGHTNESS, -1.0, 1.0),
        "contrast": _clamp_float(video.get("contrast", QUEST_VIDEO_CONTRAST), QUEST_VIDEO_CONTRAST, 0.0, 2.0),
        "saturation": _clamp_float(video.get("saturation", QUEST_VIDEO_SATURATION), QUEST_VIDEO_SATURATION, 0.0, 2.0),
        "flip_method": _flip_method(video.get("flip_method", QUEST_VIDEO_FLIP_METHOD)),
    }


def _discover_streams_with_runtime(
    *,
    running: dict[str, subprocess.Popen],
    errors: dict[str, str],
    quest_host: str | None,
) -> list[QuestVideoStream]:
    streams = discover_streams()
    out: list[QuestVideoStream] = []
    for stream in streams:
        proc = running.get(stream.role)
        is_running = proc is not None and proc.poll() is None
        out.append(
            QuestVideoStream(
                role=stream.role,
                camera_name=stream.camera_name,
                device_path=stream.device_path,
                width=stream.width,
                height=stream.height,
                fps=stream.fps,
                fourcc=stream.fourcc,
                mount=stream.mount,
                gst_launch=stream.gst_launch,
                udp_port=stream.udp_port,
                receiver_pipeline=stream.receiver_pipeline,
                active_gst_launch=(
                    _gst_launch_for_camera(stream, host=quest_host, port=stream.udp_port)
                    if is_running and quest_host is not None else None
                ),
                running=is_running,
                pid=proc.pid if is_running else None,
                last_error=errors.get(stream.role),
            )
        )
    return out


class QuestVideoBridgeManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._processes: dict[str, subprocess.Popen] = {}
        self._errors: dict[str, str] = {}
        self._quest_host: str | None = None
        self._started_at: float | None = None
        self._receive_health: dict[str, dict[str, Any]] = {}

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._reap_locked()
            streams = _discover_streams_with_runtime(
                running=self._processes,
                errors=self._errors,
                quest_host=self._quest_host,
            )
            config = _video_config()
            expected_roles = tuple(config["roles"])
            configured = {stream.role for stream in streams}
            missing = [role for role in expected_roles if role not in configured]
            gst_available = shutil.which("gst-launch-1.0") is not None
            running_roles = [role for role, proc in self._processes.items() if proc.poll() is None]
            return {
                "ready": not missing and gst_available and len(running_roles) == len(expected_roles),
                "transport": "gstreamer-rtp-h264",
                "gst_available": gst_available,
                "running": bool(running_roles),
                "quest_host": self._quest_host,
                "started_at": self._started_at,
                "roles": list(expected_roles),
                "base_port": config["base_port"],
                "bitrate_kbps": config["bitrate_kbps"],
                "brightness": config["brightness"],
                "contrast": config["contrast"],
                "saturation": config["saturation"],
                "flip_method": config["flip_method"],
                "running_roles": running_roles,
                "missing_roles": missing,
                "suspended_capture_roles": cameras.suspended_capture_roles(),
                "receive_health": dict(self._receive_health),
                "streams": [stream.to_dict() for stream in streams],
            }

    def start(self, *, quest_host: str, roles: list[str] | None = None) -> dict[str, Any]:
        host = str(quest_host or "").strip()
        if not host or any(ch.isspace() for ch in host):
            raise ValueError("quest_host must be a hostname or IP address")
        if shutil.which("gst-launch-1.0") is None:
            raise RuntimeError("gst-launch-1.0 is not installed")

        config = _video_config()
        configured_roles = tuple(config["roles"])
        requested = list(roles or configured_roles)
        invalid = [role for role in requested if role not in configured_roles]
        if invalid:
            raise ValueError(f"roles must be from {configured_roles}, got {invalid}")
        with self._lock:
            self._reap_locked()
            streams = {stream.role: stream for stream in discover_streams()}
            missing = [role for role in requested if role not in streams]
            if missing:
                raise RuntimeError(f"missing camera roles: {', '.join(missing)}")
            contended = sorted(set(requested) & set(cameras.active_capture_roles()))
            if contended:
                cameras.suspend_capture_roles(
                    contended,
                    reason="Quest RTP video is using this camera",
                )
            self._quest_host = host
            self._started_at = time.time()
            try:
                for role in requested:
                    proc = self._processes.get(role)
                    if proc is not None and proc.poll() is None:
                        continue
                    stream = streams[role]
                    command = _gst_launch_for_camera(
                        stream,
                        host=host,
                        port=stream.udp_port,
                        bitrate_kbps=config["bitrate_kbps"],
                    )
                    self._processes[role] = subprocess.Popen(
                        ["gst-launch-1.0", "-q", *shlex.split(command)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._errors.pop(role, None)
            except Exception:
                for proc in self._processes.values():
                    if proc.poll() is None:
                        proc.terminate()
                self._processes.clear()
                self._quest_host = None
                self._started_at = None
                cameras.resume_capture_roles(requested)
                raise
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            roles_to_resume = list(_video_config()["roles"])
            for proc in self._processes.values():
                if proc.poll() is None:
                    proc.terminate()
            deadline = time.monotonic() + 2.0
            for proc in self._processes.values():
                while proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                if proc.poll() is None:
                    proc.kill()
            self._processes.clear()
            self._quest_host = None
            self._started_at = None
            cameras.resume_capture_roles(roles_to_resume)
            return self.status()

    def report_receive_health(self, payload: dict[str, Any]) -> dict[str, Any]:
        role = str(payload.get("role") or "").strip()
        config = _video_config()
        if role not in config["roles"]:
            raise ValueError(f"role must be from {config['roles']}")
        with self._lock:
            self._receive_health[role] = {
                "role": role,
                "received_at": time.time(),
                "state": str(payload.get("state") or "unknown"),
                "fps": float(payload.get("fps") or 0.0),
                "latency_ms": float(payload.get("latency_ms") or 0.0),
                "frames": int(payload.get("frames") or 0),
                "error": str(payload.get("error") or ""),
            }
            return self.status()

    def _reap_locked(self) -> None:
        for role, proc in list(self._processes.items()):
            if proc.poll() is None:
                continue
            self._errors[role] = f"gst exited with code {proc.returncode}"
            del self._processes[role]


_MANAGER = QuestVideoBridgeManager()


def discover_streams() -> list[QuestVideoStream]:
    out: list[QuestVideoStream] = []
    config = _video_config()
    roles = tuple(config["roles"])
    base_port = int(config["base_port"])
    for spec in cameras.enumerate_cameras():
        if spec.role not in roles:
            continue
        udp_port = base_port + roles.index(spec.role)
        out.append(
            QuestVideoStream(
                role=spec.role,
                camera_name=spec.name,
                device_path=spec.path,
                width=spec.width,
                height=spec.height,
                fps=spec.fps,
                fourcc=spec.fourcc,
                mount=f"/quest/video/{spec.role}",
                gst_launch=_gst_launch_for_camera(spec, bitrate_kbps=config["bitrate_kbps"]),
                udp_port=udp_port,
                receiver_pipeline=_receiver_pipeline(udp_port),
            )
        )
    out.sort(key=lambda stream: roles.index(stream.role))
    return out


def _gst_launch_for_camera(
    spec: cameras.CameraSpec | QuestVideoStream,
    *,
    host: str | None = None,
    port: int | None = None,
    bitrate_kbps: int | None = None,
) -> str:
    caps_fourcc = "MJPG" if str(spec.fourcc).upper() in {"MJPG", "MJPEG"} else str(spec.fourcc).upper()
    caps = (
        f"video/x-raw,width={spec.width},height={spec.height},framerate={spec.fps}/1"
        if caps_fourcc not in {"MJPG", "MJPEG"}
        else f"image/jpeg,width={spec.width},height={spec.height},framerate={spec.fps}/1"
    )
    decode = "jpegdec ! " if caps_fourcc in {"MJPG", "MJPEG"} else ""
    device_path = spec.path if hasattr(spec, "path") else spec.device_path
    transform = _video_transform_pipeline()
    encoder = _h264_encoder_pipeline(int(bitrate_kbps or QUEST_VIDEO_BITRATE_KBPS))
    pipeline = (
        f"v4l2src device={device_path} do-timestamp=true ! "
        f"{caps} ! {decode}videoconvert ! {transform}"
        "queue leaky=downstream max-size-buffers=1 ! "
        f"{encoder} ! "
        "rtph264pay config-interval=1 pt=96"
    )
    if host is not None and port is not None:
        pipeline += f" ! udpsink host={host} port={port} sync=false async=false"
    return pipeline


def _video_transform_pipeline() -> str:
    config = _video_config()
    parts: list[str] = []
    flip_method = str(config["flip_method"])
    if flip_method != QUEST_VIDEO_FLIP_METHOD:
        parts.append(f"videoflip method={flip_method}")
    balance = _video_balance_pipeline(config)
    if balance:
        parts.append(balance)
    return " ! ".join(parts) + (" ! " if parts else "")


def _video_balance_pipeline(config: dict[str, Any] | None = None) -> str:
    config = config or _video_config()
    brightness = float(config["brightness"])
    contrast = float(config["contrast"])
    saturation = float(config["saturation"])
    if brightness == QUEST_VIDEO_BRIGHTNESS and contrast == QUEST_VIDEO_CONTRAST and saturation == QUEST_VIDEO_SATURATION:
        return ""
    return (
        f"videobalance brightness={brightness:.3f} "
        f"contrast={contrast:.3f} "
        f"saturation={saturation:.3f}"
    )


def _flip_method(value: Any) -> str:
    method = str(value or QUEST_VIDEO_FLIP_METHOD).strip().lower().replace("_", "-")
    allowed = {
        "none",
        "clockwise",
        "rotate-180",
        "counterclockwise",
        "horizontal-flip",
        "vertical-flip",
    }
    return method if method in allowed else QUEST_VIDEO_FLIP_METHOD


def _clamp_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    return max(low, min(high, out))


def _h264_encoder_pipeline(bitrate_kbps: int) -> str:
    bitrate_kbps = int(bitrate_kbps or QUEST_VIDEO_BITRATE_KBPS)
    if _gst_element_available("x264enc"):
        return (
            "x264enc tune=zerolatency speed-preset=ultrafast "
            f"key-int-max=30 bitrate={bitrate_kbps}"
        )
    if _gst_element_available("openh264enc"):
        return (
            "video/x-raw,format=I420 ! "
            f"openh264enc complexity=low rate-control=bitrate bitrate={bitrate_kbps * 1000} "
            "gop-size=30 enable-frame-skip=true"
        )
    if _gst_element_available("nvh264enc"):
        return f"nvh264enc bitrate={bitrate_kbps}"
    if _gst_element_available("nvcudah264enc"):
        return f"nvcudah264enc bitrate={bitrate_kbps}"
    raise RuntimeError(
        "no GStreamer H.264 encoder found; install x264enc "
        "(gstreamer1.0-plugins-ugly) or openh264enc (gstreamer1.0-plugins-bad)"
    )


@lru_cache(maxsize=None)
def _gst_element_available(element: str) -> bool:
    if shutil.which("gst-inspect-1.0") is None:
        return False
    result = subprocess.run(
        ["gst-inspect-1.0", element],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _receiver_pipeline(port: int) -> str:
    return (
        f"udpsrc port={port} caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96\" ! "
        "rtph264depay ! avdec_h264 ! videoconvert ! appsink sync=false"
    )
