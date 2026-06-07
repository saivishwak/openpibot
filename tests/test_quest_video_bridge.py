from openpibot.server.runtime import quest_video_bridge as bridge
from openpibot.server.runtime.cameras import CameraSpec


def _cam(role):
    return CameraSpec(
        name=role,
        path=f"/dev/{role}",
        width=640,
        height=480,
        fps=30,
        fourcc="MJPG",
        role=role,
        by_path=None,
        card="test",
    )


def test_video_bridge_reports_ready_for_required_roles(monkeypatch):
    monkeypatch.setattr(bridge.shutil, "which", lambda name: "/usr/bin/gst-launch-1.0")
    monkeypatch.setattr(bridge, "_h264_encoder_pipeline", lambda bitrate_kbps: "x264enc bitrate=2500")
    monkeypatch.setattr(
        bridge.cameras,
        "enumerate_cameras",
        lambda: [_cam("right_wrist"), _cam("head"), _cam("left_wrist")],
    )

    status = bridge.bridge_status()

    assert status["ready"] is False
    assert status["gst_available"] is True
    assert status["running"] is False
    assert status["missing_roles"] == []
    assert status["brightness"] == 0.0
    assert status["contrast"] == 1.0
    assert status["saturation"] == 1.0
    assert status["flip_method"] == "rotate-180"
    assert [stream["role"] for stream in status["streams"]] == ["head", "left_wrist", "right_wrist"]
    assert "videoflip method=rotate-180" in status["streams"][0]["gst_launch"]
    assert "x264enc bitrate=2500" in status["streams"][0]["gst_launch"]
    assert status["streams"][0]["receiver_pipeline"].startswith("udpsrc port=5600")


def test_video_bridge_reports_missing_roles(monkeypatch):
    monkeypatch.setattr(bridge.shutil, "which", lambda name: "/usr/bin/gst-launch-1.0")
    monkeypatch.setattr(bridge.cameras, "enumerate_cameras", lambda: [_cam("head")])

    status = bridge.bridge_status()

    assert status["ready"] is False
    assert status["missing_roles"] == ["left_wrist", "right_wrist"]


def test_video_bridge_starts_and_stops_gstreamer_processes(monkeypatch):
    class FakeStderr:
        def read(self):
            return ""

    class FakeStdin:
        def write(self, data):
            return len(data)

        def flush(self):
            pass

        def close(self):
            pass

    class FakeProcess:
        _next_pid = 1000

        def __init__(self, args, stdin=None, stdout=None, stderr=None, text=None):
            self.args = args
            self.stdin = FakeStdin()
            self.stdout = stdout
            self.stderr = FakeStderr()
            self.text = text
            self.pid = FakeProcess._next_pid
            FakeProcess._next_pid += 1
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

    manager = bridge.QuestVideoBridgeManager()
    monkeypatch.setattr(bridge, "_MANAGER", manager)
    monkeypatch.setattr(bridge.shutil, "which", lambda name: "/usr/bin/gst-launch-1.0")
    monkeypatch.setattr(bridge, "_h264_encoder_pipeline", lambda bitrate_kbps: "x264enc bitrate=2500")
    monkeypatch.setattr(
        bridge.cameras,
        "enumerate_cameras",
        lambda: [_cam("head"), _cam("left_wrist"), _cam("right_wrist")],
    )
    monkeypatch.setattr(bridge.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(manager, "_pump_camera_frames", lambda *args: None)

    status = bridge.start_bridge("192.168.1.55")

    assert status["ready"] is True
    assert status["running_roles"] == ["head", "left_wrist", "right_wrist"]
    assert all(stream["running"] for stream in status["streams"])
    assert "udpsink" in status["streams"][0]["active_gst_launch"]
    assert manager._processes["head"].args[0] == "gst-launch-1.0"
    assert "fdsrc" in manager._processes["head"].args
    assert "v4l2src" not in manager._processes["head"].args

    stopped = bridge.stop_bridge()
    assert stopped["running"] is False


def test_video_bridge_uses_shared_camera_service_without_suspension(monkeypatch):
    class FakeStderr:
        def read(self):
            return ""

    class FakeStdin:
        def write(self, data):
            return len(data)

        def flush(self):
            pass

        def close(self):
            pass

    class FakeProcess:
        def __init__(self, args, stdin=None, stdout=None, stderr=None, text=None):
            self.args = args
            self.stdin = FakeStdin()
            self.stdout = stdout
            self.stderr = FakeStderr()
            self.text = text
            self.pid = 2000
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

    suspended = []
    resumed = []
    manager = bridge.QuestVideoBridgeManager()
    monkeypatch.setattr(bridge, "_MANAGER", manager)
    monkeypatch.setattr(bridge.shutil, "which", lambda name: "/usr/bin/gst-launch-1.0")
    monkeypatch.setattr(bridge, "_h264_encoder_pipeline", lambda bitrate_kbps: "x264enc bitrate=2500")
    monkeypatch.setattr(
        bridge.cameras,
        "enumerate_cameras",
        lambda: [_cam("head"), _cam("left_wrist"), _cam("right_wrist")],
    )
    monkeypatch.setattr(bridge.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(bridge.cameras, "active_capture_roles", lambda: ["head"])
    monkeypatch.setattr(
        bridge.cameras,
        "suspend_capture_roles",
        lambda roles, reason="": suspended.extend(roles) or sorted(roles),
    )
    monkeypatch.setattr(
        bridge.cameras,
        "resume_capture_roles",
        lambda roles=None: resumed.extend(list(roles or [])) or list(roles or []),
    )
    monkeypatch.setattr(bridge.cameras, "suspended_capture_roles", lambda: {"head": "Quest RTP video is using this camera"})
    monkeypatch.setattr(manager, "_pump_camera_frames", lambda *args: None)

    status = bridge.start_bridge("192.168.1.55", roles=["head"])

    assert status["running_roles"] == ["head"]
    assert "fdsrc" in status["streams"][0]["active_gst_launch"]
    assert "v4l2src" not in status["streams"][0]["active_gst_launch"]
    assert suspended == []

    bridge.stop_bridge()
    assert resumed == []


def test_video_bridge_records_quest_receive_health(monkeypatch):
    manager = bridge.QuestVideoBridgeManager()
    monkeypatch.setattr(bridge, "_MANAGER", manager)
    monkeypatch.setattr(bridge.shutil, "which", lambda name: "/usr/bin/gst-launch-1.0")
    monkeypatch.setattr(
        bridge.cameras,
        "enumerate_cameras",
        lambda: [_cam("head"), _cam("left_wrist"), _cam("right_wrist")],
    )

    status = bridge.report_receive_health({
        "role": "head",
        "state": "receiving",
        "fps": 29.5,
        "latency_ms": 42.0,
        "frames": 120,
    })

    assert status["receive_health"]["head"]["state"] == "receiving"
    assert status["receive_health"]["head"]["fps"] == 29.5


def test_video_bridge_ready_false_when_running_role_has_error(monkeypatch):
    class FakeProcess:
        pid = 3000
        returncode = None
        stdin = None

        def poll(self):
            return self.returncode

    manager = bridge.QuestVideoBridgeManager()
    monkeypatch.setattr(bridge, "_MANAGER", manager)
    monkeypatch.setattr(bridge.shutil, "which", lambda name: "/usr/bin/gst-launch-1.0")
    monkeypatch.setattr(bridge, "_h264_encoder_pipeline", lambda bitrate_kbps: "x264enc bitrate=2500")
    monkeypatch.setattr(
        bridge.cameras,
        "enumerate_cameras",
        lambda: [_cam("head"), _cam("left_wrist"), _cam("right_wrist")],
    )
    manager._processes["head"] = FakeProcess()
    manager._processes["left_wrist"] = FakeProcess()
    manager._processes["right_wrist"] = FakeProcess()
    manager._errors["head"] = "camera stream unavailable"
    manager._quest_host = "192.168.1.55"

    status = bridge.bridge_status()

    assert status["ready"] is False
    assert status["errored_roles"] == ["head"]
    assert status["streams"][0]["last_error"] == "camera stream unavailable"


def test_video_bridge_restarts_when_quest_host_changes(monkeypatch):
    class FakeStdin:
        def close(self):
            pass

    class FakeProcess:
        _next_pid = 4000
        terminated = []

        def __init__(self, args, stdin=None, stdout=None, stderr=None, text=None):
            self.args = args
            self.stdin = FakeStdin()
            self.stdout = stdout
            self.stderr = stderr
            self.text = text
            self.pid = FakeProcess._next_pid
            FakeProcess._next_pid += 1
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            FakeProcess.terminated.append(self.pid)
            self.returncode = 0

        def kill(self):
            self.returncode = -9

    manager = bridge.QuestVideoBridgeManager()
    monkeypatch.setattr(bridge, "_MANAGER", manager)
    monkeypatch.setattr(bridge.shutil, "which", lambda name: "/usr/bin/gst-launch-1.0")
    monkeypatch.setattr(bridge, "_h264_encoder_pipeline", lambda bitrate_kbps: "x264enc bitrate=2500")
    monkeypatch.setattr(
        bridge.cameras,
        "enumerate_cameras",
        lambda: [_cam("head")],
    )
    monkeypatch.setattr(bridge.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(manager, "_pump_camera_frames", lambda *args: None)

    first = bridge.start_bridge("192.168.1.55", roles=["head"])
    first_pid = first["streams"][0]["pid"]
    second = bridge.start_bridge("192.168.1.56", roles=["head"])

    assert first_pid in FakeProcess.terminated
    assert second["quest_host"] == "192.168.1.56"
    assert second["streams"][0]["pid"] != first_pid
    assert "host=192.168.1.56" in second["streams"][0]["active_gst_launch"]


def test_video_bridge_pump_terminates_when_camera_stream_unavailable(monkeypatch):
    class FakeStdin:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeProcess:
        def __init__(self):
            self.stdin = FakeStdin()
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

    manager = bridge.QuestVideoBridgeManager()
    proc = FakeProcess()
    stream = bridge.QuestVideoStream(
        role="head",
        camera_name="head",
        device_path="/dev/head",
        width=640,
        height=480,
        fps=30,
        fourcc="MJPG",
        mount="/quest/video/head",
        gst_launch="",
        udp_port=5600,
        receiver_pipeline="",
    )
    monkeypatch.setattr(bridge.cameras, "get_stream", lambda role: None)

    manager._pump_camera_frames("head", stream, proc, bridge.threading.Event())

    assert proc.returncode == 0
    assert proc.stdin.closed is True
    assert manager._errors["head"] == "camera stream unavailable"


def test_video_bridge_pump_reacquires_after_camera_stream_reset(monkeypatch):
    class FakeStdin:
        def __init__(self, proc):
            self.proc = proc
            self.writes = []
            self.closed = False

        def write(self, data):
            self.writes.append(data)
            self.proc.returncode = 0
            return len(data)

        def flush(self):
            pass

        def close(self):
            self.closed = True

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.stdin = FakeStdin(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

    class FakeCameraStream:
        def __init__(self, frame, *, stopped=False):
            self.frame = frame
            self.stop_evt = bridge.threading.Event()
            if stopped:
                self.stop_evt.set()
            self.last_error = None
            self.acquired = 0
            self.released = 0

        def acquire(self):
            self.acquired += 1

        def release(self):
            self.released += 1

        def get_jpeg(self, timeout=1.0):
            return self.frame

    old_stream = FakeCameraStream(b"old-cached-frame", stopped=True)
    new_stream = FakeCameraStream(b"new-live-frame")
    streams = [old_stream, new_stream]

    def get_stream(role):
        return streams.pop(0)

    manager = bridge.QuestVideoBridgeManager()
    proc = FakeProcess()
    stream = bridge.QuestVideoStream(
        role="head",
        camera_name="head",
        device_path="/dev/head",
        width=640,
        height=480,
        fps=30,
        fourcc="MJPG",
        mount="/quest/video/head",
        gst_launch="",
        udp_port=5600,
        receiver_pipeline="",
    )
    monkeypatch.setattr(bridge.cameras, "get_stream", get_stream)

    manager._pump_camera_frames("head", stream, proc, bridge.threading.Event())

    assert proc.stdin.writes == [b"new-live-frame"]
    assert old_stream.acquired == 1
    assert old_stream.released == 1
    assert new_stream.acquired == 1
    assert new_stream.released == 1
    assert "head" not in manager._errors
