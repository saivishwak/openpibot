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

    class FakeProcess:
        _next_pid = 1000

        def __init__(self, args, stdout=None, stderr=None, text=None):
            self.args = args
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
    monkeypatch.setattr(bridge.cameras, "active_capture_roles", lambda: [])

    status = bridge.start_bridge("192.168.1.55")

    assert status["ready"] is True
    assert status["running_roles"] == ["head", "left_wrist", "right_wrist"]
    assert all(stream["running"] for stream in status["streams"])
    assert "udpsink" in status["streams"][0]["active_gst_launch"]
    assert manager._processes["head"].args[0] == "gst-launch-1.0"
    assert "v4l2src" in manager._processes["head"].args

    stopped = bridge.stop_bridge()
    assert stopped["running"] is False


def test_video_bridge_suspends_open_cv_camera_contention(monkeypatch):
    class FakeStderr:
        def read(self):
            return ""

    class FakeProcess:
        def __init__(self, args, stdout=None, stderr=None, text=None):
            self.args = args
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

    status = bridge.start_bridge("192.168.1.55", roles=["head"])

    assert status["running_roles"] == ["head"]
    assert suspended == ["head"]

    bridge.stop_bridge()
    assert "head" in resumed


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
