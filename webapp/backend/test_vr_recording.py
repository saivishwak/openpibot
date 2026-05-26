import time

from flask import Flask

from webapp.backend import api as api_mod
from webapp.backend import vr_teleop as vr_mod


class _FakeRecorder:
    repo_id = "test/repo"
    episode_count = 0
    frame_count_in_episode = 0
    in_episode = True

    def __init__(self):
        self.frames = []

    def start_episode(self, task=""):
        self.task = task

    def add_frame(self, action, present, camera_frames):
        self.frames.append((action, present, camera_frames))


class _FakeMotors:
    connected_sides = ["left", "right"]

    def __init__(self, present):
        self._present = present

    def read_positions(self):
        return dict(self._present)


def _joint_values(side, base):
    prefix = f"{side}_arm_"
    return {
        f"{prefix}shoulder_pan": base + 0,
        f"{prefix}shoulder_lift": base + 1,
        f"{prefix}elbow_flex": base + 2,
        f"{prefix}wrist_flex": base + 3,
        f"{prefix}wrist_roll": base + 4,
        f"{prefix}gripper": base + 5,
    }


def test_record_frame_uses_same_tick_commanded_actions(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    present = {**_joint_values("left", 10), **_joint_values("right", 20)}
    left_command = _joint_values("left", 100)
    right_command = _joint_values("right", 200)

    session._recording = True
    session._recorder = recorder
    session._engaged = True
    monkeypatch.setattr(vr_mod, "MOTORS", _FakeMotors(present))
    monkeypatch.setattr(vr_mod._dataset, "grab_camera_frames", lambda: {})

    session._record_frame_if_active(
        commanded_this_tick={"left": left_command, "right": right_command}
    )

    action, observed, _ = recorder.frames[0]
    assert observed == present
    assert action == {**left_command, **right_command}


def test_record_frame_falls_back_to_present_for_passive_arm(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    present = {**_joint_values("left", 10), **_joint_values("right", 20)}
    right_command = _joint_values("right", 200)

    session._recording = True
    session._recorder = recorder
    monkeypatch.setattr(vr_mod, "MOTORS", _FakeMotors(present))
    monkeypatch.setattr(vr_mod._dataset, "grab_camera_frames", lambda: {})

    session._record_frame_if_active(commanded_this_tick={"right": right_command})

    action, _, _ = recorder.frames[0]
    assert {k: action[k] for k in _joint_values("left", 0)} == _joint_values("left", 10)
    assert {k: action[k] for k in _joint_values("right", 0)} == right_command


def test_set_recording_rejects_empty_task(monkeypatch):
    session = vr_mod.VRTeleopSession()
    monkeypatch.setattr(vr_mod._dataset, "load_dataset_config", lambda: {"home_before_episode": False})

    assert session.set_recording(True, task="   ") is False
    assert session._recording is False
    assert "task description required" in (session._last_error or "")


def test_set_recording_task_cache_can_be_cleared():
    session = vr_mod.VRTeleopSession()

    session.set_recording_task("  Pick the red block  ")
    assert session._last_task == "Pick the red block"

    session.set_recording_task("   ")
    assert session._last_task == ""


def test_recording_api_rejects_empty_task():
    app = Flask(__name__)
    app.register_blueprint(api_mod.bp)

    resp = app.test_client().post(
        "/api/vr/recording",
        json={"enabled": True, "task": "   "},
    )

    assert resp.status_code == 400


def test_recording_task_api_caches_prompt(monkeypatch):
    session = vr_mod.VRTeleopSession()
    monkeypatch.setattr(api_mod.vr_mod, "SESSION", session)
    app = Flask(__name__)
    app.register_blueprint(api_mod.bp)

    resp = app.test_client().post(
        "/api/vr/recording/task",
        json={"task": "  Pick the red block  "},
    )

    assert resp.status_code == 200
    assert session._last_task == "Pick the red block"

