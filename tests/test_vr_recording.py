import time
import threading

import pytest
import yaml
from fastapi import HTTPException

from openpibot.server.routers import vr as vr_router
from openpibot.server.runtime.native_quest import NativeQuestAdapter
from openpibot.server.runtime import vr_teleop as vr_mod
from openpibot.server.runtime import dataset as dataset_mod


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

    def end_episode(self):
        self.episode_count += 1
        return True

    def finalize(self):
        self.finalized = True


class _FakeMotors:
    connected_sides = ["left", "right"]

    def __init__(self, present):
        self._present = present

    def read_positions(self):
        return dict(self._present)


class _StrictMotors:
    connected_sides = ["left", "right"]
    bounds = {
        f"{side}_arm_{joint}": (-180.0, 180.0)
        for side in ("left", "right")
        for joint in vr_mod._motors.JOINTS_PER_ARM
    }

    def __init__(self, present=None):
        self._present = present or {
            **_joint_values("left", 10),
            **_joint_values("right", 20),
        }

    def is_connected(self, side):
        return side in self.connected_sides

    def is_torque_enabled(self, side):
        return True

    def read_positions(self, side=None):
        if side is None:
            return dict(self._present)
        return {k: v for k, v in self._present.items() if k.startswith(f"{side}_arm_")}


def _mark_strict_recording_ready(session, monkeypatch):
    monkeypatch.setattr(vr_mod, "MOTORS", _StrictMotors())
    monkeypatch.setattr(
        vr_mod._dataset,
        "role_camera_list",
        lambda: (["head", "left_wrist", "right_wrist"], (2, 2, 3)),
    )
    monkeypatch.setattr(vr_mod._cameras, "suspended_capture_roles", lambda: {})
    monkeypatch.setattr(vr_mod._home, "home_pose_status", lambda: {
        "left": {"captured": True, "joints": {}},
        "right": {"captured": True, "joints": {}},
    })
    monkeypatch.setattr(vr_mod._home, "read_home_pose", lambda: {
        **_joint_values("left", 10),
        **_joint_values("right", 20),
    })
    session._native_quest_clients = 1
    for side in ("left", "right"):
        arm = session._arms[side]
        arm.kinematics = object()
        arm.using_analytical_fallback = False
        arm.cal_confidence = "good"
        arm.robot_verify_quality = "good"
        arm.robot_verify_fit_error_cm = 0.0
        arm.robot_verify_test_completed = True
        arm.controller_anchor_T = vr_mod._pose_matrix_from_vr((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        arm.vr_ctrl_to_ee = vr_mod._R.identity()
        arm.calibrated = True
        arm.anchor_generation = arm.pose_generation
        arm.anchor_invalid_reason = ""


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


def test_write_dataset_config_persists_root_and_repo_id(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "xlerobot.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "dataset": {
            "repo_id": "old/repo",
            "root": "/old/root",
            "fps": 30,
        }
    }))
    monkeypatch.setattr(dataset_mod, "REPO_ROOT", tmp_path)

    cfg = dataset_mod.write_dataset_config(
        root="/new/root",
        repo_id="new-user/new-dataset",
    )
    written = yaml.safe_load(cfg_path.read_text())

    assert cfg["repo_id"] == "new-user/new-dataset"
    assert cfg["root"] == "/new/root"
    assert written["dataset"]["repo_id"] == "new-user/new-dataset"
    assert written["dataset"]["root"] == "/new/root"


def test_set_recording_root_persists_config_and_updates_status(monkeypatch):
    session = vr_mod.VRTeleopSession()
    calls = []

    def fake_write(*, root=None, repo_id=None):
        calls.append((root, repo_id))
        return {"repo_id": repo_id or "test/repo", "root": root, "fps": 30, "push_to_hub": False}

    monkeypatch.setattr(vr_mod._dataset, "write_dataset_config", fake_write)
    monkeypatch.setattr(vr_mod._dataset, "resolve_root", lambda root, repo_id: f"/resolved/{root.strip('/')}")
    monkeypatch.setattr(session, "_recording_calibration_blockers", lambda: [])

    status = session.set_recording_root("/tmp/xlerobot-data")

    assert calls == [("/tmp/xlerobot-data", None)]
    assert session._last_dataset_root == "/resolved/tmp/xlerobot-data"
    assert session._recording_repo_id == "test/repo"
    assert status["recording_info"]["root"] == "/resolved/tmp/xlerobot-data"


def test_set_recording_root_can_update_repo_id_without_changing_root(monkeypatch):
    session = vr_mod.VRTeleopSession()
    calls = []

    def fake_write(*, root=None, repo_id=None):
        calls.append((root, repo_id))
        return {"repo_id": repo_id, "root": "/tmp/current-root", "fps": 30, "push_to_hub": False}

    monkeypatch.setattr(vr_mod._dataset, "write_dataset_config", fake_write)
    monkeypatch.setattr(vr_mod._dataset, "resolve_root", lambda root, repo_id: f"/resolved/{repo_id}/{root.strip('/')}")
    monkeypatch.setattr(session, "_recording_calibration_blockers", lambda: [])

    status = session.set_recording_root(None, repo_id="new-user/new-dataset")

    assert calls == [(None, "new-user/new-dataset")]
    assert session._recording_repo_id == "new-user/new-dataset"
    assert status["recording_info"]["repo_id"] == "new-user/new-dataset"


def test_set_recording_root_rejects_while_recording(monkeypatch):
    session = vr_mod.VRTeleopSession()
    session._recording = True
    monkeypatch.setattr(
        vr_mod._dataset,
        "write_dataset_config",
        lambda root: pytest.fail("must not write config while recording"),
    )

    with pytest.raises(RuntimeError, match="stop recording"):
        session.set_recording_root("/tmp/new-root")


def test_recording_api_rejects_empty_task():
    with pytest.raises(HTTPException) as exc_info:
        vr_router.recording({"enabled": True, "task": "   "})

    assert exc_info.value.status_code == 400


def test_recording_root_api_persists(monkeypatch):
    session = vr_mod.VRTeleopSession()
    monkeypatch.setattr(vr_router.vr_mod, "SESSION", session)
    monkeypatch.setattr(
        session,
        "set_recording_root",
        lambda root, repo_id=None: {"ok": True, "root": root, "repo_id": repo_id},
    )

    assert vr_router.recording_root({"root": "/tmp/data", "repo_id": "test/repo"}) == {
        "ok": True,
        "root": "/tmp/data",
        "repo_id": "test/repo",
    }


def test_recording_root_api_accepts_repo_id_without_root(monkeypatch):
    session = vr_mod.VRTeleopSession()
    monkeypatch.setattr(vr_router.vr_mod, "SESSION", session)
    monkeypatch.setattr(
        session,
        "set_recording_root",
        lambda root, repo_id=None: {"ok": True, "root": root, "repo_id": repo_id},
    )

    assert vr_router.recording_root({"repo_id": "test/repo"}) == {
        "ok": True,
        "root": None,
        "repo_id": "test/repo",
    }


def test_recording_task_api_caches_prompt(monkeypatch):
    session = vr_mod.VRTeleopSession()
    monkeypatch.setattr(vr_router.vr_mod, "SESSION", session)

    resp = vr_router.recording_task({"task": "  Pick the red block  "})

    assert isinstance(resp, dict)
    assert session._last_task == "Pick the red block"


def test_recording_restart_waits_for_previous_stop_to_finish(monkeypatch):
    session = vr_mod.VRTeleopSession()
    old_rec = _FakeRecorder()
    new_rec = _FakeRecorder()
    new_rec.episode_count = 1
    entered_end = threading.Event()
    allow_end = threading.Event()
    start_done = threading.Event()

    def slow_end_episode():
        entered_end.set()
        assert allow_end.wait(timeout=2.0)
        old_rec.episode_count += 1
        return True

    old_rec.end_episode = slow_end_episode
    session._recording = True
    session._recorder = old_rec
    session._last_task = "Pick the red block"

    monkeypatch.setattr(
        vr_mod._dataset,
        "load_dataset_config",
        lambda: {
            "home_before_episode": False,
            "repo_id": "test/repo",
                "fps": 30,
                "push_to_hub": False,
                "root": None,
            },
        )
    monkeypatch.setattr(
        vr_mod._dataset,
        "role_camera_list",
        lambda: (["head", "left_wrist", "right_wrist"], (2, 2, 3)),
    )
    monkeypatch.setattr(vr_mod._dataset, "resolve_root", lambda root, repo_id: "/tmp/test/repo")
    monkeypatch.setattr(vr_mod._dataset, "DatasetRecorder", lambda **kwargs: new_rec)
    monkeypatch.setattr(session, "_recording_calibration_blockers", lambda: [])

    stop_thread = threading.Thread(target=lambda: session.set_recording(False))
    stop_thread.start()
    assert entered_end.wait(timeout=2.0)

    start_thread = threading.Thread(
        target=lambda: (session.set_recording(True, task="Pick the red block"), start_done.set())
    )
    start_thread.start()
    time.sleep(0.05)
    assert not start_done.is_set()

    allow_end.set()
    stop_thread.join(timeout=2.0)
    start_thread.join(timeout=2.0)

    assert start_done.is_set()
    assert getattr(old_rec, "finalized", False) is True
    assert session._recording is True
    assert session._recorder is new_rec
    assert new_rec.task == "Pick the red block"
    assert session._episodes_saved == 1


def test_b_button_start_requires_synced_task(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()

    monkeypatch.setattr(
        vr_mod._dataset,
        "load_dataset_config",
        lambda: {
            "home_before_episode": False,
            "repo_id": "test/repo",
                "fps": 30,
                "push_to_hub": False,
                "root": None,
            },
        )
    monkeypatch.setattr(
        vr_mod._dataset,
        "role_camera_list",
        lambda: (["head", "left_wrist", "right_wrist"], (2, 2, 3)),
    )
    monkeypatch.setattr(vr_mod._dataset, "resolve_root", lambda root, repo_id: "/tmp/test/repo")
    monkeypatch.setattr(vr_mod._dataset, "DatasetRecorder", lambda **kwargs: recorder)
    monkeypatch.setattr(session, "_recording_calibration_blockers", lambda: [])

    session._handle_record_button("right")

    assert session._recording is False
    assert "task description required" in (session._last_error or "")

    session.set_recording_task("Pick the red block")
    session._handle_record_button("right")

    assert session._recording is True
    assert recorder.task == "Pick the red block"


def test_native_quest_button_release_is_forwarded_for_repeat_b_toggles():
    adapter = NativeQuestAdapter(coordinate_frame="unity_openxr")

    packets = [
        {"controllers": {"right": {"position": [0, 0, 0], "rotation": [0, 0, 0, 1], "buttons": {"B": True}}}},
        {"controllers": {"right": {"position": [0, 0, 0], "rotation": [0, 0, 0, 1], "buttons": {"B": False}}}},
        {"controllers": {"right": {"position": [0, 0, 0], "rotation": [0, 0, 0, 1], "buttons": {"B": True}}}},
    ]

    goals = [adapter.process_packet(packet)[0] for packet in packets]

    assert [goal.buttons for goal in goals] == [{"B": True}, {"B": False}, {"B": True}]


def test_delete_last_recorded_episode_requires_stop():
    session = vr_mod.VRTeleopSession()
    session._recording = True

    out = session.delete_last_recorded_episode()

    assert out["recording"] is True
    assert "stop recording" in (out["last_error"] or "")


def test_delete_last_recorded_episode_updates_counters(monkeypatch):
    session = vr_mod.VRTeleopSession()
    session._episodes_saved = 3
    session._last_dataset_root = "/tmp/old-root"

    monkeypatch.setattr(
        vr_mod._dataset,
        "load_dataset_config",
        lambda: {"repo_id": "test/repo", "root": None},
    )
    monkeypatch.setattr(
        vr_mod._dataset,
        "delete_last_episode",
        lambda repo_id, root: (2, "/tmp/new-root"),
    )
    monkeypatch.setattr(
        vr_mod._dataset,
        "last_episode_summary",
        lambda repo_id, root: (1, 420),
    )

    out = session.delete_last_recorded_episode()

    assert out["recording_info"]["episodes_saved"] == 2
    assert out["recording_info"]["root"] == "/tmp/new-root"
    assert out["recording_info"]["last_episode_index"] == 1
    assert out["recording_info"]["last_episode_frames"] == 420
    assert session._episodes_saved == 2
    assert session._last_dataset_root == "/tmp/new-root"


def test_delete_last_recorded_episode_api(monkeypatch):
    session = vr_mod.VRTeleopSession()
    monkeypatch.setattr(vr_router.vr_mod, "SESSION", session)
    monkeypatch.setattr(
        session,
        "delete_last_recorded_episode",
        lambda: {"ok": True, "recording": False},
    )

    resp = vr_router.recording_delete_last()

    assert resp == {"ok": True, "recording": False}


def test_start_recording_after_delete_keeps_saved_count(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    recorder.episode_count = 2
    recorder.last_saved_episode_index = 1
    recorder.last_saved_episode_frames = 420

    session._episodes_saved = 2
    session._recording = False
    session._last_task = "Pick the red block"

    monkeypatch.setattr(
        vr_mod._dataset,
        "load_dataset_config",
        lambda: {
            "home_before_episode": False,
            "repo_id": "test/repo",
                "fps": 30,
                "push_to_hub": False,
                "root": None,
            },
        )
    monkeypatch.setattr(
        vr_mod._dataset,
        "role_camera_list",
        lambda: (["head", "left_wrist", "right_wrist"], (2, 2, 3)),
    )
    monkeypatch.setattr(vr_mod._dataset, "resolve_root", lambda root, repo_id: "/tmp/test/repo")
    monkeypatch.setattr(vr_mod._dataset, "DatasetRecorder", lambda **kwargs: recorder)
    monkeypatch.setattr(session, "_recording_calibration_blockers", lambda: [])

    assert session.set_recording(True, task="Pick the red block") is True
    status = session.status()
    assert status["recording_info"]["episodes_saved"] == 2
    assert status["recording_info"]["last_episode_index"] == 1
    assert status["recording_info"]["last_episode_frames"] == 420


def test_recording_start_arms_until_fresh_anchors(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    _mark_strict_recording_ready(session, monkeypatch)
    monkeypatch.setattr(
        vr_mod._dataset,
        "load_dataset_config",
        lambda: {
            "home_before_episode": False,
            "repo_id": "test/repo",
            "fps": 30,
            "push_to_hub": False,
            "root": None,
        },
    )
    monkeypatch.setattr(vr_mod._dataset, "resolve_root", lambda root, repo_id: "/tmp/test/repo")
    monkeypatch.setattr(vr_mod._dataset, "DatasetRecorder", lambda **kwargs: recorder)

    session._invalidate_teleop_anchor("left", "homing changed robot pose")

    assert session.set_recording(True, task="Pick the red block") is False
    assert session._recording is False
    assert session._recording_armed is True
    assert "waiting for fresh" in (session._last_error or "")

    with session._lock:
        left = session._arms["left"]
        left.controller_anchor_T = vr_mod._pose_matrix_from_vr((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        left.vr_ctrl_to_ee = vr_mod._R.identity()
        left.calibrated = True
        left.anchor_generation = left.pose_generation
        left.anchor_invalid_reason = ""
        session._maybe_start_armed_recording_locked()

    assert session._recording is True
    assert session._recording_armed is False
    assert recorder.task == "Pick the red block"


def test_go_home_invalidates_existing_anchor(monkeypatch):
    session = vr_mod.VRTeleopSession()
    _mark_strict_recording_ready(session, monkeypatch)
    before_generation = session._arms["right"].pose_generation

    session.go_home(side="right")

    arm = session._arms["right"]
    assert arm.pose_generation == before_generation + 1
    assert arm.calibrated is False
    assert arm.controller_anchor_T is None
    assert "homing" in arm.anchor_invalid_reason


def test_strict_frame_failure_aborts_episode(monkeypatch):
    class RejectingRecorder(_FakeRecorder):
        def __init__(self):
            super().__init__()
            self.discarded = False

        def add_frame(self, action, present, camera_frames):
            raise RuntimeError("camera frames missing roles: head")

        def discard_episode(self):
            self.discarded = True
            self.in_episode = False

    recorder = RejectingRecorder()
    session = vr_mod.VRTeleopSession()
    present = {**_joint_values("left", 10), **_joint_values("right", 20)}
    session._recording = True
    session._recorder = recorder
    monkeypatch.setattr(vr_mod, "MOTORS", _StrictMotors(present))
    monkeypatch.setattr(vr_mod._dataset, "grab_camera_frames", lambda: {})

    session._record_frame_if_active(
        commanded_this_tick={
            "left": _joint_values("left", 100),
            "right": _joint_values("right", 200),
        }
    )

    assert session._recording is False
    assert recorder.discarded is True
    assert "strict frame rejected" in (session._last_error or "")
