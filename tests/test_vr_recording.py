import time
import threading

import numpy as np
import pytest
import yaml

from openpibot.server.routers import vr as vr_router
from openpibot.server.runtime.native_quest import NativeQuestAdapter
from openpibot.server.runtime import vr_teleop as vr_mod
from openpibot.server.runtime import dataset as dataset_mod


class _FakeRecorder:
    repo_id = "test/repo"
    episode_count = 0
    frame_count_in_episode = 0
    in_episode = True
    last_end_reason = ""
    last_saved_episode_index = None
    last_saved_episode_frames = 0

    def __init__(self):
        self.frames = []

    def start_episode(self, task=""):
        self.task = task

    def add_frame(self, action, present, camera_frames):
        self.frames.append((action, present, camera_frames))

    def end_episode(self):
        self.episode_count += 1
        self.last_end_reason = ""
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


class _FakeKinematics:
    def forward_kinematics(self, q_now_deg):
        out = np.eye(4)
        out[0, 3] = 0.12
        out[1, 3] = 0.0
        out[2, 3] = 0.18
        return out


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
        arm.kinematics = _FakeKinematics()
        arm.using_analytical_fallback = False
        arm.wrist_pitch_canonical = (1.0, 0.0, 0.0)
        arm.wrist_roll_canonical = (0.0, 0.0, -1.0)
        arm.cal_confidence = "good"
        arm.robot_verify_quality = "good"
        arm.robot_verify_fit_error_cm = 0.0
        arm.robot_verify_test_completed = True
        arm.controller_anchor_T = vr_mod._pose_matrix_from_vr((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        arm.vr_ctrl_to_ee = vr_mod._R.identity()
        arm.calibrated = True
        arm.anchor_generation = arm.pose_generation
        arm.anchor_invalid_reason = ""
        arm.latest = vr_mod._LatestGoal(
            received_at=time.time(),
            has_data=True,
            mode="idle",
            controller_position=(0.1 if side == "left" else -0.1, 0.0, 0.2),
            rotation_quat=(0.0, 0.0, 0.0, 1.0),
        )


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


def _wait_until(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_homing_step_does_not_finish_while_feedback_lags():
    session = vr_mod.VRTeleopSession()
    arm = session._arms["left"]
    present = _joint_values("left", 0)
    target = _joint_values("left", 30)
    arm.home_target = dict(target)
    arm.last_sent_targets = dict(present)

    settled = False
    for _ in range(100):
        command, settled = session._homing_step_command("left", arm, present)
        arm.last_sent_targets = dict(command)
        if settled:
            break

    assert settled is False


def test_homing_requires_consecutive_feedback_settle_ticks():
    session = vr_mod.VRTeleopSession()
    arm = session._arms["left"]
    present = _joint_values("left", 0)
    target = _joint_values("left", 30)
    arm.home_target = dict(target)
    arm.home_controller = session._new_homing_controller(arm, present)

    for _ in range(100):
        step = session._homing_step(arm, present)

    assert step.command_reached is True
    assert step.present_reached is False
    assert step.settled is False

    for i in range(vr_mod.HOMING_SETTLE_TICKS):
        step = session._homing_step(arm, target)
        assert step.present_reached is True
        assert step.settled is (i == vr_mod.HOMING_SETTLE_TICKS - 1)


def test_homing_accepts_stable_so101_feedback_deadband():
    controller = vr_mod.JointHomingController(
        targets={"left_arm_elbow_flex": 78.5934},
        present={"left_arm_elbow_flex": 75.1634},
        cap_for_key=lambda _key: 5.0,
        kp=0.75,
        command_tolerance_deg=0.5,
        present_tolerance_deg=4.0,
        final_direct_tolerance_deg=5.0,
        settle_ticks=vr_mod.HOMING_SETTLE_TICKS,
    )

    for i in range(vr_mod.HOMING_SETTLE_TICKS):
        step = controller.step({"left_arm_elbow_flex": 75.1634})
        assert step.command["left_arm_elbow_flex"] == 78.5934
        assert step.present_reached is True
        assert step.settled is (i == vr_mod.HOMING_SETTLE_TICKS - 1)


def test_homing_final_approach_commands_exact_target_with_small_feedback_error():
    controller = vr_mod.JointHomingController(
        targets={"joint": 30.0},
        present={"joint": 0.0},
        cap_for_key=lambda _key: 1.0,
        kp=0.75,
        command_tolerance_deg=0.5,
        present_tolerance_deg=1.0,
        final_direct_tolerance_deg=5.0,
    )

    for _ in range(40):
        step = controller.step({"joint": 27.0})

    assert step.command["joint"] == 30.0
    assert step.present_reached is False


def test_drive_loop_sends_homing_command_after_go_home(monkeypatch):
    class HomingMotors:
        connected_sides = ["left"]
        bounds = {
            f"left_arm_{joint}": (-180.0, 180.0)
            for joint in vr_mod._motors.JOINTS_PER_ARM
        }

        def __init__(self):
            self.positions = _joint_values("left", 0)
            self.sent = []
            self.sent_event = threading.Event()

        def is_connected(self, side):
            return side == "left"

        def is_torque_enabled(self, side):
            return side == "left"

        def read_positions(self, side=None):
            if side is None:
                return dict(self.positions)
            return {k: v for k, v in self.positions.items() if k.startswith(f"{side}_arm_")}

        def send_action(self, side, action):
            assert side == "left"
            self.sent.append(dict(action))
            self.positions.update({k: float(v) for k, v in action.items()})
            self.sent_event.set()
            return dict(action)

    motors = HomingMotors()
    session = vr_mod.VRTeleopSession()
    monkeypatch.setattr(vr_mod, "MOTORS", motors)
    monkeypatch.setattr(vr_mod._home, "read_home_pose", lambda: _joint_values("left", 20))

    session.go_home("left")
    session._start_drive_loop()
    try:
        assert motors.sent_event.wait(timeout=1.0)
    finally:
        session._stop_evt.set()
        if session._drive_thread is not None:
            session._drive_thread.join(timeout=1.0)

    assert motors.sent
    assert session._arms["left"].home_last_worst_joint.startswith("left_arm_")


def test_dataset_feature_names_match_lerobot_xlerobot_pos_convention():
    features = dataset_mod.DatasetRecorder._build_features(
        dataset_mod.LEROBOT_JOINT_NAMES,
        ["head"],
        (2, 2, 3),
    )

    assert features["action"]["names"] == [f"{name}.pos" for name in dataset_mod.JOINT_ORDER]
    assert features["observation.state"]["names"] == features["action"]["names"]


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


def test_record_frame_keeps_direct_vr_wrist_command(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    present = {**_joint_values("left", 10), **_joint_values("right", 20)}
    right_command = _joint_values("right", 200)
    right_command["right_arm_wrist_flex"] = 37.5
    right_command["right_arm_wrist_roll"] = -18.25
    present["right_arm_wrist_flex"] = -70.0
    present["right_arm_wrist_roll"] = 70.0

    session._recording = True
    session._recorder = recorder
    session._engaged = True
    monkeypatch.setattr(vr_mod, "MOTORS", _FakeMotors(present))
    monkeypatch.setattr(vr_mod._dataset, "grab_camera_frames", lambda: {})

    session._record_frame_if_active(commanded_this_tick={"right": right_command})

    action, observed, _ = recorder.frames[0]
    assert observed["right_arm_wrist_flex"] == pytest.approx(-70.0)
    assert observed["right_arm_wrist_roll"] == pytest.approx(70.0)
    assert action["right_arm_wrist_flex"] == pytest.approx(37.5)
    assert action["right_arm_wrist_roll"] == pytest.approx(-18.25)


def test_record_frame_uses_recorder_owned_camera_streams(monkeypatch):
    class RecorderWithCameraGrab(_FakeRecorder):
        def __init__(self):
            super().__init__()
            self.grabbed = False

        def grab_camera_frames(self):
            self.grabbed = True
            return {"head": np.zeros((2, 2, 3), dtype=np.uint8)}

    session = vr_mod.VRTeleopSession()
    recorder = RecorderWithCameraGrab()
    present = {**_joint_values("left", 10), **_joint_values("right", 20)}
    left_command = _joint_values("left", 100)
    right_command = _joint_values("right", 200)

    session._recording = True
    session._recorder = recorder
    session._engaged = True
    monkeypatch.setattr(vr_mod, "MOTORS", _FakeMotors(present))
    monkeypatch.setattr(
        vr_mod._dataset,
        "grab_camera_frames",
        lambda: pytest.fail("recording must use recorder-owned camera streams"),
    )

    session._record_frame_if_active(
        commanded_this_tick={"left": left_command, "right": right_command}
    )

    assert recorder.grabbed is True
    assert len(recorder.frames) == 1


def test_dataset_camera_grab_uses_fresh_raw_rgb_not_jpeg():
    class RawOnlyStream:
        def __init__(self):
            self.rgb_called = False
            self.acquired = False
            self.released = False

        def acquire(self):
            self.acquired = True

        def release(self):
            self.released = True

        def get_rgb(self, timeout=0.5, *, max_age_s=None, copy=False):
            self.rgb_called = True
            assert timeout == 0.5
            assert max_age_s == 0.25
            assert copy is True
            return np.ones((2, 2, 3), dtype=np.uint8)

        def get_jpeg(self, timeout=2.0):
            raise AssertionError("dataset recording must not use JPEG/Quest video path")

    stream = RawOnlyStream()

    frames = dataset_mod.grab_camera_frames(streams={"head": stream}, max_age_s=0.25)

    assert stream.rgb_called is True
    assert stream.acquired is False
    assert stream.released is False
    assert frames["head"].shape == (2, 2, 3)


def test_dataset_camera_grab_resizes_to_expected_lerobot_shape():
    class Stream:
        def get_rgb(self, timeout=0.5, *, max_age_s=None, copy=False):
            return np.ones((4, 8, 3), dtype=np.uint8)

    frames = dataset_mod.grab_camera_frames(
        streams={"head": Stream()},
        expected_shape=(2, 3, 3),
    )

    assert frames["head"].shape == (2, 3, 3)
    assert frames["head"].dtype == np.uint8
    assert frames["head"].flags["C_CONTIGUOUS"]


def test_dataset_camera_validation_reports_actual_bad_shape_and_dtype():
    recorder = dataset_mod.DatasetRecorder.__new__(dataset_mod.DatasetRecorder)
    recorder.camera_roles = ["head"]
    recorder.camera_shape = (2, 3, 3)

    with pytest.raises(RuntimeError, match=r"head: dtype=float32 shape=\(4, 8, 3\)"):
        recorder._validate_camera_frames({
            "head": np.ones((4, 8, 3), dtype=np.float32),
        })


def test_dataset_camera_validation_rejects_empty_right_wrist_before_writer():
    class WriterMustNotRun:
        def add_frame(self, frame):
            raise AssertionError("invalid camera frame must not reach LeRobot writer")

    recorder = dataset_mod.DatasetRecorder.__new__(dataset_mod.DatasetRecorder)
    recorder._lock = threading.Lock()
    recorder._io_lock = threading.Lock()
    recorder._in_episode = True
    recorder._frame_count = 0
    recorder._current_task = "test"
    recorder._last_end_reason = ""
    recorder.camera_roles = ["head", "left_wrist", "right_wrist"]
    recorder.camera_shape = (2, 3, 3)
    recorder._dataset = WriterMustNotRun()

    valid = np.ones((2, 3, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match=r"right_wrist: empty array dtype=uint8 shape=\(0,\)"):
        recorder.add_frame(
            {**_joint_values("left", 10), **_joint_values("right", 20)},
            {**_joint_values("left", 10), **_joint_values("right", 20)},
            {
                "head": valid,
                "left_wrist": valid,
                "right_wrist": np.array([], dtype=np.uint8),
            },
        )

    assert recorder._frame_count == 0


def test_dataset_recorder_acquire_waits_for_fresh_camera_frame(monkeypatch):
    class Stream:
        def __init__(self, frame):
            self.frame = frame
            self.acquired = False
            self.released = False

        def acquire(self):
            self.acquired = True

        def release(self):
            self.released = True

        def get_rgb(self, timeout=2.0, *, max_age_s=None, copy=False):
            assert timeout == 3.0
            assert max_age_s == 1.0
            return self.frame

    stream = Stream(np.ones((2, 2, 3), dtype=np.uint8))
    recorder = dataset_mod.DatasetRecorder.__new__(dataset_mod.DatasetRecorder)
    recorder.camera_roles = ["head"]
    recorder.camera_shape = (2, 2, 3)
    monkeypatch.setattr(dataset_mod.cam_mod, "get_stream", lambda role: stream)

    streams = recorder._acquire_camera_streams()

    assert streams == {"head": stream}
    assert stream.acquired is True
    assert stream.released is False


def test_dataset_recorder_acquire_releases_when_camera_never_warms(monkeypatch):
    class Stream:
        last_error = "timeout"

        def __init__(self):
            self.acquired = False
            self.released = False

        def acquire(self):
            self.acquired = True

        def release(self):
            self.released = True

        def get_rgb(self, timeout=2.0, *, max_age_s=None, copy=False):
            return None

    stream = Stream()
    recorder = dataset_mod.DatasetRecorder.__new__(dataset_mod.DatasetRecorder)
    recorder.camera_roles = ["head"]
    recorder.camera_shape = (2, 2, 3)
    monkeypatch.setattr(dataset_mod.cam_mod, "get_stream", lambda role: stream)

    with pytest.raises(RuntimeError, match="camera head did not produce a valid fresh frame"):
        recorder._acquire_camera_streams()

    assert stream.acquired is True
    assert stream.released is True


def test_record_frame_uses_present_as_passive_arm_noop_action(monkeypatch):
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


def test_record_frame_uses_last_command_for_stale_active_goal(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    present = {**_joint_values("left", 10), **_joint_values("right", 20)}
    right_command = _joint_values("right", 200)
    session._recording = True
    session._recorder = recorder
    session._engaged = True
    session._active_arm = "right"
    right = session._arms["right"]
    right.calibrated = True
    right.last_commanded_targets = dict(right_command)
    right.latest = vr_mod._LatestGoal(
        received_at=time.time() - vr_mod.GOAL_SKIP_AGE_S - 0.5,
        has_data=True,
        mode="position",
    )
    monkeypatch.setattr(vr_mod, "MOTORS", _StrictMotors(present))
    monkeypatch.setattr(vr_mod._dataset, "grab_camera_frames", lambda: {})

    session._record_frame_if_active(commanded_this_tick={})

    assert session._recording is True
    assert session._last_error is None
    action, _, _ = recorder.frames[0]
    assert {k: action[k] for k in _joint_values("left", 0)} == _joint_values("left", 10)
    assert {k: action[k] for k in _joint_values("right", 0)} == right_command


def test_record_frame_uses_last_command_for_idle_engaged_arm(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    present = {**_joint_values("left", 10), **_joint_values("right", 20)}
    right_command = _joint_values("right", 200)
    session._recording = True
    session._recorder = recorder
    session._engaged = True
    session._active_arm = "right"
    right = session._arms["right"]
    right.calibrated = True
    right.last_commanded_targets = dict(right_command)
    right.latest = vr_mod._LatestGoal(
        received_at=time.time(),
        has_data=True,
        mode="idle",
    )
    monkeypatch.setattr(vr_mod, "MOTORS", _StrictMotors(present))
    monkeypatch.setattr(vr_mod._dataset, "grab_camera_frames", lambda: {})

    session._record_frame_if_active(commanded_this_tick={})

    assert session._recording is True
    assert session._last_error is None
    action, _, _ = recorder.frames[0]
    assert {k: action[k] for k in _joint_values("left", 0)} == _joint_values("left", 10)
    assert {k: action[k] for k in _joint_values("right", 0)} == right_command


def test_record_frame_aborts_missing_held_command_for_controlled_hold(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    present = {**_joint_values("left", 10), **_joint_values("right", 20)}
    session._recording = True
    session._recorder = recorder
    session._engaged = True
    session._active_arm = "right"
    right = session._arms["right"]
    right.calibrated = True
    right.last_commanded_targets = {"right_arm_shoulder_pan": 200.0}
    right.latest = vr_mod._LatestGoal(
        received_at=time.time(),
        has_data=True,
        mode="idle",
    )
    monkeypatch.setattr(vr_mod, "MOTORS", _StrictMotors(present))
    monkeypatch.setattr(vr_mod._dataset, "grab_camera_frames", lambda: {})

    session._record_frame_if_active(commanded_this_tick={})

    assert session._recording is False
    assert "missing held motor command for controlled arm(s): right" in (session._last_error or "")
    assert recorder.frames == []


def test_record_frame_aborts_missing_command_for_fresh_position_goal(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    present = {**_joint_values("left", 10), **_joint_values("right", 20)}
    session._recording = True
    session._recorder = recorder
    session._engaged = True
    session._active_arm = "right"
    right = session._arms["right"]
    right.calibrated = True
    right.last_commanded_targets = _joint_values("right", 200)
    right.latest = vr_mod._LatestGoal(
        received_at=time.time(),
        has_data=True,
        mode="position",
    )
    monkeypatch.setattr(vr_mod, "MOTORS", _StrictMotors(present))
    monkeypatch.setattr(vr_mod._dataset, "grab_camera_frames", lambda: {})

    session._record_frame_if_active(commanded_this_tick={})

    assert session._recording is False
    assert "missing same-tick motor command for driven arm(s): right" in (session._last_error or "")
    assert recorder.frames == []


def test_record_frame_uses_drive_tick_snapshot_when_goal_updates_after_hold(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    present = {**_joint_values("left", 10), **_joint_values("right", 20)}
    right_command = _joint_values("right", 200)
    session._recording = True
    session._recorder = recorder
    session._engaged = True
    session._active_arm = "right"
    right = session._arms["right"]
    right.calibrated = True
    right.last_commanded_targets = dict(right_command)
    # Simulate the race seen in the logs: the drive tick decided not to send a
    # right-arm command because it processed a reset/hold packet, then a fresh
    # POSITION packet arrived before the recording frame was written.
    right.latest = vr_mod._LatestGoal(
        received_at=time.time(),
        has_data=True,
        mode="position",
    )
    monkeypatch.setattr(vr_mod, "MOTORS", _StrictMotors(present))
    monkeypatch.setattr(vr_mod._dataset, "grab_camera_frames", lambda: {})

    session._record_frame_if_active(
        commanded_this_tick={},
        expected_driven_sides=[],
        expected_held_sides=["right"],
    )

    assert session._recording is True
    assert session._last_error is None
    action, _, _ = recorder.frames[0]
    assert {k: action[k] for k in _joint_values("left", 0)} == _joint_values("left", 10)
    assert {k: action[k] for k in _joint_values("right", 0)} == right_command


def test_record_frame_does_not_require_command_for_uncalibrated_active_goal(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    present = {**_joint_values("left", 10), **_joint_values("right", 20)}
    session._recording = True
    session._recorder = recorder
    session._engaged = True
    session._active_arm = "right"
    right = session._arms["right"]
    right.calibrated = False
    right.latest = vr_mod._LatestGoal(
        received_at=time.time(),
        has_data=True,
        mode="position",
    )
    monkeypatch.setattr(vr_mod, "MOTORS", _StrictMotors(present))
    monkeypatch.setattr(vr_mod._dataset, "grab_camera_frames", lambda: {})

    session._record_frame_if_active(commanded_this_tick={})

    assert session._recording is True
    assert session._last_error is None
    action, _, _ = recorder.frames[0]
    assert action == present


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


def test_set_recording_root_rejects_during_recording_transition(monkeypatch):
    session = vr_mod.VRTeleopSession()
    assert session._recording_transition_lock.acquire(blocking=False)
    monkeypatch.setattr(
        vr_mod._dataset,
        "write_dataset_config",
        lambda **kwargs: pytest.fail("must not write config during recording transition"),
    )
    try:
        with pytest.raises(RuntimeError, match="recording transition"):
            session.set_recording_root("/tmp/new-root")
    finally:
        session._recording_transition_lock.release()


def test_recording_api_delegates_empty_task_to_session(monkeypatch):
    session = vr_mod.VRTeleopSession()
    called = {}
    monkeypatch.setattr(vr_router.vr_mod, "SESSION", session)
    monkeypatch.setattr(
        session,
        "set_recording",
        lambda enabled, task="", root="", source="api": called.update(
            {"enabled": enabled, "task": task, "root": root, "source": source}
        ) or False,
    )
    monkeypatch.setattr(
        session,
        "status",
        lambda: {
            "recording": False,
            "last_error": "task description required before starting an episode",
            "recording_info": {"notice": "task description required before starting an episode"},
        },
    )

    with pytest.raises(vr_router.HTTPException) as exc:
        vr_router.recording({"enabled": True, "task": "   "})

    assert exc.value.status_code == 409
    assert "task description required" in str(exc.value.detail)
    assert called == {"enabled": True, "task": "   ", "root": "", "source": "webapp"}


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
    monkeypatch.setattr(session, "_refresh_recording_anchors_for_start", lambda: [])

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
    monkeypatch.setattr(session, "_refresh_recording_anchors_for_start", lambda: [])

    session._handle_record_button("right")

    assert _wait_until(lambda: "task description required" in (session._last_error or ""))
    assert session._recording is False

    session.set_recording_task("Pick the red block")
    session._handle_record_button("right")

    assert _wait_until(lambda: session._recording is True)
    assert recorder.task == "Pick the red block"


def test_b_button_start_uses_configured_default_task(monkeypatch):
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
            "task_default": "default headset task",
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
    monkeypatch.setattr(session, "_refresh_recording_anchors_for_start", lambda: [])

    session._handle_record_button("right")

    assert _wait_until(lambda: session._recording is True)
    assert recorder.task == "default headset task"


def test_recording_start_fails_if_dataset_config_cannot_be_read(monkeypatch):
    session = vr_mod.VRTeleopSession()
    _mark_strict_recording_ready(session, monkeypatch)

    def bad_config():
        raise RuntimeError("bad dataset config")

    monkeypatch.setattr(vr_mod._dataset, "load_dataset_config", bad_config)
    monkeypatch.setattr(
        vr_mod._dataset,
        "DatasetRecorder",
        lambda **kwargs: pytest.fail("recorder must not open without dataset config"),
    )

    assert session.set_recording(True, task="Pick the red block") is False
    assert "recording config read failed: bad dataset config" in (session._last_error or "")
    assert session._recording is False


def test_b_button_recording_request_does_not_block_when_transition_busy():
    session = vr_mod.VRTeleopSession()
    assert session._recording_transition_lock.acquire(blocking=False)
    try:
        t0 = time.monotonic()
        session._handle_record_button("right")
        elapsed = time.monotonic() - t0
    finally:
        session._recording_transition_lock.release()

    assert elapsed < 0.05
    assert session._recording is False
    assert "recording transition already in progress" in session._recording_notice


def test_native_quest_b_packet_does_not_block_when_transition_busy():
    session = vr_mod.VRTeleopSession()
    assert session._recording_transition_lock.acquire(blocking=False)
    try:
        t0 = time.monotonic()
        out = session.ingest_native_quest_packet({
            "controllers": {
                "right": {
                    "position": [0.0, 1.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "grip": False,
                    "trigger": False,
                    "buttons": {"B": True},
                },
            },
        })
        elapsed = time.monotonic() - t0
    finally:
        session._recording_transition_lock.release()

    assert out == {"ok": True, "goals": 1}
    assert elapsed < 0.05
    assert session._arms["right"].latest.buttons == {"B": True}
    assert session._recording is False
    assert "recording transition already in progress" in session._recording_notice


def test_recording_diagnostics_do_not_suspend_teleop(monkeypatch):
    session = vr_mod.VRTeleopSession()
    session._engaged = True
    session._last_error = "recorder init: camera head did not produce a fresh frame"

    status = session.status()

    assert status["operator"]["stage"] != "suspended"


def test_recording_save_failure_does_not_suspend_teleop():
    session = vr_mod.VRTeleopSession()
    session._engaged = True
    session._last_error = "save_episode failed: disk full"

    status = session.status()

    assert status["operator"]["stage"] != "suspended"


def test_recording_start_refreshes_stale_anchors_and_opens_episode(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    _mark_strict_recording_ready(session, monkeypatch)
    session._invalidate_teleop_anchor("left", "homing changed robot pose")
    monkeypatch.setattr(
        vr_mod._dataset,
        "load_dataset_config",
        lambda: {
            "home_before_episode": False,
            "repo_id": "test/repo",
            "fps": 30,
            "push_to_hub": False,
            "root": None,
            "task_default": "default headset task",
        },
    )
    monkeypatch.setattr(vr_mod._dataset, "resolve_root", lambda root, repo_id: "/tmp/test/repo")
    monkeypatch.setattr(vr_mod._dataset, "DatasetRecorder", lambda **kwargs: recorder)

    assert session.set_recording(True) is True

    status = session.status()
    assert status["recording_armed"] is False
    assert status["recording"] is True
    assert status["last_error"] is None
    assert session._teleop_anchor_fresh("left") is True
    assert recorder.task == "default headset task"


def test_recording_start_blocks_when_controller_pose_missing(monkeypatch):
    session = vr_mod.VRTeleopSession()
    _mark_strict_recording_ready(session, monkeypatch)
    session._invalidate_teleop_anchor("left", "homing changed robot pose")
    session._arms["left"].latest = vr_mod._LatestGoal()
    monkeypatch.setattr(
        vr_mod._dataset,
        "load_dataset_config",
        lambda: {
            "home_before_episode": False,
            "repo_id": "test/repo",
            "fps": 30,
            "push_to_hub": False,
            "root": None,
            "task_default": "default headset task",
        },
    )

    assert session.set_recording(True) is False

    status = session.status()
    assert status["recording_armed"] is False
    assert status["recording"] is False
    assert "left Quest controller pose missing" in status["last_error"]


def test_recording_start_blocks_if_auto_home_does_not_finish(monkeypatch):
    session = vr_mod.VRTeleopSession()
    _mark_strict_recording_ready(session, monkeypatch)
    monkeypatch.setattr(
        vr_mod._dataset,
        "load_dataset_config",
        lambda: {
            "home_before_episode": True,
            "repo_id": "test/repo",
            "fps": 30,
            "push_to_hub": False,
            "root": None,
            "task_default": "default headset task",
        },
    )
    monkeypatch.setattr(session, "go_home", lambda side=None: {})
    monkeypatch.setattr(session, "wait_for_homing", lambda sides, timeout_s=10.0: False)
    monkeypatch.setattr(
        vr_mod._dataset,
        "DatasetRecorder",
        lambda **kwargs: pytest.fail("recorder must not open before homing finishes"),
    )

    assert session.set_recording(True) is False

    assert session._recording is False
    assert "homing did not finish before opening episode" in (session._last_error or "")


def test_recording_start_blocks_if_auto_home_times_out_after_flag_clears(monkeypatch):
    session = vr_mod.VRTeleopSession()
    _mark_strict_recording_ready(session, monkeypatch)
    monkeypatch.setattr(
        vr_mod._dataset,
        "load_dataset_config",
        lambda: {
            "home_before_episode": True,
            "repo_id": "test/repo",
            "fps": 30,
            "push_to_hub": False,
            "root": None,
            "task_default": "default headset task",
        },
    )

    def fake_go_home(side=None):
        with session._lock:
            session._invalidate_teleop_anchor("left", "homing timed out")
            session._invalidate_teleop_anchor("right", "homing changed robot pose")
        return {}

    monkeypatch.setattr(session, "go_home", fake_go_home)
    monkeypatch.setattr(session, "wait_for_homing", lambda sides, timeout_s=10.0: True)
    monkeypatch.setattr(
        vr_mod._dataset,
        "DatasetRecorder",
        lambda **kwargs: pytest.fail("recorder must not open after homing timeout"),
    )

    assert session.set_recording(True) is False

    assert session._recording is False
    assert "auto-home failed" in (session._last_error or "")
    assert "left: homing timed out" in (session._last_error or "")


def test_recording_start_after_auto_home_waits_for_fresh_controller_pose(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    _mark_strict_recording_ready(session, monkeypatch)
    stale_t = time.time() - 10.0
    for side in ("left", "right"):
        session._arms[side].latest.received_at = stale_t

    monkeypatch.setattr(
        vr_mod._dataset,
        "load_dataset_config",
        lambda: {
            "home_before_episode": True,
            "repo_id": "test/repo",
            "fps": 30,
            "push_to_hub": False,
            "root": None,
            "task_default": "default headset task",
        },
    )
    monkeypatch.setattr(vr_mod._dataset, "resolve_root", lambda root, repo_id: "/tmp/test/repo")
    monkeypatch.setattr(vr_mod._dataset, "DatasetRecorder", lambda **kwargs: recorder)
    monkeypatch.setattr(vr_mod, "RECORDING_ANCHOR_INPUT_WAIT_S", 1.0)

    def fake_go_home(side=None):
        with session._lock:
            for arm_side in ("left", "right"):
                session._invalidate_teleop_anchor(arm_side, "homing changed robot pose")
        return {}

    monkeypatch.setattr(session, "go_home", fake_go_home)
    monkeypatch.setattr(session, "wait_for_homing", lambda sides, timeout_s=10.0: True)

    def refresh_controller_poses():
        time.sleep(0.1)
        with session._lock:
            for side in ("left", "right"):
                session._arms[side].latest.received_at = time.time()

    thread = threading.Thread(target=refresh_controller_poses)
    thread.start()

    assert session.set_recording(True) is True
    thread.join(timeout=1.0)

    assert session._recording is True
    assert session._last_error is None
    assert recorder.task == "default headset task"
    assert session._teleop_anchor_fresh("left") is True
    assert session._teleop_anchor_fresh("right") is True


def test_b_button_stop_saves_active_recording(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    recorder.frame_count_in_episode = 12
    session._recording = True
    session._recorder = recorder
    session._last_task = "Pick the red block"

    session._handle_record_button("right")

    assert _wait_until(lambda: session._recording is False and recorder.episode_count == 1)
    assert getattr(recorder, "finalized", False) is True
    assert session._episodes_saved == 1
    assert session._last_error is None


def test_set_recording_true_is_idempotent_while_active(monkeypatch):
    session = vr_mod.VRTeleopSession()
    session._recording = True
    session._last_error = None
    monkeypatch.setattr(
        session,
        "_refresh_recording_anchors_for_start",
        lambda: pytest.fail("active recording must not refresh anchors on idempotent start"),
    )

    assert session.set_recording(True, task="Different task") is True
    assert session._recording is True
    assert session._last_error is None


def test_recording_stop_reports_zero_frame_discard():
    class EmptyRecorder(_FakeRecorder):
        last_end_reason = "episode discarded because no frames were captured"

        def end_episode(self):
            return False

    session = vr_mod.VRTeleopSession()
    recorder = EmptyRecorder()
    session._recording = True
    session._recorder = recorder

    assert session.set_recording(False) is False

    assert session._last_error == "episode discarded because no frames were captured"


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


def test_recording_api_allows_session_default_task(monkeypatch):
    session = vr_mod.VRTeleopSession()
    called = {}
    monkeypatch.setattr(vr_router.vr_mod, "SESSION", session)
    monkeypatch.setattr(
        session,
        "set_recording",
        lambda enabled, task="", root="", source="api": called.update(
            {"enabled": enabled, "task": task, "root": root, "source": source}
        ) or False,
    )
    monkeypatch.setattr(
        session,
        "status",
        lambda: {
            "recording": False,
            "last_error": "task description required before starting an episode",
            "recording_info": {"notice": "task description required before starting an episode"},
        },
    )

    with pytest.raises(vr_router.HTTPException) as exc:
        vr_router.recording({"enabled": True})

    assert exc.value.status_code == 409
    assert "task description required" in str(exc.value.detail)
    assert called == {"enabled": True, "task": "", "root": "", "source": "webapp"}


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
    monkeypatch.setattr(session, "_refresh_recording_anchors_for_start", lambda: [])

    assert session.set_recording(True, task="Pick the red block") is True
    status = session.status()
    assert status["recording_info"]["episodes_saved"] == 2
    assert status["recording_info"]["last_episode_index"] == 1
    assert status["recording_info"]["last_episode_frames"] == 420


def test_recording_start_auto_refreshes_stale_anchors(monkeypatch):
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

    assert session.set_recording(True, task="Pick the red block") is True
    assert session._recording is True
    assert session._recording_armed is False
    assert session._last_error is None
    assert session._recording_notice == ""
    assert session._teleop_anchor_fresh("left") is True
    assert recorder.task == "Pick the red block"


def test_recording_status_marks_anchor_only_blockers_as_start_allowed(monkeypatch):
    session = vr_mod.VRTeleopSession()
    _mark_strict_recording_ready(session, monkeypatch)

    session._invalidate_teleop_anchor("left", "homing changed robot pose")

    status = session.status()
    info = status["recording_info"]

    assert info["calibration_ready"] is False
    assert info["start_allowed"] is True
    assert info["start_blockers"] == []
    assert info["anchor_pending"] is True
    assert info["anchor_blockers"]
    assert info["verification_ready"] is True
    assert info["verification_blockers"] == []
    assert status["operator"]["recording"]["start_allowed"] is True
    assert status["operator"]["recording"]["anchor_pending"] is True


def test_recording_status_blocks_missing_wrist_axes_as_hard_calibration(monkeypatch):
    session = vr_mod.VRTeleopSession()
    _mark_strict_recording_ready(session, monkeypatch)
    session._arms["right"].wrist_roll_canonical = None

    info = session.status()["recording_info"]

    assert info["calibration_ready"] is False
    assert info["start_allowed"] is False
    assert any(
        "right wrist pitch/roll calibration missing" in blocker
        for blocker in info["start_blockers"]
    )
    assert info["anchor_pending"] is False
    assert info["anchor_blockers"] == []


def test_recording_context_records_direct_wrist_mapping_provenance(monkeypatch):
    session = vr_mod.VRTeleopSession()
    _mark_strict_recording_ready(session, monkeypatch)

    meta = session._recording_context_metadata("Pick the red block")
    wrist = meta["arms"]["right"]["wrist_mapping"]

    assert wrist["source"] == "direct_controller_rotation"
    assert wrist["ready"] is True
    assert wrist["pitch_axis"] == [1.0, 0.0, 0.0]
    assert wrist["roll_axis"] == [0.0, 0.0, -1.0]
    assert wrist["motor_polarity"] == vr_mod._WRIST_MOTOR_POLARITY["right"]


def test_recording_status_keeps_verified_state_when_other_gate_blocks(monkeypatch):
    session = vr_mod.VRTeleopSession()
    _mark_strict_recording_ready(session, monkeypatch)
    monkeypatch.setattr(vr_mod._home, "home_pose_status", lambda: {
        "left": {"captured": False, "joints": {}},
        "right": {"captured": True, "joints": {}},
    })

    info = session.status()["recording_info"]

    assert info["start_allowed"] is False
    assert any("left home pose not captured" in blocker for blocker in info["start_blockers"])
    assert info["verification_ready"] is True
    assert info["verification_blockers"] == []


def test_recording_status_classifies_low_scale_test_as_verification_blocker(monkeypatch):
    session = vr_mod.VRTeleopSession()
    _mark_strict_recording_ready(session, monkeypatch)
    session._arms["right"].robot_verify_test_completed = False

    info = session.status()["recording_info"]

    assert info["start_allowed"] is False
    assert any("right low-scale calibration test not completed" in blocker for blocker in info["start_blockers"])
    assert info["verification_ready"] is False
    assert info["verification_blockers"] == ["right low-scale calibration test not completed"]


def test_recording_start_does_not_block_status_while_cameras_warm(monkeypatch):
    session = vr_mod.VRTeleopSession()
    _mark_strict_recording_ready(session, monkeypatch)
    start_entered = threading.Event()
    release_start = threading.Event()
    result = []

    class SlowRecorder(_FakeRecorder):
        repo_id = "test/repo"
        last_saved_episode_index = None
        last_saved_episode_frames = 0

        def start_episode(self, task=""):
            start_entered.set()
            assert release_start.wait(timeout=2.0)
            super().start_episode(task=task)

    monkeypatch.setattr(
        vr_mod._dataset,
        "load_dataset_config",
        lambda: {
            "home_before_episode": False,
            "repo_id": "test/repo",
            "fps": 30,
            "push_to_hub": False,
            "root": None,
            "task_default": "default task",
        },
    )
    monkeypatch.setattr(vr_mod._dataset, "resolve_root", lambda root, repo_id: "/tmp/test/repo")
    monkeypatch.setattr(vr_mod._dataset, "DatasetRecorder", lambda **kwargs: SlowRecorder())

    thread = threading.Thread(
        target=lambda: result.append(session.set_recording(True, task="Pick the red block")),
        daemon=True,
    )
    thread.start()
    assert start_entered.wait(timeout=1.0)

    t0 = time.monotonic()
    status = session.status()
    elapsed = time.monotonic() - t0

    release_start.set()
    thread.join(timeout=2.0)

    assert elapsed < 0.25
    assert status["recording"] is False
    assert result == [True]
    assert session.status()["recording"] is True


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


def test_go_home_clears_live_quest_control_state(monkeypatch):
    session = vr_mod.VRTeleopSession()
    _mark_strict_recording_ready(session, monkeypatch)
    session._engaged = True
    session._dual_mode = True
    session._active_arm = None
    session._arms["left"].pending_rel_position = (0.01, 0.02, 0.03)
    session._arms["right"].pending_rel_position = (-0.01, -0.02, -0.03)

    session.go_home(side=None)

    assert session._engaged is False
    assert session._dual_mode is False
    assert session._active_arm is None
    assert session._arms["left"].homing is True
    assert session._arms["right"].homing is True
    assert session._arms["left"].pending_rel_position == (0.0, 0.0, 0.0)
    assert session._arms["right"].pending_rel_position == (0.0, 0.0, 0.0)


def test_quest_engage_buttons_are_ignored_while_any_arm_is_homing():
    session = vr_mod.VRTeleopSession()
    session._controller_buttons_enabled = True
    session._arms["left"].homing = True

    session._handle_button_edges("left", {"Y": True}, {})
    assert session._engaged is False
    assert session._dual_mode is False
    assert session._active_arm is None

    session._handle_button_edges("right", {"A": True}, {})
    assert session._engaged is False
    assert session._dual_mode is False
    assert session._active_arm is None


def test_strict_frame_failure_aborts_episode(monkeypatch):
    class RejectingRecorder(_FakeRecorder):
        def __init__(self):
            super().__init__()
            self.discarded = False

        def add_frame(self, action, present, camera_frames):
            raise RuntimeError("observation.state missing joints: left_arm_shoulder_pan")

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


def test_isolated_camera_frame_failure_skips_tick_without_discarding(monkeypatch):
    class OneBadCameraFrameRecorder(_FakeRecorder):
        def __init__(self):
            super().__init__()
            self.discarded = False
            self.calls = 0

        @property
        def frame_count_in_episode(self):
            return len(self.frames)

        def add_frame(self, action, present, camera_frames):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError(
                    "camera frames invalid; expected uint8 RGB shape 480x640x3; "
                    "got right_wrist: empty array dtype=uint8 shape=(0,)"
                )
            super().add_frame(action, present, camera_frames)

        def discard_episode(self):
            self.discarded = True
            self.in_episode = False

    recorder = OneBadCameraFrameRecorder()
    session = vr_mod.VRTeleopSession()
    present = {**_joint_values("left", 10), **_joint_values("right", 20)}
    session._recording = True
    session._recorder = recorder
    monkeypatch.setattr(vr_mod, "MOTORS", _StrictMotors(present))
    monkeypatch.setattr(vr_mod._dataset, "grab_camera_frames", lambda: {})
    command = {
        "left": _joint_values("left", 100),
        "right": _joint_values("right", 200),
    }

    session._record_frame_if_active(commanded_this_tick=command)
    assert session._recording is True
    assert recorder.discarded is False
    assert len(recorder.frames) == 0
    assert session._recording_camera_frame_skips == 1
    assert "camera frame skipped" in session._recording_notice

    session._record_frame_if_active(commanded_this_tick=command)
    assert session._recording is True
    assert len(recorder.frames) == 1
    assert session._recording_consecutive_camera_frame_skips == 0


def test_repeated_camera_frame_failures_abort_episode(monkeypatch):
    class BadCameraRecorder(_FakeRecorder):
        def __init__(self):
            super().__init__()
            self.discarded = False

        @property
        def frame_count_in_episode(self):
            return len(self.frames)

        def add_frame(self, action, present, camera_frames):
            raise RuntimeError(
                "camera frames invalid; expected uint8 RGB shape 480x640x3; "
                "got right_wrist: empty array dtype=uint8 shape=(0,)"
            )

        def discard_episode(self):
            self.discarded = True
            self.in_episode = False

    recorder = BadCameraRecorder()
    session = vr_mod.VRTeleopSession()
    present = {**_joint_values("left", 10), **_joint_values("right", 20)}
    session._recording = True
    session._recorder = recorder
    monkeypatch.setattr(vr_mod, "MOTORS", _StrictMotors(present))
    monkeypatch.setattr(vr_mod._dataset, "grab_camera_frames", lambda: {})
    command = {
        "left": _joint_values("left", 100),
        "right": _joint_values("right", 200),
    }

    for _ in range(vr_mod.RECORDING_MAX_CONSECUTIVE_CAMERA_SKIPS + 1):
        session._record_frame_if_active(commanded_this_tick=command)

    assert session._recording is False
    assert recorder.discarded is True
    assert "recording camera unstable" in (session._last_error or "")
