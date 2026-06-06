import math

import pytest
from scipy.spatial.transform import Rotation

from openpibot.server.runtime.native_quest import (
    MAX_NATIVE_QUEST_PACKET_BYTES,
    NativeQuestAdapter,
    NativeQuestProtocolError,
)
from openpibot.server.runtime import vr_teleop as vr
from openpibot.server.routers import vr as vr_router


def _controller(position, *, grip=False, buttons=None):
    return {
        "position": position,
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "grip": grip,
        "trigger": False,
        "buttons": buttons or {},
    }


def test_native_adapter_emits_reset_position_and_idle_modes():
    adapter = NativeQuestAdapter()

    goals = adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.0], grip=True),
        },
    })
    assert [goal.mode for goal in goals] == ["reset"]
    assert goals[0].arm == "right"
    assert goals[0].relative_position == pytest.approx((0.0, 0.0, 0.0))

    goals = adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.05], grip=True),
        },
    })
    assert [goal.mode for goal in goals] == ["position"]
    assert goals[0].relative_position == pytest.approx((0.05, 0.0, 0.0))

    goals = adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.05], grip=False),
        },
    })
    assert [goal.mode for goal in goals] == ["idle"]
    assert goals[0].relative_position == pytest.approx((0.0, 0.0, 0.0))


def test_native_adapter_forwards_button_releases_for_repeat_edges():
    adapter = NativeQuestAdapter()

    button_states = []
    for pressed in (True, False, True):
        goals = adapter.process_packet({
            "controllers": {
                "right": _controller([0.0, 1.0, 0.0], buttons={"B": pressed}),
            },
        })
        button_states.append(goals[0].buttons)

    assert button_states == [{"B": True}, {"B": False}, {"B": True}]


def test_native_adapter_accepts_flat_controller_packets():
    adapter = NativeQuestAdapter()

    goals = adapter.process_packet({
        "right": _controller([0.1, 1.0, 0.2], grip=True),
        "leftController": _controller([-0.1, 1.0, 0.2], grip=False, buttons={"X": True}),
    })

    assert [(goal.arm, goal.mode) for goal in goals] == [("left", "idle"), ("right", "reset")]
    assert goals[0].buttons == {"X": True}


def test_unity_openxr_frame_maps_forward_to_backend_forward():
    adapter = NativeQuestAdapter(coordinate_frame="unity_openxr")
    adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.0], grip=True),
        },
    })

    goals = adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.05], grip=True),
        },
    })

    assert goals[0].mode == "position"
    assert goals[0].relative_position == pytest.approx((0.0, 0.0, -0.05))


def test_unity_openxr_frame_maps_rotation_into_backend_basis():
    adapter = NativeQuestAdapter(coordinate_frame="unity_openxr")
    unity_yaw = Rotation.from_euler("y", 30, degrees=True).as_quat()

    goals = adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.0], grip=True)
            | {"rotation": list(unity_yaw)},
        },
    })

    backend_rotvec = goals[0].vr_ctrl_rotation.as_rotvec()
    assert backend_rotvec[1] == pytest.approx(-Rotation.from_euler("y", 30, degrees=True).as_rotvec()[1])


def test_quest_operator_frame_matches_reachy_hand_tracker_basis():
    adapter = NativeQuestAdapter(coordinate_frame="quest_operator_frame")
    adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 0.0, 0.0], grip=True),
        },
    })

    goals = adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 0.0, 0.10], grip=True),
        },
    })

    assert goals[0].relative_position == pytest.approx((0.10, 0.0, 0.0))


def test_default_native_frame_matches_shipped_quest_app_forward_axis():
    adapter = NativeQuestAdapter()
    adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 0.0, 0.0], grip=True),
        },
    })

    goals = adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 0.0, 0.10], grip=True),
        },
    })

    assert goals[0].metadata["coordinate_frame"] == "quest_operator_frame"
    assert goals[0].relative_position == pytest.approx((0.10, 0.0, 0.0))


def test_legacy_unity_reachy_alias_maps_to_quest_operator_frame():
    adapter = NativeQuestAdapter(coordinate_frame="unity_reachy")

    assert adapter.coordinate_frame == "quest_operator_frame"


def test_quest_operator_frame_rotation_matches_basis_matrix():
    adapter = NativeQuestAdapter(coordinate_frame="quest_operator_frame")
    unity_rot = Rotation.from_euler("xyz", [10, 20, 30], degrees=True)
    basis = adapter._basis_matrix()

    goals = adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 0.0, 0.0], grip=True)
            | {"rotation": list(unity_rot.as_quat())},
        },
    })

    expected = basis @ unity_rot.as_matrix() @ basis.T
    assert goals[0].vr_ctrl_rotation.as_matrix() == pytest.approx(expected)


@pytest.mark.parametrize("coordinate_frame", ["unity_openxr", "quest_operator_frame"])
@pytest.mark.parametrize(
    ("axis", "angle_deg"),
    [
        ("x", 15),
        ("y", -20),
        ("z", 25),
    ],
)
def test_controller_rotation_basis_conversion_matches_matrix_conjugation(
    coordinate_frame,
    axis,
    angle_deg,
):
    adapter = NativeQuestAdapter(coordinate_frame=coordinate_frame)
    unity_rot = Rotation.from_euler(axis, angle_deg, degrees=True)
    basis = adapter._basis_matrix()

    goals = adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 0.0, 0.0], grip=True)
            | {"rotation": list(unity_rot.as_quat())},
        },
    })

    expected = basis @ unity_rot.as_matrix() @ basis.T
    assert goals[0].vr_ctrl_rotation.as_matrix() == pytest.approx(expected)


def test_native_adapter_accumulates_reset_relative_displacement_after_invalid_packet():
    adapter = NativeQuestAdapter(coordinate_frame="unity_openxr")
    adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.0], grip=True),
        },
    })
    first = adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.01], grip=True),
        },
    })[0]
    invalid = adapter.process_packet({
        "controllers": {
            "right": _controller([99.0, 99.0, 99.0], grip=True, buttons={"A": True})
            | {"valid": False},
        },
    })[0]
    second = adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.03], grip=True),
        },
    })[0]

    assert invalid.mode == "idle"
    assert invalid.buttons == {}
    assert first.relative_position == pytest.approx((0.0, 0.0, -0.01))
    assert second.mode == "position"
    assert second.relative_position == pytest.approx((0.0, 0.0, -0.02))


def test_native_adapter_parses_string_grip_and_trigger_safely():
    adapter = NativeQuestAdapter()

    goals = adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.0], grip=False)
            | {"grip": "false", "trigger": "0.1"},
        },
    })

    assert goals[0].mode == "idle"
    assert goals[0].trigger is False


def test_native_adapter_parses_string_button_values_safely():
    adapter = NativeQuestAdapter()

    goals = adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.0], buttons={"A": "false", "B": "true"}),
        },
    })

    assert goals[0].buttons == {"A": False, "B": True}


def test_native_adapter_invalid_controller_emits_idle_without_pose_update():
    adapter = NativeQuestAdapter()
    adapter.process_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.0], grip=True),
        },
    })

    goals = adapter.process_packet({
        "controllers": {
            "right": {
                "valid": False,
                "position": [10.0, 10.0, 10.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "grip": True,
                "buttons": {"A": "false"},
            },
        },
    })

    assert goals[0].mode == "idle"
    assert goals[0].vr_ctrl_position is None
    assert goals[0].buttons == {}


def test_native_adapter_rejects_malformed_controller_pose():
    adapter = NativeQuestAdapter()

    with pytest.raises(NativeQuestProtocolError):
        adapter.process_packet({
            "controllers": {
                "right": {"position": [0.0, 1.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
            },
        })


def test_session_ingests_native_quest_packet_into_latest_goal():
    session = vr.VRTeleopSession()

    out = session.ingest_native_quest_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.0], grip=True, buttons={"A": False}),
        },
    })

    assert out == {"ok": True, "goals": 1}
    right = session._arms["right"]
    assert right.latest.mode == "reset"
    assert right.reset_pending is True
    assert right.latest.rotation_quat == pytest.approx((0.0, 0.0, 0.0, 1.0))

    session.ingest_native_quest_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.02], grip=True),
        },
    })

    assert right.latest.mode == "position"
    assert right.latest.rel_position == pytest.approx((0.02, 0.0, 0.0))


def test_quest_router_ingest_uses_session_adapter(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr_router.vr_mod, "SESSION", session)

    out = vr_router.quest_ingest(
        {
            "controllers": {
                "left": _controller([0.0, 1.0, 0.0], grip=False, buttons={"X": True}),
            },
        },
        x_quest_pairing_token=session._native_quest_pairing_token,
    )

    assert out == {"ok": True, "goals": 1}
    assert session._arms["left"].latest.buttons == {"X": True}


def test_quest_router_ingest_rejects_missing_pairing_token(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(vr_router.vr_mod, "SESSION", session)

    with pytest.raises(vr_router.HTTPException) as exc:
        vr_router.quest_ingest({"controllers": {}})

    assert exc.value.status_code == 401


def test_session_rejects_oversized_native_quest_packet():
    session = vr.VRTeleopSession()

    with pytest.raises(NativeQuestProtocolError, match="packet exceeds"):
        session.ingest_native_quest_packet({"padding": "x" * MAX_NATIVE_QUEST_PACKET_BYTES})


def test_native_quest_connection_satisfies_operator_readiness_without_legacy_https():
    session = vr.VRTeleopSession()
    session.note_native_quest_client(True)

    operator = session._operator_status(
        arms_status={},
        recording_info={"calibration_blockers": []},
        now=0.0,
    )

    assert "VR HTTPS endpoint is not running" not in operator["ready_blockers"]


def test_quest_bridge_status_tracks_native_clients():
    session = vr.VRTeleopSession()
    session.note_native_quest_client(True)

    out = session.quest_bridge_status()

    assert out["clients"] == 1
    assert out["endpoint"] == "/api/vr/quest/ws"
    assert out["pairing_required"] is True
    assert "pairing_token" not in out
    assert out["ws_url"].endswith("/api/vr/quest/ws")
    assert out["last_seen_ms"] is not None

    session.note_native_quest_client(False)
    assert session.quest_bridge_status()["clients"] == 0


def test_quest_operator_status_is_flat_for_unity_client():
    session = vr.VRTeleopSession()
    session.note_native_quest_client(True)

    out = session.quest_operator_status(public_base_url="http://robot.local:8021/")

    assert out["stage"] in {"connect_required", "mirror_waiting_robot", "mirror_ready"}
    assert out["native_quest_ready"] is True
    assert out["native_quest_clients"] == 1
    assert out["recording_active"] is False
    assert out["recording_ready"] is False
    assert out["calibration_active"] is False
    assert out["calibration_state"] == "idle"
    assert out["ws_url"] == "ws://robot.local:8021/api/vr/quest/ws"


def test_quest_operator_status_exposes_active_calibration_summary():
    session = vr.VRTeleopSession()
    arm = session._arms["left"]
    arm.cal_state = "motioning_fwd"
    arm.cal_motion_acc = (0.03, 0.04, 0.0)
    arm.cal_confidence = "good"

    out = session.quest_operator_status(public_base_url="http://robot.local:8021/")

    assert out["calibration_active"] is True
    assert out["calibration_side"] == "left"
    assert out["calibration_state"] == "motioning_fwd"
    assert out["calibration_motion_m"] == pytest.approx(0.05)
    assert out["calibration_target_m"] == pytest.approx(vr.CALIBRATION_TARGET_MOTION_M)
    assert out["calibration_min_m"] == pytest.approx(vr.CALIBRATION_MIN_MOTION_M)
    assert out["calibration_confidence"] == "good"


def test_quest_operator_status_exposes_robot_verification_summary():
    session = vr.VRTeleopSession()
    arm = session._arms["right"]
    arm.robot_verify_state = "vr_start_captured"
    arm.robot_verify_label = "forward"
    arm.robot_verify_robot_start = (0.0, 0.0, 0.0)
    arm.robot_verify_robot_end = (0.10, 0.0, 0.0)
    arm.robot_verify_vr_start = (0.0, 1.0, 0.0)
    arm.robot_verify_vr_delta_accum = (0.08, 0.0, 0.0)
    arm.latest.has_data = True
    arm.latest.received_at = vr.time.time()
    arm.latest.controller_position = (0.08, 1.0, 0.0)

    out = session.quest_operator_status(public_base_url="http://robot.local:8021/")

    assert out["robot_verification_active"] is True
    assert out["robot_verification_side"] == "right"
    assert out["robot_verification_label"] == "forward"
    assert out["robot_verification_controls"] == "Keep grip held; press A to capture VR end"
    assert out["robot_verification_target_motion_m"] == pytest.approx(0.10)
    assert out["robot_verification_vr_motion_m"] == pytest.approx(0.08)
    assert out["robot_verification_target_robot_delta"] == pytest.approx([0.10, 0.0, 0.0])
    predicted = out["robot_verification_predicted_robot_delta"]
    assert len(predicted) == 3
    assert math.sqrt(sum(v * v for v in predicted)) == pytest.approx(0.08)
    assert out["robot_verification_vr_delta"] == pytest.approx([0.08, 0.0, 0.0])


def test_quest_a_button_captures_robot_verification_vr_start_and_end():
    session = vr.VRTeleopSession()
    arm = session._arms["right"]
    arm.robot_verify_state = "robot_end_captured"
    arm.robot_verify_robot_start = (0.0, 0.0, 0.0)
    arm.robot_verify_robot_end = (0.10, 0.0, 0.0)
    arm.robot_verify_label = "forward"

    session.ingest_native_quest_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.0], grip=True, buttons={"A": True}),
        },
    })

    assert arm.robot_verify_state == "vr_start_captured"
    assert arm.robot_verify_vr_start == pytest.approx((0.0, -0.0, 1.0))

    session.ingest_native_quest_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.0], grip=True, buttons={"A": False}),
        },
    })
    session.ingest_native_quest_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.10], grip=True, buttons={"A": False}),
        },
    })
    session.ingest_native_quest_packet({
        "controllers": {
            "right": _controller([0.0, 1.0, 0.10], grip=True, buttons={"A": True}),
        },
    })

    assert arm.robot_verify_state == "collecting"
    assert len(arm.robot_verify_samples) == 1
    assert arm.robot_verify_samples[0]["label"] == "forward"
    assert arm.robot_verify_samples[0]["vr_delta"] == pytest.approx([0.10, 0.0, 0.0])


def test_quest_operator_status_does_not_read_motor_positions(monkeypatch):
    session = vr.VRTeleopSession()

    def fail_read_positions(*_args, **_kwargs):
        raise AssertionError("quest operator status must not read motor positions")

    monkeypatch.setattr(vr.MOTORS, "read_positions", fail_read_positions)

    out = session.quest_operator_status(public_base_url="http://robot.local:8021/")

    assert "stage" in out


def test_quest_operator_status_includes_video_summary(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setattr(
        vr._quest_video_bridge,
        "bridge_status",
        lambda: {
            "ready": True,
            "running": True,
            "transport": "gstreamer-rtp-h264",
            "running_roles": ["head"],
            "receive_health": {"head": {"state": "receiving"}},
        },
    )

    out = session.quest_operator_status(public_base_url="http://robot.local:8021/")

    assert out["video_ready"] is True
    assert out["video_running"] is True
    assert out["video_transport"] == "gstreamer-rtp-h264"
    assert out["video_base_port"] == 5600
    assert out["video_bitrate_kbps"] == 0
    assert out["video_running_roles"] == ["head"]
    assert out["video_receive_health"]["head"]["state"] == "receiving"


def test_quest_ws_url_uses_environment_fallback(monkeypatch):
    session = vr.VRTeleopSession()
    monkeypatch.setenv("OPENPIBOT_PUBLIC_HOST", "xlerobot.local")
    monkeypatch.setenv("OPENPIBOT_PORT", "8031")
    monkeypatch.setenv("OPENPIBOT_TLS", "true")

    assert session.quest_bridge_status()["ws_url"] == "wss://xlerobot.local:8031/api/vr/quest/ws"
