import pytest

from openpibot.server.runtime import motors as motors_mod


def test_arm_session_send_action_returns_so_follower_safe_command():
    class FakeFollower:
        def send_action(self, action):
            assert action == {
                "shoulder_pan.pos": 12.0,
                "elbow_flex.pos": 30.0,
            }
            return {
                "shoulder_pan.pos": 10.0,
                "elbow_flex.pos": 28.0,
            }

    session = motors_mod._ArmSession("left")
    session._arm = FakeFollower()

    sent = session.send_action({
        "left_arm_shoulder_pan": 12.0,
        "left_arm_elbow_flex": 30.0,
        "right_arm_elbow_flex": 99.0,
    })

    assert sent == {
        "left_arm_shoulder_pan": 10.0,
        "left_arm_elbow_flex": 28.0,
    }


def test_arm_session_send_action_requires_reported_sent_goal():
    class FakeFollower:
        def send_action(self, action):
            return {"shoulder_pan.pos": action["shoulder_pan.pos"]}

    session = motors_mod._ArmSession("left")
    session._arm = FakeFollower()

    with pytest.raises(RuntimeError, match="did not report sent goals"):
        session.send_action({
            "left_arm_shoulder_pan": 12.0,
            "left_arm_elbow_flex": 30.0,
        })
