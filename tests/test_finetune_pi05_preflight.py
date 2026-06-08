import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "finetune_pi05.py"
SPEC = importlib.util.spec_from_file_location("finetune_pi05", SCRIPT)
finetune = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(finetune)


def _fake_dataset_with_wrist_roll_step(delta: float, state_delta: float = 0.0):
    names = [f"{name}.pos" for name in finetune.XLEROBOT_JOINT_ORDER]
    action0 = np.zeros(len(names), dtype=np.float32)
    action1 = np.zeros(len(names), dtype=np.float32)
    state0 = np.zeros(len(names), dtype=np.float32)
    state1 = np.zeros(len(names), dtype=np.float32)
    joint_idx = names.index("right_arm_wrist_roll.pos")
    action1[joint_idx] = delta
    state1[joint_idx] = state_delta
    return SimpleNamespace(
        hf_dataset={
            "action": np.stack([action0, action1]),
            "observation.state": np.stack([state0, state1]),
            "episode_index": np.array([0, 0], dtype=np.int64),
        },
        meta=SimpleNamespace(
            features={
                "action": {
                    "names": names,
                },
            },
        ),
    )


def test_finetune_preflight_loads_recording_joint_caps_from_config():
    caps = finetune._load_recording_joint_caps({
        "vr": {
            "joint_deg_caps": {
                "wrist_roll": 10.0,
            },
        },
    })

    assert caps["wrist_roll"] == pytest.approx(10.0)


def test_finetune_preflight_accepts_configured_wrist_roll_cap():
    ds = _fake_dataset_with_wrist_roll_step(10.0)
    caps = finetune._load_recording_joint_caps({
        "vr": {
            "joint_deg_caps": {
                "wrist_roll": 10.0,
            },
        },
    })

    finetune._preflight_action_continuity(ds, caps)


def test_finetune_preflight_rejects_wrist_roll_step_above_default_cap():
    ds = _fake_dataset_with_wrist_roll_step(10.0)

    with pytest.raises(SystemExit):
        finetune._preflight_action_continuity(ds, finetune.RECORDING_PER_TICK_DEG_CAPS)


def test_finetune_preflight_accepts_sofollower_safety_clipped_action_jump():
    names = [f"{name}.pos" for name in finetune.XLEROBOT_JOINT_ORDER]
    joint_idx = names.index("right_arm_wrist_flex.pos")
    action0 = np.zeros(len(names), dtype=np.float32)
    action1 = np.zeros(len(names), dtype=np.float32)
    state0 = np.zeros(len(names), dtype=np.float32)
    state1 = np.zeros(len(names), dtype=np.float32)
    action0[joint_idx] = 63.560
    state0[joint_idx] = 55.648
    action1[joint_idx] = 72.022
    state1[joint_idx] = 65.934
    ds = SimpleNamespace(
        hf_dataset={
            "action": np.stack([action0, action1]),
            "observation.state": np.stack([state0, state1]),
            "episode_index": np.array([0, 0], dtype=np.int64),
        },
        meta=SimpleNamespace(features={"action": {"names": names}}),
    )

    finetune._preflight_action_continuity(
        ds,
        finetune.RECORDING_PER_TICK_DEG_CAPS,
        max_relative_target=8.0,
    )


def test_finetune_preflight_rejects_state_fallback_jump():
    ds = _fake_dataset_with_wrist_roll_step(10.0, state_delta=10.0)

    with pytest.raises(SystemExit):
        finetune._preflight_action_continuity(
            ds,
            finetune.RECORDING_PER_TICK_DEG_CAPS,
            max_relative_target=8.0,
        )
