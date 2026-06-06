import importlib.util
import pathlib
import subprocess
import sys

import pytest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "infer_pi05_finetuned.py"
SPEC = importlib.util.spec_from_file_location("infer_pi05_finetuned", SCRIPT)
infer = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(infer)


def test_inference_loads_vr_caps_and_deadbands_from_yaml_shape():
    caps, deadbands = infer._load_vr_control_shaping({
        "vr": {
            "joint_deg_caps": {
                "shoulder_pan": 4.0,
                "wrist_roll": 10.0,
            },
            "joint_command_deadband_deg": {
                "shoulder_pan": 0.18,
                "wrist_roll": 0.25,
                "gripper": 0.0,
            },
        },
    })

    assert caps["shoulder_pan"] == pytest.approx(4.0)
    assert caps["wrist_roll"] == pytest.approx(10.0)
    assert deadbands["shoulder_pan"] == pytest.approx(0.18)
    assert deadbands["wrist_roll"] == pytest.approx(0.25)
    assert deadbands["gripper"] == pytest.approx(0.0)


def test_inference_shape_action_uses_configured_wrist_roll_cap():
    key = "right_arm_wrist_roll.pos"
    shaped = infer._shape_action_like_recording(
        {key: 25.0},
        {key: 0.0},
        {key: 0.0},
        kp=1.0,
        caps={"wrist_roll": 10.0},
    )

    assert shaped[key] == pytest.approx(10.0)


def test_inference_final_ema_bypasses_wrist_roll_and_gripper():
    command = {
        "right_arm_shoulder_pan.pos": 10.0,
        "right_arm_wrist_roll.pos": 10.0,
        "right_arm_gripper.pos": 100.0,
    }
    prev = {
        "right_arm_shoulder_pan.pos": 0.0,
        "right_arm_wrist_roll.pos": 0.0,
        "right_arm_gripper.pos": 0.0,
    }

    out = infer._ema_command(
        command,
        prev,
        0.2,
        bypass_suffixes=infer._FINAL_SMOOTHING_BYPASS,
    )

    assert out["right_arm_shoulder_pan.pos"] == pytest.approx(2.0)
    assert out["right_arm_wrist_roll.pos"] == pytest.approx(10.0)
    assert out["right_arm_gripper.pos"] == pytest.approx(100.0)


def test_inference_deadband_is_per_joint():
    command = {
        "right_arm_shoulder_pan.pos": 0.10,
        "right_arm_wrist_roll.pos": 0.30,
    }
    prev = {
        "right_arm_shoulder_pan.pos": 0.0,
        "right_arm_wrist_roll.pos": 0.0,
    }

    out = infer._apply_joint_deadband(
        command,
        prev,
        {
            "shoulder_pan": 0.18,
            "wrist_roll": 0.25,
        },
    )

    assert out["right_arm_shoulder_pan.pos"] == pytest.approx(0.0)
    assert out["right_arm_wrist_roll.pos"] == pytest.approx(0.30)


def test_inference_dry_run_does_not_require_policy_path():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        cwd=str(SCRIPT.parents[1]),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "Policy checkpoint : (not required for --dry-run)" in proc.stdout
    assert "VR joint caps" in proc.stdout
