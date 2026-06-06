import numpy as np
import pytest
import yaml

from openpibot.server.runtime import vr_calibration as cal


def test_robot_verification_persists_separately_from_vr_direction(tmp_path, monkeypatch):
    cfg_path = tmp_path / "vr_calibration.yaml"
    monkeypatch.setattr(cal, "CFG_PATH", cfg_path)

    base = np.eye(3)
    translation = 0.5 * np.eye(3)
    cal.write_for_arm(
        "right",
        base,
        forward_motion_m=0.10,
        up_motion_m=0.11,
        left_motion_m=0.09,
        confidence="good",
    )
    cal.write_robot_verification_for_arm(
        "right",
        base_matrix=base,
        verified_matrix=base,
        translation_matrix=translation,
        translation_scale=0.5,
        fit_error_cm=1.2,
        sample_residuals=[{"label": "forward", "residual_cm": 0.4}],
        samples=[{"label": "forward", "vr_delta": [0.0, 0.0, -0.1], "robot_delta": [0.05, 0.0, 0.0]}],
        quality="good",
        low_scale_test_completed=True,
    )

    raw = yaml.safe_load(cfg_path.read_text())
    right = raw["profiles"]["default"]["right"]
    robot = right["robot_verification"]

    assert right["calibration_mode"] == "vr_direction"
    assert right["coordinate_frame"] == "quest_operator_frame"
    assert "translation_scale" not in right
    assert robot["calibration_mode"] == "robot_verified"
    assert robot["coordinate_frame"] == "quest_operator_frame"
    assert robot["translation_scale"] == pytest.approx(0.5)
    assert robot["low_scale_test_completed"] is True
    assert cal.status()["right"]["robot_verified"] is True
    assert cal.status()["right"]["calibration_mode"] == "vr_direction"
    assert cal.translation_scale_for_arm("right") == pytest.approx(0.5)
    np.testing.assert_allclose(cal.matrix_for_arm("right"), base)
    np.testing.assert_allclose(cal.verified_matrix_for_arm("right"), base)


def test_robot_verification_reader_accepts_legacy_flat_entry():
    legacy = {
        "calibration_mode": "robot_verified",
        "teleop_source": "native_quest",
        "session_vr_to_robot": np.eye(3).tolist(),
        "base_vr_direction_matrix": np.eye(3).tolist(),
        "verified_vr_to_robot_matrix": np.eye(3).tolist(),
        "translation_vr_to_robot_matrix": (0.5 * np.eye(3)).tolist(),
        "translation_scale": 0.5,
        "fit_error_cm": 1.1,
        "calibration_quality": "good",
        "verified_at": "2026-06-05T10:00:00",
        "robot_verified_samples": [{"label": "forward"}],
        "robot_verified_sample_residuals": [{"label": "forward", "residual_cm": 0.2}],
    }

    robot = cal.robot_verification_entry(legacy)

    assert robot["calibration_mode"] == "robot_verified"
    assert robot["translation_scale"] == pytest.approx(0.5)
    assert robot["low_scale_test_completed"] is False
    assert robot["robot_verified_samples"] == [{"label": "forward"}]
