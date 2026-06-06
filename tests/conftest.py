import pytest


@pytest.fixture(autouse=True)
def isolate_vr_calibration_file(tmp_path, monkeypatch):
    from openpibot.server.runtime import vr_calibration

    monkeypatch.setattr(vr_calibration, "CFG_PATH", tmp_path / "vr_calibration.yaml")
