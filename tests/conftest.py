import pytest


@pytest.fixture(autouse=True)
def isolate_vr_calibration_file(tmp_path, monkeypatch):
    from openpibot.server.runtime import vr_calibration

    monkeypatch.setattr(vr_calibration, "CFG_PATH", tmp_path / "vr_calibration.yaml")


@pytest.fixture(autouse=True)
def isolate_lerobot_dataset_cache(tmp_path, monkeypatch):
    """Keep tests from touching the operator's real LeRobot cache."""
    from pathlib import Path

    from openpibot.server.runtime import dataset

    test_lerobot_home = tmp_path / "lerobot"
    real_lerobot_home = Path.home() / ".cache" / "huggingface" / "lerobot"
    original_resolve_root = dataset.resolve_root

    monkeypatch.setenv("HF_LEROBOT_HOME", str(test_lerobot_home))

    def resolve_root_in_test_cache(root, repo_id):
        resolved = Path(original_resolve_root(root, repo_id))
        try:
            resolved.relative_to(real_lerobot_home)
        except ValueError:
            return str(resolved)
        return str(test_lerobot_home / str(repo_id))

    monkeypatch.setattr(dataset, "resolve_root", resolve_root_in_test_cache)
