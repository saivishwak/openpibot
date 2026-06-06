import yaml

from openpibot.server.runtime import cameras


def test_enumerate_cameras_lists_stale_config_and_all_video_candidates(tmp_path, monkeypatch):
    cfg_path = tmp_path / "xlerobot.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "cameras": {
                "right_wrist": {
                    "type": "opencv",
                    "path": "/missing/right-wrist",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "fourcc": "MJPG",
                }
            }
        })
    )
    video0 = tmp_path / "video0"
    video1 = tmp_path / "video1"
    video0.touch()
    video1.touch()

    monkeypatch.setattr(cameras, "CONFIG_YAML", cfg_path)
    monkeypatch.setattr(cameras, "_video_device_candidates", lambda: [str(video0), str(video1)])
    monkeypatch.setattr(cameras, "_canonical_video_node", lambda path: str(path))

    def fake_query(path):
        if path == str(video0):
            return True, "USB Camera"
        return None

    monkeypatch.setattr(cameras, "_query_v4l2_cap", fake_query)

    out = cameras.enumerate_cameras()

    assert out[0].name == "right_wrist"
    assert out[0].available is False
    assert out[0].role == "right_wrist"
    raw = out[1:]
    assert [cam.path for cam in raw] == [str(video0), str(video1)]
    assert [cam.by_path for cam in raw] == [str(video0), str(video1)]
    assert raw[0].capture is True
    assert raw[0].card == "USB Camera"
    assert raw[1].capture is None


def test_assign_role_accepts_raw_video_node_when_no_by_path_exists(tmp_path, monkeypatch):
    cfg_path = tmp_path / "xlerobot.yaml"
    cfg_path.write_text(yaml.safe_dump({"cameras": {}}))
    raw_node = tmp_path / "video2"
    raw_node.touch()

    monkeypatch.setattr(cameras, "CONFIG_YAML", cfg_path)

    cameras.assign_role(str(raw_node), "right_wrist")

    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["cameras"]["right_wrist"]["path"] == str(raw_node)
    assert cfg["cameras"]["right_wrist"]["type"] == "opencv"


def test_opencv_capture_path_resolves_stable_symlink_to_video_node(monkeypatch):
    monkeypatch.setattr(
        cameras,
        "_canonical_video_node",
        lambda path: "/dev/video4" if path == "/dev/v4l/by-path/right-wrist" else path,
    )

    assert cameras._opencv_capture_path("/dev/v4l/by-path/right-wrist") == "/dev/video4"
