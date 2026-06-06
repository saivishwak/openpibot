"""LeRobot v2 dataset recording for VR teleop.

Wraps `lerobot.datasets.LeRobotDataset.create()` for write-mode capture during
VR sessions. Each frame contains:
  - `action`              : commanded joint positions for both arms (12-vector)
  - `observation.state`   : present joint positions for both arms (12-vector)
  - `observation.images.<role>` : (H, W, 3) RGB frame from each configured camera

Episode boundary control:
  - `start_episode(task)` is called on the rising edge of either the B button
    on the right Quest controller or the UI "Start recording" toggle.
  - `add_frame(...)` is called every drive-loop tick while in an episode.
  - `end_episode()` is called on the next falling edge.
  - `finalize()` flushes everything to disk and (optionally) pushes to the Hub.

Camera frames come from `openpibot.server.runtime.cameras.get_stream(role).snapshot()`
which returns JPEG bytes — we decode once per frame to keep CameraStream's
existing single-frame producer architecture untouched.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import threading
import time
from typing import Any, Optional

import cv2  # decode JPEG → numpy (BGR)
import numpy as np
import yaml

from openpibot.server.config import REPO_ROOT

from . import cameras as cam_mod
from .motors import JOINTS_PER_ARM

log = logging.getLogger(__name__)

# Order must match what the drive loop writes into `action` / `observation.state`.
# All 12 joints (6 per arm x 2) are required for strict dataset recording.
JOINT_ORDER: list[str] = [
    f"{side}_arm_{j}" for side in ("left", "right") for j in JOINTS_PER_ARM
]
REQUIRED_CAMERA_ROLES: tuple[str, ...] = ("head", "left_wrist", "right_wrist")


class DatasetRecorder:
    """LeRobotDataset writer wrapper. One instance lives inside `VRTeleopSession`.

    Thread model: instantiation + start/end calls happen on whichever thread
    flips the recording flag (drive loop, for the B button; API handler thread
    for the UI button). `add_frame` is only called from the drive loop. Internal
    state mutations are guarded by `self._lock` since they can race.
    """

    def __init__(self, repo_id: str, fps: int,
                 camera_roles: list[str], camera_shape: tuple[int, int, int],
                 root: Optional[pathlib.Path] = None,
                 push_to_hub: bool = False) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._lock = threading.Lock()
        # Serializes writes/finalize calls on the underlying LeRobotDataset.
        # The VR drive loop can be preparing a frame while a UI/B-button stop
        # ends the episode; this prevents add_frame/save/finalize interleaving.
        self._io_lock = threading.Lock()
        self.repo_id = repo_id
        self.fps = int(fps)
        self.camera_roles = list(camera_roles)
        self.camera_shape = tuple(camera_shape)
        self.push_to_hub_flag = bool(push_to_hub)
        self._in_episode = False
        self._episode_count = 0
        self._frame_count = 0
        self._current_task: str = ""
        self._last_saved_episode_index: Optional[int] = None
        self._last_saved_episode_frames: int = 0

        features = self._build_features(JOINT_ORDER, self.camera_roles, self.camera_shape)
        resolved_root = pathlib.Path(resolve_root(str(root) if root else None, repo_id))
        has_info = (resolved_root / "meta" / "info.json").is_file()
        has_episode_meta = any((resolved_root / "meta" / "episodes").glob("*/*.parquet"))
        if has_info and has_episode_meta:
            self._dataset = LeRobotDataset.resume(
                repo_id=repo_id,
                root=str(resolved_root),
                revision="main",
                image_writer_threads=2,
            )
        else:
            needs_create = True
            if resolved_root.exists():
                if has_info and not has_episode_meta:
                    info_path = resolved_root / "meta" / "info.json"
                    try:
                        info = json.loads(info_path.read_text())
                    except Exception as e:
                        raise RuntimeError(
                            "existing dataset root is not finalized/readable "
                            f"(missing meta/episodes parquet and unreadable info.json): {resolved_root}. "
                            "Move that directory aside or choose a new recording root."
                        ) from e
                    total_eps = int(info.get("total_episodes", -1))
                    total_frames = int(info.get("total_frames", -1))
                    if total_eps == 0 and total_frames == 0:
                        # Valid empty dataset after deleting the last episode:
                        # rebuild a clean writable root from info.json.
                        backup = resolved_root.parent / f".{resolved_root.name}.empty-backup-{int(time.time()*1000)}"
                        os.replace(resolved_root, backup)
                        try:
                            self._dataset = LeRobotDataset.create(
                                repo_id=repo_id,
                                fps=int(info["fps"]),
                                features=info["features"],
                                root=str(resolved_root),
                                robot_type=str(info.get("robot_type") or "xlerobot-bimanual-so101"),
                                use_videos=any(
                                    isinstance(v, dict) and v.get("dtype") == "video"
                                    for v in (info.get("features") or {}).values()
                                ),
                                image_writer_threads=2,
                            )
                            self._dataset.finalize()
                            shutil.rmtree(backup, ignore_errors=True)
                            needs_create = False
                        except Exception:
                            if not resolved_root.exists() and backup.exists():
                                os.replace(backup, resolved_root)
                            raise
                    else:
                        raise RuntimeError(
                            "existing dataset root is not finalized/readable "
                            f"(missing meta/episodes parquet): {resolved_root}. "
                            "Move that directory aside or choose a new recording root."
                        )
                else:
                    raise RuntimeError(f"dataset root exists but is not a LeRobot dataset: {resolved_root}")
            if needs_create:
                self._dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=self.fps,
                    features=features,
                    root=str(resolved_root),
                    robot_type="xlerobot-bimanual-so101",
                    use_videos=True,                     # MP4-encode image streams
                    image_writer_threads=2,              # async JPEG → video encode
                )
        self._episode_count = int(getattr(self._dataset.meta, "total_episodes", 0))
        if self._episode_count > 0:
            try:
                last_ep = self._dataset.meta.episodes[self._episode_count - 1]
                self._last_saved_episode_index = self._episode_count - 1
                self._last_saved_episode_frames = int(last_ep.get("length", 0))
            except Exception:
                self._last_saved_episode_index = self._episode_count - 1
                self._last_saved_episode_frames = 0
        log.info("dataset recorder ready: repo_id=%s fps=%d cameras=%s root=%s",
                 repo_id, self.fps, self.camera_roles, self._dataset.root)

    @staticmethod
    def _build_features(joint_order: list[str], camera_roles: list[str],
                         camera_shape: tuple[int, int, int]) -> dict:
        """Construct the LeRobot v2 features dict. The `names` field is required
        for state/action features; for video it's ['height','width','channels']."""
        nj = len(joint_order)
        feats: dict[str, dict[str, Any]] = {
            "action": {
                "dtype": "float32",
                "shape": (nj,),
                "names": list(joint_order),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (nj,),
                "names": list(joint_order),
            },
        }
        for role in camera_roles:
            feats[f"observation.images.{role}"] = {
                "dtype": "video",
                "shape": tuple(camera_shape),         # (H, W, 3)
                "names": ["height", "width", "channels"],
            }
        return feats

    # ── episode lifecycle ───────────────────────────────────────────────────
    @property
    def in_episode(self) -> bool:
        with self._lock:
            return self._in_episode

    @property
    def episode_count(self) -> int:
        with self._lock:
            return self._episode_count

    @property
    def frame_count_in_episode(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def last_saved_episode_index(self) -> Optional[int]:
        with self._lock:
            return self._last_saved_episode_index

    @property
    def last_saved_episode_frames(self) -> int:
        with self._lock:
            return self._last_saved_episode_frames

    def start_episode(self, task: str = "") -> None:
        with self._lock:
            if self._in_episode:
                log.warning("start_episode called while already in episode; ignoring")
                return
            self._in_episode = True
            self._frame_count = 0
            self._current_task = task or "bimanual-vr-teleop"
            log.info("episode %d started (task=%r)", self._episode_count + 1, self._current_task)

    def end_episode(self) -> bool:
        """Save the current episode buffer to disk. Returns True if saved, False
        if there was no active episode or no frames."""
        with self._lock:
            if not self._in_episode:
                log.warning("end_episode called while not in an episode; ignoring")
                return False
            had_frames = self._frame_count > 0
            self._in_episode = False
            if not had_frames:
                # Discard empty buffer rather than save a 0-frame episode.
                try: self._dataset.clear_episode_buffer()
                except Exception as e: log.warning("clear_episode_buffer: %s", e)
                log.info("episode discarded (0 frames)")
                return False
            episode_count = self._episode_count + 1
            frame_count = self._frame_count
        # Save outside the lock — save_episode does I/O + may take a while
        # (especially with batch_encoding_size=1, the video gets encoded now).
        try:
            with self._io_lock:
                self._dataset.save_episode()
            with self._lock:
                self._episode_count = episode_count
                self._last_saved_episode_index = episode_count - 1
                self._last_saved_episode_frames = frame_count
            log.info("episode %d saved (%d frames)",
                     episode_count, frame_count)
            return True
        except Exception as e:
            log.exception("save_episode failed: %s", e)
            return False

    def add_frame(self, action: dict[str, float],
                   present: dict[str, float],
                   camera_frames: dict[str, Optional[np.ndarray]]) -> None:
        """Append one frame to the current episode buffer. Caller must already
        have called `start_episode`.

        Dataset-quality recording is strict: all joints and camera frames must be
        present for every frame. Missing hardware data aborts the caller's active
        recording instead of writing zero-filled or partial training examples."""
        with self._lock:
            if not self._in_episode:
                return
            self._validate_joint_dict("action", action)
            self._validate_joint_dict("observation.state", present)
            self._validate_camera_frames(camera_frames)
            action_vec  = self._joint_dict_to_array(action)
            present_vec = self._joint_dict_to_array(present)

            frame: dict[str, Any] = {
                "task": self._current_task,
                "action": action_vec,
                "observation.state": present_vec,
            }
            for role in self.camera_roles:
                img = camera_frames.get(role)
                frame[f"observation.images.{role}"] = img

            with self._io_lock:
                self._dataset.add_frame(frame)
            self._frame_count += 1

    @staticmethod
    def _validate_joint_dict(label: str, joints: dict[str, float]) -> None:
        missing = [k for k in JOINT_ORDER if k not in joints]
        if missing:
            raise RuntimeError(f"{label} missing joints: {', '.join(missing)}")
        bad: list[str] = []
        for key in JOINT_ORDER:
            try:
                value = float(joints[key])
            except (TypeError, ValueError):
                bad.append(key)
                continue
            if not np.isfinite(value):
                bad.append(key)
        if bad:
            raise RuntimeError(f"{label} has non-finite joints: {', '.join(bad)}")

    def _validate_camera_frames(self, camera_frames: dict[str, Optional[np.ndarray]]) -> None:
        missing = [role for role in self.camera_roles if role not in camera_frames]
        if missing:
            raise RuntimeError(f"camera frames missing roles: {', '.join(missing)}")
        bad: list[str] = []
        for role in self.camera_roles:
            img = camera_frames.get(role)
            if not isinstance(img, np.ndarray):
                bad.append(role)
                continue
            if img.shape != self.camera_shape or img.dtype != np.uint8:
                bad.append(role)
        if bad:
            shape = "x".join(str(v) for v in self.camera_shape)
            raise RuntimeError(f"camera frames invalid for roles {', '.join(bad)}; expected uint8 RGB shape {shape}")

    @staticmethod
    def _joint_dict_to_array(joints: dict[str, float]) -> np.ndarray:
        """Project a validated complete joint dict onto the fixed JOINT_ORDER."""
        return np.array(
            [float(joints.get(k, 0.0)) for k in JOINT_ORDER],
            dtype=np.float32,
        )

    # ── teardown ────────────────────────────────────────────────────────────
    def finalize(self) -> None:
        """Flush all pending state to disk. If `push_to_hub_flag` is set, push
        the dataset after finalize. Safe to call multiple times."""
        with self._lock:
            if self._in_episode and self._frame_count > 0:
                # Save the in-flight episode before finalizing.
                self._in_episode = False
                try:
                    with self._io_lock:
                        self._dataset.save_episode()
                    self._episode_count += 1
                except Exception as e:
                    log.warning("finalize: save_episode failed: %s", e)
        try:
            with self._io_lock:
                self._dataset.finalize()
            log.info("dataset finalized; %d episode(s) at %s",
                     self._episode_count, self._dataset.root)
        except Exception as e:
            log.warning("finalize failed: %s", e)
        if self.push_to_hub_flag:
            try:
                self._dataset.push_to_hub()
                log.info("pushed dataset %s to Hub", self.repo_id)
            except Exception as e:
                log.warning("push_to_hub failed: %s", e)

    def discard_episode(self) -> None:
        """Drop the current in-memory episode after a strict recording failure."""
        with self._lock:
            was_in_episode = self._in_episode
            self._in_episode = False
            self._frame_count = 0
        if was_in_episode:
            try:
                with self._io_lock:
                    self._dataset.clear_episode_buffer()
            except Exception as e:
                log.warning("discard_episode: clear_episode_buffer failed: %s", e)

    def write_recording_context(self, metadata: dict[str, Any]) -> None:
        """Append an audit record for a recording episode without changing features."""
        try:
            root = pathlib.Path(self._dataset.root)
            path = root / "meta" / "recording_context.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metadata, sort_keys=True) + "\n")
        except Exception as e:
            log.warning("write_recording_context failed: %s", e)


# ─── helpers used by VRTeleopSession to build the recorder ─────────────────

def grab_camera_frames() -> dict[str, Optional[np.ndarray]]:
    """Snapshot every role-assigned camera. Returns {role: RGB-ndarray-or-None}.

    Uses the existing CameraStream singletons so we don't open a second
    VideoCapture per camera. Decodes the latest JPEG to numpy via cv2.imdecode,
    then converts OpenCV's BGR layout to RGB for LeRobot/PIL video encoding.
    Subscriptions are reference-counted; we acquire while reading and release
    afterwards (matches how `/camera/<id>/snapshot` works)."""
    out: dict[str, Optional[np.ndarray]] = {}
    cams = cam_mod.enumerate_cameras()
    for c in cams:
        if not c.role:
            continue
        stream = cam_mod.get_stream(c.role)
        if stream is None:
            out[c.role] = None
            continue
        stream.acquire()
        try:
            # Prefer raw RGB from the capture thread (matches LeRobot OpenCVCamera path).
            img = stream.last_rgb
            if img is None:
                img = stream.get_rgb(timeout=0.5)
            if img is None:
                jpeg = stream.last_jpeg
                if jpeg is None:
                    out[c.role] = None
                    continue
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out[c.role] = img
        finally:
            stream.release()
    return out


def load_dataset_config() -> dict[str, Any]:
    """Read the `dataset:` block from config/xlerobot.yaml. Returns sensible
    defaults if the section is missing."""
    defaults = {
        "repo_id": "xlerobot/vr-teleop",
        "fps": 30,
        "push_to_hub": False,
        "task_default": "bimanual-vr-teleop",
        "camera_height": 480,
        "camera_width":  640,
        # If True, every recorded episode begins with all connected arms
        # interpolating to the saved home pose first. Required for VLA data
        # collection so all episodes start from the same proprioception state.
        "home_before_episode": False,
        # Absolute filesystem path where episodes are written. Null/missing
        # → use HuggingFace's default (`$HF_LEROBOT_HOME` or
        # `~/.cache/huggingface/lerobot/<repo_id>/`). Set this in YAML or via
        # the Recording card's "Storage path" input to write elsewhere
        # (e.g., a big external drive).
        "root": None,
    }
    try:
        cfg = yaml.safe_load((REPO_ROOT / "config" / "xlerobot.yaml").read_text()) or {}
        ds = cfg.get("dataset") or {}
        defaults.update({k: v for k, v in ds.items() if v is not None})
    except Exception as e:
        log.warning("could not read dataset config: %s; using defaults", e)
    return defaults


def write_dataset_config(*, root: str | None = None, repo_id: str | None = None) -> dict[str, Any]:
    """Persist dataset recording config in config/xlerobot.yaml and return it."""
    cfg_path = REPO_ROOT / "config" / "xlerobot.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.is_file() else {}
    if not isinstance(cfg, dict):
        cfg = {}
    dataset_cfg = cfg.setdefault("dataset", {})
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
        cfg["dataset"] = dataset_cfg
    if root is not None:
        cleaned_root = str(root or "").strip()
        dataset_cfg["root"] = cleaned_root or None
    if repo_id is not None:
        cleaned_repo_id = str(repo_id or "").strip()
        if not cleaned_repo_id:
            raise ValueError("dataset repo_id cannot be empty")
        if any(ch.isspace() for ch in cleaned_repo_id) or cleaned_repo_id.startswith("/") or cleaned_repo_id.endswith("/"):
            raise ValueError("dataset repo_id must be a Hugging Face-style id like 'user/dataset-name'")
        dataset_cfg["repo_id"] = cleaned_repo_id
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return load_dataset_config()


def write_dataset_root(root: str) -> dict[str, Any]:
    """Persist `dataset.root` in config/xlerobot.yaml and return the dataset config."""
    return write_dataset_config(root=root)


def resolve_root(root: Optional[str], repo_id: str) -> str:
    """Resolve the dataset root path. Handles ~ expansion and falls back to
    the HF default (`$HF_LEROBOT_HOME/<repo_id>` or
    `~/.cache/huggingface/lerobot/<repo_id>`) when None/empty."""
    import os
    if root:
        return os.path.abspath(os.path.expanduser(str(root)))
    hf_home = os.environ.get("HF_LEROBOT_HOME")
    if hf_home:
        return os.path.abspath(os.path.expanduser(os.path.join(hf_home, repo_id)))
    return os.path.abspath(os.path.expanduser(
        f"~/.cache/huggingface/lerobot/{repo_id}"
    ))


def role_camera_list() -> tuple[list[str], tuple[int, int, int]]:
    """Return the list of camera roles to record + a single shared (H, W, 3)
    shape that all roles must conform to."""
    cfg = load_dataset_config()
    roles: list[str] = []
    for c in cam_mod.enumerate_cameras():
        if c.role:
            roles.append(c.role)
    shape = (int(cfg["camera_height"]), int(cfg["camera_width"]), 3)
    return roles, shape


def delete_last_episode(repo_id: str, root: Optional[str]) -> tuple[int, str]:
    """Delete the most recently saved episode in-place.

    Returns:
        (new_total_episodes, resolved_root_path)
    """
    from lerobot.datasets.dataset_tools import delete_episodes as _delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    resolved_root = pathlib.Path(resolve_root(root, repo_id))
    if not (resolved_root / "meta" / "info.json").is_file():
        raise RuntimeError(f"dataset not found at {resolved_root}")

    dataset = LeRobotDataset(repo_id=repo_id, root=resolved_root)
    total = int(dataset.meta.total_episodes)
    if total <= 0:
        raise RuntimeError("no saved episodes to delete")

    stamp = int(time.time() * 1000)
    backup_root = resolved_root.parent / f".{resolved_root.name}.backup-delete-{stamp}"
    if backup_root.exists():
        raise RuntimeError(f"backup path already exists: {backup_root}")

    # Special case: deleting the sole episode should leave a valid empty dataset.
    if total == 1:
        os.replace(resolved_root, backup_root)
        try:
            ds_new = LeRobotDataset.create(
                repo_id=repo_id,
                fps=int(dataset.meta.fps),
                features=dataset.meta.features,
                root=str(resolved_root),
                robot_type=str(dataset.meta.robot_type),
                use_videos=bool(dataset.meta.video_keys),
                image_writer_threads=2,
            )
            ds_new.finalize()
            shutil.rmtree(backup_root, ignore_errors=True)
            return 0, str(resolved_root)
        except Exception:
            if not resolved_root.exists() and backup_root.exists():
                os.replace(backup_root, resolved_root)
            raise

    # General case: materialize edited dataset in a temp sibling, then atomically swap.
    tmp_root = resolved_root.parent / f".{resolved_root.name}.tmp-delete-{stamp}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)

    _delete_episodes(
        dataset,
        [total - 1],
        output_dir=tmp_root,
        repo_id=repo_id,
    )

    os.replace(resolved_root, backup_root)
    try:
        os.replace(tmp_root, resolved_root)
        # Guardrail: ensure info.json reflects the deletion before committing.
        post = LeRobotDataset(repo_id=repo_id, root=resolved_root)
        if int(post.meta.total_episodes) != (total - 1):
            raise RuntimeError(
                "delete verification failed: info.json total_episodes did not update "
                f"(expected {total - 1}, got {post.meta.total_episodes})"
            )
        shutil.rmtree(backup_root, ignore_errors=True)
    except Exception:
        if not resolved_root.exists() and backup_root.exists():
            os.replace(backup_root, resolved_root)
        elif backup_root.exists():
            # If swap succeeded but verification failed, restore original dataset.
            shutil.rmtree(resolved_root, ignore_errors=True)
            os.replace(backup_root, resolved_root)
        raise
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

    return total - 1, str(resolved_root)


def last_episode_summary(repo_id: str, root: Optional[str]) -> tuple[Optional[int], int]:
    """Return (last_episode_index, last_episode_frames) for a dataset root."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    resolved_root = pathlib.Path(resolve_root(root, repo_id))
    if not (resolved_root / "meta" / "info.json").is_file():
        return None, 0
    dataset = LeRobotDataset(repo_id=repo_id, root=resolved_root)
    total = int(dataset.meta.total_episodes)
    if total <= 0:
        return None, 0
    try:
        ep = dataset.meta.episodes[total - 1]
        return total - 1, int(ep.get("length", 0))
    except Exception:
        return total - 1, 0
