import importlib.util
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_script(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


finetune = _load_script("finetune_molmoact2", REPO / "scripts" / "finetune_molmoact2.py")


def _training_args(**overrides):
    base = dict(
        resume=False,
        resume_from="",
        dataset_repo_id="local/test",
        dataset_root="",
        video_backend="pyav",
        checkpoint_path="allenai/MolmoAct2",
        policy_path="",
        output_dir="outputs/molmoact2_finetune",
        job_name="molmoact2_finetune_xlerobot",
        steps=100,
        batch_size=2,
        num_workers=0,
        device="cpu",
        model_dtype="bfloat16",
        action_mode="continuous",
        train_mode_vlm="freeze",
        chunk_size=30,
        n_action_steps=30,
        num_flow_timesteps=8,
        setup_type="dual-arm XLerobot tabletop manipulation",
        control_mode="absolute joint position in degrees",
        normalize_gripper=True,
        image_keys_json='["observation.images.head","observation.images.left_wrist","observation.images.right_wrist"]',
        normalization_mapping_json='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}',
        log_freq=10,
        save_freq=50,
        eval_freq=-1,
        wandb_enable=False,
        push_to_hub=False,
        policy_repo_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_molmoact2_training_command_uses_allenai_module_entrypoint():
    cmd, image_keys, mapping = finetune._build_cmd(_training_args())

    assert cmd[:5] == ["uv", "run", "python", "-m", "lerobot.scripts.lerobot_train"]
    assert "--policy.type=molmoact2" in cmd
    assert "--policy.checkpoint_path=allenai/MolmoAct2" in cmd
    assert "--policy.action_mode=continuous" in cmd
    assert "--policy.train_mode_vlm=freeze" in cmd
    assert "--policy.normalize_gripper=true" in cmd
    assert "--policy.image_keys=[\"observation.images.head\",\"observation.images.left_wrist\",\"observation.images.right_wrist\"]" in cmd
    assert image_keys == [
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    assert mapping == {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}


def test_molmoact2_training_command_uses_policy_path_for_lerobot_checkpoint():
    cmd, _image_keys, _mapping = finetune._build_cmd(
        _training_args(policy_path="outputs/molmoact2/checkpoints/last/pretrained_model")
    )

    assert "--policy.type=molmoact2" not in cmd
    assert "--policy.checkpoint_path=allenai/MolmoAct2" not in cmd
    assert "--policy.path=outputs/molmoact2/checkpoints/last/pretrained_model" in cmd


def test_molmoact2_subprocess_env_prepends_allenai_vendor():
    env = finetune.molmoact2_subprocess_env(extra_pythonpath=[str(REPO)])
    parts = env["PYTHONPATH"].split(os.pathsep)

    assert pathlib.Path(parts[0]) == REPO / "vendor" / "allenai-lerobot" / "src"
    assert parts[1] == str(REPO)


def test_molmoact2_inference_dry_run_does_not_require_policy_path():
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "infer_molmoact2_finetuned.py"),
            "--dry-run",
            "--task",
            "test task",
        ],
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "Policy checkpoint : (not required for --dry-run)" in proc.stdout
    assert "Action horizon    : 50" in proc.stdout


def test_molmoact2_inference_dry_run_accepts_base_checkpoint_args():
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "infer_molmoact2_finetuned.py"),
            "--dry-run",
            "--checkpoint-path",
            "allenai/MolmoAct2",
            "--dataset-repo-id",
            "local/test",
            "--task",
            "test task",
        ],
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert ".cache/molmoact2_baseline_sources" in proc.stdout
    assert "Action horizon    : 50" in proc.stdout


def test_molmoact2_base_checkpoint_args_create_descriptor(tmp_path):
    code = """
import json
import os
import pathlib
import sys

repo = pathlib.Path.cwd()
sys.path.insert(0, str(repo / "scripts"))
import infer_molmoact2_finetuned as infer

infer.BASELINE_CACHE_DIR = pathlib.Path(os.environ["MOLMO_BASELINE_CACHE"])
args = infer._parse_args(
    [
        "--checkpoint-path",
        "allenai/MolmoAct2",
        "--dataset-repo-id",
        "local/test",
        "--task",
        "pick",
        "--action-horizon",
        "16",
        "--open-loop-steps",
        "8",
    ],
    require_task=True,
)
descriptor = pathlib.Path(args.policy_path) / infer.BASELINE_DESCRIPTOR
payload = json.loads(descriptor.read_text())
assert payload["checkpoint_path"] == "allenai/MolmoAct2"
assert payload["dataset_repo_id"] == "local/test"
assert payload["chunk_size"] == 16
assert payload["n_action_steps"] == 8
assert payload["normalize_gripper"] is True
print(args.policy_path)
"""
    env = os.environ.copy()
    env["MOLMO_BASELINE_CACHE"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert str(tmp_path) in proc.stdout


def test_molmoact2_base_checkpoint_descriptor_uses_molmo_defaults(tmp_path):
    code = """
import json
import os
import pathlib
import sys

repo = pathlib.Path.cwd()
sys.path.insert(0, str(repo / "scripts"))
import infer_molmoact2_finetuned as infer

infer.BASELINE_CACHE_DIR = pathlib.Path(os.environ["MOLMO_BASELINE_CACHE"])
args = infer._parse_args(
    [
        "--checkpoint-path",
        "allenai/MolmoAct2",
        "--dataset-repo-id",
        "local/test",
        "--task",
        "pick",
    ],
    require_task=True,
)
descriptor = pathlib.Path(args.policy_path) / infer.BASELINE_DESCRIPTOR
payload = json.loads(descriptor.read_text())
assert args.action_horizon == 50
assert args.open_loop_steps == 30
assert payload["chunk_size"] == 50
assert payload["n_action_steps"] == 30
assert payload["normalize_gripper"] is True
"""
    env = os.environ.copy()
    env["MOLMO_BASELINE_CACHE"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr


def test_molmoact2_job_endpoints_start_expected_scripts(monkeypatch):
    from openpibot.server.routers import jobs

    started = []

    class FakeJob:
        def __init__(self, command):
            self.command = command

        def public(self):
            return {"command": self.command, "status": "running"}

    def fake_start(command):
        started.append(command)
        return FakeJob(command)

    monkeypatch.setattr(jobs.JOBS, "start", fake_start)

    train = jobs.train_molmoact2({"args": ["--steps=1"]})
    infer = jobs.infer_molmoact2({"args": ["--task=pick"]})

    assert train["job"]["command"] == [
        "uv",
        "run",
        "python",
        "scripts/finetune_molmoact2.py",
        "--steps=1",
    ]
    assert infer["job"]["command"] == [
        "uv",
        "run",
        "python",
        "scripts/infer_molmoact2_finetuned.py",
        "--task=pick",
    ]
    assert len(started) == 2


def test_molmoact2_action_names_must_match_xlerobot_order():
    code = """
import pathlib
import sys
from types import SimpleNamespace

repo = pathlib.Path.cwd()
sys.path.insert(0, str(repo / "scripts"))
import infer_molmoact2_finetuned as infer

policy = SimpleNamespace(
    config=SimpleNamespace(
        action_feature_names=["right_arm_shoulder_pan.pos"] * 12,
        output_features={},
    )
)
try:
    infer._strict_action_names(policy)
except RuntimeError as exc:
    print(str(exc))
    raise SystemExit(0)
raise SystemExit("metadata check unexpectedly passed")
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "do not match XLerobot joint order" in proc.stdout


def test_molmoact2_action_names_use_saved_dataset_feature_metadata():
    code = """
import pathlib
import sys
from types import SimpleNamespace

repo = pathlib.Path.cwd()
sys.path.insert(0, str(repo / "scripts"))
import infer_molmoact2_finetuned as infer

names = [f"{side}_arm_{joint}.pos" for side in ("left", "right") for joint in (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)]
policy = SimpleNamespace(
    config=SimpleNamespace(
        action_feature_names=None,
        dataset_feature_names={"action": {"motors": names}},
    )
)
assert infer._strict_action_names(policy) == names
infer._ensure_runtime_action_feature_names(policy)
assert policy.config.action_feature_names == names
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr


def test_molmoact2_action_horizon_alignment_updates_action_expert_config():
    code = """
import pathlib
import sys
from types import SimpleNamespace

repo = pathlib.Path.cwd()
sys.path.insert(0, str(repo / "scripts"))
import infer_molmoact2_finetuned as infer

action_expert = SimpleNamespace(config=SimpleNamespace(max_action_horizon=30))
backbone = SimpleNamespace(
    config=SimpleNamespace(
        max_action_horizon=30,
        action_expert_config=SimpleNamespace(max_action_horizon=30),
    ),
    action_expert=action_expert,
)
hf_model = SimpleNamespace(
    config=SimpleNamespace(
        max_action_horizon=30,
        action_expert_config=SimpleNamespace(max_action_horizon=30),
    ),
    model=backbone,
)

class FakePolicy:
    config = SimpleNamespace(chunk_size=50)

    def _hf_model(self):
        return hf_model

    def _backbone(self):
        return backbone

    def _override_loaded_max_action_horizon(self, horizon):
        hf_model.config.max_action_horizon = int(horizon)
        backbone.config.max_action_horizon = int(horizon)

policy = FakePolicy()
infer._align_molmoact2_action_horizon(policy)
assert hf_model.config.max_action_horizon == 50
assert hf_model.config.action_expert_config.max_action_horizon == 50
assert backbone.config.max_action_horizon == 50
assert backbone.config.action_expert_config.max_action_horizon == 50
assert action_expert.config.max_action_horizon == 50
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
