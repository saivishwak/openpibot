import asyncio
import os
import threading

import pytest
from typer.testing import CliRunner

from openpibot import cli
from openpibot import agent_reasoning as ar
from openpibot.cli import app


def test_system2_prompt_loads_from_text_file_without_json_contract():
    prompt = ar.load_system2_prompt()

    assert "stage_pi05_execution_plan" in prompt
    assert "get_execution_state" in prompt
    assert "track_execution_plan_progress" in prompt
    assert "Return only a JSON object" not in prompt
    assert "tools natively" in prompt
    assert "You are the robot" in prompt
    assert "user-facing name is Tess" in prompt
    assert "identify yourself as Tess" in prompt
    assert "OpenPiBot System 2" in prompt
    assert "my arms" in prompt
    assert 'Do not refer to "the robot" when you mean yourself' in prompt


def test_parse_plan_progress_accepts_tracker_payload():
    progress = ar._parse_plan_progress(
        {
            "mode": "progress",
            "message": "Item 1 is done.",
            "progress": {
                "next_action": "continue",
                "item_statuses": [
                    {"index": 1, "status": "completed", "note": "marker is in cup"},
                    {"index": 2, "status": "pending", "note": "second marker remains"},
                ],
                "updated_tasks": [
                    {"index": 2, "task": "Pick up the second marker and place it in the cup."}
                ],
            },
        }
    )

    assert progress.next_action == "continue"
    assert [(item.index, item.status) for item in progress.item_statuses] == [
        (1, "completed"),
        (2, "pending"),
    ]
    assert progress.updated_tasks[0].index == 2
    assert progress.updated_tasks[0].task == "Pick up the second marker and place it in the cup."


def test_build_system1_contract_exposes_runtime_defaults(tmp_path):
    config = ar.AgentReasoningConfig(policy_path=tmp_path / "pretrained_model")

    contract = ar.build_system1_contract(config)

    assert contract["controller"] == "finetuned PI0.5 VLA"
    assert contract["execution_defaults"]["policy_path"] == str(config.policy_path)
    assert "episodes" not in contract["execution_defaults"]
    assert "joint angles" in contract["not_allowed"]


def test_agent_respond_posts_rest_call_to_pi_agent_for_chat(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    calls = []

    def rest_client(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {"mode": "chat", "message": "I am Tess."}

    reasoner = ar.AgentReasoner(
        ar.AgentReasoningConfig(policy_path=tmp_path),
        snapshot_factory=lambda camera: b"jpeg",
        rest_client=rest_client,
    )

    turn = asyncio.run(reasoner.respond("What is your name?"))

    assert turn.mode == "chat"
    assert turn.message == "I am Tess."
    assert calls[0][0].endswith("/turn")
    payload = calls[0][1]
    assert payload["query"] == "What is your name?"
    assert payload["image"]["data"]
    assert payload["model"] == "gpt-5.4"
    assert payload["thinking_level"] == "medium"
    assert "OpenPiBot System 2" in payload["system_prompt"]


def test_review_plan_progress_posts_execution_state(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    payloads = []

    def rest_client(url, payload, timeout):
        payloads.append(payload)
        return {
            "mode": "progress",
            "message": "Done.",
            "progress": {
                "next_action": "done",
                "item_statuses": [
                    {"index": 1, "status": "completed", "note": "done"}
                ],
            },
        }

    reasoner = ar.AgentReasoner(
        ar.AgentReasoningConfig(policy_path=tmp_path),
        snapshot_factory=lambda camera: b"jpeg",
        rest_client=rest_client,
    )
    plan = ar.AgentPlan(summary="clean", steps=[ar.PlannedStep("put marker in cup")])

    progress = asyncio.run(
        reasoner.review_plan_progress(
            original_query="clean the desk",
            plan=plan,
            statuses=["pending"],
            current_index=0,
            last_exit_code=0,
            attempt=1,
        )
    )

    assert progress.next_action == "done"
    assert payloads[0]["execution_state"]["original_query"] == "clean the desk"
    assert payloads[0]["execution_state"]["plan"]["steps"][0]["task"] == "put marker in cup"


def test_agent_respond_returns_staged_plan_from_bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    def rest_client(url, payload, timeout):
        return {
            "mode": "plan",
            "message": "I staged the plan.",
            "plan": {
                "summary": "Move the pencil into the cup.",
                "steps": [
                    {
                        "task": "Pick up the pencil from the table and place it inside the cup.",
                        "reason": "Simple visible pick and place.",
                    }
                ],
                "safety_notes": ["Avoid the keyboard."],
            },
        }

    reasoner = ar.AgentReasoner(
        ar.AgentReasoningConfig(policy_path=tmp_path),
        snapshot_factory=lambda camera: b"jpeg",
        rest_client=rest_client,
    )

    turn = asyncio.run(reasoner.respond("put the pencil in the cup"))

    assert turn.mode == "plan"
    assert turn.message == "I staged the plan."
    assert turn.summary == "Move the pencil into the cup."
    assert [step.task for step in turn.steps] == [
        "Pick up the pencil from the table and place it inside the cup."
    ]
    assert turn.steps[0].reason == "Simple visible pick and place."
    assert turn.safety_notes == ["Avoid the keyboard."]


def test_agent_respond_preserves_recent_chat_history(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    payloads = []

    def rest_client(url, payload, timeout):
        payloads.append(payload)
        if payloads[-1]["query"] == "What is your name?":
            return {"mode": "chat", "message": "I am Tess."}
        return {"mode": "chat", "message": "Hi. I can help with the desk scene."}

    reasoner = ar.AgentReasoner(
        ar.AgentReasoningConfig(policy_path=tmp_path),
        snapshot_factory=lambda camera: b"jpeg",
        rest_client=rest_client,
    )

    first = asyncio.run(reasoner.respond("Heyy"))
    second = asyncio.run(reasoner.respond("What is your name?"))

    assert first.mode == "chat"
    assert second.message == "I am Tess."
    assert payloads[1]["history"] == [
        {"role": "user", "content": "Heyy"},
        {"role": "assistant", "content": "Hi. I can help with the desk scene."},
    ]


def test_rest_failure_is_wrapped_for_repl(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    def rest_client(url, payload, timeout):
        raise ar.AgentReasoningError("bad model")

    reasoner = ar.AgentReasoner(
        ar.AgentReasoningConfig(policy_path=tmp_path),
        snapshot_factory=lambda camera: b"jpeg",
        rest_client=rest_client,
    )

    with pytest.raises(ar.AgentReasoningError, match="bad model"):
        asyncio.run(reasoner.respond("Heyy"))


def test_missing_bun_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    def process_factory(command, **kwargs):
        raise FileNotFoundError("bun")

    reasoner = ar.AgentReasoner(
        ar.AgentReasoningConfig(
            policy_path=tmp_path,
            pi_agent_url="http://127.0.0.1:9",
            pi_agent_start_timeout_s=1,
        ),
        snapshot_factory=lambda camera: b"jpeg",
        process_factory=process_factory,
    )

    with pytest.raises(ar.AgentReasoningError, match="bun is required"):
        asyncio.run(reasoner.respond("Heyy"))


def test_missing_openai_key_blocks_before_camera(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    snapshots = []
    reasoner = ar.AgentReasoner(
        ar.AgentReasoningConfig(policy_path=tmp_path),
        snapshot_factory=lambda camera: snapshots.append(camera) or b"jpeg",
    )

    with pytest.raises(ar.AgentReasoningError, match="OPENAI_API_KEY"):
        asyncio.run(reasoner.plan("pick up the pencil"))

    assert snapshots == []


def test_empty_query_blocks_before_camera(tmp_path):
    snapshots = []
    reasoner = ar.AgentReasoner(
        ar.AgentReasoningConfig(policy_path=tmp_path),
        snapshot_factory=lambda camera: snapshots.append(camera) or b"jpeg",
    )

    with pytest.raises(ar.AgentReasoningError, match="cannot be empty"):
        asyncio.run(reasoner.plan("   "))

    assert snapshots == []


def test_snapshot_failure_blocks_planning(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    reasoner = ar.AgentReasoner(
        ar.AgentReasoningConfig(policy_path=tmp_path),
        snapshot_factory=lambda camera: (_ for _ in ()).throw(
            ar.AgentReasoningError("camera unavailable")
        ),
    )

    with pytest.raises(ar.AgentReasoningError, match="camera unavailable"):
        asyncio.run(reasoner.plan("pick up the pencil"))


def test_bridge_invalid_json_is_rejected():
    with pytest.raises(ar.AgentReasoningError, match="invalid JSON"):
        ar._parse_bridge_turn("not json")


def test_build_inference_command_matches_known_good_shape(tmp_path):
    config = ar.AgentReasoningConfig(policy_path=tmp_path / "pretrained_model")
    step = ar.PlannedStep(
        task="Pick up the pencil from the table and place it inside the cup",
    )

    command = ar.build_inference_command(config, step)

    assert command[:4] == ["uv", "run", "python", "scripts/infer_pi05_finetuned.py"]
    assert command[command.index("--policy-path") + 1] == str(config.policy_path)
    assert command[command.index("--task") + 1] == step.task
    assert command[command.index("--camera-backend") + 1] == "dashboard"
    assert command[command.index("--episodes") + 1] == "1"
    assert command[command.index("--episode-time") + 1] == "120"
    assert command[command.index("--fps") + 1] == "30"
    assert command[command.index("--settle-steps") + 1] == "30"
    assert command[command.index("--gripper-max-relative-target") + 1] == "15.0"
    assert "--no-show-cameras" in command


def test_build_pi05_runner_options_omits_task_and_loads_policy_once(tmp_path):
    config = ar.AgentReasoningConfig(policy_path=tmp_path / "pretrained_model")

    options = ar.build_pi05_runner_options(config)

    assert options.task is None
    assert options.policy_path == config.policy_path
    assert options.camera_backend == "dashboard"
    assert options.episodes == 1
    assert options.episode_time == 120
    assert options.gripper_max_relative_target == 15.0
    assert options.print_startup_summary is False


def test_warm_pi05_runner_rejects_extra_cli_passthrough(tmp_path):
    config = ar.AgentReasoningConfig(
        policy_path=tmp_path / "pretrained_model",
        extra_infer_args=("--open-loop-steps", "20"),
    )

    with pytest.raises(ar.AgentReasoningError, match="subprocess fallback"):
        ar.build_pi05_runner_options(config)


def test_ensure_system1_ready_embeds_runner_module(tmp_path, monkeypatch):
    from openpibot import pi05_runner

    started = []

    class FakeRunnerServer:
        def __init__(self, *, host, port, runtime_options):
            started.append((host, port, runtime_options))

        def start_background(self):
            started.append("background")

        def close(self):
            started.append("closed")

    monkeypatch.setattr(pi05_runner, "PI05RunnerServer", FakeRunnerServer)
    monkeypatch.setattr(ar, "_get_json", lambda url, timeout_s: {"ok": False})

    reasoner = ar.AgentReasoner(
        ar.AgentReasoningConfig(policy_path=tmp_path / "pretrained_model"),
    )

    with pytest.raises(ar.AgentReasoningError, match="did not start"):
        reasoner.ensure_system1_ready()

    assert started[0][0] == "127.0.0.1"
    assert started[0][1] == 8767
    assert started[0][2].policy_path == (tmp_path / "pretrained_model").resolve()
    assert started[0][2].task is None
    assert "background" in started


def test_config_resolves_policy_path_and_rejects_invalid_runtime_values(tmp_path):
    rel = tmp_path / "policy"
    rel.mkdir()

    config = ar.AgentReasoningConfig(policy_path=rel)

    assert config.policy_path == rel.resolve()
    with pytest.raises(ar.AgentReasoningError, match="invalid camera backend"):
        ar.AgentReasoningConfig(policy_path=rel, camera_backend="bad")
    with pytest.raises(ar.AgentReasoningError, match="invalid reasoning effort"):
        ar.AgentReasoningConfig(policy_path=rel, reasoning_effort="bad")
    with pytest.raises(ar.AgentReasoningError, match="invalid inference device"):
        ar.AgentReasoningConfig(policy_path=rel, device="bad")


def test_run_plan_stops_after_failed_step(tmp_path):
    calls = []

    def rest_client(url, payload, timeout):
        calls.append((url, payload))
        return {"exit_code": 7}

    reasoner = ar.AgentReasoner(
        ar.AgentReasoningConfig(policy_path=tmp_path),
        rest_client=rest_client,
    )
    plan = ar.AgentPlan(
        summary="two steps",
        steps=[ar.PlannedStep("first"), ar.PlannedStep("second")],
    )

    assert reasoner.run_plan(plan) == 7
    assert len(calls) == 1
    assert calls[0][0].endswith("/run")
    assert calls[0][1]["task"] == "first"
    assert "episodes" not in calls[0][1]


def test_interrupt_current_run_posts_stop_to_warm_runner(tmp_path):
    calls = []

    def rest_client(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {"ok": True, "stop_requested": True}

    reasoner = ar.AgentReasoner(
        ar.AgentReasoningConfig(policy_path=tmp_path),
        rest_client=rest_client,
    )

    reasoner.interrupt_current_run("change task")

    assert calls == [
        (
            "http://127.0.0.1:8767/stop",
            {"reason": "change task"},
            5.0,
        )
    ]


def test_interactive_plan_execution_interrupts_and_returns_pending_query(monkeypatch):
    stop_event = threading.Event()
    calls = []

    class FakeReasoner:
        def run_plan(self, plan):
            stop_event.wait(timeout=2.0)
            return 130

        def interrupt_current_run(self, reason):
            calls.append(reason)
            stop_event.set()

    monkeypatch.setattr(cli, "_read_available_stdin_line", lambda timeout: "new task")

    code, pending_query = cli._run_plan_interactively(FakeReasoner(), object())

    assert code == 130
    assert pending_query == "new task"
    assert calls == ["new task"]


def test_interactive_plan_execution_tracks_items_until_done():
    calls = []

    class FakeReasoner:
        def run_plan(self, plan):
            calls.append(plan.steps[0].task)
            return 0

        async def review_plan_progress(self, **kwargs):
            current = kwargs["current_index"]
            if current == 0:
                return ar.PlanProgress(
                    message="first item done",
                    next_action="continue",
                    item_statuses=[
                        ar.PlanItemProgress(index=1, status="completed"),
                        ar.PlanItemProgress(index=2, status="pending"),
                    ],
                )
            return ar.PlanProgress(
                message="all done",
                next_action="done",
                item_statuses=[
                    ar.PlanItemProgress(index=1, status="completed"),
                    ar.PlanItemProgress(index=2, status="completed"),
                ],
            )

    plan = ar.AgentPlan(
        summary="clean",
        steps=[ar.PlannedStep("put marker one in cup"), ar.PlannedStep("put marker two in cup")],
    )

    code, pending_query = cli._run_plan_to_completion_interactively(
        FakeReasoner(),
        plan,
        original_query="clean the desk",
    )

    assert code == 0
    assert pending_query is None
    assert calls == ["put marker one in cup", "put marker two in cup"]


def test_interactive_plan_execution_applies_replanned_task_after_timeout(tmp_path):
    calls = []
    reviewed = []

    class FakeReasoner:
        config = ar.AgentReasoningConfig(policy_path=tmp_path)

        def run_plan(self, plan):
            calls.append(plan.steps[0].task)
            return 7 if len(calls) == 1 else 0

        def last_execution_result(self):
            return {"exit_code": 7, "stop_reason": "wall-clock limit (120s)"}

        async def review_plan_progress(self, **kwargs):
            reviewed.append(kwargs["last_exit_code"])
            if len(reviewed) == 1:
                return ar.PlanProgress(
                    message="replan current item",
                    next_action="retry",
                    item_statuses=[ar.PlanItemProgress(index=1, status="pending")],
                    updated_tasks=[
                        ar.PlanTaskUpdate(
                            index=1,
                            task="Pick up the marker from its new position and place it in the cup.",
                        )
                    ],
                )
            return ar.PlanProgress(
                message="done",
                next_action="done",
                item_statuses=[ar.PlanItemProgress(index=1, status="completed")],
            )

    plan = ar.AgentPlan(summary="clean", steps=[ar.PlannedStep("put marker in cup")])

    code, pending_query = cli._run_plan_to_completion_interactively(
        FakeReasoner(),
        plan,
        original_query="clean the desk",
    )

    assert code == 0
    assert pending_query is None
    assert reviewed == [7, 0]
    assert calls == [
        "put marker in cup",
        "Pick up the marker from its new position and place it in the cup.",
    ]


def test_run_step_reports_subprocess_start_failure(tmp_path):
    def failing_runner(command, **kwargs):
        raise FileNotFoundError("uv")

    reasoner = ar.AgentReasoner(
        ar.AgentReasoningConfig(policy_path=tmp_path, use_pi05_runner=False),
        runner=failing_runner,
    )

    with pytest.raises(ar.AgentReasoningError, match="failed to start"):
        reasoner.run_step(ar.PlannedStep("do task"))


def test_agent_cli_confirmation_no_never_executes(monkeypatch, tmp_path):
    policy_path = tmp_path / "pretrained_model"
    policy_path.mkdir()
    instances = []

    class FakeReasoner:
        def __init__(self, config):
            self.config = config
            self.run_called = False
            instances.append(self)

        def ensure_system1_ready(self):
            return None

        def run_plan(self, plan):
            self.run_called = True
            return 0

        async def review_plan_progress(self, **kwargs):
            return ar.PlanProgress(
                next_action="done",
                item_statuses=[ar.PlanItemProgress(index=1, status="completed")],
            )

    def fake_turn_sync(reasoner, query):
        return ar.AgentTurn(
            mode="plan",
            message="ready",
            summary="plan",
            steps=[ar.PlannedStep("do task")],
        )

    monkeypatch.setattr(ar, "AgentReasoner", FakeReasoner)
    monkeypatch.setattr(ar, "turn_sync", fake_turn_sync)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["agent", "--policy-path", str(policy_path)],
        input="pick\nno\nquit\n",
        env={**os.environ, "OPENAI_API_KEY": "test"},
    )

    assert result.exit_code == 0
    assert instances
    assert instances[0].run_called is False


def test_agent_cli_confirmation_yes_executes(monkeypatch, tmp_path):
    policy_path = tmp_path / "pretrained_model"
    policy_path.mkdir()
    instances = []

    class FakeReasoner:
        def __init__(self, config):
            self.config = config
            self.run_called = False
            instances.append(self)

        def ensure_system1_ready(self):
            return None

        def run_plan(self, plan):
            self.run_called = True
            return 0

        async def review_plan_progress(self, **kwargs):
            return ar.PlanProgress(
                next_action="done",
                item_statuses=[ar.PlanItemProgress(index=1, status="completed")],
            )

    monkeypatch.setattr(ar, "AgentReasoner", FakeReasoner)
    monkeypatch.setattr(
        ar,
        "turn_sync",
        lambda reasoner, query: ar.AgentTurn(
            mode="plan",
            message="ready",
            summary="plan",
            steps=[ar.PlannedStep("do task")],
        ),
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "agent",
            "--policy-path",
            str(policy_path),
            "--episode-time",
            "10",
            "--max-steps",
            "2",
            "--llm-timeout",
            "30",
        ],
        input="pick\nyes\nquit\n",
        env={**os.environ, "OPENAI_API_KEY": "test"},
    )

    assert result.exit_code == 0
    assert instances
    assert instances[0].run_called is True
    assert instances[0].config.episodes is None
    assert instances[0].config.episode_time == 10
    assert instances[0].config.max_steps == 2
    assert instances[0].config.llm_timeout_s == 30


def test_agent_cli_chat_turn_does_not_ask_for_confirmation(monkeypatch, tmp_path):
    policy_path = tmp_path / "pretrained_model"
    policy_path.mkdir()
    instances = []

    class FakeReasoner:
        def __init__(self, config):
            self.run_called = False
            instances.append(self)

        def ensure_system1_ready(self):
            return None

        def run_plan(self, plan):
            self.run_called = True
            return 0

    monkeypatch.setattr(ar, "AgentReasoner", FakeReasoner)
    monkeypatch.setattr(
        ar,
        "turn_sync",
        lambda reasoner, query: ar.AgentTurn(
            mode="chat",
            message="Hello. Give me a physical task when ready.",
        ),
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["agent", "--policy-path", str(policy_path)],
        input="Heyy\nquit\n",
        env={**os.environ, "OPENAI_API_KEY": "test"},
    )

    assert result.exit_code == 0
    assert "Hello. Give me a physical task when ready." in result.output
    assert instances
    assert instances[0].run_called is False
