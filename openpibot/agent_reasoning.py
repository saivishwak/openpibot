"""System 2 reasoning layer for PI0.5 robot inference."""

from __future__ import annotations

import asyncio
import atexit
import base64
import json
import os
import pathlib
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Callable, Literal

from openpibot.server.config import REPO_ROOT

VisionCamera = Literal["head", "left_wrist", "right_wrist"]
BridgeRunner = Callable[..., subprocess.CompletedProcess[str]]
RestClient = Callable[[str, dict[str, Any], float], dict[str, Any]]
ProcessFactory = Callable[..., subprocess.Popen[str]]


class AgentReasoningError(RuntimeError):
    """Raised when System 2 planning cannot safely continue."""


@dataclass(slots=True)
class PlannedStep:
    """One bounded instruction for the System 1 PI0.5 controller."""

    task: str
    reason: str = ""
    episode_time: int | None = None
    episodes: int | None = None


@dataclass(slots=True)
class AgentPlan:
    """Structured System 2 plan."""

    summary: str
    steps: list[PlannedStep] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)


PlanItemStatus = Literal["pending", "completed", "blocked"]
PlanNextAction = Literal["continue", "retry", "done", "blocked"]


@dataclass(slots=True)
class PlanItemProgress:
    """Visual progress status for one staged plan item."""

    index: int
    status: PlanItemStatus
    note: str = ""


@dataclass(slots=True)
class PlanTaskUpdate:
    """Updated or appended PI0.5 task prompt from a progress replan."""

    index: int
    task: str


@dataclass(slots=True)
class PlanProgress:
    """System 2 progress review for an executing physical plan."""

    message: str = ""
    next_action: PlanNextAction = "continue"
    item_statuses: list[PlanItemProgress] = field(default_factory=list)
    updated_tasks: list[PlanTaskUpdate] = field(default_factory=list)


@dataclass(slots=True)
class AgentTurn:
    """One System 2 REPL response, either normal chat or a physical plan."""

    mode: Literal["chat", "plan"] = "chat"
    message: str = ""
    summary: str = ""
    steps: list[PlannedStep] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)

    def as_plan(self) -> AgentPlan:
        return AgentPlan(
            summary=self.summary or self.message,
            steps=list(self.steps),
            safety_notes=list(self.safety_notes),
        )


@dataclass(slots=True)
class AgentReasoningConfig:
    """Runtime configuration for the System 2 REPL."""

    policy_path: pathlib.Path
    model: str = "gpt-5.4"
    vision_camera: VisionCamera = "head"
    api_key: str | None = None
    reasoning_effort: str = "medium"
    llm_timeout_s: int = 60
    max_steps: int = 6
    episodes: int | None = None
    episode_time: int = 120
    fps: int = 30
    settle_steps: int = 30
    gripper_max_relative_target: float | None = 15.0
    camera_backend: str = "dashboard"
    device: str = "cuda"
    show_cameras: bool = False
    extra_infer_args: tuple[str, ...] = ()
    chat_history_turns: int = 12
    bun_executable: str = "bun"
    bridge_script: pathlib.Path = REPO_ROOT / "openpibot" / "pi_agent_bridge.ts"
    pi_agent_url: str = "http://127.0.0.1:8765"
    pi_agent_autostart: bool = True
    pi_agent_start_timeout_s: int = 10
    pi05_runner_url: str = "http://127.0.0.1:8767"
    pi05_runner_autostart: bool = True
    pi05_runner_start_timeout_s: int = 180
    use_pi05_runner: bool = True

    def __post_init__(self) -> None:
        self.policy_path = self.policy_path.expanduser().resolve()
        self.bridge_script = self.bridge_script.expanduser().resolve()
        self.llm_timeout_s = _clamp_int(self.llm_timeout_s, minimum=1, maximum=600)
        self.max_steps = _clamp_int(self.max_steps, minimum=1, maximum=12)
        if self.episodes is not None:
            self.episodes = _clamp_int(self.episodes, minimum=1, maximum=20)
        self.episode_time = _clamp_int(self.episode_time, minimum=1, maximum=3600)
        self.fps = _clamp_int(self.fps, minimum=1, maximum=120)
        self.settle_steps = _clamp_int(self.settle_steps, minimum=0, maximum=10000)
        self.pi_agent_start_timeout_s = _clamp_int(
            self.pi_agent_start_timeout_s, minimum=1, maximum=60
        )
        self.pi05_runner_start_timeout_s = _clamp_int(
            self.pi05_runner_start_timeout_s, minimum=1, maximum=600
        )
        self.chat_history_turns = _clamp_int(
            self.chat_history_turns, minimum=0, maximum=50
        )
        if self.gripper_max_relative_target is not None:
            self.gripper_max_relative_target = max(
                0.1, min(100.0, float(self.gripper_max_relative_target))
            )
        if self.vision_camera not in ("head", "left_wrist", "right_wrist"):
            raise AgentReasoningError(f"invalid vision camera: {self.vision_camera}")
        if self.reasoning_effort not in ("none", "low", "medium", "high", "xhigh"):
            raise AgentReasoningError(f"invalid reasoning effort: {self.reasoning_effort}")
        if self.camera_backend not in ("dashboard", "lerobot"):
            raise AgentReasoningError(f"invalid camera backend: {self.camera_backend}")
        if self.device not in ("cuda", "cpu", "mps"):
            raise AgentReasoningError(f"invalid inference device: {self.device}")
        if self.bun_executable == "bun" and shutil.which("bun") is None:
            local_bun = pathlib.Path.home() / ".bun" / "bin" / "bun"
            if local_bun.is_file():
                self.bun_executable = str(local_bun)
        self.pi_agent_url = self.pi_agent_url.rstrip("/")
        self.pi05_runner_url = self.pi05_runner_url.rstrip("/")


PROMPT_PACKAGE = "openpibot.prompts"
SYSTEM2_PROMPT_FILE = "system2_agent.txt"


def load_system2_prompt() -> str:
    """Load the System 2 operational prompt from package data."""
    return resources.files(PROMPT_PACKAGE).joinpath(SYSTEM2_PROMPT_FILE).read_text()


def capture_dashboard_snapshot(camera: VisionCamera) -> bytes:
    """Capture a required JPEG snapshot from the configured camera role."""
    from openpibot.server.runtime import cameras

    stream = cameras.get_stream(camera)
    if stream is None:
        reason = cameras.capture_suspension_reason(camera)
        detail = f": {reason}" if reason else ""
        raise AgentReasoningError(f"camera '{camera}' is unavailable{detail}")
    try:
        jpeg, err = stream.snapshot()
        if not jpeg:
            raise AgentReasoningError(
                f"camera '{camera}' snapshot failed: {err or 'no frame'}"
            )
        return jpeg
    finally:
        # The inference subprocess opens the same dashboard CameraStream path.
        # Ensure this planning snapshot releases the V4L handle first.
        try:
            cameras.suspend_capture_roles(
                [camera],
                reason="System 2 planning snapshot complete",
            )
        finally:
            cameras.resume_capture_roles([camera])


def build_system1_contract(config: AgentReasoningConfig) -> dict[str, Any]:
    """Return the System 1 contract passed to the Pi agent tools."""
    return {
        "controller": "finetuned PI0.5 VLA",
        "input": "short natural-language manipulation task prompt plus live robot observations",
        "not_allowed": [
            "joint angles",
            "coordinates",
            "code",
            "open-ended exploration",
            "commands to contact fragile electronics unless explicitly requested",
        ],
        "recommended_step_count": f"1-{config.max_steps}",
        "execution_defaults": build_execution_defaults(config),
        "available_cameras": ["head", "left_wrist", "right_wrist"],
    }


def build_execution_defaults(config: AgentReasoningConfig) -> dict[str, Any]:
    """Return default PI0.5 runtime settings visible to System 2."""
    return {
        "policy_path": str(config.policy_path),
        "camera_backend": config.camera_backend,
        "episode_time": config.episode_time,
        "fps": config.fps,
        "settle_steps": config.settle_steps,
        "gripper_max_relative_target": config.gripper_max_relative_target,
        "device": config.device,
        "show_cameras": config.show_cameras,
        "extra_infer_args": list(config.extra_infer_args),
    }


def _clamp_int(value: Any, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise AgentReasoningError(f"expected integer value, got {value!r}") from exc
    return max(minimum, min(maximum, parsed))


class AgentReasoner:
    """Coordinates required image capture, System 2 planning, and System 1 execution."""

    def __init__(
        self,
        config: AgentReasoningConfig,
        *,
        snapshot_factory: Callable[[VisionCamera], bytes] = capture_dashboard_snapshot,
        rest_client: RestClient = None,  # type: ignore[assignment]
        process_factory: ProcessFactory = subprocess.Popen,
        runner: BridgeRunner = subprocess.run,
    ) -> None:
        self.config = config
        self._snapshot_factory = snapshot_factory
        self._rest_client_injected = rest_client is not None
        self._rest_client = rest_client or _post_json
        self._process_factory = process_factory
        self._runner = runner
        self._pi_agent_process: subprocess.Popen[str] | None = None
        self._pi05_runner_server: Any | None = None
        self._history: list[dict[str, str]] = []
        self._last_execution_result: dict[str, Any] = {}
        atexit.register(self.close)

    async def plan(self, query: str) -> AgentPlan:
        return (await self.respond(query)).as_plan()

    async def respond(self, query: str) -> AgentTurn:
        query = query.strip()
        if not query:
            raise AgentReasoningError("agent query cannot be empty")
        self._require_openai_key()
        image = self._snapshot_factory(self.config.vision_camera)
        if not image:
            raise AgentReasoningError("camera snapshot is required for planning")
        payload = self._build_bridge_payload(query, image)
        result = await asyncio.to_thread(self._call_pi_agent, payload)
        turn = _parse_agent_turn(result)
        self._remember_turn(query, _turn_history_text(turn))
        return turn

    def _require_openai_key(self) -> None:
        if self.config.api_key:
            return
        if os.environ.get("OPENAI_API_KEY"):
            return
        raise AgentReasoningError("OPENAI_API_KEY is required for the System 2 VLM")

    def _build_bridge_payload(self, query: str, image: bytes) -> dict[str, Any]:
        return {
            "query": query,
            "system_prompt": load_system2_prompt(),
            "model": self.config.model,
            "thinking_level": _pi_thinking_level(self.config.reasoning_effort),
            "timeout_ms": self.config.llm_timeout_s * 1000,
            "max_steps": self.config.max_steps,
            "image": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image).decode("ascii"),
            },
            "history": list(self._history[-self.config.chat_history_turns * 2 :])
            if self.config.chat_history_turns > 0
            else [],
            "system1_contract": build_system1_contract(self.config),
        }

    def _build_bridge_payload_with_state(
        self,
        query: str,
        image: bytes,
        execution_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._build_bridge_payload(query, image)
        if execution_state is not None:
            payload["execution_state"] = execution_state
        return payload

    def _call_pi_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._rest_client_injected:
            self._ensure_pi_agent_service()
        return self._rest_client(
            f"{self.config.pi_agent_url}/turn",
            payload,
            float(self.config.llm_timeout_s + 10),
        )

    async def review_plan_progress(
        self,
        *,
        original_query: str,
        plan: AgentPlan,
        statuses: list[PlanItemStatus],
        current_index: int,
        last_exit_code: int,
        attempt: int,
        result_reason: str = "",
    ) -> PlanProgress:
        self._require_openai_key()
        image = self._snapshot_factory(self.config.vision_camera)
        if not image:
            raise AgentReasoningError("camera snapshot is required for progress review")
        execution_state = {
            "original_query": original_query,
            "plan": {
                "summary": plan.summary,
                "steps": [
                    {"index": idx + 1, "task": step.task, "status": statuses[idx]}
                    for idx, step in enumerate(plan.steps)
                ],
            },
            "current_step": {
                "index": current_index + 1,
                "task": plan.steps[current_index].task,
                "attempt": attempt,
                "last_exit_code": last_exit_code,
                "result_reason": result_reason,
            },
        }
        query = (
            "Review execution progress for the active physical plan. "
            "Use get_execution_state first, then use the track_execution_plan_progress tool exactly once. "
            "Mark completed items from the current camera image, update pending task prompts if the "
            "plan should change, and choose whether to continue, retry, finish, or block."
        )
        payload = self._build_bridge_payload_with_state(query, image, execution_state)
        result = await asyncio.to_thread(self._call_pi_agent, payload)
        return _parse_plan_progress(result)

    def _ensure_pi_agent_service(self) -> None:
        if self._pi_agent_healthcheck():
            return
        if not self.config.pi_agent_autostart:
            raise AgentReasoningError(
                f"Pi System 2 agent is not reachable at {self.config.pi_agent_url}"
            )
        try:
            bridge_script = str(self.config.bridge_script.relative_to(REPO_ROOT))
        except ValueError:
            bridge_script = str(self.config.bridge_script)
        command = [
            self.config.bun_executable,
            "run",
            bridge_script,
            "--serve",
            "--host",
            _agent_host_from_url(self.config.pi_agent_url),
            "--port",
            str(_agent_port_from_url(self.config.pi_agent_url)),
        ]
        env = os.environ.copy()
        if self.config.api_key:
            env["OPENAI_API_KEY"] = self.config.api_key
        try:
            self._pi_agent_process = self._process_factory(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise AgentReasoningError(
                "bun is required for the Pi System 2 agent. Install Bun and run `bun install`."
            ) from exc
        except OSError as exc:
            raise AgentReasoningError(f"failed to start Pi System 2 agent: {exc}") from exc
        deadline = time.monotonic() + self.config.pi_agent_start_timeout_s
        while time.monotonic() < deadline:
            if self._pi_agent_process.poll() is not None:
                raise AgentReasoningError("Pi System 2 agent exited during startup")
            if self._pi_agent_healthcheck():
                return
            time.sleep(0.1)
        raise AgentReasoningError(
            f"Pi System 2 agent did not start at {self.config.pi_agent_url}"
        )

    def _pi_agent_healthcheck(self) -> bool:
        try:
            data = _get_json(f"{self.config.pi_agent_url}/health", timeout_s=1.0)
        except AgentReasoningError:
            return False
        return data.get("ok") is True

    def close(self) -> None:
        if self._pi05_runner_server is not None:
            self._pi05_runner_server.close()
            self._pi05_runner_server = None
        self._terminate_process(self._pi_agent_process)
        self._pi_agent_process = None

    def _terminate_process(self, proc: subprocess.Popen[str] | None) -> None:
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    def ensure_system1_ready(self) -> None:
        if not self.config.use_pi05_runner:
            return
        self._ensure_pi05_runner_service()

    def _ensure_pi05_runner_service(self) -> None:
        if self._pi05_runner_healthcheck():
            return
        if not self.config.pi05_runner_autostart:
            raise AgentReasoningError(
                f"PI0.5 runner is not reachable at {self.config.pi05_runner_url}"
            )
        try:
            from openpibot.pi05_runner import PI05RunnerServer

            self._pi05_runner_server = PI05RunnerServer(
                host=_agent_host_from_url(self.config.pi05_runner_url),
                port=_agent_port_from_url(self.config.pi05_runner_url),
                runtime_options=build_pi05_runner_options(self.config),
            )
        except OSError as exc:
            raise AgentReasoningError(f"failed to start PI0.5 runner: {exc}") from exc
        except Exception as exc:
            raise AgentReasoningError(f"failed to initialize PI0.5 runner: {exc}") from exc
        self._pi05_runner_server.start_background()
        if not self._pi05_runner_healthcheck():
            self._pi05_runner_server.close()
            self._pi05_runner_server = None
            raise AgentReasoningError(
                f"PI0.5 runner did not start at {self.config.pi05_runner_url}"
            )

    def _pi05_runner_healthcheck(self) -> bool:
        try:
            data = _get_json(f"{self.config.pi05_runner_url}/health", timeout_s=1.0)
        except AgentReasoningError:
            return False
        return data.get("ok") is True

    def _run_step_via_runner(self, step: PlannedStep) -> int:
        if not self._rest_client_injected:
            self._ensure_pi05_runner_service()
        payload = {
            "task": step.task,
            "episode_time": step.episode_time or self.config.episode_time,
        }
        if step.episodes is not None:
            payload["episodes"] = step.episodes
        result = self._rest_client(
            f"{self.config.pi05_runner_url}/run",
            payload,
            float((step.episode_time or self.config.episode_time) + 120),
        )
        self._last_execution_result = dict(result)
        return int(result.get("exit_code", 0))

    def interrupt_current_run(self, reason: str) -> None:
        """Request cooperative cancellation of the active warm PI0.5 run."""
        if not self.config.use_pi05_runner:
            raise AgentReasoningError(
                "live interruption requires the warm PI0.5 runner"
            )
        if not self._rest_client_injected:
            self._ensure_pi05_runner_service()
        self._rest_client(
            f"{self.config.pi05_runner_url}/stop",
            {"reason": reason or "operator interruption"},
            5.0,
        )

    def _remember_turn(self, user: str, assistant: str) -> None:
        if self.config.chat_history_turns <= 0:
            return
        self._history.append({"role": "user", "content": user})
        self._history.append({"role": "assistant", "content": assistant})
        max_items = self.config.chat_history_turns * 2
        if len(self._history) > max_items:
            self._history = self._history[-max_items:]

    def run_step(self, step: PlannedStep) -> int:
        if self.config.use_pi05_runner:
            return self._run_step_via_runner(step)
        command = build_inference_command(self.config, step)
        try:
            result = self._runner(command, cwd=str(REPO_ROOT), text=True)
        except OSError as exc:
            raise AgentReasoningError(f"failed to start PI0.5 inference: {exc}") from exc
        self._last_execution_result = {"exit_code": int(result.returncode)}
        return int(result.returncode)

    def run_plan(self, plan: AgentPlan) -> int:
        if not plan.steps:
            raise AgentReasoningError("System 2 returned no executable steps")
        for step in plan.steps:
            code = self.run_step(step)
            if code != 0:
                return code
        return 0

    def last_execution_result(self) -> dict[str, Any]:
        return dict(self._last_execution_result)


def build_inference_command(
    config: AgentReasoningConfig,
    step: PlannedStep,
) -> list[str]:
    """Build the existing PI0.5 inference command for one planned step."""
    command = [
        "uv",
        "run",
        "python",
        "scripts/infer_pi05_finetuned.py",
        "--policy-path",
        str(config.policy_path),
        "--task",
        step.task,
        "--camera-backend",
        config.camera_backend,
        "--episodes",
        str(step.episodes or 1),
        "--episode-time",
        str(step.episode_time or config.episode_time),
        "--fps",
        str(config.fps),
        "--settle-steps",
        str(config.settle_steps),
        "--device",
        config.device,
    ]
    if config.gripper_max_relative_target is not None:
        command.extend(
            ["--gripper-max-relative-target", str(config.gripper_max_relative_target)]
        )
    if not config.show_cameras:
        command.append("--no-show-cameras")
    command.extend(config.extra_infer_args)
    return command


def build_pi05_runner_options(config: AgentReasoningConfig):
    """Build typed options for the embedded warm PI0.5 runtime."""
    if config.extra_infer_args:
        raise AgentReasoningError(
            "extra inference CLI args are only supported by the subprocess fallback; "
            "use --no-warm-pi05-runner or add the option to PI05InferenceOptions"
        )

    from openpibot.pi05_inference_runtime import PI05InferenceOptions

    return PI05InferenceOptions(
        policy_path=config.policy_path,
        camera_backend=config.camera_backend,
        episodes=1,
        episode_time=config.episode_time,
        fps=config.fps,
        settle_steps=config.settle_steps,
        device=config.device,
        gripper_max_relative_target=config.gripper_max_relative_target,
        show_cameras=config.show_cameras,
        print_startup_summary=False,
    )


def plan_sync(reasoner: AgentReasoner, query: str) -> AgentPlan:
    return asyncio.run(reasoner.plan(query))


def turn_sync(reasoner: AgentReasoner, query: str) -> AgentTurn:
    return asyncio.run(reasoner.respond(query))


def _parse_agent_turn(payload: dict[str, Any]) -> AgentTurn:
    mode = payload.get("mode")
    message = str(payload.get("message") or "").strip()
    if mode == "plan":
        plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
        steps = [
            PlannedStep(
                task=str(item.get("task") or "").strip(),
                reason=str(item.get("reason") or "").strip(),
            )
            for item in plan.get("steps", [])
            if isinstance(item, dict) and str(item.get("task") or "").strip()
        ]
        return AgentTurn(
            mode="plan",
            message=message,
            summary=str(plan.get("summary") or "").strip(),
            steps=steps,
            safety_notes=[
                str(note).strip()
                for note in plan.get("safety_notes", [])
                if str(note).strip()
            ],
        )
    if mode == "chat":
        return AgentTurn(mode="chat", message=message or "I am ready.")
    raise AgentReasoningError("Pi System 2 agent returned an unknown mode")


def _parse_bridge_turn(stdout: str) -> AgentTurn:
    try:
        return _parse_agent_turn(json.loads(stdout))
    except json.JSONDecodeError as exc:
        raise AgentReasoningError("Pi System 2 agent returned invalid JSON") from exc


def _parse_plan_progress(payload: dict[str, Any]) -> PlanProgress:
    if payload.get("mode") != "progress":
        raise AgentReasoningError("Pi System 2 agent did not track execution progress")
    progress = payload.get("progress") if isinstance(payload.get("progress"), dict) else {}
    action = str(progress.get("next_action") or "continue").strip().lower()
    if action not in ("continue", "retry", "done", "blocked"):
        action = "continue"
    items: list[PlanItemProgress] = []
    for item in progress.get("item_statuses", []):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        status = str(item.get("status") or "").strip().lower()
        if status not in ("pending", "completed", "blocked"):
            continue
        items.append(
            PlanItemProgress(
                index=index,
                status=status,  # type: ignore[arg-type]
                note=str(item.get("note") or "").strip(),
            )
        )
    if not items:
        raise AgentReasoningError("Pi System 2 agent returned empty progress tracking")
    updates: list[PlanTaskUpdate] = []
    for item in progress.get("updated_tasks", []):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        task = str(item.get("task") or "").strip()
        if index < 1 or not task:
            continue
        updates.append(PlanTaskUpdate(index=index, task=task))
    return PlanProgress(
        message=str(payload.get("message") or progress.get("message") or "").strip(),
        next_action=action,  # type: ignore[arg-type]
        item_statuses=items,
        updated_tasks=updates,
    )


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    return _read_json_response(request, timeout_s)


def _get_json(url: str, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    return _read_json_response(request, timeout_s)


def _read_json_response(
    request: urllib.request.Request,
    timeout_s: float,
) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise AgentReasoningError(
            f"Pi System 2 agent rejected request: {detail or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise AgentReasoningError(f"Pi System 2 agent is unreachable: {exc.reason}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AgentReasoningError("Pi System 2 agent returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise AgentReasoningError("Pi System 2 agent returned invalid JSON")
    return parsed


def _agent_host_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return parsed.hostname or "127.0.0.1"


def _agent_port_from_url(url: str) -> int:
    parsed = urllib.parse.urlparse(url)
    if parsed.port is None:
        return 443 if parsed.scheme == "https" else 80
    return parsed.port


def _turn_history_text(turn: AgentTurn) -> str:
    if turn.mode == "plan":
        tasks = "; ".join(step.task for step in turn.steps)
        return f"{turn.message or 'Staged a PI0.5 plan.'} Plan: {turn.summary}. Steps: {tasks}"
    return turn.message


def _pi_thinking_level(value: str) -> str:
    if value == "none":
        return "off"
    return value
