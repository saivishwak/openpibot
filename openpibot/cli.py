"""OpenPiBot command line interface."""

from __future__ import annotations

import asyncio
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import Annotated

import typer
import uvicorn
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from openpibot import __version__
from openpibot.server.config import REPO_ROOT, get_robot_profiles, load_project_config
from openpibot.server.logging import configure_logging
from openpibot.server.runtime.openpi_policy import build_openpi_policy_server_command

app = typer.Typer(
    name="openpibot",
    help="OpenPiBot robot dashboard, VR teleop, recording, and PI0.5 tooling.",
    no_args_is_help=True,
)
dataset_app = typer.Typer(help="Dataset utilities.")
app.add_typer(dataset_app, name="dataset")
console = Console()
MAX_AUTONOMOUS_ATTEMPTS_PER_ITEM = 2


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"openpibot {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    """OpenPiBot CLI root."""


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return False
        except OSError:
            return True


def _run_script(args: list[str], dry_run: bool = False) -> None:
    console.print("[bold]Running:[/bold] " + " ".join(args))
    if dry_run:
        return
    raise typer.Exit(subprocess.run(args, cwd=str(REPO_ROOT)).returncode)


def _read_available_stdin_line(timeout_s: float) -> str | None:
    try:
        import select

        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    except (OSError, ValueError, AttributeError):
        time.sleep(timeout_s)
        return None
    if not ready:
        return None
    line = sys.stdin.readline()
    if line == "":
        return ""
    return line.strip()


@contextmanager
def _execution_input_mode():
    if not sys.stdin.isatty():
        yield None
        return
    try:
        import termios
        import tty
    except ImportError:
        yield None
        return
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def _read_execution_input(
    *,
    fd: int | None,
    buffer: str,
    timeout_s: float,
    echo: bool = False,
) -> tuple[str | None, str]:
    if fd is None:
        line = _read_available_stdin_line(timeout_s)
        return line, buffer
    try:
        import select

        ready, _, _ = select.select([fd], [], [], timeout_s)
    except (OSError, ValueError):
        time.sleep(timeout_s)
        return None, buffer
    if not ready:
        return None, buffer
    try:
        chunk = os.read(fd, 1024).decode("utf-8", errors="ignore")
    except BlockingIOError:
        return None, buffer
    if "\x03" in chunk:
        raise KeyboardInterrupt
    for char in chunk:
        if char in ("\r", "\n"):
            line = buffer.strip()
            if echo:
                sys.stdout.write("\n")
                sys.stdout.flush()
            return line, ""
        if char in ("\x7f", "\b"):
            if buffer:
                buffer = buffer[:-1]
                if echo:
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            continue
        if char.isprintable() or char == " ":
            buffer += char
            if echo:
                sys.stdout.write(char)
                sys.stdout.flush()
    return None, buffer


def _run_plan_interactively(reasoner: object, plan: object) -> tuple[int, str | None]:
    result: dict[str, object] = {}

    def worker() -> None:
        try:
            result["code"] = reasoner.run_plan(plan)  # type: ignore[attr-defined]
        except BaseException as exc:  # pragma: no cover - surfaced on main thread
            result["error"] = exc

    thread = threading.Thread(target=worker, name="openpibot-agent-execution")
    thread.start()
    console.print(
        "[dim]Execution running. Type a new instruction and press Enter to interrupt, or press Ctrl-C to stop.[/dim]"
    )
    pending_query: str | None = None
    input_buffer = ""
    last_prompt_t = 0.0
    try:
        with _execution_input_mode() as fd:
            while thread.is_alive():
                now = time.monotonic()
                if fd is not None and (now - last_prompt_t) >= 5.0:
                    sys.stdout.write("\ninterrupt> " if last_prompt_t else "interrupt> ")
                    sys.stdout.flush()
                    last_prompt_t = now
                query, input_buffer = _read_execution_input(
                    fd=fd,
                    buffer=input_buffer,
                    timeout_s=0.2,
                    echo=fd is not None,
                )
                if query is None:
                    continue
                if not query:
                    continue
                pending_query = query
                console.print("[yellow]Interrupting current execution...[/yellow]")
                try:
                    reasoner.interrupt_current_run(query)  # type: ignore[attr-defined]
                except Exception as exc:
                    console.print(f"[red]Interrupt failed:[/red] {escape(str(exc))}")
                break
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupting current execution...[/yellow]")
        try:
            reasoner.interrupt_current_run("keyboard interrupt")  # type: ignore[attr-defined]
        except Exception as exc:
            console.print(f"[red]Interrupt failed:[/red] {escape(str(exc))}")

    while thread.is_alive():
        thread.join(timeout=0.2)

    error = result.get("error")
    if error is not None:
        raise error
    return int(result.get("code", 0)), pending_query


def _execution_result_reason(reasoner: object, exit_code: int) -> str:
    getter = getattr(reasoner, "last_execution_result", None)
    if callable(getter):
        try:
            result = getter()
        except Exception:
            result = {}
        if isinstance(result, dict):
            reason = str(result.get("stop_reason") or result.get("error") or "").strip()
            if reason:
                return reason
    return f"exit_code={exit_code}"


def _replace_step_task(step: object, task: str) -> object:
    step_type = type(step)
    kwargs = {
        "task": task,
        "reason": getattr(step, "reason", "") or "Replanned after visual review.",
    }
    if hasattr(step, "episode_time"):
        kwargs["episode_time"] = getattr(step, "episode_time")
    if hasattr(step, "episodes"):
        kwargs["episodes"] = getattr(step, "episodes")
    try:
        return step_type(**kwargs)
    except TypeError:
        return step


def _apply_progress_task_updates(
    *,
    reasoner: object,
    plan: object,
    steps: list[object],
    statuses: list[str],
    attempts: list[int],
    progress: object,
) -> None:
    updates = list(getattr(progress, "updated_tasks", []))
    if not updates:
        return
    max_items = int(getattr(getattr(reasoner, "config", None), "max_steps", 12) or 12)
    for update in updates:
        idx = int(getattr(update, "index", 0)) - 1
        task = str(getattr(update, "task", "")).strip()
        if not task:
            continue
        if 0 <= idx < len(steps):
            if statuses[idx] == "completed":
                continue
            steps[idx] = _replace_step_task(steps[idx], task)
            console.print(f"[dim]Replanned item {idx + 1}: {escape(task)}[/dim]")
        elif idx == len(steps) and len(steps) < max_items:
            if not steps:
                continue
            steps.append(_replace_step_task(steps[-1], task))
            statuses.append("pending")
            attempts.append(0)
            console.print(f"[dim]Added item {idx + 1}: {escape(task)}[/dim]")
    try:
        plan.steps = steps  # type: ignore[attr-defined]
    except Exception:
        pass


def _run_plan_to_completion_interactively(
    reasoner: object,
    plan: object,
    *,
    original_query: str,
    max_attempts_per_item: int = MAX_AUTONOMOUS_ATTEMPTS_PER_ITEM,
) -> tuple[int, str | None]:
    steps = list(getattr(plan, "steps", []))
    if not steps:
        return 1, None
    statuses = ["pending"] * len(steps)
    attempts = [0] * len(steps)
    plan_cls = type(plan)
    current = 0

    while True:
        while current < len(steps) and statuses[current] == "completed":
            current += 1
        if current >= len(steps):
            return 0, None

        attempts[current] += 1
        single_step_plan = plan_cls(
            summary=getattr(plan, "summary", ""),
            steps=[steps[current]],
            safety_notes=getattr(plan, "safety_notes", []),
        )
        console.print(
            f"[bold]Executing item {current + 1}/{len(steps)}:[/bold] "
            f"{escape(getattr(steps[current], 'task', ''))}"
        )
        code, pending_query = _run_plan_interactively(reasoner, single_step_plan)
        if pending_query is not None:
            return code, pending_query
        if code == 130:
            return code, None

        progress = asyncio.run(
            reasoner.review_plan_progress(  # type: ignore[attr-defined]
                original_query=original_query,
                plan=plan,
                statuses=statuses,
                current_index=current,
                last_exit_code=code,
                attempt=attempts[current],
                result_reason=_execution_result_reason(reasoner, code),
            )
        )
        if progress.message:
            console.print(escape(progress.message))
        for item in progress.item_statuses:
            idx = int(item.index) - 1
            if 0 <= idx < len(statuses):
                statuses[idx] = item.status
                note = f" - {escape(item.note)}" if item.note else ""
                console.print(f"  {idx + 1}. {item.status}{note}")
        _apply_progress_task_updates(
            reasoner=reasoner,
            plan=plan,
            steps=steps,
            statuses=statuses,
            attempts=attempts,
            progress=progress,
        )

        if all(status == "completed" for status in statuses):
            return 0, None
        if progress.next_action == "done":
            return 0, None
        if progress.next_action == "blocked" or any(
            status == "blocked" for status in statuses
        ):
            return 2, None

        if statuses[current] != "completed" or progress.next_action == "retry":
            if attempts[current] < max_attempts_per_item:
                console.print(
                    f"[yellow]Retrying item {current + 1}/{len(steps)} after visual review.[/yellow]"
                )
                continue
            console.print(
                f"[red]Item {current + 1} did not complete after "
                f"{max_attempts_per_item} attempt(s).[/red]"
            )
            return 3, None

        current += 1


def _build_dashboard() -> None:
    frontend_dir = REPO_ROOT / "dashboard" / "frontend"
    pnpm_store = REPO_ROOT / ".pnpm-store"
    if not (frontend_dir / "package.json").is_file():
        console.print(f"[yellow]Dashboard source not found:[/yellow] {frontend_dir}")
        return
    if shutil.which("pnpm") is None:
        raise typer.BadParameter("pnpm is required to build the dashboard. Install pnpm or run with --no-build-dashboard.")

    commands = [
        ["pnpm", "install", "--frozen-lockfile", "--store-dir", str(pnpm_store)],
        ["pnpm", "build"],
    ]
    env = {**os.environ, "CI": "true"}
    for command in commands:
        console.print("[bold]Dashboard:[/bold] " + " ".join(command))
        result = subprocess.run(command, cwd=str(frontend_dir), env=env)
        if result.returncode != 0:
            raise typer.Exit(result.returncode)


@app.command()
def run(
    host: str = typer.Option("0.0.0.0", help="Host interface to bind."),
    port: int = typer.Option(5000, help="HTTP port for the dashboard/API."),
    reload: bool = typer.Option(False, "--reload", help="Enable uvicorn reload for development."),
    log_level: str = typer.Option("info", help="Uvicorn log level."),
    log_file: pathlib.Path | None = typer.Option(
        None,
        "--log-file",
        help="Write OpenPiBot server logs to this file.",
    ),
    build_dashboard: bool = typer.Option(
        True,
        "--build-dashboard/--no-build-dashboard",
        help="Build dashboard/frontend before starting the backend.",
    ),
) -> None:
    """Start the production OpenPiBot dashboard and API server."""
    if build_dashboard:
        _build_dashboard()

    log_file_path = configure_logging(level=log_level, log_file=log_file)
    ports = [port]
    if port in (80, 5000, 8000, 8020):
        ports.extend(p for p in range(8021, 8040) if p != port)

    for candidate in ports:
        if _port_in_use(host, candidate):
            console.print(f"[yellow]Port {candidate} is in use; trying next.[/yellow]")
            continue
        os.environ["OPENPIBOT_HOST"] = host
        os.environ["OPENPIBOT_PORT"] = str(candidate)
        console.print(f"[green]OpenPiBot dashboard:[/green] http://127.0.0.1:{candidate}")
        console.print(f"[dim]Logs:[/dim] {log_file_path}")
        uvicorn.run(
            "openpibot.server.app:create_app",
            factory=True,
            host=host,
            port=candidate,
            reload=reload,
            log_level=log_level,
            log_config=None,
        )
        return

    raise typer.BadParameter("No available port found in the configured range.")


@app.command()
def info() -> None:
    """Print project, config, and robot profile information."""
    cfg = load_project_config()
    profiles = get_robot_profiles(cfg)
    table = Table(title="OpenPiBot")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("Version", __version__)
    table.add_row("Repo root", str(REPO_ROOT))
    table.add_row("Config", str(REPO_ROOT / "config" / "xlerobot.yaml"))
    table.add_row("Robot profiles", ", ".join(p["id"] for p in profiles) or "(none)")
    table.add_row("Dataset repo", str((cfg.get("dataset") or {}).get("repo_id") or ""))
    table.add_row("PI0.5 server", f"{(cfg.get('pi05') or {}).get('server_host', '127.0.0.1')}:{(cfg.get('pi05') or {}).get('server_port', 8000)}")
    console.print(table)


@app.command()
def doctor(json: bool = typer.Option(False, "--json", help="Print raw JSON instead of a table.")) -> None:
    """Run local diagnostics."""
    from openpibot.server.runtime.doctor import run_doctor

    checks = run_doctor()
    if json:
        import json as json_mod

        console.print(json_mod.dumps(checks, indent=2))
        return
    table = Table(title="Diagnostics")
    table.add_column("Status")
    table.add_column("Check")
    table.add_column("Detail")
    for check in checks:
        table.add_row(str(check["status"]), str(check["name"]), str(check["detail"]))
    console.print(table)


@app.command()
def train(
    args: list[str] = typer.Argument(None, help="Arguments passed to scripts/finetune_pi05.py."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the command only."),
) -> None:
    """Fine-tune PI0.5 using the existing LeRobot wrapper."""
    _run_script(["uv", "run", "python", "scripts/finetune_pi05.py", *(args or [])], dry_run=dry_run)


@app.command()
def infer(
    args: list[str] = typer.Argument(None, help="Arguments passed to scripts/infer_pi05_finetuned.py."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the command only."),
) -> None:
    """Run local finetuned PI0.5 inference on the robot."""
    _run_script(["uv", "run", "python", "scripts/infer_pi05_finetuned.py", *(args or [])], dry_run=dry_run)


@app.command()
def agent(
    policy_path: pathlib.Path = typer.Option(
        ...,
        "--policy-path",
        help="Path to finetuned checkpoint (.../pretrained_model).",
    ),
    model: str = typer.Option(
        "gpt-5.4",
        "--model",
        envvar="OPENPIBOT_AGENT_MODEL",
        help="OpenAI model used as the System 2 VLM.",
    ),
    vision_camera: str = typer.Option(
        "head",
        "--vision-camera",
        help="Camera role used for mandatory System 2 image grounding.",
    ),
    reasoning_effort: str = typer.Option(
        "medium",
        "--reasoning-effort",
        help="System 2 Pi thinking level: none, low, medium, high, or xhigh.",
    ),
    llm_timeout: int = typer.Option(
        60,
        "--llm-timeout",
        help="Seconds before a System 2 planning call times out.",
    ),
    max_steps: int = typer.Option(
        6,
        "--max-steps",
        help="Maximum PI0.5 subgoals System 2 may emit.",
    ),
    chat_history_turns: int = typer.Option(
        12,
        "--chat-history-turns",
        help="Number of recent REPL turns injected into each System 2 prompt.",
    ),
    pi05_runner_url: str = typer.Option(
        "http://127.0.0.1:8767",
        "--pi05-runner-url",
        help="Local warm PI0.5 runner REST URL.",
    ),
    warm_pi05_runner: bool = typer.Option(
        True,
        "--warm-pi05-runner/--no-warm-pi05-runner",
        help="Load PI0.5 once at agent startup and execute via the warm REST runner.",
    ),
    camera_backend: str = typer.Option(
        "dashboard",
        "--camera-backend",
        help="Camera backend passed to the PI0.5 inference script.",
    ),
    episode_time: int = typer.Option(120, "--episode-time", help="Default seconds per planned step."),
    fps: int = typer.Option(30, "--fps", help="Control loop Hz passed to inference."),
    settle_steps: int = typer.Option(30, "--settle-steps", help="Settle steps passed to inference."),
    gripper_max_relative_target: float | None = typer.Option(
        15.0,
        "--gripper-max-relative-target",
        help="Gripper max-relative override passed to inference.",
    ),
    device: str = typer.Option("cuda", "--device", help="Inference device."),
    show_cameras: bool = typer.Option(
        False,
        "--show-cameras/--no-show-cameras",
        help="Show camera preview during PI0.5 inference.",
    ),
    extra_infer_args: list[str] = typer.Argument(
        None,
        help=(
            "Additional inference CLI arguments. Supported only with "
            "--no-warm-pi05-runner."
        ),
    ),
) -> None:
    """Start the image-grounded System 2 agent REPL for PI0.5 inference."""
    from openpibot.agent_reasoning import (
        AgentReasoner,
        AgentReasoningConfig,
        AgentReasoningError,
        turn_sync,
    )

    if vision_camera not in {"head", "left_wrist", "right_wrist"}:
        raise typer.BadParameter("--vision-camera must be head, left_wrist, or right_wrist")
    if reasoning_effort not in {"none", "low", "medium", "high", "xhigh"}:
        raise typer.BadParameter("--reasoning-effort must be none, low, medium, high, or xhigh")
    if camera_backend not in {"dashboard", "lerobot"}:
        raise typer.BadParameter("--camera-backend must be dashboard or lerobot")
    if device not in {"cuda", "cpu", "mps"}:
        raise typer.BadParameter("--device must be cuda, cpu, or mps")
    if warm_pi05_runner and extra_infer_args:
        raise typer.BadParameter(
            "extra inference arguments require --no-warm-pi05-runner; "
            "the warm runner uses typed PI05InferenceOptions"
        )
    if not policy_path.is_dir():
        raise typer.BadParameter(f"policy path not found: {policy_path}")
    if not os.environ.get("OPENAI_API_KEY"):
        raise typer.BadParameter("OPENAI_API_KEY is required for the System 2 VLM")

    config = AgentReasoningConfig(
        policy_path=policy_path.resolve(),
        model=model,
        vision_camera=vision_camera,  # type: ignore[arg-type]
        reasoning_effort=reasoning_effort,
        llm_timeout_s=llm_timeout,
        max_steps=max_steps,
        chat_history_turns=chat_history_turns,
        camera_backend=camera_backend,
        episode_time=episode_time,
        fps=fps,
        settle_steps=settle_steps,
        gripper_max_relative_target=gripper_max_relative_target,
        device=device,
        show_cameras=show_cameras,
        extra_infer_args=tuple(extra_infer_args or ()),
        pi05_runner_url=pi05_runner_url,
        use_pi05_runner=warm_pi05_runner,
        pi05_runner_autostart=warm_pi05_runner,
    )
    reasoner = AgentReasoner(config)
    console.print("[green]Tess ready.[/green]")
    if warm_pi05_runner:
        try:
            console.print("[dim]Starting warm PI0.5 runner...[/dim]")
            reasoner.ensure_system1_ready()
            console.print("[green]PI0.5 runner ready.[/green]")
        except AgentReasoningError as exc:
            console.print(f"[red]PI0.5 runner failed:[/red] {exc}")
            raise typer.Exit(1)
    console.print("[dim]Enter a physical task, or type exit/quit.[/dim]")

    pending_query: str | None = None
    while True:
        if pending_query is None:
            try:
                query = typer.prompt("agent").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return
        else:
            query = pending_query
            pending_query = None
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            return

        try:
            turn = turn_sync(reasoner, query)
        except AgentReasoningError as exc:
            console.print(f"[red]Planning failed:[/red] {exc}")
            continue

        if turn.mode == "chat":
            console.print(escape(turn.message))
            continue

        plan = turn.as_plan()
        if turn.message:
            console.print(escape(turn.message))
        console.print(f"[bold]Plan:[/bold] {escape(plan.summary or '(no summary)')}")
        if plan.safety_notes:
            console.print(
                "[bold]Safety notes:[/bold] "
                + escape("; ".join(plan.safety_notes))
            )
        if not plan.steps:
            console.print("[yellow]No executable PI0.5 steps were returned.[/yellow]")
            continue
        for idx, step in enumerate(plan.steps, start=1):
            reason = f" [dim]({escape(step.reason)})[/dim]" if step.reason else ""
            console.print(f"  {idx}. {escape(step.task)}{reason}")

        confirm = typer.prompt("Execute this plan? Type yes").strip().lower()
        if confirm != "yes":
            console.print("[yellow]Skipped.[/yellow]")
            continue
        try:
            code, pending_query = _run_plan_to_completion_interactively(
                reasoner,
                plan,
                original_query=query,
            )
        except AgentReasoningError as exc:
            console.print(f"[red]Execution blocked:[/red] {exc}")
            continue
        except BaseException as exc:
            console.print(f"[red]Execution failed:[/red] {escape(str(exc))}")
            continue
        if pending_query is not None:
            console.print("[yellow]Execution interrupted.[/yellow]")
            continue
        if code != 0:
            console.print(f"[red]Execution stopped with exit code {code}.[/red]")
        else:
            console.print("[green]Plan execution complete.[/green]")


@app.command("pi05-server")
def pi05_server(dry_run: bool = typer.Option(False, "--dry-run", help="Print the command only.")) -> None:
    """Start the package-managed OpenPI PI0.5 policy server."""
    _run_script(build_openpi_policy_server_command(), dry_run=dry_run)


@dataset_app.command("push")
def dataset_push(
    args: list[str] = typer.Argument(None, help="Arguments passed to scripts/push_dataset.py."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the command only."),
) -> None:
    """Push the configured LeRobot dataset to Hugging Face Hub."""
    _run_script(["uv", "run", "python", "scripts/push_dataset.py", *(args or [])], dry_run=dry_run)


if __name__ == "__main__":
    app()
