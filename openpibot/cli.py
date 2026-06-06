"""OpenPiBot command line interface."""

from __future__ import annotations

import os
import pathlib
import shutil
import socket
import subprocess
import sys
from typing import Annotated

import typer
import uvicorn
from rich.console import Console
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
    build_dashboard: bool = typer.Option(
        True,
        "--build-dashboard/--no-build-dashboard",
        help="Build dashboard/frontend before starting the backend.",
    ),
) -> None:
    """Start the production OpenPiBot dashboard and API server."""
    if build_dashboard:
        _build_dashboard()

    log_file = configure_logging()
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
        console.print(f"[dim]Logs:[/dim] {log_file}")
        uvicorn.run(
            "openpibot.server.app:create_app",
            factory=True,
            host=host,
            port=candidate,
            reload=reload,
            log_level=log_level,
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
