"""Embeddable warm PI0.5 inference runner service."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from openpibot.server.config import REPO_ROOT


SCRIPTS_DIR = REPO_ROOT / "scripts"


def _ensure_script_import_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))


def _load_runtime_symbols():
    _ensure_script_import_path()
    from openpibot.pi05_inference_runtime import (
        PI05InferenceOptions,
        PI05InferenceRuntime,
        _parse_args,
        inference_options_from_args,
    )

    return PI05InferenceOptions, PI05InferenceRuntime, _parse_args, inference_options_from_args


def runtime_options_from_infer_args(infer_args: list[str]) -> Any:
    """Parse standalone runner CLI passthrough into typed inference options."""
    _, _, parse_infer_args, options_from_args = _load_runtime_symbols()
    return options_from_args(parse_infer_args(infer_args, require_task=False))


def parse_runner_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a warm PI0.5 robot inference runtime over local HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args, infer_args = parser.parse_known_args(argv)
    if not infer_args:
        parser.error("pass PI0.5 inference arguments after the server options")
    args.infer_args = infer_args
    return args


def _json_response(handler: BaseHTTPRequestHandler, status: int, body: dict[str, Any]) -> None:
    raw = json.dumps(body).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


class RunnerState:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.lock = threading.Lock()


def _handler_class(state: RunnerState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("pi05-runner: " + (format % args) + "\n")

        def do_GET(self) -> None:
            if self.path != "/health":
                _json_response(self, 404, {"error": "not found"})
                return
            _json_response(self, 200, {"ok": True, "service": "openpibot-pi05-runner"})

        def do_POST(self) -> None:
            if self.path == "/stop":
                length = int(self.headers.get("Content-Length") or "0")
                payload: dict[str, Any] = {}
                if length:
                    try:
                        parsed = json.loads(self.rfile.read(length).decode("utf-8"))
                    except json.JSONDecodeError:
                        _json_response(self, 400, {"error": "invalid JSON body"})
                        return
                    if isinstance(parsed, dict):
                        payload = parsed
                reason = str(payload.get("reason") or "operator interruption")
                result = state.runtime.request_stop(reason)
                _json_response(self, 200, result)
                return
            if self.path != "/run":
                _json_response(self, 404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or "0")
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                _json_response(self, 400, {"error": "invalid JSON body"})
                return
            task = str(payload.get("task") or "").strip()
            if not task:
                _json_response(self, 400, {"error": "task is required"})
                return
            if not state.lock.acquire(blocking=False):
                _json_response(self, 409, {"error": "PI0.5 runner is already executing"})
                return
            try:
                result = state.runtime.run_task(
                    task,
                    episodes=payload.get("episodes"),
                    episode_time=payload.get("episode_time"),
                )
            except Exception as exc:
                _json_response(self, 500, {"ok": False, "error": str(exc)})
            else:
                interrupted = str(result.get("stop_reason") or "").startswith(
                    "interrupted"
                )
                _json_response(
                    self,
                    200,
                    result | {"exit_code": 130 if interrupted else 0},
                )
            finally:
                state.lock.release()

    return Handler


class PI05RunnerServer:
    """Owns a warm PI0.5 runtime and serves it from a background HTTP thread."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        runtime_options: Any | None = None,
        infer_args: list[str] | None = None,
    ) -> None:
        _, runtime_cls, _, _ = _load_runtime_symbols()
        if runtime_options is None:
            if infer_args is None:
                raise ValueError("runtime_options or infer_args is required")
            runtime_options = runtime_options_from_infer_args(infer_args)
        if not runtime_options.policy_path:
            raise ValueError("--policy-path is required")
        self.runtime = runtime_cls(runtime_options)
        self.server = ThreadingHTTPServer(
            (host, int(port)),
            _handler_class(RunnerState(self.runtime)),
        )
        self.host = host
        self.port = int(port)
        self._thread: threading.Thread | None = None
        self._closed = False

    def serve_forever(self) -> None:
        print(f"OpenPiBot PI0.5 runner listening on http://{self.host}:{self.port}")
        try:
            self.server.serve_forever()
        finally:
            self.close()

    def start_background(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self.server.serve_forever,
            name="openpibot-pi05-runner",
            daemon=True,
        )
        self._thread.start()
        print(f"OpenPiBot PI0.5 runner listening on http://{self.host}:{self.port}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if (
            self._thread is not None
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self.server.shutdown()
        self.server.server_close()
        self.runtime.close()


def serve_runner(
    *,
    host: str,
    port: int,
    runtime_options: Any | None = None,
    infer_args: list[str] | None = None,
) -> None:
    server = PI05RunnerServer(
        host=host,
        port=port,
        runtime_options=runtime_options,
        infer_args=infer_args,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.close()
