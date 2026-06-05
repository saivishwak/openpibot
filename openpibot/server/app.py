"""FastAPI application factory for OpenPIBot."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from openpibot import __version__
from openpibot.server.config import REPO_ROOT, STATIC_ROOT
from openpibot.server.logging import configure_logging
from openpibot.server.routers import cameras, config, doctor, jobs, logs, vr

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    log.info("OpenPIBot server starting")
    try:
        yield
    finally:
        log.info("OpenPIBot server shutting down")
        try:
            from openpibot.server.runtime import cameras as cam_mod

            cam_mod.reset_streams()
        except Exception:
            log.exception("camera cleanup failed")
        try:
            from openpibot.server.runtime import vr_teleop as vr_mod

            vr_mod.SESSION.emergency_stop()
        except Exception:
            log.exception("VR/motor cleanup failed")


def create_app() -> FastAPI:
    app = FastAPI(
        title="OpenPIBot",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
        log.warning("HTTP %s %s: %s", exc.status_code, request.url.path, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"http_{exc.status_code}",
                    "message": str(exc.detail),
                    "details": None,
                    "request_id": request_id,
                }
            },
        )

    @app.exception_handler(Exception)
    async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
        log.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "details": None,
                    "request_id": request_id,
                }
            },
        )

    app.include_router(config.router)
    app.include_router(doctor.router)
    app.include_router(cameras.router)
    app.include_router(vr.router)
    app.include_router(jobs.router)
    app.include_router(logs.router)

    xlerobot_root = REPO_ROOT / "vendor" / "xlerobot"
    app.mount("/robot_assets/xlerobot", StaticFiles(directory=xlerobot_root, check_dir=False), name="xlerobot-assets")
    app.mount("/assets", StaticFiles(directory=STATIC_ROOT / "assets", check_dir=False), name="assets")

    @app.get("/status")
    def server_status() -> dict:
        return {"status": "ok", "name": "OpenPIBot", "version": __version__}

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        target = STATIC_ROOT / path
        if path and target.is_file():
            return FileResponse(target)
        index = STATIC_ROOT / "index.html"
        if index.is_file():
            return FileResponse(index)
        return HTMLResponse(
            "<h1>Frontend not built</h1>"
            "<p>Run <code>uv run openpibot run</code> to build and serve the dashboard, "
            "or build manually with <code>pnpm --dir dashboard/frontend build</code>.</p>",
            status_code=503,
        )

    return app


app = create_app()
