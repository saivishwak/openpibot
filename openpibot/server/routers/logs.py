from __future__ import annotations

from fastapi import APIRouter

from openpibot.server.logging import LOG_FILE

router = APIRouter(prefix="/api/logs", tags=["logs"])


@router.get("/recent")
def recent_logs(lines: int = 200) -> dict:
    max_lines = max(1, min(int(lines), 1000))
    if not LOG_FILE.is_file():
        return {"path": str(LOG_FILE), "lines": []}
    content = LOG_FILE.read_text(errors="replace").splitlines()
    return {"path": str(LOG_FILE), "lines": content[-max_lines:]}

