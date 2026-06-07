from __future__ import annotations

from fastapi import APIRouter

from openpibot.server.logging import current_log_file

router = APIRouter(prefix="/api/logs", tags=["logs"])


@router.get("/recent")
def recent_logs(lines: int = 200) -> dict:
    max_lines = max(1, min(int(lines), 1000))
    log_file = current_log_file()
    if not log_file.is_file():
        return {"path": str(log_file), "lines": []}
    content = log_file.read_text(errors="replace").splitlines()
    return {"path": str(log_file), "lines": content[-max_lines:]}
