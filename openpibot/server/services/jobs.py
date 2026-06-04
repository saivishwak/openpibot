"""Small subprocess job manager for training/inference actions."""

from __future__ import annotations

import dataclasses
import subprocess
import threading
import time
import uuid
from collections import deque
from typing import Literal

from openpibot.server.config import REPO_ROOT

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


@dataclasses.dataclass
class Job:
    id: str
    command: list[str]
    status: JobStatus = "queued"
    returncode: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    log: deque[str] = dataclasses.field(default_factory=lambda: deque(maxlen=1000))
    process: subprocess.Popen[str] | None = None

    def public(self) -> dict:
        return {
            "id": self.id,
            "command": self.command,
            "status": self.status,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "log": list(self.log),
        }


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}

    def start(self, command: list[str]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], command=command)
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def list(self) -> list[dict]:
        with self._lock:
            return [j.public() for j in self._jobs.values()]

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.public() if job else None

    def cancel(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            process = job.process
        if process and process.poll() is None:
            process.terminate()
            job.status = "cancelled"
        return job.public()

    def _run(self, job: Job) -> None:
        job.status = "running"
        job.started_at = time.time()
        try:
            process = subprocess.Popen(
                job.command,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            job.process = process
            assert process.stdout is not None
            for line in process.stdout:
                job.log.append(line.rstrip())
            job.returncode = process.wait()
            if job.status != "cancelled":
                job.status = "succeeded" if job.returncode == 0 else "failed"
        except Exception as exc:
            job.log.append(f"{type(exc).__name__}: {exc}")
            job.status = "failed"
            job.returncode = -1
        finally:
            job.finished_at = time.time()


JOBS = JobManager()

