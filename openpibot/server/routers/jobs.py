from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException

from openpibot.server.runtime.openpi_policy import build_openpi_policy_server_command
from openpibot.server.services.jobs import JOBS

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs() -> dict[str, Any]:
    return {"jobs": JOBS.list()}


@router.get("/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    return {"job": job}


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    job = JOBS.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    return {"job": job}


@router.post("/train/pi05")
def train_pi05(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    args = list((body or {}).get("args") or [])
    job = JOBS.start(["uv", "run", "python", "scripts/finetune_pi05.py", *map(str, args)])
    return {"job": job.public()}


@router.post("/train/molmoact2")
def train_molmoact2(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    args = list((body or {}).get("args") or [])
    job = JOBS.start(["uv", "run", "python", "scripts/finetune_molmoact2.py", *map(str, args)])
    return {"job": job.public()}


@router.post("/inference/pi05")
def infer_pi05(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    args = list((body or {}).get("args") or [])
    job = JOBS.start(["uv", "run", "python", "scripts/infer_pi05_finetuned.py", *map(str, args)])
    return {"job": job.public()}


@router.post("/inference/molmoact2")
def infer_molmoact2(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    args = list((body or {}).get("args") or [])
    job = JOBS.start(["uv", "run", "python", "scripts/infer_molmoact2_finetuned.py", *map(str, args)])
    return {"job": job.public()}


@router.post("/pi05/server")
def pi05_server() -> dict[str, Any]:
    job = JOBS.start(build_openpi_policy_server_command())
    return {"job": job.public()}


@router.post("/dataset/push")
def dataset_push(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    args = list((body or {}).get("args") or [])
    job = JOBS.start(["uv", "run", "python", "scripts/push_dataset.py", *map(str, args)])
    return {"job": job.public()}
