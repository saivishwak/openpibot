from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response, StreamingResponse

from openpibot.server.runtime import cameras as cam_mod

router = APIRouter(tags=["cameras"])


@router.get("/api/cameras")
def cameras() -> dict[str, Any]:
    cams = [asdict(c) for c in cam_mod.enumerate_cameras()]
    return {"cameras": cams, "roles": list(cam_mod._ROLES)}


@router.post("/api/cameras/assign")
def assign(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = body or {}
    by_path = payload.get("by_path")
    role = payload.get("role")
    if not by_path:
        raise HTTPException(status_code=400, detail="by_path required")
    try:
        cam_mod.assign_role(str(by_path), role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    cam_mod.reset_streams()
    cams = [asdict(c) for c in cam_mod.enumerate_cameras()]
    return {"ok": True, "cameras": cams}


@router.get("/camera/{cam_id:path}/stream")
def camera_stream(cam_id: str) -> StreamingResponse:
    if cam_mod.find_camera(cam_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown camera: {cam_id}")
    return StreamingResponse(
        cam_mod.mjpeg_iter(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/camera/{cam_id:path}/snapshot")
def camera_snapshot(cam_id: str) -> Response:
    if cam_mod.find_camera(cam_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown camera: {cam_id}")
    stream = cam_mod.get_stream(cam_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"unknown camera: {cam_id}")
    jpeg, err = stream.snapshot()
    if jpeg is None:
        raise HTTPException(status_code=503, detail=f"camera unavailable: {err or 'no frame'}")
    return Response(jpeg, media_type="image/jpeg")
