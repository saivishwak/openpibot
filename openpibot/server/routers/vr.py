from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, Body, Header, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from starlette.concurrency import run_in_threadpool

from openpibot.server.runtime import vr_teleop as vr_mod
from openpibot.server.runtime.native_quest import MAX_NATIVE_QUEST_PACKET_BYTES, NativeQuestProtocolError
from openpibot.server.runtime import quest_video_bridge

router = APIRouter(prefix="/api/vr", tags=["vr"])

QUEST_WS_OPERATOR_STATUS_INTERVAL_S = 0.5


def _runtime_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


def _require_quest_pairing_token(token: str | None) -> None:
    if not vr_mod.SESSION.verify_native_quest_pairing_token(token):
        raise HTTPException(status_code=401, detail="invalid Quest pairing token")


@router.get("/status")
def status() -> dict[str, Any]:
    return vr_mod.SESSION.status()


@router.get("/quest/status")
def quest_status(request: Request) -> dict[str, Any]:
    return vr_mod.SESSION.quest_bridge_status(public_base_url=str(request.base_url))


@router.get("/quest/operator")
def quest_operator(
    request: Request,
    x_quest_pairing_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_quest_pairing_token(x_quest_pairing_token)
    return vr_mod.SESSION.quest_operator_status(public_base_url=str(request.base_url))


@router.get("/quest/video/status")
def quest_video_status() -> dict[str, Any]:
    return quest_video_bridge.bridge_status()


@router.post("/quest/video/start")
def quest_video_start(
    request: Request,
    body: dict[str, Any] | None = Body(default=None),
    x_quest_pairing_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_quest_pairing_token(x_quest_pairing_token)
    payload = body or {}
    try:
        roles = payload.get("roles")
        if roles is not None and not isinstance(roles, list):
            raise ValueError("roles must be a list")
        quest_host = str(payload.get("quest_host") or "").strip()
        if not quest_host and request.client is not None:
            quest_host = request.client.host
        return quest_video_bridge.start_bridge(
            quest_host=quest_host,
            roles=roles,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/quest/video/stop")
def quest_video_stop(
    x_quest_pairing_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_quest_pairing_token(x_quest_pairing_token)
    return quest_video_bridge.stop_bridge()


@router.post("/quest/video/health")
def quest_video_health(
    body: dict[str, Any] | None = Body(default=None),
    x_quest_pairing_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_quest_pairing_token(x_quest_pairing_token)
    try:
        return quest_video_bridge.report_receive_health(body or {})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/quest/ingest")
def quest_ingest(
    body: dict[str, Any] | None = Body(default=None),
    x_quest_pairing_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_quest_pairing_token(x_quest_pairing_token)
    try:
        return vr_mod.SESSION.ingest_native_quest_packet(body or {})
    except NativeQuestProtocolError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.websocket("/quest/ws")
async def quest_ws(websocket: WebSocket) -> None:
    token = websocket.query_params.get("token") or websocket.headers.get("x-quest-pairing-token")
    if not vr_mod.SESSION.verify_native_quest_pairing_token(token):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="invalid Quest pairing token")
        return
    await websocket.accept()
    vr_mod.SESSION.note_native_quest_client(True)
    next_operator_status_at = 0.0
    ws_public_url = str(websocket.url)
    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw.encode("utf-8")) > MAX_NATIVE_QUEST_PACKET_BYTES:
                await websocket.send_json({"ok": False, "error": f"packet exceeds {MAX_NATIVE_QUEST_PACKET_BYTES} byte limit"})
                continue
            try:
                payload = json.loads(raw)
                result = await run_in_threadpool(vr_mod.SESSION.ingest_native_quest_packet, payload)
                now = time.monotonic()
                if now >= next_operator_status_at:
                    result["operator"] = await run_in_threadpool(
                        lambda: vr_mod.SESSION.quest_operator_status(public_base_url=ws_public_url)
                    )
                    next_operator_status_at = now + QUEST_WS_OPERATOR_STATUS_INTERVAL_S
                await websocket.send_json(result)
            except (json.JSONDecodeError, NativeQuestProtocolError) as exc:
                await websocket.send_json({"ok": False, "error": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        vr_mod.SESSION.note_native_quest_client(False)


@router.post("/connect")
def connect(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    arm = (body or {}).get("arm")
    if arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left' or 'right'")
    try:
        return vr_mod.SESSION.connect(arm)
    except RuntimeError as exc:
        raise _runtime_error(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@router.post("/disconnect")
def disconnect(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    arm = (body or {}).get("arm")
    if arm is not None and arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left', 'right', or omitted")
    return vr_mod.SESSION.disconnect(side=arm)


@router.post("/engage")
def engage(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = body or {}
    if "engaged" not in payload:
        raise HTTPException(status_code=400, detail="engaged required")
    scale = payload.get("scale")
    if scale is not None:
        try:
            scale = float(scale)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="scale must be numeric") from exc
    active_arm = payload.get("active_arm")
    if active_arm is not None and active_arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="active_arm must be 'left' or 'right'")
    try:
        return vr_mod.SESSION.engage(bool(payload["engaged"]), scale=scale, active_arm=active_arm)
    except (RuntimeError, ValueError) as exc:
        raise _runtime_error(exc) from exc


@router.post("/calibration_profile/select")
def select_calibration_profile(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    try:
        return vr_mod.SESSION.select_calibration_profile(str((body or {}).get("profile") or ""))
    except (RuntimeError, ValueError) as exc:
        raise _runtime_error(exc) from exc


@router.post("/calibration_profile/create")
def create_calibration_profile(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = body or {}
    raw = payload.get("copy_from_active", True)
    copy_from_active = raw.strip().lower() not in ("0", "false", "no", "off") if isinstance(raw, str) else bool(raw)
    try:
        return vr_mod.SESSION.create_calibration_profile(str(payload.get("profile") or ""), copy_from_active=copy_from_active)
    except (RuntimeError, ValueError) as exc:
        raise _runtime_error(exc) from exc


@router.post("/calibration_profile/delete")
def delete_calibration_profile(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    try:
        return vr_mod.SESSION.delete_calibration_profile(str((body or {}).get("profile") or ""))
    except (RuntimeError, ValueError) as exc:
        raise _runtime_error(exc) from exc


@router.post("/emergency_stop")
def emergency_stop() -> dict[str, Any]:
    return vr_mod.SESSION.emergency_stop()


@router.post("/torque/release")
def torque_release(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    arm = (body or {}).get("arm")
    if arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left' or 'right'")
    try:
        return vr_mod.SESSION.release_torque_for_posing(arm)
    except RuntimeError as exc:
        raise _runtime_error(exc) from exc


@router.post("/torque/lock")
def torque_lock(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    arm = (body or {}).get("arm")
    if arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left' or 'right'")
    try:
        return vr_mod.SESSION.lock_torque(arm)
    except RuntimeError as exc:
        raise _runtime_error(exc) from exc


@router.post("/home/capture")
def home_capture(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    arm = (body or {}).get("arm")
    if arm is not None and arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left', 'right', or omitted")
    try:
        return vr_mod.SESSION.capture_home(side=arm)
    except RuntimeError as exc:
        raise _runtime_error(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@router.post("/home/go")
def home_go(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    arm = (body or {}).get("arm")
    if arm is not None and arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left', 'right', or omitted")
    try:
        return vr_mod.SESSION.go_home(side=arm)
    except RuntimeError as exc:
        raise _runtime_error(exc) from exc


@router.post("/home/cancel")
def home_cancel(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    arm = (body or {}).get("arm")
    if arm is not None and arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left', 'right', or omitted")
    return vr_mod.SESSION.cancel_homing(side=arm)


@router.post("/calibrate/start")
def calibrate_start(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    arm = (body or {}).get("arm")
    if arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left' or 'right'")
    try:
        return vr_mod.SESSION.start_calibration(arm)
    except (RuntimeError, ValueError) as exc:
        raise _runtime_error(exc) from exc


@router.post("/calibrate/cancel")
def calibrate_cancel(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    arm = (body or {}).get("arm")
    if arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left' or 'right'")
    return vr_mod.SESSION.cancel_calibration(arm)


@router.post("/calibrate/skip_wrist_verify")
def skip_wrist_verify(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    arm = (body or {}).get("arm")
    if arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left' or 'right'")
    return vr_mod.SESSION.skip_wrist_verify(arm)


@router.post("/calibrate/robot_verify/start")
def robot_verify_start(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = body or {}
    arm = payload.get("arm")
    if arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left' or 'right'")
    try:
        return vr_mod.SESSION.start_robot_verification(arm, release_torque=bool(payload.get("release_torque", False)))
    except (RuntimeError, ValueError) as exc:
        raise _runtime_error(exc) from exc


@router.post("/calibrate/robot_verify/cancel")
def robot_verify_cancel(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    arm = (body or {}).get("arm")
    if arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left' or 'right'")
    return vr_mod.SESSION.cancel_robot_verification(arm)


@router.post("/calibrate/robot_verify/robot_pose")
def robot_verify_robot_pose(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = body or {}
    arm = payload.get("arm")
    point = payload.get("point")
    if arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left' or 'right'")
    if point not in ("start", "end"):
        raise HTTPException(status_code=400, detail="point must be 'start' or 'end'")
    try:
        return vr_mod.SESSION.capture_robot_verification_pose(arm, point, label=str(payload.get("label") or ""))
    except (RuntimeError, ValueError) as exc:
        raise _runtime_error(exc) from exc


@router.post("/calibrate/robot_verify/vr_pose")
def robot_verify_vr_pose(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = body or {}
    arm = payload.get("arm")
    point = payload.get("point")
    if arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left' or 'right'")
    if point not in ("start", "end"):
        raise HTTPException(status_code=400, detail="point must be 'start' or 'end'")
    try:
        return vr_mod.SESSION.capture_robot_verification_vr(arm, point, label=str(payload.get("label") or ""))
    except (RuntimeError, ValueError) as exc:
        raise _runtime_error(exc) from exc


@router.post("/calibrate/robot_verify/discard_last")
def robot_verify_discard_last(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    arm = (body or {}).get("arm")
    if arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left' or 'right'")
    return vr_mod.SESSION.discard_last_robot_verification_sample(arm)


@router.post("/calibrate/robot_verify/solve")
def robot_verify_solve(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    arm = (body or {}).get("arm")
    if arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left' or 'right'")
    try:
        return vr_mod.SESSION.solve_robot_verification(arm)
    except (RuntimeError, ValueError) as exc:
        raise _runtime_error(exc) from exc


@router.post("/calibrate/robot_verify/test_start")
def robot_verify_test_start(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = body or {}
    arm = payload.get("arm")
    if arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left' or 'right'")
    try:
        scale = float(payload.get("scale", 0.2))
        return vr_mod.SESSION.start_robot_verification_test(arm, scale=scale)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="scale must be numeric") from exc
    except RuntimeError as exc:
        raise _runtime_error(exc) from exc


@router.post("/calibrate/robot_verify/test_stop")
def robot_verify_test_stop(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    arm = (body or {}).get("arm")
    if arm not in ("left", "right"):
        raise HTTPException(status_code=400, detail="arm must be 'left' or 'right'")
    return vr_mod.SESSION.stop_robot_verification_test(arm)


@router.post("/recording")
def recording(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = body or {}
    if "enabled" not in payload:
        raise HTTPException(status_code=400, detail="enabled required")
    task = str(payload.get("task") or "")
    root = str(payload.get("root") or "")
    if bool(payload["enabled"]) and not task.strip():
        raise HTTPException(status_code=400, detail="task description required before starting an episode")
    vr_mod.SESSION.set_recording(bool(payload["enabled"]), task=task, root=root)
    return vr_mod.SESSION.status()


@router.post("/recording/task")
def recording_task(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = body or {}
    if "task" not in payload:
        raise HTTPException(status_code=400, detail="task required")
    return vr_mod.SESSION.set_recording_task(str(payload.get("task") or ""))


@router.post("/recording/delete_last")
def recording_delete_last() -> dict[str, Any]:
    return vr_mod.SESSION.delete_last_recorded_episode()
