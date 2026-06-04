from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException

from openpibot.server.runtime import vr_teleop as vr_mod

router = APIRouter(prefix="/api/vr", tags=["vr"])


def _runtime_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


@router.get("/status")
def status() -> dict[str, Any]:
    return vr_mod.SESSION.status()


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
        return vr_mod.SESSION.capture_robot_verification_pose(arm, point)
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

