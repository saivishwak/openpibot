from __future__ import annotations

from fastapi import APIRouter

from openpibot.server.schemas import DoctorResponse
from openpibot.server.runtime import doctor as doctor_mod

router = APIRouter(tags=["diagnostics"])


@router.get("/api/doctor", response_model=DoctorResponse)
def doctor() -> dict:
    return {"checks": doctor_mod.run_doctor()}

