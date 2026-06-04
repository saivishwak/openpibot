from __future__ import annotations

from fastapi import APIRouter

from openpibot.server.config import get_active_robot_id, get_robot_profiles, load_project_config
from openpibot.server.schemas import RobotsResponse

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("/robots", response_model=RobotsResponse)
def robots() -> RobotsResponse:
    cfg = load_project_config()
    return RobotsResponse(active_robot=get_active_robot_id(cfg), robots=get_robot_profiles(cfg))

