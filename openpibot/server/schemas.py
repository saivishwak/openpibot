"""Pydantic schemas for OpenPIBot API boundaries."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ApiError(BaseModel):
    code: str
    message: str
    details: Any | None = None
    request_id: str


class ErrorResponse(BaseModel):
    error: ApiError


class RobotProfile(BaseModel):
    id: str
    name: str | None = None
    active: bool = False
    port_left_base: str | None = None
    port_right_head: str | None = None
    left_arm_id: str | None = None
    right_arm_id: str | None = None
    max_relative_target: float | None = None
    use_degrees: bool | None = None
    home_pose: dict[str, float] = Field(default_factory=dict)


class RobotsResponse(BaseModel):
    active_robot: str
    robots: list[RobotProfile]


class DoctorCheck(BaseModel):
    name: str
    status: Literal["ok", "warn", "fail", "info"]
    detail: str


class DoctorResponse(BaseModel):
    checks: list[DoctorCheck]


class JobCreateResponse(BaseModel):
    id: str
    status: str
    command: list[str]

