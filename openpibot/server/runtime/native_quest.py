"""Native Quest controller protocol adapter.

The Unity/OpenXR app streams absolute controller poses and button states. The
existing teleop runtime consumes reset/position/idle goals with per-frame
relative deltas, so this module is the narrow compatibility layer between them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation

ArmSide = Literal["left", "right"]
MAX_NATIVE_QUEST_PACKET_BYTES = 64 * 1024


class NativeQuestProtocolError(ValueError):
    """Raised when a native Quest packet is malformed."""


@dataclass(frozen=True)
class NativeQuestControlGoal:
    arm: ArmSide
    mode: str
    relative_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    relative_rotvec: tuple[float, float, float] = (0.0, 0.0, 0.0)
    trigger: bool = False
    thumbstick: dict[str, float] = field(default_factory=dict)
    buttons: dict[str, bool] = field(default_factory=dict)
    vr_ctrl_position: tuple[float, float, float] | None = None
    vr_ctrl_rotation: Rotation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ControllerState:
    grip: bool = False
    position: np.ndarray | None = None
    rotation: Rotation | None = None


class NativeQuestAdapter:
    """Convert Unity/OpenXR controller snapshots into runtime control goals."""

    def __init__(self, *, coordinate_frame: str = "unity_openxr") -> None:
        if coordinate_frame not in {"unity_openxr", "unity_reachy"}:
            raise ValueError("coordinate_frame must be 'unity_openxr' or 'unity_reachy'")
        self.coordinate_frame = coordinate_frame
        self._state: dict[ArmSide, _ControllerState] = {
            "left": _ControllerState(),
            "right": _ControllerState(),
        }

    def process_packet(self, packet: dict[str, Any]) -> list[NativeQuestControlGoal]:
        controllers = self._extract_controllers(packet)
        goals: list[NativeQuestControlGoal] = []
        for side in ("left", "right"):
            raw = controllers.get(side)
            if raw is None:
                continue
            goals.append(self._process_controller(side, raw, packet))
        return goals

    def _extract_controllers(self, packet: dict[str, Any]) -> dict[ArmSide, dict[str, Any]]:
        raw_controllers = packet.get("controllers")
        controllers: dict[ArmSide, dict[str, Any]] = {}
        if isinstance(raw_controllers, dict):
            for side in ("left", "right"):
                value = raw_controllers.get(side)
                if isinstance(value, dict):
                    controllers[side] = value

        aliases = {
            "left": "left",
            "right": "right",
            "leftController": "left",
            "rightController": "right",
        }
        for key, side in aliases.items():
            value = packet.get(key)
            if isinstance(value, dict):
                controllers[side] = value
        return controllers

    def _process_controller(
        self,
        side: ArmSide,
        raw: dict[str, Any],
        packet: dict[str, Any],
    ) -> NativeQuestControlGoal:
        state = self._state[side]
        valid = self._bool(raw.get("valid", True))
        if not valid:
            return NativeQuestControlGoal(
                arm=side,
                mode="idle",
                metadata={
                    "source": "native_quest",
                    "timestamp": packet.get("timestamp"),
                    "coordinate_frame": self.coordinate_frame,
                    "valid": False,
                },
            )
        pos = self._vector3(raw.get("position") or raw.get("controller_position"), "position")
        rot = self._transform_rotation(self._rotation(raw.get("rotation") or raw.get("quaternion")))
        grip = self._bool(raw.get("grip", raw.get("gripActive", raw.get("grip_active", False))))
        trigger_value = raw.get("trigger", raw.get("triggerPressed", False))
        trigger = self._bool(trigger_value, threshold=0.5)
        buttons = {
            str(k): self._bool(v)
            for k, v in (raw.get("buttons") or {}).items()
        }
        thumbstick_raw = raw.get("thumbstick") or {}
        thumbstick = {
            "x": float(thumbstick_raw.get("x", 0.0)),
            "y": float(thumbstick_raw.get("y", 0.0)),
        }

        if grip and not state.grip:
            mode = "reset"
            rel = np.zeros(3, dtype=float)
            rel_rot = np.zeros(3, dtype=float)
        elif grip and state.grip:
            mode = "position"
            rel = pos - state.position if state.position is not None else np.zeros(3, dtype=float)
            if state.rotation is not None:
                rel_rot = (state.rotation.inv() * rot).as_rotvec()
            else:
                rel_rot = np.zeros(3, dtype=float)
        else:
            mode = "idle"
            rel = np.zeros(3, dtype=float)
            rel_rot = np.zeros(3, dtype=float)

        state.grip = grip
        state.position = pos
        state.rotation = rot

        return NativeQuestControlGoal(
            arm=side,
            mode=mode,
            relative_position=self._tuple3(rel),
            relative_rotvec=self._tuple3(rel_rot),
            trigger=trigger,
            thumbstick=thumbstick,
            buttons=buttons,
            vr_ctrl_position=self._tuple3(pos),
            vr_ctrl_rotation=rot,
            metadata={
                "source": "native_quest",
                "timestamp": packet.get("timestamp"),
                "coordinate_frame": self.coordinate_frame,
            },
        )

    def _vector3(self, raw: Any, label: str) -> np.ndarray:
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise NativeQuestProtocolError(f"{label} must be a 3-vector")
        try:
            vec = np.array([float(v) for v in raw], dtype=float)
        except (TypeError, ValueError) as exc:
            raise NativeQuestProtocolError(f"{label} must contain numbers") from exc
        if not np.all(np.isfinite(vec)):
            raise NativeQuestProtocolError(f"{label} must contain finite numbers")
        if self.coordinate_frame == "unity_reachy":
            # Match the Reachy reference's Unity-to-robot basis documentation:
            # reachy = (unity.z, -unity.x, unity.y).
            vec = np.array([vec[2], -vec[0], vec[1]], dtype=float)
        elif self.coordinate_frame == "unity_openxr":
            # Unity/OpenXR uses +Z as forward in the operator-origin space. The
            # backend's calibrated VR convention treats operator forward as -Z.
            # Keep X/Y unchanged and flip Z before calibration.
            vec = np.array([vec[0], vec[1], -vec[2]], dtype=float)
        return vec

    def _transform_rotation(self, rot: Rotation) -> Rotation:
        basis = self._basis_matrix()
        if basis is None:
            return rot
        return Rotation.from_matrix(basis @ rot.as_matrix() @ basis.T)

    def _basis_matrix(self) -> np.ndarray | None:
        if self.coordinate_frame == "unity_reachy":
            return np.array(
                [
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=float,
            )
        if self.coordinate_frame == "unity_openxr":
            return np.diag([1.0, 1.0, -1.0])
        return None

    @staticmethod
    def _rotation(raw: Any) -> Rotation:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise NativeQuestProtocolError("rotation must be a quaternion [x, y, z, w]")
        try:
            quat = np.array([float(v) for v in raw], dtype=float)
        except (TypeError, ValueError) as exc:
            raise NativeQuestProtocolError("rotation quaternion must contain numbers") from exc
        if not np.all(np.isfinite(quat)):
            raise NativeQuestProtocolError("rotation quaternion must contain finite numbers")
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-9:
            raise NativeQuestProtocolError("rotation quaternion must be non-zero")
        return Rotation.from_quat(quat / norm)

    @staticmethod
    def _tuple3(vec: np.ndarray) -> tuple[float, float, float]:
        return (float(vec[0]), float(vec[1]), float(vec[2]))

    @staticmethod
    def _bool(raw: Any, *, threshold: float | None = None) -> bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on", "pressed"}
        if threshold is not None:
            try:
                return float(raw or 0.0) > threshold
            except (TypeError, ValueError):
                return False
        return bool(raw)
