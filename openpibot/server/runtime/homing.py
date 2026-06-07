"""Shared joint-space homing controller.

The controller separates the safe command trajectory from the completion
condition. Commands are rate-limited toward the target, but homing only
finishes after measured joint feedback has settled near the target.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalized_joint_name(key: str) -> str:
    """Return the unprefixed joint name for server and LeRobot observation keys."""
    base = key.removesuffix(".pos")
    if "_arm_" in base:
        return base.split("_arm_", 1)[1]
    return base


@dataclass(frozen=True)
class HomingStep:
    command: dict[str, float]
    command_reached: bool
    present_reached: bool
    settled: bool
    max_command_error_deg: float
    max_present_error_deg: float
    worst_present_joint: str


class JointHomingController:
    """Rate-limited homing that completes from physical feedback, not commands."""

    def __init__(
        self,
        *,
        targets: Mapping[str, float],
        present: Mapping[str, float],
        cap_for_key: Callable[[str], float] | None = None,
        kp: float = 0.75,
        command_tolerance_deg: float = 0.5,
        present_tolerance_deg: float = 1.0,
        final_direct_tolerance_deg: float = 5.0,
        settle_ticks: int = 5,
    ) -> None:
        self.targets = {str(k): float(v) for k, v in targets.items()}
        self._desired = {
            key: float(present.get(key, target))
            for key, target in self.targets.items()
        }
        self._cap_for_key = cap_for_key or (lambda _key: 1.0)
        self.kp = _clamp(float(kp), 0.05, 1.0)
        self.command_tolerance_deg = max(0.0, float(command_tolerance_deg))
        self.present_tolerance_deg = max(0.0, float(present_tolerance_deg))
        self.final_direct_tolerance_deg = max(
            self.present_tolerance_deg,
            float(final_direct_tolerance_deg),
        )
        self.settle_ticks = max(1, int(settle_ticks))
        self._settled_ticks = 0

    def step(self, present: Mapping[str, float]) -> HomingStep:
        command: dict[str, float] = {}
        command_reached = True
        present_reached = True
        max_command_error = 0.0
        max_present_error = 0.0
        worst_present_joint = ""

        for key, target in self.targets.items():
            previous_desired = float(self._desired.get(key, present.get(key, target)))
            cap = max(0.01, float(self._cap_for_key(key)))
            remaining = target - previous_desired
            if abs(remaining) <= self.command_tolerance_deg:
                next_desired = target
            else:
                next_desired = previous_desired + _clamp(remaining, -cap, cap)
            self._desired[key] = next_desired

            current = present.get(key)
            if current is None:
                current = previous_desired
                present_reached = False
            current = float(current)

            if abs(target - next_desired) <= self.command_tolerance_deg:
                # The planned trajectory has reached home; keep commanding from
                # measured feedback toward the true target so motor deadband or
                # safe-action clamping cannot leave the physical arm a few
                # degrees short forever.
                present_delta = target - current
                if abs(present_delta) <= self.final_direct_tolerance_deg:
                    cmd = target
                else:
                    cmd = current + _clamp(present_delta, -cap, cap)
            else:
                cmd = current + self.kp * (next_desired - current)
            if abs(target - cmd) <= self.command_tolerance_deg:
                cmd = target
            command[key] = float(cmd)

            command_error = abs(target - next_desired)
            present_error = abs(target - current)
            max_command_error = max(max_command_error, command_error)
            if present_error > max_present_error:
                max_present_error = present_error
                worst_present_joint = key
            if command_error > self.command_tolerance_deg:
                command_reached = False
            if present_error > self.present_tolerance_deg:
                present_reached = False

        if present_reached:
            self._settled_ticks += 1
        else:
            self._settled_ticks = 0

        return HomingStep(
            command=command,
            command_reached=command_reached,
            present_reached=present_reached,
            settled=self._settled_ticks >= self.settle_ticks,
            max_command_error_deg=max_command_error,
            max_present_error_deg=max_present_error,
            worst_present_joint=worst_present_joint,
        )
