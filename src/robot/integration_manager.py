# Copyright (c), The Rusoil Software Development Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# author : Konstantin Ustiuzhanin
# date   : 2026-Oct-17
# ==============================================================================
"""Turn detections into robot commands - carefully.

This is the layer that can hurt something, so it is built as **plan, then execute**,
with every refusal typed:

* ``STALE_DETECTION`` - the frame is older than the configured tolerance. Acting on
  stale coordinates is precisely the failure the pipeline's load shedding exists to
  avoid, so it is refused here too rather than trusted.
* ``OUT_OF_REACH`` - the target falls outside the configured safe workspace in
  millimetres. A normalised coordinate is always in ``[0, 1]``, so the workspace is
  what actually bounds the robot; without this check a spurious detection at the
  frame edge becomes a command to drive the arm into a limit.
* ``HUMAN_ON_SITE`` - an interlock state in the payload forbids motion.
* ``COMM_ERROR`` - the bus failed.

A planner that returns a command and an executor that applies it is also what makes
this testable without hardware: `plan()` is pure and can be asserted exactly, and
`execute()` is exercised against the simulated transport.

Conversion chain, and why it is split:

    normalised [0,1]  ->  millimetres  (SCALE_FACTOR_MM_PER_UNIT, the camera's
    working height) -> controller units (RobotMapping.units_per_mm)

Keeping millimetres as the intermediate unit means the calibration constant and the
controller's register scale are separate numbers, so recalibrating the camera does
not silently change what the robot does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Sequence

from src.robot.exceptions import (
    RobotError,
    StaleDetectionError,
    UnsafeTargetError,
    WorkspaceBoundsError,
)
from src.robot.service import ModbusService
from src.vision.constants import SCALE_FACTOR_MM_PER_UNIT

__all__ = ["MoveCommand", "RobotCommandManager", "RobotMapping"]

logger = logging.getLogger(__name__)

#: Interlock states a payload may carry that forbid motion.
BLOCKING_STATES = {"HUMAN_ON_SITE", "OUT_OF_REACH", "EMERGENCY_STOP"}


@dataclass(frozen=True, slots=True)
class RobotMapping:
    """How detections map onto the controller's registers.

    Defaults are placeholders for the controller register map, which is a robot-side
    decision and must be confirmed when the hardware exists. They are grouped here
    (rather than scattered through the code) so that confirming them is a one-line
    change and a review can see exactly what is assumed.
    """

    #: First holding register of a contiguous (x, y) block, written with one FC16.
    xy_register: int = 40001
    #: Register that latches the command; the controller acts on it being written.
    trigger_register: int = 40003
    #: Value written to the trigger register.
    trigger_value: int = 1
    #: Controller units per millimetre, e.g. 10 for tenth-of-a-millimetre steps.
    units_per_mm: float = 10.0
    #: Safe workspace, in millimetres, measured from the frame origin.
    workspace_x_mm: tuple[float, float] = (0.0, 3000.0)
    workspace_y_mm: tuple[float, float] = (0.0, 3000.0)
    #: Ignore targets below this confidence.
    min_confidence: float = 0.75
    #: Object types worth moving for.
    allowed_types: tuple[str, ...] = ("metal_sheet",)
    #: Refuse a detection older than this. A 60 FPS pipeline should never approach
    #: it; the threshold exists to catch a stalled worker, not normal jitter.
    stale_after_seconds: float = 2.0


@dataclass(frozen=True, slots=True)
class MoveCommand:
    """A validated move, ready to write."""

    object_type: str
    confidence: float
    x_mm: float
    y_mm: float
    x_units: int
    y_units: int
    mapping: RobotMapping

    def register_writes(self) -> list[tuple[int, Sequence[int]]]:
        """The writes to perform, in order."""
        return [
            (self.mapping.xy_register, (self.x_units, self.y_units)),
            (self.mapping.trigger_register, (self.mapping.trigger_value,)),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_type": self.object_type,
            "confidence": round(self.confidence, 4),
            "x_mm": round(self.x_mm, 2),
            "y_mm": round(self.y_mm, 2),
            "x_units": self.x_units,
            "y_units": self.y_units,
        }


class RobotCommandManager:
    """Maps detections to commands and applies them over Modbus."""

    def __init__(
        self,
        service: ModbusService,
        mapping: Optional[RobotMapping] = None,
        *,
        dry_run: bool = False,
        scale_factor_mm: float = SCALE_FACTOR_MM_PER_UNIT,
    ) -> None:
        self._service = service
        self._mapping = mapping or RobotMapping()
        self._dry_run = dry_run
        self._scale_factor_mm = scale_factor_mm
        self.commands_executed = 0
        self.commands_refused = 0

    @property
    def mapping(self) -> RobotMapping:
        return self._mapping

    # -- planning (pure) ---------------------------------------------------
    def plan(self, payload: Any) -> Optional[MoveCommand]:
        """Choose the best target and convert it to a command.

        Returns:
            The command, or ``None`` when there is nothing worth moving for. A
            missing target is normal operation, not an error, so it is not raised.

        Raises:
            StaleDetectionError, UnsafeTargetError, WorkspaceBoundsError: when there
            is a target but acting on it would be unsafe or wrong.
        """
        self._check_freshness(payload)
        self._check_interlocks(payload)

        detections: Iterable[Any] = getattr(payload, "detections", None) or []
        candidates = [
            detection
            for detection in detections
            if self._is_actionable(detection)
        ]
        if not candidates:
            return None

        # Prefer the most confident target; break ties by area, so a large sheet
        # wins over a small fragment the model is equally sure about.
        best = max(
            candidates,
            key=lambda d: (float(d.confidence), float(d.width_norm) * float(d.height_norm)),
        )

        x_mm, y_mm = self._to_millimetres(best)
        self._check_workspace(x_mm, y_mm, best)

        return MoveCommand(
            object_type=str(best.object_type),
            confidence=float(best.confidence),
            x_mm=x_mm,
            y_mm=y_mm,
            x_units=self._to_units(x_mm),
            y_units=self._to_units(y_mm),
            mapping=self._mapping,
        )

    def _is_actionable(self, detection: Any) -> bool:
        if str(getattr(detection, "object_type", "")) not in self._mapping.allowed_types:
            return False
        return float(getattr(detection, "confidence", 0.0)) >= self._mapping.min_confidence

    def _to_millimetres(self, detection: Any) -> tuple[float, float]:
        """Centre of the box, in millimetres from the frame's top-left corner.

        The centre, not the corner: the robot needs where the part *is*, and a box's
        top-left corner is a function of its size.
        """
        x_centre = float(detection.x_norm) + float(detection.width_norm) / 2.0
        y_centre = float(detection.y_norm) + float(detection.height_norm) / 2.0
        return (
            x_centre * self._scale_factor_mm,
            y_centre * self._scale_factor_mm,
        )

    def _to_units(self, millimetres: float) -> int:
        units = int(round(millimetres * self._mapping.units_per_mm))
        if not 0 <= units <= 0xFFFF:
            raise WorkspaceBoundsError(
                f"{millimetres:.1f} mm converts to {units} units, outside the "
                "16-bit register range"
            )
        return units

    def _check_workspace(self, x_mm: float, y_mm: float, detection: Any) -> None:
        x_min, x_max = self._mapping.workspace_x_mm
        y_min, y_max = self._mapping.workspace_y_mm
        if not (x_min <= x_mm <= x_max and y_min <= y_mm <= y_max):
            raise WorkspaceBoundsError(
                f"target {detection.object_type} at ({x_mm:.0f}, {y_mm:.0f}) mm is "
                f"outside the workspace x[{x_min:.0f}, {x_max:.0f}] "
                f"y[{y_min:.0f}, {y_max:.0f}] mm"
            )

    def _check_freshness(self, payload: Any) -> None:
        timestamp = getattr(payload, "processed_timestamp", None)
        if not isinstance(timestamp, datetime):
            return
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        if age > self._mapping.stale_after_seconds:
            raise StaleDetectionError(
                f"detection is {age:.1f}s old (limit "
                f"{self._mapping.stale_after_seconds:.1f}s); refusing to move on a "
                "stale view of the cell"
            )

    def _check_interlocks(self, payload: Any) -> None:
        state = _workspace_state(payload)
        if state in BLOCKING_STATES:
            raise UnsafeTargetError(
                f"workspace state is {state}; motion is inhibited"
            )

    # -- execution ---------------------------------------------------------
    def execute(self, payload: Any) -> dict[str, Any]:
        """Plan and apply a move.

        Returns a result dict rather than a bool: a caller needs to know *why*
        nothing moved, and ``False`` cannot carry that.
        """
        try:
            command = self.plan(payload)
        except RobotError as exc:
            self.commands_refused += 1
            logger.info("command refused: %s", exc.detail)
            return {"executed": False, "reason": exc.code, "detail": exc.detail}

        if command is None:
            return {"executed": False, "reason": "NO_TARGET", "detail": "no actionable target"}

        if self._dry_run:
            logger.info("dry run: would move to %s", command.to_dict())
            return {"executed": False, "reason": "DRY_RUN", "command": command.to_dict()}

        for address, values in command.register_writes():
            self._service.write_registers(address, values)

        self.commands_executed += 1
        logger.info("command executed: %s", command.to_dict())
        return {"executed": True, "command": command.to_dict()}

    def status(self) -> dict[str, Any]:
        return {
            "dry_run": self._dry_run,
            "commands_executed": self.commands_executed,
            "commands_refused": self.commands_refused,
            "mapping": {
                "xy_register": self._mapping.xy_register,
                "trigger_register": self._mapping.trigger_register,
                "units_per_mm": self._mapping.units_per_mm,
                "workspace_x_mm": list(self._mapping.workspace_x_mm),
                "workspace_y_mm": list(self._mapping.workspace_y_mm),
                "min_confidence": self._mapping.min_confidence,
            },
            "link": self._service.status(),
        }


def _workspace_state(payload: Any) -> Optional[str]:
    """Read a workspace state from a payload, whether it is an attribute or a dict.

    The pipeline carries it in the frame's ``metadata`` (the documented
    NO_TARGET / OUT_OF_REACH / HUMAN_ON_SITE vocabulary); a direct attribute is also
    accepted so the manager works with either shape.
    """
    state = getattr(payload, "workspace_state", None)
    if isinstance(state, str):
        return state
    metadata = getattr(payload, "metadata", None)
    if isinstance(metadata, dict):
        candidate = metadata.get("workspace_state")
        if isinstance(candidate, str):
            return candidate
    return None
