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
"""Typed robot/Modbus failures.

`doc/conventions.md` §5 requires two things of this layer:

* system failures are reported as *specific, actionable* codes (e.g. ``COMM_ERROR``);
* Modbus exceptions are translated into a standard Python exception class and raw
  bus errors never propagate.

So every failure here carries a stable ``code`` that callers can branch on and an
operator can search for, and the transport layer wraps pyserial/socket errors into
:class:`ModbusCommunicationError` rather than letting them escape.

The codes are deliberately the vocabulary the rest of the system already uses
(`doc/vision.md` "Failure Handling"): ``COMM_ERROR``, ``OUT_OF_REACH``,
``HUMAN_ON_SITE``. A robot command that cannot be issued safely is not a generic
error - it is one of a handful of states the operator interface knows how to
display.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "ModbusCommunicationError",
    "ModbusDeviceException",
    "RobotError",
    "RobotNotConnectedError",
    "StaleDetectionError",
    "UnsafeTargetError",
    "WorkspaceBoundsError",
]


class RobotError(Exception):
    """Base class for every robot-interface failure.

    Attributes:
        code: stable, machine-readable identifier.
        detail: human-readable explanation, safe to log and to return over HTTP.
    """

    code: str = "ROBOT_ERROR"

    def __init__(self, detail: str, *, code: Optional[str] = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if code is not None:
            self.code = code

    def to_dict(self) -> dict[str, str]:
        return {"error": type(self).__name__, "code": self.code, "detail": self.detail}


class RobotNotConnectedError(RobotError):
    """A command was attempted before the transport was opened."""

    code = "COMM_ERROR"


class ModbusCommunicationError(RobotError):
    """The bus, port or socket failed: timeout, CRC mismatch, short read, I/O error."""

    code = "COMM_ERROR"


class ModbusDeviceException(RobotError):
    """The controller answered with a Modbus exception PDU (function code | 0x80).

    ``exception_code`` is the controller's own reason:
    1 illegal function, 2 illegal data address, 3 illegal data value,
    4 slave device failure, 5 acknowledge, 6 slave busy, 8 memory parity error,
    10 gateway path unavailable, 11 gateway target failed to respond.
    """

    code = "DEVICE_EXCEPTION"

    #: Human wording for the standard exception codes, so a log line means
    #: something without a copy of the Modbus spec to hand.
    MEANINGS = {
        1: "illegal function",
        2: "illegal data address",
        3: "illegal data value",
        4: "slave device failure",
        5: "acknowledge (command accepted, still processing)",
        6: "slave device busy",
        8: "memory parity error",
        10: "gateway path unavailable",
        11: "gateway target device failed to respond",
    }

    def __init__(self, function_code: int, exception_code: int) -> None:
        self.function_code = function_code
        self.exception_code = exception_code
        meaning = self.MEANINGS.get(exception_code, "unknown exception")
        super().__init__(
            f"controller rejected function 0x{function_code:02X}: "
            f"exception {exception_code} ({meaning})"
        )


class WorkspaceBoundsError(RobotError):
    """The target lies outside the configured safe workspace (``OUT_OF_REACH``)."""

    code = "OUT_OF_REACH"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)


class UnsafeTargetError(RobotError):
    """A safety interlock refuses the command (e.g. a human is inside the cell)."""

    code = "HUMAN_ON_SITE"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)


class StaleDetectionError(RobotError):
    """The detection is too old to act on; the world has moved.

    Acting on a stale frame is the dangerous case the whole load-shedding design
    exists to avoid, so it is refused explicitly rather than judged case by case.
    """

    code = "STALE_DETECTION"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
