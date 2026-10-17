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
"""Modbus service: the only way the rest of the system talks to the controller.

Responsibilities are narrow on purpose (`doc/conventions.md` §2, SRP): build PDUs,
run them through the transport, translate failures, keep counters. Mapping a
detection to a register value is the *manager's* job, not this one's.

Failure policy, following `doc/conventions.md` §5:

* transport errors are retried a bounded number of times, then raised as
  :class:`~src.robot.exceptions.ModbusCommunicationError` - never as a raw
  ``SerialException``;
* a device exception (the controller refused the command) is **not** retried.
  Retrying an illegal-data-address or a rejected move cannot succeed, and on a
  motion command it risks issuing a second movement;
* nothing ever returns ``True`` on a silent failure. The old stub printed and
  returned booleans, which is how a robot command can be "successful" while
  nothing was written.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

from src.robot import protocol
from src.robot.exceptions import (
    ModbusCommunicationError,
    ModbusDeviceException,
    RobotError,
    RobotNotConnectedError,
)
from src.robot.transports import Transport, build_transport

__all__ = ["ModbusService"]

logger = logging.getLogger(__name__)


class ModbusService:
    """Connection-scoped Modbus client.

    One instance per process: an RS-485 bus is half-duplex, and two writers
    interleave frames and corrupt the protocol (`doc/conventions.md` §3). The K8s
    `robot` deployment is a singleton for the same reason, and its HPA is capped at
    one replica.
    """

    def __init__(
        self,
        transport: Optional[Transport] = None,
        *,
        retries: int = 2,
        retry_delay: float = 0.05,
    ) -> None:
        self._transport = transport or build_transport()
        self._retries = max(0, retries)
        self._retry_delay = retry_delay
        self._connected = False
        self.commands_sent = 0
        self.commands_failed = 0
        self.retries_used = 0
        self.last_error: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------
    @property
    def transport(self) -> Transport:
        return self._transport

    @property
    def is_connected(self) -> bool:
        return self._connected and self._transport.is_open

    def connect(self) -> bool:
        """Open the transport.

        Returns ``False`` instead of raising so a caller can decide whether a
        missing controller is fatal. The *reason* is kept in ``last_error`` and in
        the log, because "connect returned False" alone is not actionable.
        """
        try:
            self._transport.open()
            self._connected = True
            self.last_error = None
            logger.info("robot link open: %s", self._transport.describe())
            return True
        except RobotError as exc:
            self.commands_failed += 1
            self.last_error = f"{exc.code}: {exc.detail}"
            logger.error("robot link failed to open: %s", self.last_error)
            self._connected = False
            return False

    def close(self) -> None:
        try:
            self._transport.close()
        finally:
            self._connected = False

    def __enter__(self) -> "ModbusService":
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- operations --------------------------------------------------------
    def read_registers(
        self, address: int, count: int = 1, *, input_registers: bool = False
    ) -> list[int]:
        """Read holding (default) or input registers."""
        pdu = (
            protocol.pdu_read_input(address, count)
            if input_registers
            else protocol.pdu_read_holding(address, count)
        )
        response = self._exchange(pdu, expected_response_bytes=5 + 2 * count)
        return protocol.parse_read_registers(response, count)

    def read_register(self, address: int, *, input_registers: bool = False) -> int:
        values = self.read_registers(address, 1, input_registers=input_registers)
        return values[0]

    def write_register(self, address: int, value: int) -> None:
        """Write one holding register."""
        response = self._exchange(
            protocol.pdu_write_single(address, value), expected_response_bytes=8
        )
        protocol.parse_write_ack(response)

    def write_registers(self, start: int, values: Sequence[int]) -> None:
        """Write a contiguous block atomically (FC16).

        One transaction rather than several FC06 writes: the controller applies the
        whole target in one step instead of passing through intermediate positions,
        which matters for a multi-axis move.
        """
        response = self._exchange(
            protocol.pdu_write_multiple(start, values), expected_response_bytes=8
        )
        protocol.parse_write_ack(response)

    # -- internals ---------------------------------------------------------
    def _exchange(
        self, pdu: bytes, expected_response_bytes: Optional[int] = None
    ) -> bytes:
        """Send *pdu*, retrying only on communication failures."""
        if not self.is_connected:
            raise RobotNotConnectedError(
                "robot link is not open; call connect() first"
            )

        attempt = 0
        while True:
            try:
                response = self._transport.exchange(pdu, expected_response_bytes)
                # Transport-agnostic refusal check: see protocol.ensure_no_exception.
                response = protocol.ensure_no_exception(response)
                self.commands_sent += 1
                self.last_error = None
                return response
            except ModbusDeviceException as exc:
                # The controller answered; retrying cannot change its mind.
                self.commands_failed += 1
                self.last_error = f"{exc.code}: {exc.detail}"
                raise
            except ModbusCommunicationError as exc:
                attempt += 1
                if attempt > self._retries:
                    self.commands_failed += 1
                    self.last_error = f"{exc.code}: {exc.detail}"
                    raise ModbusCommunicationError(
                        f"{exc.detail} (after {attempt} attempt(s))"
                    ) from exc
                self.retries_used += 1
                logger.warning(
                    "modbus attempt %d failed (%s); retrying", attempt, exc.detail
                )
                time.sleep(self._retry_delay)

    # -- introspection -----------------------------------------------------
    def status(self) -> dict:
        """Snapshot for /health and the robot dashboard."""
        return {
            "connected": self.is_connected,
            "transport": self._transport.describe(),
            "simulated": self._transport.describe().startswith("simulated"),
            "commands_sent": self.commands_sent,
            "commands_failed": self.commands_failed,
            "retries_used": self.retries_used,
            "last_error": self.last_error,
        }
