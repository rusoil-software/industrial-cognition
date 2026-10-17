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
"""Physical transports for the Modbus link.

The interface is one method - ``exchange(pdu) -> pdu`` - because a transport's only
job is to move a protocol data unit to the controller and bring the answer back.
Whoever implements it owns the encapsulation (RS-485 framing or MBAP), so adding a
transport cannot change how a command is encoded.

There are three implementations, and the third is the point of this module:

* :class:`SerialRtuTransport` - real RS-485 via pyserial. The documented authority
  (`doc/conventions.md` §3). Never imported unless selected, so the package works
  on a machine without pyserial or a serial port.
* :class:`TcpTransport` - the Ethernet converter fallback.
* :class:`SimulatedTransport` - a controller in memory, with a register file and
  fault injection. **This is what the MVP runs**, because there is no RS-485
  converter and the robot is still being rebuilt. It speaks real Modbus PDUs, so the
  service, the mapping and the error paths are the same code that will later drive
  hardware - only the last centimetre changes.

That last property is why this is not a "mock": a simulator that bypasses the
protocol would let protocol bugs ship. This one parses and answers actual PDUs,
including exception responses, so the day a converter arrives the remaining
unknowns are wiring and timings, not framing.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

from src.robot import protocol
from src.robot.exceptions import ModbusCommunicationError, RobotError

__all__ = [
    "SerialRtuTransport",
    "SimulatedTransport",
    "TcpTransport",
    "Transport",
    "build_transport",
]

logger = logging.getLogger(__name__)


@runtime_checkable
class Transport(Protocol):
    """Moves one PDU to the controller and returns the response PDU."""

    @property
    def is_open(self) -> bool: ...

    def open(self) -> None: ...

    def close(self) -> None: ...

    def exchange(self, pdu: bytes, expected_response_bytes: Optional[int] = None) -> bytes:
        """Send *pdu*, return the response PDU.

        ``expected_response_bytes`` is a length hint the caller can supply (a read
        knows how many registers it asked for); a transport that must size its read
        buffer before sending should use it.

        Raises:
            ModbusCommunicationError: on any transport failure.
            ModbusDeviceException: when the controller reports an exception.
        """

    def describe(self) -> str: ...


class SerialRtuTransport:
    """RS-485 Modbus RTU over pyserial."""

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 9600,
        parity: str = "N",
        stopbits: int = 1,
        bytesize: int = 8,
        timeout: float = 1.0,
        slave: int = 1,
        rs485: bool = True,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._parity = parity
        self._stopbits = stopbits
        self._bytesize = bytesize
        self._timeout = timeout
        self._slave = slave
        self._rs485 = rs485
        self._serial = None
        # One transaction at a time: a shared RS-485 bus is half-duplex, so two
        # threads writing interleave frames and corrupt the protocol. The service
        # is a singleton for the same reason.
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self._serial is not None and getattr(self._serial, "is_open", False)

    def describe(self) -> str:
        return (
            f"rtu:{self._port}@{self._baudrate}"
            f"{self._bytesize}{self._parity}{self._stopbits} slave={self._slave}"
        )

    def open(self) -> None:
        try:
            import serial  # noqa: PLC0415 - optional dependency, selected at runtime
        except ImportError as exc:  # pragma: no cover - depends on the image
            raise ModbusCommunicationError(
                "pyserial is not installed; install `serial` to drive RS-485"
            ) from exc

        try:
            connection = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                parity=self._parity,
                stopbits=self._stopbits,
                bytesize=self._bytesize,
                timeout=self._timeout,
            )
            if self._rs485 and hasattr(connection, "rs485_mode"):
                # Let the UART drive the DE/RE line where the adapter supports it;
                # otherwise the converter does it in hardware and this is a no-op.
                mode = serial.rs485.RS485Settings(rts_level_for_tx=True)
                connection.rs485_mode = mode
        except Exception as exc:  # noqa: BLE001 - pyserial raises many types
            raise ModbusCommunicationError(
                f"could not open {self._port}: {type(exc).__name__}: {exc}"
            ) from exc

        self._serial = connection
        logger.info("opened %s", self.describe())

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None

    def exchange(self, pdu: bytes, expected_response_bytes: Optional[int] = None) -> bytes:
        if self._serial is None:
            raise ModbusCommunicationError("transport is not open")

        frame = _rtu_frame(self._slave, pdu)

        # A read response is 5 bytes of overhead; writes echo 8 bytes.
        size = expected_response_bytes or _guess_response_size(pdu)

        with self._lock:
            try:
                self._serial.reset_input_buffer()
                self._serial.write(frame)
                self._serial.flush()
                response = self._serial.read(size)
            except Exception as exc:  # noqa: BLE001 - translate, never leak
                raise ModbusCommunicationError(
                    f"serial I/O failed on {self._port}: {type(exc).__name__}: {exc}"
                ) from exc

        if len(response) < size:
            raise ModbusCommunicationError(
                f"timed out: {len(response)} of {size} bytes from {self._port} "
                "(check address, baud rate and termination)"
            )
        return protocol.split_rtu_response(response, self._slave)


class TcpTransport:
    """Modbus TCP over a socket (Ethernet-to-RS-485 converter or native controller)."""

    def __init__(
        self,
        host: str,
        port: int = 502,
        *,
        timeout: float = 1.0,
        unit_id: int = 1,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._unit_id = unit_id
        self._socket: Optional[socket.socket] = None
        self._transaction = 0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self._socket is not None

    def describe(self) -> str:
        return f"tcp:{self._host}:{self._port} unit={self._unit_id}"

    def open(self) -> None:
        try:
            connection = socket.create_connection(
                (self._host, self._port), timeout=self._timeout
            )
            connection.settimeout(self._timeout)
        except OSError as exc:
            raise ModbusCommunicationError(
                f"could not connect to {self._host}:{self._port}: {exc}"
            ) from exc
        self._socket = connection
        logger.info("opened %s", self.describe())

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def exchange(self, pdu: bytes, expected_response_bytes: Optional[int] = None) -> bytes:
        del expected_response_bytes  # TCP is self-describing via the MBAP length
        if self._socket is None:
            raise ModbusCommunicationError("transport is not open")

        with self._lock:
            self._transaction = (self._transaction + 1) & 0xFFFF
            transaction = self._transaction
            request = protocol.wrap_tcp_request(transaction, self._unit_id, pdu)
            try:
                self._socket.sendall(request)
                header = self._recv_exactly(7)
                length = (header[4] << 8) | header[5]
                body = self._recv_exactly(length - 1)
            except OSError as exc:
                raise ModbusCommunicationError(
                    f"TCP I/O failed: {type(exc).__name__}: {exc}"
                ) from exc
        return protocol.parse_tcp_response(header + body, transaction)

    def _recv_exactly(self, size: int) -> bytes:
        assert self._socket is not None
        buffer = b""
        while len(buffer) < size:
            chunk = self._socket.recv(size - len(buffer))
            if not chunk:
                raise ModbusCommunicationError(
                    f"connection closed after {len(buffer)} of {size} bytes"
                )
            buffer += chunk
        return buffer


class SimulatedTransport:
    """An in-memory controller that speaks real Modbus PDUs.

    Holds a register file (holding + input), answers FC03/FC04/FC06/FC16 exactly as
    a controller would, and can be told to fail: ``fail_next`` raises a communication
    error, ``exception_next`` returns a Modbus exception PDU, ``latency_ms`` adds bus
    delay. That makes the timeout, retry and error-translation paths executable
    without hardware.

    ``trace`` keeps every request PDU, so a test can assert what was written to
    which register - the only reliable way to check a command sequence.
    """

    def __init__(
        self,
        *,
        slave: int = 1,
        latency_ms: float = 0.0,
        holding: Optional[dict[int, int]] = None,
        inputs: Optional[dict[int, int]] = None,
    ) -> None:
        self._slave = slave
        self._latency_ms = latency_ms
        self.holding: dict[int, int] = dict(holding or {})
        self.inputs: dict[int, int] = dict(inputs or {})
        self.trace: list[bytes] = []
        self._open = False
        self.fail_next = 0
        self.exception_next: list[int] = []

    @property
    def is_open(self) -> bool:
        return self._open

    def describe(self) -> str:
        return f"simulated:slave={self._slave} latency={self._latency_ms}ms"

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def exchange(self, pdu: bytes, expected_response_bytes: Optional[int] = None) -> bytes:
        del expected_response_bytes
        if not self._open:
            raise ModbusCommunicationError("transport is not open")

        self.trace.append(bytes(pdu))
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000.0)

        if self.fail_next > 0:
            self.fail_next -= 1
            raise ModbusCommunicationError("simulated bus timeout")

        function_code = protocol.pdu_function_code(pdu)
        if self.exception_next:
            exception_code = self.exception_next.pop(0)
            return bytes([function_code | protocol.EXCEPTION_FLAG, exception_code])

        return self._handle(function_code, pdu)

    def _handle(self, function_code: int, pdu: bytes) -> bytes:
        if function_code == protocol.FunctionCode.READ_HOLDING_REGISTERS:
            start = (pdu[1] << 8) | pdu[2]
            count = (pdu[3] << 8) | pdu[4]
            values = [self.holding.get(start + offset, 0) for offset in range(count)]
            return self._read_response(function_code, values)

        if function_code == protocol.FunctionCode.READ_INPUT_REGISTERS:
            start = (pdu[1] << 8) | pdu[2]
            count = (pdu[3] << 8) | pdu[4]
            values = [self.inputs.get(start + offset, 0) for offset in range(count)]
            return self._read_response(function_code, values)

        if function_code == protocol.FunctionCode.WRITE_SINGLE_REGISTER:
            address = (pdu[1] << 8) | pdu[2]
            value = (pdu[3] << 8) | pdu[4]
            self.holding[address] = value
            return bytes(pdu)  # FC06 echoes the request

        if function_code == protocol.FunctionCode.WRITE_MULTIPLE_REGISTERS:
            start = (pdu[1] << 8) | pdu[2]
            count = (pdu[3] << 8) | pdu[4]
            values = [
                (pdu[6 + 2 * index] << 8) | pdu[7 + 2 * index]
                for index in range(count)
            ]
            for offset, value in enumerate(values):
                self.holding[start + offset] = value
            # FC16 answers with the start address and the quantity written.
            return bytes([function_code]) + bytes(pdu[1:5])

        # Unsupported function: exactly what a controller does.
        return bytes([function_code | protocol.EXCEPTION_FLAG, 0x01])

    @staticmethod
    def _read_response(function_code: int, values: Sequence[int]) -> bytes:
        body = bytearray([function_code, len(values) * 2])
        for value in values:
            body += bytes([(value >> 8) & 0xFF, value & 0xFF])
        return bytes(body)


def _rtu_frame(slave: int, pdu: bytes) -> bytes:
    """Local RTU encapsulation (keeps the transport independent of diagnostics)."""
    body = bytes([slave]) + bytes(pdu)
    crc = protocol.crc16_modbus(body)
    return body + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def _guess_response_size(pdu: bytes) -> int:
    """Response length for a request PDU, in RTU bytes including slave and CRC."""
    function_code = protocol.pdu_function_code(pdu)
    if function_code in (
        protocol.FunctionCode.READ_HOLDING_REGISTERS,
        protocol.FunctionCode.READ_INPUT_REGISTERS,
    ):
        count = (pdu[3] << 8) | pdu[4]
        return 5 + 2 * count
    return 8  # writes echo slave + function + address(2) + value(2) + crc(2)


def build_transport(kind: Optional[str] = None) -> Transport:
    """Construct the configured transport.

    ``MODBUS_TRANSPORT`` selects ``simulated`` (default for the MVP), ``rtu`` or
    ``tcp``. Simulated being the default is deliberate: with no converter attached,
    silently trying to open ``/dev/ttyUSB0`` and failing at every command would be
    worse than an explicit, logged simulation.
    """
    resolved = (kind or os.getenv("MODBUS_TRANSPORT", "simulated")).strip().lower()

    if resolved in ("simulated", "sim", "mock", "none"):
        logger.warning(
            "MODBUS_TRANSPORT=%s - commands are written to an in-memory register "
            "file, not to hardware. No robot will move.",
            resolved,
        )
        return SimulatedTransport(
            slave=int(os.getenv("MODBUS_SLAVE_ID", "1")),
            latency_ms=float(os.getenv("MODBUS_SIMULATED_LATENCY_MS", "0") or 0.0),
        )

    if resolved in ("rtu", "serial", "rs485"):
        return SerialRtuTransport(
            port=os.getenv("MODBUS_INTERFACE", "/dev/ttyUSB0"),
            baudrate=int(os.getenv("MODBUS_BAUD_RATE", "9600")),
            slave=int(os.getenv("MODBUS_SLAVE_ID", "1")),
            timeout=float(os.getenv("MODBUS_TIMEOUT_SECONDS", "1.0")),
        )

    if resolved == "tcp":
        host = os.getenv("MODBUS_HOST", "")
        if not host:
            raise RobotError(
                "MODBUS_TRANSPORT=tcp requires MODBUS_HOST", code="CONFIG_ERROR"
            )
        return TcpTransport(
            host=host,
            port=int(os.getenv("MODBUS_PORT", "502")),
            unit_id=int(os.getenv("MODBUS_SLAVE_ID", "1")),
            timeout=float(os.getenv("MODBUS_TIMEOUT_SECONDS", "1.0")),
        )

    raise RobotError(
        f"unknown MODBUS_TRANSPORT={resolved!r}; expected simulated, rtu or tcp",
        code="CONFIG_ERROR",
    )
