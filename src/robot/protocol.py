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
"""Modbus framing, as pure functions over bytes.

No I/O here at all. That is the point: the part of a Modbus integration that is
actually hard - CRC, byte order, exception decoding, response-length arithmetic -
is separated from the part that needs hardware, so it can be tested exhaustively
without a converter attached. `src/robot/transports.py` only moves these frames.

Two encapsulations are supported, because the deployment may use either:

* **RTU** - binary over RS-485, the authority per `doc/conventions.md` §3. Frame is
  ``slave | pdu | crc16(lo, hi)``.
* **TCP** - the optional fallback when the controller is on Ethernet. Frame is
  ``mbap | pdu`` where MBAP is ``transaction(2) protocol(2) length(2) unit(1)``.

Register values are 16-bit unsigned on the wire. Positions are often conceptually
signed, so :func:`encode_i16` / :func:`decode_i16` do the two's-complement mapping
in one reviewed place instead of at every call site.
"""

from __future__ import annotations

from typing import Sequence

from src.robot.exceptions import ModbusCommunicationError, ModbusDeviceException

__all__ = [
    "EXCEPTION_FLAG",
    "FunctionCode",
    "build_read_holding_registers",
    "build_read_input_registers",
    "build_write_multiple_registers",
    "build_write_single_register",
    "crc16_modbus",
    "decode_i16",
    "encode_i16",
    "parse_read_registers",
    "parse_tcp_response",
    "parse_write_ack",
    "ensure_no_exception",
    "pdu_function_code",
    "pdu_read_holding",
    "pdu_read_input",
    "pdu_write_multiple",
    "pdu_write_single",
    "split_rtu_response",
    "wrap_tcp_request",
]

#: Set in the response function code when the controller reports an exception.
EXCEPTION_FLAG = 0x80

#: Modbus ceiling: 125 registers per read, 123 per write (PDU size limit).
MAX_READ_REGISTERS = 125
MAX_WRITE_REGISTERS = 123
MAX_REGISTER_VALUE = 0xFFFF


class FunctionCode:
    """The subset of Modbus function codes this interface uses."""

    READ_HOLDING_REGISTERS = 0x03
    READ_INPUT_REGISTERS = 0x04
    WRITE_SINGLE_REGISTER = 0x06
    WRITE_MULTIPLE_REGISTERS = 0x10


def crc16_modbus(data: bytes) -> int:
    """CRC-16/MODBUS (poly 0xA001 reflected, init 0xFFFF).

    Check value: ``crc16_modbus(b"123456789") == 0x4B37``, the published value for
    this algorithm - asserted in the tests so a refactor cannot silently change the
    polynomial and leave every frame rejected by the controller.
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def _check_u16(value: int, what: str) -> int:
    if not 0 <= value <= MAX_REGISTER_VALUE:
        raise ValueError(f"{what} {value} does not fit in a 16-bit register")
    return value


def encode_i16(value: int) -> int:
    """Two's-complement encode a signed position into a register value."""
    if not -32768 <= value <= 32767:
        raise ValueError(f"{value} does not fit in a signed 16-bit register")
    return value & 0xFFFF


def decode_i16(value: int) -> int:
    """Two's-complement decode a register value into a signed position."""
    _check_u16(value, "register value")
    return value - 0x10000 if value & 0x8000 else value


def _frame_rtu(slave: int, pdu: bytes) -> bytes:
    if not 0 <= slave <= 247:
        raise ValueError(f"slave address {slave} is outside 1..247")
    body = bytes([slave]) + pdu
    crc = crc16_modbus(body)
    # Modbus RTU sends the CRC low byte first.
    return body + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def build_read_holding_registers(slave: int, start: int, count: int) -> bytes:
    """FC03 request: read *count* holding registers from *start*."""
    if not 1 <= count <= MAX_READ_REGISTERS:
        raise ValueError(f"count must be 1..{MAX_READ_REGISTERS}, got {count}")
    _check_u16(start, "start address")
    pdu = bytes(
        [
            FunctionCode.READ_HOLDING_REGISTERS,
            (start >> 8) & 0xFF,
            start & 0xFF,
            (count >> 8) & 0xFF,
            count & 0xFF,
        ]
    )
    return _frame_rtu(slave, pdu)


def build_read_input_registers(slave: int, start: int, count: int) -> bytes:
    """FC04 request: read input registers (the usual place for live positions)."""
    if not 1 <= count <= MAX_READ_REGISTERS:
        raise ValueError(f"count must be 1..{MAX_READ_REGISTERS}, got {count}")
    _check_u16(start, "start address")
    pdu = bytes(
        [
            FunctionCode.READ_INPUT_REGISTERS,
            (start >> 8) & 0xFF,
            start & 0xFF,
            (count >> 8) & 0xFF,
            count & 0xFF,
        ]
    )
    return _frame_rtu(slave, pdu)


def build_write_single_register(slave: int, address: int, value: int) -> bytes:
    """FC06 request: write one holding register."""
    _check_u16(address, "register address")
    _check_u16(value, "register value")
    pdu = bytes(
        [
            FunctionCode.WRITE_SINGLE_REGISTER,
            (address >> 8) & 0xFF,
            address & 0xFF,
            (value >> 8) & 0xFF,
            value & 0xFF,
        ]
    )
    return _frame_rtu(slave, pdu)


def build_write_multiple_registers(
    slave: int, start: int, values: Sequence[int]
) -> bytes:
    """FC16 request: write a contiguous block of holding registers.

    Preferred over several FC06 writes for a multi-axis move: one transaction means
    the controller applies the whole target atomically instead of moving through
    intermediate positions.
    """
    if not 1 <= len(values) <= MAX_WRITE_REGISTERS:
        raise ValueError(
            f"cannot write {len(values)} registers; limit is {MAX_WRITE_REGISTERS}"
        )
    _check_u16(start, "start address")
    payload = bytearray()
    for value in values:
        _check_u16(value, "register value")
        payload += bytes([(value >> 8) & 0xFF, value & 0xFF])

    pdu = bytes(
        [
            FunctionCode.WRITE_MULTIPLE_REGISTERS,
            (start >> 8) & 0xFF,
            start & 0xFF,
            (len(values) >> 8) & 0xFF,
            len(values) & 0xFF,
            len(values) * 2,
        ]
    ) + bytes(payload)
    return _frame_rtu(slave, pdu)


def split_rtu_response(frame: bytes, slave: int) -> bytes:
    """Validate an RTU response frame and return its PDU payload.

    Checks, in order, with a distinct message for each - because "Modbus failed" is
    useless when commissioning a converter at 9600 baud:

    1. the frame is long enough to contain a CRC;
    2. the trailing CRC matches;
    3. the responding slave is the one we addressed (a bus with several devices can
       answer out of turn);
    4. the controller did not set the exception flag.

    Raises:
        ModbusCommunicationError: for 1-3.
        ModbusDeviceException: for 4.
    """
    if len(frame) < 5:
        raise ModbusCommunicationError(
            f"response too short: {len(frame)} bytes (need at least 5)"
        )

    body, crc_bytes = frame[:-2], frame[-2:]
    expected = crc16_modbus(body)
    received = crc_bytes[0] | (crc_bytes[1] << 8)
    if expected != received:
        raise ModbusCommunicationError(
            f"CRC mismatch: computed 0x{expected:04X}, frame carried "
            f"0x{received:04X} (check baud rate, parity and RS-485 termination)"
        )

    responding = body[0]
    if responding != slave:
        raise ModbusCommunicationError(
            f"response came from slave {responding}, expected {slave}"
        )

    pdu = body[1:]
    if not pdu:
        raise ModbusCommunicationError("response contained no function code")
    if pdu[0] & EXCEPTION_FLAG:
        function_code = pdu[0] & ~EXCEPTION_FLAG
        exception_code = pdu[1] if len(pdu) > 1 else 0
        raise ModbusDeviceException(function_code, exception_code)
    return pdu


def parse_read_registers(pdu: bytes, expected_count: int) -> list[int]:
    """Decode an FC03/FC04 response PDU into register values."""
    if len(pdu) < 2:
        raise ModbusCommunicationError("read response is missing the byte count")
    byte_count = pdu[1]
    if byte_count != expected_count * 2:
        raise ModbusCommunicationError(
            f"read response announced {byte_count} bytes, expected "
            f"{expected_count * 2} for {expected_count} registers"
        )
    values = pdu[2:]
    if len(values) != byte_count:
        raise ModbusCommunicationError(
            f"read response truncated: {len(values)} of {byte_count} value bytes"
        )
    return [(values[i] << 8) | values[i + 1] for i in range(0, byte_count, 2)]


def parse_write_ack(pdu: bytes) -> None:
    """Validate an FC06/FC16 response PDU (an echo of the request header)."""
    if len(pdu) < 5:
        raise ModbusCommunicationError(
            f"write acknowledgement too short: {len(pdu)} bytes"
        )


def wrap_tcp_request(transaction_id: int, unit_id: int, pdu: bytes) -> bytes:
    """Add the MBAP header for Modbus TCP."""
    if not 0 <= transaction_id <= 0xFFFF:
        raise ValueError("transaction id must fit in 16 bits")
    if not 0 <= unit_id <= 255:
        raise ValueError("unit id must fit in a byte")
    length = len(pdu) + 1  # unit id counts toward the length field
    return (
        bytes(
            [
                (transaction_id >> 8) & 0xFF,
                transaction_id & 0xFF,
                0x00,  # protocol id: always 0 for Modbus
                0x00,
                (length >> 8) & 0xFF,
                length & 0xFF,
                unit_id,
            ]
        )
        + pdu
    )


def parse_tcp_response(frame: bytes, transaction_id: int) -> bytes:
    """Validate a Modbus TCP response and return its PDU."""
    if len(frame) < 9:
        raise ModbusCommunicationError(
            f"TCP response too short: {len(frame)} bytes"
        )
    received_transaction = (frame[0] << 8) | frame[1]
    if received_transaction != transaction_id:
        raise ModbusCommunicationError(
            f"transaction id {received_transaction} does not match request "
            f"{transaction_id}"
        )
    protocol = (frame[2] << 8) | frame[3]
    if protocol != 0:
        raise ModbusCommunicationError(f"unexpected protocol id {protocol}")
    length = (frame[4] << 8) | frame[5]
    pdu = frame[7:]
    if len(pdu) < length - 1:
        raise ModbusCommunicationError(
            f"TCP response truncated: {len(pdu)} of {length - 1} PDU bytes"
        )
    if pdu[0] & EXCEPTION_FLAG:
        function_code = pdu[0] & ~EXCEPTION_FLAG
        exception_code = pdu[1] if len(pdu) > 1 else 0
        raise ModbusDeviceException(function_code, exception_code)
    return pdu


# --------------------------------------------------------------------------- #
# PDU-level primitives
# --------------------------------------------------------------------------- #
# Transports encapsulate frames, not PDUs: RTU needs slave + CRC, TCP needs an MBAP
# header. The service builds a PDU and hands it to whichever transport is
# configured, so adding a transport cannot change how a command is encoded.

def pdu_read_holding(start: int, count: int) -> bytes:
    """FC03 PDU: read *count* holding registers from *start*."""
    if not 1 <= count <= MAX_READ_REGISTERS:
        raise ValueError(f"count must be 1..{MAX_READ_REGISTERS}, got {count}")
    _check_u16(start, "start address")
    return bytes(
        [
            FunctionCode.READ_HOLDING_REGISTERS,
            (start >> 8) & 0xFF,
            start & 0xFF,
            (count >> 8) & 0xFF,
            count & 0xFF,
        ]
    )


def pdu_read_input(start: int, count: int) -> bytes:
    """FC04 PDU: read *count* input registers from *start*."""
    if not 1 <= count <= MAX_READ_REGISTERS:
        raise ValueError(f"count must be 1..{MAX_READ_REGISTERS}, got {count}")
    _check_u16(start, "start address")
    return bytes(
        [
            FunctionCode.READ_INPUT_REGISTERS,
            (start >> 8) & 0xFF,
            start & 0xFF,
            (count >> 8) & 0xFF,
            count & 0xFF,
        ]
    )


def pdu_write_single(address: int, value: int) -> bytes:
    """FC06 PDU: write one holding register."""
    _check_u16(address, "register address")
    _check_u16(value, "register value")
    return bytes(
        [
            FunctionCode.WRITE_SINGLE_REGISTER,
            (address >> 8) & 0xFF,
            address & 0xFF,
            (value >> 8) & 0xFF,
            value & 0xFF,
        ]
    )


def pdu_write_multiple(start: int, values: Sequence[int]) -> bytes:
    """FC16 PDU: write a contiguous block of holding registers."""
    if not 1 <= len(values) <= MAX_WRITE_REGISTERS:
        raise ValueError(
            f"cannot write {len(values)} registers; limit is {MAX_WRITE_REGISTERS}"
        )
    _check_u16(start, "start address")
    payload = bytearray()
    for value in values:
        _check_u16(value, "register value")
        payload += bytes([(value >> 8) & 0xFF, value & 0xFF])
    header = bytes(
        [
            FunctionCode.WRITE_MULTIPLE_REGISTERS,
            (start >> 8) & 0xFF,
            start & 0xFF,
            (len(values) >> 8) & 0xFF,
            len(values) & 0xFF,
            len(values) * 2,
        ]
    )
    return header + bytes(payload)


def pdu_function_code(pdu: bytes) -> int:
    """Function code of a PDU (exception flag cleared)."""
    if not pdu:
        raise ModbusCommunicationError("empty PDU")
    return pdu[0] & ~EXCEPTION_FLAG


def ensure_no_exception(pdu: bytes) -> bytes:
    """Raise :class:ModbusDeviceException if *pdu* is an exception response.

    Exception decoding must not live only in the RTU/TCP parsers: a transport that
    returns a PDU directly (the simulated one, or a future in-process bridge) would
    otherwise hand an exception response to the value parsers, which report a
    confusing framing error instead of the controller's actual refusal. The service
    calls this for every response, so the guarantee holds for all transports.
    """
    if pdu and pdu[0] & EXCEPTION_FLAG:
        function_code = pdu[0] & ~EXCEPTION_FLAG
        exception_code = pdu[1] if len(pdu) > 1 else 0
        raise ModbusDeviceException(function_code, exception_code)
    return pdu
