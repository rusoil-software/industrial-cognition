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
# ==============================================================================
"""Modbus framing tests - the part that must be right before hardware exists.

With no RS-485 converter, these tests are the only evidence that the framing is
correct, so they lean on the published CRC check value and on byte-exact
expectations rather than on round-tripping the code against itself.
"""

from __future__ import annotations

import pytest

from src.robot import protocol
from src.robot.exceptions import ModbusCommunicationError, ModbusDeviceException


# --------------------------------------------------------------------------- #
# CRC
# --------------------------------------------------------------------------- #
def test_crc16_matches_the_published_check_value() -> None:
    """CRC-16/MODBUS of "123456789" is 0x4B37.

    Pinning the published value (rather than a value produced by this code) is what
    catches a changed polynomial or reflected-ness - the kind of change that makes
    a controller reject every frame with no useful diagnostic.
    """
    assert protocol.crc16_modbus(b"123456789") == 0x4B37


def test_crc16_of_empty_is_the_initial_value() -> None:
    assert protocol.crc16_modbus(b"") == 0xFFFF


# --------------------------------------------------------------------------- #
# RTU frames
# --------------------------------------------------------------------------- #
def test_read_request_is_byte_exact() -> None:
    frame = protocol.build_read_holding_registers(slave=1, start=0x0000, count=1)
    # slave | FC03 | addr hi/lo | count hi/lo | CRC lo/hi
    assert frame[:6] == bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x01])
    assert len(frame) == 8
    crc = frame[-2] | (frame[-1] << 8)
    assert crc == protocol.crc16_modbus(frame[:-2])


def test_write_single_register_frame_round_trips_through_validation() -> None:
    frame = protocol.build_write_single_register(slave=7, address=40001, value=1234)
    response = protocol.split_rtu_response(frame, slave=7)
    # FC06 echoes the request PDU (function + address + value).
    assert response == frame[1:-2]


def test_write_multiple_register_pdu_shape() -> None:
    pdu = protocol.pdu_write_multiple(40001, [10, 20])
    assert pdu[0] == protocol.FunctionCode.WRITE_MULTIPLE_REGISTERS
    assert pdu[5] == 4  # byte count for two registers
    assert pdu[6:] == bytes([0x00, 0x0A, 0x00, 0x14])


def test_frame_rejects_an_out_of_range_slave() -> None:
    with pytest.raises(ValueError):
        protocol.build_read_holding_registers(slave=300, start=0, count=1)


def test_register_count_limits_are_enforced() -> None:
    with pytest.raises(ValueError):
        protocol.pdu_read_holding(0, 0)
    with pytest.raises(ValueError):
        protocol.pdu_read_holding(0, protocol.MAX_READ_REGISTERS + 1)
    with pytest.raises(ValueError):
        protocol.pdu_write_multiple(0, [0] * (protocol.MAX_WRITE_REGISTERS + 1))


def test_register_values_must_fit_in_16_bits() -> None:
    with pytest.raises(ValueError):
        protocol.pdu_write_single(0, 0x10000)


# --------------------------------------------------------------------------- #
# Response validation
# --------------------------------------------------------------------------- #
def test_bad_crc_is_reported_as_a_communication_error() -> None:
    frame = bytearray(protocol.build_write_single_register(1, 10, 20))
    frame[-1] ^= 0xFF
    with pytest.raises(ModbusCommunicationError, match="CRC"):
        protocol.split_rtu_response(bytes(frame), slave=1)


def test_a_response_from_the_wrong_slave_is_rejected() -> None:
    frame = protocol.build_write_single_register(slave=2, address=1, value=1)
    with pytest.raises(ModbusCommunicationError, match="slave"):
        protocol.split_rtu_response(frame, slave=1)


def test_a_short_response_is_rejected_with_a_useful_message() -> None:
    with pytest.raises(ModbusCommunicationError, match="too short"):
        protocol.split_rtu_response(b"\x01\x03", slave=1)


def test_device_exception_is_translated_with_its_meaning() -> None:
    """The controller's own refusal must surface as a typed, readable exception."""
    pdu = bytes([0x03 | protocol.EXCEPTION_FLAG, 0x02])  # illegal data address
    body = bytes([0x01]) + pdu
    crc = protocol.crc16_modbus(body)
    frame = body + bytes([crc & 0xFF, (crc >> 8) & 0xFF])

    with pytest.raises(ModbusDeviceException) as caught:
        protocol.split_rtu_response(frame, slave=1)

    error = caught.value
    assert error.code == "DEVICE_EXCEPTION"
    assert error.function_code == 0x03
    assert error.exception_code == 0x02
    assert "illegal data address" in error.detail


def test_read_response_parsing() -> None:
    pdu = bytes([0x03, 0x04, 0x00, 0x0A, 0x01, 0x02])
    assert protocol.parse_read_registers(pdu, expected_count=2) == [10, 258]


def test_read_response_byte_count_mismatch_is_rejected() -> None:
    pdu = bytes([0x03, 0x04, 0x00, 0x0A])
    with pytest.raises(ModbusCommunicationError, match="truncated|announced"):
        protocol.parse_read_registers(pdu, expected_count=2)


# --------------------------------------------------------------------------- #
# Signed registers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [0, 1, -1, 12345, -12345, 32767, -32768])
def test_signed_register_round_trip(value: int) -> None:
    assert protocol.decode_i16(protocol.encode_i16(value)) == value


def test_signed_register_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        protocol.encode_i16(40000)


# --------------------------------------------------------------------------- #
# Modbus TCP
# --------------------------------------------------------------------------- #
def test_tcp_wrapping_uses_the_mbap_layout() -> None:
    pdu = protocol.pdu_read_holding(0, 1)
    frame = protocol.wrap_tcp_request(transaction_id=0x1234, unit_id=1, pdu=body_pdu(pdu))

    assert frame[:2] == b"\x12\x34"
    assert frame[2:4] == b"\x00\x00"  # protocol id
    assert frame[4:6] == len(frame[6:]).to_bytes(2, "big")


def test_tcp_response_parsing_returns_the_pdu() -> None:
    pdu = bytes([0x03, 0x02, 0x00, 0x2A])
    frame = protocol.wrap_tcp_request(1, 1, pdu)
    assert protocol.parse_tcp_response(frame, transaction_id=1) == pdu


def test_tcp_transaction_mismatch_is_rejected() -> None:
    frame = protocol.wrap_tcp_request(5, 1, bytes([0x03, 0x02, 0x00, 0x01]))
    with pytest.raises(ModbusCommunicationError, match="transaction"):
        protocol.parse_tcp_response(frame, transaction_id=6)


def body_pdu(pdu: bytes) -> bytes:
    """Identity helper, kept so the TCP test reads as an intention."""
    return pdu
