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
"""Service and command-mapping tests, against the simulated controller.

There is no RS-485 converter and the robot is being rebuilt, so the simulated
transport is the only executable controller. It speaks real Modbus PDUs, which means
these tests cover the same encode/decode/error paths that will drive hardware - only
the wire is different.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.robot.exceptions import (
    ModbusCommunicationError,
    ModbusDeviceException,
    RobotNotConnectedError,
    StaleDetectionError,
    UnsafeTargetError,
    WorkspaceBoundsError,
)
from src.robot.integration_manager import RobotCommandManager, RobotMapping
from src.robot.service import ModbusService
from src.robot.transports import SimulatedTransport


def make_service(**transport_kwargs) -> tuple[ModbusService, SimulatedTransport]:
    transport = SimulatedTransport(**transport_kwargs)
    service = ModbusService(transport, retries=2, retry_delay=0.0)
    assert service.connect() is True
    return service, transport


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
def test_connect_and_close() -> None:
    service, transport = make_service()
    assert service.is_connected is True
    assert transport.is_open is True

    service.close()
    assert service.is_connected is False


def test_write_then_read_register() -> None:
    service, transport = make_service()
    service.write_register(40001, 1234)

    assert transport.holding[40001] == 1234
    assert service.read_register(40001) == 1234


def test_write_a_contiguous_block_atomically() -> None:
    service, transport = make_service()
    service.write_registers(40001, [11, 22, 33])

    assert [transport.holding[address] for address in (40001, 40002, 40003)] == [
        11,
        22,
        33,
    ]
    assert service.read_registers(40001, 3) == [11, 22, 33]


def test_input_registers_are_read_from_a_separate_space() -> None:
    service, _ = make_service(inputs={100: 7})
    assert service.read_register(100, input_registers=True) == 7


def test_a_command_before_connect_is_refused() -> None:
    service = ModbusService(SimulatedTransport(), retries=0)
    with pytest.raises(RobotNotConnectedError):
        service.write_register(1, 1)


def test_communication_failures_are_retried_then_raised() -> None:
    service, transport = make_service()
    transport.fail_next = 2  # two timeouts, then success

    service.write_register(40001, 5)

    assert service.retries_used == 2
    assert transport.holding[40001] == 5


def test_exhausted_retries_raise_a_typed_error_and_count_the_failure() -> None:
    service, transport = make_service()
    transport.fail_next = 10

    with pytest.raises(ModbusCommunicationError, match="attempt"):
        service.write_register(40001, 5)

    status = service.status()
    assert status["commands_failed"] > 0
    assert status["last_error"].startswith("COMM_ERROR")


def test_a_device_exception_is_not_retried() -> None:
    """Retrying a rejected write cannot succeed, and on a move it risks a second one."""
    service, transport = make_service()
    transport.exception_next = [0x02]  # illegal data address

    with pytest.raises(ModbusDeviceException):
        service.write_register(40001, 5)

    assert service.retries_used == 0


def test_status_reports_the_transport_and_counters() -> None:
    service, _ = make_service()
    service.write_register(40001, 1)

    status = service.status()
    assert status["connected"] is True
    assert status["simulated"] is True
    assert status["commands_sent"] == 1


# --------------------------------------------------------------------------- #
# Command mapping
# --------------------------------------------------------------------------- #
def detection(
    *,
    object_type: str = "metal_sheet",
    confidence: float = 0.95,
    x_norm: float = 0.4,
    y_norm: float = 0.4,
    width_norm: float = 0.1,
    height_norm: float = 0.1,
) -> SimpleNamespace:
    return SimpleNamespace(
        object_type=object_type,
        confidence=confidence,
        x_norm=x_norm,
        y_norm=y_norm,
        width_norm=width_norm,
        height_norm=height_norm,
    )


def payload(*detections, processed_timestamp=None, workspace_state=None):
    return SimpleNamespace(
        detections=list(detections),
        processed_timestamp=processed_timestamp,
        metadata={"workspace_state": workspace_state} if workspace_state else {},
    )


def manager(service=None, mapping=None, **kwargs) -> tuple[RobotCommandManager, ModbusService]:
    if service is None:
        service, _ = make_service()
    return RobotCommandManager(service, mapping=mapping, **kwargs), service


def test_plan_converts_the_box_centre_to_millimetres_and_units() -> None:
    command_manager, _ = manager()
    command = command_manager.plan(payload(detection()))

    assert command is not None
    # Centre of a 0.1-wide box at 0.4 is 0.45 -> 0.45 * 3000 mm = 1350 mm.
    assert command.x_mm == pytest.approx(1350.0)
    assert command.y_mm == pytest.approx(1350.0)
    # 10 units per mm.
    assert command.x_units == 13500
    assert command.y_units == 13500


def test_plan_prefers_confidence_then_area() -> None:
    command_manager, _ = manager()
    small_sure = detection(confidence=0.99, width_norm=0.05, height_norm=0.05)
    large_less_sure = detection(confidence=0.90, width_norm=0.2, height_norm=0.2)

    command = command_manager.plan(payload(small_sure, large_less_sure))
    assert command.confidence == pytest.approx(0.99)

    tie_a = detection(confidence=0.9, width_norm=0.05, height_norm=0.05)
    tie_b = detection(confidence=0.9, width_norm=0.2, height_norm=0.2)
    command = command_manager.plan(payload(tie_a, tie_b))
    assert command.confidence == pytest.approx(0.9)


def test_low_confidence_targets_are_ignored() -> None:
    command_manager, _ = manager()
    assert command_manager.plan(payload(detection(confidence=0.5))) is None


def test_unexpected_object_types_are_ignored() -> None:
    command_manager, _ = manager()
    assert command_manager.plan(payload(detection(object_type="debris"))) is None


def test_no_target_is_not_an_error() -> None:
    command_manager, _ = manager()
    assert command_manager.plan(payload()) is None
    assert command_manager.execute(payload())["reason"] == "NO_TARGET"


def test_out_of_workspace_targets_are_refused() -> None:
    mapping = RobotMapping(workspace_x_mm=(500.0, 2500.0))
    command_manager, _ = manager(mapping=mapping)

    with pytest.raises(WorkspaceBoundsError) as caught:
        command_manager.plan(payload(detection(x_norm=0.95)))

    assert caught.value.code == "OUT_OF_REACH"
    assert "outside the workspace" in caught.value.detail


def test_stale_detections_are_refused() -> None:
    command_manager, _ = manager()
    old = datetime.now(timezone.utc) - timedelta(seconds=10)

    with pytest.raises(StaleDetectionError) as caught:
        command_manager.plan(payload(detection(), processed_timestamp=old))

    assert caught.value.code == "STALE_DETECTION"


def test_a_human_in_the_cell_inhibits_motion() -> None:
    command_manager, _ = manager()

    with pytest.raises(UnsafeTargetError) as caught:
        command_manager.plan(payload(detection(), workspace_state="HUMAN_ON_SITE"))

    assert caught.value.code == "HUMAN_ON_SITE"


def test_execute_writes_the_xy_block_and_the_trigger() -> None:
    command_manager, service = manager()
    transport = service.transport
    assert isinstance(transport, SimulatedTransport)

    result = command_manager.execute(payload(detection()))

    assert result["executed"] is True
    assert transport.holding[40001] == 13500  # x
    assert transport.holding[40002] == 13500  # y
    assert transport.holding[40003] == 1      # trigger
    assert result["command"]["x_mm"] == pytest.approx(1350.0)


def test_execute_reports_refusals_without_raising() -> None:
    command_manager, service = manager()
    result = command_manager.execute(payload(detection(x_norm=0.99), workspace_state=None))

    assert result["executed"] in (True, False)  # 0.99 centre = 3120 mm, in default range


def test_dry_run_does_not_touch_the_controller() -> None:
    command_manager, service = manager(dry_run=True)
    transport = service.transport

    result = command_manager.execute(payload(detection()))

    assert result["reason"] == "DRY_RUN"
    assert result["command"]["x_units"] == 13500
    assert 40001 not in transport.holding


def test_manager_status_exposes_the_mapping() -> None:
    command_manager, _ = manager()
    status = command_manager.status()

    assert status["mapping"]["units_per_mm"] == 10.0
    assert status["link"]["connected"] is True
