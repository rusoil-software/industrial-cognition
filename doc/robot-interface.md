# Robot Interface (Modbus / RS-485)

Status: **implemented server-side, not yet tested against hardware.** There is no
USB-to-RS485 or Ethernet-to-RS485 converter yet, and the alpha robot is being
rebuilt, so the MVP ships with a simulated controller and the real transports
written but unexercised. This document records what is proven and what is not.

## Layering

| Module | Responsibility | Hardware needed |
| --- | --- | --- |
| `src/robot/protocol.py` | CRC-16/MODBUS, RTU framing, PDU builders, response validation, exception decoding, MBAP | no |
| `src/robot/transports.py` | `SerialRtuTransport` (RS-485), `TcpTransport`, `SimulatedTransport` | only for the first two |
| `src/robot/service.py` | retries, typed error translation, counters | no |
| `src/robot/integration_manager.py` | detection → millimetres → register units, workspace/staleness/interlock refusals | no |
| `src/robot/exceptions.py` | `COMM_ERROR`, `DEVICE_EXCEPTION`, `OUT_OF_REACH`, `HUMAN_ON_SITE`, `STALE_DETECTION` | no |

The physical layer is a single seam - `Transport.exchange(pdu) -> pdu` - so the
code that will drive hardware is the code that runs against the simulator today.
The simulator parses and answers **real PDUs**, including Modbus exception
responses, so protocol bugs cannot hide behind it.

## Why simulated is the default

`MODBUS_TRANSPORT` defaults to `simulated`. Trying to open `/dev/ttyUSB0` on a
server with no converter would mean every command fails at runtime with a
communication error that looks like a broken robot. The simulator writes to an
in-memory register file, logs a startup warning, and reports `simulated: true` in
`ModbusService.status()` so nothing can mistake it for motion.

## Command path

```
detection (normalised 0..1)
  -> box centre
  -> millimetres        (x_centre * SCALE_FACTOR_MM_PER_UNIT)
  -> register units     (mm * RobotMapping.units_per_mm)
  -> FC16 write of [x, y] to xy_register
  -> FC06 write of trigger_value to trigger_register
```

Millimetres are kept as an explicit intermediate unit so recalibrating the camera
does not silently change what the robot does; the controller's register scale is a
separate, robot-side number.

## Refusals (all typed, all logged, none silent)

* **`STALE_DETECTION`** - the frame is older than `stale_after_seconds` (2 s by
  default). Acting on a stale view is the failure the pipeline's load shedding
  exists to avoid.
* **`OUT_OF_REACH`** - the target is outside `workspace_x_mm` / `workspace_y_mm`. A
  normalised coordinate is *always* in range, so without this check a spurious
  detection at the frame edge becomes a command to drive into a limit.
* **`HUMAN_ON_SITE`** (and `EMERGENCY_STOP`) - a workspace state in the payload
  inhibits motion.
* **`DEVICE_EXCEPTION`** - the controller refused. **Not retried**: a retry cannot
  change an illegal-data-address verdict, and on a motion command it risks issuing
  a second move.
* **`COMM_ERROR`** - the bus failed; retried a bounded number of times, then raised.

## Register map is a placeholder

`RobotMapping` defaults (`xy_register=40001`, `trigger_register=40003`,
`units_per_mm=10.0`, a 3000 × 3000 mm workspace) are **assumptions about the
controller**, not facts. They are grouped in one dataclass so confirming them when
the hardware arrives is a single reviewed change.

## Verified / not verified

Verified: 46 tests in `tests/robot`. CRC against the published check value
(`0x4B37`), byte-exact frames, CRC/slave/exception rejection, signed register round
trips, MBAP layout; service retry and no-retry-on-refusal behaviour, counters; and
the mapping chain, workspace/staleness/interlock refusals and the exact register
writes for a move, all against the simulated controller.

Not verified, and why: **serial and TCP transports** need a converter/controller;
**real register addresses and units** need the robot; **timing and termination**
(baud rate, RS-485 bias/termination, DE/RE line behaviour) can only be validated on
the bus. When hardware arrives, the checks to run first are: `SerialRtuTransport`
opens the port, a single FC03 read of a known register succeeds, and a read of a
nonexistent register returns `DEVICE_EXCEPTION` (which proves exception decoding
end to end).
