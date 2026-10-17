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
"""Dispatch captured frames to vision workers and collect their results.

This is where the "60 FPS across several workers" requirement is actually
implemented, and the governing rule is:

    **Bound the work in flight, and shed frames when the bound is reached.**

A naive dispatcher publishes every captured frame and lets the broker queue grow.
That looks correct and is not: if capture runs at 60 FPS and analysis capacity is
30 FPS, the backlog grows at 30 frames/s forever. Throughput stays at the 30 FPS
the GPU can do while *latency* climbs without limit, so the operator sees
detections that are many seconds stale - actively dangerous when they drive a
robot arm. The failure is invisible in every metric except latency p99.

With a bounded in-flight window the system instead:

* keeps latency flat (the queue can never exceed ``max_in_flight`` frames),
* reports the true analysed FPS, and
* makes the capacity shortfall explicit through ``dropped_capacity``.

Dropping frames is the correct behaviour for a real-time control loop: a stale
detection is worth nothing, and a fresh one is worth more than two stale ones. The
60 FPS requirement is then met by **adding workers until analysed FPS reaches
60**, which `doc/vision-pipeline.md` §5 sizes from measured latency.

The transport is injected, so this class is testable without a broker and the same
logic drives Celery in production and an in-process fake in the tests.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from src.camera.frames import CapturedFrame
from src.pipeline.stats import PipelineStats
from src.vision.detections import FrameResult

__all__ = [
    "FrameDispatcher",
    "InProcessHandle",
    "InProcessTransport",
    "TaskHandle",
    "TaskTransport",
]

logger = logging.getLogger(__name__)


@runtime_checkable
class TaskHandle(Protocol):
    """A submitted task's result, polled without blocking."""

    @property
    def task_id(self) -> str: ...

    def ready(self) -> bool: ...

    def result(self) -> dict[str, Any]:
        """The task's return value. Only valid once :meth:`ready` is true."""


@runtime_checkable
class TaskTransport(Protocol):
    """How a frame payload reaches a worker.

    Implementations: the Celery transport in `src/pipeline/celery_transport.py`
    (production) and :class:`InProcessTransport` (tests, and a single-process mode
    for development without a broker).
    """

    @property
    def name(self) -> str: ...

    def submit(self, payload: dict[str, Any]) -> TaskHandle: ...


@dataclass
class _Pending:
    """A dispatched frame awaiting its result."""

    handle: TaskHandle
    frame: CapturedFrame
    dispatched_ns: int


class InProcessHandle:
    """A handle whose result is already available.

    Exists so the dispatcher can run against a synchronous transport. Useful for
    development without RabbitMQ and for tests that assert the dispatch/collect
    contract without broker timing noise.
    """

    def __init__(self, task_id: str, value: dict[str, Any]) -> None:
        self._task_id = task_id
        self._value = value

    @property
    def task_id(self) -> str:
        return self._task_id

    def ready(self) -> bool:
        return True

    def result(self) -> dict[str, Any]:
        return self._value


class InProcessTransport:
    """Runs the worker function synchronously in the calling thread.

    This is a *test and single-process development* transport, not a production
    one: it provides no parallelism, so it cannot reach 60 FPS with a real model.
    Its purpose is to make the dispatcher's contract executable and to allow a
    broker-free demo with the synthetic backend.
    """

    name = "in-process"

    def __init__(self, callable_: Any) -> None:
        """Args: callable_ taking a payload dict and returning a result dict."""
        self._callable = callable_
        self._counter = 0

    def submit(self, payload: dict[str, Any]) -> InProcessHandle:
        self._counter += 1
        task_id = f"inproc-{self._counter}"
        try:
            value = self._callable(payload)
        except Exception as exc:  # noqa: BLE001 - mirror the Celery failure shape
            value = {
                "camera_id": payload.get("camera_id", "unknown"),
                "frame_number": payload.get("frame_number", -1),
                "capture_ts_ns": payload.get("capture_ts_ns", 0),
                "detections": [],
                "error": f"{type(exc).__name__}: {exc}",
                "timings": {},
            }
        return InProcessHandle(task_id, value)


class FrameDispatcher:
    """Submits frames within a bounded window and collects completed results."""

    def __init__(
        self,
        transport: TaskTransport,
        stats: PipelineStats,
        *,
        max_in_flight: int = 32,
        clock=time.time_ns,
    ) -> None:
        if max_in_flight < 1:
            raise ValueError(
                "max_in_flight must be >= 1; 0 would drop every frame"
            )
        self._transport = transport
        self._stats = stats
        self._max_in_flight = max_in_flight
        self._pending: list[_Pending] = []
        self._clock = clock

    # -- introspection -----------------------------------------------------
    @property
    def transport_name(self) -> str:
        return self._transport.name

    @property
    def in_flight(self) -> int:
        return len(self._pending)

    @property
    def max_in_flight(self) -> int:
        return self._max_in_flight

    @property
    def saturated(self) -> bool:
        """Whether the window is full, i.e. capture now outpaces capacity."""
        return len(self._pending) >= self._max_in_flight

    # -- dispatch ----------------------------------------------------------
    def dispatch(
        self,
        frame: CapturedFrame,
        prompt: Any = None,
    ) -> bool:
        """Submit *frame* for analysis.

        Returns:
            ``True`` when the frame was accepted, ``False`` when it was shed
            because the in-flight window is full. The caller must treat ``False``
            as normal operation - it is the designed response to a capacity
            shortfall, not an error.
        """
        if self.saturated:
            self._stats.record_drop_capacity()
            return False

        payload = frame.to_payload()
        if prompt is not None:
            # The prompt travels with the frame so workers stay stateless:
            # no shared cache to invalidate across pods when a user changes it.
            payload["prompt"] = (
                prompt.to_dict() if hasattr(prompt, "to_dict") else prompt
            )
        handle = self._transport.submit(payload)
        self._pending.append(
            _Pending(handle=handle, frame=frame, dispatched_ns=self._clock())
        )
        self._stats.record_dispatch()
        return True

    # -- collection --------------------------------------------------------
    def collect(self) -> list[FrameResult]:
        """Return every result ready right now; never blocks.

        Non-blocking by design: the capture loop calls this once per frame, so a
        blocking collect would stall capture - the exact coupling this design
        exists to avoid.
        """
        if not self._pending:
            return []

        completed: list[FrameResult] = []
        still_pending: list[_Pending] = []

        for pending in self._pending:
            if not pending.handle.ready():
                still_pending.append(pending)
                continue
            completed.append(self._finalise(pending))

        self._pending = still_pending
        return completed

    def _finalise(self, pending: _Pending) -> FrameResult:
        """Turn one completed task into a result and update the statistics."""
        try:
            payload = pending.handle.result()
        except Exception as exc:  # noqa: BLE001 - a broker-level failure
            self._stats.record_failure()
            logger.warning(
                "task %s failed: %s", pending.handle.task_id, exc
            )
            return FrameResult(
                camera_id=pending.frame.camera_id,
                frame_number=pending.frame.frame_number,
                capture_ts_ns=pending.frame.capture_ts_ns,
                e2e_latency_ms=self._e2e_ms(pending),
                error=f"transport: {type(exc).__name__}: {exc}",
            )

        if not payload:
            self._stats.record_failure()
            return FrameResult(
                camera_id=pending.frame.camera_id,
                frame_number=pending.frame.frame_number,
                capture_ts_ns=pending.frame.capture_ts_ns,
                e2e_latency_ms=self._e2e_ms(pending),
                error="empty task result",
            )

        result = FrameResult.from_wire(payload)
        e2e_ms = self._e2e_ms(pending)
        # `dataclasses.replace` rather than mutation: FrameResult is frozen so a
        # result cannot be quietly altered by a consumer after accounting.
        from dataclasses import replace

        result = replace(result, e2e_latency_ms=e2e_ms)

        if result.error:
            self._stats.record_failure()
        else:
            self._stats.record_completion(
                e2e_latency_ms=e2e_ms,
                queue_ms=result.queue_ms,
                inference_ms=result.timings.inference_ms,
            )
        return result

    def _e2e_ms(self, pending: _Pending) -> float:
        """Capture-to-completion latency, clamped at zero.

        Clamped because capture and worker use wall-clock timestamps across
        process boundaries; an NTP step during a run would otherwise produce a
        negative latency and poison the percentile accounting.
        """
        return max(0.0, (self._clock() - pending.frame.capture_ts_ns) / 1e6)

    # -- shutdown ----------------------------------------------------------
    def drain(self, timeout_ms: float = 5000.0, poll_interval_ms: float = 5.0) -> list[FrameResult]:
        """Collect until the in-flight window is empty or *timeout_ms* elapses.

        Used at shutdown so that frames already paid for are not discarded; the
        timeout prevents a wedged worker from blocking a rolling restart forever.
        """
        deadline = self._clock() + int(timeout_ms * 1e6)
        collected: list[FrameResult] = []
        while self._pending and self._clock() < deadline:
            collected.extend(self.collect())
            if self._pending:
                time.sleep(poll_interval_ms / 1000.0)
        collected.extend(self.collect())
        if self._pending:
            logger.warning(
                "shutdown drain left %d frame(s) in flight", len(self._pending)
            )
        return collected

    def reset(self) -> None:
        """Forget in-flight work (tests, or after a transport failure)."""
        self._pending.clear()
