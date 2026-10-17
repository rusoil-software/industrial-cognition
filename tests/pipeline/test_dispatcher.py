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
"""Tests for frame dispatch and load shedding.

The requirement these protect is subtle: the pipeline must **never** let its queue
grow without bound. A dispatcher that happily submits everything looks correct and
is not - when capture outruns analysis, latency climbs forever. So the assertions
here are about what *does not* happen: the in-flight window never exceeds its
bound, and excess frames are shed and counted.
"""

from __future__ import annotations

import pytest

from src.camera.frames import CapturedFrame
from src.pipeline.dispatcher import (
    FrameDispatcher,
    InProcessTransport,
    TaskHandle,
)
from src.pipeline.stats import PipelineStats


class FakeHandle:
    """A task handle the test completes explicitly."""

    def __init__(self, task_id: str) -> None:
        self._task_id = task_id
        self._ready = False
        self._value: dict | None = None
        self._error: Exception | None = None

    @property
    def task_id(self) -> str:
        return self._task_id

    def ready(self) -> bool:
        return self._ready

    def complete(self, value: dict) -> None:
        self._value = value
        self._ready = True

    def fail(self, error: Exception) -> None:
        self._error = error
        self._ready = True

    def result(self) -> dict:
        if self._error is not None:
            raise self._error
        if self._value is None:
            raise AssertionError("result() called before completion")
        return self._value


class FakeTransport:
    """Records submissions so the test controls completion timing."""

    name = "fake"

    def __init__(self) -> None:
        self.handles: list[FakeHandle] = []

    def submit(self, payload: dict) -> FakeHandle:
        handle = FakeHandle(f"task-{len(self.handles)}")
        self.handles.append(handle)
        return handle


def make_frame(number: int = 0, capture_ts_ns: int | None = None) -> CapturedFrame:
    import time

    return CapturedFrame(
        camera_id="cam0",
        frame_number=number,
        capture_ts_ns=capture_ts_ns if capture_ts_ns is not None else time.time_ns(),
        jpeg=b"\xff\xd8fake-jpeg\xff\xd9",
        width=960,
        height=960,
    )


def ok_result(frame: CapturedFrame, *, inference_ms: float = 12.0) -> dict:
    return {
        "camera_id": frame.camera_id,
        "frame_number": frame.frame_number,
        "capture_ts_ns": frame.capture_ts_ns,
        "detections": [
            {
                "object_type": "metal_sheet",
                "x_norm": 0.1,
                "y_norm": 0.2,
                "width_norm": 0.3,
                "height_norm": 0.1,
                "confidence": 0.9,
            }
        ],
        "queue_ms": 1.0,
        "worker_id": "w1",
        "error": None,
        "timings": {
            "preprocess_ms": 3.0,
            "inference_ms": inference_ms,
            "decode_ms": 1.0,
        },
    }


# --------------------------------------------------------------------------- #
# Bounded window
# --------------------------------------------------------------------------- #
def test_dispatch_accepts_until_the_window_is_full() -> None:
    transport = FakeTransport()
    stats = PipelineStats()
    dispatcher = FrameDispatcher(transport, stats, max_in_flight=3)

    accepted = [dispatcher.dispatch(make_frame(i)) for i in range(5)]

    assert accepted == [True, True, True, False, False]
    assert dispatcher.in_flight == 3
    assert len(transport.handles) == 3
    # Shed frames are counted separately from failures.
    assert stats.snapshot()["counters"]["dropped_capacity"] == 2
    assert stats.snapshot()["counters"]["failed"] == 0


def test_saturated_reflects_the_window() -> None:
    dispatcher = FrameDispatcher(FakeTransport(), PipelineStats(), max_in_flight=2)
    assert dispatcher.saturated is False
    dispatcher.dispatch(make_frame(0))
    assert dispatcher.saturated is False
    dispatcher.dispatch(make_frame(1))
    assert dispatcher.saturated is True


def test_dispatch_becomes_possible_again_after_collection() -> None:
    transport = FakeTransport()
    dispatcher = FrameDispatcher(transport, PipelineStats(), max_in_flight=1)

    first = make_frame(0)
    assert dispatcher.dispatch(first) is True
    assert dispatcher.dispatch(make_frame(1)) is False

    transport.handles[0].complete(ok_result(first))
    dispatcher.collect()

    assert dispatcher.dispatch(make_frame(2)) is True


def test_rejects_a_zero_window() -> None:
    with pytest.raises(ValueError, match="max_in_flight"):
        FrameDispatcher(FakeTransport(), PipelineStats(), max_in_flight=0)


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def test_collect_is_non_blocking() -> None:
    """Blocking here would stall capture, which is the coupling to avoid."""
    transport = FakeTransport()
    dispatcher = FrameDispatcher(transport, PipelineStats(), max_in_flight=4)
    dispatcher.dispatch(make_frame(0))

    assert dispatcher.collect() == []
    assert dispatcher.in_flight == 1


def test_collect_returns_completed_results_only() -> None:
    transport = FakeTransport()
    dispatcher = FrameDispatcher(transport, PipelineStats(), max_in_flight=4)
    first, second = make_frame(0), make_frame(1)
    dispatcher.dispatch(first)
    dispatcher.dispatch(second)

    transport.handles[0].complete(ok_result(first))
    results = dispatcher.collect()

    assert len(results) == 1
    assert results[0].frame_number == 0
    assert dispatcher.in_flight == 1


def test_completion_updates_latency_and_counters() -> None:
    import time

    transport = FakeTransport()
    stats = PipelineStats()
    dispatcher = FrameDispatcher(transport, stats, max_in_flight=2)

    frame = make_frame(5, capture_ts_ns=time.time_ns() - 20_000_000)  # 20 ms ago
    dispatcher.dispatch(frame)
    transport.handles[0].complete(ok_result(frame))
    results = dispatcher.collect()

    assert len(results) == 1
    assert results[0].detection_count == 1
    assert results[0].e2e_latency_ms >= 15.0

    snapshot = stats.snapshot()
    assert snapshot["counters"]["completed"] == 1
    assert snapshot["latency_ms"]["end_to_end"]["count"] == 1
    assert snapshot["latency_ms"]["inference"]["p50"] == pytest.approx(12.0)


def test_a_failed_result_is_counted_as_failure_not_completion() -> None:
    transport = FakeTransport()
    stats = PipelineStats()
    dispatcher = FrameDispatcher(transport, stats, max_in_flight=2)

    frame = make_frame(0)
    dispatcher.dispatch(frame)
    payload = ok_result(frame)
    payload["error"] = "ValueError: boom"
    payload["detections"] = []
    transport.handles[0].complete(payload)

    results = dispatcher.collect()

    assert results[0].error == "ValueError: boom"
    counters = stats.snapshot()["counters"]
    assert counters["failed"] == 1
    assert counters["completed"] == 0


def test_a_transport_exception_is_counted_as_failure() -> None:
    transport = FakeTransport()
    stats = PipelineStats()
    dispatcher = FrameDispatcher(transport, stats, max_in_flight=2)

    dispatcher.dispatch(make_frame(0))
    transport.handles[0].fail(RuntimeError("broker gone"))
    results = dispatcher.collect()

    assert results[0].error is not None
    assert stats.snapshot()["counters"]["failed"] == 1


def test_an_empty_task_result_is_counted_as_failure() -> None:
    transport = FakeTransport()
    stats = PipelineStats()
    dispatcher = FrameDispatcher(transport, stats, max_in_flight=2)

    dispatcher.dispatch(make_frame(0))
    transport.handles[0].complete({})
    dispatcher.collect()

    assert stats.snapshot()["counters"]["failed"] == 1


def test_drain_collects_pending_results() -> None:
    transport = FakeTransport()
    dispatcher = FrameDispatcher(transport, PipelineStats(), max_in_flight=4)

    frames = [make_frame(i) for i in range(3)]
    for frame in frames:
        dispatcher.dispatch(frame)
    for handle, frame in zip(transport.handles, frames):
        handle.complete(ok_result(frame))

    collected = dispatcher.drain(timeout_ms=100)

    assert len(collected) == 3
    assert dispatcher.in_flight == 0


def test_drain_does_not_hang_on_a_stuck_worker() -> None:
    transport = FakeTransport()
    dispatcher = FrameDispatcher(transport, PipelineStats(), max_in_flight=4)
    dispatcher.dispatch(make_frame(0))  # never completed

    collected = dispatcher.drain(timeout_ms=30)

    assert collected == []
    assert dispatcher.in_flight == 1


def test_reset_discards_in_flight_work() -> None:
    dispatcher = FrameDispatcher(FakeTransport(), PipelineStats(), max_in_flight=2)
    dispatcher.dispatch(make_frame(0))
    dispatcher.reset()
    assert dispatcher.in_flight == 0


# --------------------------------------------------------------------------- #
# In-process transport
# --------------------------------------------------------------------------- #
def test_in_process_transport_returns_the_callable_result() -> None:
    transport = InProcessTransport(lambda payload: {"frame_number": payload["frame_number"]})
    handle = transport.submit({"frame_number": 42})

    assert isinstance(handle, TaskHandle)  # structural conformance
    assert handle.ready() is True
    assert handle.result() == {"frame_number": 42}


def test_in_process_transport_turns_an_exception_into_an_error_result() -> None:
    def boom(payload: dict) -> dict:
        raise RuntimeError("worker exploded")

    handle = InProcessTransport(boom).submit({"camera_id": "cam0", "frame_number": 1})
    result = handle.result()

    assert "worker exploded" in result["error"]
    assert result["detections"] == []


def test_dispatcher_works_end_to_end_with_the_in_process_transport() -> None:
    """The broker-free path used for local development."""
    stats = PipelineStats()
    dispatcher = FrameDispatcher(
        InProcessTransport(lambda payload: ok_result(make_frame(payload["frame_number"]))),
        stats,
        max_in_flight=8,
    )

    for i in range(5):
        assert dispatcher.dispatch(make_frame(i)) is True

    results = dispatcher.collect()
    assert len(results) == 5
    assert stats.snapshot()["counters"]["completed"] == 5


def test_a_lost_task_expires_instead_of_stalling_the_window() -> None:
    """A task that never returns must not occupy the window forever.

    A message can be purged, expire, or be lost with an evicted worker. Its AsyncResult
    then never becomes ready, so without a timeout the entry stays in flight, the window
    fills, and dispatch() sheds every subsequent frame - a permanent stall from one lost
    message. Found live: purging the broker queue froze the camera for minutes.
    """
    from src.pipeline.stats import PipelineStats

    now = [0.0]
    stats = PipelineStats()
    dispatcher = FrameDispatcher(
        FakeTransport(),
        stats,
        max_in_flight=1,
        task_timeout_ms=1_000.0,
        clock=lambda: now[0],
    )

    assert dispatcher.dispatch(make_frame(0)) is True
    assert dispatcher.in_flight == 1
    # The window is full, so a second frame is shed while the first is outstanding.
    assert dispatcher.dispatch(make_frame(1)) is False

    now[0] = 2_000_000_000.0  # two seconds later, still no result
    dispatcher.collect()

    assert dispatcher.in_flight == 0, 'the lost task should have been dropped'
    assert stats.snapshot()['counters']['failed'] >= 1
    assert dispatcher.dispatch(make_frame(2)) is True, 'the window must free up'
