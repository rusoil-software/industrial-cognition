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
"""Throughput and latency accounting for the vision pipeline.

The 60 FPS requirement is only meaningful if the system reports, continuously:

* **capture FPS**     - what the camera actually delivered,
* **completion FPS**  - how many frames received detections (the number the
  requirement is about),
* **in-flight** depth - how much work is outstanding, and
* **end-to-end latency** - capture-to-result, not just inference time.

Those four numbers together are what distinguishes "60 FPS achieved" from "60 FPS
captured and 12 FPS analysed", and that distinction is the whole point of this
module. A system that silently queues is worse than one that drops frames: the
queue hides capacity shortfall behind ever-growing latency, so the operator sees
stale detections while every internal metric looks healthy.

Frames dropped on purpose (*load shedding*) are therefore counted separately from
frames that failed, and both are surfaced. See `doc/vision-pipeline.md` §4.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

__all__ = ["FpsMeter", "LatencyTracker", "PipelineStats", "percentile"]


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile of an already-sorted list.

    Implemented here rather than pulled from NumPy so the accounting has no heavy
    dependency and can be unit-tested exactly.
    """
    if not sorted_values:
        return 0.0
    if fraction <= 0:
        return float(sorted_values[0])
    if fraction >= 1:
        return float(sorted_values[-1])
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


class FpsMeter:
    """Rate over a sliding time window.

    A window rather than a cumulative average: a cumulative average takes minutes
    to reveal that throughput collapsed, which is far too slow for a control loop
    feeding a robot.
    """

    def __init__(self, window_seconds: float = 5.0, clock=time.monotonic) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window = window_seconds
        self._clock = clock
        self._events: Deque[float] = deque()
        self._lock = threading.Lock()
        self._total = 0

    def record(self, count: int = 1) -> None:
        now = self._clock()
        with self._lock:
            for _ in range(count):
                self._events.append(now)
            self._total += count
            self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self._window
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    def fps(self) -> float:
        now = self._clock()
        with self._lock:
            self._trim(now)
            if not self._events:
                return 0.0
            span = now - self._events[0]
            if span <= 0:
                # All events share this instant; report the count over one window
                # rather than dividing by zero.
                return float(len(self._events)) / self._window
            return float(len(self._events)) / span


class LatencyTracker:
    """Bounded ring of latency samples with percentile readout.

    Bounded on purpose: at 60 FPS an unbounded list grows by 216k samples an hour
    per pipeline, which is a slow memory leak in a process that must run for
    weeks.
    """

    def __init__(self, capacity: int = 2048) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._samples: Deque[float] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record(self, milliseconds: float) -> None:
        with self._lock:
            self._samples.append(float(milliseconds))

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            values = sorted(self._samples)
        if not values:
            return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        return {
            "count": len(values),
            "p50": round(percentile(values, 0.50), 3),
            "p95": round(percentile(values, 0.95), 3),
            "p99": round(percentile(values, 0.99), 3),
            "max": round(values[-1], 3),
        }


@dataclass
class PipelineStats:
    """Mutable counters for one pipeline instance (one camera).

    Deliberately a plain mutable object behind a lock rather than per-thread
    copies: the numbers are read by the health endpoint and written by the capture
    thread and the dispatcher, and correctness here is worth one lock.
    """

    camera_id: str = "unknown"
    window_seconds: float = 5.0

    captured: int = 0
    encoded: int = 0
    dispatched: int = 0
    completed: int = 0
    failed: int = 0
    dropped_capacity: int = 0
    dropped_encode: int = 0
    dropped_stale: int = 0

    capture_fps: FpsMeter = field(init=False)
    completion_fps: FpsMeter = field(init=False)
    dispatch_fps: FpsMeter = field(init=False)
    latency_ms: LatencyTracker = field(init=False)
    queue_ms: LatencyTracker = field(init=False)
    inference_ms: LatencyTracker = field(init=False)
    encode_ms: LatencyTracker = field(init=False)

    _lock: threading.Lock = field(init=False, repr=False)
    _started: float = field(init=False, default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self.capture_fps = FpsMeter(self.window_seconds)
        self.completion_fps = FpsMeter(self.window_seconds)
        self.dispatch_fps = FpsMeter(self.window_seconds)
        self.latency_ms = LatencyTracker()
        self.queue_ms = LatencyTracker()
        self.inference_ms = LatencyTracker()
        self.encode_ms = LatencyTracker()

    # -- counters ----------------------------------------------------------
    def record_capture(self) -> None:
        with self._lock:
            self.captured += 1
        self.capture_fps.record()

    def record_encode(self, milliseconds: float) -> None:
        with self._lock:
            self.encoded += 1
        self.encode_ms.record(milliseconds)

    def record_dispatch(self) -> None:
        with self._lock:
            self.dispatched += 1
        self.dispatch_fps.record()

    def record_completion(
        self,
        *,
        e2e_latency_ms: float,
        queue_ms: float,
        inference_ms: float,
    ) -> None:
        with self._lock:
            self.completed += 1
        self.completion_fps.record()
        self.latency_ms.record(e2e_latency_ms)
        self.queue_ms.record(queue_ms)
        self.inference_ms.record(inference_ms)

    def record_failure(self) -> None:
        with self._lock:
            self.failed += 1

    def record_drop_capacity(self, count: int = 1) -> None:
        with self._lock:
            self.dropped_capacity += count

    def record_drop_stale(self, count: int = 1) -> None:
        with self._lock:
            self.dropped_stale += count

    def record_drop_encode(self, count: int = 1) -> None:
        with self._lock:
            self.dropped_encode += count

    # -- readout -----------------------------------------------------------
    def snapshot(self) -> dict[str, object]:
        """Return a JSON-serialisable snapshot for logging, /metrics or the API."""
        with self._lock:
            counters = {
                "captured": self.captured,
                "encoded": self.encoded,
                "dispatched": self.dispatched,
                "completed": self.completed,
                "failed": self.failed,
                "dropped_capacity": self.dropped_capacity,
                "dropped_stale": self.dropped_stale,
                "dropped_encode": self.dropped_encode,
            }
        uptime = max(1e-9, time.monotonic() - self._started)
        capture_fps = round(self.capture_fps.fps(), 2)
        completion_fps = round(self.completion_fps.fps(), 2)
        return {
            "camera_id": self.camera_id,
            "uptime_seconds": round(uptime, 1),
            "counters": counters,
            "rates": {
                "capture_fps": capture_fps,
                "dispatch_fps": round(self.dispatch_fps.fps(), 2),
                "completion_fps": completion_fps,
                # The headline number for the 60 FPS requirement. `capture_fps`
                # alone is what a naive implementation would report.
                "completion_ratio": (
                    round(completion_fps / capture_fps, 3) if capture_fps else 0.0
                ),
            },
            "latency_ms": {
                "end_to_end": self.latency_ms.snapshot(),
                "queue": self.queue_ms.snapshot(),
                "inference": self.inference_ms.snapshot(),
                "encode": self.encode_ms.snapshot(),
            },
        }

    def meets_target(self, target_fps: float = 60.0, tolerance: float = 0.95) -> bool:
        """Whether analysed throughput is within ``tolerance`` of *target_fps*.

        Used by the integration test and by the readiness probe: "we are capturing
        60 FPS" is not the same claim, and the system should not report healthy on
        the weaker one.
        """
        return self.completion_fps.fps() >= target_fps * tolerance
