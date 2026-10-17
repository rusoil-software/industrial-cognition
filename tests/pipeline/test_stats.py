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
"""Tests for the throughput/latency accounting.

These are the numbers the 60 FPS requirement is judged on, so they are tested with
an injected clock rather than by sleeping.
"""

from __future__ import annotations

import pytest

from src.pipeline.stats import FpsMeter, LatencyTracker, PipelineStats, percentile


class FakeClock:
    """A monotonic clock the test advances explicitly."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --------------------------------------------------------------------------- #
# percentile
# --------------------------------------------------------------------------- #
def test_percentile_edge_cases() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([42.0], 0.0) == 42.0
    assert percentile([42.0], 0.99) == 42.0


def test_percentile_interpolates() -> None:
    values = sorted([10.0, 20.0, 30.0, 40.0, 50.0])
    assert percentile(values, 0.5) == pytest.approx(30.0)
    assert percentile(values, 0.0) == pytest.approx(10.0)
    assert percentile(values, 1.0) == pytest.approx(50.0)
    # p25 sits a quarter of the way between 10 and 20.
    assert percentile(values, 0.25) == pytest.approx(20.0)


# --------------------------------------------------------------------------- #
# FpsMeter
# --------------------------------------------------------------------------- #
def test_fps_meter_computes_a_windowed_rate() -> None:
    clock = FakeClock()
    meter = FpsMeter(window_seconds=2.0, clock=clock)

    # 120 frames over 2 seconds = 60 FPS.
    for _ in range(120):
        meter.record()
        clock.advance(2.0 / 120.0)

    assert meter.fps() == pytest.approx(60.0, rel=0.05)
    assert meter.total == 120


def test_fps_meter_drops_events_outside_the_window() -> None:
    clock = FakeClock()
    meter = FpsMeter(window_seconds=1.0, clock=clock)

    for _ in range(10):
        meter.record()

    clock.advance(5.0)  # everything ages out
    assert meter.fps() == 0.0
    # The cumulative total is retained for reporting.
    assert meter.total == 10


def test_fps_meter_rejects_a_non_positive_window() -> None:
    with pytest.raises(ValueError):
        FpsMeter(window_seconds=0)


# --------------------------------------------------------------------------- #
# LatencyTracker
# --------------------------------------------------------------------------- #
def test_latency_tracker_reports_percentiles() -> None:
    tracker = LatencyTracker(capacity=100)
    for value in range(1, 101):
        tracker.record(float(value))

    snapshot = tracker.snapshot()
    assert snapshot["count"] == 100
    assert snapshot["p50"] == pytest.approx(50.5, abs=1.0)
    assert snapshot["p99"] == pytest.approx(99.0, abs=1.5)
    assert snapshot["max"] == pytest.approx(100.0)


def test_latency_tracker_is_bounded() -> None:
    """Unbounded samples would be a slow leak in a weeks-long process."""
    tracker = LatencyTracker(capacity=10)
    for value in range(50):
        tracker.record(float(value))

    assert tracker.snapshot()["count"] == 10


def test_latency_tracker_empty_snapshot() -> None:
    assert LatencyTracker().snapshot()["count"] == 0


# --------------------------------------------------------------------------- #
# PipelineStats
# --------------------------------------------------------------------------- #
def test_pipeline_stats_counts_and_ratios() -> None:
    stats = PipelineStats(camera_id="cam0", window_seconds=5.0)
    stats.record_capture()
    stats.record_encode(2.0)
    stats.record_dispatch()
    stats.record_completion(e2e_latency_ms=30.0, queue_ms=5.0, inference_ms=20.0)

    snapshot = stats.snapshot()
    assert snapshot["camera_id"] == "cam0"
    counters = snapshot["counters"]
    assert counters["captured"] == 1
    assert counters["dispatched"] == 1
    assert counters["completed"] == 1
    assert counters["failed"] == 0
    assert snapshot["latency_ms"]["end_to_end"]["p50"] == pytest.approx(30.0)


def test_pipeline_stats_separates_drops_from_failures() -> None:
    """Load shedding and real failures are different operational signals."""
    stats = PipelineStats()
    stats.record_drop_capacity(3)
    stats.record_drop_stale(2)
    stats.record_failure()

    counters = stats.snapshot()["counters"]
    assert counters["dropped_capacity"] == 3
    assert counters["dropped_stale"] == 2
    assert counters["failed"] == 1


def test_completion_ratio_exposes_capacity_shortfall() -> None:
    """Capturing 60 and analysing 30 must not look like "60 FPS achieved"."""
    clock = FakeClock()
    stats = PipelineStats(window_seconds=1.0)
    stats.capture_fps = FpsMeter(window_seconds=1.0, clock=clock)
    stats.completion_fps = FpsMeter(window_seconds=1.0, clock=clock)

    for _ in range(60):
        stats.capture_fps.record()
        clock.advance(1.0 / 60.0)
    for _ in range(30):
        stats.completion_fps.record()

    rates = stats.snapshot()["rates"]
    assert rates["completion_ratio"] < 1.0


def test_meets_target_uses_completion_fps() -> None:
    clock = FakeClock()
    stats = PipelineStats(window_seconds=1.0)
    stats.completion_fps = FpsMeter(window_seconds=1.0, clock=clock)

    assert stats.meets_target(60.0) is False
    for _ in range(60):
        stats.completion_fps.record()
        clock.advance(1.0 / 60.0)
    assert stats.meets_target(60.0) is True
