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
"""Tests for the detection persistence policy.

The writer is a counter and the clock is fake, so these assert the *policy* - which frames
are worth a row - without a database. That is the part that decides how fast the disk fills;
the repository's SQL is covered separately where a database is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.api.detection_sink import DetectionSink


@dataclass
class FakeDetection:
    object_type: str = "robot"
    confidence: float = 0.9
    x_norm: float = 0.1
    y_norm: float = 0.2
    width_norm: float = 0.3
    height_norm: float = 0.1


@dataclass
class FakeResult:
    frame_number: int = 1
    camera_id: str = "main"
    detections: tuple = field(default_factory=lambda: (FakeDetection(),))
    error: str | None = None


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_sink(*, interval: float = 1.0, only: bool = True):
    written: list[FakeResult] = []

    def writer(result: FakeResult) -> int:
        written.append(result)
        return len(result.detections)

    clock = Clock()
    sink = DetectionSink(
        writer=writer,
        min_interval_seconds=interval,
        only_with_detections=only,
        clock=clock,
    )
    return sink, written, clock


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
def test_a_frame_with_detections_is_persisted() -> None:
    sink, written, _ = make_sink()

    assert sink( FakeResult() ) is True
    assert len(written) == 1
    assert sink.rows_written == 1


def test_a_failed_frame_is_never_persisted() -> None:
    sink, written, _ = make_sink()

    assert sink(FakeResult(error="worker: boom")) is False
    assert written == []
    assert sink.frames_skipped == 1


def test_empty_frames_are_skipped_by_default() -> None:
    sink, written, _ = make_sink()
    assert sink(FakeResult(detections=())) is False
    assert written == []


def test_empty_frames_can_be_kept_for_presence_history() -> None:
    sink, written, _ = make_sink(only=False)
    assert sink(FakeResult(detections=())) is True
    assert len(written) == 1


def test_only_one_frame_per_interval_per_camera() -> None:
    """Capture may run at 60 FPS; the write rate must not follow it."""
    sink, written, clock = make_sink(interval=1.0)

    assert sink(FakeResult(frame_number=1)) is True
    for number in range(2, 30):
        assert sink(FakeResult(frame_number=number)) is False

    assert len(written) == 1
    assert sink.frames_skipped == 28

    clock.advance(1.0)
    assert sink(FakeResult(frame_number=30)) is True
    assert len(written) == 2


def test_the_interval_is_per_camera() -> None:
    """Two cameras must not silence each other."""
    sink, written, _ = make_sink(interval=1.0)

    assert sink(FakeResult(camera_id="cam0")) is True
    assert sink(FakeResult(camera_id="cam1")) is True
    assert sink(FakeResult(camera_id="cam0")) is False
    assert len(written) == 2


def test_a_zero_interval_persists_every_frame() -> None:
    sink, written, _ = make_sink(interval=0.0)
    for number in range(5):
        assert sink(FakeResult(frame_number=number)) is True
    assert len(written) == 5


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #
def test_a_writer_failure_is_counted_and_does_not_raise() -> None:
    """This runs on the capture thread: a database outage must not stop detection."""

    def broken(result) -> int:
        raise RuntimeError("database is down")

    sink = DetectionSink(writer=broken, min_interval_seconds=0.0, clock=Clock())

    assert sink(FakeResult()) is False
    assert sink.errors == 1
    # The interval is not advanced on failure, so the next frame retries immediately.
    assert sink(FakeResult()) is False
    assert sink.errors == 2


def test_row_count_is_accumulated() -> None:
    sink, _written, clock = make_sink(interval=0.0)
    multiple = FakeResult(detections=(FakeDetection(), FakeDetection(), FakeDetection()))

    assert sink(multiple) is True
    clock.advance(1.0)
    assert sink(multiple) is True
    assert sink.rows_written == 6


def test_snapshot_reports_the_configuration_and_counters() -> None:
    sink, _written, _ = make_sink(interval=2.5)
    sink(FakeResult())

    snapshot = sink.snapshot()
    assert snapshot["frames_written"] == 1
    assert snapshot["min_interval_seconds"] == 2.5
    assert snapshot["only_with_detections"] is True
