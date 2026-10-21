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
# date   : 2026-Oct-21
# ==============================================================================
"""The runner's share of the station question: is the part on the table?

The geometry rules live in ``tests/vision/test_presence.py``. What is pinned here is
what the console and the robot can rely on: presence is answered from the boxes
inference already produced, a failed inference is never reported as an empty station,
and a gate that misbehaves cannot stop capture.
"""

from __future__ import annotations

import pytest

from src.vision.detections import Detection
from src.vision.presence import PresenceGate
from tests.pipeline.test_broadcast_runner import build_runner

STATION = (0.15, 0.33, 0.85, 0.85)
GATE = PresenceGate(
    roi=STATION,
    min_area=0.02,
    max_area=0.20,
    min_aspect=1.0,
    max_aspect=2.6,
    min_confidence=0.10,
)


def box(x, y, width, height, confidence=0.2, label="steel blank") -> Detection:
    return Detection(
        object_type=label,
        x_norm=x,
        y_norm=y,
        width_norm=width,
        height_norm=height,
        confidence=confidence,
    )


def table() -> Detection:
    return box(0.188, 0.378, 0.590, 0.421, confidence=0.2459)


def sheet() -> Detection:
    return box(0.3385, 0.454, 0.311, 0.232, confidence=0.1594)


class ResultStub:
    def __init__(self, detections=(), error=None, frame_number=7) -> None:
        self.detections = tuple(detections)
        self.error = error
        self.frame_number = frame_number
        self.camera_id = "cam0"
        self.capture_ts_ns = 1_700_000_000_000_000_000

    def to_payload(self) -> dict:
        return {"metadata": {"frame_number": self.frame_number}, "detections": []}


def build(gate=GATE):
    kwargs = {} if gate is None else {"presence_gate": gate}
    runner, _ = build_runner(**kwargs)
    return runner


def of_kind(runner, kind: str) -> list[dict]:
    """Drain the broadcaster and keep the messages of one kind."""
    subscription = runner.broadcaster.subscribe()
    messages = []
    while (payload := subscription.poll()) is not None:
        if payload.get("type") == kind:
            messages.append(payload)
    return messages


# -- publishing ---------------------------------------------------------------


def test_presence_is_published_as_its_own_message_kind() -> None:
    runner = build()
    subscription = runner.broadcaster.subscribe()

    runner._publish(ResultStub([table(), sheet()]))

    messages = []
    while (payload := subscription.poll()) is not None:
        messages.append(payload)
    kinds = [message.get("type") for message in messages]
    assert "presence" in kinds
    presence = next(message for message in messages if message.get("type") == "presence")
    assert presence["camera_id"] == "cam0"
    assert presence["frame_number"] == 7
    assert presence["present"] is True
    assert presence["accepted"] == 1
    assert presence["offered"] == 2
    assert presence["reason"] is None
    assert presence["detection"]["object_type"] == "steel blank"
    assert presence["detection"]["confidence"] == pytest.approx(0.1594)


def test_a_table_alone_is_reported_as_no_part_with_a_reason() -> None:
    runner = build()
    subscription = runner.broadcaster.subscribe()

    runner._publish(ResultStub([table()]))

    presence = next(
        message
        for message in iter(lambda: subscription.poll(), None)
        if message.get("type") == "presence"
    )
    assert presence["present"] is False
    assert presence["detection"] is None
    assert "a table, not the part" in presence["reason"]


def test_a_failed_inference_is_not_an_empty_station() -> None:
    """A worker timeout must not read as "the part is gone" - that stops a line."""
    runner = build()

    runner._publish(ResultStub([table()], error="worker timed out"))

    assert of_kind(runner, "presence") == []
    assert runner.snapshot()["presence"]["checks"] == 0


def test_without_a_gate_nothing_is_computed_or_published() -> None:
    runner = build(gate=None)

    runner._publish(ResultStub([sheet()]))

    assert of_kind(runner, "presence") == []
    assert runner.snapshot()["presence"] == {"enabled": False}


# -- counters -----------------------------------------------------------------


def test_the_snapshot_reports_what_the_station_has_seen() -> None:
    runner = build()

    runner._publish(ResultStub([table(), sheet()]))
    runner._publish(ResultStub([sheet()]))
    runner._publish(ResultStub([table()]))

    presence = runner.snapshot()["presence"]
    assert presence["enabled"] is True
    assert presence["checks"] == 3
    assert presence["present"] == 2
    assert presence["errors"] == 0
    assert "a table, not the part" in presence["last_reason"]
    assert presence["gate"]["roi"] == list(STATION)
    assert presence["gate"]["max_area"] == pytest.approx(0.20)
    assert presence["gate"]["min_confidence"] == pytest.approx(0.10)


def test_an_accepted_frame_clears_the_previous_reason() -> None:
    runner = build()

    runner._publish(ResultStub([table()]))
    assert runner.snapshot()["presence"]["last_reason"] is not None

    runner._publish(ResultStub([sheet()]))
    assert runner.snapshot()["presence"]["last_reason"] is None


# -- robustness ---------------------------------------------------------------


class BrokenGate:
    """A gate whose geometry already matches but whose check raises."""

    roi = STATION
    min_area = 0.02
    max_area = 0.20
    min_aspect = 1.0
    max_aspect = 2.6
    min_confidence = 0.10
    min_inside = 0.5

    def accepts(self, detection) -> bool:
        raise RuntimeError("gate exploded")

    def rejection(self, detection):
        raise RuntimeError("gate exploded")


def test_a_broken_gate_is_counted_and_never_raised() -> None:
    runner = build(gate=BrokenGate())

    runner._publish(ResultStub([sheet()]))
    runner._publish(ResultStub([sheet()]))

    assert of_kind(runner, "presence") == []
    assert runner.snapshot()["presence"]["errors"] == 2
