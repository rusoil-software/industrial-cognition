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
"""Tests for the runner's share of box tracking.

The tracker itself is tested in ``tests/vision/test_tracking.py``. What matters here
is the contract between the capture loop and the console: tracked boxes are published
as their own message, seeding skips failed frames, and tracking never stops capture.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("cv2")

from src.pipeline.runner import LatestFrame, PipelineRunner  # noqa: E402
from tests.pipeline.test_broadcast_runner import FrameSourceStub, build_runner  # noqa: E402


class TrackerStub:
    """Records what the runner asked of the tracker."""

    enabled = True

    def __init__(self) -> None:
        self.seeds: list[tuple[object, int]] = []
        self.updates: list[object] = []
        self.resets = 0
        self.fail_on_update = False
        self.results: list[dict] = []

    def seed(self, image, detections) -> None:
        self.seeds.append((image, len(list(detections))))

    def update(self, image):
        if self.fail_on_update:
            raise RuntimeError("tracker exploded")
        self.updates.append(image)
        return self.results

    def snapshot(self) -> dict:
        return {"enabled": True, "tracker": "stub", "tracks": len(self.results)}

    def reset(self) -> None:
        self.resets += 1


class ResultStub:
    def __init__(self, detections=(), error=None, frame_number=7) -> None:
        self.detections = tuple(detections)
        self.error = error
        self.frame_number = frame_number

    def to_payload(self) -> dict:
        return {"metadata": {"frame_number": self.frame_number}, "detections": []}


def make_frame(frame_number: int = 7) -> LatestFrame:
    return LatestFrame(
        jpeg=b"\xff\xd8not-a-real-jpeg\xff\xd9",
        frame_number=frame_number,
        capture_ts_ns=time.time_ns(),
        width=64,
        height=48,
    )


def build(**overrides):
    tracker = overrides.pop("tracker", None)
    runner, _ = build_runner(tracker=tracker, **overrides)
    runner._latest = make_frame()
    return runner, tracker


def test_tracking_publishes_its_own_message_kind() -> None:
    tracker = TrackerStub()
    tracker.results = [{"track_id": 1, "object_type": "robot", "x_norm": 0.1}]
    runner, _ = build(tracker=tracker, track_fps=60.0)
    subscription = runner.broadcaster.subscribe()
    frame = make_frame(frame_number=11)

    runner._advance_tracks(frame, "cam0")

    payload = subscription.poll()
    assert payload is not None
    assert payload["type"] == "tracks"
    assert payload["camera_id"] == "cam0"
    assert payload["frame_number"] == 11
    assert payload["tracks"] == tracker.results


def test_an_empty_update_is_not_published_until_there_were_tracks() -> None:
    """Noise control: "no tracks" ten times a second teaches the console to ignore it."""
    tracker = TrackerStub()
    runner, _ = build(tracker=tracker, track_fps=60.0)
    subscription = runner.broadcaster.subscribe()

    runner._advance_tracks(make_frame(), "cam0")
    assert subscription.poll() is None

    tracker.results = [{"track_id": 1}]
    runner._track_next = 0.0
    runner._advance_tracks(make_frame(), "cam0")
    assert subscription.poll() is not None

    # Once boxes were visible, an empty update is the message that clears them.
    tracker.results = []
    runner._track_next = 0.0
    runner._advance_tracks(make_frame(), "cam0")
    payload = subscription.poll()
    assert payload is not None
    assert payload["tracks"] == []


def test_tracks_are_advanced_at_most_track_fps_times_a_second() -> None:
    tracker = TrackerStub()
    runner, _ = build(tracker=tracker, track_fps=2.0)

    for _ in range(5):
        runner._advance_tracks(make_frame(), "cam0")

    # 2 updates/s against a loop that ran five times instantly: exactly one update fits.
    assert len(tracker.updates) == 1


def test_seeding_uses_the_newest_frame_and_its_detections() -> None:
    tracker = TrackerStub()
    runner, _ = build(tracker=tracker)
    runner._latest = make_frame(frame_number=21)

    runner._publish(ResultStub(detections=[object(), object()]))

    assert tracker.seeds == [(runner._latest.jpeg, 2)]


def test_a_failed_inference_does_not_move_the_boxes() -> None:
    tracker = TrackerStub()
    runner, _ = build(tracker=tracker)

    runner._publish(ResultStub(detections=[object()], error="worker timed out"))

    assert tracker.seeds == []
    assert runner.snapshot()["track_errors"] == 0


def test_a_tracker_that_raises_does_not_stop_the_loop() -> None:
    tracker = TrackerStub()
    tracker.fail_on_update = True
    runner, _ = build(tracker=tracker, track_fps=60.0)

    runner._advance_tracks(make_frame(), "cam0")
    runner._track_next = 0.0
    runner._advance_tracks(make_frame(), "cam0")

    snapshot = runner.snapshot()
    assert snapshot["track_errors"] == 2
    assert snapshot["running"] is False  # never started; the point is that nothing raised


def test_without_a_tracker_nothing_is_published() -> None:
    runner, _ = build(tracker=None)
    subscription = runner.broadcaster.subscribe()

    runner._advance_tracks(make_frame(), "cam0")
    runner._publish(ResultStub(detections=[object()]))

    payload = subscription.poll()
    assert payload is None or payload.get("type") != "tracks"
    assert runner.snapshot()["tracking"] == {"enabled": False}


def test_snapshot_reports_the_frame_and_a_cadence_aware_staleness_budget() -> None:
    runner, _ = build()
    snapshot = runner.snapshot()

    latest = snapshot["latest_frame"]
    assert latest["frame_number"] == 7
    assert latest["age_seconds"] >= 0.0
    # build_runner asks for 90 FPS, so three missed intervals is below the five-second
    # floor: a fast camera goes stale in five seconds, not in 33 milliseconds.
    assert latest["stale_after_seconds"] == 5.0


def test_a_slow_cadence_gets_a_larger_staleness_budget() -> None:
    """A 0.2 FPS dev replay is healthy at four seconds of age and must not read stale."""
    runner, _ = build(target_fps=0.2)

    assert runner.snapshot()["latest_frame"]["stale_after_seconds"] == 15.0


def test_stopping_resets_the_trackers() -> None:
    tracker = TrackerStub()
    runner, _ = build(tracker=tracker)
    runner.start()
    time.sleep(0.05)
    runner.stop()

    assert tracker.resets == 1


def test_a_seed_after_capture_still_finds_a_frame_when_none_arrived() -> None:
    """A pipeline that never captured must not crash seeding."""
    tracker = TrackerStub()
    runner, _ = build(tracker=tracker)
    runner._latest = None

    runner._publish(ResultStub(detections=[object()]))

    assert tracker.seeds == []


def test_tracking_can_be_disabled_without_touching_capture() -> None:
    runner, _ = build(tracker=None)
    assert runner.snapshot()["tracking"]["enabled"] is False
    assert runner.start() is True
    try:
        time.sleep(0.2)
    finally:
        runner.stop()
    assert runner.snapshot()["pipeline"]["counters"]["captured"] > 0
