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
# date   : 2026-Oct-20
# ==============================================================================
"""The tracker pool: identity across re-seeds, expiry, and payload geometry.

The pool is exercised with a fake tracker on purpose. A real CSRT/KCF tracker is a
statistical object: it may or may not follow a synthetic square, so asserting its
output would test OpenCV, not this module. The fake tracker returns a known box,
which lets the tests pin the behaviour this module owns - matching, identity, expiry,
normalisation - and one optional test exercises the real backend end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.vision.detections import Detection
from src.vision.tracking import (
    DEFAULT_TRACKER_TYPE,
    TRACKER_TYPE_ENV,
    TrackPool,
    available_trackers,
    resolve_tracker,
)

FRAME_HEIGHT, FRAME_WIDTH = 480, 640


def frame() -> np.ndarray:
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


class FakeTracker:
    """A tracker that reports the box it was initialised with, moved by ``offset``."""

    def __init__(self, offset=(4, 2), ok=True) -> None:
        self.offset = offset
        self.ok = ok
        self.box = (0, 0, 10, 10)
        self.inits = 0
        self.updates = 0
        self.frames = []

    def init(self, image, box):
        self.inits += 1
        self.box = tuple(box)
        return True

    def update(self, image):
        self.updates += 1
        if not self.ok:
            return False, (0, 0, 0, 0)
        x, y, width, height = self.box
        self.box = (x + self.offset[0], y + self.offset[1], width, height)
        return True, self.box


def pool_with(trackers=None, **kwargs) -> TrackPool:
    created = list(trackers or [])
    factory = lambda: created.pop(0) if created else FakeTracker()  # noqa: E731
    return TrackPool("kcf", factory=factory, **kwargs)


def detection(label="person", *, x=0.1, y=0.2, w=0.25, h=0.4, confidence=0.8):
    return Detection(
        object_type=label,
        x_norm=x,
        y_norm=y,
        width_norm=w,
        height_norm=h,
        confidence=confidence,
    )


def test_tracker_selection_falls_back_when_the_requested_one_is_missing(monkeypatch):
    from src.vision import tracking

    monkeypatch.setattr(tracking, "available_trackers", lambda: ("mil",))
    monkeypatch.setattr(
        tracking, "_constructor", lambda name: (lambda: "mil-tracker") if name == "mil" else None
    )

    factory, name, reason = resolve_tracker("csrt")

    assert name == "mil"
    assert factory() == "mil-tracker"
    assert "csrt" in reason and "not available" in reason


def test_tracker_selection_explains_a_build_without_trackers(monkeypatch):
    from src.vision import tracking

    monkeypatch.setattr(tracking, "available_trackers", lambda: ())

    factory, name, reason = resolve_tracker(DEFAULT_TRACKER_TYPE)

    assert factory is None and name is None
    assert "opencv-contrib-python-headless" in reason


def test_tracker_selection_accepts_the_default_and_named_trackers():
    available = available_trackers()

    if not available:  # pragma: no cover - depends on the installed OpenCV
        pytest.skip("this OpenCV build has no trackers")

    requested = available[0]
    factory, name, reason = resolve_tracker(requested)

    assert name == requested
    assert factory is not None
    assert reason == ""


def test_a_pool_without_a_tracker_is_a_no_op(monkeypatch):
    from src.vision import tracking

    monkeypatch.setattr(tracking, "available_trackers", lambda: ())
    pool = TrackPool()

    pool.seed(frame(), [detection()])
    assert pool.update(frame()) == []
    assert pool.enabled is False
    assert pool.snapshot()["tracks"] == 0
    assert "opencv-contrib-python-headless" in pool.snapshot()["reason"]


def test_seeding_starts_one_tracker_per_detection():
    pool = pool_with()

    pool.seed(frame(), [detection(), detection("robot", x=0.6, y=0.1)])

    assert len(pool._tracks) == 2
    assert [track.label for track in pool._tracks] == ["person", "robot"]
    assert pool._tracks[0].tracker.inits == 1
    assert pool.snapshot()["seeded"] == 1


def test_a_seed_boxes_are_normalised_for_the_console():
    pool = pool_with()
    pool.seed(frame(), [detection(x=0.1, y=0.2, w=0.25, h=0.4, confidence=0.81234)])

    tracks = pool.update(frame())

    assert len(tracks) == 1
    track = tracks[0]
    x, y, width, height = pool._tracks[0].box
    assert track["track_id"] == 1
    assert track["object_type"] == "person"
    assert track["x_norm"] == pytest.approx(x / FRAME_WIDTH)
    assert track["y_norm"] == pytest.approx(y / FRAME_HEIGHT)
    assert track["width_norm"] == pytest.approx(width / FRAME_WIDTH)
    assert track["height_norm"] == pytest.approx(height / FRAME_HEIGHT)
    assert track["confidence"] == pytest.approx(0.8123)
    assert track["stale"] is False


def test_updating_moves_the_box_with_the_tracker():
    tracker = FakeTracker(offset=(10, 5))
    pool = pool_with([tracker])
    pool.seed(frame(), [detection(x=0.1, y=0.1, w=0.2, h=0.2)])
    start = pool._tracks[0].box

    first = pool.update(frame())[0]
    second = pool.update(frame())[0]

    assert tracker.updates == 2
    assert (first["x_norm"], first["y_norm"]) == pytest.approx(
        ((start[0] + 10) / FRAME_WIDTH, (start[1] + 5) / FRAME_HEIGHT)
    )
    assert second["x_norm"] > first["x_norm"]
    assert pool.snapshot()["updated"] == 2


def test_reseeding_the_same_object_keeps_its_identity_and_tracker():
    pool = pool_with()
    pool.seed(frame(), [detection()])
    original = pool._tracks[0]
    tracker = original.tracker

    pool.seed(frame(), [detection(confidence=0.95)])

    assert len(pool._tracks) == 1
    assert pool._tracks[0] is original
    assert pool._tracks[0].tracker is tracker
    assert pool._tracks[0].confidence == pytest.approx(0.95)
    assert tracker.inits == 1  # not re-initialised


def test_a_distant_detection_starts_a_new_track():
    pool = pool_with()
    pool.seed(frame(), [detection(x=0.05, y=0.05, w=0.1, h=0.1)])

    pool.seed(frame(), [detection(x=0.7, y=0.7, w=0.1, h=0.1)])

    assert len(pool._tracks) == 2
    assert sorted(track.track_id for track in pool._tracks) == [1, 2]


def test_a_label_change_does_not_reuse_a_track():
    pool = pool_with()
    pool.seed(frame(), [detection("person")])

    pool.seed(frame(), [detection("robot")])

    assert sorted(track.label for track in pool._tracks) == ["person", "robot"]


def test_a_track_survives_a_brief_miss_and_dies_after_max_missing():
    pool = pool_with(max_missing=2)
    pool.seed(frame(), [detection()])

    for _ in range(2):
        pool.seed(frame(), [])
        assert len(pool._tracks) == 1, "the box should not blink off the screen"

    pool.seed(frame(), [])
    assert pool._tracks == []
    assert pool.snapshot()["lost"] == 1


def test_a_failing_tracker_is_reported_stale_then_dropped():
    pool = pool_with([FakeTracker(ok=False)], max_missing=2)
    pool.seed(frame(), [detection()])

    assert pool.update(frame())[0]["stale"] is True
    assert pool.update(frame())[0]["stale"] is True
    assert pool.update(frame()) == []
    assert pool.snapshot()["lost"] == 1


def test_a_new_seed_revives_a_stale_track():
    pool = pool_with([FakeTracker(ok=False)], max_missing=5)
    pool.seed(frame(), [detection()])
    pool.update(frame())
    assert pool._tracks[0].missed == 1

    pool.seed(frame(), [detection()])

    assert pool._tracks[0].missed == 0


def test_tiny_detections_are_not_tracked():
    pool = pool_with(min_box_pixels=16)

    pool.seed(frame(), [detection(w=0.001, h=0.001)])

    assert pool._tracks == []


def test_the_track_count_is_capped_by_confidence():
    pool = pool_with(max_tracks=2)
    detections = [
        detection("a", x=0.05, y=0.05, w=0.1, h=0.1, confidence=0.5),
        detection("b", x=0.30, y=0.05, w=0.1, h=0.1, confidence=0.9),
        detection("c", x=0.55, y=0.05, w=0.1, h=0.1, confidence=0.7),
    ]

    pool.seed(frame(), detections)

    assert len(pool._tracks) == 2
    assert sorted(track.label for track in pool._tracks) == ["b", "c"]


def test_a_jpeg_payload_and_a_captured_frame_object_are_accepted():
    import cv2

    ok, buffer = cv2.imencode(".jpg", frame())
    assert ok
    pool = pool_with()

    pool.seed(buffer.tobytes(), [detection()])
    assert len(pool._tracks) == 1

    class _Frame:
        jpeg = buffer.tobytes()

    pool.reset()
    pool.seed(_Frame(), [detection()])
    assert len(pool._tracks) == 1


def test_reset_forgets_every_track_and_restarts_ids():
    pool = pool_with()
    pool.seed(frame(), [detection(), detection("robot", x=0.6)])
    assert len(pool._tracks) == 2

    pool.reset()

    assert pool._tracks == []
    pool.seed(frame(), [detection()])
    assert pool._tracks[0].track_id == 1


def test_an_unknown_tracker_name_is_reported_not_fatal():
    pool = TrackPool("nonsense", factory=None)

    snapshot = pool.snapshot()

    assert snapshot["requested"] == "nonsense"
    # The pool either fell back to an available tracker or explains why it cannot.
    assert snapshot["enabled"] is (snapshot["tracker"] is not None)
    if snapshot["enabled"]:
        assert "nonsense" in snapshot["reason"]
    else:
        assert "opencv-contrib" in snapshot["reason"]


def test_the_environment_variable_name_is_stable():
    assert TRACKER_TYPE_ENV == "TRACKER_TYPE"


def test_an_object_that_moved_further_than_its_width_keeps_one_track():
    # The report: an object that moved faster than inference could follow appeared twice
    # once it stood still - the old box expiring next to a fresh one. The fresh detection
    # no longer overlaps the previous box, but its centre is inside the distance gate.
    pool = pool_with()
    pool.seed(frame(), [detection(x=0.10, y=0.20, w=0.10, h=0.10)])
    first = pool._tracks[0]

    pool.seed(frame(), [detection(x=0.34, y=0.20, w=0.10, h=0.10)])

    assert len(pool._tracks) == 1
    assert pool._tracks[0] is first
    assert pool._tracks[0].tracker.inits == 1


def test_the_distance_gate_can_be_switched_off():
    pool = pool_with(match_distance=0.0)
    pool.seed(frame(), [detection(x=0.10, y=0.20, w=0.10, h=0.10)])

    pool.seed(frame(), [detection(x=0.34, y=0.20, w=0.10, h=0.10)])

    assert len(pool._tracks) == 2


def test_two_tracks_that_end_up_on_one_object_are_collapsed():
    # Distance matching stops new duplicates; this is the clean-up for a pair that
    # already exists (two seeds, or two trackers converging on the same object).
    pool = pool_with()
    pool.seed(
        frame(),
        [
            detection(x=0.10, y=0.20, w=0.10, h=0.10, confidence=0.9),
            detection(x=0.70, y=0.70, w=0.10, h=0.10, confidence=0.8),
        ],
    )
    older, younger = sorted(pool._tracks, key=lambda track: track.track_id)
    # Both the reported box and the tracker's own box, because the fake tracker reports
    # what it was initialised with, moved by its offset.
    for track, box in ((older, (100.0, 100.0, 80.0, 60.0)), (younger, (106.0, 103.0, 80.0, 60.0))):
        track.box = box
        track.tracker.box = box

    tracks = pool.update(frame())

    assert len(tracks) == 1
    assert tracks[0]["track_id"] == older.track_id
    assert pool.snapshot()["merged"] == 1
    assert [track.track_id for track in pool._tracks] == [older.track_id]


def test_different_labels_on_one_box_are_left_alone():
    pool = pool_with()
    pool.seed(
        frame(),
        [
            detection("person", x=0.10, y=0.20, w=0.10, h=0.10),
            detection("robot", x=0.10, y=0.20, w=0.10, h=0.10),
        ],
    )

    tracks = pool.update(frame())

    assert len(tracks) == 2
    assert {track["object_type"] for track in tracks} == {"person", "robot"}
    assert pool.snapshot()["merged"] == 0



@pytest.mark.skipif("csrt" not in available_trackers(), reason="needs opencv-contrib trackers")
def test_a_real_tracker_follows_a_moving_square():
    """A last, honest check that the wiring works with OpenCV's own tracker."""
    import cv2

    image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    pool = TrackPool("csrt", max_missing=10)
    box = (100, 100, 120, 80)
    cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 255, 255), -1)
    pool.seed(image, [detection(x=box[0] / FRAME_WIDTH, y=box[1] / FRAME_HEIGHT,
                              w=box[2] / FRAME_WIDTH, h=box[3] / FRAME_HEIGHT)])

    moved = np.zeros_like(image)
    cv2.rectangle(moved, (130, 110), (250, 190), (255, 255, 255), -1)
    tracks = pool.update(moved)

    assert len(tracks) == 1
    assert tracks[0]["stale"] is False
    # The box should have moved toward the square, not stayed on its seed position.
    assert tracks[0]["x_norm"] > box[0] / FRAME_WIDTH
