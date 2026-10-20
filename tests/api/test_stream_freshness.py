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
"""Tests for preview freshness.

An MJPEG stream is a sequence of stills and a browser keeps the last one on screen when
the stream stops. That makes "is the camera still there?" a property of the API, not of
the image: the console needs a frame age, and a dead stream must end rather than freeze.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from src.api import runtime, stream  # noqa: E402
from src.api.prompts import service as prompt_service  # noqa: E402
from src.api.storage import camera_repository  # noqa: E402


class FrameStub:
    def __init__(self, frame_number: int = 3) -> None:
        self.frame_number = frame_number
        self.jpeg = b"\xff\xd8frame\xff\xd9"


class StatsStub:
    camera_id = "cam0"

    def snapshot(self) -> dict:
        return {"rates": {"capture_fps": 1.5}}


class RunnerStub:
    def __init__(self, latest: FrameStub | None, age: float | None = None) -> None:
        self.stats = StatsStub()
        self._latest = latest
        self._age = age

    def latest_frame(self):
        return self._latest

    def snapshot(self) -> dict:
        if self._latest is None:
            return {"latest_frame": None}
        return {
            "latest_frame": {
                "frame_number": self._latest.frame_number,
                "age_seconds": self._age,
                "stale_after_seconds": 5.0,
            }
        }


@pytest.fixture
def no_registry(monkeypatch):
    monkeypatch.setattr(prompt_service, "list_camera_prompts", lambda session: [])
    monkeypatch.setattr(camera_repository, "list_cameras", lambda session: [])


def test_a_running_camera_reports_the_age_of_its_newest_frame(monkeypatch, no_registry) -> None:
    monkeypatch.setattr(runtime, "get_runner", lambda: RunnerStub(FrameStub(), age=3.2))

    body = stream.list_cameras(session=None)

    entry = body["cameras"][0]
    assert entry["camera_id"] == "cam0"
    assert entry["running"] is True
    assert entry["last_frame_age_seconds"] == 3.2
    assert entry["streaming"] is True


def test_a_camera_without_a_frame_is_not_streaming(monkeypatch, no_registry) -> None:
    monkeypatch.setattr(runtime, "get_runner", lambda: RunnerStub(None))

    entry = stream.list_cameras(session=None)["cameras"][0]

    # None, not zero: zero would read as "a fresh frame" in the console.
    assert entry["last_frame_age_seconds"] is None
    assert entry["streaming"] is False


async def test_the_preview_closes_when_capture_stops(monkeypatch) -> None:
    """The one behaviour that keeps a frozen image from looking live."""
    monkeypatch.setattr(runtime, "get_runner", lambda: RunnerStub(FrameStub(), age=99.0))
    monkeypatch.setattr(stream, "STREAM_IDLE_TIMEOUT", 0.0)
    monkeypatch.setattr(stream, "STREAM_FPS", 50)

    response = await stream.stream("cam0")
    frames = response.body_iterator

    first = await anext(frames)
    assert b"X-Frame-Number: 3" in first
    assert b"\xff\xd8frame\xff\xd9" in first

    # Same frame number, timeout already exceeded: the response ends instead of repeating.
    with pytest.raises(StopAsyncIteration):
        await anext(frames)


async def test_the_preview_repeats_nothing_until_a_new_frame_arrives(monkeypatch) -> None:
    runner = RunnerStub(FrameStub(frame_number=1), age=0.1)
    monkeypatch.setattr(runtime, "get_runner", lambda: runner)
    monkeypatch.setattr(stream, "STREAM_IDLE_TIMEOUT", 30.0)
    monkeypatch.setattr(stream, "STREAM_FPS", 50)

    response = await stream.stream("cam0")
    frames = response.body_iterator
    assert b"X-Frame-Number: 1" in await anext(frames)

    runner._latest = FrameStub(frame_number=2)
    assert b"X-Frame-Number: 2" in await anext(frames)


def test_stream_fps_is_bounded() -> None:
    """A typo in the config must not become a bandwidth incident."""
    assert 1 <= stream.STREAM_FPS <= 30
    assert stream.STREAM_IDLE_TIMEOUT >= 2.0
