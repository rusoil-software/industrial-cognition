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
"""RTSP transport selection.

OpenCV has no API for it, so the choice travels through an environment variable
that its FFmpeg backend reads when a capture is opened. These tests pin the
ordering (it must be set *before* ``VideoCapture``), the merge with operator
options, and the restoration of the process environment afterwards.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest

from src.camera import capture as capture_module
from src.camera.capture import (
    FFMPEG_CAPTURE_OPTIONS_ENV,
    CaptureConfig,
    DeviceFrameSource,
    is_rtsp_source,
    rtsp_transport,
)

OPTIONS = FFMPEG_CAPTURE_OPTIONS_ENV


class FakeCapture:
    """Records the FFmpeg options in force when OpenCV would have opened."""

    opened: list["FakeCapture"] = []

    def __init__(self, source, *_args, **_kwargs) -> None:
        self.source = source
        self.options_seen = os.environ.get(OPTIONS)
        FakeCapture.opened.append(self)

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV's name
        return True

    def set(self, *_args) -> bool:
        return True

    def get(self, _prop) -> float:
        return 0.0

    def read(self):
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        return None


@pytest.fixture
def fake_capture(monkeypatch):
    FakeCapture.opened = []
    monkeypatch.setattr(capture_module.cv2, "VideoCapture", FakeCapture)
    monkeypatch.delenv(OPTIONS, raising=False)
    monkeypatch.delenv("RTSP_TRANSPORT", raising=False)
    return FakeCapture


def open_source(source) -> FakeCapture:
    frame_source = DeviceFrameSource(CaptureConfig(source=source))
    assert frame_source.open() is True
    assert FakeCapture.opened, "VideoCapture was never constructed"
    return FakeCapture.opened[-1]


@pytest.mark.parametrize(
    "source",
    ["rtsp://10.0.0.5:554/stream1", "RTSP://10.0.0.5:554/stream1", "rtsps://10.0.0.5:322/s"],
)
def test_an_rtsp_url_negotiates_tcp_by_default(fake_capture, source):
    # UDP RTP cannot cross a NAT boundary: the camera sends to an address it cannot
    # route back to, and every frame is lost. TCP interleaved goes down the control
    # connection instead, which is why it is the default.
    assert open_source(source).options_seen == "rtsp_transport;tcp"


def test_udp_is_honoured_when_configured(fake_capture, monkeypatch):
    monkeypatch.setenv("RTSP_TRANSPORT", "UDP")

    assert open_source("rtsp://10.0.0.5:554/stream1").options_seen == "rtsp_transport;udp"


def test_an_unknown_transport_falls_back_to_tcp(fake_capture, monkeypatch, caplog):
    monkeypatch.setenv("RTSP_TRANSPORT", "sctp")

    with caplog.at_level(logging.WARNING):
        opened = open_source("rtsp://10.0.0.5:554/stream1")

    assert opened.options_seen == "rtsp_transport;tcp"
    assert "sctp" in caplog.text


def test_operator_options_are_kept_and_the_transport_is_added(fake_capture, monkeypatch):
    monkeypatch.setenv(OPTIONS, "timeout;5000;stimeout;5000000")

    opened = open_source("rtsp://10.0.0.5:554/stream1")

    assert opened.options_seen == "timeout;5000;stimeout;5000000;rtsp_transport;tcp"


def test_an_existing_transport_option_is_replaced_not_duplicated(fake_capture, monkeypatch):
    monkeypatch.setenv(OPTIONS, "rtsp_transport;udp")

    opened = open_source("rtsp://10.0.0.5:554/stream1")

    assert opened.options_seen == "rtsp_transport;tcp"


def test_the_process_environment_is_restored(fake_capture):
    open_source("rtsp://10.0.0.5:554/stream1")

    assert OPTIONS not in os.environ
    assert "RTSP_TRANSPORT" not in os.environ


def test_a_previous_value_is_restored_exactly(fake_capture, monkeypatch):
    monkeypatch.setenv(OPTIONS, "timeout;5000")

    open_source("rtsp://10.0.0.5:554/stream1")

    assert os.environ[OPTIONS] == "timeout;5000"


def test_a_device_source_is_left_alone(fake_capture):
    # A webcam is not FFmpeg's business; the transport must not leak into it.
    opened = open_source(0)

    assert opened.options_seen is None
    assert OPTIONS not in os.environ


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("rtsp://10.0.0.5/s", True),
        ("RTSP://10.0.0.5/s", True),
        (" rtsps://10.0.0.5/s ", True),
        ("http://10.0.0.5/mjpeg", False),
        ("/dev/video0", False),
        (0, False),
        ("", False),
        (None, False),
    ],
)
def test_only_rtsp_urls_are_recognised(source, expected):
    assert is_rtsp_source(source) is expected


def test_the_transport_reader_ignores_surrounding_space(monkeypatch):
    monkeypatch.setenv("RTSP_TRANSPORT", "  udp  ")

    assert rtsp_transport() == "udp"
