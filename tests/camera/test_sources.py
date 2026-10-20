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
"""Which frame source a configured string turns into - without opening anything."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.camera.capture import CaptureConfig, DeviceFrameSource
from src.camera.sources import (
    ReplayFrameSource,
    SyntheticFrameSource,
    build_frame_source,
)


def config(source=0) -> CaptureConfig:
    return CaptureConfig(camera_id="cam9", source=source)


@pytest.mark.parametrize("prefix", ["replay:", "image:", "still:"])
def test_a_still_image_source_is_a_replay_source(prefix, tmp_path):
    # A replay source keeps the path, so the file need not exist yet to classify it.
    source = build_frame_source(f"{prefix}{tmp_path / 'frame.jpg'}", config())

    assert isinstance(source, ReplayFrameSource)


@pytest.mark.parametrize("name", ["synthetic", "fake", "test", "SYNTHETIC"])
def test_the_synthetic_names_are_case_insensitive(name):
    assert isinstance(build_frame_source(name, config()), SyntheticFrameSource)


def test_a_device_number_as_text_becomes_a_device_index():
    source = build_frame_source("2", config())

    assert isinstance(source, DeviceFrameSource)
    assert source._config.source == 2


def test_an_integer_source_is_a_device_index():
    assert build_frame_source(1, config())._config.source == 1


@pytest.mark.parametrize(
    "source",
    [
        "rtsp://10.0.0.5:554/stream1",
        "rtsps://10.0.0.5:322/stream1",
        "http://10.0.0.5/mjpeg",
        "/dev/video0",
    ],
)
def test_a_url_or_device_path_keeps_the_source_it_was_given(source):
    # Regression: a URL used to fall through to DeviceFrameSource(config), which opened
    # the *device index* from the config and silently ignored the camera address.
    built = build_frame_source(source, config())

    assert isinstance(built, DeviceFrameSource)
    assert built._config.source == source


def test_a_url_keeps_the_rest_of_the_capture_settings():
    built = build_frame_source("rtsp://10.0.0.5:554/stream1", config())

    assert built._config.camera_id == "cam9"
    assert built._config.width == CaptureConfig().width


def test_surrounding_whitespace_is_trimmed_from_a_url():
    built = build_frame_source("  rtsp://10.0.0.5:554/stream1  ", config())

    assert built._config.source == "rtsp://10.0.0.5:554/stream1"


@pytest.mark.parametrize("source", ["", "   "])
def test_an_empty_source_is_a_configuration_error(source):
    with pytest.raises(ValueError, match="empty"):
        build_frame_source(source, config())


def _write_still(path):
    # A white square on black is all the pan needs: the centroid of the bright pixels is
    # the offset of the frame, so sub-pixel accuracy is not required.
    image = np.zeros((60, 80, 3), np.uint8)
    image[20:40, 30:50] = 255
    assert cv2.imwrite(str(path), image)
    return path


def _centroid(frame):
    columns = np.nonzero(frame[:, :, 0] > 128)[1]
    assert columns.size, "the still should still be in frame"
    return float(columns.mean())


def test_a_replay_source_pans_so_the_console_shows_movement(tmp_path):
    # The user's complaint was literal: one frame per 10 s, so movement did not exist.
    # A still image published as a camera now pans across its amplitude once per period.
    source = ReplayFrameSource(_write_still(tmp_path / "still.png"), drift_pixels=12)
    assert source.open()

    frames = [source.read() for _ in range(ReplayFrameSource.PAN_PERIOD_FRAMES)]

    assert not np.array_equal(frames[0], frames[1])
    # The pan reaches its extremes a quarter of a period apart, 12 px each way.
    assert abs(_centroid(frames[ReplayFrameSource.PAN_PERIOD_FRAMES // 4]) - _centroid(frames[0])) >= 11.0


def test_a_replay_source_pans_visibly_by_default(tmp_path):
    # The default amplitude has to be big enough to see in a 640-960 px preview; 2 px was
    # not, which is part of why the console looked frozen.
    source = ReplayFrameSource(_write_still(tmp_path / "still.png"))
    assert source.open()

    frames = [source.read() for _ in range(ReplayFrameSource.PAN_PERIOD_FRAMES)]

    assert max(abs(_centroid(frame) - _centroid(frames[0])) for frame in frames) >= 20.0


def test_a_replay_source_without_drift_repeats_the_still(tmp_path):
    source = ReplayFrameSource(_write_still(tmp_path / "still.png"), drift_pixels=0)
    assert source.open()

    assert np.array_equal(source.read(), source.read())

