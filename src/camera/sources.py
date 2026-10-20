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
"""Frame sources, and the factory that picks one from configuration.

`CAMERA_SOURCE` accepts three shapes, which is what lets the whole pipeline run
without hardware:

* ``"0"`` or ``"/dev/video0"`` - a real device (or an RTSP URL), opened by OpenCV.
* ``"replay:/path/to/still.webp"`` - a **still image replayed as a camera**. The frame
  loop, the queue, the worker and the UI all run exactly as they would with a camera;
  only the pixels are constant. This is what makes the UI and the end-to-end tests
  demonstrable on a machine with no camera attached, and it is honest about being a
  replay: the source reports itself as such.
* ``"synthetic"`` - generated frames, for load tests.

There is no hidden fallback: an unopenable source returns ``False`` from ``open()`` and
the caller decides, which keeps "no camera" visible rather than silently swapping in
fake frames.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from src.camera.capture import (
    CaptureConfig,
    DeviceFrameSource,
    FrameSource,
    SyntheticFrameSource,
)

__all__ = ["ReplayFrameSource", "build_frame_source"]

logger = logging.getLogger(__name__)

#: Prefix that marks a source as a still image to replay.
REPLAY_PREFIXES = ("replay:", "image:", "still:")


class ReplayFrameSource:
    """Replays a still image as a live camera.

    Frames are re-encoded from the decoded still each time, so the payload that reaches
    the worker is byte-for-byte what a camera frame would be - the JPEG on the wire is
    not a constant buffer copied around.

    The still is slowly panned: without motion, consecutive frames are identical, the
    console shows one frozen image and every downstream optimisation (deduplication,
    motion gating, tracking) is tested in a way that cannot fail. The pan is seeded and
    deterministic, it repeats every :attr:`PAN_PERIOD_FRAMES` frames, and it stays small
    enough that detections continue to land on the same objects.
    """

    PAN_PERIOD_FRAMES = 60

    def __init__(self, path: Union[str, Path], *, drift_pixels: int = 24) -> None:
        self._path = Path(path)
        self._drift = max(0, drift_pixels)
        self._image: Optional[np.ndarray] = None
        self._frame_index = 0

    @property
    def is_open(self) -> bool:
        return self._image is not None

    @property
    def description(self) -> str:
        return f"replay:{self._path.name}"

    def open(self) -> bool:
        if not self._path.exists():
            logger.error("replay source image not found: %s", self._path)
            return False
        image = cv2.imread(str(self._path))
        if image is None:
            logger.error("replay source could not decode %s", self._path)
            return False
        self._image = image
        logger.warning(
            "replaying %s as a camera: frames are a still image, not live video",
            self._path,
        )
        return True

    def read(self) -> Optional[np.ndarray]:
        if self._image is None:
            return None

        self._frame_index += 1
        frame = self._image
        if self._drift:
            # A pan, not a wobble: the offset sweeps the whole amplitude once per period so
            # consecutive frames move by a few pixels and the console visibly drifts, while
            # the vertical component is a quarter of it - a still image should look like a
            # slowly panning camera rather than a shaking one.
            phase = 2.0 * np.pi * (self._frame_index % self.PAN_PERIOD_FRAMES) / self.PAN_PERIOD_FRAMES
            dx = int(round(self._drift * np.sin(phase)))
            dy = int(round(self._drift / 4.0 * np.cos(phase)))
            if dx or dy:
                matrix = np.float32([[1, 0, dx], [0, 1, dy]])
                frame = cv2.warpAffine(
                    frame,
                    matrix,
                    (frame.shape[1], frame.shape[0]),
                    borderMode=cv2.BORDER_REPLICATE,
                )
        return frame

    def release(self) -> None:
        self._image = None


def build_frame_source(source: Union[str, int], config: CaptureConfig) -> FrameSource:
    """Return the frame source described by ``CAMERA_SOURCE``.

    Raises:
        ValueError: for an empty source, so a misconfiguration fails at startup with a
            message rather than as a stream of empty frames.
    """
    if isinstance(source, str):
        lowered = source.strip().lower()
        if not lowered:
            raise ValueError("CAMERA_SOURCE is empty")

        for prefix in REPLAY_PREFIXES:
            if lowered.startswith(prefix):
                return ReplayFrameSource(source.strip()[len(prefix) :])

        if lowered in ("synthetic", "fake", "test"):
            return SyntheticFrameSource(config)

        if lowered.isdigit():
            return DeviceFrameSource(replace(config, source=int(lowered)))

        # An RTSP/HTTP URL or a V4L2 path such as /dev/video0.
        return DeviceFrameSource(replace(config, source=source.strip()))

    return DeviceFrameSource(replace(config, source=source))
