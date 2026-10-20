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
"""Frame capture and encoding.

For the MVP the camera is a device attached directly to the server
(`/dev/video0`), so capture is an OpenCV ``VideoCapture`` on a device index or a
V4L2 path. The same class accepts an RTSP URL, which is what the multi-camera
deployment uses - keeping one code path for both is worth it because the failure
modes and the tuning knobs are the same.

Two things here matter more than they look:

* **Request MJPG, not raw**. A USB3/GigE industrial camera at 60 FPS is only
  reachable over USB 3 with a compressed pixel format; requesting
  ``CAP_PROP_FOURCC = MJPG`` before setting resolution/FPS is what makes 60 FPS
  negotiable at all. With the default YUYV the link saturates around 1080p30.
* **Do not let the driver queue frames**. ``CAP_PROP_BUFFERSIZE = 1`` (where the
  backend honours it) plus reading in a tight loop means we always analyse the
  newest frame. Otherwise a slow consumer drains a backlog of stale frames and
  latency grows without bound while FPS looks fine - the classic video pipeline
  bug. Where the backend ignores the property, the dispatcher's load shedding
  (§`src/pipeline/dispatcher.py`) is the backstop.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import cv2
import numpy as np

from src.camera.frames import CapturedFrame, now_ns

__all__ = [
    "CaptureConfig",
    "DeviceFrameSource",
    "FrameSource",
    "RTSP_TRANSPORT_ENV",
    "SyntheticFrameSource",
    "encode_jpeg",
    "is_rtsp_source",
    "rtsp_transport",
]

logger = logging.getLogger(__name__)

#: Environment variable holding the RTSP transport OpenCV should negotiate.
RTSP_TRANSPORT_ENV = "RTSP_TRANSPORT"
#: OpenCV's FFmpeg backend reads its options from here (``;``-separated pairs).
FFMPEG_CAPTURE_OPTIONS_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
RTSP_TRANSPORTS = ("tcp", "udp")
RTSP_PREFIXES = ("rtsp://", "rtsps://")
#: TCP is the default: an RTSP stream behind NAT or across a lossy link loses every
#: UDP fragment it drops, and one lost fragment costs a whole frame. TCP retransmits
#: at the price of a little jitter, which a detector sampling frames does not feel.
DEFAULT_RTSP_TRANSPORT = "tcp"


def is_rtsp_source(source: object) -> bool:
    """True when ``source`` is an RTSP URL rather than a device."""
    return isinstance(source, str) and source.strip().lower().startswith(RTSP_PREFIXES)


def rtsp_transport() -> str:
    """The RTSP transport to negotiate, from ``RTSP_TRANSPORT``."""
    configured = (os.getenv(RTSP_TRANSPORT_ENV) or DEFAULT_RTSP_TRANSPORT).strip().lower()
    if configured in RTSP_TRANSPORTS:
        return configured
    logger.warning(
        "ignoring %s=%r; expected one of %s. Using %s",
        RTSP_TRANSPORT_ENV,
        configured,
        ", ".join(RTSP_TRANSPORTS),
        DEFAULT_RTSP_TRANSPORT,
    )
    return DEFAULT_RTSP_TRANSPORT


def _push_rtsp_transport(transport: Optional[str] = None) -> Optional[str]:
    """Point OpenCV's FFmpeg backend at ``transport`` and return the previous value.

    OpenCV exposes no API for RTSP transport options; the FFmpeg backend reads
    ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` when a capture is opened. Operator-supplied
    options are kept, and any existing ``rtsp_transport`` pair is replaced rather
    than duplicated.
    """
    previous = os.environ.get(FFMPEG_CAPTURE_OPTIONS_ENV)
    # Options come as "name;value;name;value", so dropping an option means dropping
    # its value with it - otherwise a stray "udp" would become a value for another
    # option.
    pairs = (previous or "").split(";")
    kept: list[str] = []
    index = 0
    while index < len(pairs):
        pair = pairs[index]
        if pair == "rtsp_transport":
            index += 2
            continue
        if pair:
            kept.append(pair)
        index += 1
    kept.extend(("rtsp_transport", transport or rtsp_transport()))
    os.environ[FFMPEG_CAPTURE_OPTIONS_ENV] = ";".join(kept)
    return previous


def _restore_capture_options(previous: Optional[str]) -> None:
    if previous is None:
        os.environ.pop(FFMPEG_CAPTURE_OPTIONS_ENV, None)
    else:
        os.environ[FFMPEG_CAPTURE_OPTIONS_ENV] = previous


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """How to open and tune one camera."""

    camera_id: str = "cam0"
    #: Device index (``0`` for ``/dev/video0``), a V4L2 path, or an RTSP URL.
    source: str | int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 60
    #: JPEG quality for the dispatch payload. 80 is the knee: visually lossless for
    #: detection purposes at roughly a third of the bytes of q=95.
    jpeg_quality: int = 80
    #: Encoded long-edge cap. The model input is 960x960, so encoding above that
    #: only inflates the broker payload; 960 keeps a 1080p frame at ~100 KB.
    encode_max_side: int = 960
    #: Ask the driver for MJPG. Needed for 60 FPS over USB 3.
    prefer_mjpeg: bool = True
    read_timeout_ms: int = 5000


@runtime_checkable
class FrameSource(Protocol):
    """A source of BGR frames."""

    @property
    def is_open(self) -> bool: ...

    def open(self) -> bool: ...

    def read(self) -> Optional[np.ndarray]: ...

    def release(self) -> None: ...


class DeviceFrameSource:
    """OpenCV-backed capture from a device index, V4L2 path or RTSP URL."""

    def __init__(self, config: CaptureConfig) -> None:
        self._config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._negotiated: dict[str, float] = {}

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def negotiated(self) -> dict[str, float]:
        """What the driver actually agreed to, as opposed to what was requested.

        Cameras routinely ignore a request for 60 FPS or 1080p and silently fall
        back; logging the negotiated values is the difference between "we asked
        for 60 FPS" and "we are getting 60 FPS".
        """
        return dict(self._negotiated)

    def open(self) -> bool:
        config = self._config
        source: str | int = config.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        previous_options: Optional[str] = None
        if is_rtsp_source(source):
            transport = rtsp_transport()
            logger.info("camera %s: negotiating RTSP over %s", config.camera_id, transport)
            previous_options = _push_rtsp_transport(transport)
        try:
            capture = cv2.VideoCapture(source)
        finally:
            # The transport is chosen while opening, so the process-wide variable
            # does not need to stay set (and a non-RTSP camera is unaffected).
            if is_rtsp_source(source):
                _restore_capture_options(previous_options)
        if not capture.isOpened():
            logger.error("could not open capture source %r", config.source)
            capture.release()
            return False

        if config.prefer_mjpeg:
            # Must be set before the resolution/framerate or the driver may have
            # already committed to an uncompressed format.
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        capture.set(cv2.CAP_PROP_FPS, config.fps)
        # Best-effort: keeps the driver queue at one frame so read() returns the
        # newest frame rather than a backlog.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = capture
        self._negotiated = {
            "width": float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "fourcc": float(capture.get(cv2.CAP_PROP_FOURCC)),
        }
        logger.info(
            "camera %s opened: requested %sx%s@%s, negotiated %sx%s@%s",
            config.camera_id,
            config.width,
            config.height,
            config.fps,
            int(self._negotiated["width"]),
            int(self._negotiated["height"]),
            self._negotiated["fps"],
        )
        if self._negotiated["fps"] and self._negotiated["fps"] < config.fps * 0.9:
            logger.warning(
                "camera %s negotiated %.1f FPS, below the requested %s FPS; "
                "the 60 FPS requirement cannot be met by this device/link",
                config.camera_id,
                self._negotiated["fps"],
                config.fps,
            )
        return True

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("camera %s released", self._config.camera_id)


class SyntheticFrameSource:
    """Deterministic frame generator for tests and for demoing without hardware.

    Simulates a workcell: a light sheet moving slowly across a darker background,
    at a configurable rate. Deterministic in the frame counter, so a test can
    assert exact behaviour.
    """

    def __init__(
        self,
        config: Optional[CaptureConfig] = None,
        *,
        width: int = 320,
        height: int = 240,
    ) -> None:
        self._config = config or CaptureConfig(camera_id="synthetic")
        self._width = width
        self._height = height
        self._frame_number = 0
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> bool:
        self._open = True
        return True

    def read(self) -> Optional[np.ndarray]:
        if not self._open:
            return None
        frame = np.full((self._height, self._width, 3), 25, dtype=np.uint8)
        offset = (self._frame_number * 4) % max(1, self._width - 80)
        cv2.rectangle(
            frame,
            (offset, self._height // 3),
            (offset + 80, self._height // 3 + 60),
            (215, 210, 205),
            -1,
        )
        self._frame_number += 1
        return frame

    def release(self) -> None:
        self._open = False


def encode_jpeg(
    frame: np.ndarray,
    *,
    quality: int = 80,
    max_side: int = 960,
) -> tuple[bytes, int, int]:
    """Downscale-if-needed and JPEG-encode a frame for dispatch.

    Returns:
        ``(jpeg_bytes, width, height)`` of the *encoded* image.

    The frame is downscaled only when its long edge exceeds ``max_side``. Encoding
    at capture resolution when the model input is 960x960 would triple the broker
    payload for bytes the model never sees.

    Raises:
        ValueError: when encoding fails, which indicates a corrupt frame rather
            than a transient condition.
    """
    if frame is None or frame.size == 0:
        raise ValueError("cannot encode an empty frame")

    height, width = frame.shape[:2]
    long_side = max(height, width)
    if max_side and long_side > max_side:
        scale = max_side / float(long_side)
        frame = cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        height, width = frame.shape[:2]

    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buffer.tobytes(), width, height


def capture_frame(
    source: FrameSource,
    config: CaptureConfig,
    frame_number: int,
) -> Optional[CapturedFrame]:
    """Read, encode and stamp one frame; ``None`` when the source has no frame.

    Kept as a free function so the capture loop stays a thin, readable policy
    (pacing, shedding) and this per-frame work is unit-testable in isolation.
    """
    started = time.perf_counter()
    frame = source.read()
    if frame is None:
        return None
    jpeg, width, height = encode_jpeg(
        frame, quality=config.jpeg_quality, max_side=config.encode_max_side
    )
    encode_ms = (time.perf_counter() - started) * 1000.0
    return CapturedFrame(
        camera_id=config.camera_id,
        frame_number=frame_number,
        capture_ts_ns=now_ns(),
        jpeg=jpeg,
        width=width,
        height=height,
        encode_ms=encode_ms,
    )
