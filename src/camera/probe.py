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
"""Open a camera source once and report what it is.

This is the difference between "the operator typed an address" and "the address works". It is
deliberately separate from the pipeline: a probe must never disturb a running capture, and a
camera that fails to open must produce a message the operator can act on ("connection refused",
"authentication required") rather than a dead stream twenty seconds later.

The probe does what the pipeline's first second does - open with the same ``CaptureConfig`` for
resolution/FPS hints, read one frame, measure the time - and then releases the device. On a USB
camera the release matters: many devices allow a single open, so a leaked handle would make the
next activation fail with "device busy".
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.camera.capture import CaptureConfig
from src.camera.registry import InvalidSourceError, parse_source, redact_source
from src.camera.sources import build_frame_source

__all__ = ["ProbeResult", "probe_source"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeResult:
    """What a single-frame probe learned."""

    ok: bool
    width: int = 0
    height: int = 0
    latency_ms: float = 0.0
    detail: str = ""

    def to_dict(self, source: str) -> dict:
        return {
            "ok": self.ok,
            "source": redact_source(source),
            "width": self.width,
            "height": self.height,
            "latency_ms": round(self.latency_ms, 1),
            "detail": self.detail,
        }


def probe_source(source: str | int, config: CaptureConfig) -> ProbeResult:
    """Open *source*, read one frame, close it.

    Never raises: a probe is a question, and "it did not work" is an answer. The failure text is
    included because the operator needs to distinguish a wrong password from an unreachable host.

    The source is validated with the same parser the registry uses before it is handed to OpenCV.
    Without that step an unsupported scheme is opened as if it were a device path and reported as
    "opened but produced no frame", which sends the operator looking at the wrong thing.
    """
    if not isinstance(source, int):
        try:
            parse_source(str(source))
        except InvalidSourceError as exc:
            return ProbeResult(ok=False, detail=str(exc))

    started = time.monotonic()
    try:
        frame_source = build_frame_source(source, config)
    except ValueError as exc:
        return ProbeResult(ok=False, detail=str(exc))

    # Open explicitly: a replay source has decoded nothing until it is opened, and a device source
    # negotiates its resolution here. Without this call a probe would report "no frame" for a
    # perfectly good camera, which is exactly the wrong answer.
    opener = getattr(frame_source, "open", None)
    if callable(opener):
        try:
            opened = opener()
        except Exception as exc:  # noqa: BLE001 - any driver failure is a probe result
            logger.info("camera probe of %s failed to open: %s", redact_source(str(source)), exc)
            return ProbeResult(ok=False, detail=f"{type(exc).__name__}: {exc}")
        if opened is False:
            return ProbeResult(
                ok=False,
                latency_ms=(time.monotonic() - started) * 1000.0,
                detail=(
                    f"the source did not open: {redact_source(str(source))}. For a replay source "
                    "the file is missing or undecodable; for a camera the address, credentials or "
                    "codec is wrong."
                ),
            )

    try:
        frame = frame_source.read()
    except Exception as exc:  # noqa: BLE001 - any driver failure is a probe result
        logger.info("camera probe of %s failed: %s", redact_source(str(source)), exc)
        return ProbeResult(ok=False, detail=f"{type(exc).__name__}: {exc}")
    finally:
        release = getattr(frame_source, "release", None)
        if callable(release):
            release()

    latency_ms = (time.monotonic() - started) * 1000.0

    if frame is None:
        return ProbeResult(
            ok=False,
            latency_ms=latency_ms,
            detail=(
                "the source opened but produced no frame within the read timeout; "
                "check the stream path and the camera's main/sub stream setting"
            ),
        )

    height, width = frame.shape[:2]
    return ProbeResult(
        ok=True,
        width=int(width),
        height=int(height),
        latency_ms=latency_ms,
        detail="one frame read",
    )
