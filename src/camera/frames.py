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
"""The message that carries one captured frame to a vision worker.

This is the wire format between the capture loop and the Celery workers, and its
design is a throughput decision, not a detail.

At 60 FPS a 1920x1080 frame is 6.2 MB raw, or ~166 MB/s for RGB float32. Neither
can cross a message broker per frame. The frame is therefore JPEG-encoded before
dispatch, at the capture resolution the model actually needs:

    960x960 JPEG q=80  ~=  80-120 KB   ->  ~6 MB/s at 60 FPS

which RabbitMQ, Redis and Celery's JSON serialiser all handle comfortably, and
which keeps the worker's job (decode + normalise + infer) smaller than the
inference itself.

The alternative designs and why they were not chosen for the MVP:

* **Shared memory / mmap ring buffer.** Fastest, but couples capture and worker to
  a single host, which defeats the point of running workers as separate pods.
* **Object storage (MinIO) per frame.** Adds a network round-trip and a GET per
  frame; correct for archival, wasteful at 60 FPS.
* **Raw BGR bytes in the payload.** ~2.7 MB per 960x960 frame, ~165 MB/s at 60 FPS.

The chosen form keeps the worker stateless and horizontally scalable, which is what
the "several workers in parallel" requirement needs. `doc/vision-pipeline.md` §3
records the trade-off and when to revisit it.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Optional

__all__ = ["CapturedFrame"]


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """One frame, compressed and ready to dispatch.

    ``capture_ts_ns`` is taken at the moment the frame left the camera driver and
    travels with the frame, so end-to-end latency is measured from capture rather
    than from dispatch. Without it, a worker that starts late reports a fast
    inference on an already-stale frame.
    """

    camera_id: str
    frame_number: int
    capture_ts_ns: int
    jpeg: bytes
    width: int
    height: int
    encode_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.jpeg:
            raise ValueError("CapturedFrame carries no image data")
        if self.frame_number < 0:
            raise ValueError("frame_number must be non-negative")

    @property
    def size_bytes(self) -> int:
        return len(self.jpeg)

    def to_payload(self) -> dict[str, Any]:
        """Serialise for Celery (JSON-compatible: bytes become base64 text).

        ``base64`` rather than ``latin-1`` decoding: Celery's default JSON
        serialiser must be able to round-trip the payload, and a raw bytes object
        fails that.
        """
        return {
            "camera_id": self.camera_id,
            "frame_number": self.frame_number,
            "capture_ts_ns": self.capture_ts_ns,
            "width": self.width,
            "height": self.height,
            "encode_ms": self.encode_ms,
            "image_b64": base64.b64encode(self.jpeg).decode("ascii"),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CapturedFrame":
        """Rebuild from the dispatched payload.

        Raises:
            ValueError: on a malformed payload. The worker validates at its own
                trust boundary rather than trusting the producer, so one corrupt
                message cannot take down a worker loop.
        """
        try:
            image = base64.b64decode(payload["image_b64"], validate=True)
            return cls(
                camera_id=str(payload["camera_id"]),
                frame_number=int(payload["frame_number"]),
                capture_ts_ns=int(payload["capture_ts_ns"]),
                jpeg=image,
                width=int(payload.get("width", 0)),
                height=int(payload.get("height", 0)),
                encode_ms=float(payload.get("encode_ms", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed frame payload: {exc}") from exc


def now_ns(clock: Optional[Any] = None) -> int:
    """Monotonic-ish timestamp in nanoseconds for latency accounting.

    Uses ``time.time_ns`` rather than ``time.monotonic_ns`` because the timestamp
    crosses process boundaries (capture in one pod, worker in another), where
    ``monotonic`` is unrelated between hosts. Wall-clock can step, so latency is
    clamped at zero downstream rather than allowed to go negative.
    """
    import time

    del clock
    return time.time_ns()
