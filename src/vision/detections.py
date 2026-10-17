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
"""Value objects that cross the pipeline boundary.

Why plain dataclasses and not pydantic, unlike `src/camera/schemas.py`:

The pipeline is a 60 FPS hot path. At 60 frames/s every frame allocates a result
object, and pydantic validates on construction - model validation, field
coercion and a dict round-trip per frame, on every worker, for data the system
produced itself. That is measurable overhead with no safety benefit, because the
only untrusted input is the camera stream, which is validated once at the edge
(decode + size checks) rather than 60 times a second.

Pydantic remains the right tool at the *edges* - the HTTP/WebSocket boundary in
`src/camera/schemas.py` and `src/owl/inference/schemas.py` - and
:meth:`FrameResult.to_payload` produces exactly the plain dict those models
accept. `doc/vision-pipeline.md` §2 records this split.

`slots=True` matters here: a `Detection` is a handful of floats and there can be
tens of them per frame, so the per-instance `__dict__` is pure overhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["Detection", "FrameResult", "StageTimings"]


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object, in normalised frame coordinates.

    The four geometry fields are normalised to ``[0, 1]`` against the *captured*
    frame, not against the model's square input. `src/vision/preprocess.py`
    stretches the frame to the model's 960x960 input, so normalised coordinates
    are resolution independent and map straight back onto the original frame -
    which is what the robot side needs when converting to millimetres via
    `src/vision/constants.py::SCALE_FACTOR_MM_PER_UNIT`.

    The fields mirror `src/camera/schemas.py::DetectionResult` so that
    :meth:`to_payload` can hand a result to the existing API models unchanged.
    """

    object_type: str
    x_norm: float
    y_norm: float
    width_norm: float
    height_norm: float
    confidence: float

    def to_payload(self) -> dict[str, Any]:
        """Return the dict shape `src/camera/schemas.py::DetectionResult` expects."""
        return {
            "object_type": self.object_type,
            "x_norm": self.x_norm,
            "y_norm": self.y_norm,
            "width_norm": self.width_norm,
            "height_norm": self.height_norm,
            "confidence": self.confidence,
        }

    @property
    def x_max(self) -> float:
        return min(1.0, self.x_norm + self.width_norm)

    @property
    def y_max(self) -> float:
        return min(1.0, self.y_norm + self.height_norm)

    @property
    def area(self) -> float:
        """Normalised area, used to prefer the largest sheet when several match."""
        return self.width_norm * self.height_norm


@dataclass(frozen=True, slots=True)
class StageTimings:
    """Per-stage wall-clock timings, in milliseconds.

    Kept separate from the result so that the *cheap* path (a frame with no
    detections) still carries its timings, which is what makes throughput
    debuggable: "60 FPS captured, 12 FPS inferred" is only actionable when you
    can see where the time went.
    """

    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    decode_ms: float = 0.0
    encode_ms: float = 0.0

    @property
    def compute_ms(self) -> float:
        """Time spent doing work inside the worker (excludes queueing/transport)."""
        return self.preprocess_ms + self.inference_ms + self.decode_ms


@dataclass(frozen=True, slots=True)
class FrameResult:
    """The outcome of analysing one captured frame.

    ``capture_ts_ns`` is carried through from capture so that end-to-end latency
    can be measured where it matters - at the consumer - rather than only inside
    the worker. A worker that reports 20 ms while the frame sat in the queue for
    400 ms is a healthy-looking system with a broken latency budget.

    ``queue_ms`` is the time between capture and the worker starting on the
    frame; it is the canary for insufficient worker capacity.
    """

    camera_id: str
    frame_number: int
    capture_ts_ns: int
    detections: tuple[Detection, ...] = ()
    timings: StageTimings = field(default_factory=StageTimings)
    queue_ms: float = 0.0
    #: Capture-to-result latency, filled in by the dispatcher. The worker cannot
    #: know it, because the frame may have waited in the broker on the way.
    e2e_latency_ms: float = 0.0
    worker_id: str = ""
    error: Optional[str] = None

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    def to_payload(self) -> dict[str, Any]:
        """Return the dict shape `src/camera/schemas.py::StreamingPayload` expects.

        ``metadata`` mirrors ``FrameMetadata``; ``detections`` mirrors a list of
        ``DetectionResult``. Timestamps are emitted as ISO-8601 UTC so the
        Pydantic model parses them without a custom encoder.
        """
        import datetime

        captured = datetime.datetime.fromtimestamp(
            self.capture_ts_ns / 1e9, tz=datetime.timezone.utc
        )
        return {
            "metadata": {
                "camera_id": self.camera_id,
                "frame_number": self.frame_number,
                "timestamp": captured.isoformat(),
            },
            "detections": [detection.to_payload() for detection in self.detections],
            "timings": {
                "preprocess_ms": round(self.timings.preprocess_ms, 3),
                "inference_ms": round(self.timings.inference_ms, 3),
                "decode_ms": round(self.timings.decode_ms, 3),
                "queue_ms": round(self.queue_ms, 3),
                "e2e_latency_ms": round(self.e2e_latency_ms, 3),
            },
            "worker_id": self.worker_id,
            "error": self.error,
        }

    def to_wire(self) -> dict[str, Any]:
        """Flat, transport-friendly form, used as the Celery task's return value.

        Distinct from :meth:`to_payload`, which is shaped for the WebSocket/API
        clients (`metadata` + `detections`, per `src/camera/schemas.py`). Two
        shapes exist because a client wants a self-describing envelope, while the
        dispatcher wants cheap, flat fields it can reconcile against the frame it
        sent. Conflating them was a real defect caught by the tests: the dispatcher
        read `payload["camera_id"]` from the API-shaped dict and got ``KeyError``.
        """
        return {
            "camera_id": self.camera_id,
            "frame_number": self.frame_number,
            "capture_ts_ns": self.capture_ts_ns,
            "detections": [detection.to_payload() for detection in self.detections],
            "queue_ms": round(self.queue_ms, 3),
            "worker_id": self.worker_id,
            "error": self.error,
            "timings": {
                "preprocess_ms": round(self.timings.preprocess_ms, 3),
                "inference_ms": round(self.timings.inference_ms, 3),
                "decode_ms": round(self.timings.decode_ms, 3),
                "encode_ms": round(self.timings.encode_ms, 3),
            },
        }

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> "FrameResult":
        """Rebuild a result from :meth:`to_wire` output.

        Used by the dispatcher, which holds the ``capture_ts_ns`` of the frame it
        dispatched and can therefore compute end-to-end latency - something the
        worker cannot know, since the frame may have waited in the broker before
        reaching it.
        """
        timings = payload.get("timings") or {}
        detections = tuple(
            Detection(
                object_type=str(item["object_type"]),
                x_norm=float(item["x_norm"]),
                y_norm=float(item["y_norm"]),
                width_norm=float(item["width_norm"]),
                height_norm=float(item["height_norm"]),
                confidence=float(item["confidence"]),
            )
            for item in payload.get("detections") or ()
        )
        return cls(
            camera_id=str(payload.get("camera_id", "")),
            frame_number=int(payload.get("frame_number", -1)),
            capture_ts_ns=int(payload.get("capture_ts_ns", 0)),
            detections=detections,
            timings=StageTimings(
                preprocess_ms=float(timings.get("preprocess_ms", 0.0)),
                inference_ms=float(timings.get("inference_ms", 0.0)),
                decode_ms=float(timings.get("decode_ms", 0.0)),
                encode_ms=float(timings.get("encode_ms", 0.0)),
            ),
            queue_ms=float(payload.get("queue_ms", 0.0)),
            e2e_latency_ms=float(payload.get("e2e_latency_ms", 0.0)),
            worker_id=str(payload.get("worker_id", "")),
            error=payload.get("error"),
        )
