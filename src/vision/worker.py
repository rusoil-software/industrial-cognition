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
"""The per-frame work a Celery worker performs.

Kept free of Celery imports on purpose: this module is the unit under test, and
requiring a broker to test detection logic would make the tests slow and flaky.
`src/tasks/celery.py` is a three-line wrapper around :func:`process_frame_payload`.

The worker is deliberately stateless between frames except for the cached
inference session. That is what lets it be replicated arbitrarily (`-c 2` per pod,
N pods) to reach the 60 FPS target by parallelism rather than by making one process
faster than physics allows.

Error policy: a failure is returned as a *result with an ``error`` field*, not
raised. A single malformed frame must not kill the worker and take the other 59
frames-per-second down with it; the dispatcher counts it as ``failed`` and the
pipeline keeps running. The exceptions this preserves are configuration errors
(missing model, bad text query), which are raised at session creation so the pod
fails fast and visibly instead of silently returning no detections.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.camera.frames import CapturedFrame
from src.vision.backend import InferenceBackend, build_backend
from src.vision.decode import decode_queries, decode_frame
from src.vision.detections import FrameResult, StageTimings
from src.vision.preprocess import PreprocessConfig, decode_image, to_model_input
from src.vision.prompt import CameraPrompt, CameraPromptSet, PromptEncodingError

__all__ = ["VisionRuntimeConfig", "VisionWorker", "get_worker", "process_frame_payload"]

logger = logging.getLogger(__name__)

#: Score below which a query is discarded. 0.25 rather than the 0.75 in
#: `src/vision/constants.py`: OWLv2's zero-shot sigmoid scores are typically
#: 0.1-0.4 for a correct detection on a hard target, so a 0.75 gate would report
#: "nothing found" on real frames. Measured on the shipped artifact (see
#: `doc/vision-pipeline.md` §6) the score distribution is far below 0.75.
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_MAX_DETECTIONS = 32


@dataclass(frozen=True, slots=True)
class VisionRuntimeConfig:
    """Detection settings, resolved once per worker process."""

    labels: tuple[str, ...] = ("metal_sheet",)
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    iou_threshold: Optional[float] = DEFAULT_IOU_THRESHOLD
    max_detections: int = DEFAULT_MAX_DETECTIONS
    #: Discard boxes smaller than this normalised area. OWLv2 emits ~3600 queries
    #: and a few will always be tiny accidental hits.
    min_area: float = 0.0
    #: Discard boxes covering most of the frame; for a top-down workcell that is
    #: a mis-detection, never an object to pick.
    max_area: float = 0.95
    preprocess: PreprocessConfig = PreprocessConfig()

    @classmethod
    def from_env(cls) -> "VisionRuntimeConfig":
        """Build from environment, so the deployment configures behaviour."""
        labels_env = os.getenv("VISION_LABELS", "").strip()
        labels = (
            tuple(label.strip() for label in labels_env.split(",") if label.strip())
            if labels_env
            else ("metal_sheet",)
        )
        iou_env = os.getenv("VISION_IOU_THRESHOLD", "").strip()
        return cls(
            labels=labels,
            score_threshold=float(
                os.getenv("VISION_SCORE_THRESHOLD", DEFAULT_SCORE_THRESHOLD)
            ),
            # An empty value disables NMS, which is useful when inspecting raw
            # graph output; "0" is a valid threshold and must not be conflated.
            iou_threshold=(
                None
                if iou_env.lower() in ("none", "off")
                else float(iou_env or DEFAULT_IOU_THRESHOLD)
            ),
            max_detections=int(
                os.getenv("VISION_MAX_DETECTIONS", DEFAULT_MAX_DETECTIONS)
            ),
            min_area=float(os.getenv("VISION_MIN_AREA", 0.0)),
            max_area=float(os.getenv("VISION_MAX_AREA", 0.95)),
        )


class VisionWorker:
    """Turns a dispatched frame payload into a :class:`FrameResult`."""

    def __init__(
        self,
        backend: InferenceBackend,
        config: Optional[VisionRuntimeConfig] = None,
        *,
        worker_id: Optional[str] = None,
    ) -> None:
        self._backend = backend
        self._config = config or VisionRuntimeConfig.from_env()
        self._worker_id = worker_id or (
            f"{socket.gethostname()}:{os.getpid()}"
        )

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def warmup(self) -> None:
        """Load the model and run one throwaway pass.

        Called at worker start rather than on the first frame: without it the
        first dispatched frame pays the ~2 s session load, shows up as a wild
        latency outlier, and may exceed the message's visibility timeout.
        """
        self._backend.warmup()

    def process_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Entry point used by the Celery task. Never raises for a bad frame."""
        try:
            frame = CapturedFrame.from_payload(payload)
        except ValueError as exc:
            logger.warning("discarding malformed frame payload: %s", exc)
            return {
                "camera_id": str(payload.get("camera_id", "unknown")),
                "frame_number": int(payload.get("frame_number", -1)),
                "capture_ts_ns": int(payload.get("capture_ts_ns", 0)),
                "detections": [],
                "queue_ms": 0.0,
                "worker_id": self._worker_id,
                "error": f"malformed payload: {exc}",
                "timings": {},
            }

        # The dispatcher embeds the camera's prompt in the payload so workers stay
        # stateless. Reading it here is what makes a stored prompt actually affect
        # inference: without this, the prompt is persisted, sent, and ignored.
        #
        # An unusable prompt fails the frame instead of falling back to the default.
        # Falling back would report detections for a *different* class than the one
        # configured, which is worse than no detection: the operator sees confident
        # results attributed to the wrong object.
        try:
            prompt = self._prompt_from_payload(payload)
        except PromptEncodingError as exc:
            logger.error("payload carries an unusable prompt: %s", exc)
            return {
                "camera_id": frame.camera_id,
                "frame_number": frame.frame_number,
                "capture_ts_ns": frame.capture_ts_ns,
                "detections": [],
                "queue_ms": 0.0,
                "worker_id": self._worker_id,
                "error": f"unusable prompt: {exc}",
                "timings": {},
            }
        return self.process_frame(frame, prompt=prompt).to_wire()

    def _prompt_from_payload(self, payload: dict[str, Any]) -> Optional[CameraPrompt]:
        """Decode the per-camera prompt carried by the payload, if any.

        A malformed prompt falls back to the backend's configured default rather
        than failing the frame: detection at the default prompt is more useful than
        no detection, and the warning names the problem.
        """
        raw = payload.get("prompt")
        if not raw:
            return None
        try:
            return CameraPromptSet.from_dict(raw)
        except (PromptEncodingError, AttributeError, TypeError) as exc:
            raise PromptEncodingError(str(exc)) from exc


    def process_frame(
        self,
        frame: CapturedFrame,
        prompt: "CameraPromptSet | None" = None,
    ) -> FrameResult:
        """Decode, preprocess, infer, decode. Failures become ``error`` results."""
        started_ns = time.time_ns()
        queue_ms = max(0.0, (started_ns - frame.capture_ts_ns) / 1e6)

        try:
            return self._process_frame_checked(frame, queue_ms, prompt)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the worker
            logger.exception(
                "frame %s from %s failed", frame.frame_number, frame.camera_id
            )
            return FrameResult(
                camera_id=frame.camera_id,
                frame_number=frame.frame_number,
                capture_ts_ns=frame.capture_ts_ns,
                queue_ms=queue_ms,
                worker_id=self._worker_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _decode(self, output: Any, prompt: Optional[CameraPromptSet]) -> list:
        """Decode the graph's output, honouring per-query thresholds when present.

        With a configured prompt set the output has one class channel per query, so
        each box is attributed to its best query and compared against *that* query's
        threshold. Without one (local runs, tests) the backend's single default query
        applies and the runtime configuration is used.
        """
        logits = output.logits
        pred_boxes = output.pred_boxes

        if prompt is not None:
            return decode_queries(
                logits,
                pred_boxes,
                labels=prompt.labels,
                thresholds=prompt.thresholds,
                iou_threshold=self._config.iou_threshold,
                max_detections=self._config.max_detections,
                min_area=self._config.min_area,
                max_area=self._config.max_area,
            )

        query_count = int(np.asarray(logits).shape[-1])
        labels = self._config.labels or ("object",)
        if len(labels) < query_count:
            labels = tuple(labels) + tuple(
                f"class_{index}" for index in range(len(labels), query_count)
            )
        return decode_queries(
            logits,
            pred_boxes,
            labels=tuple(labels)[:query_count],
            thresholds=(self._config.score_threshold,) * query_count,
            iou_threshold=self._config.iou_threshold,
            max_detections=self._config.max_detections,
            min_area=self._config.min_area,
            max_area=self._config.max_area,
        )

    def _process_frame_checked(
        self,
        frame: CapturedFrame,
        queue_ms: float,
        prompt: "CameraPromptSet | None" = None,
    ) -> FrameResult:
        config = self._config

        started = time.perf_counter()
        image = decode_image(frame.jpeg)
        pixel_values = to_model_input(image, config.preprocess)
        preprocess_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        output = self._backend.infer(
            pixel_values, prompt
        )
        inference_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        detections = self._decode(output, prompt)
        decode_ms = (time.perf_counter() - started) * 1000.0

        return FrameResult(
            camera_id=frame.camera_id,
            frame_number=frame.frame_number,
            capture_ts_ns=frame.capture_ts_ns,
            detections=tuple(detections),
            timings=StageTimings(
                preprocess_ms=preprocess_ms,
                inference_ms=inference_ms,
                decode_ms=decode_ms,
                encode_ms=frame.encode_ms,
            ),
            queue_ms=queue_ms,
            worker_id=self._worker_id,
        )


#: One backend per worker process. Loading the ONNX session costs ~2 s and ~293 MB,
#: so it is created once and reused for every frame that process handles. Celery
#: prefork workers inherit it via the parent process only if created pre-fork; this
#: lazy singleton is safe either way because it is created on first use in whichever
#: process serves the task.
_WORKER: Optional[VisionWorker] = None


def get_worker() -> VisionWorker:
    """Return the process-wide worker, building the backend on first use."""
    global _WORKER
    if _WORKER is None:
        config = VisionRuntimeConfig.from_env()
        simulation = float(os.getenv("VISION_SYNTHETIC_LATENCY_MS", "0") or 0.0)
        backend = build_backend(simulate_latency_ms=simulation)
        _WORKER = VisionWorker(backend, config)
        logger.info(
            "vision worker ready: backend=%s worker_id=%s labels=%s",
            backend.name,
            _WORKER.worker_id,
            config.labels,
        )
    return _WORKER


def process_frame_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Convenience entry point for the Celery task."""
    return get_worker().process_payload(payload)
