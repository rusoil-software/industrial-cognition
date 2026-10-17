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
"""The per-camera prompt must actually reach the model.

A prompt that is stored but never used is worse than no prompt: the operator
believes they configured detection. These tests assert the two hops that carry it -
dispatcher payload, then worker -> backend - with the backend recording what it was
asked for.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.camera.frames import CapturedFrame  # noqa: E402
from src.pipeline.dispatcher import FrameDispatcher, InProcessTransport  # noqa: E402
from src.pipeline.stats import PipelineStats  # noqa: E402
from src.vision.backend import InferenceOutput  # noqa: E402
from src.vision.prompt import encode_prompt  # noqa: E402
from src.vision.worker import VisionRuntimeConfig, VisionWorker  # noqa: E402


class FakeTokenizer:
    name = "fake"
    max_length = 16

    def encode(self, text: str):
        ids = tuple((ord(ch) % 49407) or 1 for ch in text[:15])
        padding = 16 - len(ids)
        return ids + (49407,) * padding, (1,) * len(ids) + (0,) * padding


class RecordingBackend:
    """Records the text query it was called with."""

    name = "recording"

    def __init__(self) -> None:
        self.queries: list[object] = []

    def warmup(self) -> None:
        return

    def close(self) -> None:
        return

    def infer(self, pixel_values, text_query=None):
        self.queries.append(text_query)
        batch = pixel_values.shape[0]
        logits = np.full((batch, 4, 1), -20.0, dtype=np.float32)
        logits[:, 0, 0] = 2.0  # sigmoid ~0.88
        boxes = np.zeros((batch, 4, 4), dtype=np.float16)
        boxes[:, 0] = np.asarray([0.5, 0.5, 0.2, 0.2], dtype=np.float16)
        return InferenceOutput(logits=logits, pred_boxes=boxes)


def jpeg_frame() -> CapturedFrame:
    frame = np.full((32, 32, 3), 120, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    return CapturedFrame(
        camera_id="cam0",
        frame_number=1,
        capture_ts_ns=time.time_ns(),
        jpeg=buffer.tobytes(),
        width=32,
        height=32,
    )


def test_dispatcher_embeds_the_prompt_in_the_payload() -> None:
    prompt = encode_prompt(["metal sheet"], FakeTokenizer(), camera_id="cam0")
    submitted: list[dict] = []

    dispatcher = FrameDispatcher(
        InProcessTransport(lambda payload: submitted.append(payload) or {}),
        PipelineStats(),
        max_in_flight=2,
    )
    dispatcher.dispatch(jpeg_frame(), prompt=prompt)

    assert submitted, "the frame was not submitted"
    assert submitted[0]["prompt"]["labels"] == ["metal sheet"]
    assert submitted[0]["prompt"]["camera_id"] == "cam0"
    # The token IDs travel too, so the worker needs no shared prompt store.
    assert len(submitted[0]["prompt"]["input_ids"]) == 16


def test_dispatcher_payload_has_no_prompt_when_none_is_given() -> None:
    submitted: list[dict] = []
    dispatcher = FrameDispatcher(
        InProcessTransport(lambda payload: submitted.append(payload) or {}),
        PipelineStats(),
        max_in_flight=2,
    )
    dispatcher.dispatch(jpeg_frame())

    assert "prompt" not in submitted[0]


def test_worker_passes_the_camera_prompt_to_the_backend() -> None:
    backend = RecordingBackend()
    worker = VisionWorker(backend, VisionRuntimeConfig(), worker_id="test")
    prompt = encode_prompt(["metal sheet"], FakeTokenizer(), camera_id="cam0")

    worker.process_frame(jpeg_frame(), prompt=prompt)

    assert backend.queries, "the backend was never called"
    query = backend.queries[0]
    assert query is not None
    assert query.labels == ("metal sheet",)
    assert len(query.input_ids) == 16


def test_worker_without_a_prompt_uses_the_backend_default() -> None:
    backend = RecordingBackend()
    worker = VisionWorker(backend, VisionRuntimeConfig(), worker_id="test")

    worker.process_frame(jpeg_frame())

    assert backend.queries == [None]


def test_worker_payload_carries_the_prompt_to_the_result_path() -> None:
    """End to end through the wire format used by the Celery task."""
    backend = RecordingBackend()
    worker = VisionWorker(backend, VisionRuntimeConfig(), worker_id="test")
    prompt = encode_prompt(["metal sheet"], FakeTokenizer(), camera_id="cam0")

    frame = jpeg_frame()
    payload = frame.to_payload()
    payload["prompt"] = prompt.to_dict()

    result = worker.process_payload(payload)

    assert result["error"] is None
    assert result["frame_number"] == 1
