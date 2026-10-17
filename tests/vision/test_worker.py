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
"""Tests for the per-frame worker logic.

Deliberately broker-free: `src/vision/worker.py` has no Celery import, so these
tests exercise exactly the code a worker runs, with the synthetic backend standing
in for OWLv2.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.camera.frames import CapturedFrame  # noqa: E402
from src.vision.backend import SyntheticBackend  # noqa: E402
from src.vision.detections import FrameResult  # noqa: E402
from src.vision.worker import (  # noqa: E402
    VisionRuntimeConfig,
    VisionWorker,
)


def jpeg_bytes(width: int = 64, height: int = 48, value: int = 128) -> bytes:
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    return buffer.tobytes()


def make_frame(frame_number: int = 1) -> CapturedFrame:
    return CapturedFrame(
        camera_id="cam0",
        frame_number=frame_number,
        capture_ts_ns=time.time_ns(),
        jpeg=jpeg_bytes(),
        width=64,
        height=48,
        encode_ms=1.5,
    )


def make_worker(**overrides) -> VisionWorker:
    backend = SyntheticBackend(num_queries=32, num_detections=2, score=0.9)
    config = VisionRuntimeConfig(labels=("metal_sheet",), **overrides)
    return VisionWorker(backend, config, worker_id="test-worker")


def test_process_frame_produces_detections_and_timings() -> None:
    worker = make_worker()
    result = worker.process_frame(make_frame())

    assert isinstance(result, FrameResult)
    assert result.error is None
    assert result.camera_id == "cam0"
    assert result.frame_number == 1
    assert result.worker_id == "test-worker"
    assert result.detection_count == 2
    assert result.detections[0].object_type == "metal_sheet"
    assert result.detections[0].confidence == pytest.approx(0.9, abs=1e-3)
    # Every stage is timed, which is what makes a throughput regression locatable.
    assert result.timings.preprocess_ms > 0
    assert result.timings.inference_ms >= 0
    assert result.timings.decode_ms >= 0
    assert result.timings.encode_ms == pytest.approx(1.5)
    assert result.queue_ms >= 0


def test_worker_processes_a_frame_before_its_capture_timestamp_gracefully() -> None:
    """A clock step must not produce a negative queue delay."""
    frame = CapturedFrame(
        camera_id="cam0",
        frame_number=2,
        capture_ts_ns=time.time_ns() + 10_000_000_000,  # 10 s in the future
        jpeg=jpeg_bytes(),
        width=64,
        height=48,
    )
    result = make_worker().process_frame(frame)
    assert result.queue_ms == 0.0


def test_payload_round_trip() -> None:
    frame = make_frame(7)
    restored = CapturedFrame.from_payload(frame.to_payload())

    assert restored.camera_id == frame.camera_id
    assert restored.frame_number == 7
    assert restored.jpeg == frame.jpeg
    assert restored.capture_ts_ns == frame.capture_ts_ns


def test_process_payload_matches_process_frame() -> None:
    worker = make_worker()
    frame = make_frame(3)
    payload = worker.process_payload(frame.to_payload())

    assert payload["camera_id"] == "cam0"
    assert payload["frame_number"] == 3
    assert len(payload["detections"]) == 2
    assert payload["error"] is None
    assert "timings" in payload


def test_malformed_payload_is_reported_not_raised() -> None:
    """One corrupt message must not take the worker down."""
    payload = make_worker().process_payload({"camera_id": "cam0"})

    assert payload["error"] is not None
    assert "malformed" in payload["error"]
    assert payload["detections"] == []


def test_undecodable_image_is_reported_not_raised() -> None:
    frame = CapturedFrame(
        camera_id="cam0",
        frame_number=9,
        capture_ts_ns=time.time_ns(),
        jpeg=b"this is not a jpeg",
        width=10,
        height=10,
    )
    result = make_worker().process_frame(frame)

    assert result.error is not None
    assert "ValueError" in result.error
    assert result.detection_count == 0


def test_result_payload_matches_the_api_schema() -> None:
    """`to_payload` must line up with `src/camera/schemas.py::StreamingPayload`."""
    payload = make_worker().process_frame(make_frame()).to_payload()

    assert set(payload["metadata"]) == {"camera_id", "frame_number", "timestamp"}
    for detection in payload["detections"]:
        assert set(detection) == {
            "object_type",
            "x_norm",
            "y_norm",
            "width_norm",
            "height_norm",
            "confidence",
        }


def test_frame_result_from_wire_round_trips() -> None:
    """The transport round trip must preserve identity and timings."""
    worker = make_worker()
    original = worker.process_frame(make_frame(11))
    rebuilt = FrameResult.from_wire(original.to_wire())

    assert rebuilt.frame_number == original.frame_number
    assert rebuilt.camera_id == original.camera_id
    assert rebuilt.detection_count == original.detection_count
    assert rebuilt.timings.preprocess_ms == pytest.approx(
        original.timings.preprocess_ms, abs=1e-3
    )


def test_wire_and_api_shapes_are_deliberately_different() -> None:
    """Two consumers, two contracts - conflating them was a real defect.

    `to_wire` is flat for the dispatcher; `to_payload` is the nested envelope for
    WebSocket/API clients (`src/camera/schemas.py`). The dispatcher reads
    `payload["camera_id"]`, which only exists in the flat form.
    """
    result = make_worker().process_frame(make_frame(12))
    wire = result.to_wire()
    api = result.to_payload()

    assert "camera_id" in wire
    assert "metadata" in api
    assert "camera_id" not in api


def test_runtime_config_reads_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("VISION_LABELS", "metal_sheet,detail,debris")
    monkeypatch.setenv("VISION_SCORE_THRESHOLD", "0.4")
    monkeypatch.setenv("VISION_MAX_DETECTIONS", "8")
    monkeypatch.setenv("VISION_IOU_THRESHOLD", "none")

    config = VisionRuntimeConfig.from_env()

    assert config.labels == ("metal_sheet", "detail", "debris")
    assert config.score_threshold == pytest.approx(0.4)
    assert config.max_detections == 8
    assert config.iou_threshold is None  # explicit opt-out, not the default


def test_runtime_config_defaults_are_sane(monkeypatch) -> None:
    for name in (
        "VISION_LABELS",
        "VISION_SCORE_THRESHOLD",
        "VISION_MAX_DETECTIONS",
        "VISION_IOU_THRESHOLD",
        "VISION_MIN_AREA",
        "VISION_MAX_AREA",
    ):
        monkeypatch.delenv(name, raising=False)

    config = VisionRuntimeConfig.from_env()
    assert config.labels == ("metal_sheet",)
    assert config.score_threshold == pytest.approx(0.25)
    assert config.iou_threshold == pytest.approx(0.5)
    assert config.max_area == pytest.approx(0.95)


def test_max_area_discards_a_box_covering_the_frame() -> None:
    """A detection spanning the whole frame is a mis-fire, not a pickable part."""
    worker = make_worker(max_area=0.5)
    # The synthetic backend's 0.1 x 0.08 boxes (area 0.008) are well under 0.5.
    result = worker.process_frame(make_frame())
    assert result.detection_count == 2

    aggressive = make_worker(max_area=0.001)
    assert aggressive.process_frame(make_frame()).detection_count == 0
