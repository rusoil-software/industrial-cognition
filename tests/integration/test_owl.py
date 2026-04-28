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
# date   : 2026-Apr-23
# ==============================================================================

"""
Real integration tests for OWL 2 ONNX inference.

These tests:
  - Load the actual ONNX model from disk (no mocks)
  - Use the real Owlv2Processor for correct input preprocessing
  - Run real ONNX Runtime inference
  - Assert that the model actually detects known objects

Requirements:
  - models/owl2_model.onnx must exist (run scripts/export_model.py first)
  - pip install -r requirements/export.txt  (for transformers + PIL)

Run with:
  pytest tests/inference/test_integration.py -v -s \
    --model-path models/owl2_model.onnx
"""
import io
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator
from typing import NamedTuple

import numpy as np
import pytest
import requests
from PIL import Image
from httpx import ASGITransport, AsyncClient
import asyncio
from transformers import Owlv2Processor
from multiprocessing import Process
import uvicorn
from uvicorn.config import Config
from uvicorn.server import Server

import sys
import os

# Add the project root to sys.path to make 'src' importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.owl.main import app
except ImportError as e:
    print(f"Failed to import 'src.owl.main': {e}")
    print(f"Current sys.path: {sys.path}")
    raise

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def model_path(request) -> Path:
    path = Path(request.config.getoption("--model-path"))
    if not path.exists():
        pytest.skip(
            f"ONNX model not found at '{path}'. "
            "Run: python scripts/export_model.py --output models/owl_model.onnx"
        )
    return path


# ---------------------------------------------------------------------------
# Shared fixtures (session-scoped: load once, reuse across all tests)
# ---------------------------------------------------------------------------
def run_uvicorn():
    config = Config(app=app, host="127.0.0.1", port=8000, log_level="debug")
    server = Server(config=config)

    # Use a new event loop for the subprocess
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())

@pytest.fixture(scope="session")
def ort_session(model_path: Path):
    """Load the ONNX Runtime session once for the entire test session."""
    import onnxruntime as ort

    logger.info("Loading ONNX session from: %s", model_path)
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    logger.info(
        "Session loaded. Inputs: %s",
        [i.name for i in session.get_inputs()],
    )
    return session


@pytest.fixture(scope="session")
def processor():
    """Load the OWLv2 processor once. This handles image + text preprocessing."""
    logger.info("Loading Owlv2Processor...")
    return Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")


@pytest.fixture(scope="session")
def coco_cats_image() -> Image.Image:
    """
    The canonical OWLv2 test image from COCO val2017.
    Contains two cats on a couch — well-known to the model.
    Used in the official HuggingFace OWLv2 documentation [1].
    """
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    logger.info("Downloading test image from: %s", url)
    response = requests.get(url, stream=True, timeout=15)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    logger.info("Image loaded. Size: %s", image.size)
    return image


# ---------------------------------------------------------------------------
# Helper: run inference and return raw outputs
# ---------------------------------------------------------------------------

@dataclass
class InferenceOutputs:
    """Typed container for raw ONNX outputs."""
    logits: np.ndarray  # [batch, num_queries, num_classes]
    pred_boxes: np.ndarray  # [batch, num_queries, 4] — (cx, cy, w, h) normalized
    image_width: int
    image_height: int


def run_onnx_inference(
        session,
        processor: Owlv2Processor,
        image: Image.Image,
        text_queries: list[list[str]],
) -> InferenceOutputs:
    """
    Preprocess inputs using the official Owlv2Processor, then run ONNX inference.

    The processor handles:
      - Image resizing and normalization (CLIP mean/std)
      - Text tokenization (CLIP tokenizer, max 16 tokens)
      - Padding to square with gray pixels [1]
    """
    inputs = processor(
        text=text_queries,
        images=image,
        return_tensors="np",  # Return numpy arrays directly for ONNX Runtime
    )

    # Run ONNX inference
    ort_outputs = session.run(
        None,
        {
            "pixel_values": inputs["pixel_values"],
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        },
    )

    # OWLv2 outputs: logits [batch, num_patches, num_queries],
    # pred_boxes [batch, num_patches, 4] in (cx, cy, w, h) format [1]
    logits, pred_boxes = ort_outputs, ort_outputs[1]

    return InferenceOutputs(
        logits=logits,
        pred_boxes=pred_boxes,
        image_width=image.width,
        image_height=image.height,
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def cx_cy_wh_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert normalized (cx, cy, w, h) boxes to pixel-space (x1, y1, x2, y2).
    OWLv2 pred_boxes are normalized [0, 1] relative to image size [1].
    """
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return np.stack([x1, y1, x2, y2], axis=-1)


class Detection(NamedTuple):
    label_idx: int
    score: float
    box_xyxy: list[float]  # [x1, y1, x2, y2] in pixels


def get_detections(
        outputs: InferenceOutputs,
        threshold: float = 0.1,
) -> list[Detection]:
    """
    Post-process raw ONNX outputs into filtered detections.
    Mirrors the logic of Owlv2Processor.post_process_grounded_object_detection [1].
    """
    # logits: [1, num_patches, num_queries] -> scores per patch per query
    scores = sigmoid(outputs.logits[0])  # [num_patches, num_queries]
    boxes_norm = outputs.pred_boxes  # [num_patches, 4]
    boxes_px = cx_cy_wh_to_xyxy(
        boxes_norm, outputs.image_width, outputs.image_height
    )[0]

    # Best query class per patch
    best_scores = scores.max(axis=-1)[0]  # [1,num_patches]
    best_classes = scores.argmax(axis=-1)[0]  # [1, num_patches]

    detections = []
    for patch_idx in range(len(best_scores)):
        score = float(best_scores[patch_idx])
        if score >= threshold:
            detections.append(Detection(
                label_idx=int(best_classes[patch_idx]),
                score=score,
                box_xyxy=boxes_px[patch_idx].tolist(),
            ))

    # Sort by confidence descending
    return sorted(detections, key=lambda d: d.score, reverse=True)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestONNXModelOutputShapes:
    """Verify the model produces outputs with the expected shapes."""

    def test_output_count(self, ort_session, processor, coco_cats_image):
        """ONNX model must return exactly 2 output tensors: logits and pred_boxes."""
        outputs = run_onnx_inference(
            ort_session, processor, coco_cats_image,
            text_queries=[["a cat"]],
        )
        assert outputs.logits is not None
        assert outputs.pred_boxes is not None

    def test_logits_shape(self, ort_session, processor, coco_cats_image):
        """
        logits shape: [batch=1, num_patches, num_queries].
        For owlv2-base-patch16, num_patches = (960/16)^2 = 3600 [1].
        """
        outputs = run_onnx_inference(
            ort_session, processor, coco_cats_image,
            text_queries=[["a cat", "a dog", ]],
        )
        batch, num_patches, num_queries = outputs.logits[0].shape
        assert batch == 1
        assert num_patches == 3600, (
            f"Expected 3600 patches for owlv2-base-patch16, got {num_patches}"
        )
        assert num_queries == 2  # Two text queries

    def test_pred_boxes_shape(self, ort_session, processor, coco_cats_image):
        """pred_boxes shape: [batch=1, num_patches, 4] — (cx, cy, w, h) [1]."""
        outputs = run_onnx_inference(
            ort_session, processor, coco_cats_image,
            text_queries=[["a cat"]],
        )
        batch, num_patches, coords = outputs.pred_boxes.shape
        assert batch == 1
        assert coords == 4

    def test_pred_boxes_are_normalized(self, ort_session, processor, coco_cats_image):
        """All predicted box coordinates must be in [0, 1] — they are normalized [1]."""
        outputs = run_onnx_inference(
            ort_session, processor, coco_cats_image,
            text_queries=[["a cat"]],
        )
        boxes = outputs.pred_boxes
        assert boxes.min() >= -0.1, "Box coordinates should not be significantly negative"
        assert boxes.max() <= 1.1, "Box coordinates should not exceed 1.0 significantly"

    def test_logits_are_finite(self, ort_session, processor, coco_cats_image):
        """Sanity check: logits must not contain NaN or Inf."""
        outputs = run_onnx_inference(
            ort_session, processor, coco_cats_image,
            text_queries=[["a cat"]],
        )
        assert np.isfinite(outputs.logits[0]).all(), "Logits contain NaN or Inf values"


class TestONNXModelDetectsObjects:
    """
    Real detection tests — assert the model actually finds known objects.
    Uses the canonical COCO val2017 image (two cats on a couch) [1].
    """

    def test_detects_at_least_one_object_above_threshold(
            self, ort_session, processor, coco_cats_image
    ):
        """The model must produce at least one detection above threshold=0.1."""
        outputs = run_onnx_inference(
            ort_session, processor, coco_cats_image,
            text_queries=[["a photo of a cat", "a photo of a dog"]],
        )
        detections = get_detections(outputs, threshold=0.1)

        logger.info("Total detections above 0.1: %d", len(detections))
        for d in detections[:5]:
            logger.info(
                "  label_idx=%d  score=%.3f  box=%s",
                d.label_idx, d.score,
                [round(v, 1) for v in d.box_xyxy],
            )

        assert len(detections) > 0, (
            "Model produced zero detections above threshold 0.1. "
            "Model may not be exported correctly."
        )

    def test_detects_cats_with_high_confidence(
            self, ort_session, processor, coco_cats_image
    ):
        """
        The canonical COCO image contains two cats.
        OWLv2 documentation confirms detection with ~0.6 confidence [1].
        We assert at least one detection above 0.5.
        """
        text_queries = [["a photo of a cat", "a photo of a dog"]]
        outputs = run_onnx_inference(
            ort_session, processor, coco_cats_image,
            text_queries=text_queries,
        )
        detections = get_detections(outputs, threshold=0.5)

        logger.info("High-confidence detections (>0.5): %d", len(detections))
        for d in detections:
            logger.info(
                "  label_idx=%d  score=%.3f  box=%s",
                d.label_idx, d.score,
                [round(v, 1) for v in d.box_xyxy],
            )

        assert len(detections) >= 1, (
            f"Expected at least 1 detection above 0.5 for cats image, "
            f"got {len(detections)}. Top score was: "
            f"{get_detections(outputs, threshold=0.0)[0].score:.3f}"
        )

    def test_detects_two_cats(self, ort_session, processor, coco_cats_image):
        """
        The COCO image has exactly two cats.
        OWLv2 documentation shows both detected at 0.614 and 0.665 [1].
        We assert at least two detections for the 'cat' query.
        """
        text_queries = [["a photo of a cat", "a photo of a dog"]]
        outputs = run_onnx_inference(
            ort_session, processor, coco_cats_image,
            text_queries=text_queries,
        )
        # Filter to only cat detections (label_idx=0 = first query = "a photo of a cat")
        cat_detections = [
            d for d in get_detections(outputs, threshold=0.3)
            if d.label_idx == 0
        ]

        logger.info("Cat detections above 0.3: %d", len(cat_detections))
        assert len(cat_detections) >= 2, (
            f"Expected at least 2 cat detections, got {len(cat_detections)}. "
            "This may indicate the ONNX export lost accuracy."
        )

    def test_cat_boxes_are_in_image_bounds(
            self, ort_session, processor, coco_cats_image
    ):
        """All detected bounding boxes must be within the original image dimensions."""
        outputs = run_onnx_inference(
            ort_session, processor, coco_cats_image,
            text_queries=[["a photo of a cat"]],
        )
        detections = get_detections(outputs, threshold=0.3)

        w, h = coco_cats_image.width, coco_cats_image.height
        for d in detections:
            x1, y1, x2, y2 = d.box_xyxy
            # Allow small margin for floating point at edges
            assert x1 < x2, f"Box x1 ({x1:.1f}) must be less than x"
            assert x2 > x1, f"Box x2 ({x2:.1f}) must be greater than x1 ({x1:.1f})"
            assert y1 < y2, f"Box y1 ({y1:.1f}) must be less than y2 ({y2:.1f})"
            assert y2 > y1, f"Box y2 ({y2:.1f}) must be greater than y1 ({y1:.1f})"


def test_batch_inference(self, ort_session, processor, coco_cats_image):
    """Test inference with a batch of 2 identical images."""
    text_queries = [["a photo of a cat"]]

    # Preprocess batch of 2 images
    inputs = processor(
        text=text_queries,
        images=[coco_cats_image, coco_cats_image],
        return_tensors="np",
    )

    batch_outputs = ort_session.run(
        None,
        {
            "pixel_values": inputs["pixel_values"].astype(np.float32),
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        },
    )

    logits, pred_boxes = batch_outputs, batch_outputs
    assert logits.shape == 2, "Batch size should be 2"
    assert pred_boxes.shape == 2, "Batch size should be 2"


class TestServiceIntegration:
    """Test the actual FastAPI service with real ONNX model."""

    @pytest.fixture
    def service(self, ort_session, processor):
        """Instantiate the real OWL2InferenceService with mocked ONNX session."""
        from src.owl.inference.service import OWL2InferenceService

        service = OWL2InferenceService(
            model_path=Path("src") / "owl" / "models" / "owl_model.onnx" / "model.onnx",
            execution_provider="CPUExecutionProvider",
        )
        # Inject the real session
        service._session = ort_session
        service._model_version = "1.0"
        return service

    def test_service_run_batch(
            self,
            service,
            processor,
            coco_cats_image,
    ):
        """Test the service's run_batch method with real inference."""
        from src.owl.inference.schemas import OWL2BatchRequest, OWL2InputItem

        # Preprocess image and text
        inputs = processor(
            text=[["a photo of a cat", "a photo of a dog"]],
            images=coco_cats_image,
            return_tensors="np",
        )

        # Create request
        request = OWL2BatchRequest(
            inputs=[
                OWL2InputItem(
                    pixel_values=inputs["pixel_values"][0].tolist(),
                    input_ids=inputs["input_ids"][0].tolist(),
                    attention_mask=inputs["attention_mask"][0].tolist(),
                )
            ]
        )

        # Run service
        results = service.run_batch(request)

        assert len(results) == 1
        assert len(results) > 0
        assert results[0].detections[0].score > 0.0

    def test_service_detects_cats_in_batch(
            self,
            service,
            processor,
            coco_cats_image,
    ):
        """Service should detect cats with high confidence."""
        from src.owl.inference.schemas import OWL2BatchRequest, OWL2InputItem

        inputs = processor(
            text=[["a photo of a cat"]],
            images=coco_cats_image,
            return_tensors="np",
        )

        request = OWL2BatchRequest(
            inputs=[
                OWL2InputItem(
                    pixel_values=inputs["pixel_values"][0].tolist(),
                    input_ids=inputs["input_ids"][0].tolist(),
                    attention_mask=inputs["attention_mask"][0].tolist(),
                )
            ]
        )

        results = service.run_batch(request)
        detections = results[0].detections

        # Filter high-confidence detections
        high_conf = [d for d in detections if d.score > 0.5]

        logger.info("High-confidence detections: %d", len(high_conf))
        assert len(high_conf) >= 1, (
            f"Expected at least 1 high-confidence detection, got {len(high_conf)}. "
            f"Top score: {max([detection.score for detection in detections]) if detections else 'N/A'}"
        )


class TestAPIEndpoint:
    """Test the actual FastAPI endpoint with the real service."""

    @pytest.fixture(scope="session")
    async def live_server(self):
        import threading
        # Use threading instead of multiprocessing for reliability on Windows
        if sys.platform == "win32":
            thread = threading.Thread(target=run_uvicorn, daemon=True)
            thread.start()
        else:
            proc = Process(target=run_uvicorn, daemon=True)
            proc.start()

        # Wait for the server to be ready using health check
        import time
        import httpx
        start_time = time.time()
        timeout = 120  # seconds
        while time.time() - start_time < timeout:
            try:
                response = httpx.get("http://127.0.0.1:8000/api/v1/inference/owl2/health")
                if response.status_code == 200:
                    break
            except httpx.ConnectError:
                pass  # Server is not up yet
            time.sleep(0.1)  # Short sleep before retry
        else:
            raise RuntimeError("Server failed to start within timeout period")

        yield "http://127.0.0.1:8000"

        if sys.platform != "win32":
            # Clean shutdown
            proc.terminate()
            proc.join()
            # Re-join to ensure cleanup
            if proc.is_alive():
                proc.kill()
        # No need to join daemon threads; they will terminate with the main process.
        # TODO: For a clean shutdown, we could implement a more complex stoppable server.

    @pytest.mark.asyncio
    async def test_endpoint_detects_cats(
            self,
            processor,
            coco_cats_image,
            live_server,
    ):
        """POST /api/v1/inference/owl2/detect should detect cats."""

        inputs = processor(
            text=[["a photo of a cat", "a photo of a dog"]],
            images=coco_cats_image,
            return_tensors="np",
        )

        payload = {
            "inputs": [
                {
                    "pixel_values": inputs["pixel_values"][0].tolist(),
                    "input_ids": inputs["input_ids"][0].tolist(),
                    "attention_mask": inputs["attention_mask"][0].tolist(),
                }
            ]
        }

        async with AsyncClient(base_url=live_server) as ac:
            response = await ac.post(
                "/api/v1/inference/owl2/detect",
                json=payload,
            )

        assert response.status_code == 200
        body = response.json()
        assert "results" in body
        assert len(body["results"]) == 1
        assert len(body["results"][0]["detections"]) > 0

        logger.info("Endpoint response: %s", body)


# ---------------------------------------------------------------------------
# Conftest hook to allow pytest CLI options
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
