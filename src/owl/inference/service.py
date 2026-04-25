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

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.owl.inference.config import inference_settings
from src.owl.inference.exceptions import (
    InferenceFailedException,
    InvalidInputShapeException,
    ModelNotLoadedException,
)
from src.owl.inference.schemas import (
    DetectionBox,
    OWL2BatchRequest,
    OWL2InferenceResult,
)

logger = logging.getLogger(__name__)


class OWL2InferenceService:
    """
    Wraps an ONNX Runtime session for OWL 2 object detection.
    Instantiated once at startup and injected via dependency.
    """

    def __init__(self, model_path: Path, execution_provider: str) -> None:
        self._model_path = model_path
        self._execution_provider = execution_provider
        self._session: ort.InferenceSession | None = None
        self._model_version: str = "unknown"

    def load(self) -> None:
        """Load the ONNX model. Called once during app lifespan startup."""
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self._model_path}")

        logger.info("Loading OWL 2 ONNX model from %s ...", self._model_path)
        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=[self._execution_provider],
        )
        # Store metadata for response tracing
        meta = self._session.get_modelmeta()
        self._model_version = meta.version or "1.0"
        logger.info("OWL 2 model loaded. Version: %s", self._model_version)

    def unload(self) -> None:
        """Release the session. Called during app shutdown."""
        self._session = None
        logger.info("OWL 2 ONNX session released.")

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def session(self):
        return self._session

    def assert_loaded(self) -> ort.InferenceSession:
        if self._session is None:
            raise ModelNotLoadedException()
        return self._session

    def run_batch(self, request: OWL2BatchRequest) -> list[OWL2InferenceResult]:
        """
        Synchronous inference — intentionally NOT async.
        This is CPU-bound work; the router offloads it to a thread pool.
        """
        session = self.assert_loaded()

        results: list[OWL2InferenceResult] = []

        for idx, item in enumerate(request.inputs):
            try:
                pixel_values = np.array(
                    item.pixel_values, dtype=np.float32
                )[np.newaxis, ...]  # shape: [1, C, H, W]

                input_ids = np.array(
                    item.input_ids, dtype=np.int64
                )[np.newaxis, ...]  # shape: [1, seq_len]

                attention_mask = np.array(
                    item.attention_mask, dtype=np.int64
                )[np.newaxis, ...]  # shape: [1, seq_len]

            except ValueError as exc:
                raise InvalidInputShapeException(str(exc)) from exc

            try:
                outputs = session.run(
                    None,
                    {
                        "pixel_values": pixel_values,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    },
                )
            except ort.capi.onnxruntime_pybind11_state.InvalidArgument as exc:
                raise InvalidInputShapeException(str(exc)) from exc
            except Exception as exc:
                logger.exception("ONNX inference error at index %d", idx)
                raise InferenceFailedException(str(exc)) from exc

            # Unpack OWL 2 outputs: logits [1, num_queries, num_classes],
            # pred_boxes [1, num_queries, 4]
            logits, pred_boxes = outputs, outputs[1]
            scores = _sigmoid(logits)  # [num_queries, num_classes]
            boxes = pred_boxes  # [num_queries, 4]

            detections = _build_detections(scores, boxes)
            results.append(OWL2InferenceResult(index=idx, detections=detections))

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _build_detections(
        scores: np.ndarray,
        boxes: np.ndarray,
        score_threshold: float = 0.1,
) -> list[DetectionBox]:
    detections: list[DetectionBox] = []
    max_scores = scores.max(axis=-1)  # [num_queries]
    best_classes = scores.argmax(axis=-1)  # [num_queries]

    for query_idx, (score, cls_idx) in enumerate(zip(max_scores, best_classes)):
        if float(score) < score_threshold:
            continue
        detections.append(
            DetectionBox(
                box=boxes[query_idx].tolist(),
                score=float(score),
                label=str(int(cls_idx)),
            )
        )
    return detections


@lru_cache(maxsize=1)
def get_inference_service() -> OWL2InferenceService:
    """
    Cached factory — returns the same service instance for the app's lifetime.
    Used as a FastAPI dependency.
    """
    return OWL2InferenceService(
        model_path=inference_settings.OWL2_MODEL_PATH,
        execution_provider=inference_settings.OWL2_EXECUTION_PROVIDER,
    )
