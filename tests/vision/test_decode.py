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
"""Tests for decoding OWLv2 graph outputs.

Every case builds logits/boxes with a known answer, so a regression in the box
convention, the sigmoid or the suppression shows up as a specific numeric failure
rather than "detections look wrong".
"""

from __future__ import annotations

import numpy as np
import pytest

from src.vision.decode import (
    center_to_corners,
    decode_batch,
    decode_frame,
    nms,
    sigmoid,
)

LABELS = ("metal_sheet",)
QUERIES = 64  # small stand-in for the model's 3600


def blank_outputs(
    queries: int = QUERIES, classes: int = 1, logit: float = -20.0
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.full((1, queries, classes), logit, dtype=np.float32)
    boxes = np.zeros((1, queries, 4), dtype=np.float16)
    return logits, boxes


def place(
    logits: np.ndarray,
    boxes: np.ndarray,
    index: int,
    *,
    cx: float,
    cy: float,
    w: float,
    h: float,
    score: float,
    class_index: int = 0,
) -> None:
    """Write one query's box and a logit that sigmoids to *score*."""
    boxes[0, index] = np.asarray([cx, cy, w, h], dtype=np.float16)
    logits[0, index, class_index] = np.log(score / (1.0 - score))


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def test_sigmoid_is_stable_at_extremes() -> None:
    """The naive form overflows for large negative input."""
    values = np.asarray([-800.0, -50.0, 0.0, 50.0, 800.0], dtype=np.float32)
    with np.errstate(over="raise", invalid="raise"):
        result = sigmoid(values)

    assert np.isfinite(result).all()
    assert result[0] == pytest.approx(0.0)
    assert result[2] == pytest.approx(0.5)
    assert result[4] == pytest.approx(1.0)


def test_center_to_corners_converts_the_convention() -> None:
    """OWLv2 emits (cx, cy, w, h); everything downstream uses (x1, y1, x2, y2)."""
    boxes = np.asarray([[0.5, 0.5, 0.2, 0.1]], dtype=np.float32)
    corners = center_to_corners(boxes)
    np.testing.assert_allclose(corners, np.asarray([[0.4, 0.45, 0.6, 0.55]]), atol=1e-6)


def test_center_to_corners_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        center_to_corners(np.zeros((4,), dtype=np.float32))


def test_nms_collapses_overlapping_boxes_and_keeps_the_best() -> None:
    boxes = np.asarray(
        [
            [0.10, 0.10, 0.30, 0.30],
            [0.11, 0.11, 0.31, 0.31],  # ~heavy overlap with the first
            [0.70, 0.70, 0.90, 0.90],  # separate
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)

    kept = nms(boxes, scores, iou_threshold=0.5)
    assert list(kept) == [0, 2]


def test_nms_on_empty_input() -> None:
    assert nms(np.empty((0, 4), np.float32), np.empty((0,), np.float32)).size == 0


# --------------------------------------------------------------------------- #
# decode_frame
# --------------------------------------------------------------------------- #
def test_single_high_scoring_query_decodes_to_one_detection() -> None:
    logits, boxes = blank_outputs()
    place(logits, boxes, 5, cx=0.5, cy=0.5, w=0.2, h=0.1, score=0.9)

    detections = decode_frame(logits, boxes, labels=LABELS)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.object_type == "metal_sheet"
    assert detection.confidence == pytest.approx(0.9, abs=1e-4)
    # Centre (0.5, 0.5) with size (0.2, 0.1) -> top-left (0.4, 0.45).
    assert detection.x_norm == pytest.approx(0.4, abs=1e-2)
    assert detection.y_norm == pytest.approx(0.45, abs=1e-2)
    assert detection.width_norm == pytest.approx(0.2, abs=1e-2)
    assert detection.height_norm == pytest.approx(0.1, abs=1e-2)


def test_scores_below_threshold_are_discarded() -> None:
    logits, boxes = blank_outputs()
    place(logits, boxes, 1, cx=0.5, cy=0.5, w=0.2, h=0.2, score=0.10)
    place(logits, boxes, 2, cx=0.5, cy=0.5, w=0.2, h=0.2, score=0.90)

    detections = decode_frame(logits, boxes, labels=LABELS, score_threshold=0.25)
    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.9, abs=1e-3)


def test_detections_are_sorted_by_descending_confidence() -> None:
    logits, boxes = blank_outputs()
    place(logits, boxes, 0, cx=0.2, cy=0.2, w=0.1, h=0.1, score=0.40)
    place(logits, boxes, 1, cx=0.6, cy=0.2, w=0.1, h=0.1, score=0.95)
    place(logits, boxes, 2, cx=0.4, cy=0.8, w=0.1, h=0.1, score=0.70)

    detections = decode_frame(logits, boxes, labels=LABELS, iou_threshold=None)
    confidences = [d.confidence for d in detections]
    assert confidences == sorted(confidences, reverse=True)


def test_boxes_are_clipped_to_the_frame() -> None:
    """A box hanging off the edge must not produce negative coordinates."""
    logits, boxes = blank_outputs()
    place(logits, boxes, 0, cx=0.05, cy=0.05, w=0.4, h=0.4, score=0.9)

    detection = decode_frame(logits, boxes, labels=LABELS)[0]
    assert detection.x_norm >= 0.0
    assert detection.y_norm >= 0.0
    assert detection.x_max <= 1.0
    assert detection.y_max <= 1.0


def test_duplicate_queries_are_suppressed_by_default() -> None:
    """OWLv2 emits no NMS; without it one sheet becomes a cloud of boxes."""
    logits, boxes = blank_outputs()
    for index in range(10):
        place(logits, boxes, index, cx=0.5, cy=0.5, w=0.2, h=0.2, score=0.9 - index * 0.01)

    dense = decode_frame(logits, boxes, labels=LABELS, iou_threshold=0.5)
    raw = decode_frame(logits, boxes, labels=LABELS, iou_threshold=None)

    assert len(dense) == 1
    assert len(raw) == 10


def test_max_detections_caps_the_result() -> None:
    logits, boxes = blank_outputs()
    for index in range(20):
        # Spread them out so NMS keeps them all.
        place(
            logits,
            boxes,
            index,
            cx=0.02 + 0.05 * index,
            cy=0.5,
            w=0.01,
            h=0.01,
            score=0.9,
        )
    detections = decode_frame(
        logits, boxes, labels=LABELS, max_detections=5, iou_threshold=None
    )
    assert len(detections) == 5


def test_area_filters_discard_noise_and_frame_spanning_boxes() -> None:
    logits, boxes = blank_outputs()
    place(logits, boxes, 0, cx=0.5, cy=0.5, w=0.001, h=0.001, score=0.95)  # tiny
    place(logits, boxes, 1, cx=0.5, cy=0.5, w=0.99, h=0.99, score=0.95)  # huge
    place(logits, boxes, 2, cx=0.3, cy=0.3, w=0.2, h=0.2, score=0.95)  # plausible

    detections = decode_frame(
        logits, boxes, labels=LABELS, min_area=0.001, max_area=0.95, iou_threshold=None
    )
    assert len(detections) == 1
    assert detections[0].x_norm == pytest.approx(0.2, abs=1e-2)


def test_float16_boxes_are_handled() -> None:
    """pred_boxes really is float16 in the artifact; arithmetic must upcast."""
    logits = np.full((1, 4, 1), -20.0, dtype=np.float32)
    boxes = np.zeros((1, 4, 4), dtype=np.float16)
    assert boxes.dtype == np.float16
    place(logits, boxes, 0, cx=0.5, cy=0.5, w=0.2, h=0.2, score=0.9)

    detections = decode_frame(logits, boxes, labels=LABELS)
    assert len(detections) == 1
    assert np.isfinite(detections[0].width_norm)


def test_single_class_model_maps_to_the_configured_label() -> None:
    """The shipped artifact emits one class; the label comes from configuration."""
    logits, boxes = blank_outputs(classes=1)
    place(logits, boxes, 0, cx=0.5, cy=0.5, w=0.2, h=0.2, score=0.9, class_index=0)

    detections = decode_frame(logits, boxes, labels=("detail",))
    assert detections[0].object_type == "detail"


def test_mismatched_label_count_is_a_configuration_error() -> None:
    logits, boxes = blank_outputs(classes=3)
    with pytest.raises(ValueError, match="labels"):
        decode_frame(logits, boxes, labels=("only_one",))


def test_two_dimensional_outputs_are_accepted() -> None:
    """Unbatched (Q, C) tensors decode identically to (1, Q, C)."""
    logits, boxes = blank_outputs()
    place(logits, boxes, 0, cx=0.5, cy=0.5, w=0.2, h=0.2, score=0.9)

    batched = decode_frame(logits, boxes, labels=LABELS)
    unbatched = decode_frame(logits[0], boxes[0], labels=LABELS)
    assert len(batched) == len(unbatched) == 1


def test_query_count_mismatch_is_rejected() -> None:
    logits = np.full((1, 10, 1), -20.0, dtype=np.float32)
    boxes = np.zeros((1, 12, 4), dtype=np.float16)
    with pytest.raises(ValueError, match="query count"):
        decode_frame(logits, boxes, labels=LABELS)


def test_no_detections_when_nothing_passes_threshold() -> None:
    logits, boxes = blank_outputs(logit=-20.0)
    assert decode_frame(logits, boxes, labels=LABELS) == []


def test_decode_batch_returns_one_result_per_image() -> None:
    """Batching is the main GPU throughput lever, so per-image decoding must work."""
    batch = 3
    per_image = 8
    logits = np.full((batch, per_image, 1), -20.0, dtype=np.float32)
    boxes = np.zeros((batch, per_image, 4), dtype=np.float16)

    score_logit = float(np.log(0.9 / 0.1))
    for image in range(batch):
        # Each image gets exactly one confident box, at a different query index.
        index = image
        logits[image, index, 0] = score_logit
        boxes[image, index] = np.asarray(
            [0.5, 0.5, 0.2, 0.2], dtype=np.float16
        )

    results = decode_batch(logits, boxes, labels=LABELS)

    assert len(results) == batch
    assert all(len(result) == 1 for result in results)
    assert all(result[0].confidence == pytest.approx(0.9, abs=1e-3) for result in results)


def test_decode_batch_accepts_unbatched_input() -> None:
    logits, boxes = blank_outputs()
    place(logits, boxes, 0, cx=0.5, cy=0.5, w=0.2, h=0.2, score=0.9)

    results = decode_batch(logits[0], boxes[0], labels=LABELS)
    assert len(results) == 1
    assert len(results[0]) == 1
