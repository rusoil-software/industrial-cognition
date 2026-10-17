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
"""Decode raw OWLv2 graph outputs into normalised :class:`Detection` objects.

The exported graph produces, for one image:

    logits      (1, 3600, C)   float32  - class logits per query
    pred_boxes  (1, 3600, 4)   float16  - boxes as (cx, cy, w, h), normalised

Three details here are easy to get wrong and are handled explicitly:

1. **Boxes are centre-based.** OWLv2 emits ``(cx, cy, w, h)``; the rest of the
   system (`src/camera/schemas.py`, the robot coordinate mapping) uses
   top-left ``(x, y, w, h)``. :func:`center_to_corners` performs that conversion
   once, here, so no other module has to know about it.
2. **``pred_boxes`` is float16.** The exporter casts the box head to fp16 while
   keeping the graph IO types, so the array must be upcast before arithmetic
   (fp16 overflows at 65504 and has ~3 decimal digits of precision, which is
   visibly coarse on coordinates).
3. **Scores are sigmoid, not softmax.** OWLv2's class head is trained with a
   binary sigmoid per (query, text token) pair, so a softmax would make
   confidence values depend on how many labels happen to be queried.

The number of classes ``C`` is fixed at export time. For the artifact in this
repository ``C == 1``: the text query is baked into the exported graph, so
``input_ids``/``attention_mask`` are present in the signature but no longer
select a label, and every detection decodes to ``labels[0]``. Re-querying with
different text requires re-exporting the model - see `doc/vision-pipeline.md` §3.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from src.vision.detections import Detection

__all__ = [
    "center_to_corners",
    "decode_batch",
    "decode_frame",
    "decode_queries",
    "nms",
    "sigmoid",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    ``1 / (1 + exp(-x))`` overflows for large negative ``x``; the two-branch form
    below is exact and never raises. Prompt logits are unbounded, so this matters
    in practice, not just in theory.
    """
    out = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def center_to_corners(boxes: np.ndarray) -> np.ndarray:
    """Convert ``(cx, cy, w, h)`` to ``(x1, y1, x2, y2)``.

    Both are normalised to ``[0, 1]``; the conversion is a pure translation so it
    needs no knowledge of the image size.
    """
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"expected (N, 4) boxes, got shape {boxes.shape}")
    boxes = boxes.astype(np.float32, copy=False)
    cx, cy, w, h = (boxes[:, i] for i in range(4))
    half_w, half_h = w / 2.0, h / 2.0
    return np.stack(
        [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
        axis=1,
    )


def _box_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU of one ``(x1, y1, x2, y2)`` box against an ``(N, 4)`` array."""
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_box = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_others = np.clip(others[:, 2] - others[:, 0], 0.0, None) * np.clip(
        others[:, 3] - others[:, 1], 0.0, None
    )
    union = area_box + area_others - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Greedy non-maximum suppression; returns the kept indices, best first.

    OWLv2 emits one prediction per query token (3600 of them) and the graph
    contains no NMS, so the same physical sheet is reported by dozens of
    overlapping queries. Without suppression a single sheet becomes a cloud of
    boxes and any downstream "pick the target" logic becomes unstable.

    Implemented in NumPy rather than ``cv2.dnn.NMSBoxes`` so that the behaviour
    is identical on every OpenCV build (the ``dnn`` module is absent from some
    slim wheels) and so it is directly unit-testable.
    """
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    remaining = order
    while remaining.size:
        best = int(remaining[0])
        keep.append(best)
        if remaining.size == 1:
            break
        ious = _box_iou(boxes[best], boxes[remaining[1:]])
        remaining = remaining[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def _resolve_labels(labels: Sequence[str], num_classes: int) -> Sequence[str]:
    """Map the graph's class dimension onto configured labels.

    ``C == 1`` is the baked-query case: the export traced a single text query, so
    every detection is that label. ``C == len(labels)`` is the multi-query export.
    Anything else is a configuration error worth failing fast on, because it means
    the served model and the configured vocabulary disagree.
    """
    if not labels:
        raise ValueError("at least one label is required")
    if num_classes == 1:
        return (labels[0],)
    if num_classes == len(labels):
        return labels
    raise ValueError(
        f"model emits {num_classes} classes but {len(labels)} labels are "
        f"configured ({list(labels)}); re-export the model with matching queries "
        "or set VISION_LABELS accordingly"
    )


def decode_frame(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    *,
    labels: Sequence[str],
    score_threshold: float = 0.25,
    iou_threshold: float | None = 0.5,
    max_detections: int = 32,
    min_area: float = 0.0,
    max_area: float = 1.0,
) -> list[Detection]:
    """Decode one image's graph outputs into detections.

    Args:
        logits: ``(Q, C)`` or ``(1, Q, C)`` class logits.
        pred_boxes: ``(Q, 4)`` or ``(1, Q, 4)`` centre-based normalised boxes.
        labels: class vocabulary; see :func:`_resolve_labels`.
        score_threshold: sigmoid confidence below which a query is discarded.
        iou_threshold: NMS IoU threshold, or ``None`` to keep every box. Kept
            configurable so the raw graph output can be inspected when tuning.
        max_detections: cap on returned detections, best first.
        min_area: discard boxes smaller than this normalised area (model noise).
        max_area: discard boxes larger than this (a box covering most of the
            frame is a mis-detection, not an object to pick).

    Returns:
        Detections sorted by descending confidence, in top-left normalised
        coordinates, clipped to the frame.
    """
    logits = np.asarray(logits)
    pred_boxes = np.asarray(pred_boxes)

    if logits.ndim == 3:
        logits = logits[0]
    if pred_boxes.ndim == 3:
        pred_boxes = pred_boxes[0]
    if logits.ndim != 2 or pred_boxes.ndim != 2:
        raise ValueError(
            f"unexpected output shapes logits={logits.shape} boxes={pred_boxes.shape}"
        )
    if logits.shape[0] != pred_boxes.shape[0]:
        raise ValueError(
            f"logits and boxes disagree on query count: "
            f"{logits.shape[0]} vs {pred_boxes.shape[0]}"
        )

    resolved_labels = _resolve_labels(labels, logits.shape[1])

    # fp16 -> fp32 before any arithmetic; see the module docstring.
    corners = center_to_corners(pred_boxes.astype(np.float32, copy=False))
    np.clip(corners, 0.0, 1.0, out=corners)

    scores_per_class = sigmoid(logits.astype(np.float32, copy=False))
    best_class = scores_per_class.argmax(axis=1)
    best_score = scores_per_class.max(axis=1)

    widths = corners[:, 2] - corners[:, 0]
    heights = corners[:, 3] - corners[:, 1]
    areas = np.clip(widths, 0.0, None) * np.clip(heights, 0.0, None)

    keep = (
        (best_score >= score_threshold)
        & (widths > 0)
        & (heights > 0)
        & (areas >= min_area)
        & (areas <= max_area)
    )
    indices = np.flatnonzero(keep)
    if indices.size == 0:
        return []

    filtered_boxes = corners[indices]
    filtered_scores = best_score[indices]

    if iou_threshold is not None and indices.size > 1:
        kept = nms(filtered_boxes, filtered_scores, iou_threshold)
        indices = indices[kept]
        filtered_boxes = corners[indices]
        filtered_scores = best_score[indices]
        filtered_classes = best_class[indices]
    else:
        filtered_classes = best_class[indices]
        order = np.argsort(filtered_scores)[::-1]
        indices = indices[order]
        filtered_boxes = filtered_boxes[order]
        filtered_scores = filtered_scores[order]
        filtered_classes = filtered_classes[order]

    detections: list[Detection] = []
    for box, score, class_index in zip(
        filtered_boxes[:max_detections],
        filtered_scores[:max_detections],
        filtered_classes[:max_detections],
    ):
        x1, y1, x2, y2 = (float(v) for v in box)
        detections.append(
            Detection(
                object_type=str(resolved_labels[int(class_index)]),
                x_norm=x1,
                y_norm=y1,
                width_norm=max(0.0, x2 - x1),
                height_norm=max(0.0, y2 - y1),
                confidence=float(score),
            )
        )
    return detections


def decode_batch(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    *,
    labels: Sequence[str],
    **kwargs: object,
) -> list[list[Detection]]:
    """Decode every item of a batched graph output.

    Batching is the main throughput lever on a GPU: one 8-image forward pass costs
    far less than eight separate passes, because the per-call overhead and the
    fixed-size image encoder dominate. Supporting it here means the worker can
    raise throughput without changing the decode contract.
    """
    logits = np.asarray(logits)
    pred_boxes = np.asarray(pred_boxes)
    if logits.ndim == 2:
        return [decode_frame(logits, pred_boxes, labels=labels, **kwargs)]  # type: ignore[arg-type]

    batch = logits.shape[0]
    if pred_boxes.shape[0] != batch:
        raise ValueError(
            f"batch mismatch: logits {logits.shape[0]} vs boxes {pred_boxes.shape[0]}"
        )
    return [
        decode_frame(logits[i], pred_boxes[i], labels=labels, **kwargs)  # type: ignore[arg-type]
        for i in range(batch)
    ]


def decode_queries(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    labels: Sequence[str],
    thresholds: Sequence[float],
    *,
    iou_threshold: Optional[float] = 0.5,
    max_detections: int = 32,
    min_area: float = 0.0,
    max_area: float = 1.0,
) -> list[Detection]:
    """Decode a multi-query output: one class per query, one threshold per query.

    This is the multi-object path. The graph returns ``logits (1, 3600, Q)`` for a
    text batch of Q prompts, so each box has a score *per query*; the box is attributed
    to the query that scored it highest, and only if that score clears **that query's**
    threshold. Per-query thresholds are what let "human" run at 0.10 while "metal
    sheet" stays at 0.25.

    NMS is applied **per class**, not globally: a confident robot box must not
    suppress an overlapping human box. Overlapping boxes of the same class still get
    merged, which is what NMS is for.

    Args:
        logits: ``(1, N, Q)`` or ``(N, Q)`` float32.
        pred_boxes: ``(1, N, 4)`` or ``(N, 4)`` centre-form boxes.
        labels: one label per query, in the model's query order.
        thresholds: one threshold per query, same order.
    """
    scores = sigmoid(_squeeze_batch(logits))

    if scores.ndim != 2:
        raise ValueError(f"expected a 2-D score array, got shape {scores.shape}")

    num_boxes, num_queries = scores.shape
    if len(labels) != num_queries:
        raise ValueError(
            f"{len(labels)} labels for {num_queries} queries; the text batch and the "
            "label list must match"
        )
    if len(thresholds) == 1 and num_queries > 1:
        thresholds = tuple(float(thresholds[0]) for _ in range(num_queries))
    if len(thresholds) != num_queries:
        raise ValueError(
            f"{len(thresholds)} thresholds for {num_queries} queries"
        )

    boxes = center_to_corners(_squeeze_batch(pred_boxes))
    limits = np.asarray(thresholds, dtype=np.float32)

    # Per box: the best-scoring query, and whether that score clears its threshold.
    best_query = scores.argmax(axis=1)
    best_score = scores[np.arange(num_boxes), best_query]
    keep = best_score >= limits[best_query]

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = np.clip(widths, 0, None) * np.clip(heights, 0, None)
    keep &= (areas >= min_area) & (areas <= max_area)

    candidates: dict[int, list[Detection]] = {}
    for index in np.flatnonzero(keep):
        query = int(best_query[index])
        box = boxes[index]
        candidates.setdefault(query, []).append(
            Detection(
                object_type=labels[query],
                x_norm=float(box[0]),
                y_norm=float(box[1]),
                width_norm=float(box[2] - box[0]),
                height_norm=float(box[3] - box[1]),
                confidence=float(best_score[index]),
            )
        )

    merged: list[Detection] = []
    for detections in candidates.values():
        if not detections:
            continue
        boxes = np.asarray(
            [
                [d.x_norm, d.y_norm, d.x_norm + d.width_norm, d.y_norm + d.height_norm]
                for d in detections
            ],
            dtype=np.float32,
        )
        scores = np.asarray([d.confidence for d in detections], dtype=np.float32)
        if iou_threshold is None or len(detections) == 1:
            kept = list(range(len(detections)))
        else:
            kept = nms(boxes, scores, iou_threshold=iou_threshold).tolist()
        merged.extend(detections[index] for index in kept)

    merged.sort(key=lambda detection: detection.confidence, reverse=True)
    return merged[:max_detections]


def _squeeze_batch(array: np.ndarray) -> np.ndarray:
    """Drop a leading batch dimension of one, keeping the query axis intact.

    A single-query output is ``(1, N, 1)`` and must become ``(N, 1)``, *not* ``(N,)``:
    the query axis carries the class, so squeezing it away would turn a one-class
    output into something the per-query logic cannot index. (Getting this wrong made
    every single-query decode fail with "expected a 2-D score array".)
    """
    result = np.asarray(array)
    if result.ndim == 3:
        if result.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got {result.shape[0]}")
        result = result[0]
    return result
