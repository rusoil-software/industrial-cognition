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
# date   : 2026-Oct-21
# ==============================================================================
"""Turn "what a model returned" into the numbers the CSV column claims.

Two rules are fixed here so that no model can be measured kindly:

* a **hit** is a box that overlaps the ground truth with IoU >= :data:`HIT_IOU`, and the
  best overlapping box per label counts once - a model that returns forty boxes around
  the part does not get forty hits;
* the **station answer** is produced by the same :class:`~src.vision.presence.PresenceGate`
  the pipeline runs in production, with the values measured for this stand, so the
  benchmark measures the system that ships rather than a benchmark-only heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

from src.benchmark.dataset import Box, Sample
from src.vision.detections import Detection
from src.vision.presence import PresenceGate, select_presence

#: Overlap at which a box counts as the object. 0.5 is the usual detection convention;
#: 0.3 is used for the station because the part is a plain rectangle whose true corners
#: are ambiguous by a few percent of the frame.
HIT_IOU = 0.3

#: The gate measured for this stand in ``doc/vision-pipeline.md`` section 19.2.
STAND_GATE = PresenceGate(
    roi=(0.15, 0.42, 0.85, 0.85),
    min_area=0.02,
    max_area=0.20,
    min_aspect=1.0,
    max_aspect=2.0,
    min_confidence=0.05,
    min_inside=0.5,
)


@dataclass(frozen=True, slots=True)
class Prediction:
    """One box a model returned, in normalised top-left coordinates."""

    label: str
    confidence: float
    box: Box

    def as_detection(self) -> Detection:
        x, y, w, h = self.box
        return Detection(
            object_type=self.label,
            x_norm=x,
            y_norm=y,
            width_norm=w,
            height_norm=h,
            confidence=self.confidence,
        )


@dataclass(slots=True)
class SampleScore:
    """What one model did on one image."""

    sample_id: str
    scenario: str
    positive: bool
    boxes: int
    best_iou: float = 0.0
    hit: bool = False
    present: Optional[bool] = None
    chosen_confidence: float = 0.0
    reason: str = ""
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "scenario": self.scenario,
            "positive": self.positive,
            "boxes": self.boxes,
            "best_iou": round(self.best_iou, 4),
            "hit": self.hit,
            "present": self.present,
            "chosen_confidence": round(self.chosen_confidence, 4),
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass(slots=True)
class ModelScore:
    """The aggregate row for one model."""

    key: str
    samples: list[SampleScore] = field(default_factory=list)

    def add(self, score: SampleScore) -> None:
        self.samples.append(score)

    def metrics(self) -> dict[str, float]:
        return aggregate(self.samples)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two normalised ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


def best_iou(predictions: Iterable[Prediction], label: str, target: Box) -> tuple[float, Optional[Prediction]]:
    """Highest IoU any box for ``label`` achieves against ``target``.

    The first qualifying box is kept even when it misses entirely, so the details CSV can
    say "this is the box the model preferred" instead of "there was no box".
    """
    best = 0.0
    chosen: Optional[Prediction] = None
    for prediction in predictions:
        if prediction.label != label:
            continue
        value = iou(prediction.box, target)
        if chosen is None or value > best:
            best, chosen = value, prediction
    return best, chosen


def score_sample(
    sample: Sample,
    predictions: Sequence[Prediction],
    *,
    gate: PresenceGate = STAND_GATE,
    latency_ms: float = 0.0,
) -> SampleScore:
    """Score one model's answer for one image."""
    score = SampleScore(
        sample_id=sample.sample_id,
        scenario=sample.scenario,
        positive=sample.positive,
        boxes=len(predictions),
        latency_ms=latency_ms,
    )
    if sample.ground_truth:
        best = 0.0
        for label, target in sample.ground_truth.items():
            value, _ = best_iou(predictions, label, target)
            best = max(best, value)
        score.best_iou = best
        score.hit = best >= HIT_IOU

    if sample.scenario == "station":
        detections = [prediction.as_detection() for prediction in predictions if prediction.label in sample.labels]
        chosen, report = select_presence(detections, gate)
        score.present = bool(report["present"])
        score.reason = str(report.get("reason") or "")
        score.chosen_confidence = float(chosen.confidence) if chosen is not None else 0.0
    return score


def aggregate(scores: Sequence[SampleScore]) -> dict[str, float]:
    """The row of the CSV: recall, specificity, localisation and pace."""
    positives = [s for s in scores if s.scenario == "station" and s.positive]
    negatives = [s for s in scores if s.scenario == "station" and not s.positive]
    warehouse = [s for s in scores if s.scenario == "warehouse"]

    presence_recall = (
        sum(1 for s in positives if s.present) / len(positives) if positives else 0.0
    )
    presence_specificity = (
        sum(1 for s in negatives if s.present is False) / len(negatives) if negatives else 0.0
    )
    return {
        "presence_recall": presence_recall,
        "presence_specificity": presence_specificity,
        "sheet_iou_mean": _mean([s.best_iou for s in positives]),
        "sheet_hit_rate": (sum(1 for s in positives if s.hit) / len(positives)) if positives else 0.0,
        "warehouse_hits": float(sum(1 for s in warehouse if s.hit)),
        "warehouse_iou_mean": _mean([s.best_iou for s in warehouse]),
        "samples": float(len(scores)),
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


__all__ = [
    "HIT_IOU",
    "STAND_GATE",
    "ModelScore",
    "Prediction",
    "SampleScore",
    "aggregate",
    "best_iou",
    "iou",
    "score_sample",
]
