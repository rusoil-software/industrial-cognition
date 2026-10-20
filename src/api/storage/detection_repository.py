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
# date   : 2026-Oct-20
# ==============================================================================
"""Persist detection results.

Until now detections existed only as WebSocket messages: `detection_results` and its ORM
were written in Stage 1 but nothing ever inserted a row, so there was no history for the
console or for Stage 3 analytics.

Two things this module is careful about:

* **The enum is narrower than the prompts.** `object_type` has four values; a camera can
  ask for "robot" or "person". `coerce_object_type` keeps the coarse category (unknown
  labels become `clutter`) while `label` stores what the operator actually typed. Writing
  the raw label into the enum column would raise, which is the failure this avoids.
* **Rows are written per detection, not per frame**, matching the table's shape (frame_id
  is deliberately not unique). One frame with four detections is four rows.

Callers own the transaction: this module flushes and returns a count, so a caller can batch
several frames into one commit.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.storage.models import DetectionResult, ObjectType

__all__ = [
    "coerce_object_type",
    "count_detections",
    "latest_detections",
    "save_detections",
]

logger = logging.getLogger(__name__)

#: Enum members by value, for the coercion below.
_OBJECT_TYPES = {member.value: member for member in ObjectType}


def coerce_object_type(label: str) -> ObjectType:
    """Map a prompt label onto the persisted enum.

    The enum predates free-form prompts, so anything outside it keeps the operator's label
    in the `label` column and is categorised as clutter rather than rejected. Losing the
    coarse category is a reporting inconvenience; losing the detection is data loss.
    """
    return _OBJECT_TYPES.get((label or "").strip().lower(), ObjectType.CLUTTER)


def save_detections(
    session: Session,
    *,
    camera_id: str,
    frame_id: int,
    detections: Iterable[Any],
    timestamp: Optional[datetime] = None,
    context: Optional[dict[str, Any]] = None,
) -> int:
    """Insert one row per detection; returns the number of rows written.

    `detections` are the worker's `Detection` objects (`object_type`, `x_norm`, `y_norm`,
    `width_norm`, `height_norm`, `confidence`). `context` is merged into the row's
    `metadata` and is the place for frame-level facts - worker id, timings, workspace
    state - that would otherwise be repeated per detection or lost.
    """
    written = 0
    when = timestamp or datetime.now(timezone.utc)

    for detection in detections:
        label = str(getattr(detection, "object_type", "") or "").strip()
        metadata: dict[str, Any] = {"camera_id": camera_id}
        if context:
            metadata.update(context)

        session.add(
            DetectionResult(
                timestamp=when,
                frame_id=int(frame_id),
                camera_id=camera_id,
                label=label or None,
                object_type=coerce_object_type(label),
                coordinates={
                    "x": float(detection.x_norm),
                    "y": float(detection.y_norm),
                    "width": float(detection.width_norm),
                    "height": float(detection.height_norm),
                },
                confidence=float(detection.confidence),
                detection_metadata=metadata,
            )
        )
        written += 1

    if written:
        session.flush()
    return written


def count_detections(session: Session, camera_id: Optional[str] = None) -> int:
    """Total persisted detections, optionally for one camera."""
    statement = select(DetectionResult)
    if camera_id is not None:
        statement = statement.where(DetectionResult.camera_id == camera_id)
    return len(session.scalars(statement).all())


def latest_detections(
    session: Session, *, camera_id: Optional[str] = None, limit: int = 20
) -> Sequence[DetectionResult]:
    """Most recent rows, newest first - what the console history view needs."""
    statement = select(DetectionResult).order_by(DetectionResult.timestamp.desc())
    if camera_id is not None:
        statement = statement.where(DetectionResult.camera_id == camera_id)
    return session.scalars(statement.limit(limit)).all()
