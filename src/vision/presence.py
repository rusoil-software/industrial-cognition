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
"""Turn "a flat metal thing is somewhere in this photo" into "the blank is on the table".

The reason this module exists is a measurement, not a theory. On the real station
photograph (`assets/images/metal_sheet_test_image.jpg`) the zero-shot model proposes the
workpiece *and* the table it lies on, and prefers the table:

    box                       centre          size            score (steel blank)
    the table (whole)         0.483, 0.588    0.590 x 0.421   0.2459
    the workpiece (inset)     0.494, 0.570    0.311 x 0.232   0.1594

Both are rectangles of flat metal, and the table is simply bigger and more sheet-like.
No prompt fixes that - "polished metal sheet" (0.1932), "reflective metal plate" (0.1200)
and "brushed steel plate" (0.1428) all rank the table first as well. So the fix is not
another phrase; it is geometry:

* the workpiece has a known **size** and **aspect ratio** at a fixed camera,
* it appears **inside a known region** of the frame (the station),
* the table, the floor and the doorway are outside those bounds.

A zero-shot model plus four numbers is a presence check. The numbers come from the
installation, not from the model: measure the blank in one photograph and write down
where it is. `tests/vision/test_presence.py` pins the selection rules; the live
measurement that motivated them is in `doc/vision-pipeline.md` §19.2.

Deliberately kept model-free: this is arithmetic over detected boxes, so it runs in
microseconds, is testable without a 293 MiB artifact, and cannot make the pipeline worse
when the model does something unexpected.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from src.vision.detections import Detection

logger = logging.getLogger(__name__)

__all__ = [
    "Box",
    "FULL_FRAME",
    "PresenceGate",
    "enabled",
    "from_environment",
    "parse_roi",
    "select_presence",
]

#: A normalised rectangle: `(x_min, y_min, x_max, y_max)`, each in [0, 1].
Box = Tuple[float, float, float, float]

#: The whole frame, used when the caller has no opinion about where the station is.
FULL_FRAME: Box = (0.0, 0.0, 1.0, 1.0)

#: Environment variable names, so a deployment can describe its station without a
#: schema change. `PRESENCE_ENABLED` is the switch: without it nothing is computed and
#: the pipeline behaves exactly as before.
ENABLED_ENV = "PRESENCE_ENABLED"
ROI_ENV = "PRESENCE_ROI"
MIN_AREA_ENV = "PRESENCE_MIN_AREA"
MAX_AREA_ENV = "PRESENCE_MAX_AREA"
MIN_ASPECT_ENV = "PRESENCE_MIN_ASPECT"
MAX_ASPECT_ENV = "PRESENCE_MAX_ASPECT"
MIN_CONFIDENCE_ENV = "PRESENCE_MIN_CONFIDENCE"
MIN_INSIDE_ENV = "PRESENCE_MIN_INSIDE"

_TRUTHY = ("1", "true", "yes", "on")


def enabled(value: Optional[str]) -> bool:
    """Read a boolean-ish environment value; anything unrecognised means off."""
    return str(value or "").strip().lower() in _TRUTHY


def parse_roi(text: str) -> Box:
    """Parse `"x1,y1,x2,y2"` into a validated normalised box.

    Raises :class:`ValueError` rather than guessing: a station region that silently
    becomes the whole frame turns the gate into "any sheet-like blob anywhere", which
    is the failure this module exists to prevent.
    """
    parts = [part.strip() for part in str(text).replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError(f"expected four numbers (x1,y1,x2,y2), got {text!r}")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"not a number in {text!r}") from exc
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"coordinates must be within 0..1, got {text!r}")
    if values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError(f"empty region: x1 < x2 and y1 < y2 are required, got {text!r}")
    return (values[0], values[1], values[2], values[3])



@dataclass(frozen=True)
class PresenceGate:
    """Which boxes are allowed to mean "the part is there".

    `roi` is the station region. `min_area`/`max_area` bound the box area as a fraction
    of the frame - a workpiece is small, the table it lies on is not. `min_aspect` and
    `max_aspect` bound width/height so that a tall fragment of racking cannot pass as a
    sheet. `min_confidence` is the calibrated threshold from
    `scripts/calibrate_presence.py`.

    Every default is deliberately loose: a gate that is too tight reports "no part" for
    a part that is there, which for a station means a stopped line, while a gate that is
    too loose reports a box on the table. The defaults sit at the loose end because the
    second failure is visible in the console and the first is not.
    """

    roi: Box = FULL_FRAME
    min_area: float = 0.01
    max_area: float = 0.25
    min_aspect: float = 0.5
    max_aspect: float = 4.0
    min_confidence: float = 0.0
    #: How much of the box must lie inside the region. 0.5 tolerates a part half over
    #: the edge of the station, which is exactly the interesting case.
    min_inside: float = 0.5

    def area(self, detection: Detection) -> float:
        return detection.width_norm * detection.height_norm

    def aspect(self, detection: Detection) -> float:
        if detection.height_norm <= 0.0:
            return float("inf")
        return detection.width_norm / detection.height_norm

    def inside(self, detection: Detection) -> float:
        """Fraction of the box that lies inside :attr:`roi`."""
        x1 = max(self.roi[0], detection.x_norm)
        y1 = max(self.roi[1], detection.y_norm)
        x2 = min(self.roi[2], detection.x_norm + detection.width_norm)
        y2 = min(self.roi[3], detection.y_norm + detection.height_norm)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        box_area = self.area(detection)
        if box_area <= 0.0:
            return 0.0
        return ((x2 - x1) * (y2 - y1)) / box_area

    def rejection(self, detection: Detection) -> Optional[str]:
        """Why this box cannot mean "the part is there", or None when it can."""
        if detection.confidence < self.min_confidence:
            return f"confidence {detection.confidence:.3f} < {self.min_confidence:.3f}"
        area = self.area(detection)
        if area < self.min_area:
            return f"area {area:.4f} < {self.min_area:.4f} (too small to be the part)"
        if area > self.max_area:
            return f"area {area:.4f} > {self.max_area:.4f} (a table, not the part)"
        aspect = self.aspect(detection)
        if aspect < self.min_aspect or aspect > self.max_aspect:
            return f"aspect {aspect:.2f} outside {self.min_aspect:.2f}..{self.max_aspect:.2f}"
        inside = self.inside(detection)
        if inside < self.min_inside:
            return f"only {inside:.0%} inside the station region"
        return None

    def accepts(self, detection: Detection) -> bool:
        return self.rejection(detection) is None


def from_environment(environ: Optional[Mapping[str, str]] = None) -> Optional[PresenceGate]:
    """Build the gate a deployment asked for, or None when presence is switched off.

    Raises :class:`ValueError` on unusable values instead of falling back to the full
    frame: a mistyped region that quietly becomes "anywhere in the picture" would make a
    station report a part whenever the floor looks metallic.
    """
    source = environ if environ is not None else os.environ
    if not enabled(source.get(ENABLED_ENV)):
        return None

    gate = PresenceGate()
    roi = source.get(ROI_ENV)
    if roi:
        gate = replace(gate, roi=parse_roi(roi))
    for name, field in (
        (MIN_AREA_ENV, "min_area"),
        (MAX_AREA_ENV, "max_area"),
        (MIN_ASPECT_ENV, "min_aspect"),
        (MAX_ASPECT_ENV, "max_aspect"),
        (MIN_CONFIDENCE_ENV, "min_confidence"),
        (MIN_INSIDE_ENV, "min_inside"),
    ):
        raw = source.get(name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} is not a number: {raw!r}") from exc
        gate = replace(gate, **{field: value})

    if gate.min_area < 0.0 or gate.max_area <= 0.0 or gate.min_area >= gate.max_area:
        raise ValueError(
            f"area bounds are empty: {MIN_AREA_ENV}={gate.min_area} "
            f"{MAX_AREA_ENV}={gate.max_area}"
        )
    if gate.min_aspect <= 0.0 or gate.max_aspect <= gate.min_aspect:
        raise ValueError(
            f"aspect bounds are empty: {MIN_ASPECT_ENV}={gate.min_aspect} "
            f"{MAX_ASPECT_ENV}={gate.max_aspect}"
        )
    if not 0.0 <= gate.min_inside <= 1.0:
        raise ValueError(f"{MIN_INSIDE_ENV} must be within 0..1, got {gate.min_inside}")
    return gate



def select_presence(
    detections: Iterable[Detection],
    gate: PresenceGate,
) -> Tuple[Optional[Detection], dict]:
    """Return the most confident acceptable box and a small report.

    The report is what a deployment can log: the chosen box, how many boxes were
    offered, and the reason the best-looking one was refused when nothing was chosen.
    Without that, "no part" and "the gate is misconfigured" look identical from the
    outside - and they are the two failures that matter.
    """
    offered: Sequence[Detection] = list(detections)
    accepted = [detection for detection in offered if gate.accepts(detection)]
    if accepted:
        best = max(accepted, key=lambda detection: detection.confidence)
        return best, {
            "present": True,
            "offered": len(offered),
            "accepted": len(accepted),
            "confidence": round(best.confidence, 4),
            "reason": None,
        }

    if offered:
        best_any = max(offered, key=lambda detection: detection.confidence)
        reason = gate.rejection(best_any)
    else:
        reason = "the model produced no box at all"
    return None, {
        "present": False,
        "offered": len(offered),
        "accepted": 0,
        "confidence": 0.0,
        "reason": reason,
    }
