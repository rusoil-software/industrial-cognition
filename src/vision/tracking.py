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
"""Follow detected objects between inferences with an OpenCV tracker.

The model is the slow stage: on CPU one OWLv2 pass costs seconds, so a detection
result is a snapshot of where objects were, not where they are. A tracker closes
that gap cheaply. It is seeded once from a detection (``seed``), then advanced on
every frame the capture loop already has (``update``), so the console can draw boxes
that move at camera rate while the boxes themselves are refreshed by the next
inference.

Why not run inference more often? Because the cost is the model, not the drawing;
tracking buys motion fidelity for milliseconds per frame.

What the tracker is *not*: it does not decide what an object is. It follows a box
until the next inference replaces it, and it is dropped after ``max_missing`` frames
without a fresh detection so a stale box cannot drift forever.

Seeding uses the newest frame rather than the frame that produced the detection.
The detection may be a second or two old on CPU; initialising the tracker on the
current frame with a slightly aged box loses a little accuracy but keeps every box
in the coordinate space of the frames the console is drawing on (the same reason
the tracker decodes the JPEG the client sees, not the raw capture).

Tracker choice follows the old proof of work in ``scripts/proof_of_work.py``
(CSRT/KCF/MOSSE) but defaults to KCF: CSRT is more accurate and several times more
expensive, and at 5-10 track updates per second on a shared CPU the budget matters.
Every tracker except MIL lives in OpenCV's ``contrib`` modules, which is why
``requirements/api.txt`` installs ``opencv-contrib-python-headless``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DEFAULT_TRACKER_TYPE",
    "TRACKER_TYPE_ENV",
    "Track",
    "TrackPool",
    "available_trackers",
    "resolve_tracker",
]

logger = logging.getLogger(__name__)

#: Environment variable that overrides the tracker. Read by the API runtime.
TRACKER_TYPE_ENV = "TRACKER_TYPE"
#: KCF: cheap and accurate enough for a console overlay; CSRT when quality wins.
DEFAULT_TRACKER_TYPE = "kcf"
#: Preference order when the requested one is unavailable.
FALLBACK_ORDER: Tuple[str, ...] = ("kcf", "csrt", "mil")
TRACKER_TYPES: Tuple[str, ...] = FALLBACK_ORDER

Box = Tuple[int, int, int, int]  # x, y, width, height in pixels


def _constructor(name: str) -> Optional[Callable[[], Any]]:
    """Return a zero-argument factory for one tracker, or ``None`` if absent.

    OpenCV 4.5 exposed ``cv2.TrackerKCF_create``; 4.6+ also offers
    ``cv2.TrackerKCF.create``. Both are tried so the module works across the
    versions the worker and the API image pin.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - cv2 is a hard dependency in practice
        return None

    legacy = {
        "kcf": "TrackerKCF_create",
        "csrt": "TrackerCSRT_create",
        "mil": "TrackerMIL_create",
    }[name]
    factory = getattr(cv2, legacy, None)
    if factory is not None:
        return factory
    modern = getattr(getattr(cv2, "Tracker" + name.upper(), None), "create", None)
    if modern is not None:
        return modern
    legacy_module = getattr(cv2, "legacy", None)
    if legacy_module is not None and name == "mil":
        return getattr(legacy_module, "TrackerMIL_create", None)
    return None


def available_trackers() -> Tuple[str, ...]:
    """Tracker names this OpenCV build can create, in preference order."""
    return tuple(name for name in FALLBACK_ORDER if _constructor(name) is not None)


def resolve_tracker(
    requested: Optional[str] = None,
) -> Tuple[Optional[Callable[[], Any]], Optional[str], str]:
    """Pick a tracker factory.

    Returns:
        ``(factory, name, reason)``. ``factory`` is ``None`` when the installed
        OpenCV has no tracker at all (plain ``opencv-python`` without contrib),
        in which case ``reason`` explains it for the operator rather than failing
        the pipeline - the video feed is still useful without overlay boxes.
    """
    wanted = (requested or DEFAULT_TRACKER_TYPE).strip().lower()
    available = available_trackers()
    if wanted in available:
        return _constructor(wanted), wanted, ""
    if not available:
        return (
            None,
            None,
            "no OpenCV tracker is available (install opencv-contrib-python-headless); "
            "boxes are drawn only on inference results",
        )
    fallback = available[0]
    reason = (
        f"tracker {wanted!r} is not available in this OpenCV build; using {fallback}"
        if wanted
        else f"using {fallback}"
    )
    if wanted not in TRACKER_TYPES:
        reason = f"{reason} (expected one of {', '.join(TRACKER_TYPES)})"
    return _constructor(fallback), fallback, reason


def _centre_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Distance between box centres, in pixels."""
    lx = left[0] + left[2] / 2.0
    ly = left[1] + left[3] / 2.0
    rx = right[0] + right[2] / 2.0
    ry = right[1] + right[3] / 2.0
    return float(np.hypot(lx - rx, ly - ry))


def _intersection_over_union(left: Sequence[float], right: Sequence[float]) -> float:
    """IoU of two ``(x, y, w, h)`` boxes; 0.0 when they do not overlap."""
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class Track:
    """One followed object: identity, last box, and the tracker that owns it."""

    track_id: int
    label: str
    confidence: float
    box: Box
    tracker: Any = field(repr=False)
    age: int = 0
    missed: int = 0

    def to_payload(self, width: int, height: int) -> Dict[str, Any]:
        """Shape the console already understands, plus the track identity.

        The geometry keys match `src/vision/detections.py::Detection.to_payload`
        so the overlay draws a track and an inference result with the same code.
        """
        x, y, w, h = self.box
        return {
            "track_id": self.track_id,
            "object_type": self.label,
            "confidence": round(float(self.confidence), 4),
            "x_norm": max(0.0, min(1.0, x / float(width))),
            "y_norm": max(0.0, min(1.0, y / float(height))),
            "width_norm": max(0.0, min(1.0, w / float(width))),
            "height_norm": max(0.0, min(1.0, h / float(height))),
            "age": self.age,
            "stale": self.missed > 0,
        }


class TrackPool:
    """A small pool of OpenCV trackers, seeded by detections.

    Thread-safe: seeding happens where results are collected and updates happen in
    the capture loop, and OpenCV tracker objects are not themselves thread-safe.

    Args:
        tracker_type: ``kcf``, ``csrt`` or ``mil``; ``None`` uses the default and
            falls back when the requested one is missing.
        iou_threshold: how much a fresh detection must overlap an existing track to
            be treated as the same object. Lower values keep identity across fast
            motion; higher values avoid two boxes on one object.
        match_distance: with no overlap at all, a detection whose centre is within this
            fraction of the frame diagonal from a track's centre is still the same
            object. Without it, an object that moved further than its own width between
            two inferences (seconds on CPU) gets a second tracker and ends up with two
            boxes once it stands still.
        max_missing: frames without a fresh detection before a track is dropped.
        max_tracks: hard cap on simultaneous trackers, by confidence.
    """

    def __init__(
        self,
        tracker_type: Optional[str] = None,
        *,
        iou_threshold: float = 0.3,
        match_distance: float = 0.25,
        max_missing: int = 5,
        max_tracks: int = 16,
        min_box_pixels: int = 8,
        factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.requested = (tracker_type or DEFAULT_TRACKER_TYPE).strip().lower()
        if factory is not None:
            self._factory: Optional[Callable[[], Any]] = factory
            self.tracker_name: Optional[str] = self.requested
            self.reason = ""
        else:
            self._factory, self.tracker_name, self.reason = resolve_tracker(tracker_type)
        self.iou_threshold = float(iou_threshold)
        self.match_distance = float(match_distance)
        self.max_missing = int(max_missing)
        self.max_tracks = int(max_tracks)
        self.min_box_pixels = int(min_box_pixels)
        self._tracks: List[Track] = []
        self._next_id = 1
        self._seeded = 0
        self._updated = 0
        self._lost = 0
        self._merged = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._factory is not None

    # -- seeding ----------------------------------------------------------- #

    def seed(self, image: Any, detections: Iterable[Any]) -> None:
        """(Re)initialise trackers from one inference result.

        Detections that overlap an existing track of the same label keep that
        track's identity and tracker; the rest start new trackers. Tracks without a
        matching detection are given ``max_missing`` frames of grace before they
        are dropped, so a brief miss does not blink the box off the screen.

        ``image`` may be a BGR array, a JPEG byte string, or an object with a
        ``.jpeg`` attribute (`CapturedFrame`).
        """
        if not self.enabled:
            return
        frame = self._decode(image)
        if frame is None:
            return
        height, width = frame.shape[:2]
        candidates = []
        for detection in detections or ():
            label = getattr(detection, "object_type", None)
            if label is None and isinstance(detection, dict):
                label = detection.get("object_type")
            confidence = getattr(detection, "confidence", None)
            if confidence is None and isinstance(detection, dict):
                confidence = detection.get("confidence", 0.0)
            box = self._box_from_detection(detection, width, height)
            if box is None or label is None:
                continue
            candidates.append((str(label), float(confidence or 0.0), box))
        # Confident detections win the limited track slots. An empty result is still a
        # seed: it is how a track learns that the model no longer sees its object.
        candidates.sort(key=lambda item: item[1], reverse=True)

        with self._lock:
            self._seeded += 1
            free = list(self._tracks)
            fresh: List[Track] = []
            gate = self.match_distance * float(np.hypot(width, height))
            for label, confidence, box in candidates:
                best: Optional[Track] = None
                best_overlap = 0.0
                for track in free:
                    if track.label != label:
                        continue
                    overlap = _intersection_over_union(track.box, box)
                    if overlap >= self.iou_threshold and overlap > best_overlap:
                        best, best_overlap = track, overlap
                if best is None and gate > 0.0:
                    # No overlap: the object may simply have moved further than its own
                    # width between two inferences, which is normal when inference takes
                    # seconds. Nearest centre within the gate keeps one track instead of
                    # leaving the old one to expire next to a new one.
                    nearest = None
                    nearest_distance = gate
                    for track in free:
                        if track.label != label:
                            continue
                        distance = _centre_distance(track.box, box)
                        if distance < nearest_distance:
                            best, nearest_distance = track, distance
                if best is not None:
                    free.remove(best)
                    best.confidence = confidence
                    best.missed = 0
                    # A matched track keeps its identity, its tracker and its place.
                    fresh.append(best)
                    continue
                if len(free) + len(fresh) >= self.max_tracks:
                    continue
                track = self._start(label, confidence, box, frame)
                if track is not None:
                    fresh.append(track)
            for track in free:
                track.missed += 1
                if track.missed > self.max_missing:
                    self._lost += 1
                else:
                    fresh.append(track)
            self._tracks = fresh

    def _start(
        self, label: str, confidence: float, box: Box, frame: np.ndarray
    ) -> Optional[Track]:
        try:
            tracker = self._factory() if self._factory is not None else None
        except Exception as exc:  # noqa: BLE001 - a broken backend must not stop capture
            logger.warning("cannot create tracker %s: %s", self.tracker_name, exc)
            self._factory = None
            self.reason = f"the tracker backend failed: {exc}"
            return None
        if tracker is None:
            return None
        try:
            initialised = tracker.init(frame, tuple(int(value) for value in box))
        except Exception as exc:  # noqa: BLE001 - cv2 raises on bad boxes
            logger.debug("tracker init failed for %s: %s", label, exc)
            return None
        if initialised is False:
            return None
        track = Track(
            track_id=self._next_id,
            label=label,
            confidence=confidence,
            box=box,
            tracker=tracker,
        )
        self._next_id += 1
        logger.debug("track %s follows a %s at %s", track.track_id, label, box)
        return track

    # -- following --------------------------------------------------------- #

    def update(self, image: Any) -> List[Dict[str, Any]]:
        """Advance every tracker one frame and return the boxes as payload dicts."""
        if not self.enabled:
            return []
        frame = self._decode(image)
        if frame is None:
            return []
        height, width = frame.shape[:2]
        with self._lock:
            alive: List[Track] = []
            for track in self._tracks:
                try:
                    ok, rect = track.tracker.update(frame)
                except Exception as exc:  # noqa: BLE001 - a lost tracker is data, not an error
                    logger.debug("tracker %s failed to update: %s", track.track_id, exc)
                    ok, rect = False, None
                if ok and rect is not None:
                    track.box = self._clamp(rect, width, height)
                    track.age += 1
                    track.missed = 0
                else:
                    track.missed += 1
                if track.missed > self.max_missing:
                    self._lost += 1
                    continue
                alive.append(track)
            self._tracks = self._collapse_duplicates(alive)
            results = [track.to_payload(width, height) for track in self._tracks]
            self._updated += 1
        return results

    def _collapse_duplicates(self, tracks: List[Track]) -> List[Track]:
        """Drop a track that follows the same object as an older track.

        Two trackers on one object are worse than a missing box: the operator cannot
        tell which one to trust. The older id survives, because the console label is the
        identity an operator has been reading for the last few seconds.
        """
        kept: List[Track] = []
        for track in sorted(tracks, key=lambda item: item.track_id):
            duplicate = next(
                (
                    other
                    for other in kept
                    if other.label == track.label
                    and _intersection_over_union(other.box, track.box) >= 0.6
                ),
                None,
            )
            if duplicate is None:
                kept.append(track)
            else:
                self._merged += 1
                logger.debug(
                    "track %s collapsed into %s: same %s",
                    track.track_id,
                    duplicate.track_id,
                    track.label,
                )
        return kept

    def snapshot(self) -> Dict[str, Any]:
        """Status for ``/status``: what is following what, and whether it works."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "tracker": self.tracker_name,
                "requested": self.requested,
                "reason": self.reason or None,
                "tracks": len(self._tracks),
                "seeded": self._seeded,
                "updated": self._updated,
                "lost": self._lost,
                "merged": self._merged,
                "match_distance": round(self.match_distance, 3),
            }

    def reset(self) -> None:
        """Forget every track, e.g. when capture moves to another camera."""
        with self._lock:
            self._tracks = []
            self._next_id = 1

    # -- geometry ---------------------------------------------------------- #

    @staticmethod
    def _decode(image: Any) -> Optional[np.ndarray]:
        if isinstance(image, np.ndarray):
            return image if image.size else None
        payload = image
        if hasattr(payload, "jpeg"):
            payload = payload.jpeg
        if isinstance(payload, (bytes, bytearray, memoryview)):
            import cv2

            buffer = np.frombuffer(bytes(payload), dtype=np.uint8)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            return frame if frame is not None and frame.size else None
        return None

    def _box_from_detection(
        self, detection: Any, width: int, height: int
    ) -> Optional[Box]:
        values = {}
        for key in ("x_norm", "y_norm", "width_norm", "height_norm"):
            value = getattr(detection, key, None)
            if value is None and isinstance(detection, dict):
                value = detection.get(key)
            if value is None:
                return None
            values[key] = float(value)
        x1 = int(round(values["x_norm"] * width))
        y1 = int(round(values["y_norm"] * height))
        box_width = int(round(values["width_norm"] * width))
        box_height = int(round(values["height_norm"] * height))
        if box_width < self.min_box_pixels or box_height < self.min_box_pixels:
            return None
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        box_width = min(box_width, width - x1)
        box_height = min(box_height, height - y1)
        if box_width < self.min_box_pixels or box_height < self.min_box_pixels:
            return None
        return (x1, y1, box_width, box_height)

    @staticmethod
    def _clamp(rect: Sequence[float], width: int, height: int) -> Box:
        x, y, box_width, box_height = (int(round(float(value))) for value in rect)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        box_width = max(1, min(box_width, width - x))
        box_height = max(1, min(box_height, height - y))
        return (x, y, box_width, box_height)
