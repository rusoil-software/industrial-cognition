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
"""Decide which completed frames are worth persisting.

The queue produces results far faster than a database should absorb: a 60 FPS target means
60 frames per second per camera, each with zero to several detections. Writing every one is
not a feature, it is a disk-filling bug - and most of those frames say the same thing.

So persistence is **sampled**, and the policy lives here rather than inside the pipeline or
the repository:

* a failed frame is never persisted - the error is already counted and logged;
* a frame with no detections is skipped by default (an empty workcell is not history);
* at most one frame per camera per `min_interval_seconds`, which bounds the write rate
  independently of capture rate. 1 s at 60 FPS is a 60x reduction.

This is downsampling, not an audit log, and the distinction is deliberately written down:
an event that occurs for less than the interval can be missed. If a future requirement needs
every detection, it should be a separate stream (or an event bus), not a lower interval here
- the cost is linear in frame rate and unbounded in time.

The sink is called from the capture thread, so it never raises: a database outage must
degrade history, not stop detection. Failures are counted and logged once.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

__all__ = ["DetectionSink"]

logger = logging.getLogger(__name__)


@dataclass
class DetectionSink:
    """Sampling wrapper around a persistence writer.

    `writer` receives a `FrameResult` and returns the number of rows written. Keeping it a
    callable is what lets the pipeline stay free of SQLAlchemy: the API wires a
    session-per-write writer, and tests pass a counter.
    """

    writer: Callable[[Any], int]
    min_interval_seconds: float = 1.0
    only_with_detections: bool = True
    clock: Callable[[], float] = time.monotonic

    frames_written: int = 0
    rows_written: int = 0
    frames_skipped: int = 0
    errors: int = 0
    _last_write: dict[str, float] = field(default_factory=dict, repr=False)

    def __call__(self, result: Any) -> bool:
        """Persist *result* if the policy allows it. Returns whether it was written.

        Never raises: an exception from the writer is counted and logged, because this
        runs on the capture thread where an exception would stop detection.
        """
        if getattr(result, "error", None):
            self.frames_skipped += 1
            return False

        detections = getattr(result, "detections", ()) or ()
        if self.only_with_detections and not detections:
            self.frames_skipped += 1
            return False

        camera_id = getattr(result, "camera_id", "") or ""
        now = self.clock()
        last = self._last_write.get(camera_id)
        if last is not None and (now - last) < self.min_interval_seconds:
            self.frames_skipped += 1
            return False

        try:
            rows = int(self.writer(result))
        except Exception:  # noqa: BLE001 - see the class docstring
            self.errors += 1
            # Log the first failure loudly, then stay quiet: a database outage at one
            # attempt per interval would otherwise flood the log.
            if self.errors == 1:
                logger.exception(
                    "could not persist detections for camera %s; history is degrading "
                    "while detection continues",
                    camera_id,
                )
            return False

        self._last_write[camera_id] = now
        self.frames_written += 1
        self.rows_written += rows
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "frames_written": self.frames_written,
            "rows_written": self.rows_written,
            "frames_skipped": self.frames_skipped,
            "errors": self.errors,
            "min_interval_seconds": self.min_interval_seconds,
            "only_with_detections": self.only_with_detections,
        }
