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
"""The capture loop: source -> dispatcher -> broadcaster.

This ties the pieces together and is the one place that owns a thread. Keeping it
out of `src/api/main.py` matters: the loop is testable without FastAPI, and the API
process is not obliged to run it (capture can live in its own deployment, which is
what the K8s `camera` component is for).

Design points:

* **Pacing is best-effort.** The loop aims at the configured FPS and never spins to
  catch up: if a frame takes longer than the interval, the next one starts
  immediately. Spinning would burn a core and starve the timeout handling in the
  dispatcher.
* **The prompt is refreshed, not read per frame.** Resolving a camera's prompt from
  the database 60 times a second would be absurd; it is cached and re-read every
  ``prompt_refresh_seconds``, so a prompt change takes effect within seconds without
  a worker restart.
* **A bad frame never stops the loop.** Encode/decode failures are counted and the
  loop continues; only a failure to open the source is fatal.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.camera.capture import CaptureConfig, FrameSource, capture_frame
from src.camera.frames import now_ns
from src.pipeline.broadcast import ResultBroadcaster
from src.pipeline.dispatcher import FrameDispatcher
from src.pipeline.stats import PipelineStats

__all__ = ["LatestFrame", "PipelineRunner", "RunnerConfig"]

logger = logging.getLogger(__name__)

#: How often the camera's prompt is re-read. Short enough that "I saved the prompt
#: and it still uses the old one" is not a report anyone files; long enough that a
#: 60 FPS loop does not hammer the database.
DEFAULT_PROMPT_REFRESH_SECONDS = 5.0

#: How often the box trackers are advanced. Tracking is for the human watching the
#: overlay, not for the robot: at 5-10 updates a second a box lags the object by well
#: under a tenth of a second, and the alternative (tracking at 60 FPS) spends a whole
#: core on drawing. Decoupled from the capture rate for that reason.
DEFAULT_TRACK_FPS = 10.0


@dataclass(frozen=True, slots=True)
class LatestFrame:
    """The most recent encoded frame, kept so a browser can see what the model saw.

    Frames otherwise live only as a task payload: by the time a result comes back the
    JPEG has been consumed. Keeping the last one costs a few hundred kilobytes and is
    what makes an overlay UI (and "what did it actually look at?") possible.
    """

    jpeg: bytes
    frame_number: int
    capture_ts_ns: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    capture: CaptureConfig = CaptureConfig()
    # float: a CPU-only worker may only manage a frame every few seconds, and
    # "one frame per 5 s" is 0.2 FPS - an int would round that to zero.
    target_fps: float = 60.0
    prompt_refresh_seconds: float = DEFAULT_PROMPT_REFRESH_SECONDS
    #: Maximum frames in flight before the dispatcher sheds load.
    max_in_flight: int = 64
    #: Stop waiting for in-flight work after this long on shutdown.
    drain_timeout_ms: float = 5000.0
    #: A submitted frame that has not returned by now is assumed lost. Must be
    #: comfortably larger than one inference, or healthy slow frames are killed.
    task_timeout_ms: float = 180_000.0


class PipelineRunner:
    """Owns the capture thread for one camera."""

    def __init__(
        self,
        source: FrameSource,
        transport: Any,
        config: Optional[RunnerConfig] = None,
        *,
        broadcaster: Optional[ResultBroadcaster] = None,
        prompt_provider: Optional[Callable[[str], Optional[Any]]] = None,
        sink: Optional[Callable[[Any], bool]] = None,
        tracker: Optional[Any] = None,
        track_fps: float = DEFAULT_TRACK_FPS,
    ) -> None:
        self._source = source
        self._config = config or RunnerConfig()
        self._broadcaster = broadcaster or ResultBroadcaster()
        self._prompt_provider = prompt_provider
        # Called for every completed frame after fan-out: the persistence policy decides
        # whether it becomes a row. Injected as a callable so `src/pipeline` stays free of
        # SQLAlchemy, and optional so tests and the smoke scripts run without a database.
        self._sink = sink
        # Optional `src.vision.tracking.TrackPool`. Injected rather than constructed here
        # so `src/pipeline` does not depend on OpenCV, and so a deployment without
        # opencv-contrib simply runs without overlay tracks.
        self._tracker = tracker
        self._track_interval = 1.0 / max(0.1, float(track_fps))
        self._track_next = 0.0
        self._track_errors = 0
        self._had_tracks = False
        self._stats = PipelineStats(
            camera_id=self._config.capture.camera_id,
            window_seconds=max(1.0, float(self._config.target_fps)),
        )
        self._dispatcher = FrameDispatcher(
            transport,
            self._stats,
            max_in_flight=self._config.max_in_flight,
            task_timeout_ms=self._config.task_timeout_ms,
        )
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._prompt: Optional[Any] = None
        self._prompt_fetched_at = 0.0
        self._prompt_errors = 0
        # Counts unexpected failures inside the loop that were survived. A non-zero value
        # here with a healthy "running: true" means frames are being degraded - visible in
        # /status instead of only in the log.
        self._loop_errors = 0
        self._latest: Optional[LatestFrame] = None
        # Assignment of a whole immutable object is atomic enough: a reader sees
        # either the previous frame or the new one, never a half-written buffer.
        self._latest_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------
    @property
    def stats(self) -> PipelineStats:
        return self._stats

    @property
    def broadcaster(self) -> ResultBroadcaster:
        return self._broadcaster

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Open the source and start the loop.

        Returns:
            ``False`` when the source cannot be opened. The caller decides whether
            that is fatal - the API treats it as "no camera attached" and keeps
            serving the rest of its surface.

        Raises:
            RuntimeError: if the runner is already running.
        """
        if self.running:
            raise RuntimeError("pipeline runner is already running")
        if not self._source.open():
            logger.error(
                "could not open capture source for camera %s",
                self._config.capture.camera_id,
            )
            return False

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"capture-{self._config.capture.camera_id}",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the loop to stop, then drain in-flight frames."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        # Collect results that were already paid for, so a rolling restart does not
        # throw away finished work.
        for result in self._dispatcher.drain(
            timeout_ms=self._config.drain_timeout_ms
        ):
            self._publish(result)
        if self._tracker is not None:
            # The trackers hold OpenCV objects that belong to this camera; the next
            # camera starts from a fresh seed anyway.
            self._tracker.reset()
        self._source.release()

    # -- loop --------------------------------------------------------------
    def _run(self) -> None:
        config = self._config
        camera_id = config.capture.camera_id
        interval = 1.0 / max(0.01, float(config.target_fps))
        frame_number = 0
        next_at = time.perf_counter()
        logger.info("capture loop started for camera %s", camera_id)

        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_at:
                # Sleep rather than spin: a busy-wait holds the GIL and starves the
                # worker threads.
                self._stop.wait(min(next_at - now, interval))
                continue

            try:
                frame = capture_frame(self._source, config.capture, frame_number)
            except Exception:  # noqa: BLE001 - one bad frame must not stop capture
                self._stats.record_drop_encode()
                logger.exception("failed to capture frame %s", frame_number)
                frame = None

            if frame is not None:
                with self._latest_lock:
                    self._latest = LatestFrame(
                        jpeg=frame.jpeg,
                        frame_number=frame.frame_number,
                        capture_ts_ns=frame.capture_ts_ns,
                        width=frame.width,
                        height=frame.height,
                    )
                self._stats.record_capture()
                self._stats.record_encode(frame.encode_ms)

                # Follow the boxes the last inference produced, so the overlay moves with
                # the objects between detections. Guarded: a tracker backend failure
                # degrades the overlay, never the detection path.
                self._advance_tracks(frame, camera_id)

                # Dispatch runs inside a guard because an unexpected failure here used to
                # kill the capture thread outright - a non-dict task result did exactly
                # that, stopping detection silently while the pod stayed "Running".
                # Degrading one frame is always better than stopping the loop.
                try:
                    prompt = self._current_prompt(camera_id)
                    self._dispatcher.dispatch(frame, prompt=prompt)
                except Exception:  # noqa: BLE001 - see above
                    self._loop_errors += 1
                    logger.exception(
                        "frame %s could not be dispatched; continuing",
                        frame.frame_number,
                    )

            # Collection is guarded separately and runs even when capture failed, so a
            # result already paid for is never abandoned because of an unrelated error.
            try:
                for result in self._dispatcher.collect():
                    self._publish(result)
            except Exception:  # noqa: BLE001 - see above
                self._loop_errors += 1
                logger.exception("could not collect results; continuing")

            frame_number += 1
            next_at += interval
            # If we fell far behind, resynchronise instead of burning through a
            # backlog of missed ticks.
            if time.perf_counter() - next_at > interval:
                next_at = time.perf_counter() + interval

        logger.info("capture loop stopped for camera %s", camera_id)

    # -- prompts -----------------------------------------------------------
    def _current_prompt(self, camera_id: str) -> Optional[Any]:
        """Cached per-camera prompt; ``None`` when unset or unavailable."""
        if self._prompt_provider is None:
            return None

        now = time.monotonic()
        if now - self._prompt_fetched_at < self._config.prompt_refresh_seconds:
            return self._prompt

        self._prompt_fetched_at = now
        try:
            self._prompt = self._prompt_provider(camera_id)
        except Exception:  # noqa: BLE001 - a database blip must not stop capture
            self._prompt_errors += 1
            # Log the first failure loudly, then stay quiet: at 1 refresh per 5 s a
            # persistent outage would otherwise fill the log.
            if self._prompt_errors == 1:
                logger.exception(
                    "could not load the prompt for camera %s; continuing with the "
                    "last known prompt",
                    camera_id,
                )
        return self._prompt

    # -- tracking ----------------------------------------------------------
    def _tracking_enabled(self) -> bool:
        return self._tracker is not None and bool(getattr(self._tracker, "enabled", False))

    def _advance_tracks(self, frame: Any, camera_id: str) -> None:
        """Advance the box trackers, at most ``track_fps`` times a second."""
        if not self._tracking_enabled():
            return
        now = time.perf_counter()
        if now < self._track_next:
            return
        self._track_next = now + self._track_interval
        try:
            tracks = self._tracker.update(frame)
        except Exception:  # noqa: BLE001 - the overlay must not stop detection
            self._track_errors += 1
            if self._track_errors == 1:
                logger.exception("tracking failed; boxes stay on inference results")
            return
        # An empty update is published only when boxes were visible: that is the message
        # that clears them. Publishing "no tracks" ten times a second would be noise.
        if not tracks and not self._had_tracks:
            return
        self._had_tracks = bool(tracks)
        try:
            self._broadcaster.publish(
                {
                    "type": "tracks",
                    "camera_id": camera_id,
                    "frame_number": frame.frame_number,
                    "capture_ts_ns": frame.capture_ts_ns,
                    "tracks": tracks,
                }
            )
        except Exception:  # noqa: BLE001 - a consumer must not kill the loop
            logger.exception("failed to publish tracks for frame %s", frame.frame_number)

    def _seed_tracks(self, result: Any) -> None:
        """(Re)initialise trackers from one inference result on the newest frame."""
        if not self._tracking_enabled():
            return
        if getattr(result, "error", None):
            # A failed inference says nothing about where objects are; seeding on it
            # would move every box or expire every track.
            return
        latest = self.latest_frame()
        if latest is None:
            return
        try:
            self._tracker.seed(latest.jpeg, getattr(result, "detections", ()) or ())
        except Exception:  # noqa: BLE001 - see _advance_tracks
            self._track_errors += 1
            if self._track_errors == 1:
                logger.exception("could not seed the trackers from frame %s",
                                 getattr(result, "frame_number", "?"))

    # -- output ------------------------------------------------------------
    def _publish(self, result: Any) -> None:
        """Hand a result to the broadcaster, then offer it to the persistence sink."""
        try:
            self._broadcaster.publish(result.to_payload())
        except Exception:  # noqa: BLE001 - a consumer must not kill the loop
            logger.exception("failed to publish frame %s", getattr(result, "frame_number", "?"))

        self._seed_tracks(result)

        # Fan-out first and persistence second: a slow or failing database must not delay
        # the live view, which is the operator's primary signal. The sink enforces its own
        # sampling and never raises, so one lost row cannot stop the loop.
        if self._sink is not None:
            try:
                self._sink(result)
            except Exception:  # noqa: BLE001 - belt and braces around a third-party writer
                logger.exception(
                    "detection sink failed for frame %s",
                    getattr(result, "frame_number", "?"),
                )

    # -- introspection -----------------------------------------------------
    def latest_frame(self) -> Optional[LatestFrame]:
        """The most recent encoded frame, or ``None`` before the first one."""
        with self._latest_lock:
            return self._latest

    def snapshot(self) -> dict[str, Any]:
        """Combined pipeline + fan-out state for /status and /health."""
        latest = self.latest_frame()
        now = now_ns()
        # A frame is "late" relative to the cadence capture was asked for, not to a fixed
        # number of seconds: a 0.2 FPS dev replay is four frames per 20 seconds and must not
        # be reported as stalled. Three missed intervals, never less than five seconds.
        stale_after = max(5.0, 3.0 / max(0.01, float(self._config.target_fps)))
        return {
            "running": self.running,
            "pipeline": self._stats.snapshot(),
            # The age of the newest frame is the honest answer to "is the camera still
            # there?". Counters alone cannot tell a working pipeline from one whose
            # source went silent: captured stops growing in both cases, but a frozen
            # image looks live in the console until you compare timestamps.
            "latest_frame": (
                {
                    "frame_number": latest.frame_number,
                    "capture_ts_ns": latest.capture_ts_ns,
                    "width": latest.width,
                    "height": latest.height,
                    "age_seconds": max(0.0, (now - latest.capture_ts_ns) / 1_000_000_000),
                    "stale_after_seconds": round(stale_after, 3),
                }
                if latest is not None
                else None
            ),
            "broadcast": self._broadcaster.stats(),
            "in_flight": self._dispatcher.in_flight,
            "prompt_errors": self._prompt_errors,
            "loop_errors": self._loop_errors,
            "prompt_loaded": self._prompt is not None,
            "persistence": (
                self._sink.snapshot()
                if self._sink is not None and hasattr(self._sink, "snapshot")
                else {"enabled": False}
            ),
            "tracking": (
                self._tracker.snapshot()
                if self._tracking_enabled() and hasattr(self._tracker, "snapshot")
                else {"enabled": False}
            ),
            "track_errors": self._track_errors,
            "server_time_ns": now_ns(),
        }
