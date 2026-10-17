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

__all__ = ["PipelineRunner", "RunnerConfig"]

logger = logging.getLogger(__name__)

#: How often the camera's prompt is re-read. Short enough that "I saved the prompt
#: and it still uses the old one" is not a report anyone files; long enough that a
#: 60 FPS loop does not hammer the database.
DEFAULT_PROMPT_REFRESH_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    capture: CaptureConfig = CaptureConfig()
    target_fps: int = 60
    prompt_refresh_seconds: float = DEFAULT_PROMPT_REFRESH_SECONDS
    #: Maximum frames in flight before the dispatcher sheds load.
    max_in_flight: int = 64
    #: Stop waiting for in-flight work after this long on shutdown.
    drain_timeout_ms: float = 5000.0


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
    ) -> None:
        self._source = source
        self._config = config or RunnerConfig()
        self._broadcaster = broadcaster or ResultBroadcaster()
        self._prompt_provider = prompt_provider
        self._stats = PipelineStats(
            camera_id=self._config.capture.camera_id,
            window_seconds=max(1.0, float(self._config.target_fps)),
        )
        self._dispatcher = FrameDispatcher(
            transport,
            self._stats,
            max_in_flight=self._config.max_in_flight,
        )
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._prompt: Optional[Any] = None
        self._prompt_fetched_at = 0.0
        self._prompt_errors = 0

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
        self._source.release()

    # -- loop --------------------------------------------------------------
    def _run(self) -> None:
        config = self._config
        camera_id = config.capture.camera_id
        interval = 1.0 / max(1, config.target_fps)
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
                self._stats.record_capture()
                self._stats.record_encode(frame.encode_ms)
                prompt = self._current_prompt(camera_id)
                self._dispatcher.dispatch(frame, prompt=prompt)

            for result in self._dispatcher.collect():
                self._publish(result)

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

    # -- output ------------------------------------------------------------
    def _publish(self, result: Any) -> None:
        """Hand a result to the broadcaster in the API-shaped envelope."""
        try:
            self._broadcaster.publish(result.to_payload())
        except Exception:  # noqa: BLE001 - a consumer must not kill the loop
            logger.exception("failed to publish frame %s", getattr(result, "frame_number", "?"))

    # -- introspection -----------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Combined pipeline + fan-out state for /status and /health."""
        return {
            "running": self.running,
            "pipeline": self._stats.snapshot(),
            "broadcast": self._broadcaster.stats(),
            "in_flight": self._dispatcher.in_flight,
            "prompt_errors": self._prompt_errors,
            "prompt_loaded": self._prompt is not None,
            "server_time_ns": now_ns(),
        }
