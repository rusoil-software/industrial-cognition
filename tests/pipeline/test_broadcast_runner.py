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
"""Tests for real-time fan-out and the capture loop.

The broadcaster's contract is about what must *not* happen: a slow subscriber must
never block the producer or grow without bound. Those are asserted directly, not
inferred from throughput.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.camera.capture import CaptureConfig  # noqa: E402
from src.pipeline.broadcast import ResultBroadcaster  # noqa: E402
from src.pipeline.dispatcher import InProcessTransport  # noqa: E402
from src.pipeline.runner import PipelineRunner, RunnerConfig  # noqa: E402
from src.vision.backend import SyntheticBackend  # noqa: E402
from src.vision.worker import VisionRuntimeConfig, VisionWorker  # noqa: E402


# --------------------------------------------------------------------------- #
# Broadcaster
# --------------------------------------------------------------------------- #
def test_broadcaster_fans_out_to_every_subscriber() -> None:
    broadcaster = ResultBroadcaster()
    first, second = broadcaster.subscribe(), broadcaster.subscribe()

    broadcaster.publish({"frame_number": 1})

    assert first.poll() == {"frame_number": 1}
    assert second.poll() == {"frame_number": 1}
    assert broadcaster.subscriber_count == 2


def test_slow_subscriber_drops_the_oldest_and_never_blocks() -> None:
    """The property that protects the capture loop from a lagging dashboard."""
    broadcaster = ResultBroadcaster(max_queue_per_subscriber=3)
    slow = broadcaster.subscribe()

    for frame in range(10):
        broadcaster.publish({"frame_number": frame})

    assert slow.depth == 3
    assert slow.dropped == 7
    # What remains is the newest window, not the oldest.
    assert [slow.poll()["frame_number"] for _ in range(3)] == [7, 8, 9]


def test_publish_does_not_block_on_a_full_subscriber() -> None:
    broadcaster = ResultBroadcaster(max_queue_per_subscriber=1)
    broadcaster.subscribe()

    started = time.perf_counter()
    for frame in range(1000):
        broadcaster.publish({"frame_number": frame})
    elapsed = time.perf_counter() - started

    # 1000 publishes against a never-drained subscriber must be effectively free.
    assert elapsed < 0.5


def test_unsubscribe_stops_delivery() -> None:
    broadcaster = ResultBroadcaster()
    subscription = broadcaster.subscribe()
    broadcaster.unsubscribe(subscription)

    broadcaster.publish({"frame_number": 1})

    assert broadcaster.subscriber_count == 0
    assert subscription.poll() is None
    assert subscription.closed is True


def test_poll_on_an_empty_subscription_returns_none() -> None:
    assert ResultBroadcaster().subscribe().poll() is None


def test_broadcast_stats_report_backlog() -> None:
    broadcaster = ResultBroadcaster(max_queue_per_subscriber=2)
    broadcaster.subscribe()
    for frame in range(5):
        broadcaster.publish({"frame_number": frame})

    stats = broadcaster.stats()
    assert stats["published"] == 5
    assert stats["subscribers"] == 1
    assert stats["dropped_per_subscriber"] == [3]
    assert stats["depth_per_subscriber"] == [2]


def test_broadcaster_rejects_a_zero_queue() -> None:
    with pytest.raises(ValueError):
        ResultBroadcaster(max_queue_per_subscriber=0)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class FrameSourceStub:
    """A synthetic source that can be made to fail opening."""

    def __init__(self, *, fail_open: bool = False, size: int = 64) -> None:
        self._open = False
        self._fail_open = fail_open
        self._size = size
        self.released = False

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> bool:
        if self._fail_open:
            return False
        self._open = True
        return True

    def read(self):
        if not self._open:
            return None
        frame = np.full((self._size, self._size, 3), 90, dtype=np.uint8)
        cv2.rectangle(frame, (8, 8), (24, 24), (210, 205, 200), -1)
        return frame

    def release(self) -> None:
        self._open = False
        self.released = True


class RecordingBackend(SyntheticBackend):
    """Records the text query each infer call received."""

    def __init__(self) -> None:
        super().__init__(num_queries=8, num_detections=1, score=0.9)
        self.queries: list[object] = []

    def infer(self, pixel_values, text_query=None):
        self.queries.append(text_query)
        return super().infer(pixel_values)


class PromptStub:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.value = None

    def __call__(self, camera_id: str):
        self.calls.append(camera_id)
        return self.value


def build_runner(**overrides):
    backend = RecordingBackend()
    worker = VisionWorker(backend, VisionRuntimeConfig(), worker_id="test")
    prompt_provider = overrides.pop("prompt_provider", None)
    config = RunnerConfig(
        capture=CaptureConfig(camera_id="cam0", encode_max_side=64, jpeg_quality=60),
        target_fps=overrides.pop("target_fps", 90),
        prompt_refresh_seconds=overrides.pop("prompt_refresh_seconds", 0.05),
        dispatch_interval_seconds=overrides.pop("dispatch_interval_seconds", 0.0),
        max_in_flight=overrides.pop("max_in_flight", 32),
    )
    runner = PipelineRunner(
        overrides.pop("source", FrameSourceStub()),
        InProcessTransport(worker.process_payload),
        config,
        prompt_provider=prompt_provider,
        **overrides,
    )
    return runner, backend


def test_runner_captures_analyses_and_publishes() -> None:
    runner, _ = build_runner()
    subscription = runner.broadcaster.subscribe()

    assert runner.start() is True
    try:
        time.sleep(0.4)
    finally:
        runner.stop()

    snapshot = runner.snapshot()
    assert snapshot["pipeline"]["counters"]["captured"] > 0
    assert snapshot["pipeline"]["counters"]["completed"] > 0
    assert snapshot["pipeline"]["counters"]["failed"] == 0
    assert snapshot["broadcast"]["published"] > 0
    assert subscription.poll() is not None


def test_published_envelope_matches_the_api_schema() -> None:
    runner, _ = build_runner()
    subscription = runner.broadcaster.subscribe()

    runner.start()
    time.sleep(0.3)
    runner.stop()

    payload = subscription.poll()
    assert payload is not None
    assert set(payload["metadata"]) == {"camera_id", "frame_number", "timestamp"}
    assert isinstance(payload["detections"], list)


def test_runner_fails_to_start_when_the_source_cannot_open() -> None:
    runner, _ = build_runner(source=FrameSourceStub(fail_open=True))
    assert runner.start() is False
    assert runner.running is False


def test_runner_releases_the_source_on_stop() -> None:
    source = FrameSourceStub()
    runner, _ = build_runner(source=source)
    runner.start()
    time.sleep(0.15)
    runner.stop()
    assert source.released is True
    assert runner.running is False


def test_runner_loads_the_camera_prompt_and_passes_it_to_the_model() -> None:
    """A stored prompt must reach inference, not just the database."""
    from tests.vision.test_prompt import FakeTokenizer
    from src.vision.prompt import encode_prompt_set

    provider = PromptStub()
    provider.value = encode_prompt_set(
        [("metal sheet", 0.25)], FakeTokenizer(), camera_id="cam0"
    )
    runner, backend = build_runner(prompt_provider=provider)

    runner.start()
    time.sleep(0.35)
    runner.stop()

    assert provider.calls, "the prompt provider was never consulted"
    assert set(provider.calls) == {"cam0"}
    assert any(query is not None for query in backend.queries), (
        "the prompt never reached the backend"
    )
    # The cache must prevent a per-frame database read.
    assert len(provider.calls) < runner.snapshot()["pipeline"]["counters"]["captured"]


def test_runner_survives_a_failing_prompt_provider() -> None:
    def boom(camera_id: str):
        raise RuntimeError("database is down")

    runner, _ = build_runner(prompt_provider=boom)
    runner.start()
    time.sleep(0.3)
    runner.stop()

    snapshot = runner.snapshot()
    assert snapshot["prompt_errors"] >= 1
    # Capture continued despite the provider failing.
    assert snapshot["pipeline"]["counters"]["captured"] > 0


def test_runner_rejects_a_double_start() -> None:
    runner, _ = build_runner()
    runner.start()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            runner.start()
    finally:
        runner.stop()


def test_runner_loop_is_a_daemon_thread() -> None:
    """A wedged loop must not prevent interpreter shutdown."""
    runner, _ = build_runner()
    runner.start()
    try:
        assert runner.running
        assert any(
            thread.daemon
            for thread in threading.enumerate()
            if thread.name.startswith("capture-")
        )
    finally:
        runner.stop()


def test_a_dispatch_interval_samples_frames_instead_of_dropping_them() -> None:
    """Capture must stay fast for the preview while workers see one frame at a time.

    Five frames in a row within a one-second interval are captured (so the console is
    live) but only the first is offered to the workers; the rest are counted as
    deliberate sampling, which is a different fact from the in-flight window refusing
    them.
    """
    runner, _ = build_runner(dispatch_interval_seconds=60.0)
    runner.start()
    try:
        time.sleep(0.4)
    finally:
        runner.stop()

    counters = runner.snapshot()["pipeline"]["counters"]
    assert counters["captured"] >= 2
    assert counters["dispatched"] == 1
    assert counters["shed_sampling"] >= 1
    assert counters["dropped_capacity"] == 0


def test_zero_dispatch_interval_offers_every_frame() -> None:
    runner, _ = build_runner(dispatch_interval_seconds=0.0)
    runner.start()
    try:
        time.sleep(0.4)
    finally:
        runner.stop()

    counters = runner.snapshot()["pipeline"]["counters"]
    assert counters["dispatched"] >= 2
    assert counters["shed_sampling"] == 0


def test_sampling_keeps_capacity_drops_truthful() -> None:
    """A frame refused *after* sampling is a real shortfall, not "sampling".

    The two counters answer different questions: ``shed_sampling`` is "this frame was
    never meant for the workers", ``dropped_capacity`` is "the workers could not take what
    they were offered". Sampling must not quietly excuse the second - a pipeline that is
    genuinely too slow has to keep looking too slow.
    """

    class StuckHandle:
        task_id = "stuck"

        def ready(self) -> bool:  # never finishes: the window stays full
            return False

        def result(self):
            return {}

    class StuckTransport:
        def submit(self, payload):
            return StuckHandle()

        def collect(self):
            return []

    config = RunnerConfig(
        capture=CaptureConfig(camera_id="cam0", encode_max_side=64, jpeg_quality=60),
        target_fps=90,
        dispatch_interval_seconds=0.01,
        max_in_flight=1,
    )
    runner = PipelineRunner(FrameSourceStub(), StuckTransport(), config)
    runner.start()
    try:
        time.sleep(0.4)
    finally:
        runner.stop()

    counters = runner.snapshot()["pipeline"]["counters"]
    assert counters["dispatched"] == 1
    assert counters["dropped_capacity"] >= 1


