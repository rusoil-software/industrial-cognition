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
"""The 60 FPS acceptance test.

This is the test the Stage 2 requirement is judged on, so it asserts the property
in both directions:

* with **enough** worker capacity, the pipeline delivers ~60 analysed frames per
  second end to end, and
* with **insufficient** capacity, it sheds frames and keeps latency bounded
  instead of quietly building an ever-growing queue.

The worker here is the real one (`VisionWorker` -> decode -> preprocess -> decode),
with only the model replaced by the synthetic backend, whose per-frame cost is
emulated with ``simulate_latency_ms`` exactly as a GPU worker's would be. The
"worker pool" is a thread pool standing in for N Celery workers; the semantics the
dispatcher relies on (submit, poll, non-blocking result) are identical to the
Celery transport's.

Timing tests are inherently approximate, so the thresholds are deliberately loose
(95% of target, a bounded-but-generous latency ceiling) while still failing on the
regressions that matter: throughput collapsing, or latency growing without bound.
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Callable

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.camera.capture import CaptureConfig, capture_frame, encode_jpeg  # noqa: E402
from src.camera.frames import CapturedFrame  # noqa: E402
from src.pipeline.dispatcher import FrameDispatcher  # noqa: E402
from src.pipeline.stats import PipelineStats  # noqa: E402
from src.vision.backend import SyntheticBackend  # noqa: E402
from src.vision.worker import VisionRuntimeConfig, VisionWorker  # noqa: E402

TARGET_FPS = 60


class FutureHandle:
    """A `TaskHandle` backed by a `concurrent.futures.Future`."""

    def __init__(self, task_id: str, future: "concurrent.futures.Future[dict]") -> None:
        self._task_id = task_id
        self._future = future

    @property
    def task_id(self) -> str:
        return self._task_id

    def ready(self) -> bool:
        return self._future.done()

    def result(self) -> dict:
        return self._future.result()


class PoolTransport:
    """Executes tasks on a fixed-size thread pool - a stand-in for N workers.

    ``workers`` bounds concurrency exactly as ``celery --concurrency`` and the pod
    replica count do together, which is the variable the capacity assertion is
    about.
    """

    name = "pool"

    def __init__(self, callable_: Callable[[dict], dict], workers: int) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="vision-worker"
        )
        self._callable = callable_
        self._counter = 0

    def submit(self, payload: dict[str, Any]) -> FutureHandle:
        self._counter += 1
        future = self._executor.submit(self._callable, payload)
        return FutureHandle(f"pool-{self._counter}", future)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


def small_jpeg(size: int = 960, value: int = 100) -> bytes:
    """A frame at the model's input size.

    Encoding at 960x960 makes the preprocessing resize a no-op, so the measurement
    targets orchestration rather than OpenCV's resize cost. The same bytes are
    reused for every frame, which is what a static workcell looks like anyway.
    """
    frame = np.full((size, size, 3), value, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    assert ok
    return buffer.tobytes()


def run_pipeline(
    *,
    workers: int,
    simulate_latency_ms: float,
    max_in_flight: int,
    duration_s: float = 1.0,
    capture_fps: int = TARGET_FPS,
) -> tuple[PipelineStats, dict[str, Any]]:
    """Drive capture -> dispatch -> collect at `capture_fps` for `duration_s`."""
    backend = SyntheticBackend(
        num_queries=16,
        num_detections=2,
        score=0.9,
        simulate_latency_ms=simulate_latency_ms,
    )
    worker = VisionWorker(
        backend,
        VisionRuntimeConfig(labels=("metal_sheet",), score_threshold=0.25),
        worker_id="test",
    )
    stats = PipelineStats(camera_id="cam0", window_seconds=duration_s)
    transport = PoolTransport(worker.process_payload, workers=workers)
    dispatcher = FrameDispatcher(transport, stats, max_in_flight=max_in_flight)

    jpeg = small_jpeg()
    frame_number = 0
    max_observed_in_flight = 0

    try:
        started = time.perf_counter()
        frame_interval = 1.0 / capture_fps
        next_at = started
        total_results = 0

        while time.perf_counter() - started < duration_s:
            now = time.perf_counter()
            if now < next_at:
                # Sleep, never spin. A Python busy-wait holds the GIL, which
                # starves the worker threads and makes the whole measurement
                # meaningless - it produced ~380 ms latency for a 5 ms task before
                # this was fixed.
                remaining = next_at - now
                if remaining > 0.0005:
                    time.sleep(remaining)
                else:
                    while time.perf_counter() < next_at:
                        pass

            frame = CapturedFrame(
                camera_id="cam0",
                frame_number=frame_number,
                capture_ts_ns=time.time_ns(),
                jpeg=jpeg,
                width=960,
                height=960,
            )
            stats.record_capture()
            dispatcher.dispatch(frame)
            total_results += len(dispatcher.collect())

            max_observed_in_flight = max(max_observed_in_flight, dispatcher.in_flight)
            frame_number += 1
            next_at += frame_interval

        total_results += len(dispatcher.drain(timeout_ms=4000))
    finally:
        transport.shutdown()

    info = {
        "frames_submitted": frame_number,
        "results": total_results,
        "max_in_flight": max_observed_in_flight,
    }
    return stats, info


# --------------------------------------------------------------------------- #
# Requirement: 60 FPS when capacity is adequate
# --------------------------------------------------------------------------- #
def test_pipeline_sustains_60_fps_with_adequate_worker_capacity() -> None:
    """12 workers at ~5 ms/frame is ample headroom for 60 FPS."""
    duration = 1.0
    stats, info = run_pipeline(
        workers=12, simulate_latency_ms=5.0, max_in_flight=256, duration_s=duration
    )
    snapshot = stats.snapshot()
    counters = snapshot["counters"]

    # Counts rather than an instantaneous rate: the rate over a short window is
    # bursty, whereas "60 frames captured and 60 analysed in one second" is the
    # requirement itself.
    assert counters["captured"] >= TARGET_FPS * duration * 0.8, snapshot
    assert counters["completed"] >= TARGET_FPS * duration * 0.9, (
        f"analysed only {counters['completed']} frames of a 60 FPS requirement: "
        f"{snapshot}"
    )
    # Nothing was shed and nothing failed: capacity was never the constraint.
    assert counters["dropped_capacity"] == 0, snapshot
    assert counters["failed"] == 0, snapshot
    assert snapshot["rates"]["completion_ratio"] >= 0.95, snapshot
    assert info["max_in_flight"] <= 256


def test_latency_stays_well_under_the_200ms_budget_when_capacity_is_adequate() -> None:
    """`doc/vision.md` sets a sub-200 ms end-to-end target."""
    stats, _ = run_pipeline(
        workers=12, simulate_latency_ms=5.0, max_in_flight=256, duration_s=1.0
    )
    latency = stats.snapshot()["latency_ms"]["end_to_end"]
    assert latency["p99"] < 200.0, latency


# --------------------------------------------------------------------------- #
# Safety property: shortfall sheds load, it does not queue forever
# --------------------------------------------------------------------------- #
def test_insufficient_capacity_sheds_frames_instead_of_growing_latency() -> None:
    """2 workers at 50 ms each is ~40 FPS of capacity against 60 FPS of demand.

    A naive dispatcher would queue every frame and show 40 FPS with latency
    climbing without limit. The bounded window must instead drop frames and keep
    latency flat. The worst-case wait is bounded by
    ``(max_in_flight / workers + 1) x per_frame_time`` = (6/2 + 1) x 50 ms.
    """
    stats, info = run_pipeline(
        workers=2, simulate_latency_ms=50.0, max_in_flight=6, duration_s=1.0
    )
    snapshot = stats.snapshot()

    assert snapshot["counters"]["dropped_capacity"] > 0, (
        "a capacity shortfall must shed frames rather than queue them: "
        f"{snapshot}"
    )
    assert snapshot["counters"]["completed"] < snapshot["counters"]["captured"]
    # The in-flight window is the hard bound that keeps latency flat.
    assert info["max_in_flight"] <= 6

    latency = snapshot["latency_ms"]["end_to_end"]
    assert latency["p99"] < 400.0, (
        f"latency grew beyond the bounded window's guarantee: {latency}"
    )


def test_capacity_shortfall_is_visible_in_the_completion_ratio() -> None:
    """Capturing 60 while analysing ~40 must not read as "60 FPS achieved"."""
    stats, _ = run_pipeline(
        workers=2, simulate_latency_ms=50.0, max_in_flight=6, duration_s=1.0
    )
    snapshot = stats.snapshot()

    assert snapshot["counters"]["completed"] < snapshot["counters"]["captured"]
    assert stats.meets_target(TARGET_FPS) is False, snapshot


def test_scaling_workers_raises_throughput() -> None:
    """The documented lever for meeting 60 FPS is more workers, so verify it.

    Both cases use a 50 ms per-frame cost, so capacity is 40 FPS with 2 workers
    (below the 60 FPS demand) and 200 FPS with 10. The first case must therefore
    complete fewer frames, and both must shed or complete consistently. Using
    25 ms with 2 workers was a mistake in an earlier revision of this test: 80 FPS
    of capacity already exceeds the demand, so nothing was constrained.
    """
    scarce, _ = run_pipeline(
        workers=2, simulate_latency_ms=50.0, max_in_flight=16, duration_s=0.8
    )
    ample, _ = run_pipeline(
        workers=10, simulate_latency_ms=50.0, max_in_flight=16, duration_s=0.8
    )

    scarce_snapshot = scarce.snapshot()["counters"]
    ample_snapshot = ample.snapshot()["counters"]

    assert scarce_snapshot["dropped_capacity"] > 0, (
        "2 workers at 50 ms/frame cannot keep up with 60 FPS, so frames must be shed"
    )
    assert ample_snapshot["completed"] > scarce_snapshot["completed"], (
        scarce_snapshot["completed"],
        ample_snapshot["completed"],
    )
    # With ample capacity there is no reason to shed.
    assert ample_snapshot["dropped_capacity"] == 0, ample_snapshot


# --------------------------------------------------------------------------- #
# Capture-side helpers used by the loop above
# --------------------------------------------------------------------------- #
def test_capture_frame_encodes_and_stamps_a_frame() -> None:
    class OneFrame:
        is_open = True

        def __init__(self) -> None:
            self._frame = np.zeros((240, 320, 3), dtype=np.uint8)

        def read(self):
            return self._frame

    config = CaptureConfig(camera_id="cam0", encode_max_side=960, jpeg_quality=70)
    captured = capture_frame(OneFrame(), config, frame_number=5)

    assert captured is not None
    assert captured.camera_id == "cam0"
    assert captured.frame_number == 5
    assert captured.jpeg.startswith(b"\xff\xd8")  # JPEG SOI
    assert (captured.width, captured.height) == (320, 240)
    assert captured.encode_ms >= 0
    assert captured.size_bytes > 0


def test_encode_jpeg_downscales_to_the_long_edge_cap() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    data, width, height = encode_jpeg(frame, quality=70, max_side=960)

    assert max(width, height) == 960
    # 16:9 preserved.
    assert height == pytest.approx(540, abs=1)
    assert data.startswith(b"\xff\xd8")


def test_encode_jpeg_does_not_upscale_a_small_frame() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    _, width, height = encode_jpeg(frame, max_side=960)
    assert (width, height) == (160, 120)


def test_encode_jpeg_rejects_an_empty_frame() -> None:
    with pytest.raises(ValueError):
        encode_jpeg(np.empty((0, 0, 3), dtype=np.uint8))
