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
"""Process-level holder for the vision pipeline runner.

Kept separate from `main.py` so the WebSocket route and the lifespan handler can
share one runner without importing each other, and so the API can be started with
the pipeline disabled (``CAMERA_ENABLED=0``) - useful when capture runs in its own
deployment, which is what the K8s `camera` component does.

Startup must never be fatal on a missing camera. The previous bootstrap raised from
the startup handler when Modbus, the model or the camera was absent, which is why
the API pod crash-looped in the cluster: a service that owns an HTTP surface and a
detection pipeline should serve the former and report the latter as degraded.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

__all__ = [
    "capture_config",
    "get_runner",
    "pipeline_status",
    "registry_status",
    "restart_pipeline",
    "set_runner",
    "shutdown_pipeline",
    "start_error",
    "start_pipeline",
]

logger = logging.getLogger(__name__)

_RUNNER: Optional[Any] = None
_START_ERROR: Optional[str] = None
_SUPERVISOR: Optional[Any] = None


def _registry_watch_enabled() -> bool:
    return os.getenv("CAMERA_REGISTRY_WATCH", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "",
    )


def _registry_activate(source: str, camera_id: str) -> bool:
    """Restart capture on the registry's camera. Called from the supervisor thread."""
    return restart_pipeline(source=source, camera_id=camera_id)


def _start_registry_watch() -> None:
    """Follow the camera registry, when this process is the capture process.

    Opt-in (`CAMERA_REGISTRY_WATCH=1`): the API pod in production is not the capture pod,
    and a second process restarting capture would fight the real one over the device.
    """
    global _SUPERVISOR
    if _SUPERVISOR is not None:
        return
    from src.camera.credentials import CredentialVault
    from src.camera.supervisor import RegistrySupervisor, registry_scan

    vault = CredentialVault.from_environment()
    poll_seconds = float(os.getenv("CAMERA_REGISTRY_POLL_SECONDS", "5") or 5)
    supervisor = RegistrySupervisor(
        scan=lambda: registry_scan(vault),
        on_activate=_registry_activate,
        poll_seconds=poll_seconds,
        current=lambda: (pipeline_status().get("pipeline") or {}).get("camera_id"),
    )
    supervisor.start()
    _SUPERVISOR = supervisor
    logger.info("camera registry watch enabled (poll every %.1fs)", poll_seconds)


def registry_status() -> dict:
    """Snapshot of the registry watcher for /status."""
    supervisor = _SUPERVISOR
    if supervisor is None:
        return {"running": False, "enabled": _registry_watch_enabled()}
    return {**supervisor.snapshot(), "enabled": True}


def get_runner():
    """The running pipeline, or ``None`` when capture is not active."""
    return _RUNNER


def set_runner(runner) -> None:  # noqa: ANN001 - PipelineRunner
    global _RUNNER
    _RUNNER = runner


def _capture_config(
    source: Optional[str] = None, camera_id: Optional[str] = None
):  # noqa: ANN202
    from src.camera.capture import CaptureConfig

    resolved_source = source if source is not None else os.getenv("CAMERA_SOURCE", "0")
    return CaptureConfig(
        camera_id=camera_id or os.getenv("CAMERA_ID", "cam0"),
        source=(
            int(resolved_source) if resolved_source.isdigit() else resolved_source
        ),
        width=int(os.getenv("CAMERA_WIDTH", "1920")),
        height=int(os.getenv("CAMERA_HEIGHT", "1080")),
        # Device negotiation wants whole frames per second; the pipeline target
        # below accepts fractions (0.2 = one frame every five seconds).
        fps=max(1, int(round(float(os.getenv("CAMERA_FPS", "60"))))),
        jpeg_quality=int(os.getenv("CAMERA_JPEG_QUALITY", "80")),
        encode_max_side=int(os.getenv("CAMERA_ENCODE_MAX_SIDE", "960")),
    )


def capture_config():  # noqa: ANN202
    """The capture configuration the pipeline would use now.

    Public so the camera probe (`src/camera/probe.py`) opens a device with the same resolution and
    FPS hints as capture - a probe that negotiates differently can claim a camera works while the
    pipeline then fails to open it.
    """
    return _capture_config()


def _write_detections(result) -> int:
    """Persist one frame's detections using a short-lived session.

    A session per write, like the prompt provider: the capture thread has no request scope
    and holding a connection open between frames (seconds apart) would idle in the pool for
    no reason.
    """
    from src.api.prompts.dependencies import get_session_factory
    from src.api.storage import detection_repository

    session = get_session_factory()()
    try:
        rows = detection_repository.save_detections(
            session,
            camera_id=result.camera_id,
            frame_id=result.frame_number,
            detections=result.detections,
            context={
                "worker_id": result.worker_id,
                "queue_ms": round(result.queue_ms, 3),
                "e2e_latency_ms": round(result.e2e_latency_ms, 3),
                "inference_ms": round(result.timings.inference_ms, 3),
            },
        )
        session.commit()
        return rows
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _build_sink():
    """The persistence policy, or ``None`` when disabled.

    Downsampled on purpose: 60 FPS per camera is not a write rate a database should absorb,
    and most frames say the same thing. PERSIST_MIN_INTERVAL_SECONDS bounds the rate
    independently of capture rate; see `src/api/detection_sink.py`.
    """
    if os.getenv("PERSIST_DETECTIONS", "1").strip().lower() in ("0", "false", "no"):
        logger.info("detection persistence disabled by PERSIST_DETECTIONS=0")
        return None

    from src.api.detection_sink import DetectionSink

    return DetectionSink(
        writer=_write_detections,
        min_interval_seconds=float(os.getenv("PERSIST_MIN_INTERVAL_SECONDS", "1.0")),
        only_with_detections=os.getenv("PERSIST_ONLY_DETECTIONS", "1").strip().lower()
        not in ("0", "false", "no"),
    )


def _prompt_provider(camera_id: str):
    """Read one camera's stored prompt using a short-lived session.

    Called by the runner every few seconds, not per frame. Errors propagate to the
    runner, which keeps the last known prompt and logs once.
    """
    from src.api.prompts.dependencies import get_session_factory
    from src.api.prompts import service

    session = get_session_factory()()
    try:
        return service.get_camera_prompt(session, camera_id)
    finally:
        session.close()


def _build_tracker():
    """The box tracker, or ``None`` when disabled or unavailable.

    Tracking is a console aid: it keeps boxes on moving objects between inferences.
    It is deliberately non-fatal - a build without opencv-contrib, or a bad
    ``TRACKER_TYPE``, yields ``None`` and the overlay falls back to inference boxes.
    """
    if os.getenv("TRACKING_ENABLED", "1").strip().lower() in ("0", "false", "no"):
        logger.info("box tracking disabled by TRACKING_ENABLED=0")
        return None
    try:
        from src.vision.tracking import TrackPool

        pool = TrackPool(os.getenv("TRACKER_TYPE") or None)
    except Exception:  # noqa: BLE001 - the overlay must never block capture
        logger.exception("box tracking is unavailable; continuing without it")
        return None
    if pool.enabled:
        logger.info("tracking boxes with %s", pool.tracker_name)
    elif pool.reason:
        logger.warning("box tracking is disabled: %s", pool.reason)
    return pool


def start_pipeline(
    source: Optional[str] = None, camera_id: Optional[str] = None
) -> bool:
    """Build and start the runner. Returns whether capture is running.

    Never raises: a missing camera, model or tokenizer degrades the service rather
    than preventing it from starting.

    ``source`` and ``camera_id`` override the environment defaults so the console can
    switch the service to a registered camera (`POST /cameras/{id}/activate`) without a
    redeploy. Leaving them ``None`` keeps the overlay's ``CAMERA_SOURCE`` in force.
    """
    global _START_ERROR

    if os.getenv("CAMERA_ENABLED", "1").strip().lower() in ("0", "false", "no"):
        _START_ERROR = "disabled by CAMERA_ENABLED"
        logger.info("vision pipeline disabled by configuration")
        return False

    try:
        from src.camera.sources import build_frame_source
        from src.pipeline.runner import PipelineRunner, RunnerConfig
        from src.pipeline.celery_transport import CeleryTransport

        capture = _capture_config(source=source, camera_id=camera_id)
        config = RunnerConfig(
            capture=capture,
            target_fps=float(os.getenv("CAMERA_FPS", "60")),
            max_in_flight=int(os.getenv("PIPELINE_MAX_IN_FLIGHT", "64")),
            # Match the dispatch rate to the workers instead of to the preview: with
            # 11 s CPU inference, offering every captured frame only fills the window.
            dispatch_interval_seconds=float(
                os.getenv("PIPELINE_DISPATCH_INTERVAL_SECONDS", "0") or 0
            ),
        )
        runner = PipelineRunner(
            build_frame_source(config.capture.source, config.capture),
            CeleryTransport(),
            config,
            prompt_provider=_prompt_provider,
            sink=_build_sink(),
            tracker=_build_tracker(),
            track_fps=float(os.getenv("TRACK_FPS", "10")),
        )
        if not runner.start():
            _START_ERROR = "capture source could not be opened"
            logger.warning(
                "vision pipeline not started: %s (the API will still serve)",
                _START_ERROR,
            )
            return False

        set_runner(runner)
        _START_ERROR = None
        if _registry_watch_enabled():
            try:
                _start_registry_watch()
            except Exception:  # noqa: BLE001 - the registry is optional, capture is not
                logger.exception("camera registry watch could not be started")
        return True
    except Exception as exc:  # noqa: BLE001 - startup must not be fatal
        _START_ERROR = f"{type(exc).__name__}: {exc}"
        logger.exception("vision pipeline failed to start; continuing without it")
        return False


def _stop_registry_watch() -> None:
    global _SUPERVISOR
    supervisor, _SUPERVISOR = _SUPERVISOR, None
    if supervisor is not None:
        supervisor.stop()


def shutdown_pipeline(timeout: float = 10.0) -> None:
    """Stop the runner if it is running."""
    global _RUNNER
    _stop_registry_watch()
    runner = _RUNNER
    if runner is not None:
        try:
            runner.stop(timeout=timeout)
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.exception("error while stopping the vision pipeline")
        finally:
            _RUNNER = None


def restart_pipeline(
    source: Optional[str] = None,
    camera_id: Optional[str] = None,
    timeout: float = 10.0,
) -> bool:
    """Stop whatever is capturing and start again - on *source* when one is given.

    Stopping first is not optional: one process owns one capture device, and a device that allows
    a single open (most USB cameras) would otherwise refuse the new one with "device busy". The
    stop is bounded by ``timeout``, so a wedged capture thread cannot block the HTTP request
    forever.

    Returns whether capture is running afterwards; ``start_error()`` says why when it is not.
    """
    global _START_ERROR

    shutdown_pipeline(timeout=timeout)
    _START_ERROR = None
    started = start_pipeline(source=source, camera_id=camera_id)
    if started:
        logger.info(
            "capture restarted on camera %s (source %s)",
            camera_id or os.getenv("CAMERA_ID", "cam0"),
            "registered source" if source is not None else os.getenv("CAMERA_SOURCE", "0"),
        )
    return started


def start_error() -> Optional[str]:
    """Why the pipeline is not running, or ``None`` when it is (or was never asked to)."""
    return _START_ERROR


def pipeline_status() -> dict:
    """Snapshot for /health and /status."""
    runner = _RUNNER
    if runner is None:
        return {
            "running": False,
            "reason": _START_ERROR or "not started",
            "broadcast": {"subscribers": 0, "published": 0},
        }
    return runner.snapshot()
