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
"""Real-time detection stream: ``/ws/detections``.

The telemetry contract from `doc/vision.md` ("Real-Time Telemetry:
WebSocket/FastAPI"). The handler is a thin adapter over
:class:`~src.pipeline.broadcast.ResultBroadcaster`:

* it polls a **bounded** per-client buffer, so a client that stops reading cannot
  stall capture or grow server memory - the broadcaster drops the oldest results
  and reports how many;
* it yields between polls so a quiet camera does not spin the event loop;
* it always unsubscribes, including on a client disconnect, so a flapping dashboard
  cannot leak subscriptions.

Backpressure is intentionally *not* applied to the pipeline: slowing detection to
match the slowest client would make the robot's view depend on a browser tab.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api import runtime

logger = logging.getLogger(__name__)

router = APIRouter()

#: Poll interval while a subscription is empty. Small enough to feel live, large
#: enough that an idle socket costs nothing measurable.
IDLE_POLL_SECONDS = 0.01


@router.websocket("/ws/detections")
async def detections_stream(websocket: WebSocket) -> None:
    """Stream detection results as JSON envelopes.

    Each message has the shape of `src/camera/schemas.py::StreamingPayload`::

        {"metadata": {"camera_id", "frame_number", "timestamp"},
         "detections": [{"object_type", "x_norm", ...}],
         "timings": {...}, "worker_id": "..."}

    The first message is a hello frame describing the pipeline, so a client can
    distinguish "connected, nothing detected yet" from "connected but capture is
    not running".
    """
    await websocket.accept()

    runner = runtime.get_runner()
    if runner is None:
        await websocket.send_json(
            {"type": "unavailable", "detail": "vision pipeline is not running"}
        )
        await websocket.close(code=1011, reason="pipeline not running")
        return

    subscription = runner.broadcaster.subscribe()
    logger.info("websocket client subscribed to detections")

    try:
        await websocket.send_json(
            {
                "type": "hello",
                "camera_id": runner.stats.camera_id,
                "pipeline": runtime.pipeline_status(),
            }
        )

        while True:
            payload = subscription.poll()
            if payload is None:
                await asyncio.sleep(IDLE_POLL_SECONDS)
                continue
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        logger.info("websocket client disconnected")
    except Exception:  # noqa: BLE001 - a broken client must not break the server
        logger.exception("websocket stream failed")
    finally:
        runner.broadcaster.unsubscribe(subscription)
        logger.info("websocket client unsubscribed (dropped=%d)", subscription.dropped)
