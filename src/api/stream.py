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
"""Frame and camera endpoints for the operator UI.

Two deliberate choices:

* **MJPEG for video.** ``GET /cameras/{id}/stream`` is
  ``multipart/x-mixed-replace`` - the browser renders it in a plain ``<img>`` with no
  JavaScript decoding, no WebRTC stack and no HLS segmenting. At the preview rates this
  UI needs (a few frames per second) it is the smallest thing that works, and it costs
  one streaming response per viewer.
* **Detections stay on the WebSocket.** Boxes arrive on ``/ws/detections`` already
  carrying ``metadata.frame_number``, so the UI can draw them over the MJPEG image
  without re-sending coordinates inside the video stream.

The snapshot endpoint exists for the UI's "freeze frame" and for smoke tests: it returns
the most recent JPEG the pipeline captured, which is exactly what the last inference saw.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session

from src.api import runtime
from src.api.prompts import service as prompt_service
from src.api.prompts.dependencies import get_db
from src.api.storage import camera_repository

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Cameras"])

#: Preview pacing. The capture loop may run at 60 FPS; pushing 60 JPEGs per second into
#: a browser tab would waste bandwidth and GPU-less CPU for no visual benefit. Ten is
#: still visibly smooth for an operator watching a conveyor, and a fifth of the bandwidth
#: of the capture rate. STREAM_FPS overrides it (1-30).
STREAM_FPS = max(1, min(30, int(os.getenv("STREAM_FPS", "10") or 10)))
BOUNDARY = "frame"
#: How long a preview may go without a new frame before the stream closes. A silently
#: frozen image is the worst outcome - it looks live. Ending the response makes the
#: browser fire `onerror`, and the console retries while showing "no live frames".
STREAM_IDLE_TIMEOUT = max(2.0, float(os.getenv("STREAM_IDLE_TIMEOUT", "15") or 15))


@router.get("/cameras", summary="Cameras, their configured queries and their state")
def list_cameras(session: Session = Depends(get_db)) -> dict:
    """Every camera the console should show.

    The list is the union of three sources of truth that can each know about a camera the others do
    not: the registry (an operator registered it), the prompt table (it was configured before the
    registry existed) and the running pipeline (it is capturing, however it was configured). Merging
    them means an operator never sees a camera disappear from the console because it was configured
    through a different door.
    """
    runner = runtime.get_runner()
    configured = {prompt.camera_id: prompt for prompt in prompt_service.list_camera_prompts(session)}
    registered = {camera.id: camera for camera in camera_repository.list_cameras(session)}

    camera_ids = set(configured) | set(registered)
    if runner is not None:
        camera_ids.add(runner.stats.camera_id)

    cameras = []
    for camera_id in sorted(camera_ids):
        prompt_set = configured.get(camera_id)
        running = runner is not None and runner.stats.camera_id == camera_id
        row = registered.get(camera_id)
        entry = (
            camera_repository.public_view(row, running=running)
            if row is not None
            else {
                "camera_id": camera_id,
                "label": camera_id,
                "source": None,
                "kind": "unknown",
                "username": None,
                "has_credentials": False,
                "active": False,
                "running": running,
            }
        )
        entry["prompts"] = (
            [
                {"label": query.label, "threshold": query.threshold}
                for query in prompt_set.queries
            ]
            if prompt_set
            else []
        )
        entry["capture_fps"] = (
            runner.stats.snapshot()["rates"]["capture_fps"] if running else 0.0
        )
        # Freshness per camera: the console shows a stale badge instead of an old frame
        # that looks live. Zero/None means "no frame yet", not "healthy".
        latest = runner.snapshot().get("latest_frame") if running else None
        entry["last_frame_age_seconds"] = latest["age_seconds"] if latest else None
        entry["streaming"] = bool(running and latest is not None)
        cameras.append(entry)
    return {"cameras": cameras, "count": len(cameras)}


def _latest(camera_id: str):
    runner = runtime.get_runner()
    if runner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the capture pipeline is not running in this deployment",
        )
    frame = runner.latest_frame()
    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no frame has been captured yet",
        )
    return frame


@router.get(
    "/cameras/{camera_id}/snapshot",
    summary="The most recent captured frame as JPEG",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
)
def snapshot(camera_id: str) -> Response:
    """The exact image the last inference ran on."""
    frame = _latest(camera_id)
    return Response(
        content=frame.jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Frame-Number": str(frame.frame_number),
        },
    )


@router.get(
    "/cameras/{camera_id}/stream",
    summary="Live preview as multipart/x-mixed-replace (MJPEG)",
    response_class=StreamingResponse,
)
async def stream(camera_id: str) -> StreamingResponse:
    """MJPEG preview; the browser renders it inside an <img> tag."""
    # Fail fast with a real status code rather than opening a stream that never sends.
    _latest(camera_id)

    async def frames():
        last_number = -1
        last_sent_at = time.monotonic()
        interval = 1.0 / STREAM_FPS
        while True:
            runner = runtime.get_runner()
            frame = runner.latest_frame() if runner is not None else None
            if frame is not None and frame.frame_number != last_number:
                last_number = frame.frame_number
                last_sent_at = time.monotonic()
                yield (
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame.jpeg)}\r\n"
                    f"X-Frame-Number: {frame.frame_number}\r\n\r\n"
                ).encode("ascii") + frame.jpeg + b"\r\n"
            elif time.monotonic() - last_sent_at > STREAM_IDLE_TIMEOUT:
                # Capture stopped (camera unplugged, pod restarting, source error). Close
                # the response so the browser shows the failure instead of the last frame
                # forever; the console reopens it and keeps the stale badge honest.
                logger.info(
                    "closing the preview for camera %s: no new frame for %.0fs",
                    camera_id,
                    STREAM_IDLE_TIMEOUT,
                )
                return
            await asyncio.sleep(interval)

    return StreamingResponse(
        frames(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/cameras/{camera_id}/stats", summary="Pipeline state for one camera")
def camera_stats(camera_id: str) -> dict:
    runner = runtime.get_runner()
    if runner is None:
        return {"camera_id": camera_id, "running": False, "reason": "pipeline not running"}
    if runner.stats.camera_id != camera_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"camera {camera_id!r} is not the one being captured",
        )
    return {"camera_id": camera_id, "server_time_ns": time.time_ns(), **runner.snapshot()}
