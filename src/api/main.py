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
# date   : 2026-Apr-23
# ==============================================================================
"""FastAPI application entry point.

Surface (matching `doc/vision.md` "Presentation/API Layer"):

* ``GET  /health``  - liveness/readiness, plus whether the detection pipeline runs.
  ``status: ok`` is preserved so the Kubernetes probes keep working.
* ``GET  /status``  - pipeline, throughput and fan-out snapshot.
* ``WS   /ws/detections`` - real-time detection stream.
* ``PUT/GET/DELETE /cameras/{camera_id}/prompt`` - per-camera detection prompts.

Startup policy changed deliberately. The previous bootstrap called Modbus, the
camera and the model from the startup handler and raised when any was missing, so
the API pod crash-looped in the cluster: the process could not serve ``/health``
without physical hardware attached. A service that owns an HTTP surface must serve
it and report the pipeline as degraded, which is what
:func:`src.api.runtime.start_pipeline` now does - it never raises, and the reason a
pipeline is down is visible in ``/health``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api import runtime
from src.api.prompts.router import router as prompts_router
from src.api.auth_routes import guard as auth_guard
from src.api.auth_routes import router as auth_router
from src.api.auth import get_settings as auth_settings
from src.api.realtime import router as realtime_router
from src.api.stream import router as stream_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 - required signature
    """Start the pipeline if possible; always allow the API to serve."""
    settings = auth_settings()
    if settings.enabled and not settings.login_token:
        logger.warning(
            "AUTH_REQUIRED is on but AUTH_TOKEN is empty: the console cannot be "
            "logged into and every data endpoint will answer 401"
        )
    elif not settings.enabled:
        logger.warning(
            "AUTH_REQUIRED=0 - this deployment is UNAUTHENTICATED; do not expose it"
        )
    if runtime.start_pipeline():
        logger.info("vision pipeline started")
    else:
        logger.warning(
            "vision pipeline not started (%s); the API will serve with detection "
            "degraded",
            runtime.pipeline_status().get("reason"),
        )
    try:
        yield
    finally:
        runtime.shutdown_pipeline()


app = FastAPI(
    title="Industrial Cognition Core API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(prompts_router)
app.include_router(realtime_router)
app.include_router(stream_router)
app.include_router(auth_router)


@app.middleware("http")
async def require_session(request, call_next):
    """Guard every data endpoint unless a session is present.

    Public: /health (Kubernetes probes), /auth/* (how a session is obtained) and
    the static console shell (which carries no cell data). Everything else -
    cameras, frames, prompts, stats - needs the cookie, a bearer token or
    ?token=. A middleware rather than per-route dependencies so that a route
    added later is protected by default; forgetting a dependency is exactly how
    an endpoint ends up public.
    """
    from src.api.auth import is_protected
    from fastapi.responses import JSONResponse

    if request.url.path.startswith("/ws"):
        # WebSockets are not HTTP requests; the handler checks the session itself.
        return await call_next(request)
    if is_protected(request.url.path):
        try:
            auth_guard(request)
        except Exception:
            return JSONResponse(
                {"detail": "authentication required"}, status_code=401
            )
    return await call_next(request)

# Operator UI. Served by the API for the MVP so there is one deployment and no
# CORS surface; a separate static host can replace this mount later without the
# UI code changing (it only talks to relative URLs).
app.mount(
    "/ui",
    StaticFiles(directory=os.getenv("UI_DIR", "ui"), html=True),
    name="ui",
)


@app.get("/health", tags=["Health"])
def health() -> dict:
    """Liveness and readiness.

    Returns 200 whenever the process can answer, even with detection degraded: an
    unplugged camera must not make Kubernetes restart a healthy API. The pipeline's
    real state travels in ``pipeline`` so an operator, or a stricter probe, can act
    on it.
    """
    pipeline = runtime.pipeline_status()
    return {
        "status": "ok",
        "service": "API Gateway",
        "pipeline_running": bool(pipeline.get("running")),
        "pipeline": pipeline,
    }


@app.get("/status", tags=["Health"])
def status() -> dict:
    """Detailed pipeline, throughput and fan-out snapshot."""
    return runtime.pipeline_status()
