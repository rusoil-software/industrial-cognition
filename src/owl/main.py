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

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.owl.config import settings
from src.owl.exceptions import register_exception_handlers
from src.owl.inference.router import router as inference_router
from src.owl.inference.service import get_inference_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Handles startup and shutdown events.
    Loads the ONNX model once at startup, releases it on shutdown.
    This is the modern replacement for @app.on_event("startup").
    """
    # --- Startup ---
    service = get_inference_service()
    service.load()

    yield  # App is running and serving requests here

    # --- Shutdown ---
    service.unload()


def create_application() -> FastAPI:
    """
    Application factory pattern.
    Keeps the app object configurable and testable —
    you can call create_application() with different settings in tests.
    """
    app_configs: dict = {
        "title": "OWL 2 Inference Service",
        "description": "Open-vocabulary object detection via ONNX Runtime.",
        "version": settings.APP_VERSION,
        "lifespan": lifespan,
    }

    # Hide docs in production — only expose on local/staging
    if settings.ENVIRONMENT not in settings.SHOW_DOCS_ENVIRONMENT:
        app_configs["openapi_url"] = None

    app = FastAPI(**app_configs)

    register_exception_handlers(app)  # <-- add this line

    # --- Middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routers ---
    # Each domain registers its own router with its own prefix.
    # Adding more domains later is just one more include_router() call.
    app.include_router(inference_router, prefix="/api/v1")

    return app


app = create_application()
