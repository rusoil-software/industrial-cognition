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
"""Celery implementation of the dispatcher's `TaskTransport`.

Kept in its own module so that `src/pipeline/dispatcher.py` - and everything that
tests it - stays free of Celery and broker imports. Celery is imported lazily here,
so this module can even be imported in an environment where Celery is absent
(useful for type checking and for the API image, which does not install it).

The task is submitted by *name* (`vision.process_frame`) rather than by importing
the task object: the producer must not need the worker's dependency set (OpenCV,
ONNX Runtime) just to enqueue work.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

__all__ = ["CeleryHandle", "CeleryTransport"]

logger = logging.getLogger(__name__)

DEFAULT_TASK_NAME = "vision.process_frame"


class CeleryHandle:
    """`TaskHandle` backed by an `AsyncResult`.

    `ready()` and `result()` are non-blocking in the way the dispatcher requires:
    `AsyncResult.ready()` only checks local/backend state, and `result()` is called
    exclusively after `ready()` returned true.
    """

    def __init__(self, async_result: Any) -> None:
        self._async_result = async_result

    @property
    def task_id(self) -> str:
        return str(self._async_result.id)

    def ready(self) -> bool:
        return bool(self._async_result.ready())

    def result(self) -> dict[str, Any]:
        return self._async_result.result


class CeleryTransport:
    """Submits frame payloads to the configured Celery queue."""

    name = "celery"

    def __init__(
        self,
        *,
        task_name: str = DEFAULT_TASK_NAME,
        queue: Optional[str] = None,
        app: Any = None,
    ) -> None:
        if app is None:
            from src.tasks.celery import celery_app

            app = celery_app
        self._app = app
        self._task_name = task_name
        self._queue = queue
        # Fail at construction, not on the first frame: a missing task name is a
        # wiring error that should surface at startup.
        if task_name not in self._app.tasks:
            logger.warning(
                "task %r is not registered yet; the worker registers it on import",
                task_name,
            )

    @property
    def task_name(self) -> str:
        return self._task_name

    def submit(self, payload: dict[str, Any]) -> CeleryHandle:
        options: dict[str, Any] = {}
        if self._queue:
            options["queue"] = self._queue
        async_result = self._app.send_task(self._task_name, args=[payload], **options)
        return CeleryHandle(async_result)
