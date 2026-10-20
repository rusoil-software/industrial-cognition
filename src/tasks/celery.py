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
"""Celery application and the vision task.

This module is deliberately thin. All detection logic lives in
`src/vision/worker.py`, which has no Celery import, so it can be unit-tested
without a broker. Here we only:

* build the app from the environment (docker-compose defaults preserved),
* declare the queue contract shared with the deployment manifests, and
* expose one task, :func:`process_frame`, that delegates to the worker.

The worker is the unit of horizontal scaling. To reach the 60 FPS aggregate target
the deployment runs several in parallel (``-c`` concurrency within a pod, several
pod replicas), because a single OWLv2 forward pass is far slower than a frame
interval. `doc/vision-pipeline.md` §5 sizes the pool from measured latency.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from celery import Celery
from celery.signals import worker_process_init, worker_ready
from kombu import Exchange, Queue

logger = logging.getLogger(__name__)

# Connection settings come from the environment so that the same image runs
# unchanged under docker-compose (service names `rabbitmq`/`redis`) and under
# Kubernetes (ConfigMap/Secret-provided URLs - see k8s/base/app-config.yaml).
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://myuser:mypassword@rabbitmq:5672//")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

#: Dead-letter exchange: expired work is routed here instead of vanishing silently.
DEAD_LETTER_EXCHANGE = "vision.dlx"
DEAD_LETTER_QUEUE = "vision.dlq"


def _queue_arguments() -> dict:
    """Broker-side limits for the work queue.

    The dispatcher already sheds frames when its in-flight window is full, but that only
    bounds *this* process. If no worker is consuming - all replicas busy, a crash loop, a
    scale-to-zero - the broker keeps accepting and the queue grows without limit. That is
    not hypothetical: a 41-message backlog produced `queue_ms` of 353 s, and every one of
    those frames was worthless by the time it was analysed.

    Two independent bounds, both cheap:

    * ``x-max-length`` with ``x-overflow=drop-head`` keeps the newest N frames and discards
      the oldest when full. For a real-time control loop that is the correct direction -
      the newest frame is the one worth analysing.
    * ``x-message-ttl`` expires anything older than T seconds, so a queue that drains after
      a long stall does not deliver stale work. Expired messages are dead-lettered rather
      than dropped, so the volume is visible instead of invisible.

    Note that queue arguments are immutable in RabbitMQ: changing these values does not
    alter an existing queue. `doc/k8s.md` documents the one-time recreation, and
    `scripts/queue-hygiene.sh` performs it.
    """
    max_length = int(os.getenv("CELERY_QUEUE_MAX_LENGTH", "256"))
    ttl_seconds = int(os.getenv("CELERY_QUEUE_TTL_SECONDS", "60"))
    return {
        "x-max-length": max_length,
        "x-overflow": "drop-head",
        "x-message-ttl": ttl_seconds * 1000,
        "x-dead-letter-exchange": DEAD_LETTER_EXCHANGE,
        "x-dead-letter-routing-key": DEAD_LETTER_QUEUE,
    }


#: Queue contract shared with the deployment manifests: the work queue, the
#: priority queue for urgent frames, the results queue and the dead-letter queue.
#:
#: These must be `kombu.Queue` instances, not bare strings. Celery builds
#: `self.Queues(self.app.conf.task_queues)` and then does
#: `{q.name: q for q in queues}`; a string has no `.name`, so the worker dies with
#: `AttributeError: 'str' object has no attribute 'name'` during `setup_queues`,
#: before it contacts the broker at all.
CELERY_TASK_QUEUES = [
    Queue(name)
    for name in (
        queue.strip()
        for queue in os.getenv(
            "CELERY_QUEUES", "vision,vision.priority,vision.results,vision.dlq"
        ).split(",")
    )
    if name
]

# The work queues carry the bounds; the dead-letter queue must not, or expired messages
# would be dropped instead of retained for inspection.
CELERY_TASK_QUEUES = [
    Queue(
        queue.name,
        queue_arguments=_queue_arguments()
        if queue.name in ("vision", "vision.priority")
        else None,
    )
    for queue in CELERY_TASK_QUEUES
]

# The DLQ is declared against its own exchange so dead-lettered frames land somewhere that
# can be listed, inspected and purged, rather than on the floor.
CELERY_TASK_QUEUES.append(
    Queue(
        DEAD_LETTER_QUEUE,
        exchange=Exchange(DEAD_LETTER_EXCHANGE, type="direct"),
        routing_key=DEAD_LETTER_QUEUE,
        durable=True,
    )
)

celery_app = Celery(
    "tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_queues=tuple(CELERY_TASK_QUEUES),
    task_default_queue=os.getenv("CELERY_TASK_DEFAULT_QUEUE", "vision"),
    task_create_missing_queues=True,
    # Route the real task explicitly so it cannot land on the results queue.
    task_routes={"vision.process_frame": {"queue": "vision"}},
    # Acknowledge only after the result exists: a worker eviction re-queues the
    # frame instead of losing it.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # One in-flight task per worker process: bounded GPU memory, and the broker
    # queue depth then reflects real backlog, which is what an autoscaler reads.
    worker_prefetch_multiplier=int(
        os.getenv("CELERY_WORKER_PREFETCH_MULTIPLIER", "1")
    ),
    # JSON only. The payload is camera metadata plus a base64 JPEG; pickle would
    # turn the broker into a remote-code-execution surface.
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
    broker_connection_retry_on_startup=True,
)


@celery_app.task(
    name="vision.process_frame",
    bind=True,
    # No automatic retries: a frame that failed to decode will fail again, and a
    # retry competes with fresh frames for the same capacity. The dispatcher counts
    # the failure and the pipeline moves on; a dropped frame costs one frame, a
    # retry storm costs throughput.
    max_retries=0,
    acks_late=True,
)
def process_frame(self, payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
    """Analyse one captured frame.

    The worker import is deferred so that merely importing this module (for
    ``celery inspect`` or a CLI) does not pay OpenCV/ONNX Runtime import costs.
    """
    from src.vision.worker import process_frame_payload

    return process_frame_payload(payload)


@worker_process_init.connect
def _warm_vision_model(**_kwargs: Any) -> None:
    """Load the model in each prefork child as it starts, not on the first frame.

    Measured in-cluster, the first inference in a fresh process costs ~11.5 s of ONNX
    Runtime session setup on top of the frame itself (doc/vision-pipeline.md §13). On a
    cold autoscale-up that means the first frames after a scale-out event are delayed by
    the model load rather than by the work, which is exactly when latency matters.

    Doing it here also makes the cost visible: a worker that cannot load its model fails
    its startup probe instead of accepting tasks it cannot serve. Failures are logged
    and swallowed deliberately - a missing model should degrade the worker, not stop it
    from connecting to the broker, so the reason is visible in the pod status rather
    than in a crash loop.
    """
    if os.getenv("VISION_WARMUP", "1").strip().lower() in ("0", "false", "no"):
        logger.info("model warmup disabled by VISION_WARMUP")
        return
    try:
        from src.vision.worker import get_worker

        started = __import__("time").perf_counter()
        get_worker().warmup()
        logger.info(
            "vision backend warmed in %.1fs", __import__("time").perf_counter() - started
        )
    except Exception:  # noqa: BLE001 - see the docstring
        logger.exception(
            "vision backend warmup failed; tasks will fail until the model is available"
        )


@worker_ready.connect
def _declare_dead_letter_queue(**_kwargs: Any) -> None:
    """Create the dead-letter exchange and queue if they do not exist.

    Putting ``x-dead-letter-exchange`` on the work queue does not create the target.
    RabbitMQ accepts the argument and then drops messages when it tries to dead-letter into
    an exchange that is not there - a silent failure, which is the worst kind for a queue
    whose whole purpose is to make losses visible. Celery also never declares a queue it
    does not consume from, so the DLQ has to be declared explicitly.
    """
    try:
        with celery_app.connection_for_write() as connection:
            queue = celery_app.amqp.queues[DEAD_LETTER_QUEUE]
            queue.bind(connection.default_channel).declare()
        logger.info("dead-letter queue %s declared", DEAD_LETTER_QUEUE)
    except Exception:  # noqa: BLE001 - the worker must still start without a DLQ
        logger.exception(
            "could not declare the dead-letter queue; expired frames will be dropped "
            "instead of retained"
        )
