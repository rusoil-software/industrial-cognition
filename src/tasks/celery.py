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
# date   : 2026-Jun-31
# ==============================================================================


# Placeholder for Celery task definitions used by the vision worker container.
# This file stub allows the docker-compose build process to complete without
# failing on an undefined module import.

import os

from celery import Celery
from kombu import Queue

# Connection settings come from the environment so that the same image runs
# unchanged under docker-compose (service names `rabbitmq`/`redis`) and under
# Kubernetes (ConfigMap/Secret-provided URLs - see k8s/base/app-config.yaml).
#
# The literals are the docker-compose defaults and are kept so that the
# pre-existing local workflow continues to work with no environment set.
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://myuser:mypassword@rabbitmq:5672//")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

# Queue contract shared with the deployment manifests: the work queue, the
# priority queue used for urgent frames, the results queue and the dead-letter
# queue named in the architecture diagram.
#
# These must be `kombu.Queue` instances, not bare strings. Celery builds
# `self.Queues(self.app.conf.task_queues)` and then does
# `{q.name: q for q in queues}`; a string has no `.name`, so the worker dies with
# `AttributeError: 'str' object has no attribute 'name'` during `setup_queues`,
# before it contacts the broker at all.
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

# Initialize Celery app (broker/backend overridable via environment variables)
celery_app = Celery(
    'tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
    )

celery_app.conf.update(
    # Declare the queues on worker start so that a fresh broker does not need an
    # out-of-band `rabbitmqadmin` step before the first task is routed.
    task_queues=tuple(CELERY_TASK_QUEUES),
    task_default_queue=os.getenv("CELERY_TASK_DEFAULT_QUEUE", "vision"),
    task_create_missing_queues=True,
    # Exactly-once-ish delivery: acknowledge only after the inference result has
    # been persisted, so a worker eviction re-queues instead of losing the frame.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # One in-flight task per worker process keeps GPU/CPU memory bounded and
    # makes the queue-depth autoscaler a faithful backlog signal.
    worker_prefetch_multiplier=int(os.getenv("CELERY_WORKER_PREFETCH_MULTIPLIER", "1")),
    # JSON only: the payloads crossing the broker are frame metadata, and pickle
    # would make the broker an execution surface.
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Celery writes to a logfile during `inspect` so that a liveness probe can
    # redirect stdout safely; see the probe in k8s/base/deployment-vision.yaml.
    worker_send_task_events=True,
    task_send_sent_event=True,
    broker_connection_retry_on_startup=True,
)


@celery_app.task
def process_vision_task(payload_json: str):
    """
    This task simulates the main workflow:
    1. Deserialize payload from message queue.
    2. Call VisionService.process_frame(payload).
    3. Call ModbusService.send_command(...) using results.
    """
    print(f"Received task to process: {payload_json[:50]}...")
    # Logic will go here after all services are integrated.
    return {"status": "stub_task_received"}