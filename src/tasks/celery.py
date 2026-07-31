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

from celery import Celery

# Initialize Celery app (will be configured in docker-compose.yml environment variables)
celery_app = Celery(
    'tasks',
    broker='amqp://myuser:mypassword@rabbitmq:5672//',
    backend='redis://redis:6379/0'
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