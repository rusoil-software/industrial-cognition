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

# Dockerfile for the FastAPI API Gateway (api_gateway)
# This image runs the main API server, orchestrating camera input,
# handling web/WebSocket connections, and managing the overall flow.

FROM python:3.11-slim

WORKDIR /app

# Copy requirements files first for build caching
COPY requirements/api.txt requirements/api.txt

# Install core dependencies
RUN pip install --no-cache-dir -r requirements/api.txt

# Copy the entire source code structure
COPY src/ ./src/

# Alembic configuration and revisions.
#
# Required in the image, not only on a developer's machine: the deployment runs
# migrations as a Job from this same image (k8s/scripts/deploy.sh --migrate), so an
# image without these files makes `alembic upgrade head` fail with "No such file:
# alembic.ini" and leaves the schema uncreated. The API does not import Alembic at
# runtime; this exists purely so the migration path is deployable.
COPY alembic.ini ./alembic.ini
COPY alembic/ ./alembic/

# Expose the port used by FastAPI
EXPOSE 8000

# Command to run the API when the container starts
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]