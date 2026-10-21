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
# date   : 2026-Oct-21
# ==============================================================================
#
# The detector benchmark environment. Not a production image: it carries PyTorch,
# Ultralytics and every runtime a candidate model needs, so the honest comparison can
# happen on one machine without changing what the cluster installs.
#
#   docker build -f src/benchmark/Dockerfile -t industrial-cognition/benchmark:dev .
#   docker run --rm --gpus all -v "${PWD}:/work" -w /work \
#       industrial-cognition/benchmark:dev python scripts/benchmark_detectors.py --models all
#
# The repository is mounted rather than copied so the CSV lands on the host and a re-run
# does not need a rebuild. Only the interpreter and the wheels live in the image.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/work

# libgl/libglib are what opencv-python-headless links against at import time; git and
# curl make a failed download debuggable from inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements/benchmark.txt /tmp/benchmark.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/benchmark.txt

# The ONNX artifact is large and lives on the host; mounting it keeps the image small.
ENV OWL2_MODEL_PATH=/work/src/owl/models/cuda/owl2_model.onnx

CMD ["python", "scripts/benchmark_detectors.py", "--models", "all"]
