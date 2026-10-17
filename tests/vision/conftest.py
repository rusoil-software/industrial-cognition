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
# ==============================================================================
"""Fixtures for the vision tests.

`tmp_path` is not usable in this environment: the process runs under a sandbox
whose temporary directory is not writable, so pytest's own `tmp_path` fixture
raises `PermissionError` before a test body runs. `scratch` provides the same
thing inside the repository instead, which is writable and is cleaned up
afterwards.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

SCRATCH_ROOT = Path(__file__).resolve().parent / ".scratch"


@pytest.fixture
def scratch() -> Path:
    """A writable, throwaway directory inside the repo."""
    directory = SCRATCH_ROOT / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)
