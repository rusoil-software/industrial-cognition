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

import os
import sys
from typing import AsyncGenerator
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

# Add the project root to sys.path to make 'src' importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Print for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"sys.path: {sys.path}")
print(
    f"Is src directory accessible? {'src' in os.listdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))}")

try:
    from src.owl.inference.schemas import OWL2InferenceResult, DetectionBox
    from src.owl.inference.service import get_inference_service
    from src.owl.inference.exceptions import ModelNotLoadedException
    from src.owl.main import app
except ImportError as e:
    print(f"Failed to import from 'src': {e}")
    raise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Async test client using ASGI transport — no real server needed."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
def mock_inference_service():
    """
    Override the real OWL2InferenceService with a mock for all tests.
    This avoids loading any .onnx file during the test suite.
    """
    mock_service = MagicMock()
    mock_service._session = MagicMock()  # Marks model as "loaded"
    mock_service.model_version = "test-1.0"
    # Mock the assert_loaded method which is called by the router
    mock_service.assert_loaded = MagicMock(return_value=mock_service._session)
    mock_service.run_batch.return_value = [
        OWL2InferenceResult(
            index=0,
            detections=[
                DetectionBox(box=[0.1, 0.2, 0.5, 0.8], score=0.91, label="0")
            ],
        )
    ]

    app.dependency_overrides[get_inference_service] = lambda: mock_service
    yield mock_service
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "inputs": [
        {
            "pixel_values": [[[0.5] * 224] * 224] * 3,  # [C=3, H=224, W=224]
            "input_ids": [101, 2023, 2003, 102],
            "attention_mask": [1, 1, 1, 1],
        }
    ]
}


@pytest.mark.asyncio
async def test_detect_objects_success(client: AsyncClient):
    response = await client.post("/api/v1/inference/owl2/detect", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test-1.0"
    assert len(body["results"]) == 1
    assert body["results"][0]["detections"][0]["score"] == 0.91


@pytest.mark.asyncio
async def test_detect_objects_model_not_loaded(client: AsyncClient, mock_inference_service):
    """Simulate the model session being None (not loaded)."""
    mock_inference_service._session = None
    mock_inference_service.assert_loaded.side_effect = ModelNotLoadedException()

    response = await client.post("/api/v1/inference/owl2/detect", json=VALID_PAYLOAD)

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_detect_objects_mismatched_mask(client: AsyncClient):
    """Pydantic model_validator should reject mismatched attention_mask length."""
    bad_payload = {
        "inputs": [
            {
                "pixel_values": [[[0.5] * 224] * 224] * 3,
                "input_ids": [101, 102],
                "attention_mask": [1, 1, 1],  # length mismatch
            }
        ]
    }
    response = await client.post("/api/v1/inference/owl2/detect", json=bad_payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_check_ready(client: AsyncClient):
    response = await client.get("/api/v1/inference/owl2/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_health_check_unavailable(client: AsyncClient, mock_inference_service):
    mock_inference_service._session = None
    mock_inference_service.assert_loaded.side_effect = ModelNotLoadedException()

    response = await client.get("/api/v1/inference/owl2/health")

    assert response.status_code == 200
    assert response.reason_phrase == "OK"
