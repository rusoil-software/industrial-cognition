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

from fastapi import APIRouter, Depends, status
from fastapi.concurrency import run_in_threadpool

from src.owl.inference.dependencies import valid_batch_size
from src.owl.inference.schemas import OWL2BatchRequest, OWL2BatchResponse
from src.owl.inference.service import OWL2InferenceService, get_inference_service

router = APIRouter(prefix="/inference", tags=["OWL 2 Inference"])


@router.post(
    "/owl2/detect",
    response_model=OWL2BatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Run OWL 2 object detection",
    description=(
            "Accepts a batch of image tensors and text queries. "
            "Returns bounding boxes and confidence scores per query."
    ),
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid input shape or data."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Model not loaded."},
        status.HTTP_400_BAD_REQUEST: {"description": "Batch size exceeds limit."},
    },
)
async def detect_objects(
        request: OWL2BatchRequest = Depends(valid_batch_size),
        service: OWL2InferenceService = Depends(get_inference_service),
) -> OWL2BatchResponse:
    """
    OWL 2 open-vocabulary detection endpoint.

    ONNX inference is CPU-bound, so it is explicitly offloaded to a
    thread pool to avoid blocking the async event loop.
    """
    results = await run_in_threadpool(service.run_batch, request)

    return OWL2BatchResponse(
        results=results,
        model_version=service.model_version,
    )


@router.get(
    "/owl2/health",
    status_code=status.HTTP_200_OK,
    summary="Model health check",
    tags=["Health"],
)
async def model_health(
        service: OWL2InferenceService = Depends(get_inference_service),
) -> dict:
    """Returns whether the OWL 2 model session is active."""
    is_ready = service.session is not None
    return {"status": "ready" if is_ready else "unavailable", "model_version": service.model_version}
