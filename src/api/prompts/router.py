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
"""Per-camera detection prompt endpoints.

    PUT    /cameras/{camera_id}/prompt   set (tokenise + store)
    GET    /cameras/{camera_id}/prompt   read back
    DELETE /cameras/{camera_id}/prompt   clear
    GET    /cameras/prompts              list every configured camera

Status-code mapping is deliberate:

* **422** - the prompt is malformed (blank label, a period, padding-only tokens).
  The user can fix it, so the detail explains what to change.
* **503** - this deployment has no tokenizer configured. The request is valid and
  will work once the service is provisioned, which is what 503 means; a 500 would
  wrongly imply a bug.
* **404** - no prompt is set for that camera.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.prompts import service
from src.api.prompts.dependencies import get_db
from src.api.prompts.schemas import (
    ErrorResponse,
    PromptListResponse,
    PromptRequest,
    PromptResponse,
)
from src.vision.prompt import PromptEncodingError, TokenizerUnavailable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cameras", tags=["Detection prompts"])

_COMMON_RESPONSES = {
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": "The prompt cannot be tokenised into usable text conditioning.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "No prompt tokenizer is configured in this deployment.",
    },
}


@router.put(
    "/{camera_id}/prompt",
    response_model=PromptResponse,
    summary="Set the detection prompt for one camera",
    responses=_COMMON_RESPONSES,
)
def set_prompt(
    camera_id: str,
    request: PromptRequest,
    session: Session = Depends(get_db),
) -> PromptResponse:
    """Tokenise the submitted object names and store them for this camera.

    Idempotent: setting the same prompt twice leaves one row. The tokenised form is
    returned so the caller can verify what the model will be asked for.
    """
    try:
        prompt = service.set_camera_prompt(
            session, camera_id, request.labels
        )
    except PromptEncodingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except TokenizerUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return PromptResponse.from_prompt(prompt)


@router.get(
    "/prompts",
    response_model=PromptListResponse,
    summary="List every configured camera prompt",
)
def list_prompts(session: Session = Depends(get_db)) -> PromptListResponse:
    prompts = service.list_camera_prompts(session)
    return PromptListResponse(
        prompts=[PromptResponse.from_prompt(prompt) for prompt in prompts],
        count=len(prompts),
    )


@router.get(
    "/{camera_id}/prompt",
    response_model=PromptResponse,
    summary="Read the detection prompt for one camera",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "No prompt is configured for this camera.",
        }
    },
)
def get_prompt(
    camera_id: str, session: Session = Depends(get_db)
) -> PromptResponse:
    try:
        prompt = service.get_camera_prompt(session, camera_id)
    except PromptEncodingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    if prompt is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no detection prompt is configured for camera {camera_id!r}",
        )
    return PromptResponse.from_prompt(prompt)


@router.delete(
    "/{camera_id}/prompt",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear the detection prompt for one camera",
)
def delete_prompt(camera_id: str, session: Session = Depends(get_db)) -> None:
    try:
        removed = service.delete_camera_prompt(session, camera_id)
    except PromptEncodingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no detection prompt is configured for camera {camera_id!r}",
        )
