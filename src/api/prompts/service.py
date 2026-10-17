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
"""Business logic for per-camera prompts: validate, tokenise, persist.

The service is the seam the HTTP layer talks to, so the router stays a thin
translation of domain errors into status codes and the logic here stays reachable from
a CLI, a future web UI or a test without FastAPI.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from src.api.storage import prompt_repository
from src.vision.prompt import (
    CameraPromptSet,
    PromptEncodingError,
    PromptTokenizer,
    encode_prompt_set,
    get_tokenizer,
)

__all__ = [
    "delete_camera_prompt",
    "get_camera_prompt",
    "list_camera_prompts",
    "set_camera_prompt",
]

logger = logging.getLogger(__name__)


def set_camera_prompt(
    session: Session,
    camera_id: str,
    queries: Sequence[Tuple[str, float]],
    tokenizer: Optional[PromptTokenizer] = None,
) -> CameraPromptSet:
    """Tokenise ``(label, threshold)`` pairs and store them for *camera_id*.

    Tokenisation happens before any write, so a prompt the model cannot use is
    rejected without leaving a half-configured camera behind.

    Raises:
        PromptEncodingError: invalid camera id, invalid label or threshold, or a query
            that tokenises to no usable tokens.
        TokenizerUnavailable: no tokenizer is configured in this deployment.
    """
    camera_id = prompt_repository.validate_camera_id(camera_id)
    if not queries:
        raise PromptEncodingError("at least one detection query is required")

    tokenizer = tokenizer or get_tokenizer()
    prompt_set = encode_prompt_set(queries, tokenizer, camera_id=camera_id)
    return prompt_repository.save_prompt(session, prompt_set)


def get_camera_prompt(session: Session, camera_id: str) -> Optional[CameraPromptSet]:
    """Return the queries in force for *camera_id*, or ``None``."""
    return prompt_repository.load_prompt(session, camera_id)


def delete_camera_prompt(session: Session, camera_id: str) -> bool:
    """Remove the prompt for *camera_id*; returns whether one existed."""
    return prompt_repository.delete_prompt(session, camera_id)


def list_camera_prompts(session: Session) -> list[CameraPromptSet]:
    """Return every configured camera prompt."""
    return prompt_repository.list_prompts(session)
