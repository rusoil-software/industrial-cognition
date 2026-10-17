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
"""HTTP schemas for the per-camera detection prompt.

The API accepts **object names**, not pre-tokenised IDs: the System User types
"metal sheet", not a CLIP token vector, and the tokenisation is the server's job.
The response returns the tokenised form as well, so an operator can confirm what
the model will actually be asked for - useful when a phrase tokenises unexpectedly.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PromptRequest(BaseModel):
    """What the System User submits for one camera."""

    labels: List[str] = Field(
        ...,
        min_length=1,
        max_length=8,
        description=(
            "Object names to look for, in class order, e.g. ['metal sheet']. "
            "OWLv2 expects short noun phrases; each becomes one class."
        ),
    )

    @field_validator("labels")
    @classmethod
    def _no_blank_labels(cls, labels: List[str]) -> List[str]:
        cleaned = [label.strip() for label in labels if label and label.strip()]
        if not cleaned:
            raise ValueError("at least one non-blank object name is required")
        return cleaned


class PromptResponse(BaseModel):
    """The prompt in force for a camera, with its tokenised form."""

    camera_id: str
    labels: List[str]
    prompt_text: str
    input_ids: List[int]
    attention_mask: List[int]
    tokenizer: str
    max_length: int
    active_tokens: int = Field(
        ...,
        description="Non-padding token count; 0 would mean nothing is attended to.",
    )

    @classmethod
    def from_prompt(cls, prompt) -> "PromptResponse":  # noqa: ANN001 - CameraPrompt
        return cls(
            camera_id=prompt.camera_id,
            labels=list(prompt.labels),
            prompt_text=prompt.prompt_text,
            input_ids=list(prompt.input_ids),
            attention_mask=list(prompt.attention_mask),
            tokenizer=prompt.tokenizer,
            max_length=prompt.max_length,
            active_tokens=sum(prompt.attention_mask),
        )


class PromptListResponse(BaseModel):
    prompts: List[PromptResponse]
    count: int


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
