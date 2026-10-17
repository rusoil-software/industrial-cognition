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
"""HTTP schemas for per-camera detection prompts.

The System User submits **queries**, each with its own threshold:

    {"prompts": [{"label": "metal sheet", "threshold": 0.25},
                 {"label": "person",      "threshold": 0.10}]}

One query per object: OWLv2 turns each label into its own class channel, and all of
them are evaluated in a single forward pass. Per-query thresholds are what let a loose
human-presence check coexist with strict actuation thresholds on the same camera.

The `labels` form is kept for compatibility - it expands to one query per label
sharing a single threshold.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from src.vision.prompt import DEFAULT_THRESHOLD, MAX_THRESHOLD, MIN_THRESHOLD


class PromptQueryRequest(BaseModel):
    """One object to look for, and how confident the model must be."""

    label: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Short noun phrase, e.g. 'metal sheet'. One phrase per object.",
    )
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        ge=MIN_THRESHOLD,
        le=MAX_THRESHOLD,
        description=(
            "Confidence required for this object, 0.01-0.99. Use a lower value for "
            "safety-relevant classes: a missed person costs more than a false alarm."
        ),
    )

    @field_validator("label")
    @classmethod
    def _clean_label(cls, label: str) -> str:
        cleaned = label.strip()
        if not cleaned:
            raise ValueError("an object name is required")
        return cleaned


class PromptRequest(BaseModel):
    """Either the query form or the legacy labels form."""

    prompts: Optional[List[PromptQueryRequest]] = Field(
        default=None,
        min_length=1,
        max_length=16,
        description="Detection queries; each becomes one class.",
    )
    labels: Optional[List[str]] = Field(
        default=None,
        description="Legacy form: one query per label, all using `threshold`.",
    )
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        ge=MIN_THRESHOLD,
        le=MAX_THRESHOLD,
        description="Threshold applied to the legacy `labels` form.",
    )

    @field_validator("labels")
    @classmethod
    def _no_blank_labels(cls, labels: Optional[List[str]]) -> Optional[List[str]]:
        if labels is None:
            return None
        cleaned = [label.strip() for label in labels if label and label.strip()]
        if not cleaned:
            raise ValueError("at least one non-blank object name is required")
        return cleaned

    def queries(self) -> List[PromptQueryRequest]:
        """Normalise either accepted form into a list of queries.

        Raises:
            ValueError: when neither form is supplied, or both are.
        """
        if self.prompts and self.labels:
            raise ValueError("send either 'prompts' or 'labels', not both")
        if self.prompts:
            return list(self.prompts)
        if self.labels:
            return [
                PromptQueryRequest(label=label, threshold=self.threshold)
                for label in self.labels
            ]
        raise ValueError("at least one detection query is required")


class PromptQueryResponse(BaseModel):
    """One query as stored, including its tokenisation."""

    label: str
    threshold: float
    prompt_text: str
    input_ids: List[int]
    attention_mask: List[int]
    active_tokens: int

    @classmethod
    def from_prompt(cls, prompt) -> "PromptQueryResponse":  # noqa: ANN001 - CameraPrompt
        return cls(
            label=prompt.label,
            threshold=prompt.threshold,
            prompt_text=prompt.prompt_text,
            input_ids=list(prompt.input_ids),
            attention_mask=list(prompt.attention_mask),
            active_tokens=sum(prompt.attention_mask),
        )


class PromptResponse(BaseModel):
    """The queries in force for one camera."""

    camera_id: str
    prompts: List[PromptQueryResponse]
    tokenizer: str
    max_length: int

    @classmethod
    def from_set(cls, prompt_set) -> "PromptResponse":  # noqa: ANN001 - CameraPromptSet
        return cls(
            camera_id=prompt_set.camera_id,
            prompts=[
                PromptQueryResponse.from_prompt(query) for query in prompt_set.queries
            ],
            tokenizer=prompt_set.tokenizer,
            max_length=prompt_set.max_length,
        )


class PromptListResponse(BaseModel):
    cameras: List[PromptResponse]
    count: int


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
