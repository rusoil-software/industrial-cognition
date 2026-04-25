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

from pydantic import BaseModel, Field, model_validator


class OWL2InputItem(BaseModel):
    """A single inference input — adapt field names to your OWL 2 model's actual inputs."""
    pixel_values: list[list[list[float]]] = Field(
        ...,
        description="Image tensor as [C, H, W] float32 values.",
    )
    input_ids: list[int] = Field(
        ...,
        description="Tokenized text query IDs.",
    )
    attention_mask: list[int] = Field(
        ...,
        description="Attention mask corresponding to input_ids.",
    )


class OWL2BatchRequest(BaseModel):
    inputs: list[OWL2InputItem] = Field(..., min_length=1)

    @model_validator(mode="after")
    def check_attention_mask_length(self) -> "OWL2BatchRequest":
        for item in self.inputs:
            if len(item.attention_mask) != len(item.input_ids):
                raise ValueError(
                    "attention_mask length must match input_ids length."
                )
        return self


class DetectionBox(BaseModel):
    box: list[float] = Field(..., description="[x_min, y_min, x_max, y_max] normalized.")
    score: float = Field(..., ge=0.0, le=1.0)
    label: str


class OWL2InferenceResult(BaseModel):
    index: int
    detections: list[DetectionBox]


class OWL2BatchResponse(BaseModel):
    results: list[OWL2InferenceResult]
    model_version: str
