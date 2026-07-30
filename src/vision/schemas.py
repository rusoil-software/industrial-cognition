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
# date   : 2026-Jun-30
# ==============================================================================

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from ..camera.schemas import DetectionResult, FrameMetadata
from .constants import MIN_CONFIDENCE_THRESHOLD

class DetectionTaskPayload(BaseModel):
    """Payload passed to the Vision Service from the Camera Service."""
    frame_metadata: FrameMetadata
    raw_frame_data_b64: str = Field(description="Base64 encoded raw frame buffer received from the camera stream.")

class VisionOutputPayload(BaseModel):
    """The structure of the final, processed detection results."""
    processed_timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_frame_id: int
    detections: list[DetectionResult] # Uses DetectionResult from camera/schemas.py for consistency
    processing_latency_ms: float # Time taken by the vision module to process the frame.