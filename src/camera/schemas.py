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

class CameraConfig(BaseModel):
    """Configuration for a single connected camera."""
    camera_id: str = Field(description="Unique identifier for the camera (e.g., Camera_A_RTSP).")
    rtsp_url: str = Field(description="The full RTSP stream URL (e.g., rtsp://user:pass@ip:port/stream).")
    is_active: bool = Field(default=True)
    # Other metadata like resolution/codec can be added here later.

class FrameMetadata(BaseModel):
    """Metadata associated with a single captured frame."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    camera_id: str
    frame_number: int
    
class DetectionResult(BaseModel):
    """Represents the output of the detection pipeline for one object."""
    object_type: str
    x_norm: float = Field(ge=0.0, le=1.0)
    y_norm: float = Field(ge=0.0, le=1.0)
    width_norm: float = Field(ge=0.0, le=1.0)
    height_norm: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

class StreamingPayload(BaseModel):
    """The full payload sent over the WebSocket."""
    metadata: FrameMetadata
    detections: list[DetectionResult]
    # Note: The raw frame data (e.g., Base64 or memory buffer) would be attached here
    # or handled by the underlying WebSocket transport mechanism.
