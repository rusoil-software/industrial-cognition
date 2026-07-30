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
