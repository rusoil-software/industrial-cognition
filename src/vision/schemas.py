from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
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