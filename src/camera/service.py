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

import subprocess
import subprocess.exceptions
import time
from typing import List, Dict, Any
from .constants import DEFAULT_FPS, DEFAULT_WIDTH, DEFAULT_HEIGHT, STREAM_CODEC, STREAM_PIPELINE_BASE
from .schemas import CameraConfig, FrameMetadata, DetectionResult

class CameraService:
    """
    Manages the lifecycle of video stream capture using FFmpeg/GStreamer
    for high-performance, multi-camera support.
    """
    def __init__(self, config: CameraConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.is_running: bool = False

    def _build_ffmpeg_command(self) -> List[str]:
        """
        Constructs the FFmpeg command list to capture the stream.
        This must be adapted based on the specific RTSP source and desired output format.
        """
        # Placeholder: This command needs refinement based on the actual RTSP stream complexity.
        return [
            "ffmpeg",
            "-y", # Overwrite output files without asking
            "-i", self.config.rtsp_url, # Input stream
            "-vcodec", STREAM_CODEC,
            "-r", str(DEFAULT_FPS), # Set the target frame rate
            "-s", f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}", # Set output resolution
            "pipe:1" # Output to stdout/stdout pipe
        ]

    def start_stream(self) -> bool:
        """
        Starts the FFmpeg subprocess to capture frames.
        This must run asynchronously and process stdout/stderr in the main loop.
        """
        if self.is_running:
            print("Warning: Stream is already running.")
            return True

        print(f"Attempting to start FFmpeg stream for {self.config.camera_id}...")
        
        # Use subprocess.Popen to run ffmpeg non-blocking
        try:
            self.process = subprocess.Popen(
                self._build_ffmpeg_command(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )
            self.is_running = True
            print(f"Stream started successfully. PID: {self.process.pid}")
            return True
        except FileNotFoundError:
            print("ERROR: FFmpeg or ffmpeg executable not found. Is FFmpeg installed and in PATH?")
            self.is_running = False
            self.process = None
            return False
        except Exception as e:
            print(f"Failed to start stream due to general error: {e}")
            self.is_running = False
            self.process = None
            return False

    def process_frame_data(self, raw_data: bytes, metadata: FrameMetadata) -> Optional[str]:
        """
        Processes raw binary data (e.g., JPEG/YUV buffer) from the FFmpeg stdout pipe.
        This function is where the OpenCV logic will be integrated, reading the raw bytes.
        
        Returns a serialized JSON payload string containing detections, or None if no detections are found.
        """
        # *** CRITICAL STEP: Replace this placeholder with actual image decoding logic (e.g., using cv2.imdecode) ***
        # For simulation, we will assume successful processing yields a dummy result.
        
        print(f"Processing raw frame data from Camera {metadata.camera_id}...")
        
        # Placeholder for actual CV processing:
        # 1. Decode raw_data into an OpenCV image format.
        # 2. Run OWLv2 model inference on the image.
        # 3. Extract coordinates/detections.
        
        if metadata.frame_number % 10 == 0: # Simulate detections every 10 frames
            mock_detection = DetectionResult(
                object_type="metal_sheet",
                x_norm=0.1, y_norm=0.2, width_norm=0.3, height_norm=0.15, confidence=0.92
            )
            # Construct the final payload (mocking JSON serialization)
            payload_data = f'{{"metadata": {{"camera_id": "{metadata.camera_id}", "timestamp": "{metadata.timestamp.isoformat()}", "frame_number": {metadata.frame_number}}}, "detections": [{{"object_type": "metal_sheet", "x_norm": 0.1, "y_norm": 0.2, "width_norm": 0.3, "height_norm": 0.15, "confidence": 0.92}}]}}'
            return payload_data
        
        return None # No detections in this frame

    def stop_stream(self) -> bool:
        """Stops the running FFmpeg process cleanly."""
        if self.process and self.is_running:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.is_running = False
            print("Camera stream stopped successfully.")
            return True
        return False

# Example usage (for testing the module structure)
# if __name__ == '__main__':
#     pass