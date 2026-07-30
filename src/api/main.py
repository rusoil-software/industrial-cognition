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

# Main Application Entry Point (FastAPI App)
# This file orchestrates the Camera, Vision, and Robot modules.

from typing import Optional
import asyncio
import asyncio.gather
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.camera.service import CameraService
from src.vision.service import VisionService, DetectionTaskPayload
from src.robot.service import ModbusService
from src.robot.integration_manager import RobotCommandManager
from src.vision.schemas import CameraConfig, StreamingPayload
from src.camera.schemas import DetectionResult, FrameMetadata
from src.camera.constants import DEFAULT_FPS

app = FastAPI(title="Industrial Cognition Core API")

# --- Global State ---
# State management for all initialized services
camera_service: Optional[CameraService] = None
vision_service: Optional[VisionService] = None
modbus_service: Optional[ModbusService] = None
robot_manager: Optional[RobotCommandManager] = None

async def setup_services(config: CameraConfig):
    """Initializes all core components upon application startup."""
    global camera_service, vision_service, modbus_service, robot_manager
    
    # 1. Initialize Modbus Service
    modbus_service = ModbusService(port=f"{config.camera_id}_rs485", baudrate="9600")
    if not modbus_service.connect():
        raise ConnectionError("Failed to connect to Modbus Controller. Cannot start system.")
    
    # 2. Initialize Vision Service
    vision_service = VisionService(model_path=f"./models/{'owlv2'}/{'v3.1.0'}/model.onnx")
    if not vision_service.load_model():
        raise RuntimeError("Failed to load Vision Model. Cannot proceed.")

    # 3. Initialize Camera Service
    camera_service = CameraService(config)
    if not camera_service.start_stream():
        raise ConnectionError("Failed to start the initial FFmpeg camera stream.")

    # 4. Initialize Robot Manager
    robot_manager = RobotCommandManager(modbus_service)
    print("--- SYSTEM BOOTSTRAP COMPLETE ---")


@app.on_event("startup")
async def startup_event():
    """Runs the setup routine when the API container starts."""
    # In a real scenario, this fetches config from K8s ConfigMap.
    # Here we use the default config for bootstrap.
    global camera_service, vision_service, modbus_service, robot_manager
    
    # Mock configuration load
    mock_config = CameraConfig(camera_id="MAIN_CAM_01", rtsp_url="rtsp://mock_camera_stream", is_active=True)
    try:
        await setup_services(mock_config)
    except Exception as e:
        print(f"CRITICAL STARTUP FAILURE: {e}")
        # The app startup should fail if critical services cannot connect.


@app.websocket("/ws/detections")
async def websocket_endpoint(websocket: WebSocket):
    """Handles the continuous, real-time data stream from the Camera Service."""
    await websocket.accept()
    print("WebSocket client connected. Awaiting stream data.")
    
    try:
        while True:
            # Simulate receiving processed data packets from the background camera thread/process
            # In a real setup, we would poll the CameraService's internal queue.
            await asyncio.sleep(0.1) # Yield control back to the event loop
            
            # 1. Simulate reading raw data chunk from the background FFmpeg process
            # This simulates the CameraService picking up a chunk of raw bytes.
            mock_raw_bytes = b'\xde\xad\xbe\xef' * 100 # Mock binary data
            mock_meta = FrameMetadata(camera_id="MAIN_CAM_01", frame_number=int(time.time()*1000), timestamp=datetime.utcnow())

            # 2. Hand-off to Vision Service (Core Logic)
            detection_payload = DetectionTaskPayload(
                frame_metadata=mock_meta,
                raw_frame_data_b64="MOCK_BASE64_STRING_LENGTH_FOR_LONGER_FRAME_SIMULATION"
            )
            
            vision_payload = vision_service.process_frame(detection_payload)

            if vision_payload:
                # 3. Pass structured data to the Robot Manager
                # This triggers the Modbus write sequence.
                success = robot_manager.execute_command_from_detections(vision_payload)
                if success:
                    await websocket.send_json({"status": "success", "command_executed": True, "message": "Command sent to robot."})
                else:
                    await websocket.send_json({"status": "warning", "command_executed": False, "message": "Robot command failed."})

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    finally:
        # Clean up resources on disconnect
        await camera_service.stop_stream()
        modbus_service.close()
        print("WebSocket session and all resources cleaned up.")

@app.get("/health")
def health_check():
    """Simple endpoint to check API gateway health."""
    return {"status": "ok", "service": "API Gateway", "uptime": "Operational"}

# Add additional GET/POST routes for /status and /config here...