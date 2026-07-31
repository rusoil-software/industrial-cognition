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

import time
import json
from typing import Optional, Dict, Any
from src.camera.schemas import DetectionResult
from src.robot.service import ModbusService
from src.vision.schemas import VisionOutputPayload

class RobotCommandManager:
    """
    Orchestrates the action: taking high-level detections and translating them
    into actionable, low-level Modbus commands for the physical robot.
    """
    def __init__(self, modbus_service: ModbusService):
        self.modbus_service = modbus_service
        # In a real K8s/Celery setup, this manager would be triggered by a queue listener.
        # For now, we simulate the trigger.

    def _normalize_coordinates_to_registers(self, detection: DetectionResult) -> Optional[Dict[str, Any]]:
        """
        Converts normalized, continuous detection data into discrete, register-addressable integers.
        This function enforces the mapping rules from the physical specification.
        """
        # Example mapping: Normalized X (0-1) -> Register Address (e.g., 40001)
        # This mapping must be derived from the 'physical_spec' memory record.
        
        # For simulation, we'll map the normalized X to a register, and use a fixed command register.
        # The value written must be an integer representing the required position/angle.
        
        # Simple mapping: Assuming X_norm needs to be converted to a 16-bit integer offset.
        x_offset = int(detection.x_norm * 65535)
        
        return {
            "register_address": 40001, # Example: Target X Position Register
            "value": x_offset,
            "count": 1
        }

    def execute_command_from_detections(self, payload: VisionOutputPayload) -> bool:
        """
        Processes all detected objects, prioritizing the best candidate, and executing a single
        move command via Modbus.
        
        Returns:
            True if a command was successfully dispatched.
        """
        if not self.modbus_service.read_position(40001):
             print("Warning: Could not read current position to determine target delta.")
             return False

        # 1. Triage: Find the best target object to command movement toward.
        # Convention: Prioritize metal sheets with confidence > 0.90.
        best_detection: Optional[DetectionResult] = None
        for det in payload.detections:
            if det.object_type == "metal_sheet" and det.confidence >= 0.90:
                # Simple prioritization: take the first highly confident sheet found.
                best_detection = det
                break
        
        if not best_detection:
            print("No high-confidence, actionable targets found to command movement.")
            return False

        # 2. Convert and Execute
        command_params = self._normalize_coordinates_to_registers(best_detection)
        if not command_params:
            print("Failed to generate Modbus command parameters from detection.")
            return False

        success = self.modbus_service.send_command(
            holding_register_address=command_params["register_address"],
            value=command_params["value"],
            count=command_params["count"]
        )
        
        if success:
            print(f"SUCCESS: Command sent for {best_detection.object_type} at register {command_params['register_address']}.")
        else:
            print("FAILURE: Modbus command failed to dispatch.")
        return success

# Integration with Messaging Layer (Placeholder for Celery/RabbitMQ consumption)
# In a live system, this class would be wrapped in a Celery worker listening to the 'vision_output' queue.
# The function to call would be: process_incoming_payload(payload: VisionOutputPayload)
#     command_manager = RobotCommandManager(modbus_service)
#     command_manager.execute_command_from_detections(payload)