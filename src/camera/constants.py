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

# Camera Constants & Configurations
# These define the physical and streaming parameters for all cameras.

# Protocol Constants
MODBUS_PROTOCOL: str = "Modbus RTU"
MODBUS_BAUD_RATE: str = "9600"
MODBUS_INTERFACE: str = "/dev/ttyUSB0" # Placeholder for RS-485 serial port

# Streaming Constants
DEFAULT_WIDTH: int = 1920
DEFAULT_HEIGHT: int = 1080
DEFAULT_FPS: int = 30
# FFmpeg/GStreamer Specific Flags
STREAM_CODEC: str = "h264"
STREAM_PIPELINE_BASE: str = "ffmpeg -i" # Placeholder for the base command structure

# Status Codes for reporting
CAMERA_STATUS_CONNECTED: str = "connected"
CAMERA_STATUS_DISCONNECTED: str = "disconnected"
CAMERA_STATUS_ERROR: str = "error"