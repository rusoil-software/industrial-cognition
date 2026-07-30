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