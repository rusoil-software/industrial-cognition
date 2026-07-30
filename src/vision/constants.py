# Vision Constants & Configurations
# Defines parameters specific to the object detection models (OWLv2, etc.)

# Model Constants
MODEL_NAME: str = "owlv2-optimized"
MODEL_VERSION: str = "v3.1.0-gpu"
# Model paths should point to resources managed by the deployment infrastructure.
MODEL_PATH: str = "./models/{model_name}/{model_version}/model.onnx"

# Detection Parameters
MIN_CONFIDENCE_THRESHOLD: float = 0.75
MAX_CONFIDENCE_THRESHOLD: float = 1.0
MIN_AREA_THRESHOLD_PX: int = 100 # Minimum pixel area for a detection to be considered valid.
MAX_AREA_THRESHOLD_PX: int = 500000 # Maximum allowed pixel area.

# Coordinate System Mapping
# These map the normalized [0, 1] coordinates back to real-world metric units (mm/degrees)
SCALE_FACTOR_MM_PER_UNIT: float = 3000.0 # Based on the 3000mm height specified in documentation.

# State Management
# The system should track which models are loaded and their readiness status.
MODEL_LOADED_STATUS: Dict[str, bool] = {
    "owlv2": False
}