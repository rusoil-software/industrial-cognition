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

from pathlib import Path

import onnxruntime as ort
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


def get_provider(device: str = "cuda") -> str:
    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"

class InferenceConfig(BaseSettings):
    OWL2_MODEL_PATH: Path = Path("src") / "owl" / "models" / "cuda" / "owl2_model.onnx"
    OWL2_EXECUTION_PROVIDER: str = get_provider()
    OWL2_MAX_BATCH_SIZE: int = 32

    model_config = ConfigDict()


inference_settings = InferenceConfig()