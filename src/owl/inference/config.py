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
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from pathlib import Path


class InferenceConfig(BaseSettings):
    OWL2_MODEL_PATH: Path = Path("models/owl2_model.onnx")
    OWL2_EXECUTION_PROVIDER: str = "CPUExecutionProvider"
    OWL2_MAX_BATCH_SIZE: int = 32

    model_config = ConfigDict(env_file=".env")


inference_settings = InferenceConfig()