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

from pydantic import ConfigDict
from pydantic_settings import BaseSettings

from src.owl.constants import Environment


class Config(BaseSettings):
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Environment = Environment.PRODUCTION
    CORS_ORIGINS: list[str] = ["*"]
    SHOW_DOCS_ENVIRONMENT: tuple[str, ...] = ("local", "staging")
    # ENV_FILE_PATHS_LIST: dict[Environment | None, set[list]]={Environment.PRODUCTION : {".env"}, }

    model_config = ConfigDict()


settings = Config()
