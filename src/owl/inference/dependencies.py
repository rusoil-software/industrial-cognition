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

from fastapi import Depends

from src.owl.inference.config import inference_settings
from src.owl.inference.exceptions import BatchTooLargeException
from src.owl.inference.schemas import OWL2BatchRequest
from src.owl.inference.service import OWL2InferenceService, get_inference_service


async def valid_batch_size(
        request: OWL2BatchRequest,
        service: OWL2InferenceService = Depends(get_inference_service),
) -> OWL2BatchRequest:
    """Validates batch size and that the model is ready before hitting the handler."""
    service.assert_loaded()  # Raises 503 if model not loaded

    if len(request.inputs) > inference_settings.OWL2_MAX_BATCH_SIZE:
        raise BatchTooLargeException(inference_settings.OWL2_MAX_BATCH_SIZE)

    return request
