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
# ==============================================================================
"""The `pipeline` domain: capture-to-worker orchestration for the MVM service.

`doc/conventions.md` §1 requires every subsystem to own an isolated directory.
The camera and vision domains own *devices* and *models* respectively; this domain
owns the thing that neither should: the bounded, observable flow of frames between
them, including load shedding and the 60 FPS accounting. See
`doc/vision-pipeline.md`.
"""
