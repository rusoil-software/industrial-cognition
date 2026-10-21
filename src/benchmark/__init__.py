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
# date   : 2026-Oct-21
# ==============================================================================
"""Evidence over vibes: the same bytes and the same rule for every candidate detector.

``scripts/benchmark_detectors.py`` produces the CSV; the modules here are the parts that
can be tested without downloading a model:

* :mod:`src.benchmark.registry` - what is being compared, and what its licence allows;
* :mod:`src.benchmark.dataset` - the images, the ground truth and the station protocol;
* :mod:`src.benchmark.scoring` - IoU, hits, and the production presence gate;
* :mod:`src.benchmark.measure` - time, memory, artifact size;
* :mod:`src.benchmark.report` - the CSV, the per-image detail and the console table;
* :mod:`src.benchmark.runners` - one adapter per runtime.
"""

from __future__ import annotations

__all__: list[str] = []
