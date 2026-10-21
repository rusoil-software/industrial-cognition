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
"""The measurement loop of `scripts/benchmark_detectors.py`, without any model.

The rule that makes the CSV worth reading is that a vocabulary change is a
configuration cost, not a frame cost: the samples are grouped by the labels they
ask for and each group is warmed up once, untimed, before it is measured. Grouping
is where that rule can silently break - a sample scored under the wrong warm-up
would make a 6 s CLIP encode look like per-frame latency - so it is pinned here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_detectors.py"


def load_script():
    spec = importlib.util.spec_from_file_location("benchmark_detectors_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(sample_id: str, labels: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(sample_id=sample_id, labels=list(labels))


def test_samples_are_grouped_by_vocabulary() -> None:
    module = load_script()
    samples = [
        sample("a", ("steel blank",)),
        sample("b", ("robot", "person")),
        sample("c", ("steel blank",)),
    ]

    groups = module._grouped(samples)

    assert [vocabulary for vocabulary, _ in groups] == [("steel blank",), ("robot", "person")]
    assert [index for _, members in groups for index, _ in members] == [0, 2, 1]


def test_every_sample_keeps_its_own_index() -> None:
    module = load_script()
    samples = [sample(str(index), ("steel blank",)) for index in range(5)]

    groups = module._grouped(samples)

    assert len(groups) == 1
    assert [index for index, _ in groups[0][1]] == [0, 1, 2, 3, 4]


def test_grouping_an_empty_dataset_is_empty() -> None:
    module = load_script()

    assert module._grouped([]) == []
