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
"""The registry is the claim being tested, so it is pinned here: what is compared,
which licence each model carries, and which of them cannot be measured at all."""

from __future__ import annotations

import pytest

from src.benchmark.registry import (
    MODELS,
    PERMISSIVE_LICENCES,
    license_summary,
    resolve,
    runnable,
)


def test_the_incumbent_comes_first():
    assert MODELS[0].key == "owl2-onnx"
    assert MODELS[0].runtime == "onnxruntime"


def test_every_registry_key_is_unique():
    keys = [spec.key for spec in MODELS]
    assert len(keys) == len(set(keys))


def test_the_requested_models_are_all_present():
    keys = {spec.key for spec in MODELS}
    assert keys == {
        "owl2-onnx",
        "grounding-dino-base",
        "grounding-dino-15-edge",
        "florence2-base",
        "yolo-world-m",
        "yolo-world-l",
        "dino-x",
    }


@pytest.mark.parametrize(
    ("key", "verdict"),
    [
        ("owl2-onnx", True),
        ("grounding-dino-base", True),
        ("florence2-base", True),
        ("yolo-world-m", False),
        ("yolo-world-l", False),
        ("grounding-dino-15-edge", False),
        ("dino-x", False),
    ],
)
def test_the_licence_verdict_is_apache_compatibility(key, verdict):
    spec = {spec.key: spec for spec in MODELS}[key]
    assert spec.license_ok is verdict


def test_permissive_licences_are_the_expected_set():
    assert PERMISSIVE_LICENCES == frozenset({"Apache-2.0", "MIT", "BSD-3-Clause"})


def test_the_api_only_models_are_marked_unavailable_with_a_reason():
    blocked = {spec.key: spec for spec in MODELS if not spec.available}
    assert set(blocked) == {"grounding-dino-15-edge", "dino-x"}
    assert all(spec.blocked for spec in blocked.values())
    assert all("API" in spec.blocked for spec in blocked.values())


def test_resolve_keeps_registry_order_not_the_argument_order():
    specs = resolve(["florence2-base", "owl2-onnx"])
    assert [spec.key for spec in specs] == ["owl2-onnx", "florence2-base"]


def test_resolve_rejects_an_unknown_key_and_names_the_known_ones():
    with pytest.raises(KeyError) as error:
        resolve(["yolo-world-xl"])
    assert "yolo-world-xl" in str(error.value)
    assert "owl2-onnx" in str(error.value)


def test_resolve_without_keys_returns_everything():
    assert resolve([]) == list(MODELS)


def test_runnable_excludes_the_models_that_need_an_api_key():
    available = {spec.key for spec in runnable()}
    assert "dino-x" not in available
    assert "grounding-dino-15-edge" not in available
    assert "grounding-dino-base" in available


def test_the_licence_summary_counts_every_row():
    summary = license_summary()
    assert summary == {"permissive": 3, "copyleft_or_commercial": 4, "unclear": 0}
    assert sum(summary.values()) == len(MODELS)
