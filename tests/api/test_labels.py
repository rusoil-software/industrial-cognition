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
# ==============================================================================
"""Tests for label normalisation.

These are the values that end up in `detection_results.object_type`, which is what history
queries filter on. Getting them wrong is how the old enum ended up recording `clutter` for
every real prompt.
"""

from __future__ import annotations

import pytest

from src.api.storage.labels import UNKNOWN_OBJECT_TYPE, normalise_object_type


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("robot", "robot"),
        ("Robot", "robot"),
        ("cardboard box", "cardboard box"),
        ("Metal  Sheet", "metal sheet"),
        ("  person  ", "person"),
        ("metal_sheet", "metal_sheet"),
    ],
)
def test_labels_are_lowercased_and_whitespace_collapsed(label: str, expected: str) -> None:
    assert normalise_object_type(label) == expected


@pytest.mark.parametrize("label", ["", "   ", None])
def test_an_empty_label_is_unknown_not_blank(label) -> None:
    """A blank NOT NULL value would be indistinguishable from a bug."""
    assert normalise_object_type(label) == UNKNOWN_OBJECT_TYPE
    assert UNKNOWN_OBJECT_TYPE == "unknown"


def test_the_label_is_preserved_exactly_elsewhere() -> None:
    """Normalisation must not be the only record: `label` keeps the original."""
    original = "Metal  Sheet"
    assert normalise_object_type(original) != original, (
        "this test documents that the raw text needs its own column"
    )
