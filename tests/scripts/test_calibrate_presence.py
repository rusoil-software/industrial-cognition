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
"""The threshold arithmetic of `scripts/calibrate_presence.py`, without a model.

The scoring half of that script needs a 293 MiB artifact and a GPU-less minute; the
part that can be wrong in a quiet way is the threshold itself, so it is pure and
tested here. A calibration helper that returns a threshold when the two groups overlap
would give a deployment a number that looks authoritative and is not.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "calibrate_presence.py"


def load_script():
    spec = importlib.util.spec_from_file_location("calibrate_presence_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return load_script()


def test_the_threshold_sits_between_the_two_groups(script):
    threshold = script.threshold_from([0.25, 0.31], [0.02, 0.05], margin=0.05)

    assert threshold == pytest.approx(0.15, abs=0.001)


def test_groups_that_touch_have_no_threshold(script):
    # 0.20 and 0.21 are a coin flip; any number between them is arbitrary.
    assert script.threshold_from([0.21], [0.20], margin=0.05) is None


def test_groups_that_are_close_but_not_touching_are_still_refused(script):
    # A real gap, but narrower than the required margin on both sides.
    assert script.threshold_from([0.24], [0.20], margin=0.05) is None


def test_a_wide_gap_is_accepted_at_the_default_margin(script):
    # Exactly 2 x margin of separation is the boundary: 0.05 either side.
    assert script.threshold_from([0.25], [0.15], margin=0.05) == pytest.approx(0.2)


def test_the_margin_is_configurable(script):
    assert script.threshold_from([0.25], [0.20], margin=0.01) == pytest.approx(0.225)
    assert script.threshold_from([0.25], [0.20], margin=0.1) is None


def test_a_missing_group_has_no_threshold(script):
    assert script.threshold_from([], [0.1]) is None
    assert script.threshold_from([0.3], []) is None


def test_the_weakest_positive_and_strongest_negative_decide(script):
    """More photographs cannot move the threshold past the worst one of each group."""
    threshold = script.threshold_from([0.9, 0.8, 0.22], [0.01, 0.02, 0.20], margin=0.001)

    assert threshold == pytest.approx(0.21)


def test_images_are_found_by_file_directory_and_glob(script, tmp_path):
    for name in ("a.jpg", "b.jpeg", "c.png", "d.webp", "e.txt"):
        (tmp_path / name).write_bytes(b"not really an image")

    by_glob = script.expand_images([str(tmp_path / "*.jpg")])
    assert [path.name for path in by_glob] == ["a.jpg"]

    by_directory = script.expand_images([str(tmp_path)])
    assert [path.name for path in by_directory] == ["a.jpg", "b.jpeg", "c.png", "d.webp"]

    by_file = script.expand_images([str(tmp_path / "c.png")])
    assert [path.name for path in by_file] == ["c.png"]


def test_a_glob_that_matches_nothing_is_not_an_error(script, tmp_path):
    """A typo must show up as "0 positive images", not as a stack trace."""
    assert script.expand_images([str(tmp_path / "*.jpg")]) == []


def test_a_file_listed_twice_is_scored_once(script, tmp_path):
    image = tmp_path / "one.jpg"
    image.write_bytes(b"x")

    assert len(script.expand_images([str(image), str(image)])) == 1


def test_relative_patterns_resolve_against_the_given_root(script, tmp_path):
    (tmp_path / "shot.jpg").write_bytes(b"x")

    found = script.expand_images(["*.jpg"], root=tmp_path)

    assert [path.name for path in found] == ["shot.jpg"]


def test_the_table_shows_both_groups_and_the_centre(script):
    text = script.format_table(
        [
            ("positive", "sheet.jpg", 0.2459, (0.483, 0.588)),
            ("negative", "empty.jpg", 0.0471, None),
        ]
    )

    assert "positive" in text and "negative" in text
    assert "0.246" in text and "0.047" in text
    assert "(0.483, 0.588)" in text
    assert text.splitlines()[0].lstrip().startswith("group")


def test_a_prompt_fixture_is_read_without_transformers(script):
    """The test rig may not carry `transformers`; the committed fixture is enough."""
    fixture = Path(__file__).resolve().parents[2] / "assets" / "prompts" / "steel_blank.json"

    query = script.load_query(fixture)

    assert query.labels == ("steel blank",)
    assert len(query.input_ids) == 16
    assert sum(query.attention_mask) == 5
