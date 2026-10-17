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
"""Tests for detection queries: one phrase per object, each with its own threshold.

A deterministic fake tokenizer keeps this offline; the semantics under test are the
query set's, not the tokenizer's.
"""

from __future__ import annotations

import pytest

from src.vision.prompt import (
    DEFAULT_THRESHOLD,
    CameraPrompt,
    CameraPromptSet,
    PromptEncodingError,
    TokenizerUnavailable,
    build_prompt_text,
    clamp_threshold,
    encode_prompt_set,
)


class FakeTokenizer:
    """Deterministic stand-in for CLIP's tokenizer."""

    name = "fake"
    max_length = 16

    def __init__(self, *, empty: bool = False) -> None:
        self._empty = empty
        self.seen: list[str] = []

    def encode(self, text: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        self.seen.append(text)
        if self._empty:
            return (0,) * 16, (0,) * 16
        ids = tuple((ord(ch) % 49407) or 1 for ch in text[:15])
        padding = 16 - len(ids)
        return ids + (49407,) * padding, (1,) * len(ids) + (0,) * padding


# --------------------------------------------------------------------------- #
# Text and thresholds
# --------------------------------------------------------------------------- #
def test_prompt_text_follows_the_owlv2_convention() -> None:
    assert build_prompt_text("metal sheet") == "metal sheet."
    assert build_prompt_text(" robot ") == "robot."


@pytest.mark.parametrize("label", ["", "   "])
def test_prompt_text_requires_a_label(label: str) -> None:
    with pytest.raises(PromptEncodingError):
        build_prompt_text(label)


def test_prompt_text_rejects_a_period_inside_a_label() -> None:
    """A period would split one phrase into two and shift every later query index."""
    with pytest.raises(PromptEncodingError, match="period"):
        build_prompt_text("metal. sheet")


def test_threshold_bounds_are_enforced() -> None:
    assert clamp_threshold(0.1) == 0.1
    assert clamp_threshold("0.25") == 0.25
    for bad in (0.0, 1.0, -0.5, 1.5, float("nan")):
        with pytest.raises(PromptEncodingError):
            clamp_threshold(bad)


def test_threshold_rejects_non_numeric() -> None:
    with pytest.raises(PromptEncodingError, match="number"):
        clamp_threshold("high")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Query sets
# --------------------------------------------------------------------------- #
def test_encode_prompt_set_creates_one_query_per_object() -> None:
    tokenizer = FakeTokenizer()
    prompt_set = encode_prompt_set(
        [("robot", 0.25), ("person", 0.1)], tokenizer, camera_id="cam0"
    )

    assert prompt_set.labels == ("robot", "person")
    assert prompt_set.thresholds == (0.25, 0.1)
    assert len(prompt_set.queries) == 2
    assert tokenizer.seen == ["robot.", "person."]


def test_query_set_produces_one_text_row_per_query() -> None:
    """The batch dimension is the query axis: this is what makes one pass enough."""
    prompt_set = encode_prompt_set([("robot", 0.25), ("person", 0.1)], FakeTokenizer())
    ids, mask = prompt_set.as_arrays(expected_length=16)

    assert ids.shape == (2, 16)
    assert mask.shape == (2, 16)


def test_thresholds_survive_the_round_trip() -> None:
    prompt_set = encode_prompt_set([("robot", 0.31), ("person", 0.12)], FakeTokenizer())
    restored = CameraPromptSet.from_dict(prompt_set.to_dict())

    assert restored.thresholds == (0.31, 0.12)
    assert restored.labels == ("robot", "person")


def test_a_query_set_needs_at_least_one_query() -> None:
    with pytest.raises(PromptEncodingError):
        CameraPromptSet(queries=())
    with pytest.raises(PromptEncodingError):
        encode_prompt_set([], FakeTokenizer())


def test_duplicate_labels_are_rejected() -> None:
    """Two queries for the same phrase would split one class's detections in two."""
    with pytest.raises(PromptEncodingError, match="duplicate"):
        encode_prompt_set([("robot", 0.25), ("robot", 0.5)], FakeTokenizer())


def test_inconsistent_token_lengths_are_rejected() -> None:
    short = CameraPrompt(
        label="a",
        threshold=0.3,
        prompt_text="a.",
        input_ids=(1, 2),
        attention_mask=(1, 1),
    )
    long = CameraPrompt(
        label="b",
        threshold=0.3,
        prompt_text="b.",
        input_ids=(1, 2, 3),
        attention_mask=(1, 1, 1),
    )
    with pytest.raises(PromptEncodingError, match="inconsistent"):
        CameraPromptSet(queries=(short, long))


def test_as_arrays_checks_the_expected_length() -> None:
    prompt_set = encode_prompt_set([("robot", 0.25)], FakeTokenizer())
    with pytest.raises(ValueError, match="expects"):
        prompt_set.as_arrays(expected_length=32)


# --------------------------------------------------------------------------- #
# Validation of a single query
# --------------------------------------------------------------------------- #
def test_padding_only_tokens_are_rejected() -> None:
    """The silent-outage guard: padding saturates the graph's text path."""
    with pytest.raises(PromptEncodingError, match="padding|no usable tokens"):
        encode_prompt_set([("robot", 0.25)], FakeTokenizer(empty=True))


def test_mismatched_mask_length_is_rejected() -> None:
    with pytest.raises(PromptEncodingError, match="different lengths"):
        CameraPrompt(
            label="a",
            threshold=0.3,
            prompt_text="a.",
            input_ids=(1, 2, 3),
            attention_mask=(1, 1),
        )


def test_empty_attention_mask_is_rejected() -> None:
    with pytest.raises(PromptEncodingError, match="mask"):
        CameraPrompt(
            label="a",
            threshold=0.3,
            prompt_text="a.",
            input_ids=(1, 2),
            attention_mask=(0, 0),
        )


def test_default_threshold_is_used_when_unspecified() -> None:
    assert DEFAULT_THRESHOLD == 0.25


# --------------------------------------------------------------------------- #
# Legacy shapes
# --------------------------------------------------------------------------- #
def test_a_legacy_single_prompt_row_is_still_readable() -> None:
    legacy = {
        "camera_id": "cam0",
        "labels": ["metal sheet"],
        "prompt_text": "metal sheet.",
        "input_ids": [49406, 4044, 7298, 269, 49407] + [0] * 11,
        "attention_mask": [1, 1, 1, 1, 1] + [0] * 11,
        "tokenizer": "transformers:x",
        "max_length": 16,
    }

    prompt_set = CameraPromptSet.from_dict(legacy)

    assert prompt_set.labels == ("metal sheet",)
    # No threshold was stored before thresholds existed; the default applies.
    assert prompt_set.thresholds == (DEFAULT_THRESHOLD,)


def test_a_legacy_multi_label_row_expands_to_one_query_each() -> None:
    legacy = {
        "camera_id": "cam0",
        "labels": ["metal sheet", "debris"],
        "prompt_text": "metal sheet. debris.",
        "input_ids": [49406, 1, 2, 49407] + [0] * 12,
        "attention_mask": [1, 1, 1, 1] + [0] * 12,
        "max_length": 16,
    }

    prompt_set = CameraPromptSet.from_dict(legacy)

    assert prompt_set.labels == ("metal sheet", "debris")


def test_malformed_stored_data_is_rejected() -> None:
    with pytest.raises(PromptEncodingError, match="malformed"):
        CameraPromptSet.from_dict({"nothing": "useful"})


# --------------------------------------------------------------------------- #
# Tokenizer availability
# --------------------------------------------------------------------------- #
def test_tokenizer_unavailable_is_not_reported_as_a_prompt_error() -> None:
    """A missing tokenizer is a deployment problem (503), not a bad prompt (422)."""

    class Unavailable:
        name = "unavailable"
        max_length = 16

        def encode(self, text: str):
            raise TokenizerUnavailable("no tokenizer")

    with pytest.raises(TokenizerUnavailable):
        encode_prompt_set([("robot", 0.25)], Unavailable())
