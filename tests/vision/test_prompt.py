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
"""Tests for prompt construction and tokenisation.

A deterministic fake tokenizer is used so the round trip is asserted without a
network download; the HuggingFace implementation is exercised only where it is
available.
"""

from __future__ import annotations

import pytest

from src.vision.prompt import (
    CameraPrompt,
    PromptEncodingError,
    TransformersPromptTokenizer,
    build_prompt_text,
    encode_prompt,
    get_tokenizer,
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
        # A stable pseudo-token id per character, bounded to the CLIP vocabulary.
        ids = tuple((ord(ch) % 49407) or 1 for ch in text[:15])
        padding = 16 - len(ids)
        return ids + (49407,) * padding, (1,) * len(ids) + (0,) * padding


# --------------------------------------------------------------------------- #
# Prompt text
# --------------------------------------------------------------------------- #
def test_prompt_text_follows_the_owlv2_convention() -> None:
    assert build_prompt_text(["metal sheet"]) == "metal sheet."
    assert build_prompt_text(["metal sheet", "debris"]) == "metal sheet. debris."


def test_prompt_text_trims_and_drops_blank_entries() -> None:
    assert build_prompt_text([" metal sheet ", "", "  "]) == "metal sheet."


def test_prompt_text_requires_at_least_one_label() -> None:
    with pytest.raises(PromptEncodingError, match="at least one"):
        build_prompt_text([])
    with pytest.raises(PromptEncodingError):
        build_prompt_text(["   "])


def test_prompt_text_rejects_a_period_inside_a_label() -> None:
    """A period would split one label into two and shift every class index."""
    with pytest.raises(PromptEncodingError, match="period"):
        build_prompt_text(["metal. sheet"])


def test_prompt_text_rejects_an_over_long_label() -> None:
    with pytest.raises(PromptEncodingError, match="too long"):
        build_prompt_text(["x" * 65])


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def test_encode_prompt_produces_a_consistent_encoding() -> None:
    tokenizer = FakeTokenizer()
    prompt = encode_prompt(["metal sheet"], tokenizer, camera_id="cam0")

    assert prompt.camera_id == "cam0"
    assert prompt.labels == ("metal sheet",)
    assert prompt.prompt_text == "metal sheet."
    assert len(prompt.input_ids) == len(prompt.attention_mask) == 16
    assert prompt.tokenizer == "fake"
    assert tokenizer.seen == ["metal sheet."]


def test_encode_prompt_rejects_padding_only_tokens() -> None:
    """The silent-outage guard: padding input saturates the graph's text path."""
    with pytest.raises(PromptEncodingError, match="padding|no usable tokens"):
        encode_prompt(["metal sheet"], FakeTokenizer(empty=True))


def test_encoding_round_trips_through_a_dict() -> None:
    prompt = encode_prompt(["metal sheet", "debris"], FakeTokenizer(), camera_id="cam1")
    restored = CameraPrompt.from_dict(prompt.to_dict())

    assert restored == prompt
    assert restored.to_query().labels == ("metal sheet", "debris")


def test_from_dict_rejects_malformed_stored_data() -> None:
    with pytest.raises(PromptEncodingError, match="malformed"):
        CameraPrompt.from_dict({"labels": ["a"]})


def test_prompt_rejects_mismatched_mask_length() -> None:
    with pytest.raises(PromptEncodingError, match="different lengths"):
        CameraPrompt(
            labels=("a",),
            prompt_text="a.",
            input_ids=(1, 2, 3),
            attention_mask=(1, 1),
        )


def test_prompt_rejects_an_empty_attention_mask() -> None:
    with pytest.raises(PromptEncodingError, match="mask"):
        CameraPrompt(
            labels=("a",),
            prompt_text="a.",
            input_ids=(1, 2),
            attention_mask=(0, 0),
        )


def test_to_query_is_accepted_by_the_inference_backend() -> None:
    """The prompt must satisfy the same validation the backend applies."""
    prompt = encode_prompt(["metal sheet"], FakeTokenizer())
    query = prompt.to_query()
    ids, mask = query.as_arrays(expected_length=16)
    assert ids.shape == (1, 16)
    assert mask.shape == (1, 16)


# --------------------------------------------------------------------------- #
# Tokenizer availability
# --------------------------------------------------------------------------- #
def test_get_tokenizer_returns_a_transformers_backend() -> None:
    tokenizer = get_tokenizer()
    assert isinstance(tokenizer, TransformersPromptTokenizer)
    assert tokenizer.max_length == 16


def test_transformers_tokenizer_fails_loudly_when_unavailable() -> None:
    """Without the package or local files, setting a prompt must error clearly."""
    tokenizer = TransformersPromptTokenizer("definitely-not-a-real-model-xyz")
    try:
        tokenizer.encode("metal sheet.")
    except Exception as exc:  # noqa: BLE001 - exact type depends on the image
        assert "transformers" in str(exc) or "not available" in str(exc)
    else:
        pytest.skip("transformers is installed with a usable offline cache")
