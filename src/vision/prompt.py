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
# date   : 2026-Oct-17
# ==============================================================================
"""User-configured detection prompts, tokenised for a specific camera.

The System User types what to look for ("metal sheet", "debris") and that becomes
the text conditioning for one camera's frames. This module owns the *pure* part of
that: prompt text construction, tokenisation and validation. Persistence lives in
`src/api/storage/prompt_repository.py` and the HTTP surface in
`src/api/prompts/` - so this module imports neither FastAPI nor SQLAlchemy and is
fully unit-testable offline.

Why tokenisation is an interface (`PromptTokenizer`):

* The real tokenizer is CLIP's, reached through HuggingFace `transformers`. That
  dependency is deliberately *not* in the worker image, and downloading a tokenizer
  at request time is not acceptable on an air-gapped industrial network.
* Tests need a deterministic tokenizer to assert the round trip without a 100 MB
  download.

`TransformersPromptTokenizer` is the production implementation: it loads from a
**local directory** by default (baked into the image at build time) and refuses to
reach the network unless explicitly allowed.

Validation matters here more than it looks. The graph's text path saturates at
-65504 when fed padding tokens, which decodes to *zero detections forever* while
every frame still "succeeds". A prompt that tokenises to padding-only is therefore
rejected when it is set, not silently accepted.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence, runtime_checkable

from src.vision.backend import DEFAULT_TEXT_LENGTH, TextQuery

__all__ = [
    "CameraPrompt",
    "PromptEncodingError",
    "PromptTokenizer",
    "TokenizerUnavailable",
    "TransformersPromptTokenizer",
    "build_prompt_text",
    "encode_prompt",
    "get_tokenizer",
]

logger = logging.getLogger(__name__)


class PromptEncodingError(ValueError):
    """The prompt cannot be turned into usable text conditioning."""


class TokenizerUnavailable(RuntimeError):
    """No tokenizer is configured, so prompts cannot be set at this time.

    Surfaced to the API as HTTP 503: the endpoint is correct but the service cannot
    perform the operation until the deployment provides a tokenizer.
    """


@runtime_checkable
class PromptTokenizer(Protocol):
    """Turns prompt text into fixed-length token IDs."""

    @property
    def name(self) -> str: ...

    @property
    def max_length(self) -> int: ...

    def encode(self, text: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return ``(input_ids, attention_mask)``, both ``max_length`` long."""


def build_prompt_text(labels: Sequence[str]) -> str:
    """Build OWLv2's prompt convention from object names.

    OWLv2 was trained on space-separated phrases, each terminated by a period, e.g.
    ``"metal sheet. detail."``. Each phrase is one class, and the graph's class
    dimension corresponds to them in order - which is why the order of ``labels``
    is preserved everywhere (storage, wire payload, decode).
    """
    cleaned = [label.strip() for label in labels if label and label.strip()]
    if not cleaned:
        raise PromptEncodingError("at least one object name is required")
    for label in cleaned:
        if "." in label:
            # A period would split one label into two phrases and shift every
            # subsequent class index.
            raise PromptEncodingError(
                f"object name {label!r} must not contain a period"
            )
        if len(label) > 64:
            raise PromptEncodingError(
                f"object name {label!r} is too long (max 64 characters)"
            )
    return " ".join(f"{label}." for label in cleaned)


class TransformersPromptTokenizer:
    """CLIP tokenizer via HuggingFace `transformers`.

    Loads from a local directory so an air-gapped deployment works: the tokenizer
    files are baked into the image or mounted with the model.
    """

    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        *,
        max_length: int = DEFAULT_TEXT_LENGTH,
        local_files_only: bool = True,
    ) -> None:
        self._model_id = model_id
        self._max_length = max_length
        self._local_files_only = local_files_only
        self._tokenizer = None

    @property
    def name(self) -> str:
        return f"transformers:{self._model_id}"

    @property
    def max_length(self) -> int:
        return self._max_length

    def _load(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the image
            raise TokenizerUnavailable(
                "the prompt tokenizer requires the `transformers` package; "
                "install it in the API image (export-time tier) or set prompts "
                "with scripts/encode_text_queries.py and OWL2_TEXT_IDS_PATH"
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id,
                local_files_only=self._local_files_only,
            )
        except OSError as exc:  # pragma: no cover - depends on the image
            raise TokenizerUnavailable(
                f"tokenizer {self._model_id!r} is not available locally and network "
                f"download is disabled: {exc}"
            ) from exc
        return self._tokenizer

    def encode(self, text: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        tokenizer = self._load()
        encoded = tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            max_length=self._max_length,
            truncation=True,
        )
        ids = tuple(int(value) for value in encoded["input_ids"][0])
        mask = tuple(int(value) for value in encoded["attention_mask"][0])
        return ids, mask


def get_tokenizer() -> PromptTokenizer:
    """Return the configured tokenizer, or raise :class:`TokenizerUnavailable`.

    Configuration: ``PROMPT_TOKENIZER_MODEL`` (defaults to the OWLv2 checkpoint),
    ``PROMPT_TOKENIZER_LOCAL_ONLY`` (default ``1`` - never download at runtime).
    """
    model_id = os.getenv(
        "PROMPT_TOKENIZER_MODEL", "google/owlv2-base-patch16-ensemble"
    )
    local_only = os.getenv("PROMPT_TOKENIZER_LOCAL_ONLY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    max_length = int(os.getenv("PROMPT_TOKENIZER_MAX_LENGTH", DEFAULT_TEXT_LENGTH))
    return TransformersPromptTokenizer(
        model_id, max_length=max_length, local_files_only=local_only
    )


@dataclass(frozen=True, slots=True)
class CameraPrompt:
    """The prompt in force for one camera, with its tokenised form.

    ``labels[i]`` corresponds to class ``i`` of the graph. For the single-class
    export in this repository only ``labels[0]`` is observable, but the full list is
    carried so a multi-class export works without any further change.
    """

    labels: tuple[str, ...]
    prompt_text: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    tokenizer: str = ""
    max_length: int = DEFAULT_TEXT_LENGTH
    camera_id: str = ""

    def __post_init__(self) -> None:
        if not self.labels:
            raise PromptEncodingError("a prompt needs at least one object name")
        if len(self.input_ids) != len(self.attention_mask):
            raise PromptEncodingError(
                "input_ids and attention_mask have different lengths "
                f"({len(self.input_ids)} vs {len(self.attention_mask)})"
            )
        if not self.input_ids:
            raise PromptEncodingError("the tokenised prompt is empty")
        if not any(self.attention_mask):
            raise PromptEncodingError(
                "the attention mask is empty: the prompt produced no usable tokens"
            )
        if all(value == 0 for value in self.input_ids):
            raise PromptEncodingError(
                "the prompt tokenised to padding only. This model's text path "
                "saturates at -65504 for padding input, so every frame would "
                "report no detections. Use object names the tokenizer recognises."
            )

    def to_query(self) -> TextQuery:
        """Return the `TextQuery` the inference backend consumes."""
        return TextQuery(
            labels=self.labels,
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
        )

    def to_dict(self) -> dict:
        """Serialise for storage and for the task payload."""
        return {
            "camera_id": self.camera_id,
            "labels": list(self.labels),
            "prompt_text": self.prompt_text,
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "tokenizer": self.tokenizer,
            "max_length": self.max_length,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CameraPrompt":
        """Rebuild from :meth:`to_dict`; rejects malformed stored data."""
        try:
            return cls(
                labels=tuple(str(label) for label in payload["labels"]),
                prompt_text=str(payload["prompt_text"]),
                input_ids=tuple(int(value) for value in payload["input_ids"]),
                attention_mask=tuple(
                    int(value) for value in payload["attention_mask"]
                ),
                tokenizer=str(payload.get("tokenizer", "")),
                max_length=int(payload.get("max_length", DEFAULT_TEXT_LENGTH)),
                camera_id=str(payload.get("camera_id", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PromptEncodingError(f"malformed stored prompt: {exc}") from exc


def encode_prompt(
    labels: Sequence[str],
    tokenizer: PromptTokenizer,
    *,
    camera_id: str = "",
) -> CameraPrompt:
    """Tokenise *labels* for one camera.

    Raises:
        PromptEncodingError: when the labels are invalid or tokenise to nothing
            usable. Validating here means a bad prompt is rejected at the API,
            where a human can fix it, rather than degrading detections silently.
    """
    text = build_prompt_text(labels)
    try:
        input_ids, attention_mask = tokenizer.encode(text)
    except (PromptEncodingError, TokenizerUnavailable):
        # TokenizerUnavailable must NOT be wrapped: it means the deployment lacks a
        # tokenizer, which the API maps to 503. Wrapping it made the 503 branch
        # unreachable and reported a missing package as if the user's prompt were
        # invalid - found by the live cluster smoke test.
        raise
    except Exception as exc:  # noqa: BLE001 - tokenizer backends vary
        raise PromptEncodingError(f"tokenisation failed: {exc}") from exc

    return CameraPrompt(
        labels=tuple(label.strip() for label in labels if label and label.strip()),
        prompt_text=text,
        input_ids=tuple(input_ids),
        attention_mask=tuple(attention_mask),
        tokenizer=tokenizer.name,
        max_length=tokenizer.max_length,
        camera_id=camera_id,
    )
