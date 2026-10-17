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
"""User-configured detection prompts: one query per object, each with a threshold.

How OWLv2 actually consumes prompts, and why this module looks the way it does:

* The graph's text input has a **batch** dimension and the class dimension of `logits`
  follows it: ``input_ids`` of shape ``(N, 16)`` returns ``logits (1, 3600, N)``.
  Measured on the shipped artifact with `scripts/probe_query_axis.py`: 1, 2 and 3
  prompts give 1, 2 and 3 query channels.
* So multi-object detection is **one forward pass with every label in the text batch**,
  then choosing, per box, the query that scored it highest - the same approach as
  `scripts/proof_of_work.py`, at the cost of a single prompt.
* Each query carries **its own threshold**. That is what makes a human-presence
  interlock possible without lowering the threshold for actuation: "human" at 0.10 and
  "metal sheet" at 0.25 can coexist on the same camera.

The System User therefore configures a *set* of queries per camera, not one prompt.

Validation still refuses a query that tokenises to padding: that state saturates the
graph's fp16 text path at -65504 and silently disables detection.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from src.vision.backend import DEFAULT_TEXT_LENGTH

__all__ = [
    "CameraPrompt",
    "CameraPromptSet",
    "DEFAULT_THRESHOLD",
    "MAX_THRESHOLD",
    "MIN_THRESHOLD",
    "PromptEncodingError",
    "PromptTokenizer",
    "TokenizerUnavailable",
    "TransformersPromptTokenizer",
    "build_prompt_text",
    "clamp_threshold",
    "encode_prompt_set",
    "get_tokenizer",
]

logger = logging.getLogger(__name__)

#: Default confidence threshold. OWLv2 zero-shot sigmoid scores are typically 0.1-0.5,
#: so 0.25 is a reasonable "something is there" line for rigid objects. Classes that
#: matter for safety should be configured lower - see doc/vision-pipeline.md §14.
DEFAULT_THRESHOLD = 0.25

#: Bounds so a typo cannot turn detection into "everything" or "nothing".
MIN_THRESHOLD = 0.01
MAX_THRESHOLD = 0.99


class PromptEncodingError(ValueError):
    """The prompt cannot be turned into usable text conditioning."""


class TokenizerUnavailable(RuntimeError):
    """No tokenizer is configured, so prompts cannot be set at this time."""


@runtime_checkable
class PromptTokenizer(Protocol):
    """Turns prompt text into fixed-length token IDs."""

    @property
    def name(self) -> str: ...

    @property
    def max_length(self) -> int: ...

    def encode(self, text: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return ``(input_ids, attention_mask)``, both ``max_length`` long."""


def build_prompt_text(label: str) -> str:
    """Build OWLv2's prompt convention for one object name.

    OWLv2 expects a short noun phrase terminated by a period, e.g. ``"metal sheet."``
    One phrase is one query, and each query becomes its own class channel.
    """
    cleaned = (label or "").strip()
    if not cleaned:
        raise PromptEncodingError("an object name is required")
    if "." in cleaned:
        # A period would split one phrase into two and shift every later query index.
        raise PromptEncodingError(f"object name {cleaned!r} must not contain a period")
    if len(cleaned) > 64:
        raise PromptEncodingError(f"object name {cleaned!r} is too long (max 64 chars)")
    return f"{cleaned}."


def clamp_threshold(threshold: float) -> float:
    """Validate one query's confidence threshold."""
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise PromptEncodingError(
            f"threshold must be a number, got {threshold!r}"
        ) from exc
    if value != value:  # NaN
        raise PromptEncodingError("threshold must be a number, got NaN")
    if not MIN_THRESHOLD <= value <= MAX_THRESHOLD:
        raise PromptEncodingError(
            f"threshold {value} is outside {MIN_THRESHOLD}..{MAX_THRESHOLD}"
        )
    return value


class TransformersPromptTokenizer:
    """CLIP tokenizer via HuggingFace `transformers`, loaded from a local cache.

    Offline by default because an industrial network has no egress at runtime; the
    tokenizer files are baked into the API image at build time.
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
                "the prompt tokenizer requires the `transformers` package; it is "
                "baked into the API image at build time (src/api/Dockerfile)"
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id, local_files_only=self._local_files_only
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
    """Return the configured tokenizer, or raise :class:`TokenizerUnavailable`."""
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
    """One detection query: a phrase, its tokens and its own threshold."""

    label: str
    threshold: float
    prompt_text: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    camera_id: str = ""
    tokenizer: str = ""
    max_length: int = DEFAULT_TEXT_LENGTH

    def __post_init__(self) -> None:
        if not self.label or not self.label.strip():
            raise PromptEncodingError("a query needs an object name")
        object.__setattr__(self, "threshold", clamp_threshold(self.threshold))
        if len(self.input_ids) != len(self.attention_mask):
            raise PromptEncodingError(
                "input_ids and attention_mask have different lengths "
                f"({len(self.input_ids)} vs {len(self.attention_mask)})"
            )
        if not self.input_ids:
            raise PromptEncodingError("the tokenised query is empty")
        if not any(self.attention_mask):
            raise PromptEncodingError(
                "the attention mask is empty: the query produced no usable tokens"
            )
        if all(value == 0 for value in self.input_ids):
            raise PromptEncodingError(
                "the query tokenised to padding only. The model's text path saturates "
                "at -65504 for padding input, so detections would silently vanish."
            )

    @property
    def labels(self) -> tuple[str, ...]:
        """Backwards-compatible single-label view."""
        return (self.label,)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "threshold": self.threshold,
            "prompt_text": self.prompt_text,
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "camera_id": self.camera_id,
            "tokenizer": self.tokenizer,
            "max_length": self.max_length,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CameraPrompt":
        """Rebuild one query. Accepts the legacy single-prompt shape too."""
        try:
            if "label" not in payload:
                # Legacy rows stored labels/input_ids for one joined prompt.
                labels = list(payload["labels"])
                if len(labels) != 1:
                    raise PromptEncodingError(
                        "legacy prompt with several labels cannot be migrated to one "
                        "query; set the prompt again"
                    )
                return cls(
                    label=str(labels[0]),
                    threshold=payload.get("threshold", DEFAULT_THRESHOLD),
                    prompt_text=str(payload.get("prompt_text", f"{labels[0]}.")),
                    input_ids=tuple(int(v) for v in payload["input_ids"]),
                    attention_mask=tuple(int(v) for v in payload["attention_mask"]),
                    camera_id=str(payload.get("camera_id", "")),
                    tokenizer=str(payload.get("tokenizer", "")),
                    max_length=int(payload.get("max_length", DEFAULT_TEXT_LENGTH)),
                )
            return cls(
                label=str(payload["label"]),
                threshold=payload.get("threshold", DEFAULT_THRESHOLD),
                prompt_text=str(payload["prompt_text"]),
                input_ids=tuple(int(v) for v in payload["input_ids"]),
                attention_mask=tuple(int(v) for v in payload["attention_mask"]),
                camera_id=str(payload.get("camera_id", "")),
                tokenizer=str(payload.get("tokenizer", "")),
                max_length=int(payload.get("max_length", DEFAULT_TEXT_LENGTH)),
            )
        except PromptEncodingError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise PromptEncodingError(f"malformed stored prompt: {exc}") from exc


@dataclass(frozen=True, slots=True)
class CameraPromptSet:
    """The queries in force for one camera, sent as one text batch."""

    queries: tuple[CameraPrompt, ...]
    camera_id: str = ""
    tokenizer: str = ""
    max_length: int = DEFAULT_TEXT_LENGTH

    def __post_init__(self) -> None:
        if not self.queries:
            raise PromptEncodingError("a camera needs at least one detection query")
        lengths = {len(query.input_ids) for query in self.queries}
        if len(lengths) != 1:
            raise PromptEncodingError(
                f"queries have inconsistent token lengths {sorted(lengths)}; all must "
                "come from the same tokenizer configuration"
            )
        labels = [query.label for query in self.queries]
        if len(set(labels)) != len(labels):
            raise PromptEncodingError(f"duplicate object names in {labels}")

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(query.label for query in self.queries)

    @property
    def thresholds(self) -> tuple[float, ...]:
        return tuple(query.threshold for query in self.queries)

    def as_arrays(
        self, expected_length: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(input_ids, attention_mask)`` shaped ``(Q, L)`` int64.

        The batch dimension is the *query* axis: the graph returns one logits channel
        per row, which is what makes multi-object detection a single forward pass.
        """
        if expected_length is not None:
            for query in self.queries:
                if len(query.input_ids) != expected_length:
                    raise ValueError(
                        f"query {query.label!r} has {len(query.input_ids)} tokens but "
                        f"the model expects {expected_length}; re-tokenise with "
                        "matching padding (scripts/encode_text_queries.py)"
                    )
        ids = np.asarray([query.input_ids for query in self.queries], dtype=np.int64)
        mask = np.asarray(
            [query.attention_mask for query in self.queries], dtype=np.int64
        )
        return ids, mask

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "tokenizer": self.tokenizer,
            "max_length": self.max_length,
            "queries": [query.to_dict() for query in self.queries],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CameraPromptSet":
        """Rebuild a set, accepting the legacy single-prompt shape."""
        # "prompts" is the HTTP field name; "queries" is the wire name. Accept both so
        # an operator replaying an API response as a task payload is not silently
        # ignored - which is exactly what happened the first time this was tried live.
        if "queries" in payload or "prompts" in payload:
            items = payload.get("queries") or payload.get("prompts") or []
            queries = tuple(CameraPrompt.from_dict(item) for item in items)
            return cls(
                queries=queries,
                camera_id=str(payload.get("camera_id", "")),
                tokenizer=str(payload.get("tokenizer", "")),
                max_length=int(payload.get("max_length", DEFAULT_TEXT_LENGTH)),
            )
        if "labels" in payload:
            queries = []
            for label in payload["labels"]:
                legacy = dict(payload)
                legacy["labels"] = [label]
                queries.append(CameraPrompt.from_dict(legacy))
            return cls(
                queries=tuple(queries),
                camera_id=str(payload.get("camera_id", "")),
                tokenizer=str(payload.get("tokenizer", "")),
                max_length=int(payload.get("max_length", DEFAULT_TEXT_LENGTH)),
            )
        raise PromptEncodingError(
            "malformed prompt set: neither 'queries' nor 'labels'"
        )


def encode_prompt_set(
    items: Sequence[tuple[str, float]],
    tokenizer: PromptTokenizer,
    *,
    camera_id: str = "",
) -> CameraPromptSet:
    """Tokenise ``(label, threshold)`` pairs into one query set.

    Each pair becomes its own query and its own class channel, so a camera can ask for
    "metal sheet" at 0.25 and "human" at 0.10 in the same forward pass.
    """
    if not items:
        raise PromptEncodingError("at least one detection query is required")

    queries: list[CameraPrompt] = []
    for label, threshold in items:
        text = build_prompt_text(label)
        try:
            input_ids, attention_mask = tokenizer.encode(text)
        except (PromptEncodingError, TokenizerUnavailable):
            raise
        except Exception as exc:  # noqa: BLE001 - tokenizer backends vary
            raise PromptEncodingError(f"tokenisation failed: {exc}") from exc

        queries.append(
            CameraPrompt(
                label=label.strip(),
                threshold=threshold,
                prompt_text=text,
                input_ids=tuple(input_ids),
                attention_mask=tuple(attention_mask),
                camera_id=camera_id,
                tokenizer=tokenizer.name,
                max_length=tokenizer.max_length,
            )
        )

    return CameraPromptSet(
        queries=tuple(queries),
        camera_id=camera_id,
        tokenizer=tokenizer.name,
        max_length=tokenizer.max_length,
    )
