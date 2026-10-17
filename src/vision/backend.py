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
"""The inference backend seam.

Everything downstream of this module (worker, dispatcher, capture loop) is written
against :class:`InferenceBackend` rather than against ONNX Runtime. Three reasons,
in order of importance:

1. **The 60 FPS target must be testable without a GPU.** The pipeline's ability to
   capture at 60 FPS, shard across workers, shed load and stay under its latency
   budget is a property of the *orchestration*, not of OWLv2. A synthetic backend
   lets `tests/pipeline` prove that property deterministically; the real backend's
   cost per frame is then a capacity input (measured), not an unknown.
2. **GPU/CPU parity.** The provider is a configuration detail; the worker code must
   not branch on it.
3. **Testability.** A backend that returns a known answer makes decode and
   end-to-end assertions exact.

The real backend wraps the exported ONNX graph. Its text conditioning is supplied
as pre-tokenised IDs (see :class:`TextQuery`), because:

* tokenising requires the CLIP tokenizer (a `transformers` dependency the worker
  image does not otherwise need), and the prompt only changes when the model or
  the target class changes - not per frame;
* the graph in this repository was exported with **random** dummy token IDs
  (`src/owl/inference/utils.py::_create_dummy_inputs` uses
  ``torch.randint(0, 30522, ...)``), so the IDs that actually select "metal
  sheet" are deployment data, not something the artifact carries. Passing zeros
  saturates the graph's fp16 text path at -65504 and yields no detections at all;
  :class:`TextQuery` refuses that configuration rather than failing silently.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = [
    "InferenceBackend",
    "InferenceOutput",
    "Owl2OnnxBackend",
    "SyntheticBackend",
    "TextQuery",
    "build_backend",
    "load_text_query",
]

logger = logging.getLogger(__name__)

#: The graph's text input length, read from the artifact: `input_ids` is
#: `(batch, 16)`. Tokenised prompts must be padded/truncated to exactly this.
DEFAULT_TEXT_LENGTH = 16

#: Logits this close to the fp16 maximum are a saturation artefact, not a score.
#: Observed when the text input is degenerate (e.g. all padding): the whole logits
#: tensor reads exactly -65504 and every sigmoid is 0, which looks like "no objects
#: in view" forever. Failing loudly turns a silent production outage into an
#: obvious startup error.
FP16_SATURATION_THRESHOLD = 60000.0


@dataclass(frozen=True, slots=True)
class InferenceOutput:
    """Raw graph outputs: ``logits (1, Q, C)`` and ``pred_boxes (1, Q, 4)``.

    Kept as the raw arrays (rather than decoded detections) so that decoding
    stays a separate, independently testable step and so a batch of N frames
    decodes with one pass per item.
    """

    logits: np.ndarray
    pred_boxes: np.ndarray


@dataclass(frozen=True, slots=True)
class TextQuery:
    """Pre-tokenised text conditioning for the graph.

    ``labels`` names what the classes mean. The artifact in this repository emits
    a single class per query, so ``labels[0]`` is the object type reported for
    every detection; a multi-query export maps ``labels[i]`` to class ``i``.
    """

    labels: tuple[str, ...]
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("TextQuery needs at least one label")
        if len(self.input_ids) != len(self.attention_mask):
            raise ValueError(
                f"input_ids/attention_mask length mismatch: "
                f"{len(self.input_ids)} vs {len(self.attention_mask)}"
            )
        if not self.input_ids:
            raise ValueError("TextQuery has empty input_ids")
        if all(value == 0 for value in self.input_ids):
            raise ValueError(
                "input_ids are all zero. Zero is the padding token, and feeding "
                "only padding saturates this graph's fp16 text path at -65504, "
                "producing zero detections for every frame. Provide the real "
                "tokenised prompt (scripts/encode_text_queries.py)."
            )
        if not any(self.attention_mask):
            raise ValueError("attention_mask is all zero; nothing would be attended")

    def as_arrays(self, expected_length: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(input_ids, attention_mask)`` shaped ``(1, L)`` int64.

        Raises:
            ValueError: when ``expected_length`` is given and the query does not
                match it. A mismatch means the graph and the tokenised prompt were
                produced by different exports, which cannot work.
        """
        if expected_length is not None and len(self.input_ids) != expected_length:
            raise ValueError(
                f"text query has {len(self.input_ids)} tokens but the model expects "
                f"{expected_length}; re-tokenise with matching padding "
                "(scripts/encode_text_queries.py)"
            )
        ids = np.asarray(self.input_ids, dtype=np.int64)[np.newaxis, ...]
        mask = np.asarray(self.attention_mask, dtype=np.int64)[np.newaxis, ...]
        return ids, mask


def load_text_query(path: str | Path | None = None) -> TextQuery:
    """Load the tokenised prompt from JSON.

    Resolution order: explicit ``path``, then ``OWL2_TEXT_IDS_PATH``, then
    ``src/owl/models/text_queries.json`` next to the model. The file is produced
    once per deployment by ``scripts/encode_text_queries.py`` because tokenising
    needs the CLIP tokenizer.

    Raises:
        FileNotFoundError: with instructions, because "the prompt file is missing"
            is a deployment mistake that deserves a precise message.
    """
    resolved = path or os.getenv("OWL2_TEXT_IDS_PATH")
    if resolved is None:
        resolved = Path(__file__).resolve().parents[1] / "owl" / "models" / "text_queries.json"
    resolved = Path(resolved)

    if not resolved.exists():
        raise FileNotFoundError(
            f"no tokenised text query at {resolved}. Generate it once with:\n"
            "  python scripts/encode_text_queries.py --labels metal_sheet "
            "--output src/owl/models/text_queries.json\n"
            "(needs the `transformers` package; it is an export-time dependency)"
        )

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    try:
        return TextQuery(
            labels=tuple(payload["labels"]),
            input_ids=tuple(int(v) for v in payload["input_ids"]),
            attention_mask=tuple(int(v) for v in payload["attention_mask"]),
        )
    except KeyError as exc:
        raise ValueError(
            f"{resolved} is missing the {exc.args[0]!r} key; expected keys "
            "labels, input_ids, attention_mask"
        ) from exc


@runtime_checkable
class InferenceBackend(Protocol):
    """What the worker needs from an inference engine."""

    name: str

    def warmup(self) -> None:
        """Run one throwaway inference so the first real frame is not slow."""

    def infer(
        self,
        pixel_values: np.ndarray,
        text_query: "TextQuery | None" = None,
    ) -> InferenceOutput:
        """Run inference on a ``(N, 3, H, W)`` float32 batch.

        ``text_query`` overrides the configured prompt for this call,
        which is how a per-camera prompt reaches the graph. Backends that
        do not consume text accept and ignore it.
        """

    def close(self) -> None:
        """Release resources."""


class Owl2OnnxBackend:
    """OWLv2 on ONNX Runtime.

    The session is created lazily and cached for the process lifetime: creating it
    loads ~293 MB of weights, so doing it per task would dominate every frame.

    Provider selection honours ``OWL2_EXECUTION_PROVIDER`` and defaults to CUDA
    when the installed runtime offers it. Note that a CPU-only ``onnxruntime``
    wheel reports only ``CPUExecutionProvider`` - installing ``onnxruntime-gpu``
    is what makes the 60 FPS budget reachable, which `doc/vision-pipeline.md` §5
    quantifies.
    """

    name = "owl2-onnx"

    def __init__(
        self,
        model_path: str | Path,
        *,
        execution_provider: str | None = None,
        text_query: TextQuery | None = None,
        intra_op_threads: int | None = None,
        labels: Sequence[str] | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._requested_provider = execution_provider or os.getenv(
            "OWL2_EXECUTION_PROVIDER", ""
        )
        self._text_query = text_query
        if labels:
            # Allow the label vocabulary to come from configuration even when the
            # tokenised file names something else.
            if self._text_query is None:
                raise ValueError("labels= requires a text_query (tokenised prompt)")
            self._text_query = TextQuery(
                labels=tuple(labels),
                input_ids=self._text_query.input_ids,
                attention_mask=self._text_query.attention_mask,
            )
        self._intra_op_threads = intra_op_threads or int(
            os.getenv("OWL2_INTRA_OP_THREADS", "0")
        )
        self._session = None
        self._input_ids: Optional[np.ndarray] = None
        self._attention_mask: Optional[np.ndarray] = None
        self._text_length: Optional[int] = None
        self._providers: tuple[str, ...] = ()

    # -- lifecycle ---------------------------------------------------------
    def _provider_choice(self) -> list[str]:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if self._requested_provider:
            if self._requested_provider not in available:
                raise RuntimeError(
                    f"OWL2_EXECUTION_PROVIDER={self._requested_provider!r} is not "
                    f"available; installed providers are {available}. Install "
                    "onnxruntime-gpu for CUDAExecutionProvider."
                )
            return [self._requested_provider]
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _ensure_session(self):
        if self._session is not None:
            return self._session

        import onnxruntime as ort

        if not self._model_path.exists():
            raise FileNotFoundError(f"model not found at {self._model_path}")

        providers = self._provider_choice()
        options = ort.SessionOptions()
        if self._intra_op_threads:
            options.intra_op_num_threads = self._intra_op_threads
        # Keep the graph optimiser at the default level: EXTENDED is applied
        # automatically for the CUDA provider, and re-applying it on every worker
        # start would add seconds to each replica's cold start for no benefit.

        logger.info("loading OWLv2 from %s with providers %s", self._model_path, providers)
        started = time.perf_counter()
        session = ort.InferenceSession(str(self._model_path), options, providers=providers)
        self._session = session
        self._providers = tuple(session.get_providers())
        logger.info(
            "OWLv2 loaded in %.1fs using %s",
            time.perf_counter() - started,
            self._providers,
        )

        # The text inputs are static per deployment: validate once, then reuse.
        text_input = next(
            (i for i in session.get_inputs() if i.name == "input_ids"), None
        )
        expected_length = None
        if text_input is not None and isinstance(text_input.shape[-1], int):
            expected_length = text_input.shape[-1]
        if self._text_query is None:
            self._text_query = load_text_query()
        self._text_length = expected_length
        self._input_ids, self._attention_mask = self._text_query.as_arrays(
            expected_length
        )
        return session

    def warmup(self) -> None:
        """One throwaway pass, so the first real frame is not the slow one."""
        session = self._ensure_session()
        height, width = self._expected_image_size(session)
        dummy = np.zeros((1, 3, height, width), dtype=np.float32)
        self.infer(dummy)

    @staticmethod
    def _expected_image_size(session) -> tuple[int, int]:
        shape = next(i for i in session.get_inputs() if i.name == "pixel_values").shape
        try:
            return int(shape[2]), int(shape[3])
        except (TypeError, ValueError) as exc:  # pragma: no cover - dynamic export
            raise RuntimeError(
                f"pixel_values has a dynamic spatial shape {shape}; this pipeline "
                "expects the fixed 960x960 export"
            ) from exc

    def close(self) -> None:
        self._session = None

    # -- inference ---------------------------------------------------------
    @property
    def providers(self) -> tuple[str, ...]:
        return self._providers

    @property
    def text_query(self) -> Optional[TextQuery]:
        return self._text_query

    def infer(
        self,
        pixel_values: np.ndarray,
        text_query: TextQuery | None = None,
    ) -> InferenceOutput:
        session = self._ensure_session()
        if pixel_values.ndim != 4:
            raise ValueError(
                f"expected a (N, 3, H, W) batch, got shape {pixel_values.shape}"
            )
        if pixel_values.dtype != np.float32:
            # Copy explicitly rather than letting ONNX Runtime do it internally,
            # where the cost is invisible.
            pixel_values = pixel_values.astype(np.float32, copy=False)

        batch = pixel_values.shape[0]
        if text_query is not None:
            # Per-camera prompt: validated against the graph's text length.
            ids, mask = text_query.as_arrays(self._text_length)
        else:
            ids = self._input_ids
            mask = self._attention_mask
        if batch != 1:
            ids = np.repeat(ids, batch, axis=0)
            mask = np.repeat(mask, batch, axis=0)

        logits, pred_boxes = session.run(
            None,
            {
                "pixel_values": pixel_values,
                "input_ids": ids,
                "attention_mask": mask,
            },
        )
        self._assert_not_saturated(logits)
        return InferenceOutput(logits=logits, pred_boxes=pred_boxes)

    @staticmethod
    def _assert_not_saturated(logits: np.ndarray) -> None:
        peak = float(np.abs(np.asarray(logits, dtype=np.float32)).max())
        if peak >= FP16_SATURATION_THRESHOLD:
            raise RuntimeError(
                f"model logits saturated at {peak:.0f} (fp16 limit). This happens "
                "when the text input is degenerate (e.g. all padding tokens) and "
                "would produce zero detections forever. Check "
                "OWL2_TEXT_IDS_PATH / scripts/encode_text_queries.py."
            )


class SyntheticBackend:
    """Deterministic stand-in for load tests and pipeline tests.

    Returns a fixed number of high-confidence boxes derived from a hash of the
    image, so results are reproducible while still varying frame to frame. This is
    what allows `tests/pipeline` to prove the 60 FPS orchestration without a GPU.

    Cost is intentionally ~0 ms; add ``simulate_latency_ms`` to model a slow
    backend and exercise the backpressure/drop paths realistically.
    """

    name = "synthetic"

    def __init__(
        self,
        *,
        num_queries: int = 3600,
        num_detections: int = 2,
        score: float = 0.9,
        simulate_latency_ms: float = 0.0,
    ) -> None:
        self._num_queries = num_queries
        self._num_detections = min(num_detections, num_queries)
        self._score = score
        self._simulate_latency_ms = simulate_latency_ms

    def warmup(self) -> None:  # pragma: no cover - nothing to warm up
        return

    def close(self) -> None:  # pragma: no cover - nothing to release
        return

    def infer(
        self,
        pixel_values: np.ndarray,
        text_query: TextQuery | None = None,
    ) -> InferenceOutput:
        del text_query  # no text conditioning in the synthetic backend
        if self._simulate_latency_ms > 0:
            time.sleep(self._simulate_latency_ms / 1000.0)

        batch = pixel_values.shape[0]
        # A cheap content signature so frames differ deterministically.
        seed = int(abs(float(pixel_values.reshape(batch, -1).mean())) * 1e6) % 1000

        logits = np.full((batch, self._num_queries, 1), -20.0, dtype=np.float32)
        boxes = np.zeros((batch, self._num_queries, 4), dtype=np.float16)

        # Logit that sigmoids to `self._score`.
        score_logit = float(np.log(self._score / (1.0 - self._score)))
        rng = np.random.default_rng(seed)
        for b in range(batch):
            picks = rng.choice(self._num_queries, size=self._num_detections, replace=False)
            logits[b, picks, 0] = score_logit
            # Distinct, well-separated boxes in (cx, cy, w, h) form.
            for ordinal, index in enumerate(picks):
                cx = 0.2 + 0.15 * (ordinal % 4)
                cy = 0.3 + 0.1 * (ordinal // 4)
                boxes[b, index] = np.asarray([cx, cy, 0.1, 0.08], dtype=np.float16)
        return InferenceOutput(logits=logits, pred_boxes=boxes)


def build_backend(
    kind: str | None = None,
    *,
    model_path: str | Path | None = None,
    text_query: TextQuery | None = None,
    simulate_latency_ms: float = 0.0,
) -> InferenceBackend:
    """Construct the backend named by ``VISION_BACKEND`` (``owl2`` by default).

    ``synthetic`` exists so the orchestration can be load-tested and unit-tested
    without the 293 MB model or a GPU; it must never be selected in production,
    which is why the default is ``owl2`` and the choice is logged.
    """
    resolved = (kind or os.getenv("VISION_BACKEND", "owl2")).strip().lower()

    if resolved in ("synthetic", "mock", "fake"):
        logger.warning(
            "VISION_BACKEND=%s - the synthetic backend returns fabricated "
            "detections; never use it in production",
            resolved,
        )
        return SyntheticBackend(simulate_latency_ms=simulate_latency_ms)

    if resolved in ("owl2", "owl2-onnx", "onnx"):
        resolved_model = model_path or os.getenv(
            "OWL2_MODEL_PATH",
            str(
                Path(__file__).resolve().parents[1]
                / "owl"
                / "models"
                / "cuda"
                / "owl2_model.onnx"
            ),
        )
        # Load the tokenised prompt from OWL2_TEXT_IDS_PATH when the caller did not
        # supply one. Omitting this made the deployed backend unconstructible: it
        # passed text_query=None together with VISION_LABELS and tripped its own
        # "labels= requires a text_query" check, so every real task failed. Unit
        # tests always injected a TextQuery, which is exactly why it was invisible
        # until the first live cluster run.
        if text_query is None:
            text_query = load_text_query()
        extra_labels = tuple(
            label.strip()
            for label in os.getenv("VISION_LABELS", "").split(",")
            if label.strip()
        )
        return Owl2OnnxBackend(
            resolved_model,
            text_query=text_query,
            labels=extra_labels or None,
        )

    raise ValueError(
        f"unknown VISION_BACKEND={resolved!r}; expected 'owl2' or 'synthetic'"
    )
