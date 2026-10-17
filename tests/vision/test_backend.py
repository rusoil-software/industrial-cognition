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
"""Tests for the inference backend seam.

The text-query validations here are not defensive noise: the shipped artifact was
exported with random dummy token IDs, and feeding it all-zero (padding) tokens
saturates its fp16 text path at -65504, which decodes to *zero detections on every
frame*. That failure is silent - the pipeline runs happily and reports "nothing
found" forever - so it is pinned by tests at the configuration boundary.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.vision.backend import (
    FP16_SATURATION_THRESHOLD,
    InferenceOutput,
    Owl2OnnxBackend,
    SyntheticBackend,
    TextQuery,
    build_backend,
    load_text_query,
)


# --------------------------------------------------------------------------- #
# TextQuery
# --------------------------------------------------------------------------- #
def test_text_query_rejects_all_zero_tokens() -> None:
    """Zero is the padding token; only padding yields no detections at all."""
    with pytest.raises(ValueError, match="padding"):
        TextQuery(
            labels=("metal_sheet",),
            input_ids=(0,) * 16,
            attention_mask=(1,) * 16,
        )


def test_text_query_rejects_an_empty_attention_mask() -> None:
    with pytest.raises(ValueError, match="attention_mask"):
        TextQuery(
            labels=("metal_sheet",),
            input_ids=(49406, 1, 2) + (49407,) * 13,
            attention_mask=(0,) * 16,
        )


def test_text_query_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        TextQuery(labels=("a",), input_ids=(1, 2), attention_mask=(1,))


def test_text_query_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="label"):
        TextQuery(labels=(), input_ids=(1,), attention_mask=(1,))


def test_text_query_encodes_to_int64_batches() -> None:
    query = TextQuery(
        labels=("metal_sheet",),
        input_ids=(49406, 1, 2) + (49407,) * 13,
        attention_mask=(1, 1, 1) + (0,) * 13,
    )
    ids, mask = query.as_arrays(expected_length=16)

    assert ids.shape == (1, 16)
    assert mask.shape == (1, 16)
    assert ids.dtype == np.int64
    assert mask.dtype == np.int64


def test_text_query_rejects_a_length_the_graph_cannot_accept() -> None:
    query = TextQuery(labels=("a",), input_ids=(1, 2), attention_mask=(1, 1))
    with pytest.raises(ValueError, match="expects 16"):
        query.as_arrays(expected_length=16)


# --------------------------------------------------------------------------- #
# Loading the tokenised prompt
# --------------------------------------------------------------------------- #
def test_load_text_query_reads_json(scratch) -> None:
    path = scratch / "text_queries.json"
    path.write_text(
        json.dumps(
            {
                "labels": ["metal_sheet", "detail"],
                "input_ids": [49406, 3208, 49407] + [49407] * 13,
                "attention_mask": [1, 1, 1] + [0] * 13,
            }
        ),
        encoding="utf-8",
    )

    query = load_text_query(path)
    assert query.labels == ("metal_sheet", "detail")
    assert len(query.input_ids) == 16


def test_load_text_query_explains_how_to_create_a_missing_file(scratch) -> None:
    with pytest.raises(FileNotFoundError, match="encode_text_queries"):
        load_text_query(scratch / "absent.json")


def test_load_text_query_reports_a_malformed_file(scratch) -> None:
    path = scratch / "bad.json"
    path.write_text(json.dumps({"labels": ["a"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="input_ids"):
        load_text_query(path)


# --------------------------------------------------------------------------- #
# Synthetic backend
# --------------------------------------------------------------------------- #
def test_synthetic_backend_is_deterministic_and_well_formed() -> None:
    backend = SyntheticBackend(num_queries=32, num_detections=2, score=0.9)
    pixels = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)

    first = backend.infer(pixels)
    second = backend.infer(pixels)

    assert isinstance(first, InferenceOutput)
    assert first.logits.shape == (1, 32, 1)
    assert first.pred_boxes.shape == (1, 32, 4)
    assert np.array_equal(first.logits, second.logits)
    assert np.array_equal(first.pred_boxes, second.pred_boxes)


def test_synthetic_backend_produces_decodable_detections() -> None:
    from src.vision.decode import decode_frame

    backend = SyntheticBackend(num_queries=32, num_detections=3, score=0.9)
    output = backend.infer(np.zeros((1, 3, 8, 8), dtype=np.float32))

    detections = decode_frame(
        output.logits, output.pred_boxes, labels=("metal_sheet",), iou_threshold=None
    )
    assert len(detections) == 3
    assert all(d.confidence == pytest.approx(0.9, abs=1e-3) for d in detections)


def test_synthetic_backend_can_simulate_slow_inference() -> None:
    import time

    backend = SyntheticBackend(num_queries=4, num_detections=0, simulate_latency_ms=20)
    started = time.perf_counter()
    backend.infer(np.zeros((1, 3, 4, 4), dtype=np.float32))
    assert (time.perf_counter() - started) >= 0.015


# --------------------------------------------------------------------------- #
# Factory and failure modes
# --------------------------------------------------------------------------- #
def test_build_backend_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="VISION_BACKEND"):
        build_backend("magic")


def test_build_backend_selects_the_synthetic_backend() -> None:
    assert isinstance(build_backend("synthetic"), SyntheticBackend)


def test_build_backend_defaults_to_the_real_backend(monkeypatch, scratch) -> None:
    """The default kind is `owl2`, not the synthetic stand-in.

    A tokenised prompt is supplied because `build_backend` now loads it from
    OWL2_TEXT_IDS_PATH when the caller does not pass one - which is the behaviour
    that makes the deployed backend constructible at all.
    """
    import json

    path = scratch / "text_queries.json"
    path.write_text(
        json.dumps(
            {
                "labels": ["metal_sheet"],
                "input_ids": [49406, 3208, 2368] + [49407] * 13,
                "attention_mask": [1, 1, 1] + [0] * 13,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("VISION_BACKEND", raising=False)
    monkeypatch.setenv("OWL2_TEXT_IDS_PATH", str(path))
    monkeypatch.setenv("OWL2_MODEL_PATH", str(scratch / "model.onnx"))

    backend = build_backend()

    assert isinstance(backend, Owl2OnnxBackend)


def test_saturation_guard_catches_the_degenerate_text_path() -> None:
    """-65504 is the fp16 limit; logits there mean the graph is being fed padding."""
    saturated = np.full((1, 10, 1), -FP16_SATURATION_THRESHOLD, dtype=np.float32)
    with pytest.raises(RuntimeError, match="saturat"):
        Owl2OnnxBackend._assert_not_saturated(saturated)


def test_saturation_guard_allows_normal_logits() -> None:
    Owl2OnnxBackend._assert_not_saturated(np.full((1, 4, 1), -20.0, dtype=np.float32))


def test_real_backend_requires_the_model_file(scratch) -> None:
    backend = Owl2OnnxBackend(scratch / "absent.onnx")
    with pytest.raises(FileNotFoundError):
        backend.warmup()


def test_real_backend_rejects_an_unavailable_provider(scratch) -> None:
    """A GPU provider requested on a CPU-only runtime must fail loudly."""
    import onnxruntime as ort

    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("CUDA is available on this machine")

    model = scratch / "dummy.onnx"
    model.write_bytes(b"0")
    backend = Owl2OnnxBackend(model, execution_provider="CUDAExecutionProvider")
    with pytest.raises(RuntimeError, match="not available"):
        backend.warmup()


def test_build_backend_loads_the_text_query_from_the_configured_path(monkeypatch, scratch) -> None:
    """The deployed backend must construct itself from OWL2_TEXT_IDS_PATH.

    Regression: build_backend passed text_query=None together with VISION_LABELS and
    tripped its own "labels= requires a text_query" check, so every real task failed
    in the cluster while every unit test passed (they inject a TextQuery).
    """
    import json

    path = scratch / "text_queries.json"
    path.write_text(
        json.dumps(
            {
                "labels": ["metal_sheet"],
                "input_ids": [49406, 3208, 2368, 2369, 49407] + [49407] * 11,
                "attention_mask": [1, 1, 1, 1, 1] + [0] * 11,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OWL2_TEXT_IDS_PATH", str(path))
    monkeypatch.setenv("VISION_LABELS", "metal_sheet")

    backend = build_backend("owl2", model_path=scratch / "model.onnx")

    assert backend.name == "owl2-onnx"
    assert backend._text_query.labels == ("metal_sheet",)


def test_build_backend_explains_a_missing_text_query(monkeypatch, scratch) -> None:
    monkeypatch.setenv("OWL2_TEXT_IDS_PATH", str(scratch / "absent.json"))
    with pytest.raises(FileNotFoundError, match="encode_text_queries"):
        build_backend("owl2", model_path=scratch / "model.onnx")
