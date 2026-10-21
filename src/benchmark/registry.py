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
"""Which open-vocabulary detectors the benchmark can compare, and on what terms.

The registry is deliberately separate from the runners: adding a model here must not
require touching scoring, and the licence column has to be readable without running
anything. A model that cannot be measured on this machine still gets a row - with the
reason and its licence - because "not evaluated" and "evaluated badly" are different
answers and a table that hides the first one is not evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

#: Licences that can be shipped inside a closed product without further negotiation.
PERMISSIVE_LICENCES = frozenset({"Apache-2.0", "MIT", "BSD-3-Clause"})

#: Size on disk of the weights this benchmark downloads, in mebibytes. Measured where
#: the model was run; the numbers in the CSV are measured again at run time.
ONNX_OWL2_ARTIFACT = "src/owl/models/cuda/owl2_model.onnx"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One row of the comparison table, before anything has been measured."""

    key: str
    family: str
    weights: str
    runtime: str
    license: str
    weights_mb: float
    notes: str = ""
    #: False when this benchmark cannot run the model here, with :attr:`blocked` saying why.
    available: bool = True
    blocked: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def license_ok(self) -> Optional[bool]:
        """True for a licence that needs no negotiation, False when it does, None when unclear."""
        if self.license in PERMISSIVE_LICENCES:
            return True
        if self.license in {"AGPL-3.0", "GPL-3.0", "CC-BY-NC-4.0", "Proprietary"}:
            return False
        return None


def _spec(**kwargs: object) -> ModelSpec:
    return ModelSpec(**kwargs)  # type: ignore[arg-type]


#: Order matters: the CSV lists the incumbent first, so the comparison reads as a change.
MODELS: tuple[ModelSpec, ...] = (
    _spec(
        key="owl2-onnx",
        family="OWLv2",
        weights=ONNX_OWL2_ARTIFACT,
        runtime="onnxruntime",
        license="Apache-2.0",
        weights_mb=293.4,
        notes="the incumbent: same artifact the cluster runs, CPU, 8 threads",
    ),
    _spec(
        key="grounding-dino-base",
        family="Grounding DINO (Swin-T)",
        weights="IDEA-Research/grounding-dino-base",
        runtime="transformers",
        license="Apache-2.0",
        weights_mb=694.0,
        notes="text-grounded boxes, one prompt string at a time",
    ),
    _spec(
        key="grounding-dino-15-edge",
        family="Grounding DINO 1.5 Edge",
        weights="IDEA Research API",
        runtime="api",
        license="Proprietary",
        weights_mb=0.0,
        notes="API-only release: no weights, no offline measurement, no Apache grant",
        available=False,
        blocked="no open weights; IDEA Research serves it behind an API and API keys",
    ),
    _spec(
        key="florence2-base",
        family="Florence-2",
        weights="microsoft/Florence-2-base",
        runtime="transformers",
        license="MIT",
        weights_mb=464.0,
        notes="seq2seq: boxes come back as text tokens, parsed by a prompt template",
    ),
    _spec(
        key="yolo-world-m",
        family="YOLO-World v2 M",
        weights="yolov8m-worldv2.pt",
        runtime="ultralytics",
        license="AGPL-3.0",
        weights_mb=50.0,
        notes="fast, but the ultralytics runtime is AGPL-3.0: copyleft for a hosted product",
    ),
    _spec(
        key="yolo-world-l",
        family="YOLO-World v2 L",
        weights="yolov8l-worldv2.pt",
        runtime="ultralytics",
        license="AGPL-3.0",
        weights_mb=96.0,
        notes="as M, with a larger backbone",
    ),
    _spec(
        key="dino-x",
        family="DINO-X",
        weights="IDEA Research API",
        runtime="api",
        license="Proprietary",
        weights_mb=0.0,
        notes="closed API; no local weights, so no latency or accuracy number here",
        available=False,
        blocked="closed API: no weights, no offline measurement, no Apache grant",
    ),
)

BY_KEY: dict[str, ModelSpec] = {spec.key: spec for spec in MODELS}


def resolve(keys: list[str]) -> list[ModelSpec]:
    """Return the requested models in registry order, or raise on an unknown key."""
    if not keys:
        return list(MODELS)
    unknown = [key for key in keys if key not in BY_KEY]
    if unknown:
        raise KeyError(f"unknown model(s): {', '.join(sorted(unknown))}; known: {', '.join(sorted(BY_KEY))}")
    wanted = set(keys)
    return [spec for spec in MODELS if spec.key in wanted]


def runnable(specs: list[ModelSpec] | None = None) -> list[ModelSpec]:
    """The subset that can actually be measured on this machine."""
    return [spec for spec in (specs if specs is not None else MODELS) if spec.available]


def license_summary(specs: list[ModelSpec] | None = None) -> dict[str, int]:
    """Count models per licence verdict, for the summary line of the report."""
    summary = {"permissive": 0, "copyleft_or_commercial": 0, "unclear": 0}
    for spec in specs if specs is not None else MODELS:
        ok = spec.license_ok
        if ok is True:
            summary["permissive"] += 1
        elif ok is False:
            summary["copyleft_or_commercial"] += 1
        else:
            summary["unclear"] += 1
    return summary


__all__ = [
    "BY_KEY",
    "MODELS",
    "PERMISSIVE_LICENCES",
    "ModelSpec",
    "license_summary",
    "resolve",
    "runnable",
]
