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
"""Detection quality against a real photograph.

Every other test in this repository checks plumbing: that frames flow, that boxes
decode, that prompts reach the graph. None of them would notice if the model stopped
*understanding* the prompt, because a decoder that ignores its text input still
returns well-formed boxes. This test is the one that fails in that case.

It uses `assets/images/roboflow_test_detect_human_robot.webp` (Apache 2.0, Roboflow
test dataset): an overhead warehouse scene with two geometrically separable subjects -
a yellow articulated robot arm at the top centre, and a worker in a hi-vis vest below
it. The assertion is the thing a human would say out loud:

    with the prompt "robot", the best box is the arm;
    with "human" (or "person"), the best box is the worker.

Because the points are far apart, that is a meaningful check of prompt conditioning
rather than a coincidence.

Two deliberate limits on what this proves:

* The exported artifact has a **single** class dimension (`logits` is (1, 3600, 1)), so
  the graph scores one query at a time. Prompt conditioning shows up as *where* the
  best box is and how confident it is, not as a per-class label. A multi-label prompt
  therefore reports everything as `labels[0]` - see doc/vision-pipeline.md §14.
* The prompt token IDs are committed fixtures (`assets/prompts/*.json`), produced by
  `scripts/encode_text_queries.py` through the real CLIP tokenizer. That keeps this
  test free of the `transformers` dependency, which the vision image deliberately does
  not carry.

The 293 MiB model is not in version control, so the test skips when it is absent; it is
a `model`-marked test and takes ~20-30 s on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")

from src.vision.backend import Owl2OnnxBackend, TextQuery  # noqa: E402
from src.vision.decode import decode_frame  # noqa: E402
from src.vision.preprocess import PreprocessConfig, to_model_input  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
IMAGE = REPO / "assets" / "images" / "roboflow_test_detect_human_robot.webp"
PROMPTS = REPO / "assets" / "prompts"
MODEL = REPO / "src" / "owl" / "models" / "cuda" / "owl2_model.onnx"

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not MODEL.exists(), reason=f"model artifact absent: {MODEL}"),
    pytest.mark.skipif(not IMAGE.exists(), reason=f"test image absent: {IMAGE}"),
]

#: Hand-labelled regions in normalised (x_min, y_min, x_max, y_max) coordinates.
#: Deliberately generous: the point is "did it look at the right end of the image",
#: not pixel-accurate localisation.
REGIONS = {
    "robot_arm": (0.25, 0.19, 0.73, 0.43),
    "human": (0.50, 0.49, 0.70, 0.65),
}


def load_query(name: str) -> TextQuery:
    # Fixture files are named after the label with spaces replaced, so "cardboard box"
    # maps to cardboard_box.json.
    path = PROMPTS / f"{name.replace(' ', '_')}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TextQuery(
        labels=tuple(payload["labels"]),
        input_ids=tuple(payload["input_ids"]),
        attention_mask=tuple(payload["attention_mask"]),
    )


@pytest.fixture(scope="module")
def scene() -> np.ndarray:
    image = cv2.imread(str(IMAGE))
    assert image is not None, f"could not decode {IMAGE}"
    return image


@pytest.fixture(scope="module")
def backend() -> Owl2OnnxBackend:
    return Owl2OnnxBackend(str(MODEL), text_query=load_query("robot"))


def top_box(backend: Owl2OnnxBackend, scene: np.ndarray, prompt: str):
    """Run one prompt and return (score, centre, size) of its best box."""
    output = backend.infer(
        to_model_input(scene, PreprocessConfig(width=960, height=960)),
        text_query=load_query(prompt),
    )
    detections = decode_frame(
        np.asarray(output.logits),
        np.asarray(output.pred_boxes, dtype=np.float32),
        labels=load_query(prompt).labels,
        score_threshold=0.0,
        iou_threshold=0.5,
        max_detections=10,
    )
    assert detections, f"prompt {prompt!r} produced no boxes at all"
    best = detections[0]
    centre = (
        best.x_norm + best.width_norm / 2,
        best.y_norm + best.height_norm / 2,
    )
    return best.confidence, centre, (best.width_norm, best.height_norm)


def region_of(centre: tuple[float, float]) -> str:
    for name, (x1, y1, x2, y2) in REGIONS.items():
        if x1 <= centre[0] <= x2 and y1 <= centre[1] <= y2:
            return name
    return "background"


def test_robot_prompt_finds_the_robot_arm(backend, scene) -> None:
    score, centre, size = top_box(backend, scene, "robot")

    assert region_of(centre) == "robot_arm", (
        f"'robot' looked at {region_of(centre)} (centre {centre}), expected the arm"
    )
    # Above the deployed 0.25 threshold, so this is a detection and not an artefact.
    assert score > 0.25, f"'robot' scored only {score:.3f}"
    assert size[0] > 0.05 and size[1] > 0.05, f"degenerate box {size}"


@pytest.mark.parametrize("prompt", ["human", "person"])
def test_human_prompts_find_the_worker(backend, scene, prompt: str) -> None:
    """Two synonyms, same target: the worker must win for both."""
    _score, centre, _size = top_box(backend, scene, prompt)

    assert region_of(centre) == "human", (
        f"{prompt!r} looked at {region_of(centre)} (centre {centre}), "
        "expected the worker"
    )


def test_the_prompt_changes_where_the_model_looks(backend, scene) -> None:
    """The regression this whole file exists for.

    If the text path were ignored - the failure mode that made an earlier export
    detect nothing - every prompt would return the same box. 'robot' and 'human' must
    disagree, and each must be right.
    """
    robot_centre = top_box(backend, scene, "robot")[1]
    human_centre = top_box(backend, scene, "human")[1]

    separation = abs(robot_centre[1] - human_centre[1])  # vertical distance
    assert separation > 0.15, (
        "both prompts picked nearly the same box "
        f"({robot_centre} vs {human_centre}); the prompt is probably ignored"
    )


def test_a_prompt_for_something_present_is_more_confident_than_nonsense(backend, scene) -> None:
    """Sanity: a real object outranks an absent one."""
    robot_score = top_box(backend, scene, "robot")[0]
    box_score = top_box(backend, scene, "cardboard box")[0]

    # Both are present in this warehouse scene; what matters is that neither is ~0,
    # which would mean the text conditioning collapsed.
    assert robot_score > 0.1, f"'robot' scored {robot_score:.3f}"
    assert box_score > 0.1, f"'cardboard box' scored {box_score:.3f}"
