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
"""Presence of the real metal sheet: is it there, or is the station empty?

`assets/images/roboflow_test_detect_human_robot.webp` checks that the prompt reaches
the graph - a synthetic-but-real photograph with two subjects far apart. This file
checks the thing the station actually needs, on the photograph the project owner took
of the real workpiece (`assets/images/metal_sheet_test_image.jpg`, a brushed-metal
sheet on a white table in a tiled hallway):

    with the sheet in frame, the best box is the sheet;
    with the sheet cropped out, nothing scores anywhere near it.

Measured on the exported artifact (CPU, 960x960 input, 2026-10-21):

    prompt        sheet   ceiling   wall   bare table
    steel blank  0.2459    0.0122   0.0000    0.0465
    metal blank  0.2448    0.0232   0.0000    0.0170
    metal sheet  0.1564    0.0064   0.0000    0.0232

So a threshold around 0.15 separates this scene cleanly, and 0.25 - the default the
worker ships with - would have reported an empty station for a sheet that is plainly
there. That gap is the reason this file exists: "zero-shot" is not the same as
"calibrated".

Two limits, deliberately asserted around rather than papered over:

* The **tiled floor** in the bottom of the same photograph scores 0.19 for
  "steel blank": glossy tiles read as sheet metal. A threshold alone cannot fix a
  scene where the floor looks like the workpiece, so the operating rule is "point the
  camera at the table", and `scripts/calibrate_presence.py` is how a new camera gets
  its number. The floor crop is excluded from the absence set below for the same
  reason - including it would make this test assert something untrue.
* The sheet photograph also contains a doorway and a wall; those are true negatives,
  and they stay near zero. Absence is therefore tested on *the same photograph*, which
  controls for lighting and camera in a way a second photograph could not.

The test skips without the 293 MiB artifact, and is `model`-marked (~60 s on CPU).
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
IMAGE = REPO / "assets" / "images" / "metal_sheet_test_image.jpg"
PROMPTS = REPO / "assets" / "prompts"
MODEL = REPO / "src" / "owl" / "models" / "cuda" / "owl2_model.onnx"

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not MODEL.exists(), reason=f"model artifact absent: {MODEL}"),
    pytest.mark.skipif(not IMAGE.exists(), reason=f"test image absent: {IMAGE}"),
]

#: The sheet in normalised coordinates, hand-labelled from the photograph with room
#: to spare: the reflective rectangle runs x 0.34-0.65, y 0.46-0.67.
SHEET_REGION = (0.30, 0.40, 0.72, 0.74)

#: A prompt must be this confident on the real sheet to count as a detection. Measured
#: 0.2459 / 0.2448 / 0.1564, so this is a floor, not a target.
MIN_POSITIVE_SCORE = 0.18

#: Empty parts of the same photograph must stay below this. Measured at most 0.0465.
MAX_ABSENCE_SCORE = 0.10

#: Crops of the same frame that contain no sheet: the doorway above the table, the
#: wall on the right, and the bare left end of the table.
ABSENCE_CROPS = {
    "ceiling": (0.00, 0.30, 0.00, 1.00),
    "wall": (0.10, 0.50, 0.86, 1.00),
    "bare_table": (0.40, 0.78, 0.19, 0.33),
}


def load_query(name: str) -> TextQuery:
    payload = json.loads((PROMPTS / f"{name.replace(' ', '_')}.json").read_text(encoding="utf-8"))
    return TextQuery(
        labels=tuple(payload["labels"]),
        input_ids=tuple(payload["input_ids"]),
        attention_mask=tuple(payload["attention_mask"]),
    )


def crop(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Slice a normalised `(y1, y2, x1, x2)` box out of an image."""
    height, width = image.shape[:2]
    y1, y2, x1, x2 = box
    return image[int(height * y1) : int(height * y2), int(width * x1) : int(width * x2)]


def top_box(backend: Owl2OnnxBackend, image: np.ndarray, prompt: str):
    """Return `(score, centre)` of the best box for one prompt on one image."""
    query = load_query(prompt)
    output = backend.infer(
        to_model_input(image, PreprocessConfig(width=960, height=960)),
        text_query=query,
    )
    detections = decode_frame(
        np.asarray(output.logits),
        np.asarray(output.pred_boxes, dtype=np.float32),
        labels=query.labels,
        score_threshold=0.0,
        iou_threshold=0.5,
        max_detections=1,
    )
    if not detections:
        return 0.0, (0.0, 0.0)
    best = detections[0]
    return float(best.confidence), (
        best.x_norm + best.width_norm / 2,
        best.y_norm + best.height_norm / 2,
    )


@pytest.fixture(scope="module")
def scene() -> np.ndarray:
    image = cv2.imread(str(IMAGE))
    assert image is not None, f"could not decode {IMAGE}"
    return image


@pytest.fixture(scope="module")
def backend() -> Owl2OnnxBackend:
    return Owl2OnnxBackend(str(MODEL), text_query=load_query("steel blank"))


@pytest.mark.parametrize("prompt", ["steel blank", "metal blank"])
def test_the_sheet_is_found_where_it_is(backend, scene, prompt: str) -> None:
    score, centre = top_box(backend, scene, prompt)

    x1, y1, x2, y2 = SHEET_REGION
    assert x1 <= centre[0] <= x2 and y1 <= centre[1] <= y2, (
        f"{prompt!r} looked at centre {centre}, outside the sheet region "
        f"{SHEET_REGION}; it saw something else in the hallway"
    )
    assert score >= MIN_POSITIVE_SCORE, (
        f"{prompt!r} scored only {score:.4f} on a sheet that fills half the frame"
    )


@pytest.mark.parametrize("prompt", ["steel blank", "metal blank"])
def test_an_empty_part_of_the_same_scene_stays_quiet(backend, scene, prompt: str) -> None:
    """Absence is checked on the same photograph, so lighting is not a variable."""
    for name, box in ABSENCE_CROPS.items():
        score, _centre = top_box(backend, crop(scene, box), prompt)
        assert score < MAX_ABSENCE_SCORE, (
            f"{prompt!r} scored {score:.4f} on the {name} crop, which holds no sheet; "
            "this is what an operator would see as a phantom part"
        )


def test_the_sheet_outranks_every_empty_crop_by_a_wide_margin(backend, scene) -> None:
    """The separation, not just the two absolute numbers, is what makes a threshold work."""
    positive, _centre = top_box(backend, scene, "steel blank")
    worst_absence = max(
        top_box(backend, crop(scene, box), "steel blank")[0] for box in ABSENCE_CROPS.values()
    )

    assert positive > 3 * worst_absence, (
        f"sheet {positive:.4f} vs worst empty crop {worst_absence:.4f}: a threshold "
        "cannot be placed safely between them"
    )
    assert positive - worst_absence > 2 * 0.05, (
        "the gap is narrower than the margin scripts/calibrate_presence.py requires"
    )


def test_the_sheet_is_a_blank_like_a_worker_would_call_it(backend, scene) -> None:
    """The industrial word for a cut sheet wins over the generic description.

    Kept as an assertion because the difference (0.2459 vs 0.1564) is what decides
    whether the deployed 0.25 threshold is a near miss or a plain miss, and somebody
    will eventually "simplify" the prompt back to `metal sheet`.
    """
    generic, _ = top_box(backend, scene, "metal sheet")
    industrial, _ = top_box(backend, scene, "steel blank")

    assert industrial > generic, (
        f"'steel blank' ({industrial:.4f}) did not beat 'metal sheet' ({generic:.4f})"
    )
