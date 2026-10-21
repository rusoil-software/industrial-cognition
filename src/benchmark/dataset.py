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
"""The benchmark's inputs: one labelled stand, one warehouse scene, and their rules.

Accuracy is only worth reporting if every model saw the same bytes and was scored by
the same rule, so this module owns both:

* the **station** protocol - eight pan phases of the real workpiece photograph, each a
  positive, plus four crops of the same photograph that cannot contain the part, each a
  negative. Positives and negatives differ by *content*, not by scene, which is what
  makes a presence answer meaningful.
* the **warehouse** scene - one Roboflow photograph with three hand-labelled objects,
  scored by best-box IoU.

Both are small on purpose. This is a comparison that has to be re-runnable on the
production rig, not a leaderboard entry: a bigger set that nobody can reproduce proves
less than a small set whose ground truth is in the repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

#: Normalised box, top-left origin: ``(x, y, width, height)``.
Box = tuple[float, float, float, float]

#: The workpiece in ``metal_sheet_test_image.jpg``, hand-checked against a 0.1 grid
#: overlay. The model's own box lands within 0.004 of it, which is why the photograph
#: can be used as a *positive* test even though the model put a bigger box on the table.
SHEET_GT: Box = (0.335, 0.455, 0.320, 0.220)

#: The phrase every model is asked at the station. Not ``metal sheet``: the calibration in
#: ``doc/vision-pipeline.md`` §19 showed that the industrial word for a cut blank scores
#: roughly twice as high on the real stand, and this is the phrase the deployment stores on
#: the camera. Measuring an easier phrase than the one we ship would be measuring nothing.
STATION_LABEL = "steel blank"

#: The pan the replay source applies to a still (``src/camera/sources.py``), so the
#: positives look like the frames the pipeline actually produces.
PAN_DRIFT_PIXELS = 24
PAN_PERIOD_FRAMES = 60

#: Crops of the workpiece photograph that cannot contain the part: ceiling/wall, floor,
#: and the two racking margins. Each is a negative for "the part is at the station".
ABSENCE_CROPS: dict[str, Box] = {
    "wall-band": (0.00, 0.00, 1.00, 0.30),
    "floor-band": (0.00, 0.85, 1.00, 0.15),
    "left-margin": (0.00, 0.30, 0.15, 0.55),
    "right-margin": (0.82, 0.30, 0.18, 0.55),
}

#: Hand-labelled objects in ``roboflow_test_detect_human_robot.webp`` (0.1 grid overlay,
#: boxes deliberately generous: they describe the object, not one model's output).
WAREHOUSE_GT: dict[str, Box] = {
    "robot": (0.280, 0.170, 0.470, 0.250),
    "person": (0.520, 0.510, 0.140, 0.130),
    "cardboard box": (0.655, 0.335, 0.135, 0.070),
}


@dataclass(frozen=True, slots=True)
class Sample:
    """One image the models are asked about, with what the right answer is."""

    sample_id: str
    scenario: str
    image: np.ndarray
    labels: tuple[str, ...]
    #: Ground truth per label; a missing label means "this object is not in the frame".
    ground_truth: dict[str, Box] = field(default_factory=dict)
    #: For the station scenario: is the part meant to be in this image?
    positive: bool = True
    note: str = ""

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


def pan_phases(image: np.ndarray, *, phases: int = 8, drift: int = PAN_DRIFT_PIXELS) -> list[np.ndarray]:
    """The frames the replay source would produce, one per eighth of its period.

    Mirrors ``ReplayFrameSource.read``: ``dx = drift * sin(phase)``,
    ``dy = drift/4 * cos(phase)`` with ``BORDER_REPLICATE``. Reimplemented rather than
    imported so the benchmark can pan an image it holds in memory.
    """
    height, width = image.shape[:2]
    step = max(1, PAN_PERIOD_FRAMES // max(1, phases))
    frames: list[np.ndarray] = []
    for index in range(phases):
        phase = 2.0 * np.pi * ((index * step) % PAN_PERIOD_FRAMES) / PAN_PERIOD_FRAMES
        dx = int(round(drift * np.sin(phase)))
        dy = int(round(drift / 4.0 * np.cos(phase)))
        if dx or dy:
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            frames.append(
                cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
            )
        else:
            frames.append(image.copy())
    return frames


def crop_normalised(image: np.ndarray, box: Box) -> np.ndarray:
    """Cut a normalised box out of an image, keeping at least one pixel per side."""
    height, width = image.shape[:2]
    x, y, w, h = box
    x1 = max(0, min(width - 1, int(round(x * width))))
    y1 = max(0, min(height - 1, int(round(y * height))))
    x2 = max(x1 + 1, min(width, int(round((x + w) * width))))
    y2 = max(y1 + 1, min(height, int(round((y + h) * height))))
    return image[y1:y2, x1:x2].copy()


def station_samples(image_path: str | Path, *, phases: int = 8) -> list[Sample]:
    """Eight pan phases with the part, four crops without it."""
    image = _read(image_path)
    samples = [
        Sample(
            sample_id=f"station-pan-{index}",
            scenario="station",
            image=frame,
            labels=(STATION_LABEL,),
            ground_truth={STATION_LABEL: SHEET_GT},
            positive=True,
            note="workpiece photograph, replay pan phase",
        )
        for index, frame in enumerate(pan_phases(image, phases=phases))
    ]
    samples.extend(
        Sample(
            sample_id=f"station-absent-{name}",
            scenario="station",
            image=crop_normalised(image, box),
            labels=(STATION_LABEL,),
            ground_truth={},
            positive=False,
            note=f"crop of the same photograph without the part ({name})",
        )
        for name, box in ABSENCE_CROPS.items()
    )
    return samples


def warehouse_samples(image_path: str | Path) -> list[Sample]:
    """One scene, three objects, three prompts in one pass."""
    return [
        Sample(
            sample_id="warehouse",
            scenario="warehouse",
            image=_read(image_path),
            labels=tuple(WAREHOUSE_GT),
            ground_truth=dict(WAREHOUSE_GT),
            positive=True,
            note="Roboflow warehouse photograph, hand-labelled boxes",
        )
    ]


def build_dataset(root: str | Path = "assets/images", *, deployment: bool = True) -> list[Sample]:
    """Every sample the benchmark scores, in a stable order.

    With ``deployment`` (the default) each frame is put through the same encode the
    capture loop performs - ``encode_jpeg(..., max_side=960)`` and a decode - so the
    models classify exactly the bytes the pipeline feeds them. Measuring raw photographs
    would flatter every model by a few points of confidence; the numbers that matter are
    the ones the deployment can reproduce.
    """
    root = Path(root)
    samples = station_samples(root / "metal_sheet_test_image.jpg")
    samples.extend(warehouse_samples(root / "roboflow_test_detect_human_robot.webp"))
    if not deployment:
        return samples
    return [
        Sample(
            sample_id=sample.sample_id,
            scenario=sample.scenario,
            image=deployment_frame(sample.image),
            labels=sample.labels,
            ground_truth=sample.ground_truth,
            positive=sample.positive,
            note=sample.note,
        )
        for sample in samples
    ]


def deployment_frame(image: np.ndarray, *, max_side: int = 960, quality: int = 80) -> np.ndarray:
    """The frame the capture loop would hand to a detector, as a decoded array."""
    from src.camera.capture import encode_jpeg  # noqa: PLC0415 - avoids a heavy import chain
    from src.vision.preprocess import decode_image  # noqa: PLC0415

    jpeg, _, _ = encode_jpeg(image, quality=quality, max_side=max_side)
    return decode_image(jpeg)



def _read(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"cannot read benchmark image {path}")
    return image


def labels_of(samples: Iterable[Sample]) -> tuple[str, ...]:
    """Union of the prompts the dataset needs, in first-seen order."""
    seen: list[str] = []
    for sample in samples:
        for label in sample.labels:
            if label not in seen:
                seen.append(label)
    return tuple(seen)


def summarise(samples: Sequence[Sample]) -> dict[str, int]:
    """Counts for the report header: how much evidence the table rests on."""
    positives = [s for s in samples if s.scenario == "station" and s.positive]
    negatives = [s for s in samples if s.scenario == "station" and not s.positive]
    return {
        "samples": len(samples),
        "station_positives": len(positives),
        "station_negatives": len(negatives),
        "warehouse_scenes": len({s.sample_id for s in samples if s.scenario == "warehouse"}),
        "labels": len(labels_of(samples)),
    }


__all__ = [
    "ABSENCE_CROPS",
    "Box",
    "PAN_DRIFT_PIXELS",
    "PAN_PERIOD_FRAMES",
    "SHEET_GT",
    "STATION_LABEL",
    "WAREHOUSE_GT",
    "Sample",
    "build_dataset",
    "crop_normalised",
    "deployment_frame",
    "labels_of",
    "pan_phases",
    "station_samples",
    "summarise",
    "warehouse_samples",
]
