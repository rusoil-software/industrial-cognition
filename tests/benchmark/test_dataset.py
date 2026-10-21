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
"""The dataset is the protocol: eight pan phases with the part, four crops without it,
one warehouse scene. These tests protect the two properties that make the comparison
mean anything - positives and negatives differ by content, and every model sees the
same frames."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.benchmark.dataset import (
    ABSENCE_CROPS,
    PAN_DRIFT_PIXELS,
    SHEET_GT,
    STATION_LABEL,
    WAREHOUSE_GT,
    build_dataset,
    crop_normalised,
    labels_of,
    pan_phases,
    station_samples,
    summarise,
    warehouse_samples,
)


def test_the_dataset_has_eight_positives_four_negatives_and_one_scene():
    samples = build_dataset()
    summary = summarise(samples)
    assert summary == {
        "samples": 13,
        "station_positives": 8,
        "station_negatives": 4,
        "warehouse_scenes": 1,
        "labels": 4,
    }


def test_the_dataset_feeds_the_frames_the_pipeline_would_encode():
    samples = build_dataset()
    for sample in samples:
        long_side = max(sample.height, sample.width)
        assert long_side <= 960, f"{sample.sample_id} is {sample.width}x{sample.height}"


def test_raw_photographs_are_available_for_an_unframed_comparison():
    raw = build_dataset(deployment=False)
    framed = build_dataset()
    assert max(raw[0].height, raw[0].width) == 1280
    assert raw[0].image.shape != framed[0].image.shape


def test_every_station_sample_asks_about_the_same_label():
    labels = {label for sample in station_samples(_sheet()) for label in sample.labels}
    assert labels == {STATION_LABEL}


def test_the_warehouse_sample_asks_about_three_objects():
    (sample,) = warehouse_samples(_warehouse())
    assert sample.labels == ("robot", "person", "cardboard box")
    assert set(sample.ground_truth) == set(WAREHOUSE_GT)


def test_labels_of_keeps_first_seen_order():
    samples = build_dataset()
    assert labels_of(samples) == (STATION_LABEL, "robot", "person", "cardboard box")


def test_the_pan_produces_different_frames():
    image = _striped(np.zeros((240, 320, 3), dtype=np.uint8))
    frames = pan_phases(image, phases=8)
    assert len(frames) == 8
    assert not np.array_equal(frames[0], frames[2])
    assert frames[0].shape == image.shape


def test_the_pan_moves_the_content_by_the_configured_drift():
    image = _striped(np.zeros((240, 320, 3), dtype=np.uint8))
    frames = pan_phases(image, phases=4)
    # phase 1 of 4 is a quarter period: dx = drift, dy = 0
    delta = _centroid(frames[1])[0] - _centroid(frames[0])[0]
    assert abs(delta - PAN_DRIFT_PIXELS) <= 2


def test_a_pan_of_one_phase_keeps_the_scene_where_it_was():
    # The replay source's vertical term at phase 0 is drift/4, so a single-phase pan is
    # a small offset, not an identity: the scene must stay where the ground truth says.
    image = _striped(np.zeros((120, 160, 3), dtype=np.uint8))
    (frame,) = pan_phases(image, phases=1)
    assert frame.shape == image.shape
    dx = abs(_centroid(frame)[0] - _centroid(image)[0])
    dy = abs(_centroid(frame)[1] - _centroid(image)[1])
    assert dx <= 2
    assert dy <= PAN_DRIFT_PIXELS // 4 + 2


def test_a_crop_keeps_the_requested_region():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[10:20, 30:40] = 255
    crop = crop_normalised(image, (0.15, 0.10, 0.05, 0.10))
    assert crop.shape[0] >= 1 and crop.shape[1] >= 1
    assert crop.max() == 255


def test_a_crop_beyond_the_edge_stays_inside_the_image():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    crop = crop_normalised(image, (0.95, 0.95, 0.5, 0.5))
    assert crop.shape[:2] == (2, 2)


def test_no_absence_crop_contains_the_labelled_part():
    x, y, w, h = SHEET_GT
    part = (x, y, x + w, y + h)
    for name, (cx, cy, cw, ch) in ABSENCE_CROPS.items():
        crop = (cx, cy, cx + cw, cy + ch)
        overlap_x = min(part[2], crop[2]) - max(part[0], crop[0])
        overlap_y = min(part[3], crop[3]) - max(part[1], crop[1])
        assert overlap_x <= 0 or overlap_y <= 0, f"{name} overlaps the part: {crop}"


def test_the_negatives_come_from_the_same_photograph_as_the_positives():
    samples = station_samples(_sheet())
    negatives = [sample for sample in samples if not sample.positive]
    assert len(negatives) == 4
    assert all(sample.note.startswith("crop of the same photograph") for sample in negatives)
    assert all(not sample.ground_truth for sample in negatives)


def test_a_missing_image_is_reported_by_name():
    with pytest.raises(FileNotFoundError) as error:
        station_samples("assets/images/not-a-photo.jpg")
    assert "not-a-photo.jpg" in str(error.value)


def _sheet() -> str:
    return "assets/images/metal_sheet_test_image.jpg"


def _warehouse() -> str:
    return "assets/images/roboflow_test_detect_human_robot.webp"


def _striped(image: np.ndarray) -> np.ndarray:
    image[:] = 0
    cv2.rectangle(image, (100, 80), (180, 160), (255, 255, 255), -1)
    return image


def _centroid(image: np.ndarray) -> tuple[float, float]:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(grey > 127)
    return float(xs.mean()), float(ys.mean())
