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
"""Scoring decides what "better" means, so these tests pin the rule: one hit per label,
the production presence gate for the station answer, and a row that reports missing
measurements as missing."""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmark.dataset import SHEET_GT, WAREHOUSE_GT, Sample
from src.benchmark.scoring import (
    HIT_IOU,
    STAND_GATE,
    Prediction,
    aggregate,
    best_iou,
    iou,
    score_sample,
)


def test_iou_of_a_box_with_itself_is_one():
    assert iou(SHEET_GT, SHEET_GT) == pytest.approx(1.0)


def test_disjoint_boxes_overlap_by_zero():
    assert iou((0.0, 0.0, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2)) == 0.0


def test_degenerate_boxes_are_never_a_hit():
    assert iou((0.2, 0.2, 0.0, 0.3), SHEET_GT) == 0.0


def test_half_overlap_is_about_one_third():
    left = (0.0, 0.0, 0.4, 0.4)
    right = (0.2, 0.0, 0.4, 0.4)
    assert iou(left, right) == pytest.approx(0.2 / 0.6, rel=1e-6)


def test_best_iou_only_considers_the_requested_label():
    predictions = [
        Prediction("robot", 0.9, SHEET_GT),
        Prediction("metal sheet", 0.1, (0.0, 0.0, 0.05, 0.05)),
    ]
    value, chosen = best_iou(predictions, "metal sheet", SHEET_GT)
    assert chosen is not None and chosen.label == "metal sheet"
    assert value < HIT_IOU


def test_a_sample_without_ground_truth_is_scored_on_presence_only():
    sample = station_sample(positive=False)
    score = score_sample(sample, [])
    assert score.best_iou == 0.0
    assert score.hit is False
    assert score.present is False


def test_the_part_at_the_station_is_present():
    score = score_sample(station_sample(), [Prediction("metal sheet", 0.16, SHEET_GT)])
    assert score.present is True
    assert score.hit is True
    assert score.chosen_confidence == pytest.approx(0.16)


def test_the_table_alone_is_not_the_part():
    table = (0.19, 0.375, 0.59, 0.426)
    score = score_sample(station_sample(), [Prediction("metal sheet", 0.2, table)])
    assert score.present is False
    assert "a table, not the part" in score.reason


def test_a_floor_box_is_refused_because_it_is_outside_the_station():
    floor = (0.30, 0.78, 0.29, 0.22)
    score = score_sample(station_sample(), [Prediction("metal sheet", 0.12, floor)])
    assert score.present is False
    assert "inside the station" in score.reason


def test_the_wall_band_is_refused_by_its_aspect_ratio():
    wall = (0.436, 0.310, 0.433, 0.191)
    score = score_sample(station_sample(), [Prediction("metal sheet", 0.08, wall)])
    assert score.present is False
    assert "aspect" in score.reason


def test_a_box_under_another_label_does_not_answer_the_station():
    score = score_sample(station_sample(), [Prediction("robot", 0.5, SHEET_GT)])
    assert score.present is False
    assert "no box at all" in score.reason


def test_the_warehouse_scene_is_scored_by_overlap_not_by_presence():
    robot = WAREHOUSE_GT["robot"]
    prediction = Prediction("robot", 0.32, robot)
    score = score_sample(warehouse_sample(), [prediction])
    assert score.hit is True
    assert score.present is None


def test_a_warehouse_box_missed_by_one_label_is_still_counted_for_the_others():
    score = score_sample(
        warehouse_sample(),
        [
            Prediction("robot", 0.32, WAREHOUSE_GT["robot"]),
            Prediction("person", 0.30, WAREHOUSE_GT["person"]),
        ],
    )
    assert score.hit is True
    assert score.best_iou >= HIT_IOU


def test_aggregate_reports_recall_and_specificity_separately():
    scores = [
        score_sample(station_sample(positive=True), [Prediction("metal sheet", 0.2, SHEET_GT)]),
        score_sample(station_sample(positive=True), []),
        score_sample(station_sample(positive=False), []),
        score_sample(station_sample(positive=False), [Prediction("metal sheet", 0.2, SHEET_GT)]),
    ]
    metrics = aggregate(scores)
    assert metrics["presence_recall"] == pytest.approx(0.5)
    assert metrics["presence_specificity"] == pytest.approx(0.5)
    assert metrics["samples"] == 4


def test_aggregate_of_nothing_is_zero_not_an_error():
    metrics = aggregate([])
    assert metrics["presence_recall"] == 0.0
    assert metrics["warehouse_hits"] == 0.0


def test_the_shared_gate_is_the_measured_one_from_the_pipeline():
    assert STAND_GATE.roi == (0.15, 0.42, 0.85, 0.85)
    assert STAND_GATE.max_area == pytest.approx(0.20)
    assert STAND_GATE.max_aspect == pytest.approx(2.0)
    assert STAND_GATE.min_confidence == pytest.approx(0.05)


def station_sample(*, positive: bool = True) -> Sample:
    return Sample(
        sample_id="station-pan-0",
        scenario="station",
        image=np.zeros((48, 64, 3), dtype=np.uint8),
        labels=("metal sheet",),
        ground_truth={"metal sheet": SHEET_GT} if positive else {},
        positive=positive,
    )


def warehouse_sample() -> Sample:
    return Sample(
        sample_id="warehouse",
        scenario="warehouse",
        image=np.zeros((48, 64, 3), dtype=np.uint8),
        labels=tuple(WAREHOUSE_GT),
        ground_truth=dict(WAREHOUSE_GT),
        positive=True,
    )
