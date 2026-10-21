# Copyright (c), The Rusoil Software Development Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# author : Konstantin Ustiuzhanin
# date   : 2026-Oct-21
#
# =============================================================================
"""Tests for the YOLO-World adapter's class-vocabulary handling."""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmark.runners import YoloWorldRunner


class _Tensor(list):
    """Mimics the ``.tolist()`` interface ultralytics returns for boxes and scores."""

    def tolist(self) -> list:
        return list(self)


class _Boxes:
    def __init__(self, rows: list[tuple[list[float], float, float, str]]) -> None:
        self.xyxy = _Tensor(row[0] for row in rows)
        self.conf = _Tensor(row[1] for row in rows)
        self.cls = _Tensor(row[2] for row in rows)

    def __len__(self) -> int:
        return len(self.conf)


class _Result:
    def __init__(self, rows: list[tuple[list[float], float, float, str]], names: dict[int, str]) -> None:
        self.boxes = _Boxes(rows)
        self.names = names


class _FakeModel:
    """Stands in for ``YOLOWorld``: records every vocabulary it was given."""

    def __init__(self, rows: list[tuple[list[float], float, float, str]]) -> None:
        self.rows = rows
        self.names: dict[int, str] = {}
        self.classes: list[tuple[str, ...]] = []
        self.predictions = 0

    def set_classes(self, labels: list[str]) -> None:
        self.classes.append(tuple(labels))
        self.names = {index: label for index, label in enumerate(labels)}

    def predict(self, image: np.ndarray, **_: object) -> list[_Result]:
        self.predictions += 1
        return [_Result(self.rows, self.names)]


@pytest.fixture()
def runner() -> YoloWorldRunner:
    instance = YoloWorldRunner(key="yolo-world-m", weights="unused.pt")
    instance._model = _FakeModel([])  # noqa: SLF001 - the adapter is the unit under test
    return instance


def _image() -> np.ndarray:
    return np.zeros((720, 960, 3), dtype=np.uint8)


def test_the_vocabulary_is_set_once_for_repeated_labels(runner: YoloWorldRunner) -> None:
    for _ in range(5):
        runner.detect(_image(), ["robot", "person"])

    assert runner._model.classes == [("robot", "person")]  # noqa: SLF001
    assert runner._model.predictions == 5  # noqa: SLF001


def test_a_new_vocabulary_is_applied_again(runner: YoloWorldRunner) -> None:
    runner.detect(_image(), ["robot"])
    runner.detect(_image(), ["steel blank"])
    runner.detect(_image(), ["robot"])

    assert runner._model.classes == [("robot",), ("steel blank",), ("robot",)]  # noqa: SLF001


def test_labels_are_converted_to_normalised_top_left_boxes(runner: YoloWorldRunner) -> None:
    runner._model = _FakeModel([([96.0, 72.0, 480.0, 360.0], 0.8, 0, "robot")])  # noqa: SLF001

    predictions = runner.detect(_image(), ["robot"])

    assert len(predictions) == 1
    box = predictions[0]
    assert box.label == "robot"
    assert box.confidence == pytest.approx(0.8)
    assert box.box == pytest.approx((0.1, 0.1, 0.4, 0.4))


def test_a_model_without_boxes_yields_nothing(runner: YoloWorldRunner) -> None:
    assert runner.detect(_image(), ["robot"]) == []
