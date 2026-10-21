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
"""Downloaded weights must not land in the checkout."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmark.runners import weights_path


def test_without_a_directory_the_name_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("YOLO_WEIGHTS_DIR", raising=False)

    assert weights_path("yolov8m-worldv2.pt") == "yolov8m-worldv2.pt"


def test_an_explicit_directory_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOLO_WEIGHTS_DIR", "/cache/elsewhere")

    assert weights_path("yolov8l-worldv2.pt", "/cache/yolo") == str(Path("/cache/yolo", "yolov8l-worldv2.pt"))


def test_the_environment_variable_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOLO_WEIGHTS_DIR", "/cache/yolo")

    assert weights_path("yolov8m-worldv2.pt") == str(Path("/cache/yolo", "yolov8m-worldv2.pt"))


def test_a_blank_directory_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOLO_WEIGHTS_DIR", "   ")

    assert weights_path("yolov8m-worldv2.pt") == "yolov8m-worldv2.pt"

