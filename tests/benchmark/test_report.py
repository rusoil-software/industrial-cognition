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
"""The CSV is the artefact a decision rests on, so its shape is pinned: stable column
order, tri-state licence, an empty cell for "not measured" and never a zero that could
be mistaken for a measurement."""

from __future__ import annotations

import csv

from src.benchmark.registry import MODELS
from src.benchmark.report import (
    COLUMNS,
    Row,
    format_blocked,
    format_table,
    write_csv,
    write_details,
)

OWL2 = {spec.key: spec for spec in MODELS}["owl2-onnx"]
DINO = {spec.key: spec for spec in MODELS}["grounding-dino-15-edge"]


def test_the_csv_columns_are_the_documented_contract():
    assert COLUMNS[0] == "model"
    assert "license_ok" in COLUMNS
    assert COLUMNS[-1] == "notes"
    assert len(COLUMNS) == len(set(COLUMNS))


def test_a_row_takes_its_licence_from_the_registry():
    row = Row.from_spec(OWL2)
    assert row.license == "Apache-2.0"
    assert row.license_ok is True


def test_a_missing_number_is_written_as_an_empty_cell():
    record = Row.from_spec(OWL2).as_csv()
    assert record["latency_p50_ms"] == ""
    assert record["presence_recall"] == ""
    assert record["peak_gpu_mb"] == ""


def test_accuracy_is_written_with_four_decimals_and_cost_with_two():
    row = Row.from_spec(OWL2, presence_recall=0.123456, sheet_iou_mean=0.5, latency_p50_ms=1234.567)
    record = row.as_csv()
    assert record["presence_recall"] == "0.1235"
    assert record["sheet_iou_mean"] == "0.5000"
    assert record["latency_p50_ms"] == "1234.57"


def test_the_licence_column_is_readable_without_a_glossary():
    assert Row.from_spec(OWL2).as_csv()["license_ok"] == "yes"
    assert Row.from_spec(DINO).as_csv()["license_ok"] == "no"


def test_a_boolean_is_spelled_out():
    record = Row.from_spec(OWL2, measured=True).as_csv()
    assert record["measured"] == "yes"


def test_csv_round_trip_keeps_the_values(tmp_path):
    rows = [
        Row.from_spec(OWL2, measured=True, presence_recall=0.875, latency_p50_ms=11.0),
        Row.from_spec(DINO, measured=False),
    ]
    path = write_csv(rows, tmp_path / "out" / "bench.csv")
    assert path.exists()
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 2
    assert records[0]["model"] == "owl2-onnx"
    assert records[0]["presence_recall"] == "0.8750"
    assert records[1]["measured"] == "no"


def test_details_are_one_row_per_sample_and_model(tmp_path):
    rows = [
        Row.from_spec(OWL2, measured=True),
        Row.from_spec(DINO, measured=False),
    ]
    rows[0].samples = [
        {"sample_id": "station-pan-0", "scenario": "station", "positive": True, "boxes": 2},
        {"sample_id": "station-pan-1", "scenario": "station", "positive": True, "boxes": 1},
    ]
    path = write_details(rows, tmp_path / "details.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    assert [record["sample_id"] for record in records] == ["station-pan-0", "station-pan-1"]
    assert all(record["model"] == "owl2-onnx" for record in records)


def test_the_console_table_names_every_model():
    table = format_table([Row.from_spec(OWL2, latency_p50_ms=11.0, fps=90.0, presence_recall=1.0)])
    assert "owl2-onnx" in table
    assert "100%" in table


def test_a_blocked_model_is_listed_with_its_reason():
    text = format_blocked([Row.from_spec(DINO, measured=False)])
    assert "grounding-dino-15-edge" in text
    assert "not measurable here" in text


def test_no_blocked_models_means_no_extra_section():
    assert format_blocked([Row.from_spec(OWL2)]) == ""
