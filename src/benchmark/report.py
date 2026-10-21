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
"""The evidence: one CSV row per model, plus a table a human can read.

The CSV is the deliverable - it is what a decision is justified with, so the column
names are stable and every row carries its licence verdict and the reason a model was
not measured. An empty cell means "could not be measured here"; it never means zero.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from src.benchmark.registry import ModelSpec

#: Column order of the CSV. Adding a column is fine; renaming one breaks comparisons
#: with earlier runs, so the names are part of the contract.
COLUMNS: tuple[str, ...] = (
    "model",
    "family",
    "license",
    "license_ok",
    "runtime",
    "device",
    "weights_mb",
    "params_m",
    "load_s",
    "latency_p50_ms",
    "latency_p95_ms",
    "fps",
    "peak_rss_mb",
    "peak_gpu_mb",
    "presence_recall",
    "presence_specificity",
    "sheet_iou_mean",
    "warehouse_hits",
    "measured",
    "notes",
)


@dataclass(slots=True)
class Row:
    """One model's line in the report."""

    key: str
    family: str
    license: str
    license_ok: Optional[bool]
    runtime: str
    device: str = ""
    weights_mb: Optional[float] = None
    params_m: Optional[float] = None
    load_s: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    fps: Optional[float] = None
    peak_rss_mb: Optional[float] = None
    peak_gpu_mb: Optional[float] = None
    presence_recall: Optional[float] = None
    presence_specificity: Optional[float] = None
    sheet_iou_mean: Optional[float] = None
    warehouse_hits: Optional[float] = None
    measured: bool = True
    notes: str = ""
    samples: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: ModelSpec, **values: object) -> "Row":
        """Start from the registry row, then apply the measured values.

        ``notes`` appears in both places: the registry explains what the model *is*, the
        measurement may add what went wrong. The measured value wins when it is given.
        """
        base = {
            "key": spec.key,
            "family": spec.family,
            "license": spec.license,
            "license_ok": spec.license_ok,
            "runtime": spec.runtime,
            "notes": spec.notes,
        }
        base.update(values)
        return cls(**base)  # type: ignore[arg-type]

    def as_csv(self) -> dict[str, object]:
        record: dict[str, object] = {}
        for column in COLUMNS:
            # The CSV column is "model"; the attribute is "key".
            value = self.key if column == "model" else getattr(self, column, None)
            if column == "license_ok":
                record[column] = _tri_state(value)
            elif isinstance(value, float):
                record[column] = f"{value:.4f}" if column in {"presence_recall", "presence_specificity", "sheet_iou_mean"} else f"{value:.2f}"
            elif value is None:
                record[column] = ""
            elif isinstance(value, bool):
                record[column] = "yes" if value else "no"
            else:
                record[column] = value
        return record


def _tri_state(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unclear"


def write_csv(rows: Sequence[Row], path: str | Path) -> Path:
    """Write the machine-readable table, creating the parent directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv())
    return target


def write_details(rows: Sequence[Row], path: str | Path) -> Path:
    """Per-image detail: one line per model and sample, for the "why" behind a score."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "model",
        "sample_id",
        "scenario",
        "positive",
        "boxes",
        "best_iou",
        "hit",
        "present",
        "chosen_confidence",
        "reason",
        "latency_ms",
    )
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            for sample in row.samples:
                record = {"model": row.key, **sample}
                writer.writerow({key: record.get(key, "") for key in columns})
    return target


def format_table(rows: Sequence[Row]) -> str:
    """A fixed-width table for the terminal and for `doc/`."""
    header = f"{'model':22s} {'licence':12s} {'MB':>7s} {'p50 ms':>8s} {'fps':>6s} {'present':>8s} {'iou':>6s} {'warehouse':>10s}"
    lines = [header, "-" * len(header)]
    for row in rows:
        present = "-" if row.presence_recall is None else f"{row.presence_recall:.0%}"
        iou = "-" if row.sheet_iou_mean is None else f"{row.sheet_iou_mean:.2f}"
        p50 = "-" if row.latency_p50_ms is None else f"{row.latency_p50_ms:.0f}"
        fps = "-" if row.fps is None else f"{row.fps:.1f}"
        weights = "-" if row.weights_mb is None else f"{row.weights_mb:.0f}"
        warehouse = "-" if row.warehouse_hits is None else f"{row.warehouse_hits:.0f}"
        licence = row.license if row.license_ok is not False else f"{row.license}!"
        lines.append(
            f"{row.key:22s} {licence:12s} {weights:>7s} {p50:>8s} {fps:>6s} {present:>8s} {iou:>6s} {warehouse:>10s}"
        )
    return "\n".join(lines)


def format_blocked(rows: Sequence[Row]) -> str:
    """The models that could not be measured, and why - never silently dropped."""
    blocked = [row for row in rows if not row.measured]
    if not blocked:
        return ""
    lines = ["", "not measurable here:"]
    lines.extend(f"  {row.key}: {row.notes}" for row in blocked)
    return "\n".join(lines)


__all__ = [
    "COLUMNS",
    "Row",
    "format_blocked",
    "format_table",
    "write_csv",
    "write_details",
]
