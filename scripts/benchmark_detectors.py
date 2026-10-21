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
"""Compare candidate detectors on the same bytes, the same prompts and the same rule.

    python scripts/benchmark_detectors.py --models all --out reports/detector_benchmark.csv

The output is the evidence, not the opinion: one CSV row per model with licence, weight
size, load time, latency, throughput, memory and the accuracy columns the station
protocol and the warehouse scene define. ``--details`` adds one row per image so a score
can be traced back to the box that produced it, and models that could not be measured
stay in the table with the reason, because a silently missing row is the easiest way to
pretend a comparison happened.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import dataset as dataset_module  # noqa: E402
from src.benchmark.measure import (  # noqa: E402
    Latency,
    gpu_memory_mb,
    peak_rss_mb,
    reset_gpu_peak,
    timed,
    weights_size_mb,
)
from src.benchmark.registry import MODELS, license_summary, resolve, runnable  # noqa: E402
from src.benchmark.report import (  # noqa: E402
    Row,
    format_blocked,
    format_table,
    write_csv,
    write_details,
)
from src.benchmark.runners import build_runner  # noqa: E402
from src.benchmark.scoring import aggregate, score_sample  # noqa: E402

LOGGER = logging.getLogger(__name__)

#: Where the CSV lands by default. ``reports/`` is ignored by git: the numbers are
#: reproducible from the script, the file itself is a build artifact.
DEFAULT_OUT = "reports/detector_benchmark.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        default="all",
        help="comma-separated registry keys, or 'all' (default: all)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"CSV path (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--details",
        default=None,
        help="optional per-image CSV path (default: <out stem>_details.csv)",
    )
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto", help="auto uses CUDA when present")
    parser.add_argument("--images", default="assets/images", help="directory holding the benchmark images")
    parser.add_argument("--phases", type=int, default=8, help="station pan phases (default: 8)")
    parser.add_argument("--json", default=None, help="optional path for the same rows as JSON")
    parser.add_argument("--quiet", action="store_true", help="only print the final table")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.device == "cpu":
        # Hides the GPU from every runtime that asks, including torch's own check.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    samples = dataset_module.build_dataset(args.images)
    print(f"dataset: {json.dumps(dataset_module.summarise(samples))}")

    keys = [spec.key for spec in MODELS] if args.models.strip().lower() == "all" else [
        part.strip() for part in args.models.split(",") if part.strip()
    ]
    specs = resolve(keys)
    rows: list[Row] = []
    for spec in specs:
        if not spec.available:
            rows.append(Row.from_spec(spec, measured=False, device=""))
            print(f"{spec.key}: not measurable - {spec.notes}")
            continue
        row = _measure_one(spec, samples, args)
        rows.append(row)

    out = write_csv(rows, args.out)
    details_path = args.details or str(Path(args.out).with_name(Path(args.out).stem + "_details.csv"))
    write_details(rows, details_path)
    if args.json:
        _write_json(rows, args.json)

    print()
    print(format_table(rows))
    print()
    print("licences:")
    for spec in specs:
        ok = "yes" if spec.license_ok else ("no" if spec.license_ok is False else "unclear")
        print(f"  {spec.key:22s} {spec.license:14s} Apache-2.0 compatible: {ok}")
    blocked = format_blocked(rows)
    if blocked:
        print(blocked)
    print()
    print(f"csv: {out}")
    print(f"details: {details_path}")
    print(f"dataset: {json.dumps(dataset_module.summarise(samples))}, gate: {'shared with the pipeline'}")
    print(license_summary(runnable(specs)))
    return 0


def _grouped(samples) -> list[tuple[tuple[str, ...], list[tuple[int, object]]]]:
    """Samples grouped by vocabulary, keeping the order in which each vocabulary appears."""
    order: list[tuple[str, ...]] = []
    groups: dict[tuple[str, ...], list[tuple[int, object]]] = {}
    for index, sample in enumerate(samples):
        vocabulary = tuple(sample.labels)
        if vocabulary not in groups:
            groups[vocabulary] = []
            order.append(vocabulary)
        groups[vocabulary].append((index, sample))
    return [(vocabulary, groups[vocabulary]) for vocabulary in order]


def _measure_one(spec, samples, args) -> Row:
    print(f"{spec.key}: loading ...")
    runner = build_runner(spec.key)
    latency = Latency()
    reset_gpu_peak()
    load_seconds: Optional[float] = None
    try:
        import time

        reset_gpu_peak()
        started = time.perf_counter()
        runner.load()
        load_seconds = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001 - a missing wheel must not end the run
        LOGGER.debug("load failed for %s", spec.key, exc_info=True)
        runner.close()
        return Row.from_spec(
            spec,
            measured=False,
            device=getattr(runner, "device", "") or "",
            notes=f"could not load: {type(exc).__name__}: {exc}",
        )

    scored: dict[int, object] = {}
    failures = 0
    # A detector can pay a one-off cost when it is given a new vocabulary: the YOLO-World
    # adapter rebuilds its CLIP text embeddings, measured at 6-9 s on this GPU. That is a
    # configuration cost (a prompt change), not a per-frame cost, so the samples are measured
    # vocabulary by vocabulary, warming each one untimed first. Everything else stays inside
    # the measurement, including OWLv2's per-label forward pass, which is real per-frame work.
    for vocabulary, members in _grouped(samples):
        _, first_sample = members[0]
        try:
            runner.detect(first_sample.image, first_sample.labels)
        except Exception as exc:  # noqa: BLE001 - the timed call below reports it properly
            LOGGER.warning("%s could not warm up %s: %s", spec.key, vocabulary, exc)
        for index, sample in members:
            try:
                with timed(latency):
                    predictions = runner.detect(sample.image, sample.labels)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                LOGGER.warning("%s failed on %s: %s", spec.key, sample.sample_id, exc)
                LOGGER.debug("traceback", exc_info=True)
                continue
            scored[index] = score_sample(sample, predictions, latency_ms=latency.samples_ms[-1])
    scores = [scored[index] for index in sorted(scored)]

    metrics = aggregate(scores) if scores else {}
    params = None
    try:
        params = runner.params()
    except Exception:  # noqa: BLE001
        params = None
    peak_gpu = gpu_memory_mb()
    # A downloaded .pt only exists after load(); asking the adapter where it put the file
    # is the only way to size it, because the registry knows the name, not the path.
    weights_reference = getattr(runner, "weights", None) or spec.weights
    try:
        runner.close()
    except Exception:  # noqa: BLE001
        pass

    notes = spec.notes
    if failures:
        notes = (notes + "; " if notes else "") + f"{failures} sample(s) failed"

    row = Row.from_spec(
        spec,
        device=getattr(runner, "device", "") or "",
        weights_mb=weights_size_mb(weights_reference),
        params_m=params,
        load_s=load_seconds,
        latency_p50_ms=latency.percentile(0.50),
        latency_p95_ms=latency.percentile(0.95),
        fps=latency.fps(),
        peak_rss_mb=peak_rss_mb(),
        peak_gpu_mb=peak_gpu,
        presence_recall=metrics.get("presence_recall"),
        presence_specificity=metrics.get("presence_specificity"),
        sheet_iou_mean=metrics.get("sheet_iou_mean"),
        warehouse_hits=metrics.get("warehouse_hits"),
        measured=bool(scores),
        notes=notes,
    )
    row.samples = [score.as_dict() for score in scores]
    print(
        f"{spec.key}: p50 {row.latency_p50_ms:.0f} ms, {row.fps:.1f} fps, "
        f"presence {row.presence_recall:.0%}, iou {row.sheet_iou_mean:.2f}"
        if row.measured
        else f"{spec.key}: no samples measured"
    )
    return row


def _write_json(rows: Sequence[Row], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for row in rows:
        record = row.as_csv()
        record["samples"] = row.samples
        payload.append(record)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130) from None
