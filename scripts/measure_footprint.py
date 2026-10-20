"""Measure the ONNX worker's memory and latency envelope per session configuration.

Run in the worker image, one process per configuration (peak RSS is per process, so two
configurations cannot share one):

    python scripts/measure_footprint.py --matrix
    python scripts/measure_footprint.py --config arena_off

Why this exists: item 2 of the pre-Stage-3 backlog is "make the worker lean enough for two".
One resident session peaks at ~3.4 GiB, so two workers plus the rest of the stack do not fit a
10 GiB node. Before trimming anything, the levers have to be measured rather than guessed -
ONNX Runtime's memory arena and memory-pattern flags are the first suspects, thread count the
second.

Each configuration reports:
  peak_rss_mib  - VmHWM, the number that decides how many workers fit
  load_ms       - session creation (cold start / warmup cost)
  latency_ms    - median of a few warm inferences on the 960x960 contract
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODEL = Path(os.getenv("OWL2_MODEL_PATH", "src/owl/models/cuda/owl2_model.onnx"))
TEXT_IDS = os.getenv("OWL2_TEXT_IDS_PATH", "k8s/dev/text_queries.json")

#: name -> SessionOptions overrides (and thread behaviour).
CONFIGS: dict[str, dict] = {
    "baseline": {},  # what the pipeline uses today
    "no_arena": {"enable_cpu_mem_arena": False},
    "no_mem_pattern": {"enable_mem_pattern": False},
    "lean": {"enable_cpu_mem_arena": False, "enable_mem_pattern": False},
    "single_thread": {"intra_op_num_threads": 1},
    "lean_single_thread": {
        "enable_cpu_mem_arena": False,
        "enable_mem_pattern": False,
        "intra_op_num_threads": 1,
    },
}


def peak_rss_mib() -> float:
    """Peak resident set size from /proc; 0 where /proc is unavailable."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def run_config(name: str) -> dict:
    import onnxruntime as ort

    from src.vision.backend import TextQuery, effective_cpu_count
    from src.vision.preprocess import PreprocessConfig, to_model_input

    overrides = CONFIGS[name]

    def setting(key: str, default):
        return overrides.get(key, default)

    options = ort.SessionOptions()
    # Same sizing rule as the backend: the cgroup quota, not the node's cores. Using
    # os.cpu_count() here measured a 20-thread pool against a 6-CPU limit, which reads as
    # "the arena costs 12% latency" when part of it is really thread thrash.
    options.intra_op_num_threads = int(
        setting("intra_op_num_threads", effective_cpu_count())
    )
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_mem_pattern = bool(setting("enable_mem_pattern", True))
    options.enable_cpu_mem_arena = bool(setting("enable_cpu_mem_arena", True))
    options.log_severity_level = 3

    started = time.perf_counter()
    session = ort.InferenceSession(
        str(MODEL), options, providers=["CPUExecutionProvider"]
    )
    load_ms = (time.perf_counter() - started) * 1000.0

    query = json.loads(Path(TEXT_IDS).read_text(encoding="utf-8"))
    text = TextQuery(
        labels=tuple(query["labels"]),
        input_ids=tuple(query["input_ids"]),
        attention_mask=tuple(query["attention_mask"]),
    )
    ids, mask = text.as_arrays(expected_length=16)

    image = np.full((960, 960, 3), 90, dtype=np.uint8)
    image[300:600, 350:650] = (205, 210, 215)
    pixel_values = to_model_input(image, PreprocessConfig(width=960, height=960))

    feed = {"pixel_values": pixel_values, "input_ids": ids, "attention_mask": mask}
    session.run(None, feed)  # warm up kernels and arenas

    timings = []
    for _ in range(3):
        began = time.perf_counter()
        session.run(None, feed)
        timings.append((time.perf_counter() - began) * 1000.0)

    return {
        "config": name,
        "intra_op": options.intra_op_num_threads,
        "arena": options.enable_cpu_mem_arena,
        "mem_pattern": options.enable_mem_pattern,
        "load_ms": round(load_ms),
        "latency_ms": round(statistics.median(timings)),
        "peak_rss_mib": round(peak_rss_mib()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(CONFIGS))
    parser.add_argument("--matrix", action="store_true", help="run every configuration")
    args = parser.parse_args()

    if not MODEL.exists():
        print(f"model not found: {MODEL}", file=sys.stderr)
        return 2

    if args.matrix:
        results = []
        for name in CONFIGS:
            # A fresh process per configuration: VmHWM never goes down, so measuring two
            # configurations in one process would report the first one's peak for the second.
            completed = subprocess.run(
                [sys.executable, __file__, "--config", name],
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            if completed.returncode != 0:
                print(f"{name}: failed\n{completed.stderr[-400:]}", file=sys.stderr)
                continue
            results.append(json.loads(completed.stdout))

        header = f"{'config':<20} {'threads':>7} {'arena':>6} {'pattern':>7} {'load_ms':>8} {'lat_ms':>8} {'peak_MiB':>9}"
        print(header)
        print("-" * len(header))
        for row in results:
            print(
                f"{row['config']:<20} {row['intra_op']:>7} {str(row['arena']):>6} "
                f"{str(row['mem_pattern']):>7} {row['load_ms']:>8} {row['latency_ms']:>8} "
                f"{row['peak_rss_mib']:>9}"
            )
        return 0

    if not args.config:
        parser.print_help()
        return 2

    print(json.dumps(run_config(args.config)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
