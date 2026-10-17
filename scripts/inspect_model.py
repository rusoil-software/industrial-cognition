#!/usr/bin/env python3

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
# date   : 2026-Oct-17
# ==============================================================================
"""Print an ONNX model's I/O contract and time one forward pass.

The pipeline's preprocessing is written against the exported graph's *actual*
contract - input names, spatial size, dtypes and output shapes - not against an
assumption about the upstream model. This script is how those facts were
established, and is the first thing to run when a new artifact is deployed.

Usage::

    python scripts/inspect_model.py src/owl/models/cuda/owl2_model.onnx
    python scripts/inspect_model.py --provider CUDAExecutionProvider model.onnx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="path to the .onnx file")
    parser.add_argument(
        "--provider",
        default="",
        help="execution provider (default: whatever the installed runtime prefers)",
    )
    parser.add_argument("--runs", type=int, default=3, help="timed forward passes")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("error: onnxruntime is not installed", file=sys.stderr)
        return 2

    if not args.model.exists():
        print(f"error: {args.model} does not exist", file=sys.stderr)
        return 2

    providers = [args.provider] if args.provider else ort.get_available_providers()
    session = ort.InferenceSession(str(args.model), providers=providers)

    print(f"model     : {args.model}  ({args.model.stat().st_size / 1e6:.1f} MB)")
    print(f"providers : {session.get_providers()}")
    print("\ninputs")
    for item in session.get_inputs():
        print(f"  {item.name:16} {item.shape}  {item.type}")
    print("\noutputs")
    for item in session.get_outputs():
        print(f"  {item.name:16} {item.shape}  {item.type}")

    shapes = {i.name: i.shape for i in session.get_inputs()}
    pixel_values_shape = shapes.get("pixel_values")
    if pixel_values_shape is None:
        print("\nwarning: no `pixel_values` input; this is not the OWLv2 pipeline graph")
        return 0

    spatial = pixel_values_shape[2:]
    if not all(isinstance(dim, int) for dim in spatial):
        print(f"\nwarning: dynamic spatial shape {spatial}; the pipeline expects fixed 960x960")
        height = width = 960
    else:
        height, width = int(spatial[0]), int(spatial[1])

    text_length = 16
    ids_shape = shapes.get("input_ids")
    if ids_shape and isinstance(ids_shape[-1], int):
        text_length = ids_shape[-1]

    feed = {
        "pixel_values": np.zeros((1, 3, height, width), dtype=np.float32),
        "input_ids": np.full((1, text_length), 49406, dtype=np.int64),
        "attention_mask": np.ones((1, text_length), dtype=np.int64),
    }

    session.run(None, feed)  # warm-up
    timings = []
    for _ in range(max(1, args.runs)):
        started = time.perf_counter()
        outputs = session.run(None, feed)
        timings.append((time.perf_counter() - started) * 1000.0)

    print(f"\nimage size : {width}x{height}")
    print(f"text length: {text_length}")
    print("\noutput shapes")
    for item, value in zip(session.get_outputs(), outputs):
        print(f"  {item.name:16} {np.asarray(value).shape} {np.asarray(value).dtype}")
        if np.asarray(value).dtype == np.float16:
            print("                   (float16: cast before arithmetic)")

    timings.sort()
    median = timings[len(timings) // 2]
    print(f"\nlatency    : median {median:.1f} ms  ({timings})")
    if median > 0:
        per_worker_fps = 1000.0 / median
        workers = int(np.ceil(60.0 / per_worker_fps))
        print(f"throughput : {per_worker_fps:.1f} FPS per worker")
        print(f"workers for 60 FPS: {workers}")

    logits = np.asarray(outputs[0])
    if logits.size and np.abs(logits).max() >= 60000:
        print(
            "\nwarning: logits saturated at the fp16 limit. The text input is "
            "degenerate (e.g. padding tokens); detections would always be empty. "
            "Use scripts/encode_text_queries.py to produce real token IDs.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
