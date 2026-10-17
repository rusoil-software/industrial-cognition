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
"""Tokenise the detection prompt into the IDs the ONNX graph consumes.

Why this is a separate, one-off script rather than part of the worker:

* Tokenising needs the CLIP tokenizer, which lives in `transformers`. The worker
  image does not otherwise need `transformers`, `torch` or `tokenizers`, and
  pulling them in would add hundreds of megabytes to every replica.
* The prompt is deployment data, not per-frame data. It changes when the target
  object class changes, not 60 times a second.

This matters more than it looks: the artifact in this repository was exported with
**random** dummy token IDs (`src/owl/inference/utils.py::_create_dummy_inputs`
uses `torch.randint`), so the IDs that actually select "metal sheet" are not
recoverable from the model. Feeding zeros saturates the graph's fp16 text path at
-65504 and yields zero detections forever. Running this script is what makes the
pipeline detect anything.

Usage::

    pip install 'transformers>=4.52'      # export-time only
    python scripts/encode_text_queries.py \
        --labels "metal sheet,detail,debris" \
        --output src/owl/models/text_queries.json

Then either ship that file in the image's model directory or point
``OWL2_TEXT_IDS_PATH`` at it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: OWLv2 was trained with the CLIP tokenizer and a 16-token text budget.
DEFAULT_MAX_LENGTH = 16
DEFAULT_MODEL_ID = "google/owlv2-base-patch16-ensemble"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        required=True,
        help=(
            "Comma-separated object names. OWLv2 expects space-separated phrases "
            "and a trailing period, e.g. 'metal sheet. detail. debris.'"
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model id whose tokenizer to use (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/owl/models/text_queries.json"),
        help="Where to write the JSON (default: src/owl/models/text_queries.json)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Padded token length, must match the exported graph (default: {DEFAULT_MAX_LENGTH})",
    )
    args = parser.parse_args()

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        print("error: --labels produced an empty list", file=sys.stderr)
        return 2

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "error: `transformers` is required to tokenise the prompt.\n"
            "       It is an export-time dependency: pip install 'transformers>=4.52'",
            file=sys.stderr,
        )
        return 2

    # OWLv2's convention: one phrase per class, terminated with a period, joined by
    # spaces. The tokenizer then pads to the fixed 16-token budget.
    prompt = " ".join(f"{label}." for label in labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoded = tokenizer(
        prompt,
        return_tensors="np",
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
    )

    input_ids = [int(v) for v in encoded["input_ids"][0]]
    attention_mask = [int(v) for v in encoded["attention_mask"][0]]

    payload = {
        "labels": labels,
        "prompt": prompt,
        "tokenizer": args.model,
        "max_length": args.max_length,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"prompt      : {prompt}")
    print(f"input_ids   : {input_ids}")
    print(f"attention   : {attention_mask}")
    print(f"active tokens: {sum(attention_mask)}/{len(attention_mask)}")
    print(f"written     : {args.output}")

    if all(value == 0 for value in input_ids):
        print(
            "error: every token id is zero, which is the padding token; the graph "
            "would saturate and detect nothing",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
