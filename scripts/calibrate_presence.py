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
# date   : 2026-Oct-21
# ==============================================================================
"""Work out the score threshold for a presence check from a handful of photographs.

OWLv2 scores the same object very differently in different scenes. On the real
metal-sheet photograph in `assets/images/metal_sheet_test_image.jpg` the prompt
"steel blank" peaks at 0.25, while empty crops of the same room stay under 0.06 -
a threshold of 0.20 is comfortable there. In the warehouse image the same prompt
fires on the racking above 0.25. Neither number is wrong: the model is doing zero-shot
localisation, and "is there a sheet of metal" depends on where the camera is looking.

That is why the threshold belongs to the *camera*, not to the model, and why it has
to be measured instead of guessed. This script takes a folder of photographs from
the station, some with the object and some without, scores them with one prompt, and
prints a threshold that sits between the two groups. Feed the result to
``PUT /cameras/{id}/prompt``.

Usage::

    # positive = the sheet is in the frame, negative = the station is empty
    python scripts/calibrate_presence.py \
        --query assets/prompts/steel_blank.json \
        --positive "photos/station_with_sheet/*.jpg" \
        --negative "photos/station_empty/*.jpg"

Without ``--query`` the labels are tokenised on the fly, which needs `transformers`
(the same export-time dependency `scripts/encode_text_queries.py` uses)::

    python scripts/calibrate_presence.py --labels "steel blank" \
        --positive photos/with/*.jpg --negative photos/without/*.jpg

The exit status is 0 when the two groups separate and 1 when they do not, so it can
be used as a gate in a deployment script. What it cannot do is invent evidence: five
photographs from one morning do not cover a week of lighting changes.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

#: Scores this close to a threshold are a coin flip, so a usable threshold has to be
#: at least this far from both groups. A tenth of a point is roughly the spacing
#: between a confident zero-shot hit and a plausible-looking false positive.
DEFAULT_MARGIN = 0.05

#: Formats OpenCV decodes here. WebP needs a recent OpenCV, which the images have.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

#: Copied from `scripts/encode_text_queries.py`: OWLv2 pads every phrase to 16 tokens.
DEFAULT_MAX_LENGTH = 16
DEFAULT_MODEL_ID = "google/owlv2-base-patch16-ensemble"

#: The exported graph runs at a fixed 960x960; the preprocessor letterboxes into it.
MODEL_INPUT = 960

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "src" / "owl" / "models" / "cuda" / "owl2_model.onnx"


def expand_images(patterns: Iterable[str], root: Optional[Path] = None) -> list[Path]:
    """Turn file paths, directories and globs into a sorted list of image files.

    Directories are scanned one level deep for :data:`IMAGE_SUFFIXES`. A pattern that
    matches nothing is dropped silently: the caller reports the count, and a typo shows
    up as "0 positive images" rather than as a crash in the middle of a model run.
    """
    base = root or Path.cwd()
    found: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if not candidate.is_absolute():
            candidate = base / candidate
        if candidate.is_dir():
            matches = [
                path
                for path in sorted(candidate.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
        else:
            matches = [Path(match) for match in sorted(glob.glob(str(candidate)))]
        for match in matches:
            resolved = match.resolve()
            if resolved.is_file() and resolved.suffix.lower() in IMAGE_SUFFIXES:
                if resolved not in found:
                    found.append(resolved)
    return found


def threshold_from(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    margin: float = DEFAULT_MARGIN,
) -> Optional[float]:
    """Return a threshold between the two groups, or None when they overlap.

    The midpoint is used rather than "just above the highest negative" because the
    midpoint is the point furthest from both mistakes, and the mistake that matters
    here is symmetric: a missed sheet stops the station, a phantom sheet stops it too.
    """
    if not positive_scores or not negative_scores:
        return None
    lowest_positive = min(positive_scores)
    highest_negative = max(negative_scores)
    if lowest_positive - highest_negative < 2 * margin:
        return None
    return round((lowest_positive + highest_negative) / 2, 3)


def format_table(rows: Sequence[tuple[str, str, float, Optional[tuple[float, float]]]]) -> str:
    """Render `(group, name, score, centre)` rows as an aligned block."""
    lines = [f"{'group':10s}{'score':>8s}  {'centre':22s}image"]
    for group, name, score, centre in rows:
        where = "" if centre is None else f"({centre[0]:.3f}, {centre[1]:.3f})"
        lines.append(f"{group:10s}{score:8.3f}  {where:22s}{name}")
    return "\n".join(lines)


def load_query(path: Path):
    """Build a :class:`TextQuery` from a committed prompt fixture."""
    from src.vision.backend import TextQuery

    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels") or [path.stem.replace("_", " ")]
    return TextQuery(
        labels=tuple(labels),
        input_ids=tuple(int(v) for v in payload["input_ids"]),
        attention_mask=tuple(int(v) for v in payload["attention_mask"]),
    )


def encode_labels(labels: Sequence[str], model_id: str = DEFAULT_MODEL_ID):
    """Tokenise labels with the CLIP tokenizer (needs `transformers`)."""
    from transformers import AutoTokenizer

    from src.vision.backend import TextQuery

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = " ".join(f"{label}." for label in labels)
    encoded = tokenizer(
        prompt,
        return_tensors="np",
        padding="max_length",
        max_length=DEFAULT_MAX_LENGTH,
        truncation=True,
    )
    return TextQuery(
        labels=tuple(labels),
        input_ids=tuple(int(v) for v in encoded["input_ids"][0]),
        attention_mask=tuple(int(v) for v in encoded["attention_mask"][0]),
    )


def score_image(backend, image, query) -> tuple[float, Optional[tuple[float, float]]]:
    """Return the best score for one image and where that box sits (normalised)."""
    from src.vision.decode import decode_frame
    from src.vision.preprocess import PreprocessConfig, to_model_input

    output = backend.infer(
        to_model_input(image, PreprocessConfig(width=MODEL_INPUT, height=MODEL_INPUT)),
        text_query=query,
    )
    detections = decode_frame(
        output.logits,
        output.pred_boxes,
        labels=query.labels,
        score_threshold=0.0,
        iou_threshold=0.5,
        max_detections=1,
    )
    if not detections:
        return 0.0, None
    best = detections[0]
    centre = (best.x_norm + best.width_norm / 2, best.y_norm + best.height_norm / 2)
    return float(best.confidence), centre


def _read_image(path: Path):
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"cannot decode {path}")
    return image


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=Path,
        help="Prompt fixture to score with, e.g. assets/prompts/steel_blank.json",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated labels to tokenise instead of --query (needs transformers)",
    )
    parser.add_argument(
        "--positive",
        nargs="+",
        required=True,
        help="Photos where the object IS there: files, directories or globs",
    )
    parser.add_argument(
        "--negative",
        nargs="+",
        required=True,
        help="Photos where the object is NOT there: files, directories or globs",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"ONNX artifact (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=f"Required distance from both groups (default: {DEFAULT_MARGIN})",
    )
    args = parser.parse_args(argv)

    if args.query is None and not args.labels.strip():
        print("error: pass --query or --labels", file=sys.stderr)
        return 2
    if not args.model.exists():
        print(f"error: model artifact not found: {args.model}", file=sys.stderr)
        return 2

    positives = expand_images(args.positive)
    negatives = expand_images(args.negative)
    if not positives:
        print("error: no positive images matched", file=sys.stderr)
        return 2
    if not negatives:
        print("error: no negative images matched", file=sys.stderr)
        return 2

    from src.vision.backend import Owl2OnnxBackend

    if args.query is not None:
        query = load_query(args.query)
    else:
        labels = [label.strip() for label in args.labels.split(",") if label.strip()]
        query = encode_labels(labels)

    backend = Owl2OnnxBackend(str(args.model), text_query=query)

    rows: list[tuple[str, str, float, Optional[tuple[float, float]]]] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    failures = 0
    for group, paths, bucket in (
        ("positive", positives, positive_scores),
        ("negative", negatives, negative_scores),
    ):
        for path in paths:
            try:
                score, centre = score_image(backend, _read_image(path), query)
            except Exception as exc:  # a broken file must not hide the other photos
                print(f"warning: {path.name}: {exc}", file=sys.stderr)
                failures += 1
                continue
            bucket.append(score)
            rows.append((group, path.name, score, centre))

    print(format_table(rows))
    print()
    print(f"prompt      : {' '.join(f'{label}.' for label in query.labels)}")
    print(f"images      : {len(positive_scores)} positive, {len(negative_scores)} negative")
    if positive_scores:
        print(f"positive    : min {min(positive_scores):.3f}  max {max(positive_scores):.3f}")
    if negative_scores:
        print(f"negative    : min {min(negative_scores):.3f}  max {max(negative_scores):.3f}")

    threshold = threshold_from(positive_scores, negative_scores, args.margin)
    if threshold is None:
        print(
            "result      : the groups overlap or sit closer than "
            f"2 x {args.margin:.2f}; a single threshold would make mistakes either way",
            file=sys.stderr,
        )
        print(
            "next        : photograph the same station again (lighting, angle, the sheet "
            "half-covered) until the empty and loaded shots separate, or pick another "
            "prompt",
            file=sys.stderr,
        )
        return 1

    print(f"threshold   : {threshold}")
    print(
        "next        : PUT /cameras/{id}/prompt with this score for the camera that "
        "watches the station"
    )
    if failures:
        print(f"skipped     : {failures} file(s) could not be read", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
