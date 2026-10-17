"""One forward pass, several prompts, a threshold per prompt.

Run locally or in a pod with the vision image:
    python scripts/evaluate_multi.py

This is the multi-object path: every label goes into the text batch, the graph returns
one class channel per label, and each box is attributed to its best query *and* checked
against that query's own threshold. It is deliberately a single inference call - the
whole point is that multi-object detection does not cost N passes.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.vision.backend import Owl2OnnxBackend, TextQuery
from src.vision.decode import decode_queries
from src.vision.preprocess import PreprocessConfig, to_model_input
from src.vision.prompt import CameraPrompt, CameraPromptSet

REPO = Path(__file__).resolve().parents[1]
IMAGE = REPO / "assets" / "images" / "roboflow_test_detect_human_robot.webp"
PROMPTS = REPO / "assets" / "prompts"
MODEL = REPO / "src" / "owl" / "models" / "cuda" / "owl2_model.onnx"

#: Per-query thresholds as the System User would set them: rigid objects strict, a
#: human-presence query loose, because missing a person is worse than a false alarm.
TUNING = {
    "robot": 0.25,
    "human": 0.10,
    "person": 0.25,
    "metal sheet": 0.50,
    "cardboard box": 0.25,
}


def load(name: str) -> CameraPrompt:
    path = PROMPTS / f"{name.replace(' ', '_')}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CameraPrompt(
        label=payload["labels"][0],
        threshold=TUNING.get(payload["labels"][0], 0.25),
        prompt_text=payload["prompt_text"],
        input_ids=tuple(payload["input_ids"]),
        attention_mask=tuple(payload["attention_mask"]),
        tokenizer=payload.get("tokenizer", ""),
        max_length=payload.get("max_length", 16),
    )


def main() -> None:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    image = cv2.imread(str(IMAGE))
    if image is None:
        raise SystemExit(f"could not read {IMAGE}")

    queries = tuple(load(name) for name in TUNING)
    prompt_set = CameraPromptSet(queries=queries, camera_id="eval")

    print("queries   :")
    for query in queries:
        print(
            f"  {query.label:<14} threshold={query.threshold:.2f} "
            f"tokens={sum(query.attention_mask)}"
        )

    backend = Owl2OnnxBackend(
        str(model_path),
        text_query=TextQuery(
            labels=(queries[0].label,),
            input_ids=queries[0].input_ids,
            attention_mask=queries[0].attention_mask,
        ),
        execution_provider="CPUExecutionProvider",
    )
    backend.warmup()

    pixel_values = to_model_input(image, PreprocessConfig(width=960, height=960))

    started = time.perf_counter()
    output = backend.infer(pixel_values, text_query=prompt_set)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    logits = np.asarray(output.logits)
    print(
        f"\ninference : {elapsed_ms:.0f} ms for {len(queries)} queries, "
        f"logits {logits.shape}"
    )

    detections = decode_queries(
        logits,
        np.asarray(output.pred_boxes, dtype=np.float32),
        labels=prompt_set.labels,
        thresholds=prompt_set.thresholds,
        iou_threshold=0.5,
        max_detections=32,
    )

    print(f"detections: {len(detections)} across {len(queries)} classes in ONE pass\n")
    for detection in detections:
        print(
            f"  {detection.object_type:<14} {detection.confidence:.3f} "
            f"centre=({detection.x_norm + detection.width_norm / 2:.3f}, "
            f"{detection.y_norm + detection.height_norm / 2:.3f}) "
            f"size=({detection.width_norm:.3f}, {detection.height_norm:.3f})"
        )

    per_class: dict[str, int] = {}
    for detection in detections:
        per_class[detection.object_type] = per_class.get(detection.object_type, 0) + 1
    print(f"\nper class : {per_class}")

    # What the loose human threshold buys, demonstrated rather than asserted.
    strict_human = [
        detection
        for detection in detections
        if detection.object_type == "human" and detection.confidence >= 0.25
    ]
    loose_human = [
        detection
        for detection in detections
        if detection.object_type == "human" and detection.confidence >= 0.10
    ]
    print(
        f"human     : {len(loose_human)} above 0.10 (configured), "
        f"{len(strict_human)} would survive 0.25"
    )


if __name__ == "__main__":
    main()
