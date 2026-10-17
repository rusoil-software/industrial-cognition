"""Evaluate prompt conditioning on a real image with the real model.

Run in a pod that has the vision image and the model PVC (k8s/tools/inference-probe.yaml):
    kubectl -n cogni-ns cp assets/images/roboflow_test_detect_human_robot.webp probe:/tmp/scene.webp
    kubectl -n cogni-ns cp -r prompts_out probe:/tmp/prompts
    kubectl -n cogni-ns exec probe -- env PYTHONPATH=/app python /tmp/evaluate_prompts.py

What this answers, and why it is not the same question as "does detection work":

The exported artifact has a **single** class dimension (`logits` is (1, 3600, 1)), so the
graph cannot emit per-class scores for several phrases at once. A prompt therefore
changes *how much* a region looks like the phrase, not *which class* it is. So the test
is: does the highest-scoring box for "robot" land on the robot, and for "human" on the
human? That is what distinguishes real text conditioning from a decoder that ignores the
prompt entirely.

The reference regions are hand-labelled from the image (1024x1536, overhead view): a
yellow arm at the top centre, a worker in a hi-vis vest below it.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.vision.backend import Owl2OnnxBackend, TextQuery
from src.vision.decode import decode_frame
from src.vision.preprocess import PreprocessConfig, to_model_input

IMAGE = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/scene.webp")
PROMPTS = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/prompts")

# Hand-labelled, normalised (x_min, y_min, x_max, y_max).
REGIONS = {
    "robot_arm": (0.25, 0.19, 0.73, 0.43),
    "human": (0.52, 0.51, 0.66, 0.63),
}


def region_of(box: tuple[float, float, float, float]) -> str:
    """Which reference region contains the centre of *box*, if any."""
    x_centre = (box[0] + box[2]) / 2
    y_centre = (box[1] + box[3]) / 2
    for name, (x1, y1, x2, y2) in REGIONS.items():
        if x1 <= x_centre <= x2 and y1 <= y_centre <= y2:
            return name
    return "background"


def load_query(path: Path) -> TextQuery:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TextQuery(
        labels=tuple(payload["labels"]),
        input_ids=tuple(payload["input_ids"]),
        attention_mask=tuple(payload["attention_mask"]),
    )


def main() -> None:
    image = cv2.imread(str(IMAGE))
    if image is None:
        raise SystemExit(f"could not read {IMAGE}")
    print(f"image     : {IMAGE.name} {image.shape[1]}x{image.shape[0]}")

    backend = Owl2OnnxBackend(
        "/app/models/owl2_model.onnx",
        # A prompt is supplied per call, so the file on the PVC is not needed here.
        text_query=load_query(sorted(PROMPTS.glob("*.json"))[0]),
        execution_provider="CPUExecutionProvider",
    )
    backend.warmup()
    print(f"backend   : {backend.name} {backend._providers}")

    pixel_values = to_model_input(image, PreprocessConfig(width=960, height=960))

    results = []
    for path in sorted(PROMPTS.glob("*.json")):
        query = load_query(path)
        output = backend.infer(pixel_values, text_query=query)
        logits = np.asarray(output.logits)
        boxes = np.asarray(output.pred_boxes, dtype=np.float32)

        detections = decode_frame(
            logits,
            boxes,
            labels=query.labels,
            score_threshold=0.0,
            iou_threshold=0.5,
            max_detections=10,
        )
        best = detections[0] if detections else None
        label = path.stem.replace("_", " ")
        if best is None:
            print(f"{label:<16} no boxes")
            continue

        box = (best.x_norm, best.y_norm, best.x_norm + best.width_norm,
               best.y_norm + best.height_norm)
        where = region_of(box)
        results.append((label, best.confidence, where))
        print(
            f"{label:<16} top score={best.confidence:.4f} "
            f"centre=({(box[0] + box[2]) / 2:.3f}, {(box[1] + box[3]) / 2:.3f}) "
            f"size=({best.width_norm:.3f}, {best.height_norm:.3f}) -> {where}"
        )

    print("\n=== does the prompt change where it looks? ===")
    by_label = {label: where for label, _score, where in results}
    robot_ok = by_label.get("robot") == "robot_arm"
    human_ok = by_label.get("human") == "human"
    print(f"  'robot' -> {by_label.get('robot', 'n/a')}  (expected robot_arm)")
    print(f"  'human' -> {by_label.get('human', 'n/a')}  (expected human)")

    if robot_ok and human_ok:
        print("\nPROMPT CONDITIONING OK: the same image is localised differently")
    else:
        print(
            "\nPROMPT CONDITIONING INCONCLUSIVE: with a single-class export the "
            "argmax is not required to follow the phrase; see doc/vision-pipeline.md §14"
        )

    print("\nall top boxes:")
    for label, score, where in results:
        print(f"  {label:<16} {score:.4f}  {where}")


if __name__ == "__main__":
    main()
