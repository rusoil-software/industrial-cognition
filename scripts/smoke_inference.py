"""Inspect what the real ONNX artifact actually returns in-cluster.

Run inside a pod that has the vision image and the model PVC mounted:
    kubectl -n cogni-ns cp scripts/smoke_inference.py <pod>:/tmp/probe.py
    kubectl -n cogni-ns exec <pod> -- env PYTHONPATH=/app python /tmp/probe.py

Why this exists separately from the Celery task smoke test: that one asserts the path
executes and returns a well-formed result. It cannot show whether the graph produces
*usable* numbers, because the decoder legitimately filters everything below the score
threshold. Here the threshold is disabled so the raw distribution - logits range,
scores, boxes - is visible, which is how you tell "nothing in view" apart from "the
text path is degenerate".

Note on the prompt: the tokenised file on this PVC holds **placeholder** IDs, not a
CLIP encoding of "metal sheet", so the detections are meaningless. What is verified
here is the inference and decode machinery, and the memory/latency envelope.
"""

import os
import time

import numpy as np

from src.vision.backend import build_backend
from src.vision.decode import decode_frame
from src.vision.preprocess import PreprocessConfig, to_model_input


def peak_rss_mib() -> float:
    """Peak resident set size from /proc, so the number is measured, not guessed."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def synthetic_scene(size: int = 960) -> np.ndarray:
    """A BGR frame with one bright rectangle, so there is a plausible object."""
    frame = np.full((size, size, 3), 60, dtype=np.uint8)
    frame[300:600, 350:650] = (205, 210, 215)
    return frame


def main() -> None:
    backend = build_backend("owl2")
    print(f"backend   : {backend.name}")
    print(f"model     : {os.getenv('OWL2_MODEL_PATH')}")
    print(f"labels    : {backend._text_query.labels}")
    print(f"text ids  : {list(backend._text_query.input_ids)}")

    started = time.perf_counter()
    backend.warmup()
    warmup_ms = (time.perf_counter() - started) * 1000.0
    print(f"warmup    : {warmup_ms:.0f} ms")
    print(f"providers : {getattr(backend, '_providers', 'n/a')}")
    try:
        options = backend._session.get_session_options()
        print(
            f"threads   : intra_op={options.intra_op_num_threads} "
            f"inter_op={options.inter_op_num_threads}"
        )
    except Exception:  # noqa: BLE001 - diagnostics only
        pass

    image = synthetic_scene()
    pixel_values = to_model_input(image, PreprocessConfig(width=960, height=960))

    timings = []
    output = None
    for _ in range(3):
        began = time.perf_counter()
        output = backend.infer(pixel_values)
        timings.append((time.perf_counter() - began) * 1000.0)

    logits = np.asarray(output.logits)
    boxes = np.asarray(output.pred_boxes, dtype=np.float32)
    print(f"\nlatency   : {[round(t) for t in timings]} ms")
    print(
        f"logits    : shape={logits.shape} min={logits.min():.3f} "
        f"max={logits.max():.3f} mean={logits.mean():.3f}"
    )
    print(f"boxes     : shape={boxes.shape} min={boxes.min():.4f} max={boxes.max():.4f}")
    saturated = float(logits.max()) >= 60000 or float(logits.min()) <= -60000
    print(f"saturation: {'none (text path is not degenerate)' if not saturated else 'DETECTED'}")

    # The score the threshold actually acts on is the sigmoid of the best logit.
    best_score = 1.0 / (1.0 + np.exp(-float(logits.max())))
    print(f"best score: {best_score:.5f}")

    detections = decode_frame(
        logits,
        boxes,
        labels=backend._text_query.labels,
        score_threshold=0.0,
        iou_threshold=0.5,
        max_detections=32,
    )
    print(f"\ndecoded   : {len(detections)} boxes after NMS at threshold 0.0")
    for index, detection in enumerate(detections[:5]):
        print(
            f"  #{index} {detection.object_type} score={detection.confidence:.5f} "
            f"box=({detection.x_norm:.3f}, {detection.y_norm:.3f}) "
            f"size=({detection.width_norm:.3f}, {detection.height_norm:.3f})"
        )

    for threshold in (0.01, 0.10, 0.25, 0.50):
        count = len(
            decode_frame(
                logits,
                boxes,
                labels=backend._text_query.labels,
                score_threshold=threshold,
                iou_threshold=0.5,
                max_detections=32,
            )
        )
        print(f"above {threshold:.2f}: {count}")

    print(f"\npeak RSS  : {peak_rss_mib():.0f} MiB")
    print("\nINFERENCE SMOKE OK")


if __name__ == "__main__":
    main()
