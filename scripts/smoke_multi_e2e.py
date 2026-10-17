"""Live check: set two queries with different thresholds, analyse the real image.

Run in an api pod:
    kubectl -n cogni-ns cp assets/images/roboflow_test_detect_human_robot.webp <pod>:/tmp/scene.webp
    kubectl -n cogni-ns cp scripts/smoke_multi_e2e.py <pod>:/tmp/multi_e2e.py
    kubectl -n cogni-ns exec <pod> -- env PYTHONPATH=/app python /tmp/multi_e2e.py

Exercises both halves of the multi-object feature through the real surface:

* the System User sets **one query per object, each with its own threshold**, over HTTP;
* the worker runs **one forward pass** with both labels in the text batch, attributes each
  box to its best query and applies that query's threshold.
"""

import base64
import json
import time

import cv2
import httpx

from src.tasks.celery import celery_app

BASE = "http://localhost:8000"
CAMERA = "main"
IMAGE = "/tmp/scene.webp"

#: Rigid object strict, person loose: missing a person costs more than a false alarm.
QUERIES = [
    {"label": "robot", "threshold": 0.25},
    {"label": "person", "threshold": 0.10},
    {"label": "cardboard box", "threshold": 0.35},
]


def main() -> None:
    print("=== 1. configure the queries over HTTP ===")
    response = httpx.put(
        f"{BASE}/cameras/{CAMERA}/prompt", json={"prompts": QUERIES}, timeout=60
    )
    print(f"PUT -> {response.status_code}")
    if response.status_code != 200:
        print(response.text)
        return
    stored = response.json()
    for query in stored["prompts"]:
        print(
            f"   {query['label']:<14} threshold={query['threshold']:.2f} "
            f"tokens={query['active_tokens']} ids={query['input_ids'][:5]}"
        )

    print("\n=== 2. read back ===")
    fetched = httpx.get(f"{BASE}/cameras/{CAMERA}/prompt", timeout=10).json()
    print(f"   same queries: {[q['label'] for q in fetched['prompts']]}")
    print(f"   thresholds  : {[q['threshold'] for q in fetched['prompts']]}")

    print("\n=== 3. analyse the real image in ONE pass ===")
    image = cv2.imread(IMAGE)
    if image is None:
        print(f"   cannot read {IMAGE}")
        return
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    assert ok
    print(f"   image: {image.shape[1]}x{image.shape[0]}, {len(buffer) / 1024:.0f} KiB jpeg")

    payload = {
        "camera_id": CAMERA,
        "frame_number": 1,
        "capture_ts_ns": time.time_ns(),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "encode_ms": 0.0,
        "image_b64": base64.b64encode(buffer.tobytes()).decode("ascii"),
        "prompt": stored,
    }

    started = time.time()
    task = celery_app.send_task("vision.process_frame", args=[payload], queue="vision")
    value = task.get(timeout=300)
    elapsed = time.time() - started

    print(f"   error      : {value['error']}")
    print(f"   timings    : {json.dumps(value['timings'])}")
    print(f"   round trip : {elapsed:.1f}s")

    counts: dict[str, int] = {}
    for detection in value["detections"]:
        counts[detection["object_type"]] = counts.get(detection["object_type"], 0) + 1
    print(f"   detections : {len(value['detections'])} {counts}")

    for detection in value["detections"][:6]:
        centre_x = detection["x_norm"] + detection["width_norm"] / 2
        centre_y = detection["y_norm"] + detection["height_norm"] / 2
        print(
            f"     {detection['object_type']:<14} {detection['confidence']:.3f} "
            f"centre=({centre_x:.3f}, {centre_y:.3f})"
        )

    print("\n=== 4. clean up ===")
    print(f"   DELETE -> {httpx.delete(f'{BASE}/cameras/{CAMERA}/prompt', timeout=10).status_code}")

    ok = (
        value["error"] is None
        and len(stored["prompts"]) == len(QUERIES)
        and all(q["threshold"] == r["threshold"] for q, r in zip(stored["prompts"], QUERIES))
    )
    print("\nMULTI E2E OK" if ok else "\nMULTI E2E FAILED")


if __name__ == "__main__":
    main()
