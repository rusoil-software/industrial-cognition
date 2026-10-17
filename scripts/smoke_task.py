"""End-to-end task smoke test: API-side producer -> RabbitMQ -> vision worker.

Run inside a pod that has celery + cv2 + numpy (the api image works):
    kubectl -n cogni-ns cp scripts/smoke_task.py <pod>:/tmp/smoke_task.py
    kubectl -n cogni-ns exec <pod> -- env PYTHONPATH=/app python /tmp/smoke_task.py
"""

import base64
import json
import time

import cv2
import numpy as np

from src.tasks.celery import celery_app


def build_payload(frame_number: int = 1) -> dict:
    frame = np.full((480, 640, 3), 70, dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (330, 300), (215, 210, 205), -1)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    assert ok
    return {
        "camera_id": "main",
        "frame_number": frame_number,
        "capture_ts_ns": time.time_ns(),
        "width": 640,
        "height": 480,
        "encode_ms": 0.7,
        "image_b64": base64.b64encode(buffer.tobytes()).decode("ascii"),
    }


def main() -> None:
    payload = build_payload()
    print(f"payload bytes: {len(payload['image_b64'])} (base64)")
    started = time.time()

    result = celery_app.send_task(
        "vision.process_frame", args=[payload], queue="vision"
    )
    print(f"task id: {result.id}")

    value = result.get(timeout=120)
    elapsed = time.time() - started

    summary = {key: item for key, item in value.items() if key != "detections"}
    print(json.dumps(summary, indent=2))
    print(f"detections: {len(value['detections'])}")
    if value["detections"]:
        print(f"first: {json.dumps(value['detections'][0])}")
    print(f"round trip: {elapsed:.2f}s")
    print("TASK SMOKE OK")


if __name__ == "__main__":
    main()
