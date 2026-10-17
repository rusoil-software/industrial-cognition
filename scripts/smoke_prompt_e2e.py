"""End-to-end: set a prompt over HTTP, then analyse a frame with it.

Run inside an api pod (it has httpx, celery and cv2):
    kubectl -n cogni-ns cp scripts/smoke_prompt_e2e.py <pod>:/tmp/prompt_e2e.py
    kubectl -n cogni-ns exec <pod> -- env PYTHONPATH=/app python /tmp/prompt_e2e.py

This is the chain the System User actually exercises:

    PUT /cameras/{id}/prompt  ->  tokenise with CLIP  ->  Postgres
    (dispatcher embeds it)    ->  RabbitMQ            ->  vision worker -> ONNX

The dispatcher is bypassed here on purpose: capture is not running (no camera), so the
prompt is taken from the API response and injected into the payload exactly as
`FrameDispatcher.dispatch(prompt=...)` would.
"""

import base64
import json
import time

import cv2
import httpx
import numpy as np

from src.tasks.celery import celery_app

BASE = "http://localhost:8000"
CAMERA = "main"


def frame_payload() -> dict:
    frame = np.full((480, 640, 3), 70, dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (330, 300), (215, 210, 205), -1)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    assert ok
    return {
        "camera_id": CAMERA,
        "frame_number": 1,
        "capture_ts_ns": time.time_ns(),
        "width": 640,
        "height": 480,
        "encode_ms": 0.7,
        "image_b64": base64.b64encode(buffer.tobytes()).decode("ascii"),
    }


def main() -> None:
    print("=== 1. set the prompt over HTTP (real CLIP tokenisation) ===")
    response = httpx.put(
        f"{BASE}/cameras/{CAMERA}/prompt", json={"labels": ["metal sheet"]}, timeout=30
    )
    print(f"PUT  -> {response.status_code}")
    if response.status_code != 200:
        print(response.text)
        return
    stored = response.json()
    print(f"     prompt_text   : {stored['prompt_text']!r}")
    print(f"     input_ids     : {stored['input_ids']}")
    print(f"     active_tokens : {stored['active_tokens']}")
    print(f"     tokenizer     : {stored['tokenizer']}")

    print("\n=== 2. read it back ===")
    fetched = httpx.get(f"{BASE}/cameras/{CAMERA}/prompt", timeout=10).json()
    print(f"     matches PUT   : {fetched['input_ids'] == stored['input_ids']}")

    print("\n=== 3. validation still rejects bad prompts ===")
    bad = httpx.put(
        f"{BASE}/cameras/{CAMERA}/prompt", json={"labels": ["metal. sheet"]}, timeout=10
    )
    print(f"     period inside -> {bad.status_code}")

    print("\n=== 4. list ===")
    listing = httpx.get(f"{BASE}/cameras/prompts", timeout=10).json()
    print(f"     configured    : {listing['count']} camera(s)")

    print("\n=== 5. analyse a frame with that prompt (real OWLv2) ===")
    payload = frame_payload()
    # Exactly what the dispatcher puts on the wire for this camera.
    payload["prompt"] = {
        "camera_id": CAMERA,
        "labels": stored["labels"],
        "prompt_text": stored["prompt_text"],
        "input_ids": stored["input_ids"],
        "attention_mask": stored["attention_mask"],
        "tokenizer": stored["tokenizer"],
        "max_length": stored["max_length"],
    }

    started = time.time()
    task = celery_app.send_task("vision.process_frame", args=[payload], queue="vision")
    value = task.get(timeout=300)
    elapsed = time.time() - started

    print(f"     worker_id     : {value['worker_id']}")
    print(f"     error         : {value['error']}")
    print(f"     timings       : {json.dumps(value['timings'])}")
    print(f"     detections    : {len(value['detections'])}")
    print(f"     round trip    : {elapsed:.1f}s")

    print("\n=== 6. clean up ===")
    deleted = httpx.delete(f"{BASE}/cameras/{CAMERA}/prompt", timeout=10)
    print(f"     DELETE -> {deleted.status_code}")

    ok = value["error"] is None and stored["active_tokens"] == 5
    print("\nPROMPT E2E OK" if ok else "\nPROMPT E2E FAILED")


if __name__ == "__main__":
    main()
