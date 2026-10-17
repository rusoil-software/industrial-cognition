"""Verify the operator console's backend surface inside the cluster.

Run in an api pod (needs httpx + websockets, both present):
    kubectl -n cogni-ns cp scripts/smoke_ui.py <pod>:/tmp/smoke_ui.py
    kubectl -n cogni-ns exec <pod> -- env PYTHONPATH=/app python /tmp/smoke_ui.py

Checks the four things the Vue console depends on:

1. ``/ui/`` serves the console (and its vendored Vue runtime),
2. ``/cameras`` reports the camera and whether it is capturing,
3. ``/cameras/{id}/snapshot`` returns the JPEG the last inference saw,
4. ``/cameras/{id}/stream`` speaks MJPEG, and ``/ws/detections`` delivers results.
"""

import json
import time

import httpx

BASE = "http://localhost:8000"
CAMERA = "main"
QUERIES = [
    {"label": "robot", "threshold": 0.25},
    {"label": "person", "threshold": 0.10},
    {"label": "cardboard box", "threshold": 0.4},
]


def main() -> None:
    print("=== 1. console assets ===")
    index = httpx.get(f"{BASE}/ui/", timeout=10)
    print(f"GET /ui/            -> {index.status_code} {len(index.text)} bytes")
    has_vue = "./vendor/vue.global.prod.js" in index.text
    vue = httpx.get(f"{BASE}/ui/vendor/vue.global.prod.js", timeout=10)
    print(f"     vendored vue   -> {vue.status_code} {len(vue.content)} bytes")
    app_js = httpx.get(f"{BASE}/ui/app.js", timeout=10)
    print(f"     app.js         -> {app_js.status_code} {len(app_js.content)} bytes")

    print("\n=== 2. cameras ===")
    cameras = httpx.get(f"{BASE}/cameras", timeout=10).json()
    for camera in cameras["cameras"]:
        print(
            f"     {camera['camera_id']:<10} running={camera['running']} "
            f"capture_fps={camera['capture_fps']:.1f} prompts={len(camera['prompts'])}"
        )

    print("\n=== 3. configure the cognition task ===")
    put = httpx.put(f"{BASE}/cameras/{CAMERA}/prompt", json={"prompts": QUERIES}, timeout=60)
    print(f"PUT prompt         -> {put.status_code}")
    for query in put.json().get("prompts", []):
        print(f"     {query['label']:<14} threshold={query['threshold']:.2f}")

    print("\n=== 4. snapshot ===")
    # Capture is running at 2 FPS, so wait for a frame rather than racing it.
    shot = None
    for _ in range(30):
        response = httpx.get(f"{BASE}/cameras/{CAMERA}/snapshot", timeout=10)
        if response.status_code == 200:
            shot = response
            break
        time.sleep(0.5)
    if shot is None:
        print(f"     snapshot       -> {response.status_code} {response.text[:100]}")
    else:
        print(
            f"     snapshot       -> {shot.status_code} {len(shot.content)} bytes "
            f"{shot.headers.get('content-type')} frame={shot.headers.get('x-frame-number')}"
        )
        print(f"     jpeg magic     : {shot.content[:2].hex()} (ffd8 expected)")

    print("\n=== 5. MJPEG stream ===")
    chunks = 0
    with httpx.stream("GET", f"{BASE}/cameras/{CAMERA}/stream", timeout=20) as stream:
        print(f"     content-type   : {stream.headers.get('content-type')}")
        for chunk in stream.iter_raw():
            if chunk:
                chunks += len(chunk)
            if chunks > 40000:
                break
    print(f"     received       : {chunks} bytes of multipart")

    print("\n=== 6. detection stream ===")
    from websockets.sync.client import connect

    observed = None
    try:
        with connect("ws://localhost:8000/ws/detections", open_timeout=10) as socket:
            deadline = time.time() + 150
            while time.time() < deadline:
                message = json.loads(socket.recv(timeout=155))
                if message.get("type") in ("hello", "unavailable"):
                    print(f"     greeted        : {message.get('type')}")
                    continue
                observed = message
                if message.get("detections"):
                    break
    except Exception as exc:  # noqa: BLE001 - report, do not fail the script
        print(f"     websocket error: {type(exc).__name__}: {str(exc)[:120]}")

    if observed:
        detections = observed.get("detections", [])
        counts: dict[str, int] = {}
        for detection in detections:
            counts[detection["object_type"]] = counts.get(detection["object_type"], 0) + 1
        print(f"     frame          : {observed['metadata']['frame_number']}")
        print(f"     detections     : {len(detections)} {counts}")
        print(f"     timings        : {json.dumps(observed.get('timings'))}")

    print("\n=== 7. clean up ===")
    print(f"     DELETE -> {httpx.delete(f'{BASE}/cameras/{CAMERA}/prompt', timeout=10).status_code}")

    ok = index.status_code == 200 and has_vue and shot is not None and observed is not None
    print("\nUI BACKEND OK" if ok else "\nUI BACKEND INCOMPLETE (see above)")


if __name__ == "__main__":
    main()
