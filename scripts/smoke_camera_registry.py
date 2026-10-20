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
# date   : 2026-Oct-20
# ==============================================================================
"""Manual check of the camera registry against a running deployment.

Not automated coverage (see doc/tasklist.md item 10): it needs a live API and a
camera - here a replay image standing in for one - so it verifies that the registry,
activation, capture and the preview all agree, which unit tests cannot.

    kubectl -n cogni-ns port-forward svc/api 18000:80 &
    python scripts/smoke_camera_registry.py
"""

from __future__ import annotations

import http.cookiejar
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("API_BASE", "http://127.0.0.1:18000")
TOKEN = os.environ.get("AUTH_TOKEN", "cogni-dev-console-token")
CAMERA = os.environ.get("SMOKE_CAMERA_ID", "replay-cam")
SOURCE = os.environ.get(
    "SMOKE_CAMERA_SOURCE",
    "replay:/app/assets/images/roboflow_test_detect_human_robot.webp",
)

opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))


def call(method: str, path: str, payload: dict | None = None, raw: bool = False):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(BASE + path, data=data, method=method)
    if data is not None:
        request.add_header("content-type", "application/json")
    try:
        with opener.open(request, timeout=180) as response:
            body = response.read()
            return response.status, (body if raw else json.loads(body or b"{}"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def main() -> int:
    failures = 0

    def check(name: str, ok: bool, detail: object = "") -> None:
        nonlocal failures
        print(f"{'ok  ' if ok else 'FAIL'} {name}{'' if ok else f': {detail}'}")
        if not ok:
            failures += 1

    status, health = call("GET", "/health")
    check("health", status == 200, health)

    status, login = call("POST", "/auth/login", {"token": TOKEN})
    check("login", status == 200, login)

    status, discovery = call("GET", "/cameras/discover?include_network=false")
    check(
        "discover answers with devices and network lists",
        status == 200 and "devices" in discovery and "network" in discovery,
        discovery,
    )
    check(
        "the server can store camera passwords",
        bool(discovery.get("credentials_configured")),
        "CAMERA_CRED_KEY is not configured",
    )

    status, probe = call(
        "POST", "/cameras/probe", {"source": SOURCE}
    )
    check("probe reads a frame from the replay source", probe.get("ok") is True, probe)

    status, camera = call(
        "POST",
        "/cameras",
        {"id": CAMERA, "label": "Replay test", "source": SOURCE, "activate": True},
    )
    check("register + activate", status in (200, 201), camera)
    check("the camera is active", camera.get("active") is True, camera)
    check("no password leaks into the response", "password" not in json.dumps(camera), camera)

    status, listing = call("GET", "/cameras")
    entries = {entry["camera_id"]: entry for entry in listing.get("cameras", [])}
    check("the registry lists it", CAMERA in entries, listing)
    check("it is active in the registry", entries.get(CAMERA, {}).get("active") is True, entries.get(CAMERA))
    check("capture runs", entries.get(CAMERA, {}).get("running") is True, entries.get(CAMERA))

    status, pipeline = call("GET", "/status")
    check("status reports capture running", pipeline.get("running") is True, pipeline.get("reason"))
    check(
        "status reports the registry watcher",
        pipeline.get("registry", {}).get("running") is True,
        pipeline.get("registry"),
    )
    check(
        "capture follows the registered camera",
        pipeline.get("pipeline", {}).get("camera_id") == CAMERA,
        pipeline.get("pipeline", {}).get("camera_id"),
    )

    status, snapshot = call("GET", f"/cameras/{CAMERA}/snapshot", raw=True)
    check(
        "snapshot returns a JPEG",
        status == 200 and isinstance(snapshot, bytes) and snapshot[:2] == b"\xff\xd8",
        status,
    )

    print(f"\n{'PASS' if not failures else f'{failures} check(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
