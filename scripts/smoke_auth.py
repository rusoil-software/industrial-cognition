"""Verify the console's door in-cluster.

Run in the api pod (it has httpx + websockets + the AUTH_TOKEN env):
    kubectl -n cogni-ns cp scripts/smoke_auth.py <pod>:/tmp/smoke_auth.py
    kubectl -n cogni-ns exec <pod> -- env PYTHONPATH=/app python /tmp/smoke_auth.py

Checks the properties that matter for a door:
  * data endpoints refuse an anonymous request;
  * the shared token buys a session cookie;
  * the cookie opens the data endpoints, the image stream and the WebSocket;
  * a wrong token is refused;
  * the login form itself is public (otherwise no one could ever sign in).
"""

import json
import os
import time

import httpx

BASE = "http://localhost:8000"
TOKEN = os.environ.get("AUTH_TOKEN", "")
checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    checks.append((name, condition, detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


def main() -> None:
    print(f"deployment token configured: {bool(TOKEN)}")

    print("\n=== anonymous access ===")
    anon = httpx.get(f"{BASE}/cameras", timeout=10)
    check("GET /cameras without a session -> 401", anon.status_code == 401, str(anon.status_code))
    anon_prompt = httpx.put(
        f"{BASE}/cameras/main/prompt", json={"prompts": [{"label": "robot", "threshold": 0.3}]}, timeout=10
    )
    check("PUT prompt without a session -> 401", anon_prompt.status_code == 401, str(anon_prompt.status_code))

    print("\n=== the login form is public ===")
    ui = httpx.get(f"{BASE}/ui/", timeout=10)
    check("GET /ui/ -> 200", ui.status_code == 200, f"{len(ui.text)} bytes")
    status = httpx.get(f"{BASE}/auth/status", timeout=10).json()
    check("/auth/status reports required", bool(status.get("required")), json.dumps(status))

    print("\n=== a wrong token is refused ===")
    wrong = httpx.post(f"{BASE}/auth/login", json={"token": "not-the-token"}, timeout=10)
    check("POST /auth/login wrong token -> 401", wrong.status_code == 401, str(wrong.status_code))

    print("\n=== the right token buys a session ===")
    client = httpx.Client(base_url=BASE, timeout=30)
    login = client.post("/auth/login", json={"token": TOKEN})
    check("POST /auth/login -> 200", login.status_code == 200, login.text[:80])
    check("session cookie set", "ic_session" in client.cookies, ",".join(client.cookies.keys()))
    cookie = client.cookies.get("ic_session", "")
    check("cookie is a JWT (3 segments)", cookie.count(".") == 2, cookie[:24] + "...")

    print("\n=== the session opens the data ===")
    cameras = client.get("/cameras")
    check("GET /cameras with session -> 200", cameras.status_code == 200, cameras.text[:80])

    snapshot = None
    for _ in range(20):
        response = client.get("/cameras/main/snapshot")
        if response.status_code == 200:
            snapshot = response
            break
        time.sleep(0.5)
    check(
        "GET snapshot with session -> 200 jpeg",
        snapshot is not None and snapshot.content[:2] == b"\xff\xd8",
        f"{snapshot.status_code} {len(snapshot.content) if snapshot else 0} bytes" if snapshot else "no frame",
    )

    prompt = client.put(
        "/cameras/main/prompt",
        json={"prompts": [{"label": "robot", "threshold": 0.25}, {"label": "person", "threshold": 0.1}]},
    )
    check("PUT prompt with session -> 200", prompt.status_code == 200, prompt.text[:80])

    print("\n=== the socket is protected the same way ===")
    from websockets.sync.client import connect

    closed = None
    try:
        with connect("ws://localhost:8000/ws/detections", open_timeout=10) as socket:
            socket.recv(timeout=5)
        check("anonymous WebSocket refused", False, "connected without a session")
    except Exception as exc:  # noqa: BLE001 - the close is the expected outcome
        closed = str(exc)
        check("anonymous WebSocket refused", True, closed[:60])

    try:
        with connect(
            "ws://localhost:8000/ws/detections",
            additional_headers={"Cookie": f"ic_session={cookie}"},
            open_timeout=10,
        ) as socket:
            greeting = json.loads(socket.recv(timeout=15))
        check("authenticated WebSocket greeted", greeting.get("type") in ("hello", "unavailable"), str(greeting.get("type")))
    except Exception as exc:  # noqa: BLE001
        check("authenticated WebSocket greeted", False, str(exc)[:80])

    failed = [name for name, ok, _ in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    print("AUTH OK" if not failed else f"AUTH FAILED: {failed}")


if __name__ == "__main__":
    main()
