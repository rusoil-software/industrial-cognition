"""In-cluster smoke test: health, prompt storage against real Postgres, status.

Run inside an api pod (it has httpx, sqlalchemy and the app):
    kubectl -n cogni-ns cp scripts/smoke_cluster.py <pod>:/tmp/smoke.py
    kubectl -n cogni-ns exec <pod> -- python /tmp/smoke.py
"""

import json

import httpx

BASE = "http://localhost:8000"


class FakeTokenizer:
    """Stands in for CLIP so the DB path can be exercised without `transformers`.

    The API image deliberately does not ship the tokenizer, so the HTTP endpoint
    answers 503; this class lets this smoke test still prove encode -> repository ->
    Postgres -> read back, which is the part that has no other coverage in-cluster.
    """

    name = "smoke:fake-clip"
    max_length = 16

    def encode(self, text: str):
        ids = [(ord(ch) % 49407) or 1 for ch in text[:15]]
        padding = 16 - len(ids)
        return tuple(ids + [49407] * padding), tuple([1] * len(ids) + [0] * padding)


def main() -> None:
    print("=== HTTP surface ===")
    health = httpx.get(f"{BASE}/health", timeout=10)
    body = health.json()
    print(f"GET /health            -> {health.status_code} pipeline_running={body['pipeline_running']}")
    print(f"   reason: {body['pipeline'].get('reason')}")

    status = httpx.get(f"{BASE}/status", timeout=10)
    print(f"GET /status            -> {status.status_code} running={status.json().get('running')}")

    print("\n=== prompt endpoint (no tokenizer in this image: expect 503) ===")
    put = httpx.put(f"{BASE}/cameras/main/prompt", json={"labels": ["metal sheet"]}, timeout=10)
    print(f"PUT /cameras/main/prompt -> {put.status_code} {put.text[:120]}")

    print("\n=== validation (expect 422) ===")
    bad = httpx.put(f"{BASE}/cameras/main/prompt", json={"labels": ["metal. sheet"]}, timeout=10)
    print(f"PUT with a period inside -> {bad.status_code} {bad.text[:120]}")

    empty = httpx.put(f"{BASE}/cameras/main/prompt", json={"labels": ["   "]}, timeout=10)
    print(f"PUT with a blank label   -> {empty.status_code} {empty.text[:120]}")

    print("\n=== repository against real Postgres (fake tokenizer) ===")
    from src.api.prompts import service
    from src.api.storage.database import make_session_factory

    session = make_session_factory()()
    try:
        prompt = service.set_camera_prompt(session, "main", ["metal sheet", "debris"], FakeTokenizer())
        session.commit()
        print(f"stored   camera_id={prompt.camera_id} labels={prompt.labels} tokens={sum(prompt.attention_mask)}")

        loaded = service.get_camera_prompt(session, "main")
        print(f"loaded   camera_id={loaded.camera_id} prompt_text={loaded.prompt_text!r}")
        print(f"ids      {list(loaded.input_ids)[:6]}...")
        assert loaded.labels == ("metal sheet", "debris"), loaded.labels
        assert loaded.to_query().labels == prompt.labels

        listed = service.list_camera_prompts(session)
        print(f"listed   {len(listed)} prompt(s)")

        # Upsert must not create a second row.
        service.set_camera_prompt(session, "main", ["detail"], FakeTokenizer())
        session.commit()
        again = service.get_camera_prompt(session, "main")
        print(f"upsert   labels={again.labels} total={len(service.list_camera_prompts(session))}")
        assert again.labels == ("detail",)

        removed = service.delete_camera_prompt(session, "main")
        session.commit()
        print(f"deleted  {removed} remaining={len(service.list_camera_prompts(session))}")
    finally:
        session.close()

    print("\n=== WebSocket (pipeline not running: expect an unavailable frame) ===")
    try:
        from websockets.sync.client import connect

        with connect("ws://localhost:8000/ws/detections", open_timeout=10) as socket:
            print(f"first message: {socket.recv(timeout=5)}")
    except Exception as exc:  # noqa: BLE001 - the close is the expected outcome
        print(f"ws closed: {type(exc).__name__}: {str(exc)[:120]}")

    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
