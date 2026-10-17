"""Tokenise a set of prompts through the running API and write them as JSON files.

Run inside an api pod:
    kubectl -n cogni-ns cp scripts/dump_prompts.py <pod>:/tmp/dump_prompts.py
    kubectl -n cogni-ns exec <pod> -- env PYTHONPATH=/app python /tmp/dump_prompts.py

Using the HTTP endpoint rather than importing the tokenizer is deliberate: it exercises
the path the System User actually drives (PUT -> CLIP tokenisation -> Postgres), so a
tokenisation bug cannot hide behind a direct import. The files are what the vision-side
evaluation script needs, because that image has ONNX Runtime but no `transformers`.
"""

import json
import pathlib

import httpx

BASE = "http://localhost:8000"
CAMERA = "eval"
LABELS = ["robot", "human", "person", "metal sheet", "cardboard box"]
OUT = pathlib.Path("/tmp/prompts")

KEEP = ("labels", "prompt_text", "input_ids", "attention_mask", "tokenizer", "max_length")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for label in LABELS:
        response = httpx.put(
            f"{BASE}/cameras/{CAMERA}/prompt", json={"labels": [label]}, timeout=60
        )
        if response.status_code != 200:
            print(f"{label:<16} FAILED {response.status_code} {response.text[:120]}")
            continue
        payload = response.json()
        path = OUT / f"{label.replace(' ', '_')}.json"
        path.write_text(
            json.dumps({key: payload[key] for key in KEEP}, indent=2) + "\n",
            encoding="utf-8",
        )
        ids = payload["input_ids"]
        print(
            f"{label:<16} active={payload['active_tokens']:2d} ids={ids[:6]} -> {path.name}"
        )

    httpx.delete(f"{BASE}/cameras/{CAMERA}/prompt", timeout=10)
    print(f"\nwrote {len(list(OUT.glob('*.json')))} prompt file(s) to {OUT}")


if __name__ == "__main__":
    main()
