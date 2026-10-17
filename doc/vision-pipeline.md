# Vision Pipeline: Multi-Worker Frame Analysis

This document specifies how a camera stream becomes detection results, how the
work is spread across several Celery workers, and how the **60 FPS aggregate
throughput** requirement (`doc/vision.md`) is measured and met.

It is the companion to `doc/vision.md` (the design intent) and `doc/k8s.md` (the
deployment). Everything below is based on measurements taken against the artifact
in this repository, not on published model figures.

---

## 1. Why the pipeline needs workers at all

A single OWLv2 forward pass is far slower than a 60 FPS frame interval. Measured
with `scripts/inspect_model.py` on the shipped 293 MiB artifact
(`src/owl/models/cuda/owl2_model.onnx`):

| Runtime | Per frame | Per-worker throughput | Workers for 60 FPS |
| --- | --- | --- | --- |
| CPU-only `onnxruntime` (this dev machine) | **4053 ms** | 0.25 FPS | **244** |
| GPU (`onnxruntime-gpu`, RTX 4060 8 GB) | ~20-40 ms (expected) | 25-50 FPS | **2-3** |

So the requirement cannot be met by making one process faster; it is met by
running enough workers in parallel. That makes the *orchestration* the deliverable,
and the model's latency a capacity input:

```
workers_needed = ceil(60 / (1000 / per_frame_latency_ms))
```

The dev machine has an RTX 4060 but only the CPU `onnxruntime` wheel is installed,
so the CPU row is what could be measured here. The GPU row is why the vision image
installs `onnxruntime-gpu` (`requirements/gpu.txt`) and why the prod overlay
requests `nvidia.com/gpu`.

---

## 2. Module map

| Module | Responsibility |
| --- | --- |
| `src/camera/capture.py` | Open device/RTSP, negotiate MJPG + 60 FPS, read frames, JPEG-encode to the model's resolution |
| `src/camera/frames.py` | `CapturedFrame`: the wire format between capture and worker |
| `src/vision/preprocess.py` | Compressed frame → `pixel_values (1,3,960,960)` float32, CLIP-normalised |
| `src/vision/backend.py` | `InferenceBackend` seam: `Owl2OnnxBackend`, `SyntheticBackend`, text-query validation |
| `src/vision/decode.py` | Graph outputs → normalised `Detection`s (centre→corner, sigmoid, NMS) |
| `src/vision/worker.py` | The per-frame job: decode → preprocess → infer → decode, with timings |
| `src/vision/detections.py` | `Detection`, `FrameResult`, `StageTimings` value objects |
| `src/pipeline/dispatcher.py` | Bounded in-flight window, load shedding, result collection |
| `src/pipeline/stats.py` | Capture/dispatch/completion FPS, latency percentiles, drop counters |
| `src/tasks/celery.py` | The `vision.process_frame` task; queues and broker configuration |

`doc/conventions.md` §1 (domain isolation) is what puts decoding in `camera`, the
model in `vision`, and only the *flow* in `pipeline`.

---

## 3. Frame lifecycle and the wire format

```
camera driver ──read()──► BGR frame ──resize/JPEG──► CapturedFrame (bytes)
        │                                                   │
        │                                                   ▼
        │                                    Celery task payload (base64 JPEG)
        │                                                   │
        ▼                                                   ▼
  drop if window full  ◄── FrameDispatcher ──► worker: decode → preprocess
                                                            → ONNX → decode
                                                            → FrameResult (wire)
        ▲                                                   │
        └──────────── collect() ◄───────────────────────────┘
```

**Why JPEG on the queue.** At 60 FPS a 1080p frame is 6.2 MB raw (~166 MB/s for
float32 RGB); no broker carries that per frame. A 960×960 JPEG at q80 is ~100 KB,
about **6 MB/s at 60 FPS**, which RabbitMQ and Celery's JSON serialiser handle
comfortably. The alternatives (shared memory, one object per frame in MinIO) either
couple capture to a single host or add a network round-trip per frame; both defeat
the horizontal scaling this stage exists to provide.

The worker therefore pays a decode cost (~2-4 ms) but stays **stateless**, which
is what allows N replicas.

---

## 4. Load shedding is the safety property

The dispatcher keeps at most `max_in_flight` tasks outstanding and **drops** new
frames beyond that. This is the difference between a slow system and a dangerous
one:

* If capture runs at 60 FPS and analysis capacity is 40 FPS, a naive dispatcher
  queues everything. Throughput stays at 40 FPS while **latency grows without
  bound** - detections become seconds stale, which is unacceptable when they drive
  a robot arm. Every metric except latency p99 looks healthy.
* With a bounded window, the queue can never exceed `max_in_flight` frames, so
  worst-case latency is bounded by
  `(max_in_flight / workers + 1) × per_frame_latency`, and the shortfall shows up
  explicitly as `dropped_capacity`.

Dropping is the correct behaviour for a real-time control loop: a stale detection
is worth nothing. `tests/pipeline/test_fps_target.py` asserts both directions -
60 FPS sustained with adequate workers, and bounded latency plus non-zero shedding
when workers are scarce.

---

## 5. Sizing the pool

Set `max_in_flight` to roughly **2 × workers × concurrency**, and size the worker
count from the measured per-frame latency:

```sh
# what does one forward pass cost on this box?
python scripts/inspect_model.py src/owl/models/cuda/owl2_model.onnx
# -> "workers for 60 FPS: N"
```

Then set, in `k8s/base/deployment-vision.yaml`, `replicas` to at least `N`, or
enable the KEDA queue-depth scaler (`k8s/components/keda`) so the pool follows the
`vision` queue depth rather than a fixed count.

Two honest notes:

* On CPU the target is unreachable (244 workers). The 60 FPS claim is a **GPU
  claim**; without `CUDAExecutionProvider` the service degrades to a low-rate
  detection feed and says so through `completion_fps`.
* Batching (`--concurrency` pipelines, `OWL2_MAX_BATCH_SIZE`) raises per-worker
  throughput on a GPU, because the fixed image-encoder cost dominates. `decode.py`
  already supports batched outputs.

---

## 6. The text query is deployment data

This is the least obvious part of the pipeline and the easiest to get wrong.

* The artifact was exported with **random** dummy token IDs
  (`src/owl/inference/utils.py::_create_dummy_inputs` uses `torch.randint`), so the
  IDs that select "metal sheet" cannot be recovered from the model.
* Feeding all-zero IDs (padding) saturates the graph's fp16 text path at
  **-65504**; `sigmoid(-65504) == 0`, so every query is below any threshold and the
  pipeline reports *no objects, forever*. The image path is unaffected, which makes
  the failure look like "nothing in view" rather than "misconfigured".
* The graph emits **one class** (`logits` is `(1, 3600, 1)`), so the text query is
  effectively fixed per export and `Detection.object_type` comes from
  `VISION_LABELS[0]`.

Therefore:

```sh
# one-off, needs `transformers` (export-time dependency)
python scripts/encode_text_queries.py \
  --labels "metal sheet,detail,debris" \
  --output src/owl/models/text_queries.json
```

and point `OWL2_TEXT_IDS_PATH` at the result. `TextQuery` validates the IDs and
raises on the all-padding case; `Owl2OnnxBackend` additionally raises if the graph
returns saturated logits.

---

## 7. Preprocessing contract

Read from the artifact by `scripts/inspect_model.py`, and asserted by
`tests/vision/test_preprocess.py`:

| Input | Shape | dtype |
| --- | --- | --- |
| `pixel_values` | `(batch, 3, 960, 960)` | float32 |
| `input_ids` | `(batch, 16)` | int64 |
| `attention_mask` | `(batch, 16)` | int64 |

Outputs: `logits (batch, 3600, C)` float32, `pred_boxes (batch, 3600, 4)` **float16**.

Normalisation is `(pixel/255 - CLIP_mean) / CLIP_std` on RGB, with the CLIP
statistics; boxes are centre-based `(cx, cy, w, h)` and are converted to
top-left `(x1, y1, x2, y2)` in `decode.py`. The reference HuggingFace processor is
compared against ours in `test_matches_reference_processor` whenever
`transformers` + `torch` are installed.

---

## 8. Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `VISION_BACKEND` | `owl2` | `owl2` (real) or `synthetic` (tests/load only) |
| `OWL2_MODEL_PATH` | `src/owl/models/cuda/owl2_model.onnx` | ONNX artifact |
| `OWL2_EXECUTION_PROVIDER` | auto (CUDA if available) | `CUDAExecutionProvider` for the 60 FPS budget |
| `OWL2_TEXT_IDS_PATH` | `src/owl/models/text_queries.json` | tokenised prompt |
| `VISION_LABELS` | `metal_sheet` | class names; `labels[0]` for a single-class export |
| `VISION_SCORE_THRESHOLD` | `0.25` | OWLv2 zero-shot scores are low; 0.75 would detect nothing |
| `VISION_IOU_THRESHOLD` | `0.5` | NMS; `none` disables it for raw inspection |
| `VISION_MAX_DETECTIONS` | `32` | cap per frame |
| `VISION_SYNTHETIC_LATENCY_MS` | `0` | emulate per-frame cost in tests |

---

## 9. What is verified, and what is not

Verified here:

* `tests/vision` (72 tests) - preprocessing maths and channel order, decode
  conventions (centre→corner, sigmoid stability, NMS, clipping, fp16), text-query
  validation, worker round trip and failure handling.
* `tests/pipeline` (28 tests) - bounded window, shedding, latency accounting, and
  the 60 FPS / bounded-latency acceptance tests.
* `scripts/inspect_model.py` against the real 293 MiB artifact.

Not verified here, and why:

* **Real detections.** No correct token IDs exist for the prompt (see §6), and the
  CPU-only runtime cannot run at video rate. The pipeline is proven; detection
  quality is a GPU + tokenised-prompt exercise.
* **Real camera capture.** No industrial camera is attached to this machine.
  `SyntheticFrameSource` and the `DeviceFrameSource` code path are exercised, but
  driver negotiation (MJPG, 60 FPS) needs the real device.
* **Celery transport.** `src/pipeline/celery_transport.py` and the Celery task are
  code-complete, but `celery` is not installed locally, so the broker round trip
  was not executed. The dispatcher is transport-agnostic and fully tested with an
  in-process and a thread-pool transport.

---

## 10. Next steps for Stage 2

1. Wire `FrameDispatcher.collect()` into the FastAPI WebSocket (`/ws/detections`)
   so results reach clients; the payload already matches
   `src/camera/schemas.py::StreamingPayload`.
2. Add the Celery transport adapter and run the pipeline against a real broker.
3. Tokenise the real prompt and validate detection quality on a captured frame
   from the actual camera.
4. Size the worker pool from measured GPU latency; enable the KEDA scaler.

---

## 11. Setting the prompt from the API

The System User sets what one camera looks for; the service tokenises it and stores
it against that camera. A later web interface is expected to call exactly these
endpoints.

| Method | Path | Purpose |
| --- | --- | --- |
| `PUT` | `/cameras/{camera_id}/prompt` | Tokenise `{"labels": ["metal sheet"]}` and store it for that camera |
| `GET` | `/cameras/{camera_id}/prompt` | Read back the prompt **and its token IDs** |
| `DELETE` | `/cameras/{camera_id}/prompt` | Clear it |
| `GET` | `/cameras/prompts` | List every configured camera |

Status codes are deliberate: **422** when the prompt cannot tokenise into usable
conditioning (blank name, an embedded period, or padding-only tokens), **503** when
this deployment has no tokenizer configured, **404** when no prompt is set.

```sh
curl -X PUT http://localhost:8000/cameras/main/prompt \
  -H 'content-type: application/json' \
  -d '{"labels": ["metal sheet"]}'
```

```json
{"camera_id": "main", "labels": ["metal sheet"], "prompt_text": "metal sheet.",
 "input_ids": [49406, 3208, 2368, 49407, ...], "attention_mask": [1, 1, 1, 1, 0, ...],
 "tokenizer": "transformers:google/owlv2-base-patch16-ensemble",
 "max_length": 16, "active_tokens": 4}
```

Returning the tokens is not decoration: it is how an operator confirms that
"metal sheet" did not tokenise into something unexpected, which is the difference
between a configured system and a confidently misconfigured one.

**Storage.** One row in `system_metadata`, key `camera_prompt:<camera_id>`, the
human-readable prompt in `value` and the full encoding in `value_json`. The table
exists for system-level state that grows with the product, so no migration was
needed; if prompts later gain history or relationships, a dedicated table plus an
Alembic revision is the next step.

**How it reaches the model.** The dispatcher embeds the encoding in the frame
payload (`payload["prompt"]`), so every task carries the prompt for *its* camera and
workers stay stateless - no shared cache to invalidate across pods when a user
changes a prompt mid-stream. `Owl2OnnxBackend.infer(..., text_query=...)` then
applies it for that call instead of the process-wide default.

**Prompt lifecycle.** A prompt applies to frames captured *after* it is stored;
in-flight frames keep the prompt they were dispatched with. That is intentional:
retroactively re-prompting frames already in the queue would make results depend on
when a human clicked save.

**Tokenisation is offline-first.** `PROMPT_TOKENIZER_MODEL` selects the checkpoint
and `PROMPT_TOKENIZER_LOCAL_ONLY=1` (the default) forbids a runtime download, which
an air-gapped industrial network requires. The tokenizer files must therefore be
baked into the API image or mounted beside the model.
