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

---

## 12. Real-time stream and startup policy

`WS /ws/detections` streams results as they complete. Each message is the
`src/camera/schemas.py::StreamingPayload` envelope:

```json
{"metadata": {"camera_id": "cam0", "frame_number": 412, "timestamp": "..."},
 "detections": [{"object_type": "metal_sheet", "x_norm": 0.31, "y_norm": 0.28,
                 "width_norm": 0.22, "height_norm": 0.19, "confidence": 0.83}],
 "timings": {"preprocess_ms": 3.1, "inference_ms": 24.6, "decode_ms": 0.4,
             "queue_ms": 2.2, "e2e_latency_ms": 31.8},
 "worker_id": "vision-1:7", "error": null}
```

The first frame is a `hello` describing the pipeline, so a client can tell
"connected, nothing detected yet" from "connected, capture is not running".

**A slow client never slows detection.** Each WebSocket subscription owns a
bounded buffer (120 messages); when it overflows the *oldest* result is dropped and
counted, and the count is logged on disconnect. Slowing the pipeline to match the
slowest browser tab would make the robot's view depend on a dashboard. This mirrors
the dispatcher's own load-shedding policy.

**Startup never fails on missing hardware.** `src/api/runtime.py` starts the runner
best-effort: a missing camera, model or tokenizer leaves the API serving `/health`,
`/status` and the prompt endpoints, with `pipeline_running: false` and a `reason`
explaining why. This is a deliberate reversal of the previous bootstrap, which
raised from the startup handler and made the API pod crash-loop in the cluster
because it could not serve `/health` without Modbus, a camera and a model.

Operational knobs: `CAMERA_ENABLED`, `CAMERA_ID`, `CAMERA_SOURCE` (device index,
V4L2 path or RTSP URL), `CAMERA_FPS`, `PIPELINE_MAX_IN_FLIGHT`.

---

## 13. Measured in a live cluster (CPU)

The real artifact was loaded from the `model-cache` PVC into the `vision` worker on
minikube and exercised with `scripts/smoke_inference.py`. No GPU is available, so
these are CPU numbers - they are the *worst case* and the reason the 60 FPS target is
a GPU claim (§1).

| Quantity | Value | Notes |
| --- | --- | --- |
| Model load (`warmup`) | **11.4-11.8 s** | From the PVC, `CPUExecutionProvider` |
| Inference, 2 CPU limit | **26.8 s/frame** | Through Celery, as deployed |
| Inference, 6 CPU limit | **7.5-7.8 s/frame** | Probe pod; inference is CPU-bound and scales with cores |
| Peak RSS, one session | **3432 MiB** | `VmHWM`, the 293 MiB artifact |
| Logits | min -41.8, max -4.8, mean -17.5 | Finite, **not** the -65504 saturation case |
| Best sigmoid score | **0.0084** | Below every usable threshold |

### What this proves

* The ONNX path works end to end in-cluster: artifact from a PVC, CPU session,
  `pixel_values` in, `logits`/`pred_boxes` out, decode and NMS producing boxes.
* The graph is **not** in the degenerate text state. Logits are finite and
  well-distributed, so the earlier -65504 padding failure is not what is happening.
* The geometry path is correct. With a bright rectangle drawn from
  `(0.364, 0.312)` with size `(0.312, 0.312)` in normalised coordinates, the
  highest-scoring decoded box was `(0.363, 0.310)` size `(0.317, 0.320)` - the same
  rectangle. Box decoding and the centre-to-corner conversion are right.

### What it does not prove

Detection quality. The tokenised prompt on that PVC is a **placeholder**
(`k8s/dev/text_queries.placeholder.json`), not a CLIP encoding of "metal sheet", so
every score is near zero and nothing clears 0.25. That is the expected outcome of a
meaningless prompt, not a defect in the pipeline. Producing real conditioning needs
`scripts/encode_text_queries.py` with the CLIP tokenizer (§6).

### Capacity findings

* **Memory:** one resident session peaks at 3432 MiB. The dev overlay's 4 GiB limit
  was exceeded and the container was OOMKilled (exit 137) as soon as a second
  session was loaded in the same pod. Raised to 6 GiB. This also constrains
  concurrency: Celery prefork children each load their **own** session, so
  `--concurrency=1` is not a tuning preference but a memory requirement - N children
  cost roughly N x 3.4 GiB.
* **Latency:** ~7.5 s/frame at 6 cores. Reaching 60 FPS on CPU would need ~450
  concurrent worker cores, which confirms §1's conclusion that this is GPU-only work.
* **Warmup matters:** a ~11.5 s first-inference penalty per process means a cold
  autoscale-up produces no detections for the first ~11 s. `VisionWorker.warmup()`
  exists for this; the deployment should call it at startup rather than on the first
  request.

### Real prompts and tightened runtime (same day, second pass)

Two changes, both measured in the cluster.

**1. Real CLIP tokenisation.** `scripts/encode_text_queries.py` was run against
`google/owlv2-base-patch16-ensemble` and produced the genuine encoding of the default
prompt:

```
metal sheet.  ->  [49406, 4044, 7298, 269, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                 (5 active tokens; CLIP pads with 0, not with EOS)
```

That differs completely from the earlier placeholder, which is why placeholder scores
were meaningless. The tokenizer is now **baked into the API image at build time** with
`HF_HOME=/app/.hf`, so `PROMPT_TOKENIZER_LOCAL_ONLY=1` works on an air-gapped network:
the prompt API tokenises without any runtime download. Verified offline inside the
image: the same five IDs come back with `local_files_only=True`.

The whole chain now runs live: `PUT /cameras/main/prompt` returned **200** with those
IDs, `GET` returned the same encoding, a prompt containing a period still returned
**422**, and `DELETE` returned **204**.

**2. Thread sizing was the dominant runtime cost.** ONNX Runtime sizes its intra-op
pool from the *node's* physical cores, and this node reports 20 - inside a pod limited
to 2. The pool spent its time in scheduler contention. `effective_cpu_count()` now
reads the cgroup quota (v2 `cpu.max`, v1 `cpu.cfs_quota_us`/`cpu.cfs_period_us`) and
falls back to the affinity mask, with `OWL2_INTRA_OP_THREADS` as an override. Sessions
also get sequential execution, one inter-op thread, `ORT_ENABLE_ALL` stated explicitly,
and a persistent allocation arena; CUDA gets `kSameAsRequested` arena growth and the
heuristic cuDNN search.

| Configuration | Before | After | Change |
| --- | --- | --- | --- |
| 2 CPU worker (as deployed) | 26.8 s/frame | **10.1 s/frame** | 2.6x faster |
| 6 CPU probe | 7.5-7.8 s/frame | **4.6-5.0 s/frame** | 1.6x faster |
| Model load | 11.4-11.8 s | 7.5 s (12.2 s incl. warmup pass) | - |

`--concurrency=1` remains a memory requirement, not a preference: one session peaks at
~3.4 GiB, and each prefork child loads its own.

**3. Warmup moved to worker startup.** `worker_process_init` now loads the model in
each child as it starts (`VISION_WARMUP` disables it). The log line
`vision backend warmed in 12.2s` appears before the worker accepts its first task, so a
cold scale-up no longer pays the model load on the first frame.

With the real prompt, the synthetic test scene still yields no detections (best score
0.0018 against a 0.25 threshold). That is the correct answer: it is a grey rectangle,
not a metal sheet. Detection *quality* remains the one thing that needs a real camera
image, not more pipeline work.

---

## 14. Prompt conditioning on a real image

The first real photograph in the repository
(`assets/images/roboflow_test_detect_human_robot.webp`, Apache 2.0) closes the gap that
remained after §13: an overhead warehouse scene with a robot arm and a worker, far
enough apart that "did it look at the right thing" is answerable.

Each prompt was tokenised **through the live API** (`scripts/dump_prompts.py` PUTs it,
reads back the IDs), then run against the image with the real model
(`scripts/evaluate_prompts.py`).

| Prompt | Top score | Box centre | Landed on |
| --- | --- | --- | --- |
| `robot` | 0.322 | (0.63, 0.30) | **robot arm** |
| `person` | 0.324 | (0.57, 0.58) | **worker** |
| `human` | 0.188 | (0.57, 0.58) | **worker** |
| `cardboard box` | 0.452 | (0.71, 0.38) | a stack of boxes |
| `metal sheet` | 0.272 | (0.75, 0.06) | conveyor/grating area |

`robot` and `human` are asked for the same image and return boxes ~0.28 apart
vertically, each on the correct subject. That is prompt conditioning working: a decoder
that ignored its text input would return the same box for every phrase - which is
exactly the failure the -65504 saturation produced earlier, and which no
plumbing test would have caught.

This is now a regression test: `tests/vision/test_detection_quality.py` (marked
`model`, skipped when the 293 MiB artifact is absent) asserts that `robot` finds the
arm, that both `human` and `person` find the worker, and that the two prompts do not
return the same box.

### The single-class limitation

The artifact's `logits` are `(1, 3600, 1)`: **one** class dimension. The graph scores
one query at a time, so:

* prompt conditioning is visible as *where* the best box is and how confident it is,
  not as a per-class label;
* a prompt with several labels reports every box as `labels[0]`, because there is no
  second class channel to attribute boxes to;
* `VISION_LABELS` therefore works as a *name* for the one class, not as a vocabulary.

Supporting multi-class prompts means re-exporting the model with a wider text/class
dimension, which is an export-side change (`requirements/export.txt`), not a pipeline
one.

### Threshold finding (safety-relevant)

With the shipped `VISION_SCORE_THRESHOLD=0.25`, **`human` would not be reported**
(0.188), while `robot`, `person`, `cardboard box` and `metal sheet` would. For a
metal-sheet pick-and-place MVP that default is fine; for a human-presence interlock it
is not. Two honest options, both better than silently lowering the number globally:

1. give each camera its own threshold alongside its prompt, or
2. keep 0.25 for actuation and use a lower threshold (0.15) for the interlock path.

The measured scores are recorded here so that whichever is chosen is a decision, not an
accident.

---

## 15. Several objects, one pass, a threshold each

The single-class conclusion in §14 was wrong, and the way it was wrong is worth
recording. `logits` came back `(1, 3600, 1)` because the text input was always sent as a
batch of **one**. The graph's text batch *is* the class axis: sending `input_ids` of
shape `(N, 16)` returns `logits (1, 3600, N)`. Measured with
`scripts/probe_query_axis.py`: 1, 2 and 3 prompts give 1, 2 and 3 query channels. This
is the approach `scripts/proof_of_work.py` already used.

So multi-object detection is **one forward pass with every label in the text batch**,
then per box the argmax over the query axis picks its class. Measured cost is the cost
of a single prompt: five queries took 5.98 s on CPU locally versus ~5 s for one, and
three queries took 10.4 s inside the 2-CPU worker.

### Per-query thresholds (the safety gap from §14)

Each query carries its own threshold, so a camera can ask for rigid objects strictly and
people loosely:

```http
PUT /cameras/main/prompt
{"prompts": [{"label": "robot",         "threshold": 0.25},
             {"label": "person",        "threshold": 0.10},
             {"label": "cardboard box", "threshold": 0.35}]}
```

A box is attributed to the query that scored it highest, and only if that score clears
**that query's** threshold. `decode_queries()` implements this; NMS runs per class, so a
confident robot box cannot suppress an overlapping person box.

Live in the cluster, on the real photograph, one pass:

```
inference 10.4 s -> 18 detections {cardboard box: 16, person: 1 (0.415), robot: 1 (0.322)}
```

### Two properties that are easy to get wrong

* **Synonyms cannibalise.** Asking for both `human` and `person` does not give two
  boxes: the same box is attributed to whichever scored higher, so the other query
  reports nothing. Configure one phrase per object. (`human` 0.188 vs `person` 0.324 on
  this image - `person` is the better phrase.)
* **A wrong prompt must not degrade to the default.** When a payload's prompt cannot be
  parsed, the worker now fails the frame with `unusable prompt: ...` instead of quietly
  falling back to the process default. Falling back reports confident detections
  labelled with a class the operator did not configure, which is worse than no
  detection. This was not hypothetical: the first live run of this feature used the HTTP
  field name (`prompts`) where the worker expected `queries`, and the fallback silently
  returned `metal_sheet` boxes for a robot/person prompt. `CameraPromptSet.from_dict`
  now accepts both names, and the fallback is gone either way.

### Storage and wire format

`system_metadata[camera_prompt:<id>].value_json` holds
`{"queries": [{label, threshold, prompt_text, input_ids, attention_mask}, ...]}`; the
human-readable `value` summarises it as `robot@0.25, person@0.1`. Legacy single-prompt
rows are migrated on read (their labels expand to one query each, default threshold), so
no database migration was needed. The dispatcher embeds the whole set in each frame
payload, keeping workers stateless.

---

## 16. Worker footprint: fitting two workers on one node

Backlog item 2 was "make the Celery worker lean enough for two". One resident ONNX session
peaked at **3434 MiB** (§13), so two workers plus Postgres, RabbitMQ, Redis, MinIO and the API
did not fit a 10 GiB node. The levers were measured rather than guessed, with
`scripts/measure_footprint.py` in the worker image - one process per configuration, because
`VmHWM` never goes down and two configurations in one process would report the first one's
peak for the second.

| Configuration | threads | arena | mem pattern | load | latency | peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (was shipped) | 20 | on | on | 4.3 s | 6825 ms | **3465 MiB** |
| **arena off (now shipped)** | 20 | off | on | 4.2 s | 6463 ms | **2042 MiB** |
| mem pattern off | 20 | on | off | 3.7 s | 6658 ms | 2145 MiB |
| arena + pattern off | 20 | off | off | 4.1 s | 12165 ms | 2027 MiB |
| single thread | 1 | on | on | 3.6 s | 17746 ms | 3463 MiB |
| single thread, lean | 1 | off | off | 3.6 s | 27034 ms | 2040 MiB |

### What the numbers say

* **The CPU allocation arena is the lever, and it is nearly free.** Turning it off cuts peak
  RSS by **41%** (3465 → 2042 MiB) with no measurable latency cost in this measurement. A
  second run in a dedicated pod, at 20 threads against a 6-CPU limit, showed arena-off ~12%
  slower (14.9 → 16.8 s) - so the honest statement is "same within noise up to ~12% slower,
  depending on thread competition", which is still a good trade for 1.4 GiB.
* **Thread count does not affect memory at all**, and cutting to one thread nearly triples
  latency. Threads buy speed; they are not a memory lever.
* **Memory patterns must stay on.** Turning both off saved only 15 MiB more but doubled
  latency (12.2 s), presumably because allocations are redone every run instead of replayed.

The arena exists to avoid re-allocating between many small inferences per second. This
pipeline runs one 960x960 pass every few seconds, so the arena is a large reservation held for
the process lifetime and nothing else. `OWL2_CPU_ARENA=1` restores the old behaviour for
throughput experiments.

### Two workers, verified

The `vision` Deployment was scaled to 2 replicas: both reached `1/1`, RabbitMQ reports
**2 consumers** on the `vision` queue, and the pods' combined RSS was ~3.0 GiB shortly after
scaling (one child still warming; ~2.0 GiB each once loaded). Node requests sat at 43% of a
10 GiB node.

**Caveat on measuring latency in a worker pod:** running a probe *inside* the worker pod
reports ~20 s/frame, because the probe and the warmed Celery child both hold a session and
compete for the pod's 2 CPUs. Latency has to be measured in a dedicated pod; memory can be
measured anywhere.

### If more than two workers are ever needed

Not pursued, because two is enough for the MVP and the cheap levers are exhausted:
the worker image is 1.55 GB (`onnxruntime-gpu` dominates and is required for the GPU path), and
the remaining large win is architectural - one inference service holding the model once, with
Celery workers as thin clients, instead of one model per replica.

---

## 17. Persisting history, and honest readiness

Detections used to exist only as WebSocket messages. The capture loop now offers every
completed frame to `src/api/detection_sink.py`, which applies the sampling policy and, when it
says yes, calls `src/api/storage/detection_repository.py`. The sink is injected into the runner
as a callable, so `src/pipeline` still has no SQLAlchemy import.

Sampling rules, in order: a failed frame is never persisted; a frame with no detections is
skipped by default; at most one frame per camera per `PERSIST_MIN_INTERVAL_SECONDS` (1 s).
That bounds writes independently of capture rate - 60 FPS becomes 1 write/s per camera - and
the policy is honest about being downsampling rather than an audit log.

The sink runs **after** fan-out, so a slow database delays history rather than the live view,
and it never raises: a database outage degrades history while detection continues. Counters are
in `/status` under `persistence`.

Readiness is separate from liveness (§5 of the backlog): `/health` stays 200 whenever the
process is alive, `/readyz` decides whether traffic should arrive, parameterised by the pod's
role through `READY_REQUIRE_PIPELINE`. A camera pod without capture leaves the endpoints; the
API stays, because it serves the page that explains the fault.

## 18. Boxes that move between inferences

The console draws detections, and detections arrive once per inference. Measured on this
host: 8.5-11 s per frame at two threads (§13), so a worker walking past a camera produced a
box that jumped once every ten seconds and otherwise sat still. That is not a rendering
problem, it is the honest output rate of the detector - which is why the fix is not a faster
preview but a cheaper estimator between the expensive ones.

`src/vision/tracking.py` seeds one OpenCV tracker per detection from a result, then advances
them at `TRACK_FPS` (10) on the newest frames. The capture process owns the pool: a tracker
needs pixel data and the frames are already there, and shipping frames to a worker to draw
boxes would cost more than the tracking.

Four decisions are worth recording.

**Seed on the newest frame, not on the detection's frame.** On CPU the detection is several
seconds old when it arrives. Seeding its own frame would place every box where the object was
when the frame was captured - correct geometry, wrong time, and visibly lagging. The tracker
starts where the object is *now*, which trades a little box error for a box that is on the
object.

**KCF by default, CSRT on request.** CSRT survives occlusion better and costs several times
more per update. At 10 updates/s next to a 2-thread inference, the cheaper tracker that
drifts is more useful than the accurate one that steals CPU. `TRACKER_TYPE` overrides, and
the fallback order is kcf -> csrt -> mil, so a build with only one tracker still works.

**Matching is label-gated, by overlap or by distance.** A new detection with IoU >= 0.3
against a live track *of the same label* keeps that track's identity and tracker. Overlap
alone was not enough: with inference seconds apart, a person walking across the frame moves
further than their own width between two results, so the next detection did not overlap the
old box at all - a second tracker started and the first expired next to it, leaving two boxes
on one object once the person stopped. A detection therefore also matches the nearest
same-label track whose centre is within 25 % of the frame diagonal (`match_distance`;
0 disables the rule). Unmatched tracks age out after 5 consecutive misses. Without the label
gate, `robot` and `person` boxes swap identities whenever they overlap. Two same-label tracks
that do converge to IoU >= 0.6 are collapsed into the older id, because two boxes on one
object are worse than one: the operator cannot tell which to trust.

**An empty detection result still seeds.** The pool is re-seeded from every successful
inference, including one that found nothing. That is how tracks expire when the model stops
seeing the object; skipping empty results would leave stale boxes drifting forever.

The wire format mirrors `Detection`: normalised geometry plus `track_id`, `age` and `stale`,
published as `{"type": "tracks"}` on the same WebSocket. The console draws tracks solid (dashed
only when a tracker lost its object) and, once tracks exist, does **not** draw the last
inference at all. Drawing both produced two boxes that never coincided - the inference box is a
whole inference old, so it contradicts the live frame it is painted over - and the complaint
was exactly that: "dotted-line frames that often do not coincide with the solid-line frames".
Without tracks the inference is drawn solid with its confidence, because then it is the only
answer available.

Failure policy is the same as everywhere else in the pipeline: no tracker installed, a bad
`TRACKER_TYPE`, or an exception inside a tracker disables the overlay for that frame and
leaves capture and detection alone. `/status` reports `tracking.enabled`, the chosen tracker
and the reason it is off.

### 18.1 Why "frames dropped" was the wrong thing to fix

Dropping frames is by design (§12): a bounded in-flight window refuses work the workers cannot
finish, instead of queueing it into a latency swamp. But with capture at 5 FPS for a live
preview and inference at ~11 s, *every* frame after the first two was refused - so
`dropped_capacity` grew steadily and looked like a fault while the pipeline was behaving
correctly. The number was measuring a cadence mismatch.

`PIPELINE_DISPATCH_INTERVAL_SECONDS` (dev overlay: `10`, base: `0`) makes the decision
explicit: capture as fast as the preview needs, offer the workers at most one frame per
interval. Frames kept only for the preview are counted as `shed_sampling`, separately from
`dropped_capacity`. The rule of thumb is to set the interval near the measured inference time;
then a rising `dropped_capacity` means the workers really are slower than the chosen sampling,
and a rising `shed_sampling` is the pipeline doing what it was told.

## 19. Presence: "is the sheet there or not?"

Everything above is about *finding* things in a scene the model can already see. The station
question is narrower and less forgiving: a metal blank either sits on the table or it does
not, and the answer decides whether the robot picks. A box in the wrong place, or a
confident box on an empty table, is worse than no box at all.

The real workpiece photograph changed this question and complicated it:
`assets/images/metal_sheet_test_image.jpg` (a brushed-metal sheet on a white table in a tiled
hallway, shot the way a fixed camera would see the station). Measured on the exported artifact
(CPU, 960x960 input, 2026-10-21, top box per prompt):

| prompt | sheet | doorway crop | wall crop | bare table crop | tiled floor crop |
| --- | --- | --- | --- | --- | --- |
| `steel blank` | **0.2459** | 0.0122 | 0.0000 | 0.0465 | 0.1911 |
| `metal blank` | **0.2448** | 0.0232 | 0.0000 | 0.0170 | 0.1743 |
| `metal sheet` | **0.1564** | 0.0064 | 0.0000 | 0.0232 | 0.0767 |

Three conclusions, each with a consequence for deployment:

1. **The model localises the real sheet correctly and cheaply.** Every prompt above puts its
   best box at the same place (centre `0.483, 0.588`, size `0.590 x 0.421`), which is the
   sheet and not the wall behind it. Absence crops of the same photograph score at most
   `0.0465`, so the separation between "there" and "not there" is a factor of five.
2. **The deployed default of 0.25 would call an empty station.** `steel blank` reaches
   `0.2459` - a near miss that reads as "no part" at the default threshold, while
   `metal sheet` (`0.1564`) is a plain miss. The threshold therefore belongs to the *camera*,
   set from photographs of that station, not to the model. `PUT /cameras/{id}/prompt` already
   carries a threshold per query (§15); this is what it is for.
3. **`steel blank` beats `metal sheet`.** The industrial word for a cut sheet is the one the
   text encoder places closer to the pixels of a cut sheet.
   `tests/vision/test_metal_sheet_presence.py` asserts this, so a later "simplification" back
   to `metal sheet` fails loudly.

The uncomfortable finding is the right-hand column: **the tiled floor scores 0.19**. Glossy
beige tiles read as brushed sheet metal, and no threshold separates them from a real blank at
`0.24`. That is not a bug in the model; it is a statement about where the camera may point.
A presence check needs the station to fill the frame, or a zone constraint (backlog item 4), or
both. The photograph also shows why calibration has to be per-camera: aiming tighter *did not*
help - cropping to the table dropped `steel blank` from `0.2459` to `0.1235`, and to the sheet
alone to `0.0569`, because this export uses surrounding context to decide that a flat rectangle
is a workpiece and not a table top.

### 19.1 Calibrating a station camera

`scripts/calibrate_presence.py` turns "how confident is the model here?" into a number a
deployment can use. It scores photographs from the station, some with the part and some
without, and prints the midpoint of the two groups - refusing to print a threshold when the
groups overlap or sit closer than `2 x margin`:

```
python scripts/calibrate_presence.py \
    --query assets/prompts/steel_blank.json \
    --positive "photos/station_loaded/*.jpg" \
    --negative "photos/station_empty/*.jpg"
```

On the real photograph, with the doorway, wall and bare table as the empty set, it reports
`positive min 0.241`, `negative max 0.049`, `threshold 0.145`. The same number reaches the API
as `PUT /cameras/metal-sheet/prompt` with `{"label": "steel blank", "threshold": 0.15}`.

Presence is then the ordinary detection path: any box for that label above the threshold means
"the part is there". Nothing extra is persisted, so `detection_results` answers "when was the
sheet last seen" without a new table, and the tracker (§18) keeps the box steady between the
inference results an operator actually needs to trust.

**Verified in the cluster** after registering `metal-sheet` (replay source
`/app/assets/images/metal_sheet_test_image.jpg`, prompts `steel blank` and `metal blank` at
`0.15`, activated):

| camera | label | rows | min | max |
| --- | --- | --- | --- | --- |
| `metal-sheet` | `steel blank` | 2 | 0.202 | 0.222 |
| `metal-sheet` | `metal blank` | 1 | 0.208 | 0.208 |

Persisted geometry `x 0.17-0.20, y 0.377, w 0.59, h 0.42` (centre `0.48, 0.59`) - the sheet,
in the deployed pipeline, at a threshold the defaults would have rejected.

### 19.2 When the best box is the table

Look at what the console drew on the photograph and the calibration above looks too
optimistic. The model puts its **best** box around the white table - the thing the sheet lies
on - and the workpiece is its *second* candidate. With `score_threshold=0.02`, top three boxes
per phrasing (centre, size, fraction of the box inside the sheet):

| prompt | table | part |
| --- | --- | --- |
| `steel blank` | 0.2459 _(0.483, 0.588)_ `0.590x0.421` | **0.1594** _(0.494, 0.570)_ `0.311x0.232`, 90 % inside the sheet |
| `stainless steel sheet` | 0.1501 | 0.1259 |
| `brushed steel plate` | 0.1428 | 0.1078 |
| `reflective metal plate` | 0.1200 | 0.0891 |
| `mirror` | 0.3349 (table) / 0.5684 (floor tiles) | - |
| `shiny metal plate` | 0.0559 | - |

No phrasing promotes the part above the table. The part is distinguished by *where and how
big it is*, not by how it looks: a brushed sheet, a table top and a tiled floor are the same
kind of flat, light, low-texture rectangle. Two further attempts were checked and rejected:

* **Cropping tighter does not help.** It removes the scene context the export uses: the
  station crop scored `0.1235`, the sheet-only crop `0.0569`.
* **A gloss/reflection measure is not a classifier.** Specular highlights move with the
  lighting and the camera position, so they would need their own calibration per stand and
  still would not separate a table from a sheet.

The workable answer is a **geometry gate**: the station is fixed, so the part is the only
object whose box has the part's size, aspect and position. `src/vision/presence.py` applies
that to the boxes inference already produced - no second model, no new table:

```python
PresenceGate(
    roi=(0.15, 0.42, 0.85, 0.85),   # the station region, normalised
    min_area=0.02, max_area=0.20,   # a part is small; a table is not
    min_aspect=1.0, max_aspect=2.0, # a sheet lies flat
    min_confidence=0.05,            # the part's own score moves across a pan
    min_inside=0.5,                 # half over the edge still counts
)
```

Measured on the photograph, the gate refuses 7 of the 8 boxes and keeps exactly the part:
the table is rejected as `area 0.2484 > 0.2000 (a table, not the part)`, the floor box as
`only 36% inside the station region`, and the answer is `present`, one accepted box, centre
`(0.494, 0.570)`, size `0.311 x 0.232`, confidence `0.1594`.

#### Why the confidence floor is 0.05 and the region starts at 0.42

The first deployment of this gate alternated `part present` / `no part` on every other
inference, and the cause was not the gate's idea, it was the *cadence*. The replay source pans
its still by 24 pixels in an eight-phase sine, so the same photograph produces slightly
different scores each loop. On the frame the vision worker actually sees (encoded at quality 80,
long edge 960), across the eight phases:

| box | confidence | area | aspect | centre |
| --- | --- | --- | --- | --- |
| the table | 0.176 - 0.223 | 0.251 | 1.40 | 0.48, 0.59 |
| **the part** | **0.085 - 0.125** | 0.075 | 1.31 | 0.49, 0.57 |
| the floor | 0.071 - 0.140 | 0.110 | 1.32 | 0.44, 0.88 |
| the wall behind the stand | 0.070 - 0.087 | 0.083 | **2.27** | 0.65, 0.40 |

The part and the wall **overlap in confidence** (0.085 vs 0.087), so no confidence floor can
separate them: at 0.10 the part disappears on the low phases (hence the flapping), and at a
floor low enough to keep it (0.05) the wall gets in (86 % of its box lies inside the old
region). What separates them is geometry, and only geometry:

* the **region's top edge at 0.42** sits below the wall band (y 0.31 - 0.50), so at most 43 %
  of it counts as inside the station;
* its **2.27 aspect** is outside the 1.00 - 2.00 band, independently of the region.

Both rules refuse it, which is the point: one threshold that happens to work today is not a
calibration. With the values above all eight phases of the part are accepted and the table,
the floor and both wall boxes of the pan are refused.

The pipeline publishes the answer as its own message, next to the detections that produced it:

```json
{"type": "presence", "camera_id": "metal-sheet", "frame_number": 41,
 "present": true, "offered": 8, "accepted": 1, "reason": null,
 "detection": {"object_type": "steel blank", "x_norm": 0.3385, "y_norm": 0.563,
               "width_norm": 0.314, "height_norm": 0.239, "confidence": 0.1199}}
```

The prompt that produced those numbers is `steel blank` at threshold 0.08, which is the
*detection* threshold, not the presence floor: the two are separate knobs on purpose. The
station's own prompt set (`steel blank`, `metal blank`) is stored per camera, so changing the
phrasing is a console edit, not a deploy.

`/status` carries the counters (`checks`, `present`, `errors`, `last_reason`) and the gate
that is in force. Three properties matter more than the numbers:

* **A failed inference is not an empty station.** A worker timeout publishes nothing, so a
  slow frame cannot stop a line.
* **The reason is reported.** "No part" and "the gate is misconfigured" look identical from
  outside otherwise, and they are the two failures that matter.
* **Off by default.** `PRESENCE_ENABLED` is the switch; without it the pipeline does no extra
  work at all. The region and the size band are properties of one physical stand, so the
  defaults are deliberately loose and a deployment configures its own (see
  `k8s/overlays/dev/patches/console-single-pod.yaml` for the measured values of this stand and
  `doc/k8s.md` for the keys).

What this gate does **not** do: it does not find the part anywhere in the frame (that is the
detector's job), it does not see a part that is outside the region or turned on edge, and it
cannot tell two identical parts apart. It answers one question - "is the part at the station
now?" - and answers it without pretending to more.

What this section does **not** claim: that the model is good at this in general. It was
measured on one workpiece, one camera position and one afternoon's lighting. The honest
position is that presence at a station is a *calibrated* use of a zero-shot model, not a
capability it has out of the box, and the calibration is only as good as the photographs
behind it. GPU acceleration remains deferred by decision (Stage 4 item 12): the measurements
above are CPU, and they are enough to say "there" or "not there".


## 20. Choosing a detector with a CSV, not with an opinion

Section 19 ends with a sentence worth following: presence at a station is a *calibrated* use of a
zero-shot model, and calibration is only as good as the photographs behind it. The obvious next
question is whether a different detector would need less calibration.

That question is answered by measurement, not by reading model cards on the internet. `src/benchmark/`
holds a small harness: one adapter per runtime, one dataset built from the frames this deployment
actually carries, one scoring path through the shipped `PresenceGate`, and one CSV. The details -
the protocol, the ground truth, the licences and the numbers - are in `doc/detector-benchmark.md`;
the evidence file is `reports/detector_benchmark.csv`.

Two decisions in that harness matter more than any single model, and both were paid for in
measurement:

* **Score the frames the pipeline carries.** Full-resolution photographs put 2-4 points of
  confidence on every model for free (the incumbent's top box read 0.1564 raw, 0.1291 after the
  capture encode). A benchmark that skips the encode measures a system nobody ships.
* **Use the phrase the deployment stores.** `metal sheet` gave OWLv2 recall 0.25 and sheet IoU
  0.43 on the real workpiece; the deployed `steel blank` gave 1.00 and 0.86. A benchmark with a
  friendlier phrase than production is a benchmark of a different product.

The result that closed the question for now: the incumbent is the only candidate that is both
Apache-2.0-compatible and able to see the part at all; the fast alternatives are AGPL-3.0
(YOLO-World) and blind to a bare metal blank, and the accurate closed-vocabulary alternatives
(Grounding DINO base, Florence-2) could not be downloaded on this network and are recorded as not
measured rather than as failures. Re-running the CSV on a working network is the next honest step,
and it needs no code change.
