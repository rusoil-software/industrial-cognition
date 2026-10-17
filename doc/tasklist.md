# Project Task List: Industrial AI Recognition System

This list tracks all development tasks required to bring the system from blueprint to production. Tasks must be completed sequentially to maintain functional integrity.

**Status Legend:**

* ✅: Completed and verified.
* 🟡: In Progress (Requires active work).
* 🔴: Blocked (Requires external decision or prerequisite task completion).
* ⚫: To Do (Initial task).

---

## 🚀 Development Stages

### **Stage 0: Foundation & Documentation (COMPLETED)**

* ✅ Finalize Core Vision & Tech Stack: (Completed)
* ✅ Define Communication Protocols: (Completed - Modbus/RS-485)
* ✅ Establish Development Conventions: (Completed - KISS, SRP, etc.)

### **Stage 1: Foundational Infrastructure (NEXT FOCUS AREA)**

* ✅ Setup K8s Manifests: Kustomize `base` + `dev`/`prod` overlays covering every microservice, the data
  services, networking, autoscaling and the optional observability/KEDA components. Rendered and
  contract-tested by `tests/k8s` and `.github/workflows/k8s/validate.yaml`; documented in `doc/k8s.md`.
* ✅ Database Schema Migration: Alembic is wired up (`alembic.ini` + `alembic/env.py`), the ORM models in
  `src/api/storage/models.py` define `detection_results` and `system_metadata` per `doc/vision.md`, and the
  initial revision creates both tables plus the `object_type`/`camera_status` enums. Verified against a real
  PostgreSQL: upgrade, full downgrade/re-upgrade round trip, empty second autogenerate, and insert/read-back.
  `pytest tests/db` (12 static + 10 behavioural tests); documented in `doc/database.md`.
* ✅ Base Dependency Management: `requirements/prod.txt` finalised as the union of the runtime dependency sets
  shared by both production images, excluding the `dev`/`export`/`k8s` tiers. The two images still differ in
  exactly one respect — `api` installs `onnxruntime`, `vision` installs `onnxruntime-gpu`, and those cannot
  coexist in one environment. `requirements/api.txt` and `base.txt` gained `psycopg[binary]==3.2.9` to match
  the deployment's `postgresql+psycopg://` DSN.

### **Stage 2: Core Functionality Implementation**

* ✅ Camera Service: Device/RTSP capture implemented in `src/camera/capture.py` (MJPG + 60 FPS negotiation,
  `CAP_PROP_BUFFERSIZE=1` to avoid draining stale frames, JPEG encode at the model's resolution) and unit
  tested. **Remaining:** push results to clients over the FastAPI WebSocket (`/ws/detections`); the payload
  already matches `src/camera/schemas.py::StreamingPayload`. Real device negotiation needs the camera hardware.
* ✅ Vision Service: OWLv2 on ONNX Runtime is implemented end to end and unit tested — preprocessing
  (`src/vision/preprocess.py`), graph decoding (`src/vision/decode.py`: centre→corner, sigmoid, NMS, fp16),
  backend seam (`src/vision/backend.py`) and the per-frame worker (`src/vision/worker.py`). The model contract
  was read from the artifact with `scripts/inspect_model.py`. 72 tests in `tests/vision`.
  *Caveat resolved:* the default prompt is now genuinely CLIP-tokenised (`k8s/dev/text_queries.json`), the
  tokenizer is baked into the API image so prompts can be set offline over HTTP, and the runtime was tuned
  (cgroup-aware thread pool, warmup at worker start): 26.8 s → 10.1 s per frame at the deployed 2-CPU limit.
  Detection *quality* is now verified on a real Apache-2.0 test image
  (`tests/vision/test_detection_quality.py`): 'robot' localises the arm and 'human'/'person' the worker,
  with the two prompts returning different boxes. Multi-object detection is one forward pass with
  every label in the text batch, and each query carries its own threshold, set by the System User
  through `PUT /cameras/{id}/prompt` (`doc/vision-pipeline.md` §15).
* ✅ Robot Interface: Modbus RTU/TCP implemented server-side — `protocol.py` (CRC, framing, PDU builders,
  exception decoding), `transports.py` (RS-485 via pyserial, Modbus TCP, and a PDU-accurate simulated
  controller), `service.py` (bounded retries, typed errors, no retry on device refusal) and
  `integration_manager.py` (detection → millimetres → register units, with STALE_DETECTION / OUT_OF_REACH /
  HUMAN_ON_SITE refusals). 46 tests in `tests/robot`.
  **No hardware yet:** the transport defaults to `simulated`, and the register map / units are placeholders
  to confirm when a converter and the rebuilt robot exist. See `doc/robot-interface.md`.
* ✅ Orchestration: Per-camera prompt API implemented (`src/api/prompts/`: put/get/delete/list at
  `/cameras/{camera_id}/prompt`), tokenised through `src/vision/prompt.py`, stored in
  `system_metadata` via `src/api/storage/prompt_repository.py`, and carried in each frame payload so a
  prompt applies only to its own camera. Multi-worker dispatch implemented and load tested — bounded in-flight window with
  load shedding, capture/dispatch/completion FPS and latency percentiles (`src/pipeline/`), Celery task and
  transport, and the 60 FPS acceptance tests (`tests/pipeline`, 28 tests). **Remaining:** wire the dispatcher
  into the FastAPI app. Done: `src/pipeline/runner.py` owns the capture loop, `src/api/main.py` was rewritten
  around a non-fatal lifespan, and `/health` + `/status` expose the pipeline snapshot.

**Stage 2 throughput note (measured, `doc/vision-pipeline.md` §1):** one OWLv2 pass costs **4053 ms on CPU**
and ~20-40 ms expected on a GPU, so the 60 FPS aggregate target needs **2-3 GPU workers** (or is unreachable
on CPU: 244 workers). The pipeline itself is not the bottleneck; it sustains 60 FPS with adequate workers and
sheds frames with bounded latency otherwise, both asserted by tests.

### **Pre-Stage 3 backlog (agreed 2026-10-17)**

Ordered roughly by what blocks what. Items 1-3 are the ones we named; the rest are gaps this
build exposed that will cost more to fix later than now.

**1. &#9744; Put the model back on the PVC (and fix the loader pod first).**
Recreating the cluster recreated `industrial-cognition-model-cache` empty, so `vision` sits at
`0/1` with nothing to load - the console previews and edits tasks, but detects nothing.
`k8s/tools/model-loader.yaml` runs as root with no seccomp profile and trips the
`restricted:latest` PodSecurity warning on every use. Before repopulating:
add `runAsNonRoot`, `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`,
`seccompProfile: RuntimeDefault` and an `fsGroup` that can write the volume, then
`kubectl cp` the artifact, verify it with `scripts/inspect_model.py`, and confirm one real
inference with `scripts/smoke_inference.py`.
*Done when:* `vision` is `1/1` and a task returns detections from the real model.
*Note:* a fresh cluster is also a fresh database - prompts live in Postgres, so a cluster
recreate silently loses them. Re-set them after recreating (or seed them in the installer).

**2. &#9744; Make the Celery worker lean enough for two.**
One resident ORT session peaks at **3432 MiB** (measured, `doc/vision-pipeline.md` §13); the dev
limit is 6 GiB and the node has 10. Two workers is therefore ~6.9 GiB for inference alone,
leaving too little for Postgres, RabbitMQ, Redis, MinIO and the API.
Investigate, in order: (a) ORT session options - `enable_cpu_mem_arena`,
`arena_extend_strategy`, `enable_mem_pattern` - with RSS measured for each; (b) thread count
per worker (intra-op is already cgroup-aware, but two workers sharing cores contend);
(c) `worker_max_tasks_per_child` / `max_memory_per_child` so a long-lived child cannot creep;
(d) image size (the worker image is 1.55 GB - trim what serving does not need);
(e) a single inference service that the Celery workers call, so the model is resident once
instead of per replica - the architectural answer if (a)-(d) are not enough.
*Done when:* two workers plus the rest of the stack fit the node, with measured RSS and
per-frame latency recorded in `doc/vision-pipeline.md`.

**3. &#9744; Find the real camera from the OS and run it end to end.**
Windows: `Get-PnpDevice -Class Camera` for the device, OpenCV `CAP_DSHOW` for the index.
Linux: `/dev/video*` plus `v4l2-ctl --list-devices --list-formats-ext` to confirm MJPG at the
target rate. **Containers cannot see a USB camera on Docker Desktop/WSL**, so the practical
path is to run capture on the host and publish RTSP (ffmpeg or mediamtx), then point
`CAMERA_SOURCE=rtsp://...` at it - the same code path production uses. `DeviceFrameSource`
already negotiates MJPG, sets `CAP_PROP_BUFFERSIZE=1` and warns when the negotiated FPS is
below 90% of the request; add a `scripts/probe_camera.py` that lists devices, formats and the
achievable FPS so this is diagnosable rather than guesswork.
*Done when:* real frames reach the console with overlays, at a documented resolution/FPS.

**4. &#9744; Persist detections.**
`detection_results` and its ORM exist, but nothing writes to them: results only go out over the
WebSocket. Stage 3 history and analytics need a writer with a retention policy, and at 60 FPS a
sampling rule (per class, per camera, or on change) rather than one row per frame.

**5. &#9744; Make readiness honest.**
`/health` returns 200 with `pipeline_running: false`, so Kubernetes reports the API ready while
detection is degraded - the console can look "up" with a dead pipeline. Split liveness
(`/health`) from readiness (`/readyz`, which fails when capture is expected but not running) and
decide per deployment what "ready" means (the API pod serves the UI *and* captures in dev).

**6. &#9744; Harden auth before anything faces a network.**
Dev credentials are committed in `k8s/base/secrets.env`; `AUTH_TOKEN`/`JWT_SECRET` must be
rotated, `AUTH_COOKIE_SECURE=1` behind TLS, token lifetime decided, and `/docs` +
`/openapi.json` closed in production. Today it is one shared deployment token - per-user
accounts and roles are a later step, but the shared token must not survive into a shared
network.

**7. &#9744; Real ingress and TLS.**
The dev overlay installs no ingress controller (NodePort 30080 is dev-only) and the prod ingress
references secret `industrial-cognition-tls` that nothing creates. Need cert-manager or a
provided certificate, a real hostname, and probes that match item 5.

**8. &#9744; Broker-side queue hygiene.**
We observed a 41-message backlog and `queue_ms` of 353 s. The client sheds frames, but the broker
does not: set `x-max-length` / message TTL on `vision` so stale frames die at the broker, and
give `vision.dlq` a producer (it is declared and unused) plus a retry policy for
genuinely-failed tasks.

**9. &#9744; Decide multi-camera shape.**
The runner is one source per process; per-camera prompts already exist but queues, task routing
and stats assume a single capture loop per deployment. Either run one capture Deployment per
camera (simplest, needs per-camera services/ingress paths) or teach the runner multiple sources
with per-camera windows and stats. Also per-camera `PIPELINE_MAX_IN_FLIGHT`, which must be sized
from inference latency, not left at a default.

**10. &#9744; Make the deployment itself testable in CI.**
`tests/k8s` checks manifests, but nothing exercises auth, the stream endpoints or the WebSocket
outside a manual `scripts/smoke_*.py` run. Promote `smoke_auth.py` and `smoke_ui.py` into a
pytest-marked integration suite against kind/minikube, so the six-to-eight bugs each live run
found would be caught automatically.

**11. &#9744; Detection quality beyond one photograph.**
One Apache-2.0 image proves prompt conditioning; thresholds are still chosen from a handful of
scores (`doc/vision-pipeline.md` §14). Build a small ground-truth set of the real workcell per
camera and tune per-query thresholds against it, and use
`POSE-Lab/IndustryShapes` (already noted in `assets/images/README.md`) for a broader evaluation
before Stage 3 claims quality numbers.

**12. &#9744; Decide where the GPU sits.**
The prod overlay requests `nvidia.com/gpu` and the worker image installs `onnxruntime-gpu`, but
nothing has ever run on a GPU: every measurement is CPU at ~11 s/frame. Before performance work
in Stage 3, confirm a GPU runtime actually exists (`CUDAExecutionProvider` visible in the pod)
or state plainly that 60 FPS is out of scope for now.

**13. &#9744; Model delivery stays manual, by agreement.**
The artifact lives on a PVC and is copied by hand. Revisit together with the MinIO replacement
(Stage 4) so the worker pulls it at startup with checksum verification instead of `kubectl cp`.

### **Stage 3: Polish & Hardening**

* ✅ Operator console: Vue 3 single-page UI (ui/, no build step) served at /ui/ with a
  live MJPEG preview, detection overlays and per-camera cognition tasks (one query per
  object, each with its own threshold). Backend: src/api/stream.py,
  src/camera/sources.py (including a still-image replay source so the whole pipeline
  runs without a camera). See doc/operator-console.md. Missing before any network
  exposure: authentication.

* 🟡 End-to-End Testing: Write and execute full integration tests covering the entire pipeline (Camera -> Vision -> Modbus).
  **Live deployment done (2026-10-17):** the whole stack runs on minikube — all pods `Running`, schema migrated,
  service discovery, prompt repository against real Postgres, and the full
  API → RabbitMQ → vision worker → Redis task path verified with a real JPEG (0.30 s, 2 detections).
  Six deployment bugs were found and fixed; see `doc/k8s.md` "Live run". Remaining: real camera/OWLv2/Modbus
  hardware, and automated assertions rather than a smoke script.
* 🟡 Observability Integration: Hook up Prometheus metrics scraping endpoints to all service checkpoints.
  The Kubernetes side is already in place (`k8s/components/monitoring` scrapes the
  `prometheus.io/scrape` annotations on every Deployment); what remains is the `/metrics` endpoint itself in
  the FastAPI app and the vision worker, plus a Redis exporter for `redis_exporter`. The commented jobs in
  `k8s/components/monitoring/prometheus/prometheus.yml` are the exact targets to enable.
* 🟡 Deployment Automation: Finalize Helm chart package for production deployment.
  Superseded in practice by the Kustomize overlays (`doc/k8s.md` §1 explains why Helm is deferred); revisit
  only if a Helm-native consumer appears.

### **Stage 4: Tracked Issues (deferred until the MVP runs end to end)**

* 🔴 **Object-storage backend: replace MinIO.** `minio/minio` and `minio/mc` no longer resolve on Docker Hub or
  Quay (verified with `docker manifest inspect`; `alpine`/`redis`/`postgres`/`rabbitmq` all resolve, so it is
  the upstream repository, not the network). The manifests currently use `bitnamilegacy/minio:latest`, an
  explicitly unmaintained archive repository — a stopgap, not a target. Evaluate **SeaweedFS**, **Garage**,
  **RustFS**, **Ceph RGW** or **Versity Gateway**, roughly in that order.
  *Scope:* only `k8s/base/statefulset-minio.yaml`, the `minio-bucket-init` Job and the credential key names
  change. The application contract is the S3 API plus `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` /
  `MINIO_SECRET_KEY` / `MINIO_BUCKET_*`, so no service code is affected. See `doc/k8s.md` §2
  "Object storage (S3)".
* 🔴 **RabbitMQ HA.** The broker is a replicated-single StatefulSet. True multi-node clustering needs the
  RabbitMQ Cluster Operator; see `doc/k8s.md` §2 "RabbitMQ clustering".
