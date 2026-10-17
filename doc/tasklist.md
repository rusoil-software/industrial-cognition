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
  *Caveat:* real detections additionally need a tokenised prompt
  (`scripts/encode_text_queries.py` → `OWL2_TEXT_IDS_PATH`), because the artifact was exported with random
  dummy tokens; see `doc/vision-pipeline.md` §6.
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

### **Stage 3: Polish & Hardening**

* 🟡 End-to-End Testing: Write and execute full integration tests covering the entire pipeline (Camera -> Vision -> Modbus).
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
