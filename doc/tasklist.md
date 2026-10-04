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

* 🟡 Camera Service: Implement low-level RTSP capture, using OpenCV, and streaming results via WebSocket.
* 🟡 Vision Service: Implement the core detection logic using OWLv2 on ONNX Runtime.
* 🟡 Robot Interface: Implement the Modbus/RS-485 communication service that consumes detection data and sends commands.
* 🟡 Orchestration: Develop the FastAPI layer to tie together the services (Routing/Error Handling).

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
