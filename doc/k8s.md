# Kubernetes (Kustomize) Deployment Guide

This document is the single source of truth for the Kubernetes deployment model of the Industrial Cognition
system. It describes *what* is deployed, *how* the manifests are organised and rendered, and *how* an operator
brings the stack up, verifies it and tears it down.

It replaces the earlier ad-hoc flat manifest dump in `k8s/` (commit `a532152`). That dump declared Deployments
that referenced a `ConfigMap`/`Secret` which was never created, declared the API `Service` three times with
conflicting ports, mixed two different label/naming schemes, and had no namespace, no persistent volumes, no
autoscaling, no network policy and no validation. The current layout fixes all of that and is machine-verified
(see [Validation](#validation)).

---

## 1. Why Kustomize (and not plain `kubectl apply` or a Helm chart)

The project constraints (`doc/conventions.md`) mandate the **KISS principle**: prefer the simplest working
mechanism. Therefore:

* **Kustomize** is chosen as the templating engine because it ships inside `kubectl` (`kubectl kustomize`,
  `kubectl apply -k`). There is **no extra tool to install** on an industrial server, no templating language,
  and every rendered manifest is valid, reviewable YAML.
* `doc/vision.md` mentions Helm for production packaging. Helm is *not* required by any functional
  requirement and adds a toolchain dependency, so it is deliberately deferred; the Kustomize overlays
  (`k8s/overlays/dev`, `k8s/overlays/prod`) already provide environment-specific configuration and are the
  packaging unit. Because Kustomize output is plain YAML, a future `helm template`-style wrapper can be added
  without changing the manifests.

Layout:

```tree
k8s/
├── base/                     # Environment-independent definitions
│   ├── kustomization.yaml    # Entrypoint for `kubectl kustomize k8s/base`
│   ├── namespace.yaml
│   ├── serviceaccount.yaml   # ServiceAccounts + RBAC + PriorityClasses
│   ├── app-config.yaml       # ConfigMap `industrial-cognition-config`
│   ├── secrets.env           # Secret `industrial-cognition-secrets` (DEV DEFAULTS, see §5)
│   ├── storage.yaml          # Standalone model-cache PVC
│   ├── statefulset-postgres.yaml
│   ├── statefulset-rabbitmq.yaml
│   ├── statefulset-redis.yaml
│   ├── statefulset-minio.yaml
│   ├── deployment-api.yaml
│   ├── deployment-camera.yaml
│   ├── deployment-vision.yaml
│   ├── deployment-robot.yaml
│   └── ingress.yaml
├── components/
│   ├── monitoring/           # Optional: Prometheus, Grafana, Loki, Alloy
│   └── keda/                 # Optional: queue-depth autoscaling for the vision workers
├── scripts/                  # deploy.sh, teardown.sh, validate.sh
└── overlays/
    ├── dev/                  # Lightweight, single-node data services
    │   ├── kustomization.yaml
    │   └── patches/
    └── prod/                 # Architecture parity: 4-server MinIO, monitoring, hardening
        ├── kustomization.yaml
        ├── autoscaling.yaml    # HPAs for api + camera (production only: needs a metrics-server)
        ├── hpa-vision-cpu.yaml # vision HPA - swap for components/keda to scale on queue depth
        ├── availability.yaml   # PDBs, ResourceQuota, LimitRange
        ├── networkpolicy.yaml  # default-deny + explicit allows
        └── patches/
```

`k8s/overlays/dev` and `k8s/overlays/prod` are the only directories an operator applies. `k8s/base` renders on
its own, but the overlays are what pin image tags, storage classes, replica counts and the optional
components.

Why autoscaling, availability and network policies live in `overlays/prod` rather than in `base`: all three are
policy objects whose correctness depends on the production topology — a metrics-server being installed, nodes
being drained, and the physical RTSP/Modbus network being reachable. Shipping them in `base` would force the
`dev` overlay to either require infrastructure a laptop does not have, or black-hole the pipeline behind a
default-deny egress policy. `dev` therefore renders without them and `prod` adds them explicitly.

---

## 2. Workload model

The Python application is currently a **modular monolith**: `src/api/main.py` bootstraps the camera, vision
and Modbus (robot) services in-process, while `src/tasks/celery.py` provides the asynchronous worker entry
point. The Kubernetes layout keeps the domain isolation required by `doc/conventions.md` — one Deployment,
Service, ServiceAccount and probe set per domain — so that the monolith can be split later without touching
the manifests' contract.

| Component  | Kind        | Image (default)                    | Port | Replicas (base) | Notes |
| ---------- | ----------- | ---------------------------------- | ---- | --------------- | ----- |
| `api`      | Deployment  | `industrial-cognition/api`         | 8000 | 2               | FastAPI gateway; `/health`, `/ws/detections`. Stateless, scales horizontally. |
| `camera`   | Deployment  | `industrial-cognition/api`         | 8000 | 1               | FFmpeg/RTSP capture. Pinned to RTSP-reachable nodes; may be privileged for `/dev/video*`. |
| `vision`   | Deployment  | `industrial-cognition/vision`      | 9100 | 2               | Celery worker running OWLv2 on ONNX Runtime. GPU-scheduled when `gpu.enabled=true`. |
| `robot`    | Deployment  | `industrial-cognition/api`         | 8000 | 1               | Modbus/RS-485 bridge. **Singleton**: one process must own the serial bus. |
| `postgres` | StatefulSet | `postgres:15-alpine`               | 5432 | 1               | Detection results + metadata (Alembic-managed). |
| `rabbitmq` | StatefulSet | `rabbitmq:3.13-management-alpine`  | 5672 / 15672 / 15692 | 1 | Priority queues, results queue, dead-letter queue. See the clustering note below. |
| `redis`    | StatefulSet | `redis:7-alpine`                   | 6379 / 9121 | 1     | Celery result backend + state cache. |
| `minio`    | StatefulSet | `bitnamilegacy/minio`              | 9000 / 9001 | 4     | S3-compatible store for frames and inference results. See "Object storage" below. |

Why `camera` and `robot` reuse the `api` image: `src/api/Dockerfile` is the only Dockerfile that installs the
full `src` tree with the API dependency set, and `doc/project-development.md` mandates that this Dockerfile
stays the canonical one ("do not create others"). The two components are therefore deployed as the same
image with a different `command`/`args` contract. When dedicated entrypoints are added later, only
`images:` in `k8s/overlays/*/kustomization.yaml` needs to change.

`vision` does **not** reuse the `api` image. `src/api/Dockerfile` installs only `requirements/api.txt`, which
contains no `celery`, so a worker started from it dies with
`exec: "celery": executable file not found in $PATH`. `src/vision/Dockerfile.worker` installs
`requirements/vision.txt` *and* then `requirements/api.txt`, because the worker imports `src.vision.service`
→ `src.camera.schemas`. The two files cannot simply be merged into one list: `api.txt` pins
`onnxruntime==1.24.4` while `vision.txt` pins `onnxruntime-gpu==1.24.4`, and pip cannot satisfy both.
`k8s/scripts/build-images.sh` builds both images.

### Object storage (S3): why the image is not `minio/minio`

`minio/minio` and `minio/mc` **no longer resolve** on Docker Hub or Quay. Verified with
`docker manifest inspect`: both are `MISSING`, while `docker.io/library/alpine`, `redis:7-alpine`,
`postgres:15-alpine` and `rabbitmq:3.13-management-alpine` all resolve — so it is the upstream repository
disappearing, not the network, a rate limit or a wrong tag.

The base manifest therefore uses the only pullable image found, `bitnamilegacy/minio:latest`, which ships both
the server (`/opt/bitnami/minio/bin/minio`) and the client (`/opt/bitnami/minio-client/bin/mc`, already on
`PATH`). That single image covers the StatefulSet *and* the `minio-bucket-init` Job. Two details are
load-bearing:

* The volume set must be a **positional argv**, not only `MINIO_VOLUMES`. `minio server` with no positional
  volume argument fails with `FATAL Invalid command line arguments: use path style endpoint for single node
  setup`, regardless of `MINIO_VOLUMES`. The manifest passes `"$(MINIO_VOLUMES)"` so the kubelet expands the
  same variable the distributed (base) and single-node (dev overlay) forms both set.
* The `mc` container must set `MC_CONFIG_DIR` to a writable path. `mc` persists alias state under `$HOME/.mc`,
  which is read-only under `readOnlyRootFilesystem` + a non-root UID, producing
  `Unable to save new mc config. open /.mc/...: read-only file system` on a loop and a Job that never
  completes.

> **Tracked issue — storage backend choice.** MinIO is no longer a safe dependency: the public images were
> withdrawn, and `bitnamilegacy/*` is an explicitly unmaintained archive repository, so the fallback is a
> stopgap, not a target. The application only needs an **S3-compatible API** (`MINIO_ENDPOINT`,
> `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET_FRAMES`, `MINIO_BUCKET_RESULTS` in
> `k8s/base/app-config.yaml`), so the decision is deliberately deferred until the MVP runs end to end.
>
> Candidates to evaluate, in the order they seem worth a spike:
>
> | Option | Why it is interesting |
> | --- | --- |
> | **SeaweedFS** | Mature, Apache-2.0, S3 gateway, runs happily in a single small pod, large contributor base. |
> | **Garage** | Lightweight, geo-distributed by design, S3 API, single static Rust binary; good fit for an on-prem industrial server. |
> | **RustFS** | Newer Rust S3 implementation; check project maturity/licence and operational track record first. |
> | **Ceph (RADOS Gateway)** | The heavyweight, fully-featured answer; a large operational commitment for one industrial server. |
> | **Versity Gateway** | S3 front-end over existing filesystem/tape; relevant only if the site already standardises on Versity. |
>
> Whichever is chosen, only `k8s/base/statefulset-minio.yaml`, the bucket-init Job and the credentials keys
> need to change — no application code, because the contract is S3 and the endpoint is configuration.

### RabbitMQ clustering

The broker runs as a **replicated-single** StatefulSet (one Ready replica). Multi-node Erlang clustering —
cookie distribution, quorum queues, partition handling, safe scale-down — is exactly the problem the official
RabbitMQ Cluster Operator exists to solve, and a shell-scripted approximation would be less reliable than a
single node. For HA, install the operator and replace `statefulset-rabbitmq.yaml` with a `RabbitmqCluster`
custom resource: the Service name (`rabbitmq`), ports and credential keys that the application consumes stay
identical, so no Deployment changes are needed.

### Workload identity

Each domain has its own `ServiceAccount` with `automountServiceAccountToken: false` (`api`, `camera`,
`vision`, `robot`, `data-services`). Only `prometheus` and `alloy` mount a token, because they are the only
components that talk to the API server — Prometheus for target discovery, Alloy to enumerate pods for log
collection. Alloy's RBAC is a `ClusterRole`, since pod/log enumeration is cluster-scoped; everything else is
namespace-scoped.

---

## 3. Configuration contract (`ConfigMap` `industrial-cognition-config`)

Every key below is consumed by the Deployments' `envFrom`. The first block mirrors the settings the Python
code reads today (`src/owl/inference/config.py`, `src/tasks/celery.py`, `src/vision/constants.py`), the second
block is the service-endpoint contract that the application modules are wired to.

| Key | Example | Consumer |
| --- | --- | --- |
| `ENVIRONMENT` | `production` | application |
| `APP_VERSION` | `1.0.0` | application |
| `LOG_LEVEL` | `INFO` | application |
| `CORS_ORIGINS` | `["https://mvm.example.com"]` | API |
| `OWL2_MODEL_PATH` | `/app/models/owl2_model.onnx` | `src/owl/inference/config.py` |
| `OWL2_EXECUTION_PROVIDER` | `CUDAExecutionProvider` | `src/owl/inference/config.py` |
| `OWL2_MAX_BATCH_SIZE` | `32` | `src/owl/inference/config.py` |
| `CELERY_BROKER_URL` | `amqp://user:pass@rabbitmq:5672//` | `src/tasks/celery.py` |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/0` | `src/tasks/celery.py` |
| `CELERY_QUEUES` | `vision,vision.priority,vision.results,vision.dlq` | `src/tasks/celery.py` |
| `DATABASE_URL` | `postgresql+psycopg://user:pass@postgres:5432/cogni_db` | API / Alembic |
| `RABBITMQ_HOST` / `RABBITMQ_PORT` | `rabbitmq` / `5672` | application |
| `REDIS_HOST` / `REDIS_PORT` | `redis` / `6379` | application |
| `MINIO_ENDPOINT` | `minio:9000` | application |
| `MINIO_SECURE` | `false` | application |
| `MINIO_BUCKET_FRAMES` / `MINIO_BUCKET_RESULTS` | `frames` / `results` | application |
| `CAMERA_STREAMS` | `main:rtsp://…` , comma-separated | camera |
| `CAMERA_IP` / `CAMERA_PORT` / `DEFAULT_FPS` | `192.168.1.10` / `554` / `30` | camera |
| `RTSP_TRANSPORT` | `tcp` (or `udp`) | `src/camera/capture.py` |
| `CAMERA_REGISTRY_WATCH` | `1` | capture pod (`src/api/runtime.py`) |
| `STREAM_FPS` / `STREAM_IDLE_TIMEOUT` | `10` / `15` | `src/api/stream.py` (preview) |
| `TRACKING_ENABLED` / `TRACKER_TYPE` / `TRACK_FPS` | `1` / `kcf` / `10` | `src/vision/tracking.py` |
| `PIPELINE_DISPATCH_INTERVAL_SECONDS` | `0` (dev `10`) | `src/pipeline/runner.py` (sampling) |
| `PRESENCE_ENABLED` / `PRESENCE_ROI` / `PRESENCE_MIN_AREA` / `PRESENCE_MAX_AREA` / `PRESENCE_MIN_ASPECT` / `PRESENCE_MAX_ASPECT` / `PRESENCE_MIN_CONFIDENCE` / `PRESENCE_MIN_INSIDE` | `0` / `0.15,0.33,0.85,0.85` / `0.02` / `0.20` / `1.0` / `2.6` / `0.10` / `0.5` | `src/vision/presence.py` (station answer) |
| `PERSIST_DETECTIONS` / `PERSIST_ONLY_DETECTIONS` | `1` / `1` | `src/api/detection_sink.py` |
| `PERSIST_MIN_INTERVAL_SECONDS` | `1.0` | `src/api/detection_sink.py` |
| `DETECTION_RETENTION_DAYS` | `30` | retention CronJob |
| `CELERY_QUEUE_MAX_LENGTH` / `CELERY_QUEUE_TTL_SECONDS` | `256` / `60` | `src/tasks/celery.py` |
| `MODBUS_INTERFACE` | `/dev/ttyUSB0` | robot |
| `MODBUS_BAUD_RATE` | `9600` | robot |
| `MODBUS_SLAVE_ID` | `1` | robot |
| `MODBUS_HOST` / `MODBUS_PORT` | *(empty)* / `502` | robot (TCP fallback) |
| `PROMETHEUS_MULTIPROC_DIR` | `/tmp/prometheus` | metrics |

`RABBITMQ_URL`, `MINIO_ACCESS_KEY`/`MINIO_SECRET_KEY`, `POSTGRES_*` and `REDIS_PASSWORD` are **not** in the
ConfigMap — credentials live in the Secret (`§5`).

> Migration note: the deleted `k8s/configmap.yaml` declared `app-config` with `CAMERA_IP`, `CAMERA_PORT`,
> `DEFAULT_FPS` and `SYSTEM_STATUS_MESSAGE`. Those keys (minus the unused status string) are preserved above,
> but the object was renamed to `industrial-cognition-config` because that is the name every Deployment
> already referenced.

---

## 4. Storage

| Volume | Backing | Size (prod) | Mounted by |
| --- | --- | --- | --- |
| `postgres-storage` | `volumeClaimTemplates` (StatefulSet) | 20Gi | postgres |
| `rabbitmq-data` | `volumeClaimTemplates` | 20Gi | rabbitmq |
| `redis-data` | `volumeClaimTemplates` | 10Gi | redis |
| `minio-data` | `volumeClaimTemplates` | 100Gi | minio |
| `industrial-cognition-model-cache` | standalone PVC (`ReadWriteOnce`) | 6Gi | vision (`/app/models`), camera (`/app/models`, read-only) |
| `prometheus-storage` | `volumeClaimTemplates` (monitoring component) | 20Gi | prometheus |
| `grafana-storage` | `volumeClaimTemplates` (monitoring component) | 5Gi | grafana |
| `loki-storage` | `volumeClaimTemplates` (monitoring component) | 20Gi | loki |

Stateful components use `volumeClaimTemplates` so the PVC is bound to the pod identity and survives
rescheduling — the previous manifests used bare `PersistentVolumeClaim` objects for Deployments, which do not
survive a move between nodes and cannot be scaled.

### `storageClassName`: omit it, never set it to `""`

The base manifests **omit** `storageClassName` so the cluster's default StorageClass provisions the volume.
That is a load-bearing detail, not a style choice, and it cost a real debugging cycle on a live cluster:

> Setting `storageClassName: ""` does **not** mean "use the default". It means "no class", which disables
> dynamic provisioning entirely. Every claim then sits `Pending` forever with
> `FailedBinding: no persistent volumes available for this claim and no storage class is set`, and because the
> pods mount those claims, `postgres`, `rabbitmq`, `redis`, `minio` and the vision worker never schedule at
> all.

`tests/k8s` asserts this (`test_no_claim_disables_dynamic_provisioning`) so it cannot regress. The `prod`
overlay adds the field explicitly to pin a class — e.g. `fast-ssd` for PostgreSQL/RabbitMQ/Redis and `bulk`
for MinIO — and those names must be adjusted to the target cluster's own classes.

Recovering a cluster that was already deployed with the empty string: the field is **immutable** on both a
`PersistentVolumeClaim` and a StatefulSet's `volumeClaimTemplates`, so neither can be patched in place. The
workload has to be recreated:

```sh
# 1. delete the StatefulSets FIRST, orphaning their pods so the kubelet releases
#    the claims instead of fighting the controller
kubectl -n cogni-ns delete statefulset postgres rabbitmq redis minio --cascade=orphan
# 2. drain the pods, then the claims
kubectl -n cogni-ns delete pods --all
kubectl -n cogni-ns delete pvc --all          # retry if one is stuck Terminating
# 3. recreate from the corrected manifests
kubectl apply -k k8s/overlays/dev
```

Deleting a claim *while its StatefulSet still exists* leaves the controller unable to recreate it, and a claim
still mounted by a running pod stays `Terminating` indefinitely — which is why the order above matters.

### The OWLv2 model

`src/owl/models/**` is **git-ignored** (`src/owl/models/cuda/owl2_model.onnx` is a ~300 MB artifact), so the
model can never be part of the repository manifest set. Choose one of:

1. **Baked into the image** (simplest, KISS): the build pipeline copies the model into
   `industrial-cognition/vision` and no `model-cache` PVC is needed. Remove the `model-cache` volume from
   `deployment-vision.yaml`.
2. **Populated PVC** (default here): upload the model once into the `industrial-cognition-model-cache` PVC
   (`kubectl cp` via a helper pod, or `mc cp` from MinIO), then the read-only mount is enough.

Without either, the vision worker starts but `VisionService.load_model()` finds no artifact. The worker is a
Celery consumer, so it stays `Running` and only fails when a task arrives — check
`kubectl -n cogni-ns exec deploy/vision -- ls -l /app/models` after a model rollout.

---

## 5. Secrets

`k8s/base/secrets.env` is consumed by Kustomize's `secretGenerator` to produce the Secret
`industrial-cognition-secrets` with the keys `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`,
`RABBITMQ_DEFAULT_USER`, `RABBITMQ_DEFAULT_PASS`, `REDIS_PASSWORD`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`.

It ships **development defaults** so that a fresh clone renders and deploys without a manual step. For any
real installation, override it without editing the committed file:

```sh
# 1. copy the template and fill in real values
cp k8s/base/secrets.env k8s/overlays/prod/secrets.env
$EDITOR k8s/overlays/prod/secrets.env

# 2. add to k8s/overlays/prod/kustomization.yaml
secretGenerator:
  - name: industrial-cognition-secrets
    envs: [secrets.env]
    options:
      disableNameSuffixHash: true
```

`k8s/overlays/prod/secrets.env` is covered by `.gitignore`. For a managed-secrets workflow, replace the
generator with an `ExternalSecret` (External Secrets Operator) or a CSI `SecretProviderClass`; the
Deployments' `secretKeyRef`s do not change.

Rotating a credential means editing the env file and re-running
`./k8s/scripts/deploy.sh --overlay prod`, which reapplies the Secret; restart the affected pods with
`kubectl -n cogni-ns rollout restart deployment/vision` (or `statefulset/rabbitmq`).

---

## 6. Networking

* Every component gets a `ClusterIP` Service named after the component hostname
  (`api`, `camera`, `vision`, `robot`, `rabbitmq`, `redis`, `minio`, `postgres`) on port **80** (HTTP) or its
  native port, so in-cluster URLs are stable regardless of container ports.
* `rabbitmq` and `minio` additionally get headless (`clusterIP: None`) discovery Services for StatefulSet
  peer discovery and MinIO's distributed mode.
* `Ingress` (`networking.k8s.io/v1`, class `nginx`) terminates HTTPS and routes:
  * `/` → `api:80`
  * `/ws/` → `api:80` (annotated with a long proxy read timeout for the WebSocket telemetry stream)
  * `/grafana` → `grafana:80` (monitoring component only)
  * `/minio` → `minio:9001` (console, optional)
* `NetworkPolicy`: the namespace is **default-deny** for ingress and egress. Explicit `allow-*` policies then
  permit only DNS egress, ingress-controller → `api`, Prometheus → all scrape targets, application → data
  services, `rabbitmq` ↔ `rabbitmq` (cluster), `minio` ↔ `minio` (distributed), and the RTSP/Modbus egress
  the camera and robot need. Anything not listed is dropped.

WebSocket support requires the ingress controller not to buffer SSE/WS traffic — hence
`nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"` and `proxy-buffering: "off"`.

---

## 7. Scaling

Scaling objects live in `k8s/overlays/prod/` rather than in `base`, because an HPA is inert (and its
`minReplicas` actively harmful) on a cluster without a metrics-server — which is exactly the `dev` overlay's
environment.

* `HorizontalPodAutoscaler` (`autoscaling/v2`) objects exist for `api` (2→4) and `camera` (1→2) in
  `autoscaling.yaml`, and for `vision` (2→4) in `hpa-vision-cpu.yaml`. `minReplicas` equals the Deployment's
  static `replicas`, so adding an HPA never changes the pod count on its own.
* `vision` gets its own file because it is the one workload with **two mutually exclusive scaling
  strategies**, and exactly one of them may be enabled at a time — two controllers writing `spec.replicas`
  flips the deployment every few seconds:

  | Mode | `resources:` | `components:` |
  | --- | --- | --- |
  | CPU utilisation (default) | `hpa-vision-cpu.yaml` | `../../components/monitoring` |
  | **Queue depth** (KEDA) | *(drop the line above)* | `../../components/monitoring`, `../../components/keda` |

* **Queue depth** is the metric the architecture actually cares about: the workers sit idle waiting on
  RabbitMQ while the backlog grows, so CPU utilisation is a late and noisy signal.
  `k8s/components/keda/scaledobject.yaml` contains a ready `ScaledObject` (`keda.sh/v1alpha1`) plus a
  `TriggerAuthentication` that reads the AMQP URL (`RABBITMQ_URL`) from the same Secret the workers use. It
  scales at 20 ready messages on `vision` and 5 on `vision.priority`, keeps one warm worker
  (`cooldownPeriod: 300`) and scales down one pod per 5 minutes to avoid dropping a warm ONNX session.
  It is opt-in because it requires the KEDA operator and its CRDs — without them the apply fails outright.

  > The component cannot delete the CPU HPA for you: Kustomize components are additive, and a component
  > patch cannot remove an object contributed by the kustomization that includes it. Hence the swap is a
  > `resources:` edit rather than a `$patch: delete`, and `tests/k8s` asserts that exactly one of the two
  > files is referenced so the two can never both ship.
* `PodDisruptionBudget` objects protect `api`, `vision`, `rabbitmq` and `minio` during voluntary disruptions.
* `robot` is deliberately a singleton with `strategy: Recreate`; a second replica would corrupt Modbus
  framing on the shared RS-485 bus. It has no HPA and no PDB — a `minAvailable: 1` budget on a singleton
  would block node drains forever without adding availability.
* `ResourceQuota` + `LimitRange` bound the namespace so a runaway worker cannot starve the GPU node.

---

## 8. Scheduling and GPUs

* GPU nodes are selected with the standard NVIDIA device-plugin labels:
  `nodeSelector: { nvidia.com/gpu.present: "true" }` plus a `nvidia.com/gpu` resource request, and an
  optional `toleration` for `nvidia.com/gpu:NoSchedule`. Node-pool identity can also be pinned with the
  `node-role.kubernetes.io/gpu` label — both are in `deployment-vision.yaml`, controlled by the
  `gpu.enabled` / `gpu.count` fields which the overlays patch.
* CPU-only clusters: set `gpu.enabled: false` in the overlay. The pod then uses
  `OWL2_EXECUTION_PROVIDER=CPUExecutionProvider` (the ConfigMap default) and no GPU scheduling is applied.
* `PriorityClass`es: `industrial-cognition-critical` (data services, priority 1000000) and
  `industrial-cognition-standard` (default, 1000) so the database and broker are evicted last.
* `topologySpreadConstraints` keep the 2 `api` and 2 `vision` replicas on different nodes.
* The camera and robot pods tolerate edge nodes; the robot pod's `MODBUS_INTERFACE` device must exist on the
  node it lands on (`nodeSelector: industrial-cognition/role: edge`).

---

## 9. Observability

Enabled by adding the `monitoring` component to an overlay:

```yaml
components:
  - ../../components/monitoring
```

It deploys Prometheus (scraping itself, `api`, `camera`, `vision`, `robot`, `redis`, `rabbitmq`, `minio`,
`node-exporter`, `kubernetes-pods` via annotation discovery), Grafana (with a provisioned Prometheus data
source and a "Industrial Cognition / Overview" dashboard), Loki (single-binary, filesystem storage) and
Alloy (DaemonSet log collector shipping to Loki). Dashboards live in
`k8s/components/monitoring/dashboards/` and are loaded via `configMapGenerator` + the Grafana sidecar-free
provisioning path, so no restart is needed when a dashboard changes as long as the ConfigMap hash changes
(which Kustomize guarantees).

Application metrics come from the `prometheus.io/scrape`, `prometheus.io/port` and `prometheus.io/path`
annotations that every Deployment already carries; no Prometheus Operator CRDs are required.

---

## 10. Operations

### Prerequisites

* `kubectl` ≥ 1.24 (Kustomize v4+ built in). `kubectl version --client` must succeed.
* A storage class able to satisfy the PVCs in §4, **or** the `dev` overlay (which shrinks the requests).
* An ingress controller implementing `networking.k8s.io/v1` (nginx-ingress by default).
* For GPU inference: the NVIDIA device plugin + container toolkit on the GPU nodes.
* For queue-depth autoscaling: the KEDA operator.

### Install

```sh
# Render and review first (never apply blind)
kubectl kustomize k8s/overlays/prod | less

# Deploy, then wait for rollouts and print access info
./k8s/scripts/deploy.sh --overlay prod

# Local/CI smoke deployment (single-node data services, monitoring off)
./k8s/scripts/deploy.sh --overlay dev
```

`deploy.sh` performs: `kubectl apply -k` (server-side), `kubectl rollout status` for every workload,
`kubectl wait --for=condition=Ready pod` for the data services, then prints the API, Grafana and MinIO
endpoints together with the `kubectl port-forward` commands for in-cluster-only services.

### Validate

```sh
./k8s/scripts/validate.sh --overlay dev   # kustomize render + kubeconform/schema checks
pytest tests/k8s -q                       # repo-native manifest contract tests
```

`validate.sh` renders every overlay and, when available, validates it with `kubeconform` (or `kubeval`);
otherwise it falls back to `kubectl apply --dry-run=client`, then to the pure-Python checks in
`tests/k8s/`. It is wired into `.github/workflows/k8s/validate.yaml` so a malformed manifest cannot be merged.

### Upgrade

```sh
# Roll a new application image
./k8s/scripts/deploy.sh --overlay prod --image-tag "$(git rev-parse --short HEAD)"

# Or pin new tags directly in the overlay:
#   images:
#     - name: industrial-cognition/api
#       newTag: 1.4.2
```

Rollouts are `RollingUpdate` with `maxSurge: 1, maxUnavailable: 0` for `api`, `vision` and `camera`, so a
deploy never dips below the ready replica count. `robot` uses `Recreate`.

### Database migrations

Alembic migrations run as a `Job` created from the API image, not from the API pods (so that N replicas do
not race):

```sh
kubectl -n cogni-ns delete job alembic-upgrade --ignore-not-found
kubectl -n cogni-ns create job alembic-upgrade \
  --image=industrial-cognition/api:latest \
  -- alembic upgrade head
kubectl -n cogni-ns wait --for=condition=complete job/alembic-upgrade --timeout=5m
```

`deploy.sh --migrate` performs exactly this after the data services report Ready. `alembic.ini` and
`alembic/env.py` are currently empty stubs, so the flag is a no-op-safe wrapper until
`doc/tasklist.md` Stage 1 "Database Schema Migration" lands.

### Uninstall

```sh
./k8s/scripts/teardown.sh --overlay prod          # deletes workloads, keeps PVCs
./k8s/scripts/teardown.sh --overlay prod --purge  # also deletes PVCs and the namespace
```

### Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `CreateContainerConfigError: secret "industrial-cognition-secrets" not found` | Applied `k8s/base/*.yaml` with `kubectl apply -f` instead of Kustomize, so the generated Secret/ConfigMap were skipped. | Always `kubectl apply -k <overlay>`. |
| `api` pods `CrashLoopBackOff`, log shows `Failed to connect to Modbus Controller` | The API bootstraps camera/vision/Modbus in-process and exits if the RS-485 bus is absent. | Provide the serial device (robot node) or run the API with `MODBUS_TRANSPORT=tcp`; see `doc/tasklist.md` Stage 2. |
| Vision pods `Pending`, event `0/3 nodes available: Insufficient nvidia.com/gpu` | No GPU node or device plugin. | Deploy the NVIDIA device plugin or set `gpu.enabled: false`. |
| Ingress returns 502 on `/ws/detections` | Proxy buffering / short read timeout. | Keep the `proxy-buffering: "off"` + 3600 s annotations from `base/ingress.yaml`. |
| `minio` pod stuck `Init:0/1` in the base overlay | Base assumes 4-node distributed MinIO; a single node cannot satisfy the quorum. | Use `overlays/dev` (1 server) or provide 4 nodes. |
| RabbitMQ pod `CrashLoopBackOff` with `nodedown` | Erlang cookie mismatch between replicas. | The StatefulSet shares the cookie via the `rabbitmq-erlang-cookie` Secret; delete stuck pods to force a re-join. |

---

## 11. Validation

`tests/k8s/test_manifests.py` renders every overlay (using `kubectl kustomize` when present, otherwise a
minimal pure-Python renderer) and asserts the invariants that were violated by the old manifests:

* every `configMapKeyRef` / `secretKeyRef` / `envFrom` target exists in the rendered set;
* no object name is declared twice inside one overlay;
* no `Service` selector matches zero workloads, and every workload's ports are backed by a `containerPort`;
* no placeholder tokens (`your-registry`, `<REPLACE`, `changeme`) remain anywhere;
* every workload declares resources, probes (except the migration job and the log collector DaemonSet) and a
  non-root `securityContext`;
* `robot` is a singleton with `Recreate`, and `postgres`/`rabbitmq`/`redis`/`minio` are StatefulSets with
  `volumeClaimTemplates` (the regression guards for this specific rewrite).

---

## Live run: minikube (2026-10-17)

First full deployment of Stages 1-2 to a real cluster. Cluster: minikube v1.39.0,
driver `docker`, Kubernetes v1.37.0, 8 vCPU / 10 GiB, containerd runtime.
`kubectl apply -k k8s/overlays/dev` created 35 objects; all pods reached `Running`
(api x2, camera, robot, vision, postgres, rabbitmq, redis, minio) and the
`minio-bucket-init` Job completed.

**Verified working**

* `/health` returns 200 with `pipeline_running: false` and
  `reason: capture source could not be opened`. The deliberate non-fatal startup is
  what makes this possible: the container has no `/dev/video0`, and the API still
  serves. The previous bootstrap raised here, which is why the pod used to
  CrashLoopBackOff.
* Service discovery from inside a pod: `postgres:5432`, `rabbitmq:5672`,
  `redis:6379`, `minio:9000` all reachable, and the DSN/broker URLs from the
  generated Secrets resolve.
* Alembic created `detection_results`, `system_metadata` and `alembic_version`.
* The prompt repository round trip against real Postgres: store, load, list, upsert
  (one row, not two) and delete, via `scripts/smoke_cluster.py`.
* HTTP error mapping: 422 for an invalid prompt, 503 when the image has no
  tokenizer, 404 when unset.
* `WS /ws/detections` answers `{"type":"unavailable"}` and closes when capture is
  not running.
* End-to-end task path: a JPEG produced in an api pod was sent through RabbitMQ,
  analysed by the `vision` worker (synthetic backend) and the result returned
  through Redis - 2 detections, 0.30 s round trip, `preprocess_ms` 18.0,
  `inference_ms` 0.96.

**Bugs this run found (all fixed)**

1. `alembic.ini` and `alembic/` were not copied into the API image, so the migration
   Job from `deploy.sh --migrate` could never have run. The image built fine and the
   failure only appeared on the first `alembic upgrade head`.
2. `requirements/*.txt` pinned `serial==0.0.97`, which is an unrelated PyPI package,
   not pyserial; `import serial; serial.Serial` would fail at runtime. Now
   `pyserial==3.5`.
3. The API image lacked OpenCV, Celery and Redis, yet `deployment-camera` and
   `deployment-robot` reuse it - so capture and enqueueing could not work at all.
4. `build-images.sh` re-tagged the API image as the vision image ("identical
   content"). They are not identical, and the re-tag yields a worker with no
   `celery` binary. It now builds `src/vision/Dockerfile.worker`.
5. `requirements/vision.txt` installed the *export* stack
   (`optimum-onnx`, `onnxscript`), pulling PyTorch into a serving image; the build
   never finished. Serving needs only `onnxruntime-gpu`, and export moves to
   `requirements/export.txt`.
6. `encode_prompt` wrapped `TokenizerUnavailable` in `PromptEncodingError`, so a
   deployment without a tokenizer answered 422 ("your prompt is invalid") instead of
   503. Now the two are distinguished, with a regression test.

**Operational notes for Windows workstations**

* `minikube docker-env` does not work with the containerd runtime on Windows
  (`SSH_AGENT_START: starting an SSH agent on Windows is not yet supported`). Build
  with the default Docker daemon and load the image with
  `minikube image load <ref>`; `build-images.sh --load-minikube` already does this.
* The dev overlay leaves the nginx ingress controller uninstalled, so
  `mvm.example.com` is not reachable until `minikube addons enable ingress`.
  Reach the API with `kubectl -n cogni-ns port-forward svc/api 8000:80` or from
  inside a pod.
* Migrations are not automatic: run
  `kubectl -n cogni-ns exec deploy/api -- alembic upgrade head` (or
  `deploy.sh --migrate`).

**Not exercised**

The real OWLv2 model (the `model-cache` PVC is empty and there is no GPU in the
cluster), real camera capture, and the Modbus transports. The `vision` worker ran
with `VISION_BACKEND=synthetic` for the task smoke test and was returned to the
`owl2` default afterwards.

### Second pass: the real model on the PVC

The `model-cache` PVC was populated from a workstation with a temporary
read-write pod (`k8s/tools/model-loader.yaml`; the Deployment mounts it readOnly, which
is correct) and the real 293 MiB artifact was exercised in-cluster:

* the backend loaded in 11.4-11.8 s and ran on `CPUExecutionProvider`;
* one inference took 26.8 s through Celery at the 2 CPU limit, and 7.5-7.8 s in a
  6 CPU probe pod;
* peak RSS for one session was **3432 MiB**, so the dev 4 GiB limit was too tight -
  a second session in the same pod was OOMKilled (exit 137). Raised to 6 GiB;
* logits came back finite (-41.8 to -4.8), so the graph is not in the -65504
  saturation state, and the top decoded box matched the drawn rectangle almost
  exactly, which validates the geometry path.

A seventh bug surfaced here: `build_backend()` never loaded the text query from
`OWL2_TEXT_IDS_PATH`. It passed `text_query=None` together with `VISION_LABELS` and
tripped its own "labels= requires a text_query" check, so **every real task failed** -
while all unit tests passed, because they inject a `TextQuery` explicitly. The
deployed backend was unconstructible until the first live task proved it.

Detection quality is still unverified: the prompt on that PVC is a placeholder
(`k8s/dev/text_queries.placeholder.json`), so scores are ~0.008 and nothing clears the
threshold. That is the expected result of a meaningless prompt, not a pipeline fault;
`doc/vision-pipeline.md` §13 has the numbers.

---

## Cluster doctor: when the cluster is up but unusable

A minikube cluster can start successfully and still be unable to host this stack. The
message to recognise:

```
Enabling 'storage-provisioner' returned an error: running callbacks:
  failed to download openapi: Get "https://localhost:8443/openapi/v2": dial tcp [::1]:8443: connect: connection refused
Enabling 'default-storageclass' returned an error: ... same openapi error ...
```

`minikube start` exited 0, the addons did not apply, and the cluster now has **no default
StorageClass**. Nothing that needs a volume can start: every `PersistentVolumeClaim` stays
`Pending`, so `postgres-0` never schedules (`pod does not have a host assigned`), and the
install fails much later with a scheduling error that points nowhere near the cause.

The `connection refused` on `:8443` is the API server inside the node not answering while
minikube applied its addons - almost always because the node is too small for the control
plane plus the addons.

### Diagnose and repair

```powershell
./scripts/cluster-doctor.ps1        # diagnose
./scripts/cluster-doctor.ps1 -Fix   # re-enable the addons once the API server is up
```

It checks in dependency order - minikube host, API server from the host, node `Ready`,
default StorageClass - and only then attempts the addons, so a repair happens *after* the
condition that broke them has passed. The installer runs it automatically (`install.ps1`
calls `-Fix`; `install.sh` performs the same StorageClass check), so a fresh install fails
at the real cause rather than three steps downstream.

If it still reports no StorageClass, the node needs more room:

```sh
minikube delete
minikube start --driver=docker --memory=12288 --cpus=8
./scripts/cluster-doctor.ps1 -Fix
```

A note on why this is a resource problem and not a code one: the stack runs Postgres,
RabbitMQ, Redis, MinIO, the API, the vision worker and (in a full run) a 293 MiB ONNX
model in one VM. 8 GiB is the practical floor; 12 GiB is comfortable.

---

## Queue hygiene: bounds, TTL and a real dead-letter queue

The dispatcher sheds frames when *its own* in-flight window is full, but that bounds one
process. With no consumer at all - every replica busy, a crash loop, a scale-to-zero - the
broker keeps accepting and the queue grows without limit. Measured cost of getting this wrong:
a 41-message backlog produced `queue_ms` of 353 s, and none of those frames was worth
analysing by the time it was.

`src/tasks/celery.py` now declares both work queues with:

| Argument | Value | Why |
| --- | --- | --- |
| `x-max-length` | 256 | keeps the newest frames |
| `x-overflow` | `drop-head` | discards the *oldest* when full - for a control loop the newest frame is the one worth analysing |
| `x-message-ttl` | 60 s | a frame queued longer than that is stale by definition |
| `x-dead-letter-exchange` | `vision.dlx` | so expiry is visible, not silent |

**Queue arguments are immutable.** Adding them does nothing to an existing queue, and a
worker that tries to redeclare it fails with `PRECONDITION_FAILED` (406). The queues must be
deleted once so consumers recreate them:

```sh
./scripts/queue-hygiene.sh            # report the current arguments
./scripts/queue-hygiene.sh --force    # delete; workers recreate on restart
```

The script is destructive on purpose and refuses without `--force`: deleting the work queue
discards whatever is queued. For this pipeline that is acceptable - queued frames are stale by
design - but it is a human decision, not something an installer should do silently.

### The dead-letter queue had to be declared explicitly

`x-dead-letter-exchange` on a queue does **not** create the target. RabbitMQ accepts the
argument and then drops messages when it dead-letters into a missing exchange - a silent
failure, which is the worst kind for a queue whose purpose is to make losses visible. Celery
also never declares a queue it does not consume from. A `worker_ready` handler now declares
`vision.dlx` / `vision.dlq` explicitly and logs it.

Verified live: with no consumer, a published frame left `vision` at its 60 s TTL and
**`vision.dlq` went from 0 to 3** (`vision` 2 -> 0). Expired work is now retained where it can
be counted instead of evaporating.

## Credentials are generated, not committed

`k8s/base/secrets.env` used to be tracked, with the comment that it "ships development
defaults". It also held `AUTH_TOKEN`, the single token that opens the operator console: a
committed secret has been in every clone and every backup, and rotating it does not undo that.

Now:

* `k8s/base/secrets.env.example` is the tracked template.
* `k8s/base/secrets.env` is git-ignored and created by `install.ps1` / `install.sh` from that
  template with **random** values for every password and token, keeping identifiers (`*_USER`,
  `*_DB`, `*_NAME`) stable because they appear in the manifests.
* Generation happens only when the file is missing. Re-running the installer must not rotate an
  existing file: changing `POSTGRES_PASSWORD` in the env does not change the password inside an
  initialised database, so rotating it unattended breaks the cluster instead of securing it.
  Rotation is therefore an operator action, and it has to move both sides at once.

A fresh clone that runs `kubectl apply -k` without the installer now fails with a missing
`secrets.env` - deliberately, and with the installer as the documented path.

---

## Scheduled maintenance

`cronjob-retention.yaml` is the only CronJob: it prunes detection history daily
(`python -m src.api.maintenance.retention`, `DETECTION_RETENTION_DAYS` default 30). It runs the
API image - that image already carries SQLAlchemy, the driver and the module, so no new artifact
is needed and the code under test is the code that runs.

Choices worth knowing: `concurrencyPolicy: Forbid` so a slow prune cannot race the next one;
`startingDeadlineSeconds: 3600` so a missed window is skipped rather than piled up;
`ttlSecondsAfterFinished: 86400` so daily pods do not accumulate; and the usual restricted
security context (`runAsNonRoot`, read-only root, `/tmp` emptyDir).

## CI

`.github/workflows/ci.yml` runs two jobs on every push and PR:

* **unit** - `pytest tests` with the API requirement set (not `base.txt`, which pulls a GPU
  stack), a compile-all pass, and *syntax checks for the PowerShell and shell entry points*.
  Those scripts are what users actually run; a parse error there is invisible until it is hit.
* **manifests** - renders dev, prod and tools with `kubectl kustomize`, validates them with
  `kubeconform -strict`, then runs the manifest contract tests.

A third job, **cluster**, is manual (`workflow_dispatch`): it builds both images, starts kind,
deploys, migrates through a Job, checks `/readyz` and runs `scripts/smoke_auth.py` against the real
surface. It is gated because it takes minutes, not seconds - but it is the only job that exercises
the deployment instead of the code.

Three things it deliberately does differently from the installer, each for a reason:

* **The vision workers are scaled to zero after `kubectl apply`.** The ONNX artifact is not in the
  repository, so the workers would crash-loop on a missing model while holding a 2 GiB request
  each - more than a hosted runner has to spare alongside postgres, rabbitmq, redis, minio and the
  API. Inference is covered by the `model`-marked tests, not by this job. Letting the pods fail and
  still calling the run green would be worse than saying what is tested.
* **The migrate Job is patched to a restricted security context.** `kubectl create job` emits a Pod
  with none, and the namespace only *warns* on restricted policy, so an uncompliant Job passes here
  and is rejected on a cluster that enforces it - which is the production configuration. The step
  also sets `backoffLimit: 1`: one retry for a transient connection race, rather than six passes at
  a migration that is genuinely broken.
* **`scripts/smoke_auth.py` is streamed over stdin**, not copied in. `kubectl cp` would work today
  (the container has `tar`, and `/tmp` is an emptyDir) but it depends on both of those; stdin
  depends on neither, and the script is standalone.

The kind cluster also differs from minikube in image loading (`kind load docker-image` versus
`minikube image load`), which is the one part of this job that cannot be rehearsed on a workstation
with minikube.

---

## TLS on-premise: an internal CA, no internet

The MVP runs on a factory network with no route out. That rules out Let's Encrypt (needs a
publicly reachable name) and makes cert-manager a mirroring exercise. Neither is necessary: an
internal CA needs one machine with openssl, once per rotation.

```
scripts/tls-selfsigned.sh --host mvm.example.com     # POSIX
scripts/tls-selfsigned.ps1 -HostName mvm.example.com  # Windows
```

It creates two Secrets:

| Secret | Contents | Purpose |
| --- | --- | --- |
| `industrial-cognition-tls` | `tls.crt`, `tls.key` | what the ingress serves (name already referenced by `ingress.yaml`) |
| `industrial-cognition-tls-ca` | `ca.crt` | the trust anchor clients import |

**Why a CA rather than a self-signed leaf.** A self-signed server certificate has to be trusted
individually, so every rotation re-teaches every client. With a CA, operators import `ca.crt`
once and renewals are invisible. Re-running the script rotates the leaf; only a compromised CA
key requires re-import.

The script runs openssl in a container when the host has none (a default Windows box does not),
and generates the private key **on the machine that runs it** - the CA key is never written
outside a temporary directory.

### Verified

Against the live cluster, with the ingress controller and the certificate in place:

```
curl --cacert ca.crt --resolve mvm.example.com:443:<ingress-ip> https://mvm.example.com/health
  -> HTTP/2 200, ssl_verify_result=0
  -> subject CN=mvm.example.com; issuer CN=IndustrialCognitionInternalCA
  -> without --cacert: exit 60 (certificate verify failed)
```

The last line matters as much as the first: it proves the certificate is ours and not something a
public CA would already have vouched for.

### If you do run cert-manager

`k8s/components/tls/cert-manager` holds the equivalent chain (selfsigned ClusterIssuer -> CA
Certificate -> CA Issuer -> server Certificate) for sites that already operate cert-manager and
would rather it renew on a schedule. It is **not** referenced by any overlay, because applying it
without the cert-manager CRDs fails; it is an explicit choice. Air-gapped sites must mirror
cert-manager's images and apply its release manifest from a local copy.

### Not done, deliberately

* **mTLS between pods.** It would mean issuing a certificate per workload and enforcing
  `PeerAuthentication`-equivalent policy (a service mesh, or cert-manager plus NetworkPolicy).
  Worth doing once the cluster has a CNI that can enforce it; today the namespace has no
  NetworkPolicies at all, so mTLS would protect against a threat the network does not yet
  restrict.
* **A secrets manager.** `secrets.env` is generated per checkout and git-ignored, which is
  adequate for one cluster. Vault or OpenBao becomes worthwhile when several environments share
  credentials and rotation has to be auditable.
