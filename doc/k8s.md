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
