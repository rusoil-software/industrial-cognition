#!/usr/bin/env bash

# Copyright (c), The Rusoil Software Development Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# author : Konstantin Ustiuzhanin
<<<<<<< HEAD
# date   : 2026-Oct-4
=======
# date   : 2026-Sep-10
>>>>>>> 9353103d5fc4ac064cdee843ceca45b8eb141917
# ==============================================================================
#
# Deploy the Industrial Cognition stack to Kubernetes.
#
#   ./k8s/scripts/deploy.sh --overlay dev
#   ./k8s/scripts/deploy.sh --overlay prod --image-tag 1.4.2 --migrate
#
# The script is deliberately a thin wrapper around `kubectl apply -k`: Kustomize
# does the composition, this file only sequences the apply, waits for the
# rollouts, optionally runs the Alembic migration Job, and prints how to reach
# the result. See doc/k8s.md §10.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OVERLAY="dev"
NAMESPACE="cogni-ns"
IMAGE_TAG=""
RUN_MIGRATIONS=0
TIMEOUT="600s"

usage() {
  cat <<'EOF'
Usage: deploy.sh [options]

Options:
  -o, --overlay <dev|prod>   Overlay to apply (default: dev)
  -n, --namespace <name>     Target namespace (default: cogni-ns)
  -t, --image-tag <tag>      Override the image tag for every application image
  -m, --migrate              Run the Alembic migration Job after the data
                             services report Ready (no-op while alembic/ is a stub)
      --timeout <duration>   Rollout wait timeout (default: 600s)
  -h, --help                 Show this help
EOF
}

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--overlay)   OVERLAY="${2:?}"; shift 2 ;;
    -n|--namespace) NAMESPACE="${2:?}"; shift 2 ;;
    -t|--image-tag) IMAGE_TAG="${2:?}"; shift 2 ;;
    -m|--migrate)   RUN_MIGRATIONS=1; shift ;;
    --timeout)      TIMEOUT="${2:?}"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

OVERLAY_DIR="${REPO_ROOT}/k8s/overlays/${OVERLAY}"
[[ -d "${OVERLAY_DIR}" ]] || die "overlay not found: ${OVERLAY_DIR}"

command -v kubectl >/dev/null 2>&1 || die "kubectl is not installed or not on PATH"
kubectl version --client >/dev/null 2>&1 || die "kubectl is present but not runnable"

log "Target context: $(kubectl config current-context 2>/dev/null || echo '<none>')"
log "Rendering overlay '${OVERLAY}'"

# Render once so a broken overlay fails before any cluster mutation, and so the
# operator can see exactly how many objects are about to change.
MANIFEST="$(mktemp)"
trap 'rm -f "${MANIFEST}"' EXIT
kubectl kustomize "${OVERLAY_DIR}" >"${MANIFEST}"
log "Rendered $(grep -c '^kind:' "${MANIFEST}") objects"

if [[ -n "${IMAGE_TAG}" ]]; then
  log "Setting every application image tag to '${IMAGE_TAG}'"
  # `kubectl set image` on a not-yet-applied object is a no-op, so the tag is
  # injected into the rendered stream instead: it works on a first install and
  # on an upgrade.
  sed -E "s#(industrial-cognition/(api|vision):)[^\"[:space:]]+#\1${IMAGE_TAG}#g" \
    "${MANIFEST}" >"${MANIFEST}.tagged"
  mv "${MANIFEST}.tagged" "${MANIFEST}"
fi

log "Applying (server-side)"
kubectl apply --server-side --namespace "${NAMESPACE}" -f "${MANIFEST}"

log "Waiting for the data services"
for component in postgres rabbitmq redis minio; do
  if kubectl --namespace "${NAMESPACE}" get "statefulset/${component}" >/dev/null 2>&1; then
    if ! kubectl --namespace "${NAMESPACE}" rollout status "statefulset/${component}" \
      --timeout="${TIMEOUT}"; then
      warn "statefulset/${component} did not become Ready within ${TIMEOUT}"
    fi
  fi
done

log "Waiting for the application workloads"
for workload in deployment/api deployment/camera deployment/vision deployment/robot; do
  if kubectl --namespace "${NAMESPACE}" get "${workload}" >/dev/null 2>&1; then
    if ! kubectl --namespace "${NAMESPACE}" rollout status "${workload}" --timeout="${TIMEOUT}"; then
      warn "${workload} did not finish rolling out within ${TIMEOUT}"
      warn "inspect with: kubectl -n ${NAMESPACE} describe ${workload}"
    fi
  fi
done

if [[ "${RUN_MIGRATIONS}" -eq 1 ]]; then
  if [[ -s "${REPO_ROOT}/alembic.ini" ]]; then
    log "Running Alembic migrations"
    kubectl --namespace "${NAMESPACE}" delete job alembic-upgrade --ignore-not-found
    kubectl --namespace "${NAMESPACE}" create job alembic-upgrade \
      --image="industrial-cognition/api:${IMAGE_TAG:-latest}" \
      -- alembic upgrade head
    kubectl --namespace "${NAMESPACE}" wait --for=condition=complete \
      job/alembic-upgrade --timeout="${TIMEOUT}"
  else
    warn "--migrate requested but alembic.ini is empty; skipping."
    warn "See doc/tasklist.md Stage 1 'Database Schema Migration'."
  fi
fi

log "Deployment summary"
kubectl --namespace "${NAMESPACE}" get pods,services,ingress --output=wide || true

cat <<EOF

Endpoints
  API (in-cluster)   http://api.${NAMESPACE}.svc.cluster.local
  API (local)        kubectl -n ${NAMESPACE} port-forward svc/api 8000:80
                     then: curl http://localhost:8000/health
  WebSocket          ws://localhost:8000/ws/detections
  Grafana (prod)     kubectl -n ${NAMESPACE} port-forward svc/grafana 3000:80
  MinIO console      kubectl -n ${NAMESPACE} port-forward svc/minio 9001:9001
  RabbitMQ UI        kubectl -n ${NAMESPACE} port-forward svc/rabbitmq 15672:15672

Next steps
  * upload the OWLv2 model into the model-cache PVC (doc/k8s.md §4) if the
    vision image does not already contain it
  * ./k8s/scripts/validate.sh --overlay ${OVERLAY}
EOF
