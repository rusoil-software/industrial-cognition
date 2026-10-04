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
# date   : 2026-Oct-4
# ==============================================================================
#
# Tear down the Industrial Cognition stack.
#
#   ./k8s/scripts/teardown.sh --overlay dev            # keep data
#   ./k8s/scripts/teardown.sh --overlay dev --purge    # delete PVCs too
#
# Without --purge, PersistentVolumeClaims survive: the database, the broker
# state and the object store all outlive a redeploy, which is the safe default.
# With --purge the namespace is deleted outright and every volume with it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OVERLAY="dev"
NAMESPACE="cogni-ns"
PURGE=0
ASSUME_YES=0

usage() {
  cat <<'EOF'
Usage: teardown.sh [options]

Options:
  -o, --overlay <dev|prod>   Overlay whose objects should be removed (default: dev)
  -n, --namespace <name>     Target namespace (default: cogni-ns)
      --purge                Also delete PersistentVolumeClaims and the namespace
  -y, --yes                  Do not prompt for confirmation
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
    --purge)        PURGE=1; shift ;;
    -y|--yes)       ASSUME_YES=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

OVERLAY_DIR="${REPO_ROOT}/k8s/overlays/${OVERLAY}"
[[ -d "${OVERLAY_DIR}" ]] || die "overlay not found: ${OVERLAY_DIR}"
command -v kubectl >/dev/null 2>&1 || die "kubectl is not installed or not on PATH"

if [[ "${PURGE}" -eq 1 ]]; then
  TARGET="the '${NAMESPACE}' namespace INCLUDING every persistent volume"
else
  TARGET="the workloads and configuration in '${NAMESPACE}' (PVCs are kept)"
fi
log "About to delete ${TARGET}"
if [[ "${ASSUME_YES}" -ne 1 ]]; then
  read -r -p "Continue? [y/N] " reply
  [[ "${reply}" =~ ^[Yy]$ ]] || { log "aborted"; exit 0; }
fi

if [[ "${PURGE}" -eq 1 ]]; then
  # Deleting the namespace removes every object in it, including generated
  # Secrets and ConfigMaps that `kubectl delete -k` cannot always resolve.
  kubectl delete namespace "${NAMESPACE}" --ignore-not-found --wait=true
  log "Namespace ${NAMESPACE} deleted"
  exit 0
fi

log "Deleting objects declared by k8s/overlays/${OVERLAY}"
kubectl delete -k "${OVERLAY_DIR}" --ignore-not-found --wait=true

log "Remaining in ${NAMESPACE}:"
kubectl --namespace "${NAMESPACE}" get persistentvolumeclaims,secrets,configmaps || true
cat <<EOF

Persistent volume claims were kept on purpose. Remove them with:
  ./k8s/scripts/teardown.sh --overlay ${OVERLAY} --purge

The cluster-scoped objects created by the base are shared by every overlay
(PriorityClasses and the Alloy ClusterRole), so they are intentionally NOT
removed here. Delete them by hand if this cluster will not host the stack again:
  kubectl delete priorityclass industrial-cognition-critical industrial-cognition-standard
  kubectl delete clusterrole,clusterrolebinding industrial-cognition-alloy
EOF
