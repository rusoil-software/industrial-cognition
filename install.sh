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
# date   : 2026-Oct-17
# ==============================================================================
#
# Install Industrial Cognition on a local Kubernetes cluster, then open the console.
#
#   1. preflight - docker, minikube, kubectl
#   2. cluster   - start minikube if needed and refresh the kubeconfig
#   3. images    - build both images and load them into the node
#   4. deploy    - kubectl apply -k, wait for rollouts, run migrations as a Job
#   5. console   - print the tokenised URL and open the browser
#
# Deliberately no `kubectl exec` and no port-forward to a Deployment: a Pending pod cannot
# be exec'd into ("does not have a host assigned"), while a Job and a Service port-forward
# both work regardless of which replica is up.
#
# Windows: use install.cmd, which calls install.ps1. A fresh Windows install refuses to
# run unsigned .ps1 files, and the .cmd wrapper passes -ExecutionPolicy Bypass for that
# invocation only.
#
# Usage:
#   ./install.sh [--memory 10240] [--cpus 8] [--overlay dev] [--no-console]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMORY=10240
CPUS=8
OVERLAY=dev
OPEN_CONSOLE=1

usage() { sed -n '30,45p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--memory)    MEMORY="${2:?}"; shift 2 ;;
        -c|--cpus)      CPUS="${2:?}"; shift 2 ;;
        -o|--overlay)   OVERLAY="${2:?}"; shift 2 ;;
        --no-console)   OPEN_CONSOLE=0; shift ;;
        -h|--help)      usage ;;
        *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
done

step() { printf '\n\033[36m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[31merror: %s\033[0m\n' "$1" >&2; [[ -n "${2:-}" ]] && printf '%s\n' "$2" >&2; exit 1; }

cd "${REPO_ROOT}"

step "preflight"
for tool in docker minikube kubectl; do
    command -v "${tool}" >/dev/null 2>&1 || die "${tool} is not installed or not on PATH"
done
docker version --format '{{.Server.Version}}' >/dev/null 2>&1 \
    || die "the Docker engine is not responding. Start Docker Desktop and re-run."
printf '    docker %s\n' "$(docker version --format '{{.Server.Version}}')"

step "cluster (minikube, ${MEMORY} MB, ${CPUS} CPUs)"
if [[ "$(minikube status --format '{{.Host}}' 2>/dev/null || true)" != "Running" ]]; then
    minikube start --driver=docker --memory="${MEMORY}" --cpus="${CPUS}"
else
    printf '    already running\n'
fi
# After a Docker Desktop restart the API server port changes and the kubeconfig keeps
# pointing at the old one, which presents as "the cluster is down".
minikube update-context >/dev/null
kubectl get --raw /healthz --request-timeout=15s >/dev/null 2>&1 \
    || die "the Kubernetes API server is not answering." \
           "  Recreate the cluster: minikube delete && ./install.sh"

# The cluster can be up and still unusable: if minikube's addons failed while the API
# server was starting, there is no default StorageClass, every PVC stays Pending, and the
# install fails later with a misleading scheduling error.
step "checking the default StorageClass"
default_sc="$(kubectl get storageclass -o jsonpath='{range .items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")]}{.metadata.name}{end}' 2>/dev/null || true)"
if [[ -z "${default_sc}" ]]; then
    printf '    none found; re-enabling the storage addons\n'
    minikube addons enable storage-provisioner >/dev/null 2>&1 || true
    minikube addons enable default-storageclass >/dev/null 2>&1 || true
    default_sc="$(kubectl get storageclass -o jsonpath='{range .items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")]}{.metadata.name}{end}' 2>/dev/null || true)"
fi
[[ -n "${default_sc}" ]] || die "no default StorageClass: PersistentVolumeClaims would stay Pending." \
    "  This usually means the node was too small for minikube's addons:
      minikube delete && minikube start --memory=12288 --cpus=8
  then re-run ./install.sh.  ./scripts/cluster-doctor.ps1 -Fix diagnoses this on Windows."
printf '    %s\n' "${default_sc}"

# Credentials are generated, never committed. Generated once per checkout: re-running must
# not rotate an existing file, because the database password inside Postgres would not change
# with it and the pods would simply stop connecting.
ensure_secrets() {
    local target="${REPO_ROOT}/k8s/base/secrets.env"
    local example="${REPO_ROOT}/k8s/base/secrets.env.example"
    if [[ -f "${target}" ]]; then
        printf '    k8s/base/secrets.env exists (left untouched)\n'
        return 0
    fi
    [[ -f "${example}" ]] || die "k8s/base/secrets.env.example is missing"
    local key value
    : > "${target}"
    while IFS= read -r line; do
        if [[ "${line}" =~ ^([A-Z0-9_]+)= ]]; then
            key="${BASH_REMATCH[1]}"
            if [[ "${key}" =~ _(USER|DB|NAME|URL|KEY)$ && ! "${key}" =~ SECRET|PASS|TOKEN|COOKIE ]]; then
                printf '%s\n' "${line}" >> "${target}"
            else
                value="$(head -c 24 /dev/urandom | base64 | tr -dc 'a-zA-Z0-9' | head -c 32)"
                printf '%s=%s\n' "${key}" "${value}" >> "${target}"
            fi
        else
            printf '%s\n' "${line}" >> "${target}"
        fi
    done < "${example}"
    printf '    generated k8s/base/secrets.env with random credentials\n'
}

step "credentials"
ensure_secrets

step "building and loading images"
"${REPO_ROOT}/k8s/scripts/build-images.sh" --tag dev --load-minikube

step "applying k8s/overlays/${OVERLAY}, waiting for rollouts, migrating"
"${REPO_ROOT}/k8s/scripts/deploy.sh" --overlay "${OVERLAY}" --image-tag dev --migrate

if [[ "${OPEN_CONSOLE}" -eq 0 ]]; then
    printf '\n  Done. Start the console with: ./scripts/console.sh\n'
    exit 0
fi

exec "${REPO_ROOT}/scripts/console.sh"
