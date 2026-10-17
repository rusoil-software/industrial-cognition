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
# Open the operator console, the way `jupyter lab` does: resolve the token, forward the
# API service to localhost, wait until it answers, print the URL and open the browser.
#
# It forwards the **Service**, never a pod, and never uses `kubectl exec`: a Pending pod
# cannot be exec'd into ("does not have a host assigned"), while the Service has endpoints
# as soon as one replica is ready - and this waits for that instead of failing.
#
# Usage:
#   ./scripts/console.sh [--port 8080] [--namespace cogni-ns] [--print-only] [--no-browser]

set -euo pipefail

PORT=8080
NAMESPACE=cogni-ns
SERVICE=api
PRINT_ONLY=0
OPEN_BROWSER=1
TIMEOUT=180

usage() {
    sed -n '25,34p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)        PORT="${2:?}"; shift 2 ;;
        -n|--namespace)   NAMESPACE="${2:?}"; shift 2 ;;
        --print-only)     PRINT_ONLY=1; shift ;;
        --no-browser)     OPEN_BROWSER=0; shift ;;
        --timeout)        TIMEOUT="${2:?}"; shift 2 ;;
        -h|--help)        usage ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

step() { printf '\033[36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$*"; }
die()  { printf '\033[31merror: %s\033[0m\n' "$1" >&2; [[ -n "${2:-}" ]] && printf '%s\n' "$2" >&2; exit 1; }

command -v kubectl >/dev/null 2>&1 || die "kubectl is not installed or not on PATH"

kubectl get --raw /healthz --request-timeout=10s >/dev/null 2>&1 || die \
    "the Kubernetes API server is not answering (context: $(kubectl config current-context 2>/dev/null || echo none))." \
    "  If this is minikube after a Docker restart, recreate it:
      minikube delete && minikube start --memory=10240 --cpus=8
  Then deploy: ./install.sh"

TOKEN_B64="$(kubectl -n "${NAMESPACE}" get secret industrial-cognition-secrets \
    --request-timeout=15s -o jsonpath='{.data.AUTH_TOKEN}' 2>/dev/null || true)"
[[ -n "${TOKEN_B64}" ]] || die \
    "the deployment has no AUTH_TOKEN in secret '${NAMESPACE}/industrial-cognition-secrets'." \
    "  Deploy first: ./install.sh"

TOKEN="$(printf '%s' "${TOKEN_B64}" | base64 -d 2>/dev/null || printf '%s' "${TOKEN_B64}")"
TOKEN="$(printf '%s' "${TOKEN}" | tr -d '\r\n')"

URL="http://127.0.0.1:${PORT}/auth/login?token=$(printf '%s' "${TOKEN}" | sed 's/ /%20/g')"

if [[ "${PRINT_ONLY}" -eq 1 ]]; then
    printf '%s\n' "${URL}"
    exit 0
fi

step "waiting for service ${NAMESPACE}/${SERVICE} to have endpoints"
deadline=$(( $(date +%s) + TIMEOUT ))
while true; do
    ips="$(kubectl -n "${NAMESPACE}" get endpoints "${SERVICE}" \
        --request-timeout=10s -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || true)"
    [[ -n "${ips}" ]] && break
    [[ $(date +%s) -ge ${deadline} ]] && die "service ${SERVICE} still has no ready endpoints after ${TIMEOUT}s."

    warn "no ready endpoints yet; pods:"
    kubectl -n "${NAMESPACE}" get pods --request-timeout=10s \
        -o custom-columns='NAME:.metadata.name,PHASE:.status.phase,NODE:.spec.nodeName' 2>/dev/null \
        | sed 's/^/    /' || true
    if kubectl -n "${NAMESPACE}" get pods --field-selector=status.phase=Pending \
        --request-timeout=10s -o name 2>/dev/null | grep -q .; then
        printf '    Pending pods usually mean the node is out of CPU or memory.\n' >&2
        printf '    Check: kubectl -n %s describe pod <name>\n' "${NAMESPACE}" >&2
    fi
    sleep 3
done

step "forwarding ${NAMESPACE}/${SERVICE} to 127.0.0.1:${PORT}"
kubectl -n "${NAMESPACE}" port-forward "svc/${SERVICE}" "${PORT}:80" >/tmp/cogni-portforward.log 2>&1 &
FORWARD_PID=$!
trap 'kill ${FORWARD_PID} 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 60); do
    if ! kill -0 "${FORWARD_PID}" 2>/dev/null; then
        die "port-forward exited: $(cat /tmp/cogni-portforward.log)"
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

printf '\n  Console is up.\n\n    %s\n\n' "${URL}"
printf '  The token is exchanged for a session cookie and you land on /ui/.\n'
printf '  Token: %s\n' "${TOKEN}"
printf '  Press Ctrl-C to stop the forward.\n\n'

if [[ "${OPEN_BROWSER}" -eq 1 ]]; then
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "${URL}" >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then
        open "${URL}" >/dev/null 2>&1 || true
    else
        warn "no browser opener found; open the URL above"
    fi
fi

wait "${FORWARD_PID}"
