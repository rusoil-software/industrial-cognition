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
# date   : 2026-Sep-10
# ==============================================================================
#
# Validate the Kubernetes manifests.
#
#   ./k8s/scripts/validate.sh                 # every overlay
#   ./k8s/scripts/validate.sh --overlay prod  # one overlay
#
# Validation is layered, and every layer that cannot run is *skipped with a
# warning* rather than silently passing:
#
#   1. `kubectl kustomize` must render the overlay (the real Kustomize, not a
#      reimplementation).
#   2. `kubeconform` or `kubeval` schema validation, when installed.
#   3. `kubectl apply --dry-run=client` when a cluster is reachable. Without an
#      API server kubectl cannot fetch the OpenAPI schema, so this layer is
#      skipped - it is not a failure.
#   4. `pytest tests/k8s`, the repo-native contract tests, which run everywhere.
#
# See doc/k8s.md §11.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OVERLAYS=("dev" "prod")
KUBERNETES_VERSION="1.29.0"
FAILED=0
SKIPPED=()

usage() {
  cat <<'EOF'
Usage: validate.sh [options]

Options:
  -o, --overlay <dev|prod>   Validate only this overlay (default: all)
  -k, --k8s-version <ver>    Kubernetes version for schema validation (default: 1.29.0)
  -h, --help                 Show this help
EOF
}

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; SKIPPED+=("$*"); }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; FAILED=1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--overlay)     OVERLAYS=("${2:?}"); shift 2 ;;
    -k|--k8s-version) KUBERNETES_VERSION="${2:?}"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *) die "unknown argument: $1 (try --help)"; exit 1 ;;
  esac
done

# --------------------------------------------------------------------------- #
# 1. Render with the real Kustomize
# --------------------------------------------------------------------------- #
if ! command -v kubectl >/dev/null 2>&1; then
  die "kubectl is required: Kustomize ships inside it"
  exit 1
fi

RENDERED=()
for overlay in "${OVERLAYS[@]}"; do
  directory="${REPO_ROOT}/k8s/overlays/${overlay}"
  if [[ ! -d "${directory}" ]]; then
    die "overlay not found: ${directory}"
    continue
  fi
  output="$(mktemp)"
  if kubectl kustomize "${directory}" >"${output}" 2>"${output}.err"; then
    ok "kubectl kustomize k8s/overlays/${overlay} ($(grep -c '^kind:' "${output}") objects)"
    RENDERED+=("${overlay}:${output}")
  else
    die "kubectl kustomize failed for k8s/overlays/${overlay}:"
    sed 's/^/    /' "${output}.err" >&2
  fi
done

# --------------------------------------------------------------------------- #
# 2. Schema validation, when a validator is available
# --------------------------------------------------------------------------- #
VALIDATOR=""
if command -v kubeconform >/dev/null 2>&1; then
  VALIDATOR="kubeconform"
elif command -v kubeval >/dev/null 2>&1; then
  VALIDATOR="kubeval"
fi

if [[ -n "${VALIDATOR}" ]]; then
  for entry in "${RENDERED[@]}"; do
    overlay="${entry%%:*}"; manifest="${entry#*:}"
    log "Schema validation (${VALIDATOR}) for ${overlay}"
    if [[ "${VALIDATOR}" == "kubeconform" ]]; then
      if kubeconform -strict -summary -kubernetes-version "${KUBERNETES_VERSION}" \
        -ignore-missing-schemas "${manifest}"; then
        ok "${VALIDATOR}: k8s/overlays/${overlay}"
      else
        die "${VALIDATOR} reported schema errors for k8s/overlays/${overlay}"
      fi
    else
      if kubeval --strict --kubernetes-version "${KUBERNETES_VERSION}" \
        --ignore-missing-schemas "${manifest}"; then
        ok "${VALIDATOR}: k8s/overlays/${overlay}"
      else
        die "${VALIDATOR} reported schema errors for k8s/overlays/${overlay}"
      fi
    fi
  done
else
  warn "neither kubeconform nor kubeval is installed; schema validation skipped"
fi

# --------------------------------------------------------------------------- #
# 3. Client-side dry run, when a cluster is reachable
# --------------------------------------------------------------------------- #
if kubectl cluster-info >/dev/null 2>&1; then
  for entry in "${RENDERED[@]}"; do
    overlay="${entry%%:*}"; manifest="${entry#*:}"
    log "Dry run against the live cluster for ${overlay}"
    if kubectl apply --dry-run=client -f "${manifest}" >/dev/null; then
      ok "kubectl dry-run: k8s/overlays/${overlay}"
    else
      die "kubectl dry-run failed for k8s/overlays/${overlay}"
    fi
  done
else
  warn "no reachable cluster; kubectl dry-run skipped (it needs the OpenAPI schema)"
fi

# --------------------------------------------------------------------------- #
# 4. Repo-native contract tests
# --------------------------------------------------------------------------- #
if python3 -c "import yaml" >/dev/null 2>&1 || python -c "import yaml" >/dev/null 2>&1; then
  PYTHON="python3"
  command -v python3 >/dev/null 2>&1 || PYTHON="python"
  log "Running tests/k8s"
  if (cd "${REPO_ROOT}" && "${PYTHON}" -m pytest tests/k8s -q); then
    ok "pytest tests/k8s"
  else
    die "pytest tests/k8s failed"
  fi
else
  warn "PyYAML is not installed; tests/k8s skipped (pip install -r requirements/k8s.txt)"
fi

for entry in "${RENDERED[@]}"; do
  rm -f "${entry#*:}" "${entry#*:}.err"
done

echo
if [[ "${FAILED}" -ne 0 ]]; then
  printf '\033[1;31mVALIDATION FAILED\033[0m\n'
  exit 1
fi
if [[ "${#SKIPPED[@]}" -gt 0 ]]; then
  printf '\033[1;33mVALIDATION PASSED with %d skipped layer(s):\033[0m\n' "${#SKIPPED[@]}"
  printf '  - %s\n' "${SKIPPED[@]}"
else
  printf '\033[1;32mVALIDATION PASSED (all layers)\033[0m\n'
fi
