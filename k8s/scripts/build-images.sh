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
# Build the application images.
#
#   ./k8s/scripts/build-images.sh --tag dev
#   ./k8s/scripts/build-images.sh --tag 1.4.2 --registry registry.example.com:5000 --load-minikube
#
# Both application images come from the SAME Dockerfile. That is deliberate:
# `doc/project-development.md` mandates src/api/Dockerfile as the one
# application Dockerfile ("do not create others"), and the Celery worker imports
# modules from across src/ (src.tasks.celery -> src.vision.service ->
# src.camera.schemas), so its dependency set must be a superset of the API's.
# requirements/vision.txt alone cannot run `celery -A src.tasks.celery worker`.
#
# Verify compatibility by importing the worker module inside the built image:
#   docker run --rm industrial-cognition/vision:dev \
#     python -c "import src.tasks.celery; print('ok')"
#
# See doc/k8s.md §2 for why camera/robot share the api image.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REGISTRY=""
TAG="dev"
PUSH=0
LOAD_MINIKUBE=0
API_IMAGE="industrial-cognition/api"
VISION_IMAGE="industrial-cognition/vision"

usage() {
  cat <<'EOF'
Usage: build-images.sh [options]

Options:
  -t, --tag <tag>          Image tag to apply (default: dev)
  -r, --registry <host>    Registry prefix, e.g. registry.example.com:5000
      --push               Push both images after building
      --load-minikube      Load both images into the running minikube node
                           (required for a cluster that cannot pull them)
  -h, --help               Show this help
EOF
}

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--tag)           TAG="${2:?}"; shift 2 ;;
    -r|--registry)      REGISTRY="${2:?}"; shift 2 ;;
    --push)             PUSH=1; shift ;;
    --load-minikube)    LOAD_MINIKUBE=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH"
[[ -f "${REPO_ROOT}/src/api/Dockerfile" ]] || die "src/api/Dockerfile is missing"

prefix=""
[[ -n "${REGISTRY}" ]] && prefix="${REGISTRY}/"
api_ref="${prefix}${API_IMAGE}:${TAG}"
vision_ref="${prefix}${VISION_IMAGE}:${TAG}"

log "Building ${api_ref} from src/api/Dockerfile"
docker build \
  --file "${REPO_ROOT}/src/api/Dockerfile" \
  --tag "${api_ref}" \
  --label "org.opencontainers.image.revision=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  "${REPO_ROOT}"

# A re-tag, not a rebuild: identical content, so the node stores one layer set.
log "Tagging ${vision_ref}"
docker tag "${api_ref}" "${vision_ref}"

log "Verifying the worker entrypoint imports"
if docker run --rm --entrypoint python "${api_ref}" -c "import src.tasks.celery" >/dev/null 2>&1; then
  log "  src.tasks.celery imports cleanly"
else
  warn "src.tasks.celery failed to import; the vision worker will CrashLoopBackOff"
fi

if [[ "${PUSH}" -eq 1 ]]; then
  [[ -n "${REGISTRY}" ]] || die "--push requires --registry"
  log "Pushing"
  docker push "${api_ref}"
  docker push "${vision_ref}"
fi

if [[ "${LOAD_MINIKUBE}" -eq 1 ]]; then
  command -v minikube >/dev/null 2>&1 || die "--load-minikube requires the minikube CLI"
  # A local cluster has no registry: `imagePullPolicy: IfNotPresent` only helps
  # if the image is already in the node's containerd store, otherwise the kubelet
  # tries docker.io and fails with ErrImagePull.
  log "Loading into minikube"
  minikube image load "${api_ref}"
  minikube image load "${vision_ref}"
  minikube image ls | grep -F 'industrial-cognition' || warn "images not visible in the node"
fi

cat <<EOF

Images prepared:
  ${api_ref}      (api, camera, robot)
  ${vision_ref}   (vision worker)

Point the overlay at them:
  # k8s/overlays/dev/kustomization.yaml
  images:
    - name: industrial-cognition/api
      newName: ${prefix}${API_IMAGE}
      newTag: ${TAG}
    - name: industrial-cognition/vision
      newName: ${prefix}${VISION_IMAGE}
      newTag: ${TAG}
EOF
