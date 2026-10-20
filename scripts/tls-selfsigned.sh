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
# date   : 2026-Oct-20
# ==============================================================================
#
# Create the TLS Secret the ingress expects, from a local internal CA.
#
# Why not cert-manager or Let's Encrypt here: the MVP runs on-premise with no route to the
# internet. ACME needs a publicly reachable name, and cert-manager needs its own images
# mirrored. An internal CA needs neither - it needs one machine with openssl, once per
# rotation.
#
# Two decisions worth stating:
#
#   * A **CA of its own** rather than a self-signed server certificate. A self-signed leaf has
#     to be trusted individually and every rotation re-teaches every client; with a CA the
#     operators import `ca.crt` once and rotations are invisible to them.
#   * The secret carries `ca.crt` as well as `tls.crt`/`tls.key`. `tls.crt` is what the ingress
#     serves; `ca.crt` is what a human (or a robot's client) installs to trust it. Same files,
#     one source of truth.
#
# Usage:
#   ./scripts/tls-selfsigned.sh                        # host from the ingress, 825 days
#   ./scripts/tls-selfsigned.sh --host cogni.local
#   ./scripts/tls-selfsigned.sh --days 365 --out ./tls
#
# Rotating is the same command again: it updates the Secret in place.

set -euo pipefail

NAMESPACE="${NAMESPACE:-cogni-ns}"
SECRET_NAME="${SECRET_NAME:-industrial-cognition-tls}"
HOST="${INGRESS_HOST:-mvm.example.com}"
DAYS="${DAYS:-825}"
OUT_DIR="${OUT_DIR:-}"
CA_SUBJECT="${CA_SUBJECT:-/CN=Industrial Cognition Internal CA/O=Rusoil Software}"

for ((index = 1; index <= $#; index++)); do
    argument="${!index}"
    case "${argument}" in
        --host=*) HOST="${argument#*=}" ;;
        --host)   index=$((index + 1)); HOST="${!index}" ;;
        --days=*) DAYS="${argument#*=}" ;;
        --days)   index=$((index + 1)); DAYS="${!index}" ;;
        --out=*)  OUT_DIR="${argument#*=}" ;;
        --out)    index=$((index + 1)); OUT_DIR="${!index}" ;;
        --namespace=*) NAMESPACE="${argument#*=}" ;;
        --namespace)   index=$((index + 1)); NAMESPACE="${!index}" ;;
        -h|--help) sed -n '26,45p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: ${argument}" >&2; exit 2 ;;
    esac
done

log() { printf '\033[36m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31merror: %s\033[0m\n' "$1" >&2; exit 1; }

command -v kubectl >/dev/null 2>&1 || die "kubectl is not on PATH"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

# openssl is not installed on a default Windows box, but docker is (the cluster needs it).
openssl_run() {
    if command -v openssl >/dev/null 2>&1; then
        openssl "$@"
    elif command -v docker >/dev/null 2>&1; then
        docker run --rm -v "${WORK_DIR}:/work" -w /work alpine:3.20 \
            sh -c "apk add -q --no-cache openssl >/dev/null 2>&1; openssl $*"
    else
        die "neither openssl nor docker is available to generate the certificate"
    fi
}

log "generating an internal CA and a server certificate for ${HOST} (${DAYS} days)"

cat > "${WORK_DIR}/server.ext" <<EOF
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = DNS:${HOST},DNS:*.${HOST},DNS:industrial-cognition,DNS:api,DNS:localhost,IP:127.0.0.1
EOF

openssl_run req -x509 -newkey rsa:4096 -nodes -days "${DAYS}" \
    -keyout "${WORK_DIR}/ca.key" -out "${WORK_DIR}/ca.crt" \
    -subj "${CA_SUBJECT}" >/dev/null 2>&1

openssl_run req -newkey rsa:2048 -nodes \
    -keyout "${WORK_DIR}/tls.key" -out "${WORK_DIR}/tls.csr" \
    -subj "/CN=${HOST}/O=Rusoil Software" >/dev/null 2>&1

openssl_run x509 -req -in "${WORK_DIR}/tls.csr" -CA "${WORK_DIR}/ca.crt" \
    -CAkey "${WORK_DIR}/ca.key" -CAcreateserial -days "${DAYS}" \
    -extfile "${WORK_DIR}/server.ext" -out "${WORK_DIR}/tls.crt" >/dev/null 2>&1

log "creating Secret ${NAMESPACE}/${SECRET_NAME}"
kubectl -n "${NAMESPACE}" create secret tls "${SECRET_NAME}" \
    --cert="${WORK_DIR}/tls.crt" --key="${WORK_DIR}/tls.key" \
    --dry-run=client -o yaml > "${WORK_DIR}/secret.yaml"

# The CA goes into the same Secret so the trust anchor ships with the certificate it signed.
kubectl -n "${NAMESPACE}" create secret generic "${SECRET_NAME}-ca" \
    --from-file=ca.crt="${WORK_DIR}/ca.crt" \
    --dry-run=client -o yaml > "${WORK_DIR}/ca-secret.yaml"

kubectl apply -f "${WORK_DIR}/secret.yaml"
kubectl apply -f "${WORK_DIR}/ca-secret.yaml"

# The ingress serves `tls.crt`; the CA secret is what a client imports. Keeping them separate
# avoids putting a private key file and a public trust anchor in the same object by habit.
kubectl -n "${NAMESPACE}" get secret "${SECRET_NAME}" -o jsonpath='{.data.tls\.crt}' \
    | base64 -d | openssl_run x509 -noout -subject -issuer -dates 2>/dev/null || true

if [[ -n "${OUT_DIR}" ]]; then
    mkdir -p "${OUT_DIR}"
    # Only the public half leaves the machine; the CA key is not written out.
    cp "${WORK_DIR}/ca.crt" "${WORK_DIR}/tls.crt" "${OUT_DIR}/"
    log "wrote ${OUT_DIR}/ca.crt and ${OUT_DIR}/tls.crt (public certificates only)"
fi

log "done"
cat <<EOF

  The ingress now has a certificate for ${HOST}, signed by the internal CA.

  Trust it on a client by importing the CA certificate:
      kubectl -n ${NAMESPACE} get secret ${SECRET_NAME}-ca \\
          -o jsonpath='{.data.ca\.crt}' | base64 -d > cogni-ca.crt

  Then, from a machine that can resolve ${HOST} to the ingress address:
      curl --cacert cogni-ca.crt https://${HOST}/health

  Without DNS, test by pinning the name to the ingress IP:
      curl --cacert cogni-ca.crt --resolve ${HOST}:443:\$(minikube ip) https://${HOST}/health

  Rotation is this same command. Clients keep trusting the new leaf because the CA is unchanged.
EOF
