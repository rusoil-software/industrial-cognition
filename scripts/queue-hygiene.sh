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
# Recreate the Celery queues with their configured bounds.
#
# RabbitMQ queue arguments are immutable. Adding `x-max-length`, `x-message-ttl` or a
# dead-letter exchange to `src/tasks/celery.py` therefore does nothing to an existing queue,
# and a worker that tries to redeclare it fails with PRECONDITION_FAILED (406). The queues
# have to be deleted once so the consumers can recreate them with the new arguments.
#
# This is destructive on purpose and requires --force: deleting the work queue discards the
# frames queued at that moment. For this pipeline that is acceptable - every queued frame is
# already stale by design (see doc/vision-pipeline.md §4) - but it is still a decision a human
# should make explicitly, not something an installer does silently.
#
# Usage:
#   ./scripts/queue-hygiene.sh            # report the current arguments
#   ./scripts/queue-hygiene.sh --force    # delete the queues so workers recreate them

set -euo pipefail

NAMESPACE="${NAMESPACE:-cogni-ns}"
BROKER_POD="${BROKER_POD:-rabbitmq-0}"
BROKER_CONTAINER="${BROKER_CONTAINER:-rabbitmq}"
QUEUES="${QUEUES:-vision,vision.priority,vision.dlq,vision.results}"
FORCE=0

for argument in "$@"; do
    case "${argument}" in
        --force) FORCE=1 ;;
        -h|--help) sed -n '24,37p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: ${argument}" >&2; exit 2 ;;
    esac
done

rmq() {
    kubectl -n "${NAMESPACE}" exec "${BROKER_POD}" -c "${BROKER_CONTAINER}" -- "$@"
}

echo "== current queue state =="
rmq rabbitmqctl list_queues name messages arguments 2>/dev/null | grep -E '^(vision|name)' || true

echo
echo "== expected (from src/tasks/celery.py) =="
echo "  vision, vision.priority: x-max-length, x-overflow=drop-head, x-message-ttl, x-dead-letter-exchange=${DEAD_LETTER_EXCHANGE:-vision.dlx}"
echo "  vision.dlq:             no bounds, bound to the dead-letter exchange"

if [[ "${FORCE}" -eq 0 ]]; then
    echo
    echo "Re-run with --force to delete and recreate them (drops queued frames)."
    exit 0
fi

echo
echo "== deleting =="
IFS=',' read -r -a names <<< "${QUEUES}"
for name in "${names[@]}"; do
    [[ -n "${name}" ]] || continue
    rmq rabbitmqctl delete_queue "${name}" >/dev/null 2>&1 && echo "  deleted ${name}" || echo "  ${name} absent"
done

echo
echo "  The next producer/consumer connection recreates them with the new arguments."
echo "  Restart the workers so they redeclare:"
echo "      kubectl -n ${NAMESPACE} rollout restart deploy/vision deploy/api"
