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
"""Readiness as a decision, separated from the HTTP plumbing.

`/health` answers "is this process alive" and must stay 200 - a Kubernetes liveness probe
that fails on a degraded pipeline restarts a perfectly good pod. `/readyz` answers a
different question: "should traffic be sent here right now".

What readiness means depends on the pod's job, which is why the pipeline requirement is a
parameter rather than a constant:

* The **capture** role exists to analyse frames. If capture is not running, that pod has
  nothing to offer and should leave the Service endpoints (`require_pipeline=True`).
* The **API** role serves the console, prompts and history. A dead capture pipeline makes
  it degraded, not useless - pulling it out of the endpoints would take the UI down with
  the camera and remove the very page that explains what is wrong. So it requires the
  database and reports the pipeline as a check, not a gate (`require_pipeline=False`).

The database is always required: prompt reads and history writes both need it, and every
endpoint that matters would 500 without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

__all__ = ["Readiness", "evaluate_readiness"]


@dataclass(frozen=True, slots=True)
class Readiness:
    """The outcome of a readiness check, and why."""

    ready: bool
    checks: dict[str, Any]
    detail: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"ready": self.ready, "checks": self.checks, "detail": self.detail}


def evaluate_readiness(
    *,
    database_ok: bool,
    pipeline_running: bool,
    require_pipeline: bool,
    pipeline_reason: Optional[str] = None,
) -> Readiness:
    """Decide whether this pod should receive traffic.

    Args:
        database_ok: a trivial query against PostgreSQL succeeded.
        pipeline_running: the capture thread is alive (meaningless if capture is disabled).
        require_pipeline: whether this pod's role depends on capture running.
        pipeline_reason: the reported reason capture is down, for the response body.
    """
    checks = {
        "database": database_ok,
        "pipeline": pipeline_running,
        "pipeline_required": require_pipeline,
    }

    if not database_ok:
        return Readiness(False, checks, "database is not reachable")

    if require_pipeline and not pipeline_running:
        return Readiness(
            False,
            checks,
            pipeline_reason or "capture pipeline is not running",
        )

    return Readiness(True, checks)
