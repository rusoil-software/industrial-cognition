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
# ==============================================================================
"""Readiness policy tests.

Pure function, no cluster and no database: the point under test is *which* conditions make a
pod unusable, and that differs by role.
"""

from __future__ import annotations

from src.api.readiness import evaluate_readiness


def test_healthy_database_and_pipeline_is_ready() -> None:
    result = evaluate_readiness(
        database_ok=True, pipeline_running=True, require_pipeline=True
    )
    assert result.ready is True
    assert result.checks["database"] is True


def test_no_database_is_never_ready() -> None:
    """Prompts and history both need it; every useful endpoint would 500."""
    for require in (True, False):
        result = evaluate_readiness(
            database_ok=False, pipeline_running=True, require_pipeline=require
        )
        assert result.ready is False
        assert "database" in (result.detail or "")


def test_a_capture_pod_without_capture_is_not_ready() -> None:
    """Its whole job is analysing frames; without capture it should leave the endpoints."""
    result = evaluate_readiness(
        database_ok=True,
        pipeline_running=False,
        require_pipeline=True,
        pipeline_reason="capture source could not be opened",
    )
    assert result.ready is False
    assert result.detail == "capture source could not be opened"


def test_the_api_stays_ready_with_a_dead_pipeline() -> None:
    """Pulling the console out of the endpoints would hide the page that explains the fault."""
    result = evaluate_readiness(
        database_ok=True, pipeline_running=False, require_pipeline=False
    )
    assert result.ready is True
    assert result.checks["pipeline"] is False


def test_the_default_reason_is_used_when_none_is_given() -> None:
    result = evaluate_readiness(
        database_ok=True, pipeline_running=False, require_pipeline=True
    )
    assert result.ready is False
    assert "not running" in (result.detail or "")


def test_the_response_shape_is_stable() -> None:
    """Probes and dashboards read these keys; renaming one silently breaks both."""
    payload = evaluate_readiness(
        database_ok=True, pipeline_running=True, require_pipeline=False
    ).to_dict()

    assert set(payload) == {"ready", "checks", "detail"}
    assert set(payload["checks"]) == {"database", "pipeline", "pipeline_required"}
