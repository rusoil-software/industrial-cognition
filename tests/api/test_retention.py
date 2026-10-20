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
"""Tests for the retention window.

The SQL is covered where a database exists; the decision - which rows count as old - is a pure
function and is where an off-by-one would quietly keep or delete the wrong day.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.api.maintenance.policy import DEFAULT_RETENTION_DAYS, cutoff_for

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def test_cutoff_is_the_window_before_now() -> None:
    assert cutoff_for(30, NOW) == NOW - timedelta(days=30)


def test_the_default_window_is_thirty_days() -> None:
    assert DEFAULT_RETENTION_DAYS == 30


def test_zero_days_means_everything_older_than_now() -> None:
    """Used deliberately for a wipe, and by the tests as the strongest case."""
    assert cutoff_for(0, NOW) == NOW


def test_a_negative_window_is_refused() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        cutoff_for(-1, NOW)


def test_the_cutoff_defaults_to_utc_now() -> None:
    """Naive datetimes would compare wrongly against a timestamptz column."""
    before = datetime.now(timezone.utc)
    cutoff = cutoff_for(1)
    after = datetime.now(timezone.utc)

    assert cutoff.tzinfo is not None
    assert before - timedelta(days=1) <= cutoff <= after - timedelta(days=1)


def test_a_one_day_window_boundary() -> None:
    cutoff = cutoff_for(1, NOW)
    assert cutoff < NOW
    assert NOW - cutoff == timedelta(hours=24)
