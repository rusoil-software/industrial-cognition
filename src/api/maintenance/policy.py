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
"""Retention policy, kept free of SQLAlchemy so it is testable anywhere.

The database work lives in `src/api/maintenance/retention.py`; the rule that decides *which*
rows are old is a pure function of a window and a clock. Separating them means the policy is
covered by tests that run without a database, which is where the off-by-one errors live.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

__all__ = ["DEFAULT_RETENTION_DAYS", "cutoff_for"]

#: How much history to keep when the environment does not say otherwise.
DEFAULT_RETENTION_DAYS = 30


def cutoff_for(days: int, now: Optional[datetime] = None) -> datetime:
    """The oldest timestamp to keep.

    Raises:
        ValueError: for a negative window. Zero is allowed on purpose - it means "everything
            older than now", which is how a test or a deliberate wipe expresses itself.
    """
    if days < 0:
        raise ValueError(f"retention days must be >= 0, got {days}")
    reference = now or datetime.now(timezone.utc)
    return reference - timedelta(days=days)
