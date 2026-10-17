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
"""Request-scoped database session for the API.

The session factory (and therefore the engine and its pool) is created once per
process and reused; building a new engine per request would open a connection pool
60 times a second.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterator

from sqlalchemy.orm import Session

from src.api.storage.database import make_session_factory

__all__ = ["get_db", "get_session_factory"]


@lru_cache(maxsize=1)
def get_session_factory():
    """Process-wide session factory."""
    return make_session_factory()


def get_db() -> Iterator[Session]:
    """FastAPI dependency: a session per request, committed on success.

    The commit happens here rather than in each endpoint so that a handler which
    returns successfully but forgets to commit cannot silently lose a write.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
