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
"""Engine and session factories for the PostgreSQL store.

Two entry points, because the project needs both and choosing one now would be a
guess rather than a decision:

* :func:`get_engine` / :func:`session_scope` - synchronous. Alembic needs a
  synchronous connection to run migrations, and the Celery worker is synchronous.
* :func:`get_async_engine` / :func:`async_session_scope` - asynchronous, for the
  FastAPI request path that `doc/vision.md` describes as fully async.

Both wrap the same :data:`DATABASE_URL`. `doc/conventions.md` §4 puts all
database access behind a repository, so this module is plumbing only - no
queries live here.

DSN handling: the deployment contract (`k8s/base/app-config.yaml`) is
``postgresql+psycopg://...``, which is psycopg 3 in synchronous mode. The async
engine needs the ``postgresql+psycopg_async://`` spelling, and a bare
``postgresql://`` (what a hand-written ``DATABASE_URL`` usually looks like) is
normalised to psycopg 3 on both paths, so the image never needs psycopg2 as
well.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, AsyncIterator, Iterator, Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

__all__ = [
    "DATABASE_URL",
    "async_session_scope",
    "async_url",
    "get_async_engine",
    "get_engine",
    "make_async_session_factory",
    "make_session_factory",
    "session_scope",
    "sync_url",
]

#: Deployment-provided DSN. The default mirrors the docker-compose stack so local
#: work needs no environment, exactly like `src/tasks/celery.py`.
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://cogni:cogni-dev-password@postgres:5432/cogni_db",
)

_SYNC_DRIVER = "postgresql+psycopg"
_ASYNC_DRIVER = "postgresql+psycopg_async"


def sync_url(url: Optional[str] = None) -> str:
    """Return *url* in a psycopg-3 synchronous form.

    Raises:
        ValueError: if the URL is not a PostgreSQL URL. That means the
            deployment wired the wrong database in, and failing at startup beats
            failing on the first request.
    """
    resolved = url or DATABASE_URL
    scheme = resolved.split("://", 1)[0]
    if not resolved.startswith("postgresql://") and not resolved.startswith("postgresql+"):
        raise ValueError(
            f"DATABASE_URL must be a PostgreSQL DSN, got scheme {scheme!r}"
        )
    _, _, rest = resolved.partition("://")
    if scheme in ("postgresql", "postgresql+psycopg2", "postgresql+psycopg_async"):
        return f"{_SYNC_DRIVER}://{rest}"
    return resolved


def async_url(url: Optional[str] = None) -> str:
    """Return *url* in the psycopg-3 asynchronous form."""
    _, _, rest = sync_url(url).partition("://")
    return f"{_ASYNC_DRIVER}://{rest}"


def get_engine(url: Optional[str] = None, **kwargs: Any) -> Engine:
    """Build a synchronous engine.

    ``pool_pre_ping`` is on by default: the documented failure mode
    (`COMM_ERROR` on network loss) is far easier to handle as a transparent
    reconnect than as a dead pooled connection.
    """
    kwargs.setdefault("pool_pre_ping", True)
    return create_engine(sync_url(url), **kwargs)


def get_async_engine(url: Optional[str] = None, **kwargs: Any) -> AsyncEngine:
    """Build an asynchronous engine."""
    kwargs.setdefault("pool_pre_ping", True)
    return create_async_engine(async_url(url), **kwargs)


def make_session_factory(
    engine: Optional[Engine] = None, **kwargs: Any
) -> sessionmaker[Session]:
    """Return a synchronous session factory bound to *engine*.

    ``expire_on_commit=False`` keeps ORM objects usable after a commit, which is
    what a request handler returning a persisted object wants.
    """
    kwargs.setdefault("expire_on_commit", False)
    kwargs.setdefault("autoflush", False)
    return sessionmaker(bind=engine or get_engine(), **kwargs)


@contextlib.contextmanager
def session_scope(
    session_factory: Optional[sessionmaker[Session]] = None,
) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception.

    Usage::

        with session_scope() as session:
            session.add(result)
    """
    factory = session_factory or make_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def make_async_session_factory(
    engine: Optional[AsyncEngine] = None, **kwargs: Any
) -> async_sessionmaker[AsyncSession]:
    """Return an asynchronous session factory."""
    kwargs.setdefault("expire_on_commit", False)
    return async_sessionmaker(bind=engine or get_async_engine(), **kwargs)


@contextlib.asynccontextmanager
async def async_session_scope(
    session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
) -> AsyncIterator[AsyncSession]:
    """Asynchronous equivalent of :func:`session_scope`."""
    factory = session_factory or make_async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
