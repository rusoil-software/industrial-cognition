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
# ==============================================================================
"""Behavioural tests for the Alembic migration against a real PostgreSQL.

These are the tests that would have caught the two defects the first live run
exposed, and they cannot be written against a mock:

* PostgreSQL has no ``CREATE TYPE ... IF NOT EXISTS``, so a ``downgrade()`` that
  does not drop the native enums leaves them behind and the next ``upgrade``
  fails with ``type "object_type" already exists``. That only appears on the
  *second* upgrade, i.e. on a real round trip.
* ``autogenerate`` is only trustworthy if running it twice is a no-op, which
  requires comparing the models against a database that already has the schema.

Selection::

    pytest -m db                      # needs a reachable PostgreSQL
    DATABASE_URL=... pytest -m db     # explicit target

Skipped when ``DATABASE_URL`` is unset, when SQLAlchemy/pytest-alembic are
missing, or when nothing answers on the configured host - so a bare ``pytest``
on a workstation stays green. CI provides a PostgreSQL service container.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.db

DATABASE_URL = os.getenv("DATABASE_URL", "")


def _require_sqlalchemy():
    return pytest.importorskip(
        "sqlalchemy", reason="SQLAlchemy is required for the migration tests"
    )


def _reachable(url: str) -> bool:
    try:
        from sqlalchemy import create_engine, text
    except ImportError:  # pragma: no cover - guarded by _require_sqlalchemy
        return False
    try:
        engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        engine.dispose()
    except Exception:
        return False
    return True


@pytest.fixture(scope="module")
def alembic_config():
    """Alembic Config pointed at the test database."""
    _require_sqlalchemy()
    pytest.importorskip("alembic", reason="alembic is required for the migration tests")
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL is not set; skipping migration tests")
    if not _reachable(DATABASE_URL):
        pytest.skip(f"no reachable PostgreSQL at DATABASE_URL")

    from alembic.config import Config

    from src.api.storage.database import sync_url

    config = Config(str(_repo_root() / "alembic.ini"))
    config.set_main_option("script_location", str(_repo_root() / "alembic"))
    config.set_main_option("sqlalchemy.url", sync_url(DATABASE_URL))
    return config


def _repo_root():
    import pathlib

    return pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def migrated(alembic_config):
    """Bring the database to head, then back to base after the module."""
    from alembic import command

    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")
    yield alembic_config
    command.downgrade(alembic_config, "base")


def _columns(url: str, table: str) -> dict[str, str]:
    """Return ``{column: type}`` for *table*.

    For a native enum PostgreSQL reports ``data_type = 'USER-DEFINED'`` and puts
    the actual type name in ``udt_name``, so the two are combined here - asserting
    against ``data_type`` alone would compare ``'camera_status'`` to
    ``'USER-DEFINED'`` and fail on a correct schema.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(url)
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT column_name, data_type, udt_name FROM information_schema.columns "
                "WHERE table_name = :table ORDER BY ordinal_position"
            ),
            {"table": table},
        ).fetchall()
    engine.dispose()
    return {
        name: (udt if dtype == "USER-DEFINED" else dtype) for name, dtype, udt in rows
    }


def _enum_labels(url: str, type_name: str) -> list[str]:
    from sqlalchemy import create_engine, text

    engine = create_engine(url)
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT e.enumlabel FROM pg_enum e "
                "JOIN pg_type t ON t.oid = e.enumtypid "
                "WHERE t.typname = :name ORDER BY e.enumsortorder"
            ),
            {"name": type_name},
        ).fetchall()
    engine.dispose()
    return [label for (label,) in rows]


def _table_names(url: str) -> set[str]:
    from sqlalchemy import create_engine, text

    engine = create_engine(url)
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
        ).fetchall()
    engine.dispose()
    return {name for (name,) in rows}


def test_upgrade_creates_the_documented_tables(migrated) -> None:
    from src.api.storage.database import sync_url

    url = sync_url(DATABASE_URL)
    tables = _table_names(url)
    assert "detection_results" in tables
    assert "system_metadata" in tables
    assert "alembic_version" in tables


def test_detection_results_matches_the_documented_data_model(migrated) -> None:
    from src.api.storage.database import sync_url

    columns = _columns(sync_url(DATABASE_URL), "detection_results")
    assert set(columns) == {
        "id",
        "timestamp",
        "frame_id",
        "object_type",
        "coordinates",
        "confidence",
        "metadata",
    }
    # Timestamps must be timezone-aware: a naive industrial timestamp is a bug.
    assert columns["timestamp"] == "timestamp with time zone"
    assert columns["coordinates"] == "jsonb"
    assert columns["confidence"] == "double precision"


def test_system_metadata_matches_the_documented_data_model(migrated) -> None:
    from src.api.storage.database import sync_url

    columns = _columns(sync_url(DATABASE_URL), "system_metadata")
    assert set(columns) == {
        "id",
        "key",
        "value",
        "value_json",
        "camera_status",
        "processing_fps",
        "last_heartbeat",
        "updated_at",
    }
    assert columns["last_heartbeat"] == "timestamp with time zone"
    assert columns["camera_status"] == "camera_status"


def test_enum_labels_are_the_lowercase_domain_vocabulary(migrated) -> None:
    """The regression that a live run caught: `'METAL_SHEET'` instead of `'metal_sheet'`."""
    from src.api.storage.database import sync_url

    url = sync_url(DATABASE_URL)
    assert _enum_labels(url, "object_type") == [
        "metal_sheet",
        "detail",
        "debris",
        "clutter",
    ]
    assert _enum_labels(url, "camera_status") == [
        "connected",
        "disconnected",
        "error",
    ]


def test_downgrade_then_upgrade_round_trips(alembic_config) -> None:
    """Downgrade must remove the enum types, or the second upgrade cannot run.

    This is the test for `psycopg.errors.DuplicateObject: type "object_type"
    already exists`, which is what the first live verification produced.
    """
    from alembic import command
    from sqlalchemy import create_engine, text

    from src.api.storage.database import sync_url

    url = sync_url(DATABASE_URL)
    command.downgrade(alembic_config, "base")

    assert "detection_results" not in _table_names(url)
    engine = create_engine(url)
    with engine.connect() as connection:
        remaining = connection.execute(
            text("SELECT typname FROM pg_type WHERE typname IN ('object_type','camera_status')")
        ).fetchall()
    engine.dispose()
    assert remaining == [], (
        f"downgrade left the enum types behind: {remaining}; the next upgrade "
        "would fail with DuplicateObject"
    )

    command.upgrade(alembic_config, "head")
    assert {"detection_results", "system_metadata"} <= _table_names(url)


def test_autogenerate_is_empty_after_upgrade(migrated) -> None:
    """Models and database must agree, or the next autogenerate invents drift."""
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from sqlalchemy import create_engine

    from src.api.storage import models
    from src.api.storage.database import sync_url

    engine = create_engine(sync_url(DATABASE_URL))
    with engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        diff = compare_metadata(context, models.Base.metadata)
    engine.dispose()

    # Constraint/index reflection can report cosmetic differences; ignore the
    # entries that are not schema objects we declare.
    real = [entry for entry in diff if entry[0] != "remove_index"]
    assert real == [], f"autogenerate would produce changes: {real}"


def test_a_row_can_be_inserted_and_read_back(migrated) -> None:
    """Proves the schema is usable, not merely present."""
    import uuid
    from datetime import datetime, timezone

    from src.api.storage.database import make_session_factory
    from src.api.storage.models import DetectionResult, ObjectType, SystemState, SystemMetadata

    factory = make_session_factory()
    with factory() as session:
        session.add(
            DetectionResult(
                id=uuid.uuid4(),
                timestamp=datetime.now(timezone.utc),
                frame_id=42,
                object_type=ObjectType.METAL_SHEET,
                coordinates={"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.15},
                confidence=0.92,
                detection_metadata={"model": "owlv2", "workspace_state": "NO_TARGET"},
            )
        )
        session.add(
            SystemMetadata(
                key="camera_status",
                value=SystemState.CONNECTED.value,
                camera_status=SystemState.CONNECTED,
                processing_fps=59.5,
                last_heartbeat=datetime.now(timezone.utc),
            )
        )
        session.commit()

    with factory() as session:
        detection = session.query(DetectionResult).filter_by(frame_id=42).one()
        assert detection.object_type is ObjectType.METAL_SHEET
        assert detection.coordinates["width"] == pytest.approx(0.3)
        assert detection.detection_metadata["workspace_state"] == "NO_TARGET"

        system = session.query(SystemMetadata).filter_by(key="camera_status").one()
        assert system.camera_status is SystemState.CONNECTED
        assert system.processing_fps == pytest.approx(59.5)

        session.delete(detection)
        session.delete(system)
        session.commit()


def test_confidence_outside_zero_to_one_is_rejected(migrated) -> None:
    """The CHECK constraint is the last line of defence for a bad model output."""
    import uuid
    from sqlalchemy.exc import IntegrityError

    from src.api.storage.database import make_session_factory
    from src.api.storage.models import DetectionResult, ObjectType

    factory = make_session_factory()
    with factory() as session:
        session.add(
            DetectionResult(
                id=uuid.uuid4(),
                frame_id=1,
                object_type=ObjectType.DEBRIS,
                coordinates={"x": 0.0, "y": 0.0, "width": 0.1, "height": 0.1},
                confidence=1.5,
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


def test_duplicate_system_metadata_key_is_rejected(migrated) -> None:
    from sqlalchemy.exc import IntegrityError

    from src.api.storage.database import make_session_factory
    from src.api.storage.models import SystemMetadata

    factory = make_session_factory()
    with factory() as session:
        session.add(SystemMetadata(key="processing_fps", value="60"))
        session.commit()
    with factory() as session:
        session.add(SystemMetadata(key="processing_fps", value="61"))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
    with factory() as session:
        session.query(SystemMetadata).filter_by(key="processing_fps").delete()
        session.commit()


def test_migration_dsn_rejects_a_non_postgres_scheme() -> None:
    """Fails loudly instead of silently connecting to the wrong database."""
    _require_sqlalchemy()
    from src.api.storage.database import sync_url

    with pytest.raises(ValueError):
        sync_url("mysql://user:pass@localhost/db")
