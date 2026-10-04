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
"""Alembic environment.

Three deliberate decisions:

* **The URL comes from the environment, never from `alembic.ini`.** A DSN in
  version control is a credential leak, and the deployment already supplies
  ``DATABASE_URL`` through a ConfigMap/Secret (`k8s/base/app-config.yaml`) or
  through `docker-compose.yml`. The scheme is normalised to psycopg 3 by
  `src.api.storage.database`, so one value serves the API, the worker and
  migrations.

* **`compare_type` and `compare_server_default` are on.** Without them
  autogenerate silently ignores a changed column type or default - exactly the
  drift a migration tool exists to catch.

* **Native PostgreSQL enums get a matching `DROP TYPE`.** See
  :func:`_register_enum_drop_hook`; without it a downgrade leaves the enum
  behind and the next upgrade fails with ``type "object_type" already exists``.

Revisions are produced with::

    DATABASE_URL=... alembic revision --autogenerate -m "description"

and then reviewed. `target_metadata` below is what makes that command correct.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import engine_from_config, pool

# `prepend_sys_path = .` in alembic.ini covers the normal case, but invoking the
# module from another working directory does not, so make the repository root
# importable explicitly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.api.storage import models  # noqa: E402  (sys.path set up above)
from src.api.storage.database import DATABASE_URL, sync_url  # noqa: E402

# Alembic Config object, providing access to alembic.ini.
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

#: Autogenerate target. Importing `models` is what registers every table on the
#: metadata; omitting an import here makes autogenerate propose dropping the
#: table it cannot see.
target_metadata = models.Base.metadata


def _table_enum_types(table_name: str) -> list[str]:
    """Names of the native enum types used by *table_name*, in column order."""
    table = models.Base.metadata.tables.get(table_name)
    if table is None:
        # The dropped table is not in the current models - a revision dropping a
        # table that has since been removed. Nothing to drop by name.
        return []
    names: list[str] = []
    for column in table.columns:
        column_type = column.type
        enum_name = getattr(column_type, "name", None)
        if enum_name and getattr(column_type, "native_enum", False):
            if enum_name not in names:
                names.append(enum_name)
    return names


def _register_enum_drop_hook() -> bool:
    """Teach autogenerate to ``DROP TYPE`` the enums a dropped table owned.

    Alembic writes ``CREATE TYPE`` for a native PostgreSQL enum but never the
    matching ``DROP TYPE``, because a type is not owned by any single table. A
    downgrade therefore leaves the enums behind, and the next upgrade fails with::

        psycopg.errors.DuplicateObject: type "object_type" already exists

    which is exactly what the first round-trip verification of this migration
    produced. Extending ``DropTableOp`` fixes it at the source, so every future
    ``downgrade()`` is correct without anyone remembering to add a manual drop.

    Returns:
        True when the extension is installed. Callers (and the tests) use this to
        distinguish "correct by construction" from "the private API moved and a
        generated downgrade needs a manual ``DROP TYPE`` in review".
    """
    try:
        from alembic.autogenerate import render as render_module
        from alembic.operations import ops as ops_module
    except ImportError:  # pragma: no cover - alembic always ships these
        return False

    dispatcher = getattr(render_module, "renderers", None)
    if dispatcher is None or not hasattr(dispatcher, "dispatch_for"):  # pragma: no cover
        return False

    original = dispatcher.dispatch(ops_module.DropTableOp)
    if original is None:  # pragma: no cover - a DropTableOp renderer always exists
        return False
    if getattr(original, "_emits_enum_drops", False):
        return True  # already installed; keep idempotent

    def render_drop_table_with_enums(autogen_context: Any, op: Any) -> str:
        # This renderer must return a single STRING, not a list. Alembic joins a
        # list with "\n" and re-renders each element, so returning
        # `["op.execute(...)"]` produced one character per line in the revision
        # file - `SyntaxError: unterminated string literal`. `original` returns a
        # string, which is appended to verbatim.
        lines = original(autogen_context, op)
        emitted = autogen_context.opts.setdefault("_emitted_enum_drops", set())
        extra: list[str] = []
        for type_name in _table_enum_types(op.table_name):
            if type_name in emitted:
                continue
            emitted.add(type_name)
            # The text emitted here becomes *Python source* in the revision file,
            # so the SQL is wrapped in repr() to get its quoting and escaping
            # right. Naively interpolating into an `op.execute("...")` template
            # would emit `op.execute("DROP TYPE IF EXISTS object_type")` and
            # terminate the Python string early.
            statement = f"DROP TYPE IF EXISTS {type_name}"
            extra.append(f"op.execute({statement!r})")
        if not extra:
            return lines
        return "\n".join([lines, *extra])

    render_drop_table_with_enums._emits_enum_drops = True  # type: ignore[attr-defined]
    dispatcher.dispatch_for(
        ops_module.DropTableOp,
        qualifier=autogen_context_qualifier(dispatcher, ops_module.DropTableOp),
        replace=True,
    )(render_drop_table_with_enums)
    return True


def autogen_context_qualifier(dispatcher: Any, target: Any) -> str:
    """Return the qualifier the existing *target* renderer was registered under.

    The registry is keyed by ``(target, qualifier)``; replacing a renderer under
    the wrong qualifier would leave the original in place.
    """
    registry = getattr(dispatcher, "_registry", {})
    for key in registry:
        if isinstance(key, tuple) and key and key[0] is target:
            return str(key[1])
    return "default"


ENUM_DROP_HOOK_ACTIVE = _register_enum_drop_hook()


def _database_url() -> str:
    """Resolve the migration DSN, failing loudly when it is unusable.

    `alembic.ini` ships an empty ``sqlalchemy.url`` on purpose, so an unset
    ``DATABASE_URL`` surfaces as missing configuration rather than as a
    confusing connection error against the empty string.
    """
    override = (config.get_main_option("sqlalchemy.url", default="") or "").strip()
    if override:
        return sync_url(override)
    if not DATABASE_URL:
        raise RuntimeError(
            "No database URL for migrations. Set DATABASE_URL before running "
            "alembic."
        )
    return sync_url(DATABASE_URL)


def run_migrations_offline() -> None:
    """Emit SQL instead of executing it.

    Used to review a migration before it reaches a production database::

        DATABASE_URL=... alembic upgrade head --sql
    """
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _configure_and_run(connection: Any) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_schemas=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        _configure_and_run(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
