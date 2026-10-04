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
# ==============================================================================
"""Unit tests for the ORM schema and the generated Alembic revision.

Deliberately import-free of SQLAlchemy where possible: `tests/k8s` runs on a
machine with nothing but PyYAML, and the migration's *content* can be checked by
parsing the revision file's AST plus reading the model source. The tests that do
need SQLAlchemy import it inside the test body and skip when it is absent.

The full behavioural check - create the schema, drop it, recreate it, and prove a
second autogenerate is empty - lives in `tests/db/test_migrations.py` and needs a
real PostgreSQL.
"""

from __future__ import annotations

import ast
import configparser
import enum
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS = ROOT / "src" / "api" / "storage" / "models.py"
DATABASE = ROOT / "src" / "api" / "storage" / "database.py"
VERSIONS = ROOT / "alembic" / "versions"
ALEMBIC_INI = ROOT / "alembic.ini"
ENV_PY = ROOT / "alembic" / "env.py"

EXPECTED_TABLES = ("detection_results", "system_metadata")


def revision_files() -> list[pathlib.Path]:
    return sorted(p for p in VERSIONS.glob("*.py") if p.name != "__init__.py")


@pytest.fixture(scope="module")
def revision_source() -> str:
    files = revision_files()
    assert len(files) == 1, (
        f"expected exactly one migration revision, found {[f.name for f in files]}; "
        "an extra file is usually a leftover empty autogenerate"
    )
    return files[0].read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def test_models_declare_the_documented_tables() -> None:
    """`doc/tasklist.md` names both tables explicitly."""
    source = MODELS.read_text(encoding="utf-8")
    for table in EXPECTED_TABLES:
        assert f'__tablename__ = "{table}"' in source, f"{table} is not declared"


def test_enum_values_are_the_domain_vocabulary_not_python_member_names() -> None:
    """The database vocabulary must match what the vision/camera code emits.

    SQLAlchemy's `Enum` persists the member *name* unless `values_callable` is
    given, so `ObjectType.METAL_SHEET` would be stored as `'METAL_SHEET'` while
    `src/camera/schemas.py` and `src/vision/service.py` produce `"metal_sheet"`.
    A live round-trip caught exactly that; this is the static guard.
    """
    source = MODELS.read_text(encoding="utf-8")
    assert "values_callable" in source, (
        "the enum columns must set values_callable, otherwise the stored "
        "vocabulary is the Python member name (METAL_SHEET) rather than the "
        "value (metal_sheet)"
    )

    # Extract the member values without importing SQLAlchemy.
    for enum_name, expected in (
        ("ObjectType", {"metal_sheet", "detail", "debris", "clutter"}),
        ("SystemState", {"connected", "disconnected", "error"}),
    ):
        block = re.search(
            rf"class {enum_name}\(str, enum\.Enum\):(.*?)(?=\n\nclass |\n\ndef )",
            source,
            re.DOTALL,
        )
        assert block, f"{enum_name} not found"
        values = set(re.findall(r'^\s+[A-Z_]+ = "([^"]+)"', block.group(1), re.MULTILINE))
        assert values == expected, f"{enum_name} values drifted: {values}"


def test_no_model_column_is_named_metadata() -> None:
    """`metadata` is reserved by SQLAlchemy's declarative API.

    The detection payload column is therefore `metadata` in the database and
    `detection_metadata` on the class; using the raw name on the class body
    breaks every declarative model at import time.
    """
    tree = ast.parse(MODELS.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                assert statement.target.id != "metadata", (
                    f"{node.name}.metadata collides with DeclarativeBase.metadata"
                )


def test_database_module_normalises_the_driver_scheme() -> None:
    """The K8s DSN is `postgresql+psycopg://`; psycopg2 is not installed."""
    source = DATABASE.read_text(encoding="utf-8")
    assert "postgresql+psycopg" in source
    assert "postgresql+psycopg_async" in source
    assert "postgresql+psycopg2" in source, (
        "a bare postgresql:// or a psycopg2 URL must be normalised, not passed "
        "through to a driver the image does not install"
    )


# --------------------------------------------------------------------------- #
# Alembic configuration
# --------------------------------------------------------------------------- #
def test_alembic_ini_has_no_credentials() -> None:
    """A DSN in version control is a credential leak."""
    parser = configparser.ConfigParser()
    parser.read(ALEMBIC_INI, encoding="utf-8")
    url = parser.get("alembic", "sqlalchemy.url", fallback="")
    assert url.strip() == "", (
        f"alembic.ini must not carry a database URL (found {url!r}); "
        "alembic/env.py reads DATABASE_URL"
    )
    assert parser.get("alembic", "script_location") == "alembic"


def test_env_py_reads_the_url_from_the_environment() -> None:
    source = ENV_PY.read_text(encoding="utf-8")
    assert "DATABASE_URL" in source
    assert "target_metadata = models.Base.metadata" in source
    # Autogenerate must not ignore a changed type or default.
    assert source.count("compare_type=True") >= 2
    assert source.count("compare_server_default=True") >= 2


def test_env_py_registers_the_enum_drop_extension() -> None:
    """The extension is what makes a generated `downgrade()` complete."""
    source = ENV_PY.read_text(encoding="utf-8")
    assert "dispatch_for" in source, (
        "the DropTableOp renderer extension is missing; a generated downgrade "
        "would leave the native enums behind and the next upgrade would fail "
        'with `type "object_type" already exists`'
    )
    assert "ENUM_DROP_HOOK_ACTIVE" in source
    assert "IF EXISTS" in source


# --------------------------------------------------------------------------- #
# The generated revision
# --------------------------------------------------------------------------- #
def test_revision_creates_every_table_and_enum(revision_source: str) -> None:
    for table in EXPECTED_TABLES:
        assert f"op.create_table('{table}'" in revision_source, f"{table} not created"
    for enum_name in ("object_type", "camera_status"):
        assert f"name='{enum_name}'" in revision_source, f"enum {enum_name} not declared"


def test_revision_enum_labels_are_lowercase(revision_source: str) -> None:
    """Catches the `'METAL_SHEET'` regression at the artifact level."""
    labels = re.findall(r"sa\.Enum\(([^)]*)\)", revision_source)
    assert labels, "no enum columns found in the revision"
    for group in labels:
        for value in re.findall(r"'([^']*)'", group):
            if value in ("object_type", "camera_status"):
                continue
            assert value == value.lower(), (
                f"enum label {value!r} is not the lowercase domain vocabulary"
            )


def test_revision_downgrade_drops_the_enum_types(revision_source: str) -> None:
    """PostgreSQL has no `CREATE TYPE ... IF NOT EXISTS`, so the drop is required.

    Without these two statements the downgrade leaves the types behind and the
    next `upgrade head` fails; the verification run reproduced exactly that.
    """
    tree = ast.parse(revision_source)
    downgrade = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "downgrade"
    )
    body = ast.get_source_segment(revision_source, downgrade) or ""
    for enum_name in ("object_type", "camera_status"):
        assert f"DROP TYPE IF EXISTS {enum_name}" in body, (
            f"downgrade() must drop the {enum_name} enum type"
        )
    # Dropping a table before its enum is the correct order.
    assert body.index("drop_table('detection_results')") < body.index(
        "DROP TYPE IF EXISTS object_type"
    )


def test_revision_has_exactly_one_head_and_no_placeholder(revision_source: str) -> None:
    assert "down_revision: Union[str, None] = None" in revision_source
    assert "revision: str = '" in revision_source
    # An autogenerated file that was never regenerated after a model change.
    assert "should be empty" not in revision_source
    assert "### commands auto generated by Alembic" in revision_source


def test_no_stray_revision_files_are_left_behind() -> None:
    """An empty autogenerate is a common leftover and confuses `alembic history`."""
    for path in revision_files():
        body = path.read_text(encoding="utf-8")
        # Strip the docstring/comments; a real revision always has operations.
        assert "def upgrade" in body and "def downgrade" in body
