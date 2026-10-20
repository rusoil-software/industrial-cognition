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
# date   : 2026-10-20
# ==============================================================================
"""object_type: enum -> free text, backfilled from the real label.

The column was a four-value enum (metal_sheet, detail, debris, clutter) from before prompts
were free-form. The console lets the System User type any object name, so every real detection
was persisted as ``clutter`` - a category that told the operator nothing and made history
useless for the one question it exists to answer: what did we find.

This migration converts the column to text and backfills it from ``label``, which already holds
what the operator typed (added in c4e7a1f90b12). After it, ``object_type`` is the label, and
``label`` remains the exact original text for auditing.

Revision ID: d5f8b2a1c9e3
Revises: c4e7a1f90b12
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d5f8b2a1c9e3"
down_revision: Union[str, None] = "c4e7a1f90b12"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

#: The vocabulary the enum had, needed to rebuild it on downgrade.
LEGACY_LABELS = ("metal_sheet", "detail", "debris", "clutter")


def upgrade() -> None:
    # 1. Widen the column first: the backfill writes arbitrary prompt text, which the enum
    #    would reject. `::text` also drops the enum's ordering rules, which nothing used.
    op.alter_column(
        "detection_results",
        "object_type",
        existing_type=sa.Enum(*LEGACY_LABELS, name="object_type"),
        type_=sa.String(length=64),
        existing_nullable=False,
        postgresql_using="object_type::text",
    )
    # 2. The real label wins; rows written before `label` existed keep their enum value.
    op.execute(
        "UPDATE detection_results "
        "SET object_type = lower(btrim(label)) "
        "WHERE label IS NOT NULL AND btrim(label) <> ''"
    )
    # 3. Nothing uses the type any more.
    op.execute("DROP TYPE IF EXISTS object_type")


def downgrade() -> None:
    op.execute(
        "CREATE TYPE object_type AS ENUM ("
        + ", ".join(f"'{label}'" for label in LEGACY_LABELS)
        + ")"
    )
    # Anything outside the old vocabulary has no faithful representation; `clutter` is what it
    # used to be, so that is what it becomes again rather than failing the downgrade.
    op.execute(
        "UPDATE detection_results SET object_type = 'clutter' "
        "WHERE object_type NOT IN ("
        + ", ".join(f"'{label}'" for label in LEGACY_LABELS)
        + ")"
    )
    op.alter_column(
        "detection_results",
        "object_type",
        existing_type=sa.String(length=64),
        type_=sa.Enum(*LEGACY_LABELS, name="object_type"),
        existing_nullable=False,
        postgresql_using="object_type::object_type",
    )
