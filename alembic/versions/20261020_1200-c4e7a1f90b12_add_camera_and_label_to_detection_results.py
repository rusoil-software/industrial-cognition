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
"""Add camera_id and label to detection_results.

`object_type` is a PostgreSQL enum with four values (metal_sheet, detail, debris,
clutter) that predates free-form prompts. The console lets the System User type any
object name - "robot", "person", "cardboard box" - so persisting detections with the
enum alone would fail for every label the enum does not know.

Two additions, both nullable so the change is additive:

* ``label`` keeps the prompt text exactly as typed. It is the value that answers "what
  did the operator ask for", while ``object_type`` remains the coarse category used by
  the existing constraints and indexes.
* ``camera_id`` makes per-camera history queryable. It could have gone into the JSONB
  metadata, but "the last hour for camera X" is the primary history query and JSONB is
  not indexable for ordering; the composite index below is the point.

Revision ID: c4e7a1f90b12
Revises: da2d8061e283
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c4e7a1f90b12"
down_revision: Union[str, None] = "da2d8061e283"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "detection_results",
        sa.Column("camera_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "detection_results",
        sa.Column("label", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_detection_results_camera_timestamp",
        "detection_results",
        ["camera_id", "timestamp"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_detection_results_camera_timestamp", table_name="detection_results"
    )
    op.drop_column("detection_results", "label")
    op.drop_column("detection_results", "camera_id")
