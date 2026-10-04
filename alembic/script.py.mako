"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

Copyright (c), The Rusoil Software Development Team. All rights reserved.
Licensed under the Apache License, Version 2.0 - see alembic/env.py for the full
header.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    """Apply the migration."""
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """Revert the migration."""
    ${downgrades if downgrades else "pass"}
