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
"""Prune old detection history.

Persistence is sampled (one frame per camera per second, and only frames with detections),
which bounds the *rate* but not the total: an unbounded table is still an unbounded table,
and on a factory floor the process runs for months. This is the other half of item 4.

Run as a Kubernetes CronJob (k8s/base/cronjob-retention.yaml) using the API image:

    python -m src.api.maintenance.retention                # prune rows older than the default
    python -m src.api.maintenance.retention --days 7       # override the window
    python -m src.api.maintenance.retention --dry-run      # report, delete nothing

Two decisions worth stating:

* **Delete in batches.** One `DELETE ... WHERE timestamp < cutoff` on millions of rows takes
  a long lock, bloats the WAL and can stall the writer that is persisting live frames. The
  loop deletes at most `batch_size` rows per statement and commits between batches, so a
  prune is a series of short transactions.
* **Keep the cutoff computation separate and pure.** `cutoff_for(days, now)` is what the tests
  assert; the SQL is exercised against a real database when one is available.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from src.api.maintenance.policy import DEFAULT_RETENTION_DAYS, cutoff_for
from src.api.storage.models import DetectionResult

__all__ = ["cutoff_for", "count_older_than", "prune_detections"]

logger = logging.getLogger(__name__)

#: Rows per statement. Large enough to make progress, small enough that each transaction is
#: short and the live writer is never blocked for long.
DEFAULT_BATCH_SIZE = 5000


def count_older_than(session: Session, cutoff: datetime) -> int:
    """Rows that a prune would remove - what `--dry-run` reports."""
    return int(
        session.scalar(
            select(func.count())
            .select_from(DetectionResult)
            .where(DetectionResult.timestamp < cutoff)
        )
        or 0
    )


def prune_detections(
    session: Session,
    *,
    days: int = DEFAULT_RETENTION_DAYS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
    now: Optional[datetime] = None,
) -> int:
    """Delete detections older than *days*; returns the number of rows removed.

    Commits between batches: holding one transaction over the whole table would keep locks
    and WAL for the entire prune, which on a busy workcell is long enough to matter.
    """
    cutoff = cutoff_for(days, now)
    if dry_run:
        return count_older_than(session, cutoff)

    removed = 0
    while True:
        # A subselect keeps the DELETE itself index-friendly and bounded.
        batch = session.scalars(
            select(DetectionResult.id)
            .where(DetectionResult.timestamp < cutoff)
            .limit(batch_size)
        ).all()
        if not batch:
            break

        session.execute(delete(DetectionResult).where(DetectionResult.id.in_(batch)))
        session.commit()
        removed += len(batch)
        logger.info("pruned %d detection row(s) so far", removed)

        if len(batch) < batch_size:
            break

    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=int(os.getenv("DETECTION_RETENTION_DAYS", DEFAULT_RETENTION_DAYS)),
        help="keep this many days of history",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("DETECTION_RETENTION_BATCH", DEFAULT_BATCH_SIZE)),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report how many rows would be removed, then exit",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.api.storage.database import make_session_factory

    if not os.getenv("DATABASE_URL"):
        print("error: DATABASE_URL is not set", file=sys.stderr)
        return 2

    session = make_session_factory()()
    try:
        removed = prune_detections(
            session,
            days=args.days,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    finally:
        session.close()

    cutoff = cutoff_for(args.days)
    verb = "would remove" if args.dry_run else "removed"
    logger.info(
        "%s %d detection row(s) older than %s (retention %d day(s))",
        verb,
        removed,
        cutoff.isoformat(),
        args.days,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
