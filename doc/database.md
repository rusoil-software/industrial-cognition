# Database Schema & Migrations

This document describes the durable store: how the schema is defined, how it is
migrated, and how to run those migrations. It is the companion to
`doc/k8s.md` §10 ("Database migrations"), which covers running the migration as a
Kubernetes Job.

The tasklist entry this completes is Stage 1 "Database Schema Migration".

---

## 1. Source of truth

The ORM models in `src/api/storage/models.py` are the single source of truth.
The SQL schema is *derived* from them through Alembic autogenerate, never
hand-written:

```sh
DATABASE_URL='postgresql+psycopg://user:pass@host:5432/db' \
  alembic revision --autogenerate -m "add something"
```

The generated revision is then **reviewed and committed**. Autogenerate is
reliable here only because `alembic/env.py` enables `compare_type` and
`compare_server_default`; without both, a changed column type or default is
silently ignored, which is the drift a migration tool exists to catch.

```tree
alembic.ini                     # config only - deliberately holds NO database URL
alembic/
├── env.py                      # reads DATABASE_URL, exposes target_metadata
├── script.py.mako              # revision template
└── versions/
    └── <timestamp>-<rev>_initial_schema_detection_results_system_.py
src/api/storage/
├── models.py                   # ORM models + naming convention (source of truth)
└── database.py                 # sync + async engines, session factories
tests/db/
├── test_models.py              # static checks (no database needed)
└── test_migrations.py          # behavioural checks against real PostgreSQL
```

---

## 2. The two tables

Both come from `doc/vision.md` > "Data Model".

### `detection_results` — one recognised object in one frame

| Column | Type | Notes |
| --- | --- | --- |
| `id` | `uuid` PK | application-generated (`uuid4`), so a row can be logged before flush |
| `timestamp` | `timestamptz` | UTC, server-defaulted to `now()`, indexed |
| `frame_id` | `integer` | camera's sequential counter; **not** unique, two objects share a frame |
| `object_type` | `object_type` enum | `metal_sheet`, `detail`, `debris`, `clutter`, indexed |
| `coordinates` | `jsonb` | normalised box `{x, y, width, height}`, each in `[0, 1]` |
| `confidence` | `double precision` | `CHECK 0 <= confidence <= 1` |
| `metadata` | `jsonb` | model version, latency, workspace state (`NO_TARGET`, `OUT_OF_REACH`, `HUMAN_ON_SITE`) |

Composite index `ix_detection_results_type_timestamp (object_type, timestamp)`
covers the two documented access patterns: time-range history and per-type
history ("last 30 days of metal sheets", the 30-day retention policy).

The class attribute is `detection_metadata` while the column is `metadata`:
`metadata` is reserved by SQLAlchemy's declarative API and assigning to it breaks
model import.

### `system_metadata` — system state, counters and configuration

| Column | Type | Notes |
| --- | --- | --- |
| `id` | `uuid` PK | |
| `key` | `varchar(128)` | `UNIQUE`; e.g. `camera_status`, `processing_fps` |
| `value` | `text` | scalar form, so a status needs no JSON quoting |
| `value_json` | `jsonb` | structured form (the `configuration` snapshot) |
| `camera_status` | `camera_status` enum | promoted from `value`; `connected`, `disconnected`, `error`, indexed |
| `processing_fps` | `double precision` | promoted; the 60 FPS target is measured against it |
| `last_heartbeat` | `timestamptz` | indexed; a stale value is the primary `COMM_ERROR` signal |
| `updated_at` | `timestamptz` | server-defaulted, `onupdate` |

Key/value rather than one column per field, because the documented attributes are
only a starting set and adding a metric on an industrial line must not require a
migration. The values that are actually queried or graphed are promoted to
indexed columns alongside the payload.

### Enum vocabulary

Native PostgreSQL enums are used, and they store the **values**, not the Python
member names:

```sql
CREATE TYPE object_type AS ENUM ('metal_sheet', 'detail', 'debris', 'clutter');
CREATE TYPE camera_status AS ENUM ('connected', 'disconnected', 'error');
```

This is load-bearing. SQLAlchemy's `Enum` persists the member *name* by default,
which would store `'METAL_SHEET'` while `src/camera/schemas.py` and
`src/vision/service.py` emit `"metal_sheet"` — the database would hold a
vocabulary no other part of the system uses. `models._enum()` sets
`values_callable` to pin the labels. The first live run produced exactly the
wrong vocabulary, which is why `tests/db` asserts it.

---

## 3. Migrating

```sh
# 1. what is the database currently at?
DATABASE_URL=... alembic current

# 2. what is pending?
DATABASE_URL=... alembic history

# 3. apply
DATABASE_URL=... alembic upgrade head

# 4. review the SQL first, without touching the database
DATABASE_URL=... alembic upgrade head --sql
```

`DATABASE_URL` is normalised to psycopg 3 by `src/api/storage/database.py`, so
all of these forms work: `postgresql://…`, `postgresql+psycopg://…`,
`postgresql+psycopg2://…`. A non-PostgreSQL scheme raises `ValueError` at
startup rather than failing on the first request.

In Kubernetes the same command runs as a Job, so N API replicas never race:

```sh
./k8s/scripts/deploy.sh --overlay prod --migrate
```

The first live round-trip reproduced the `DuplicateObject` failure above, which
is why the fix lives in `alembic/env.py` rather than in each revision.

### Downgrade

```sh
DATABASE_URL=... alembic downgrade base
```

A `downgrade()` must drop the native enum types as well as the tables.
PostgreSQL has no `CREATE TYPE ... IF NOT EXISTS`, so leaving a type behind makes
the next `upgrade` fail with:

```
psycopg.errors.DuplicateObject: type "object_type" already exists
```

Alembic does not emit the `DROP TYPE` itself, because a type is not owned by any
single table. `alembic/env.py` therefore extends Alembic's `DropTableOp` renderer
so every generated `downgrade()` ends each dropped table's enum with
`op.execute('DROP TYPE IF EXISTS <name>')`.

---

## 4. Testing

```sh
pytest tests/db                          # static checks only (no database)
pytest -m db tests/db                    # needs DATABASE_URL / a PostgreSQL
DATABASE_URL=... pytest -m db tests/db
```

* `tests/db/test_models.py` needs only PyYAML. It reads the model source and the
  generated revision (AST, not import) and asserts the documented tables, the
  enum vocabulary, the `metadata` naming collision, the absence of a DSN in
  `alembic.ini`, and that `downgrade()` drops the enums.
* `tests/db/test_migrations.py` needs a real PostgreSQL. It applies the
  migration, inspects `information_schema`, asserts the enum labels, does a
  **full downgrade → upgrade round trip** (the `DuplicateObject` guard), proves a
  second `compare_metadata` is empty, then inserts and reads back a detection
  and a system-metadata row and checks the constraints reject bad values.

The suite skips itself when `DATABASE_URL` is unset or nothing answers, so a bare
`pytest` on a workstation stays green. CI provides a PostgreSQL service
container.

---

## 5. Why psycopg 3

`requirements/api.txt` and `requirements/base.txt` pin
`psycopg[binary]==3.2.9`, matching the `postgresql+psycopg://` DSN that
`k8s/base/app-config.yaml` provides. **psycopg2 is not installed**, so a bare
`postgresql://` URL is normalised to psycopg 3 rather than silently reaching for
a driver that is not in the image. The async engine reuses the same value as
`postgresql+psycopg_async://`, so the FastAPI request path and the synchronous
worker/migration path share one connection string.

`requirements/prod.txt` is the union of the runtime dependency sets, because both
production images install it. It excludes `dev.txt`, `export.txt` and `k8s.txt`
tiers. The two serving images still differ in exactly one respect: `api` installs
`onnxruntime` (CPU) while `vision` installs `onnxruntime-gpu`; those two ship the
same import package and cannot coexist in one environment. See `doc/k8s.md` §2.
