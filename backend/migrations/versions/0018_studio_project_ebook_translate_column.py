"""studio_projects: dedicated ebook_translate_json column

Revision ID: 0018_studio_project_ebook_translate_column
Revises: 0017_job_events_unique_seq
Create Date: 2026-10-30 00:00:00.000000

Adds ``studio_projects.ebook_translate_json TEXT DEFAULT NULL`` — the Ebook
Translator's per-chapter progress (api/routers/audiobook.py), moved out of
the shared ``state_json`` blob. ``state_json`` is replaced wholesale (no
merge) by api/routers/projects.py's PUT/POST, so an ordinary project autosave
used to erase translation progress it didn't even know existed. A dedicated
column makes that structurally impossible instead of relying on every
state_json writer to remember a sub-key it doesn't own.

Additive + idempotent (guarded by PRAGMA table_info, matching 0009's
pattern), so re-running on a fresh-install DB where ``_BASE_SCHEMA`` already
declares the column is a no-op. The same column is mirrored into
``core/db.py::_BASE_SCHEMA`` so fresh installs and migrated DBs converge on
an identical end-state, and a DB where alembic can't run at all picks it up
via ``_reconcile_additive_columns`` (the #552/#547 self-heal) — the dual-path
discipline.

This migration also does a one-time, best-effort backfill: any row that
already carries a legacy ``state_json["ebook_translate"]`` value (only
possible from local testing against this feature before this correction
shipped — never from a released version) is moved into the new column, then
stripped out of ``state_json`` so there's no stale duplicate. Every other key
already in ``state_json`` is preserved untouched.
"""
import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0018_studio_project_ebook_translate_column"
down_revision: Union[str, None] = "0017_job_events_unique_seq"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "studio_projects"
_COLUMN = "ebook_translate_json"


def _has_table(name: str) -> bool:
    bind = op.get_bind()
    row = bind.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
        {"n": name},
    ).fetchone()
    return row is not None


def _has_column(table: str, column: str) -> bool:
    bind = op.get_bind()
    rows = bind.execute(sa.text(f"PRAGMA table_info({table})")).fetchall()
    return any(r[1] == column for r in rows)


def _backfill_legacy_state(bind) -> None:
    """Move any state_json["ebook_translate"] into the new column, leaving
    every other state_json key untouched. Best-effort: a row with malformed
    JSON is left exactly as-is rather than raising and aborting the
    migration — this is a cleanup pass, not a required step."""
    rows = bind.execute(
        sa.text(f"SELECT id, state_json FROM {_TABLE} WHERE state_json IS NOT NULL")
    ).fetchall()
    for row_id, state_json in rows:
        try:
            state = json.loads(state_json)
        except (ValueError, TypeError):
            continue
        if not isinstance(state, dict) or "ebook_translate" not in state:
            continue
        legacy_value = state.pop("ebook_translate")
        bind.execute(
            sa.text(
                f"UPDATE {_TABLE} SET state_json=:sj, {_COLUMN}=:et WHERE id=:i"
            ),
            {
                "sj": json.dumps(state, ensure_ascii=False),
                "et": json.dumps(legacy_value, ensure_ascii=False),
                "i": row_id,
            },
        )


def upgrade() -> None:
    if not _has_table(_TABLE):
        return  # init_db creates the table with the column already present
    if not _has_column(_TABLE, _COLUMN):
        op.add_column(_TABLE, sa.Column(_COLUMN, sa.Text(), nullable=True))
    _backfill_legacy_state(op.get_bind())


def downgrade() -> None:
    if _has_table(_TABLE) and _has_column(_TABLE, _COLUMN):
        op.drop_column(_TABLE, _COLUMN)
