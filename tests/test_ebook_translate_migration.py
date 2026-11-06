"""studio_projects.ebook_translate_json — migration 0018 safety.

Covers (corrective review, Priority 1 / Point C): fresh DB creation,
upgrading an existing pre-0018 DB, repeated migration execution, backfilling
pre-existing legacy ``state_json["ebook_translate"]`` data, no loss of
unrelated state_json information, and convergence via the schema-reconcile
self-heal path when alembic can't run — mirroring
tests/test_db_migration_safety.py's and tests/test_db_schema_reconcile.py's
existing harnesses rather than inventing a new one.
"""
import json
import sqlite3

import core.db as db_module
from core.db import _BASE_SCHEMA, _reconcile_additive_columns, _run_alembic_upgrade, init_db

# A pre-0018 studio_projects — missing ebook_translate_json entirely.
_LEGACY_STUDIO_PROJECTS = """
    CREATE TABLE studio_projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        video_path TEXT,
        audio_path TEXT,
        duration REAL,
        state_json TEXT,
        created_at REAL,
        updated_at REAL
    );
"""


def _cols(db_path):
    with sqlite3.connect(str(db_path)) as conn:
        return {r[1] for r in conn.execute("PRAGMA table_info(studio_projects)")}


def _base_schema_cols():
    canon = sqlite3.connect(":memory:")
    try:
        canon.executescript(_BASE_SCHEMA)
        return {r[1] for r in canon.execute("PRAGMA table_info(studio_projects)")}
    finally:
        canon.close()


def _seed_legacy_db(path, *, stamp="0017_job_events_unique_seq"):
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_LEGACY_STUDIO_PROJECTS)
        legacy_state = json.dumps({
            "dub_editor": {"tracks": ["keep-me"]},
            "ebook_translate": {
                "chapters": {"0": {"status": "done", "source_title": "Ch1"}},
                "total_chapters": 1, "target_lang": "es", "provider": "argos",
            },
        })
        conn.execute(
            "INSERT INTO studio_projects (id, name, video_path, audio_path, duration, "
            "state_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            ("p-legacy", "Book", None, None, None, legacy_state, 1.0, 1.0),
        )
        conn.execute(
            "INSERT INTO studio_projects (id, name, video_path, audio_path, duration, "
            "state_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            ("p-no-state", "NoTranslation", None, None, None, None, 1.0, 1.0),
        )
        if stamp is not None:
            conn.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
            conn.execute("INSERT INTO alembic_version VALUES (?)", (stamp,))
        conn.commit()
    finally:
        conn.close()


def _row(db_path, project_id):
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM studio_projects WHERE id=?", (project_id,)).fetchone()
    finally:
        conn.close()


# ── Fresh DB ─────────────────────────────────────────────────────────────


def test_fresh_db_has_the_column(tmp_path, monkeypatch):
    db = tmp_path / "fresh.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db))
    init_db()
    assert "ebook_translate_json" in _cols(db)


# ── Existing DB upgrade (the migration itself) ──────────────────────────


def test_upgrade_adds_column_to_existing_db(tmp_path, monkeypatch):
    db = tmp_path / "legacy.db"
    _seed_legacy_db(db)
    monkeypatch.setattr(db_module, "DB_PATH", str(db))

    assert "ebook_translate_json" not in _cols(db)
    _run_alembic_upgrade()
    assert "ebook_translate_json" in _cols(db)


# ── Repeated migration execution ────────────────────────────────────────


def test_upgrade_is_idempotent(tmp_path, monkeypatch):
    db = tmp_path / "legacy.db"
    _seed_legacy_db(db)
    monkeypatch.setattr(db_module, "DB_PATH", str(db))

    _run_alembic_upgrade()
    first = _cols(db)
    _run_alembic_upgrade()  # already at head: must be a clean no-op
    assert _cols(db) == first


def test_fresh_db_migrated_twice_is_a_no_op(tmp_path, monkeypatch):
    db = tmp_path / "fresh.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db))
    init_db()
    cols_once = _cols(db)
    init_db()
    assert _cols(db) == cols_once


# ── Backfill of existing translation state ──────────────────────────────


def test_backfill_moves_legacy_state_and_preserves_siblings(tmp_path, monkeypatch):
    db = tmp_path / "legacy.db"
    _seed_legacy_db(db)
    monkeypatch.setattr(db_module, "DB_PATH", str(db))

    _run_alembic_upgrade()

    row = _row(db, "p-legacy")
    moved = json.loads(row["ebook_translate_json"])
    assert moved["chapters"]["0"]["status"] == "done"
    assert moved["total_chapters"] == 1

    # No accidental loss of unrelated state_json information.
    remaining_state = json.loads(row["state_json"])
    assert remaining_state["dub_editor"]["tracks"] == ["keep-me"]
    # The legacy key is cleaned up — no stale duplicate left behind.
    assert "ebook_translate" not in remaining_state

    # A row with no state_json at all is untouched, not an error.
    row2 = _row(db, "p-no-state")
    assert row2["ebook_translate_json"] is None
    assert row2["state_json"] is None


def test_backfill_is_idempotent(tmp_path, monkeypatch):
    db = tmp_path / "legacy.db"
    _seed_legacy_db(db)
    monkeypatch.setattr(db_module, "DB_PATH", str(db))

    _run_alembic_upgrade()
    first_translate = _row(db, "p-legacy")["ebook_translate_json"]
    _run_alembic_upgrade()  # nothing left to backfill — must not duplicate/clear it
    second_translate = _row(db, "p-legacy")["ebook_translate_json"]
    assert first_translate == second_translate


# ── Compatibility with the schema-reconcile self-heal ───────────────────


def test_reconcile_adds_column_when_alembic_cannot_run(tmp_path):
    db = tmp_path / "reconcile.db"
    with sqlite3.connect(str(db)) as conn:
        conn.executescript(_LEGACY_STUDIO_PROJECTS)
        conn.execute(
            "INSERT INTO studio_projects (id, name, created_at, updated_at) "
            "VALUES ('p1', 'Book', 1.0, 1.0)"
        )
        conn.commit()
        _reconcile_additive_columns(conn)
    assert _cols(db) == _base_schema_cols()
    # The existing row survives the reconcile.
    with sqlite3.connect(str(db)) as conn:
        assert conn.execute(
            "SELECT name FROM studio_projects WHERE id='p1'").fetchone()[0] == "Book"


def _alembic_cfg(db_path):
    import os

    from alembic.config import Config

    root = os.path.dirname(os.path.dirname(os.path.abspath(db_module.__file__)))
    cfg = Config(os.path.join(os.path.dirname(root), "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


# ── Downgrade ───────────────────────────────────────────────────────────


def test_downgrade_removes_column_from_a_fresh_install_db(tmp_path, monkeypatch):
    """Regression: _BASE_SCHEMA once carried a comma-laden SQL comment inside
    the studio_projects column list. SQLite stores CREATE TABLE text verbatim
    and DROP COLUMN on the last column mis-parsed that comment, so rolling back
    0018 on a fresh-install DB failed with "incomplete input"."""
    from alembic import command

    db = tmp_path / "fresh.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db))
    init_db()
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "INSERT INTO studio_projects (id, name, state_json, created_at, updated_at, "
            "ebook_translate_json) VALUES ('p1', 'Book', '{\"k\": 1}', 1.0, 1.0, '{}')"
        )
        conn.commit()

    command.downgrade(_alembic_cfg(db), "0017_job_events_unique_seq")
    assert "ebook_translate_json" not in _cols(db)
    assert _row(db, "p1")["state_json"] == '{"k": 1}'  # other data survives

    command.upgrade(_alembic_cfg(db), "head")  # and the round trip converges
    assert _cols(db) == _base_schema_cols()
    assert _row(db, "p1")["name"] == "Book"


def test_reconcile_is_idempotent_for_studio_projects(tmp_path):
    db = tmp_path / "reconcile_twice.db"
    with sqlite3.connect(str(db)) as conn:
        conn.executescript(_LEGACY_STUDIO_PROJECTS)
        conn.commit()
        _reconcile_additive_columns(conn)
        after_first = _cols(db)
        _reconcile_additive_columns(conn)
        after_second = _cols(db)
    assert after_first == after_second == _base_schema_cols()
