"""API tests for POST /audiobook/translate + GET /audiobook/translate/{id} —
the Ebook Translator's chapter-by-chapter translation route (Step 3).

Mounts only the audiobook router (no ``main`` import) so it runs locally
without the main+torch segfault; conftest.py provides the hermetic data dir
and OMNIVOICE_DATA_DIR, so studio_projects persistence is a real (throwaway)
SQLite file, not mocked — that's the whole point of testing persistence here.

Route-level validation and the full happy/partial-failure/retry/status flow
go through TestClient, which drives the SSE POST to completion and hands back
the whole event stream as text. The generator's disconnect-cancellation and
request-level-setup-error branches are exercised by calling
``_translate_longform_sse`` directly instead — collecting a completed SSE
response doesn't let a test simulate a mid-stream client disconnect.
"""
from __future__ import annotations

import json
import time
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.db import db_conn, init_db
from services import ebook_translate, translation_engines
from services.audiobook import AudiobookPlan, Chapter, Span
from api.routers import audiobook as ab

init_db()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(ab.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_translate_locks():
    """_ACTIVE_TRANSLATE_RUNS is a module-level dict shared across the whole
    test session — clear it before AND after every test so one test's lock
    state (leaked by a bug, or deliberately left by a lifecycle test) can
    never leak into another test's assertions."""
    ab._ACTIVE_TRANSLATE_RUNS.clear()
    yield
    ab._ACTIVE_TRANSLATE_RUNS.clear()


@pytest.fixture(autouse=True)
def _default_provider_ready(monkeypatch):
    """Every test that reaches the provider-configuration pre-check gets a
    deterministic "ready" answer regardless of what's actually importable in
    this environment; tests that want the unready/unknown path override it
    explicitly."""
    monkeypatch.setattr(translation_engines, "get_engine", lambda p: {"id": p})
    monkeypatch.setattr(translation_engines, "is_ready", lambda p: True)


def _make_project(name: str = "Book") -> str:
    project_id = uuid.uuid4().hex[:8]
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO studio_projects (id, name, created_at, updated_at) VALUES (?,?,?,?)",
            (project_id, name, time.time(), time.time()),
        )
    return project_id


_SCRIPT = "# Chapter One\nHello there.\n\n# Chapter Two\nGoodbye now.\n"


def _events(resp) -> list:
    out = []
    for block in resp.text.split("\n\n"):
        block = block.strip()
        if block.startswith("data:"):
            out.append(json.loads(block[len("data:"):].strip()))
    return out


def _ok_translate(calls=None):
    async def fake(segments, *, source_lang, target_lang, provider, glossary=None, instructions=None):
        if calls is not None:
            calls.append(list(segments))
        return [{"id": s.id, "text": f"[{target_lang}] {s.text}"} for s in segments]
    return fake


# ── Route-level validation ──────────────────────────────────────────────────


def test_unknown_project_404(client):
    resp = client.post("/audiobook/translate", json={
        "project_id": "does-not-exist", "text": _SCRIPT, "target_lang": "es",
    })
    assert resp.status_code == 404


def test_missing_target_lang_400(client):
    pid = _make_project()
    resp = client.post("/audiobook/translate", json={
        "project_id": pid, "text": _SCRIPT, "target_lang": "  ",
    })
    assert resp.status_code == 400


def test_unknown_provider_400(client, monkeypatch):
    monkeypatch.setattr(translation_engines, "get_engine", lambda p: None)
    pid = _make_project()
    resp = client.post("/audiobook/translate", json={
        "project_id": pid, "text": _SCRIPT, "target_lang": "es",
        "provider": "not-a-real-engine",
    })
    assert resp.status_code == 400


def test_unconfigured_non_llm_provider_409(client, monkeypatch):
    monkeypatch.setattr(translation_engines, "is_ready", lambda p: False)
    pid = _make_project()
    resp = client.post("/audiobook/translate", json={
        "project_id": pid, "text": _SCRIPT, "target_lang": "es", "provider": "google",
    })
    assert resp.status_code == 409


def test_llm_provider_disabled_409(client, monkeypatch):
    from services import llm_skills
    monkeypatch.setattr(llm_skills, "resolve_skill",
                        lambda sid: type("R", (), {"ready": False, "reason": "disabled"})())
    pid = _make_project()
    resp = client.post("/audiobook/translate", json={
        "project_id": pid, "text": _SCRIPT, "target_lang": "es", "provider": "openai",
    })
    assert resp.status_code == 409
    assert "Settings -> LLM Skills" in resp.json()["detail"]


def test_too_many_chapters_422(client, monkeypatch):
    monkeypatch.setattr(ab, "_MAX_CHAPTERS", 1)
    pid = _make_project()
    resp = client.post("/audiobook/translate", json={
        "project_id": pid, "text": _SCRIPT, "target_lang": "es",
    })
    assert resp.status_code == 422


# ── Happy path: SSE progress + persistence ──────────────────────────────────


def test_translate_happy_path_streams_and_persists(client, monkeypatch):
    calls = []
    monkeypatch.setattr(ebook_translate, "translate_segments", _ok_translate(calls))
    pid = _make_project()

    resp = client.post("/audiobook/translate", json={
        "project_id": pid, "text": _SCRIPT, "target_lang": "es", "source_lang": "en",
    })
    assert resp.status_code == 200
    events = _events(resp)

    assert events[0]["type"] == "started" and events[0]["chapters"] == 2
    chapter_events = [e for e in events if e["type"] == "chapter"]
    assert [e["index"] for e in chapter_events] == [0, 1]
    assert all(e["cached"] is False for e in chapter_events)
    done = events[-1]
    assert done["type"] == "done"
    assert done["complete"] is True
    assert done["translated_chapters"] == 2
    assert done["failed_chapters"] == []
    assert len(calls) == 2  # one translate_segments call per chapter

    status = ab.get_translate_status(pid)
    assert status["complete"] is True
    assert status["translated_chapters"] == 2
    assert status["failed_chapters"] == []
    titles = {c["index"]: c for c in status["chapters"]}
    assert titles[0]["translated_title"] == "[es] Chapter One"
    assert titles[0]["source_title"] == "Chapter One"
    # Original and translated text are both present, independently.
    assert titles[0]["source_spans"][0]["text"] == "Hello there."
    assert titles[0]["translated_spans"][0]["text"] == "[es] Hello there."


# ── Partial failure: never silently "done" ──────────────────────────────────


def test_partial_failure_marks_chapter_failed_never_done(client, monkeypatch):
    async def fake(segments, *, source_lang, target_lang, provider, glossary=None, instructions=None):
        if any("Goodbye" in s.text for s in segments):
            return [{"id": s.id, "text": s.text, "error": "provider exploded"} for s in segments]
        return [{"id": s.id, "text": f"[{target_lang}] {s.text}"} for s in segments]
    monkeypatch.setattr(ebook_translate, "translate_segments", fake)
    pid = _make_project()

    resp = client.post("/audiobook/translate", json={
        "project_id": pid, "text": _SCRIPT, "target_lang": "es",
    })
    events = _events(resp)
    chapter_errors = [e for e in events if e["type"] == "chapter_error"]
    assert len(chapter_errors) == 1 and chapter_errors[0]["index"] == 1
    assert "provider exploded" in chapter_errors[0]["error"]

    done = events[-1]
    assert done["type"] == "done"
    assert done["complete"] is False
    assert done["failed_chapters"] == [1]

    status = ab.get_translate_status(pid)
    assert status["complete"] is False
    failed = next(c for c in status["chapters"] if c["index"] == 1)
    assert failed["status"] == "failed"
    # The degraded segment's text (== source text) must never be stored as
    # the chapter's translated content — that would silently read as done.
    assert failed["translated_title"] is None
    assert failed["translated_spans"] is None
    done_chapter = next(c for c in status["chapters"] if c["index"] == 0)
    assert done_chapter["status"] == "done"


# ── Retry: only failed/unfinished chapters are retranslated ─────────────────


def test_retry_skips_done_chapters_and_retranslates_failed(client, monkeypatch):
    attempt = {"n": 0}

    async def flaky(segments, *, source_lang, target_lang, provider, glossary=None, instructions=None):
        if any("Goodbye" in s.text for s in segments) and attempt["n"] == 0:
            attempt["n"] += 1
            return [{"id": s.id, "text": s.text, "error": "boom"} for s in segments]
        return [{"id": s.id, "text": f"[{target_lang}] {s.text}"} for s in segments]
    monkeypatch.setattr(ebook_translate, "translate_segments", flaky)
    pid = _make_project()

    body = {"project_id": pid, "text": _SCRIPT, "target_lang": "es"}
    first = _events(client.post("/audiobook/translate", json=body))
    assert first[-1]["failed_chapters"] == [1]

    calls = []
    original = ebook_translate.translate_segments
    async def counting(segments, **kwargs):
        calls.append(segments)
        return await original(segments, **kwargs)
    monkeypatch.setattr(ebook_translate, "translate_segments", counting)

    second = _events(client.post("/audiobook/translate", json=body))
    chapter_events = [e for e in second if e["type"] == "chapter"]
    # Chapter 0 is replayed from the persisted cache, not retranslated.
    cached = {e["index"]: e["cached"] for e in chapter_events}
    assert cached[0] is True
    assert len(calls) == 1  # only chapter 1 (previously failed) called translate_segments
    assert second[-1]["complete"] is True

    status = ab.get_translate_status(pid)
    assert status["complete"] is True
    assert status["failed_chapters"] == []


# ── Resume after a "restart": a fresh app/client, same DB ───────────────────


def test_state_survives_a_fresh_app_instance(monkeypatch):
    monkeypatch.setattr(ebook_translate, "translate_segments", _ok_translate())
    app1 = FastAPI()
    app1.include_router(ab.router)
    client1 = TestClient(app1)
    pid = _make_project()
    client1.post("/audiobook/translate", json={
        "project_id": pid, "text": _SCRIPT, "target_lang": "es",
    })

    # Simulate the backend process restarting: a brand new app/router/client
    # object, sharing nothing in-process — only the SQLite file on disk.
    app2 = FastAPI()
    app2.include_router(ab.router)
    client2 = TestClient(app2)
    resp = client2.get(f"/audiobook/translate/{pid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["complete"] is True
    assert body["translated_chapters"] == 2


def test_status_404_for_unknown_project(client):
    assert client.get("/audiobook/translate/does-not-exist").status_code == 404


def test_status_for_project_with_no_run_yet_is_not_complete(client):
    pid = _make_project()
    body = client.get(f"/audiobook/translate/{pid}").json()
    assert body["total_chapters"] == 0
    assert body["complete"] is False


# ── Cancellation: disconnect stops cleanly without losing progress ─────────


@pytest.mark.asyncio
async def test_disconnect_stops_after_current_chapter_and_keeps_progress(monkeypatch):
    monkeypatch.setattr(ebook_translate, "translate_segments", _ok_translate())
    pid = _make_project()
    plan = AudiobookPlan(chapters=[
        Chapter(title="One", spans=[Span(voice_id=None, text="Hello")]),
        Chapter(title="Two", spans=[Span(voice_id=None, text="World")]),
    ])

    seen = {"n": 0}

    async def is_disconnected():
        seen["n"] += 1
        return seen["n"] > 1  # disconnect reported just before chapter index 1

    events = []
    async for raw in ab._translate_longform_sse(
        plan, project_id=pid, source_lang="en", target_lang="es", provider="argos",
        glossary=None, instructions=None, job_id="job-disconnect-test",
        is_disconnected=is_disconnected,
    ):
        events.append(json.loads(raw[len("data: "):].strip()))

    assert events[-1]["type"] == "stopped"
    assert events[-1]["translated"] == 1

    status = ab.get_translate_status(pid)
    assert status["translated_chapters"] == 1
    assert status["total_chapters"] == 2
    assert status["complete"] is False  # one chapter never ran


# ── Request-level setup error: abort the whole run, not just one chapter ───


@pytest.mark.asyncio
async def test_setup_error_aborts_whole_run_without_marking_any_chapter(monkeypatch):
    async def fake(segments, **kwargs):
        raise ebook_translate.UnsupportedLanguageError("no Argos pack for en -> xx")
    monkeypatch.setattr(ebook_translate, "translate_segments", fake)
    pid = _make_project()
    plan = AudiobookPlan(chapters=[
        Chapter(title="One", spans=[Span(voice_id=None, text="Hello")]),
        Chapter(title="Two", spans=[Span(voice_id=None, text="World")]),
    ])

    events = []
    async for raw in ab._translate_longform_sse(
        plan, project_id=pid, source_lang="en", target_lang="xx", provider="argos",
        glossary=None, instructions=None, job_id="job-setup-error-test",
    ):
        events.append(json.loads(raw[len("data: "):].strip()))

    assert events[-1] == {"type": "error", "error": "no Argos pack for en -> xx"}
    status = ab.get_translate_status(pid)
    assert status["total_chapters"] == 0  # no chapter was ever persisted


# ── Mandatory concurrency tests (corrective review, Priority 2 / Point B) ──


def test_second_simultaneous_request_for_same_project_gets_409(client):
    """#1/#2: a second request for a project that's already running is
    rejected, not queued or silently allowed to race the first."""
    pid = _make_project()
    assert ab._acquire_translate_lock(pid, "job-first-run")
    try:
        resp = client.post("/audiobook/translate", json={
            "project_id": pid, "text": _SCRIPT, "target_lang": "es",
        })
        assert resp.status_code == 409
    finally:
        ab._release_translate_lock(pid, "job-first-run")


def test_concurrent_writers_do_not_lose_each_others_chapters(client):
    """#3: even bypassing the request-level guard entirely, the per-chapter
    persistence merge (not a blind full-dict replace) means one writer's
    completed chapter can never erase another's — exercised directly against
    the function two interleaved runs would actually call."""
    pid = _make_project()
    rec_a = {"index": 0, "key": "ka", "status": "done", "source_title": "A",
             "source_spans": [], "translated_title": "A-es",
             "translated_spans": [], "error": None, "updated_at": time.time()}
    rec_b = {"index": 1, "key": "kb", "status": "done", "source_title": "B",
             "source_spans": [], "translated_title": "B-es",
             "translated_spans": [], "error": None, "updated_at": time.time()}
    # "Run A" persists chapter 0, then — simulating a second writer whose own
    # in-memory view never included A's write — "run B" persists chapter 1
    # via a completely separate call, re-reading the DB fresh each time.
    ab._persist_chapter_record(pid, 0, rec_a, target_lang="es", provider="argos", total=2)
    ab._persist_chapter_record(pid, 1, rec_b, target_lang="es", provider="argos", total=2)

    status = ab.get_translate_status(pid)
    assert status["translated_chapters"] == 2
    assert {c["index"] for c in status["chapters"]} == {0, 1}


def test_new_request_succeeds_once_previous_run_has_finished(client, monkeypatch):
    """#4: completing a run must not leave anything that blocks the next one."""
    monkeypatch.setattr(ebook_translate, "translate_segments", _ok_translate())
    pid = _make_project()
    body = {"project_id": pid, "text": _SCRIPT, "target_lang": "es"}

    first = client.post("/audiobook/translate", json=body)
    assert first.status_code == 200
    assert pid not in ab._ACTIVE_TRANSLATE_RUNS

    second = client.post("/audiobook/translate", json=body)
    assert second.status_code == 200


def test_run_where_every_chapter_fails_still_releases_the_lock(client, monkeypatch):
    """#5a, the per-chapter-isolation path: every chapter individually
    failing (an ordinary provider error, degraded per-segment) still ends
    the run as `done` with failed_chapters populated — not a crash — and
    must not leave the lock behind either way."""
    async def boom(segments, *, target_lang, **kwargs):
        return [{"id": s.id, "text": s.text, "error": "provider blew up"} for s in segments]
    monkeypatch.setattr(ebook_translate, "translate_segments", boom)
    pid = _make_project()
    body = {"project_id": pid, "text": _SCRIPT, "target_lang": "es"}

    resp = client.post("/audiobook/translate", json=body)
    assert resp.status_code == 200
    done = _events(resp)[-1]
    assert done["type"] == "done" and done["complete"] is False
    assert done["failed_chapters"] == [0, 1]
    assert pid not in ab._ACTIVE_TRANSLATE_RUNS

    # A follow-up request for the same project is not blocked by a leaked lock.
    assert client.post("/audiobook/translate", json=body).status_code != 409


def test_run_where_persistence_itself_raises_still_releases_the_lock(client, monkeypatch):
    """#5a, the genuine outer except-Exception path: an unexpected failure
    OUTSIDE the per-chapter isolation (not a provider error — an
    infrastructure fault) surfaces as a terminal `error` SSE event, and
    still must not leave the lock behind."""
    monkeypatch.setattr(ebook_translate, "translate_segments", _ok_translate())

    def boom(*_a, **_k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(ab, "_persist_chapter_record", boom)
    pid = _make_project()
    body = {"project_id": pid, "text": _SCRIPT, "target_lang": "es"}

    resp = client.post("/audiobook/translate", json=body)
    assert resp.status_code == 200
    assert _events(resp)[-1]["type"] == "error"
    assert pid not in ab._ACTIVE_TRANSLATE_RUNS


@pytest.mark.asyncio
async def test_disconnected_run_does_not_leave_a_permanent_lock():
    """#5b: a client disconnect (the `stopped` path) releases the lock too,
    not only a clean `done`."""
    pid = _make_project()
    plan = AudiobookPlan(chapters=[
        Chapter(title="One", spans=[Span(voice_id=None, text="Hello")]),
        Chapter(title="Two", spans=[Span(voice_id=None, text="World")]),
    ])
    seen = {"n": 0}

    async def is_disconnected():
        seen["n"] += 1
        return seen["n"] > 1

    job_id = "job-disconnect-lifecycle"
    assert ab._acquire_translate_lock(pid, job_id)
    async for _ in ab._translate_longform_sse(
        plan, project_id=pid, source_lang="en", target_lang="es", provider="argos",
        glossary=None, instructions=None, job_id=job_id, is_disconnected=is_disconnected,
    ):
        pass
    assert pid not in ab._ACTIVE_TRANSLATE_RUNS


# ── Lock lifecycle (corrective review, Priority 2 / Point A) ───────────────


def test_streaming_response_construction_failure_releases_the_lock(client, monkeypatch):
    """A failure between acquiring the lock and actually returning the
    StreamingResponse must not leak it — the generator's own cleanup never
    gets a chance to run for a stream that was never constructed."""
    def boom(*_a, **_k):
        raise RuntimeError("response construction blew up")
    monkeypatch.setattr(ab, "_longform_response", boom)
    pid = _make_project()
    body = {"project_id": pid, "text": _SCRIPT, "target_lang": "es"}

    with pytest.raises(RuntimeError):
        client.post("/audiobook/translate", json=body)
    assert pid not in ab._ACTIVE_TRANSLATE_RUNS


def test_a_lock_whose_generator_never_starts_self_heals(monkeypatch):
    """The scenario Priority 2 / Point A explicitly calls out: Starlette's
    StreamingResponse can race a client disconnect against ever calling the
    body iterator once (verified against Starlette 1.6's actual source, not
    assumed), so the generator's `finally` is not guaranteed to run. Simulate
    exactly that — a lock acquired with nothing ever releasing it — and
    confirm a later request for the same project is not permanently blocked."""
    monkeypatch.setattr(ab, "_translate_lock_stale_after_s", lambda: 0.01)
    pid = "orphaned-lock-project"
    assert ab._acquire_translate_lock(pid, "job-orphaned")
    # Nobody calls _release_translate_lock for "job-orphaned" — exactly what
    # "the generator never started consuming" looks like to this function.
    time.sleep(0.05)
    assert ab._acquire_translate_lock(pid, "job-new"), "a stale lock must self-heal"
    ab._release_translate_lock(pid, "job-new")


def test_heartbeat_refresh_keeps_a_long_run_from_looking_stale():
    pid = "long-running-project"
    job_id = "job-long"
    assert ab._acquire_translate_lock(pid, job_id)
    acquired_at = ab._ACTIVE_TRANSLATE_RUNS[pid][1]
    time.sleep(0.02)
    ab._refresh_translate_lock(pid, job_id)
    assert ab._ACTIVE_TRANSLATE_RUNS[pid][1] > acquired_at
    ab._release_translate_lock(pid, job_id)


def test_refresh_never_resurrects_a_lock_this_job_no_longer_owns():
    pid = "reassigned-lock-project"
    ab._acquire_translate_lock(pid, "job-old")
    ab._release_translate_lock(pid, "job-old")
    ab._acquire_translate_lock(pid, "job-new")
    ab._refresh_translate_lock(pid, "job-old")  # must be a no-op
    assert ab._ACTIVE_TRANSLATE_RUNS[pid][0] == "job-new"
    ab._release_translate_lock(pid, "job-new")


def test_backend_restart_trivially_clears_every_lock():
    """In-memory and process-local by design: a fresh process starts with an
    empty dict, which is the entire "release on restart" story — nothing to
    sweep, nothing that can be left stuck across a restart."""
    ab._ACTIVE_TRANSLATE_RUNS["some-project"] = ("job-x", time.monotonic())
    ab._ACTIVE_TRANSLATE_RUNS.clear()  # what module (re)import / process start does
    assert ab._acquire_translate_lock("some-project", "job-y")
    ab._release_translate_lock("some-project", "job-y")


# ── Audiobook generation protection (corrective review, Priority 4 / Point D) ──


def test_no_source_project_id_is_a_complete_noop(monkeypatch):
    """#6: existing AudiobookRequest callers (every one before this field
    existed) must see zero behavior change — not even a DB lookup."""
    def boom(*_a, **_k):
        raise AssertionError("must not touch the database when source_project_id is None")
    monkeypatch.setattr(ab, "_project_exists", boom)
    ab._require_translation_complete(None)  # must not raise


def test_unknown_source_project_id_404():
    with pytest.raises(ab.HTTPException) as exc:
        ab._require_translation_complete("does-not-exist")
    assert exc.value.status_code == 404


def test_incomplete_translation_is_rejected(client, monkeypatch):
    """#5: a project with a failed/unfinished chapter must not pass."""
    async def fake(segments, *, target_lang, **kwargs):
        return [{"id": s.id, "text": s.text, "error": "boom"} for s in segments]
    monkeypatch.setattr(ebook_translate, "translate_segments", fake)
    pid = _make_project()
    client.post("/audiobook/translate", json={
        "project_id": pid, "text": _SCRIPT, "target_lang": "es",
    })
    status = ab.get_translate_status(pid)
    assert status["complete"] is False  # sanity on the fixture itself

    with pytest.raises(ab.HTTPException) as exc:
        ab._require_translation_complete(pid)
    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "translation_incomplete"
    assert exc.value.detail["failed_chapters"] == [0, 1]


def test_complete_translation_is_accepted(client, monkeypatch):
    monkeypatch.setattr(ebook_translate, "translate_segments", _ok_translate())
    pid = _make_project()
    client.post("/audiobook/translate", json={
        "project_id": pid, "text": _SCRIPT, "target_lang": "es",
    })
    status = ab.get_translate_status(pid)
    assert status["complete"] is True  # sanity on the fixture itself

    ab._require_translation_complete(pid)  # must not raise


@pytest.mark.asyncio
async def test_validation_runs_before_any_synthesis_related_work(monkeypatch):
    """The guard must be the FIRST thing audiobook_synthesize does — assert
    that when it rejects, nothing downstream (engine licence check, script
    parsing) ever ran."""
    called = {"licence": False, "parse": False}

    def licence_spy():
        called["licence"] = True
    monkeypatch.setattr(ab, "_ensure_active_engine_licence", licence_spy)

    def parse_spy(*_a, **_k):
        called["parse"] = True
    monkeypatch.setattr(ab, "parse_audiobook_script", parse_spy)

    pid = "incomplete-project-for-order-test"
    monkeypatch.setattr(ab, "_project_exists", lambda p: p == pid)
    monkeypatch.setattr(ab, "_load_translate_state", lambda p: {
        "total_chapters": 2, "chapters": {"0": {"status": "done"}, "1": {"status": "failed"}},
    })

    req = ab.AudiobookRequest(text="# Chapter\nHello.", source_project_id=pid)
    with pytest.raises(ab.HTTPException) as exc:
        await ab.audiobook_synthesize(req, request=None)
    assert exc.value.status_code == 409
    assert called == {"licence": False, "parse": False}


@pytest.mark.asyncio
async def test_existing_caller_without_source_project_id_reaches_past_the_guard(monkeypatch):
    """#6, exercised through the real route function: a request that never
    sets source_project_id must proceed exactly as it always did — proven by
    letting execution reach (and raise from) the very next line, rather than
    being stopped by the new guard."""
    sentinel = RuntimeError("reached past the translation guard")

    def licence_spy():
        raise sentinel
    monkeypatch.setattr(ab, "_ensure_active_engine_licence", licence_spy)

    req = ab.AudiobookRequest(text="# Chapter\nHello.")  # source_project_id omitted entirely
    with pytest.raises(RuntimeError) as exc:
        await ab.audiobook_synthesize(req, request=None)
    assert exc.value is sentinel
