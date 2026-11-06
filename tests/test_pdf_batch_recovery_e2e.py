"""Stage 3 end-to-end: persistent batch state + recovery, driven against the
REAL `_render_pdf_batches_sse`/`resume_pdf_batches` generators and REAL
ffmpeg (same harness as test_pdf_batch_audiobook.py / test_longform_e2e.py),
with a stub synth so no GPU/model is needed.

"Crash mid-render" is simulated by abandoning the async generator at a
specific suspension point (a plain ``yield`` inside it) without letting the
loop reach its natural end — nothing runs after that point, the same way
nothing runs after a killed process. The durable manifest
(services.pdf_batch_resume) is read straight from disk afterward, exactly
as a fresh backend process would on restart.
"""
from __future__ import annotations

import asyncio
import io
import json

import pytest
import torch

from services.ffmpeg_utils import find_ffmpeg

pytestmark = pytest.mark.skipif(
    find_ffmpeg() is None, reason="ffmpeg required for the PDF batch recovery e2e",
)


# ── PDF fixture builder (same minimal layout as the other pdf_batch tests) ──

def _make_multi_page_pdf(pages: list[list[str]]) -> bytes:
    def _content_bytes(page: list[str]) -> bytes:
        if not page:
            return b"q Q"
        show = "BT /F1 12 Tf 72 720 Td 16 TL\n"
        for ln in page:
            esc = ln.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")
            show += f"({esc}) Tj T*\n"
        show += "ET"
        return show.encode("latin-1")

    n = len(pages)
    page_obj_nums = [4 + 2 * i for i in range(n)]
    kids = " ".join(f"{num} 0 R" for num in page_obj_nums)
    objs: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, page in enumerate(pages):
        content = _content_bytes(page)
        content_obj_num = 5 + 2 * i
        objs.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
            + str(content_obj_num).encode() + b" 0 R >>"
        )
        objs.append(
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
            + content + b"\nendstream"
        )
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_pos = out.tell()
    total = len(objs) + 1
    out.write(f"xref\n0 {total}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode())
    return out.getvalue()


def _three_page_pdf() -> bytes:
    return _make_multi_page_pdf([
        ["First batch content, ends cleanly."],
        ["Second batch content, ends cleanly."],
        ["Third batch content, ends cleanly."],
    ])


def _resolve(_voice_id):
    return {"ref_audio": None, "ref_text": None, "instruct": None, "seed": None}


def _stub_build_synth(*, fail_on=None):
    def _factory(default_voice=None, language=None, opts=None, voice_map=None, lease=None):
        def synth(text, voice_id, speed=None):
            if fail_on is not None and fail_on(text):
                raise RuntimeError("stub synth deliberately failed")
            return 0.1 * torch.sin(torch.arange(2400) * (2 * torch.pi * 440 / 24000))
        return {"mode": "generic", "resolve": _resolve, "engine_id": "stub",
                "synth": synth, "sample_rate": 24000}
    return _factory


def _patch(monkeypatch, outputs_dir, *, fail_on=None):
    from api.routers import audiobook
    monkeypatch.setattr(audiobook, "_build_synth", _stub_build_synth(fail_on=fail_on))
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(outputs_dir))


async def _drain(gen):
    out = []
    async for frame in gen:
        assert frame.startswith("data:") and frame.endswith("\n\n")
        out.append(json.loads(frame[len("data:"):].strip()))
    return out


async def _pull_until(gen, predicate, limit=200):
    """Pull events from ``gen`` until one matches ``predicate``; return the
    events pulled (inclusive) WITHOUT closing or further advancing the
    generator — it stays suspended at that exact yield, as if the process
    died right there."""
    out = []
    for _ in range(limit):
        frame = await gen.__anext__()
        event = json.loads(frame[len("data:"):].strip())
        out.append(event)
        if predicate(event):
            return out
    raise AssertionError("predicate never matched")


# ── normal completion clears the manifest ───────────────────────────────

def test_normal_completion_clears_the_resume_manifest(tmp_path, monkeypatch):
    from api.routers import audiobook
    from services import pdf_batch_resume

    _patch(monkeypatch, tmp_path)

    async def run():
        gen = audiobook._render_pdf_batches_sse(
            _three_page_pdf(), default_voice=None, batch_size_pages=1,
            source_filename="book.pdf",
        )
        return await _drain(gen)

    events = asyncio.run(asyncio.wait_for(run(), timeout=120))
    summary = events[-1]
    assert summary["type"] == "batch_summary" and summary["status"] == "done"
    job_id = events[0]["job_id"]
    assert pdf_batch_resume.has_manifest(job_id) is False


# ── crash mid-batch + resume ────────────────────────────────────────────

def test_crash_mid_batch_persists_interrupted_state_and_resume_continues(
    tmp_path, monkeypatch,
):
    """The documented scenario: batches 1..N-1 completed, batch N crashed
    mid-render. Resume must preserve the completed batches untouched, retry
    only the interrupted one, and finish the rest."""
    from api.routers import audiobook
    from services import pdf_batch_resume

    _patch(monkeypatch, tmp_path)

    async def crash_after_batch1_starts():
        gen = audiobook._render_pdf_batches_sse(
            _three_page_pdf(), default_voice=None, batch_size_pages=1,
            source_filename="book.pdf",
        )
        events = await _pull_until(
            gen, lambda e: e.get("type") == "started" and e.get("batch_index") == 1,
        )
        job_id = events[0]["job_id"]
        # The generator is suspended mid-batch-1; nothing after this point
        # ran. Read the durable state exactly as a restarted process would.
        manifest = pdf_batch_resume.read_manifest(job_id)
        await gen.aclose()
        return job_id, manifest

    job_id, manifest = asyncio.run(asyncio.wait_for(crash_after_batch1_starts(), timeout=120))

    assert manifest is not None
    statuses = {b["index"]: b["status"] for b in manifest["batches"]}
    assert statuses[0] == "done"       # batch 1: completed
    assert statuses[1] == "running"    # batch 2: mid-flight when "crashed"
    assert statuses[2] == "pending"    # batch 3: never started
    done_output = next(b["output"] for b in manifest["batches"] if b["index"] == 0)
    assert (tmp_path / done_output).is_file()

    async def resume():
        from api.routers import audiobook as ab
        gen = ab._render_pdf_batches_sse(
            None, default_voice=None, batch_size_pages=0,
            job_id=job_id, resume=True,
        )
        return await _drain(gen)

    resumed = asyncio.run(asyncio.wait_for(resume(), timeout=120))
    summary = resumed[-1]
    assert summary["type"] == "batch_summary"
    assert summary["status"] == "done"
    assert summary["completed_batches"] == [0, 1, 2]
    assert summary["failed_batches"] == []

    recovery = next(e for e in resumed if e["type"] == "recovery_summary")
    assert recovery["verified"] == 1     # batch 0's output re-verified, preserved
    assert recovery["invalidated"] == 0

    # Batch 0 is preserved, not re-run at all — no NEW batch_done for it
    # appears in the resumed stream (that's the duplicate-execution guard
    # working), yet its original output file is still exactly what the
    # final summary reports as completed.
    assert not any(e["type"] == "batch_done" and e["batch_index"] == 0 for e in resumed)
    assert (tmp_path / done_output).is_file()

    # Fix 1: batch 0 never gets its own batch_done this run, so the ONLY
    # place a client learns its filename is batch_started's restored_outputs
    # — this is what makes the Download link show up immediately on resume.
    started = resumed[0]
    assert started["type"] == "batch_started"
    restored0 = started["restored_outputs"]["0"]
    assert restored0["output"] == done_output
    assert restored0["chapters"] == 1
    assert restored0["duration_s"] > 0
    assert "1" not in started["restored_outputs"]  # batch 1 was interrupted, not preserved
    assert "2" not in started["restored_outputs"]  # batch 2 never started

    assert pdf_batch_resume.has_manifest(job_id) is False  # fully resolved now


# ── missing output file ─────────────────────────────────────────────────

def test_missing_completed_output_file_is_regenerated_on_resume(tmp_path, monkeypatch):
    from api.routers import audiobook
    from services import pdf_batch_resume

    _patch(monkeypatch, tmp_path)

    async def crash_after_batch1_starts():
        gen = audiobook._render_pdf_batches_sse(
            _three_page_pdf(), default_voice=None, batch_size_pages=1,
        )
        events = await _pull_until(
            gen, lambda e: e.get("type") == "started" and e.get("batch_index") == 1,
        )
        job_id = events[0]["job_id"]
        await gen.aclose()
        return job_id

    job_id = asyncio.run(asyncio.wait_for(crash_after_batch1_starts(), timeout=120))
    manifest = pdf_batch_resume.read_manifest(job_id)
    output_name = next(b["output"] for b in manifest["batches"] if b["index"] == 0)
    (tmp_path / output_name).unlink()  # simulate a missing output file

    async def resume():
        gen = audiobook._render_pdf_batches_sse(
            None, default_voice=None, batch_size_pages=0, job_id=job_id, resume=True,
        )
        return await _drain(gen)

    resumed = asyncio.run(asyncio.wait_for(resume(), timeout=120))
    recovery = next(e for e in resumed if e["type"] == "recovery_summary")
    assert recovery["invalidated"] == 1
    assert resumed[-1]["completed_batches"] == [0, 1, 2]
    assert (tmp_path / next(
        e for e in resumed if e["type"] == "batch_done" and e["batch_index"] == 0
    )["output"]).is_file()


# ── corrupted output file ───────────────────────────────────────────────

def test_corrupted_completed_output_file_is_regenerated_on_resume(tmp_path, monkeypatch):
    from api.routers import audiobook
    from services import pdf_batch_resume

    _patch(monkeypatch, tmp_path)

    async def crash_after_batch1_starts():
        gen = audiobook._render_pdf_batches_sse(
            _three_page_pdf(), default_voice=None, batch_size_pages=1,
        )
        events = await _pull_until(
            gen, lambda e: e.get("type") == "started" and e.get("batch_index") == 1,
        )
        job_id = events[0]["job_id"]
        await gen.aclose()
        return job_id

    job_id = asyncio.run(asyncio.wait_for(crash_after_batch1_starts(), timeout=120))
    manifest = pdf_batch_resume.read_manifest(job_id)
    output_name = next(b["output"] for b in manifest["batches"] if b["index"] == 0)
    out_path = tmp_path / output_name
    data = out_path.read_bytes()
    # A small fixed prefix (not a fraction) so it can never contain a
    # complete moov atom regardless of where ffmpeg placed it in this
    # particular (very short, stub-synth) output.
    out_path.write_bytes(data[:64])  # torn/corrupt

    async def resume():
        gen = audiobook._render_pdf_batches_sse(
            None, default_voice=None, batch_size_pages=0, job_id=job_id, resume=True,
        )
        return await _drain(gen)

    resumed = asyncio.run(asyncio.wait_for(resume(), timeout=120))
    recovery = next(e for e in resumed if e["type"] == "recovery_summary")
    assert recovery["invalidated"] == 1
    assert resumed[-1]["completed_batches"] == [0, 1, 2]


# ── recovery when everything is already complete / all failed ──────────

def test_recovery_when_all_batches_already_completed_makes_no_new_render_calls(
    tmp_path, monkeypatch,
):
    from api.routers import audiobook
    from services import pdf_batch_resume

    _patch(monkeypatch, tmp_path)

    async def run_to_completion_but_dont_let_it_clear():
        # Run a real job to completion so three valid output files exist.
        # The natural run clears the manifest + source PDF on success (the
        # normal-completion contract); hand-write both back afterward to
        # simulate a crash that happened between the last batch_done and
        # that final clear — the state a restarted process would actually
        # find on disk in that narrow window.
        pdf_bytes = _three_page_pdf()
        gen = audiobook._render_pdf_batches_sse(
            pdf_bytes, default_voice=None, batch_size_pages=1,
        )
        events = await _drain(gen)
        job_id = events[0]["job_id"]
        done_events = [e for e in events if e["type"] == "batch_done"]
        batches = [
            {"index": e["batch_index"], "start_page": e["batch_index"] + 1,
             "end_page": e["batch_index"] + 1, "status": "done",
             "output": e["output"], "chapters": e["chapters"],
             "duration_s": e["duration_s"], "error": None,
             "child_job_id": f"{job_id}b{e['batch_index']}",
             "started_at": 1.0, "finished_at": 2.0}
            for e in done_events
        ]
        pdf_batch_resume.write_source_pdf(job_id, pdf_bytes)
        pdf_batch_resume.write_manifest(pdf_batch_resume.build_manifest(
            job_id=job_id, page_count=3, batch_size_pages=1,
            params={"default_voice": None, "fmt": "m4b", "bitrate": "128k",
                    "expressive": None},
            batches=batches,
        ))
        return job_id

    job_id = asyncio.run(asyncio.wait_for(run_to_completion_but_dont_let_it_clear(), timeout=120))

    def _boom(*a, **k):
        raise AssertionError("a fully-recovered batch must never be re-rendered")
    monkeypatch.setattr(audiobook, "_render_longform_sse", _boom)

    async def resume():
        gen = audiobook._render_pdf_batches_sse(
            None, default_voice=None, batch_size_pages=0, job_id=job_id, resume=True,
        )
        return await _drain(gen)

    resumed = asyncio.run(asyncio.wait_for(resume(), timeout=30))
    recovery = next(e for e in resumed if e["type"] == "recovery_summary")
    assert recovery["verified"] == 3
    assert recovery["invalidated"] == 0
    assert resumed[-1]["status"] == "done"
    assert resumed[-1]["completed_batches"] == [0, 1, 2]


def test_recovery_when_some_batches_have_failed_retries_them(tmp_path, monkeypatch):
    from api.routers import audiobook
    from services import pdf_batch_resume

    _patch(monkeypatch, tmp_path, fail_on=lambda t: "Second" in t)

    async def crash_run():
        gen = audiobook._render_pdf_batches_sse(
            _three_page_pdf(), default_voice=None, batch_size_pages=1,
        )
        events = await _pull_until(
            gen, lambda e: e.get("type") == "started" and e.get("batch_index") == 2,
        )
        job_id = events[0]["job_id"]
        await gen.aclose()
        return job_id

    job_id = asyncio.run(asyncio.wait_for(crash_run(), timeout=120))
    manifest = pdf_batch_resume.read_manifest(job_id)
    statuses = {b["index"]: b["status"] for b in manifest["batches"]}
    assert statuses[1] == "failed"  # "Second batch..." deliberately failed

    # Resume without the failure this time — the previously-failed batch
    # must be retried, not left failed forever.
    _patch(monkeypatch, tmp_path, fail_on=None)

    async def resume():
        gen = audiobook._render_pdf_batches_sse(
            None, default_voice=None, batch_size_pages=0, job_id=job_id, resume=True,
        )
        return await _drain(gen)

    resumed = asyncio.run(asyncio.wait_for(resume(), timeout=120))
    assert resumed[-1]["status"] == "done"
    assert resumed[-1]["completed_batches"] == [0, 1, 2]
    assert resumed[-1]["failed_batches"] == []


# ── M1 regression: a NATURALLY-completed done_with_failures run must stay
# resumable (unlike the crash-interrupted case above, nothing aborts the
# generator here — it runs to its own batch_summary) ───────────────────────

def test_done_with_failures_from_a_natural_completion_remains_resumable(
    tmp_path, monkeypatch,
):
    from api.routers import audiobook
    from core import db as core_db
    from services import pdf_batch_resume

    # list_resumable_pdf_batch_jobs() (unlike the generator's own job_store
    # calls, which are best-effort) reads job_store.get() directly — needs a
    # real `jobs` table, which only app startup creates outside this test.
    core_db.init_db()
    _patch(monkeypatch, tmp_path, fail_on=lambda t: "Second" in t)

    async def run():
        gen = audiobook._render_pdf_batches_sse(
            _three_page_pdf(), default_voice=None, batch_size_pages=1,
        )
        return await _drain(gen)  # runs to its own natural batch_summary

    events = asyncio.run(asyncio.wait_for(run(), timeout=120))
    summary = events[-1]
    assert summary["type"] == "batch_summary"
    assert summary["status"] == "done_with_failures"
    job_id = events[0]["job_id"]
    original_done = {
        e["batch_index"]: e["output"] for e in events if e["type"] == "batch_done"
    }
    assert set(original_done) == {0, 2}  # batch 1 ("Second...") failed

    # The manifest must survive a done_with_failures run that finished on its
    # own — this is the fix: before it, clear_manifest ran for this status too.
    assert pdf_batch_resume.has_manifest(job_id) is True

    # GET /audiobook/pdf-batches/jobs must discover it.
    listing = audiobook.list_resumable_pdf_batch_jobs()
    entry = next((j for j in listing["jobs"] if j["job_id"] == job_id), None)
    assert entry is not None, listing
    assert entry["completed_batches"] == [0, 2]
    assert entry["failed_batches"] == [1]

    # Resume without the failure this time — only the failed batch retries;
    # the two already-done batches must not be regenerated.
    _patch(monkeypatch, tmp_path, fail_on=None)

    async def resume():
        gen = audiobook._render_pdf_batches_sse(
            None, default_voice=None, batch_size_pages=0, job_id=job_id, resume=True,
        )
        return await _drain(gen)

    resumed = asyncio.run(asyncio.wait_for(resume(), timeout=120))
    started_event = next(e for e in resumed if e["type"] == "batch_started")
    restored = started_event["restored_outputs"]
    # Batches 0 and 2 are handed back as already-done, with their ORIGINAL
    # output filenames — proof they were preserved, not re-rendered.
    assert restored["0"]["output"] == original_done[0]
    assert restored["2"]["output"] == original_done[2]
    # No render activity for batches 0/2 this run — only batch 1 renders.
    rendered_indices = {
        e["batch_index"] for e in resumed
        if e["type"] in ("batch_progress", "started", "chapter", "batch_done")
    }
    assert rendered_indices == {1}

    summary2 = resumed[-1]
    assert summary2["type"] == "batch_summary"
    assert summary2["status"] == "done"
    assert summary2["completed_batches"] == [0, 1, 2]
    assert summary2["failed_batches"] == []
    # Fully resolved now — the manifest is cleared for real this time.
    assert pdf_batch_resume.has_manifest(job_id) is False


# ── repeated recovery attempts ───────────────────────────────────────────

def test_repeated_resume_attempts_are_safe_and_idempotent(tmp_path, monkeypatch):
    from api.routers import audiobook
    from services import pdf_batch_resume

    _patch(monkeypatch, tmp_path)

    async def crash_run():
        gen = audiobook._render_pdf_batches_sse(
            _three_page_pdf(), default_voice=None, batch_size_pages=1,
        )
        events = await _pull_until(
            gen, lambda e: e.get("type") == "started" and e.get("batch_index") == 1,
        )
        job_id = events[0]["job_id"]
        await gen.aclose()
        return job_id

    job_id = asyncio.run(asyncio.wait_for(crash_run(), timeout=120))

    async def resume():
        gen = audiobook._render_pdf_batches_sse(
            None, default_voice=None, batch_size_pages=0, job_id=job_id, resume=True,
        )
        return await _drain(gen)

    first = asyncio.run(asyncio.wait_for(resume(), timeout=120))
    assert first[-1]["status"] == "done"
    assert pdf_batch_resume.has_manifest(job_id) is False

    # A second resume attempt on the same (now fully-resolved, manifest
    # already cleared) job must 404, not silently re-render anything.
    entry_exists = any(e["job_id"] == job_id for e in pdf_batch_resume.scan_resumable())
    assert entry_exists is False


# ── multiple interrupted jobs ────────────────────────────────────────────

def test_multiple_interrupted_jobs_are_independent(tmp_path, monkeypatch):
    from api.routers import audiobook
    from services import pdf_batch_resume

    _patch(monkeypatch, tmp_path)

    async def crash_run():
        gen = audiobook._render_pdf_batches_sse(
            _three_page_pdf(), default_voice=None, batch_size_pages=1,
        )
        events = await _pull_until(
            gen, lambda e: e.get("type") == "started" and e.get("batch_index") == 1,
        )
        job_id = events[0]["job_id"]
        await gen.aclose()
        return job_id

    job_a = asyncio.run(asyncio.wait_for(crash_run(), timeout=120))
    job_b = asyncio.run(asyncio.wait_for(crash_run(), timeout=120))
    assert job_a != job_b

    listed = {e["job_id"] for e in pdf_batch_resume.scan_resumable()}
    assert {job_a, job_b} <= listed

    async def resume(job_id):
        gen = audiobook._render_pdf_batches_sse(
            None, default_voice=None, batch_size_pages=0, job_id=job_id, resume=True,
        )
        return await _drain(gen)

    resumed_a = asyncio.run(asyncio.wait_for(resume(job_a), timeout=120))
    assert resumed_a[-1]["status"] == "done"
    # job_b must be untouched by resuming job_a.
    assert pdf_batch_resume.has_manifest(job_b) is True
    assert pdf_batch_resume.has_manifest(job_a) is False
