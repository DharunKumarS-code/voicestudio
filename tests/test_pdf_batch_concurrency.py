"""Stage 3 Fix 3: in-process guard against two concurrent renders of the
same PDF-batch job (``_pdf_batch_jobs_in_flight``).

Reproduces the race identified in the Stage 3 review — two concurrent
resume requests for the same job_id each build their own BatchQueue from
the same persisted manifest and, without a guard, both independently claim
and render the same batch. Driven against the REAL generator + REAL ffmpeg
(same harness as test_pdf_batch_recovery_e2e.py), with a stub synth that
blocks long enough for two concurrent calls to genuinely overlap.
"""
from __future__ import annotations

import asyncio
import io
import json
import time

import pytest
import torch

from services.ffmpeg_utils import find_ffmpeg

pytestmark = pytest.mark.skipif(
    find_ffmpeg() is None, reason="ffmpeg required for the PDF batch concurrency e2e",
)


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


def _two_page_pdf() -> bytes:
    return _make_multi_page_pdf([
        ["First batch content, ends cleanly."],
        ["Second batch content, ends cleanly."],
    ])


def _resolve(_voice_id):
    return {"ref_audio": None, "ref_text": None, "instruct": None, "seed": None}


def _slow_stub_build_synth(delay_s: float):
    """A stub whose synth() blocks for ``delay_s`` — long enough for two
    concurrent calls to overlap mid-render, the window the Fix 3 guard must
    close."""
    def _factory(default_voice=None, language=None, opts=None, voice_map=None, lease=None):
        def synth(text, voice_id, speed=None):
            time.sleep(delay_s)
            return 0.1 * torch.sin(torch.arange(2400) * (2 * torch.pi * 440 / 24000))
        return {"mode": "generic", "resolve": _resolve, "engine_id": "stub",
                "synth": synth, "sample_rate": 24000}
    return _factory


def _fast_stub_build_synth():
    def _factory(default_voice=None, language=None, opts=None, voice_map=None, lease=None):
        def synth(text, voice_id, speed=None):
            return 0.1 * torch.sin(torch.arange(2400) * (2 * torch.pi * 440 / 24000))
        return {"mode": "generic", "resolve": _resolve, "engine_id": "stub",
                "synth": synth, "sample_rate": 24000}
    return _factory


async def _drain(gen):
    out = []
    async for frame in gen:
        out.append(json.loads(frame[len("data:"):].strip()))
    return out


def _seed_interrupted_job(tmp_path, monkeypatch, *, delay_s: float) -> str:
    """Run a real job partway (batch 0 starting) and abandon the generator
    mid-flight, leaving a durable "running" manifest — the same technique
    test_pdf_batch_recovery_e2e.py uses."""
    from api.routers import audiobook

    monkeypatch.setattr(audiobook, "_build_synth", _slow_stub_build_synth(delay_s))
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))

    async def seed():
        gen = audiobook._render_pdf_batches_sse(
            _two_page_pdf(), default_voice=None, batch_size_pages=1,
        )
        job_id = None
        async for raw in gen:
            payload = json.loads(raw[len("data: "):].rstrip("\n"))
            if payload.get("type") == "batch_started":
                job_id = payload["job_id"]
            if payload.get("type") == "started" and payload.get("batch_index") == 0:
                break
        await gen.aclose()
        return job_id

    return asyncio.run(asyncio.wait_for(seed(), timeout=60))


# ── 1/2/3: two simultaneous resumes — only one claims the batch, the other
# is clearly rejected ───────────────────────────────────────────────────

def test_two_simultaneous_resumes_only_one_claims_and_renders(tmp_path, monkeypatch):
    from api.routers import audiobook

    job_id = _seed_interrupted_job(tmp_path, monkeypatch, delay_s=1.0)
    # Keep the slow stub for the race window, then let the loop run to a
    # real, deterministic completion.
    monkeypatch.setattr(audiobook, "_build_synth", _slow_stub_build_synth(1.0))
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))

    claimed_batch0 = []

    async def resume(label):
        gen = audiobook._render_pdf_batches_sse(
            None, default_voice=None, batch_size_pages=0, job_id=job_id, resume=True,
        )
        events = await _drain(gen)
        for e in events:
            if e.get("type") == "batch_progress" and e.get("batch_index") == 0:
                claimed_batch0.append(label)
        return label, events

    async def main():
        return await asyncio.gather(resume("A"), resume("B"))

    results = asyncio.run(asyncio.wait_for(main(), timeout=60))

    # Exactly one of the two actually claimed/rendered the interrupted batch.
    assert claimed_batch0 == ["A"] or claimed_batch0 == ["B"]
    assert len(claimed_batch0) == 1

    # The other received a clear rejection, not a race/crash.
    rejected = [label for label, events in results if label not in claimed_batch0]
    assert len(rejected) == 1
    rejected_label = rejected[0]
    rejected_events = next(events for label, events in results if label == rejected_label)
    assert len(rejected_events) == 1
    assert rejected_events[0]["type"] == "error"
    assert "already being processed" in rejected_events[0]["error"]
    assert job_id in rejected_events[0]["error"]


# ── 4: the guard releases after both success and failure ───────────────

def test_guard_releases_after_successful_completion(tmp_path, monkeypatch):
    from api.routers import audiobook

    monkeypatch.setattr(audiobook, "_build_synth", _fast_stub_build_synth())
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))

    async def run():
        gen = audiobook._render_pdf_batches_sse(
            _two_page_pdf(), default_voice=None, batch_size_pages=1,
        )
        return await _drain(gen)

    events = asyncio.run(asyncio.wait_for(run(), timeout=60))
    job_id = events[0]["job_id"]
    assert events[-1]["type"] == "batch_summary" and events[-1]["status"] == "done"
    assert job_id not in audiobook._pdf_batch_jobs_in_flight


def test_guard_releases_after_a_failed_render(tmp_path, monkeypatch):
    """Every batch's chapter synth fails — the run ends in status "failed"
    (not an exception), and the guard must still be released."""
    from api.routers import audiobook

    def _failing_stub(*, fail_on=None):
        def _factory(default_voice=None, language=None, opts=None, voice_map=None, lease=None):
            def synth(text, voice_id, speed=None):
                raise RuntimeError("stub synth deliberately failed")
            return {"mode": "generic", "resolve": _resolve, "engine_id": "stub",
                    "synth": synth, "sample_rate": 24000}
        return _factory

    monkeypatch.setattr(audiobook, "_build_synth", _failing_stub())
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))

    async def run():
        gen = audiobook._render_pdf_batches_sse(
            _two_page_pdf(), default_voice=None, batch_size_pages=1,
        )
        return await _drain(gen)

    events = asyncio.run(asyncio.wait_for(run(), timeout=60))
    job_id = events[0]["job_id"]
    assert events[-1]["type"] == "batch_summary" and events[-1]["status"] == "failed"
    assert job_id not in audiobook._pdf_batch_jobs_in_flight


def test_guard_releases_on_early_split_error(tmp_path, monkeypatch):
    """A corrupt PDF hits the pre-queue ValueError return path — one of the
    "early return" branches now nested inside the guard's try/finally."""
    from api.routers import audiobook

    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))

    async def run():
        gen = audiobook._render_pdf_batches_sse(
            b"not a pdf at all", default_voice=None, batch_size_pages=1,
        )
        return await _drain(gen)

    events = asyncio.run(asyncio.wait_for(run(), timeout=30))
    assert events[-1]["type"] == "error"
    # No job_id is reported on this path (split fails before batch_started),
    # so confirm indirectly: the set is empty, nothing was left registered.
    assert audiobook._pdf_batch_jobs_in_flight == set()


# ── 5: a later (sequential) retry works once the first request finishes ──

def test_sequential_retry_succeeds_after_first_resume_releases_the_guard(
    tmp_path, monkeypatch,
):
    from api.routers import audiobook

    job_id = _seed_interrupted_job(tmp_path, monkeypatch, delay_s=0.0)
    monkeypatch.setattr(audiobook, "_build_synth", _fast_stub_build_synth())
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))

    async def resume_once():
        gen = audiobook._render_pdf_batches_sse(
            None, default_voice=None, batch_size_pages=0, job_id=job_id, resume=True,
        )
        return await _drain(gen)

    first = asyncio.run(asyncio.wait_for(resume_once(), timeout=60))
    assert first[-1]["type"] == "batch_summary" and first[-1]["status"] == "done"
    assert job_id not in audiobook._pdf_batch_jobs_in_flight

    # The job fully resolved and its manifest was cleared, so a second
    # resume 404s at the router layer in real use — but the generator level
    # guarantee under test here is narrower and still holds: calling the
    # renderer again for the same id is never rejected as "already being
    # processed" merely because it ran once before.
    second = asyncio.run(asyncio.wait_for(resume_once(), timeout=60))
    assert not any(
        e["type"] == "error" and "already being processed" in e.get("error", "")
        for e in second
    )


# ── 6: two different job IDs remain independent under real concurrency ──

def test_two_different_job_ids_run_concurrently_without_interfering(tmp_path, monkeypatch):
    from api.routers import audiobook

    job_a = _seed_interrupted_job(tmp_path, monkeypatch, delay_s=0.3)
    job_b = _seed_interrupted_job(tmp_path, monkeypatch, delay_s=0.3)
    assert job_a != job_b
    monkeypatch.setattr(audiobook, "_build_synth", _slow_stub_build_synth(0.3))
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))

    async def resume(job_id):
        gen = audiobook._render_pdf_batches_sse(
            None, default_voice=None, batch_size_pages=0, job_id=job_id, resume=True,
        )
        return await _drain(gen)

    async def main():
        return await asyncio.gather(resume(job_a), resume(job_b))

    results_a, results_b = asyncio.run(asyncio.wait_for(main(), timeout=60))

    # Neither run was rejected by the other's guard entry.
    assert not any(e["type"] == "error" and "already being processed" in e.get("error", "")
                   for e in results_a)
    assert not any(e["type"] == "error" and "already being processed" in e.get("error", "")
                   for e in results_b)
    assert results_a[-1]["type"] == "batch_summary" and results_a[-1]["status"] == "done"
    assert results_b[-1]["type"] == "batch_summary" and results_b[-1]["status"] == "done"
    assert audiobook._pdf_batch_jobs_in_flight == set()
