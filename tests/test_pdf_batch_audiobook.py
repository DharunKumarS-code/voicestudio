"""Integration tests for Stage 2: PDF batch splitting wired into the
audiobook backend as an ordered, sequential, one-batch-at-a-time render queue
(``POST /audiobook/pdf-batches`` → ``audiobook._render_pdf_batches_sse``).

Same harness as test_longform_e2e.py: drives the REAL generator and REAL
ffmpeg, with a stub synth (a CPU tone) so no GPU/model is needed. Each batch
reuses the existing chapterized renderer unmodified, so chapter-level cache,
watermarking, and resume are already covered by test_longform_e2e.py /
test_synthetic_audio_watermark_1169.py — these tests cover the *queue*
behavior layered on top: ordering, one-at-a-time, independent per-batch
output, and failure isolation.
"""
from __future__ import annotations

import asyncio
import io
import json

import pytest
import torch

from services.ffmpeg_utils import find_ffmpeg

pytestmark = pytest.mark.skipif(
    find_ffmpeg() is None, reason="ffmpeg required for the PDF batch e2e",
)

_FFPROBE = __import__("shutil").which("ffprobe")


# ── PDF fixture builder (same minimal hand-built layout as
# tests/test_pdf_batch_splitter.py's _make_multi_page_pdf) ─────────────────

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


# ── stub synth (same idiom as test_longform_e2e.py's _stub_build_synth) ────

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


def _collect_batch_events(data, monkeypatch, outputs_dir, *, fail_on=None, timeout=120, **kw):
    from api.routers import audiobook
    monkeypatch.setattr(audiobook, "_build_synth", _stub_build_synth(fail_on=fail_on))
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(outputs_dir))

    async def _run():
        out = []
        async for frame in audiobook._render_pdf_batches_sse(
            data, default_voice=None, batch_size_pages=1, **kw,
        ):
            assert frame.startswith("data:") and frame.endswith("\n\n")
            out.append(json.loads(frame[len("data:"):].strip()))
        return out

    return asyncio.run(asyncio.wait_for(_run(), timeout=timeout))


# ── ordering / one batch at a time ──────────────────────────────────────

def test_batches_render_in_order_one_at_a_time(tmp_path, monkeypatch):
    events = _collect_batch_events(_three_page_pdf(), monkeypatch, tmp_path)
    progress = [e for e in events if e["type"] == "batch_progress"]
    assert [e["batch_index"] for e in progress] == [0, 1, 2]
    # Each batch_progress(i+1) only ever appears after batch i resolved.
    resolved_at = {}
    for idx, e in enumerate(events):
        if e["type"] in ("batch_done", "batch_failed", "batch_empty"):
            resolved_at[e["batch_index"]] = idx
    progress_at = {e["batch_index"]: i for i, e in enumerate(events) if e["type"] == "batch_progress"}
    for i in range(1, 3):
        assert resolved_at[i - 1] < progress_at[i]


def test_batch_summary_reports_total_and_all_completed(tmp_path, monkeypatch):
    events = _collect_batch_events(_three_page_pdf(), monkeypatch, tmp_path)
    summary = events[-1]
    assert summary["type"] == "batch_summary"
    assert summary["total_batches"] == 3
    assert summary["completed_batches"] == [0, 1, 2]
    assert summary["failed_batches"] == []
    assert summary["status"] == "done"


# ── independent per-batch output files ──────────────────────────────────

def test_each_batch_saves_its_own_independent_output_file(tmp_path, monkeypatch):
    events = _collect_batch_events(_three_page_pdf(), monkeypatch, tmp_path)
    done_events = [e for e in events if e["type"] == "batch_done"]
    assert len(done_events) == 3
    outputs = [e["output"] for e in done_events]
    assert len(set(outputs)) == 3, "every batch must write a distinct output file"
    for name in outputs:
        path = tmp_path / name
        assert path.is_file() and path.stat().st_size > 0
    if _FFPROBE:
        import json as _json
        import subprocess
        for name in outputs:
            probe = subprocess.run(
                [_FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format",
                 str(tmp_path / name)],
                capture_output=True, text=True, check=True,
            ).stdout
            assert "mp4" in _json.loads(probe)["format"]["format_name"]


def test_chapter_level_events_are_tagged_with_batch_context(tmp_path, monkeypatch):
    events = _collect_batch_events(_three_page_pdf(), monkeypatch, tmp_path)
    chapter_events = [e for e in events if e["type"] == "chapter"]
    assert len(chapter_events) == 3  # one chapter per batch
    assert [e["batch_index"] for e in chapter_events] == [0, 1, 2]
    assert all(e["total_batches"] == 3 for e in chapter_events)


# ── failure isolation: a failed batch doesn't block or corrupt the rest ────

def test_a_failed_batch_does_not_block_or_corrupt_remaining_batches(tmp_path, monkeypatch):
    data = _make_multi_page_pdf([
        ["First batch, fine."],
        ["FAILME this batch must fail."],
        ["Third batch, fine."],
    ])
    events = _collect_batch_events(
        data, monkeypatch, tmp_path, fail_on=lambda t: "FAILME" in t,
    )
    summary = events[-1]
    assert summary["type"] == "batch_summary"
    assert summary["failed_batches"] == [1]
    assert summary["completed_batches"] == [0, 2]
    assert summary["status"] == "done_with_failures"

    failed_event = next(e for e in events if e["type"] == "batch_failed")
    assert failed_event["batch_index"] == 1
    assert "stub synth deliberately failed" in failed_event["error"]

    done_events = [e for e in events if e["type"] == "batch_done"]
    assert {e["batch_index"] for e in done_events} == {0, 2}
    for e in done_events:
        path = tmp_path / e["output"]
        assert path.is_file() and path.stat().st_size > 0


def test_all_batches_failing_reports_failed_status(tmp_path, monkeypatch):
    data = _make_multi_page_pdf([["FAILME one."], ["FAILME two."]])
    events = _collect_batch_events(
        data, monkeypatch, tmp_path, fail_on=lambda t: "FAILME" in t,
    )
    summary = events[-1]
    assert summary["status"] == "failed"
    assert summary["failed_batches"] == [0, 1]
    assert summary["completed_batches"] == []
    assert not any(p.suffix in (".m4b", ".mp3") for p in tmp_path.iterdir())


# ── different page counts / empty-batch handling ────────────────────────

def test_single_page_pdf_is_a_single_batch(tmp_path, monkeypatch):
    data = _make_multi_page_pdf([["Only page, ends cleanly."]])
    events = _collect_batch_events(data, monkeypatch, tmp_path)
    summary = events[-1]
    assert summary["total_batches"] == 1
    assert summary["completed_batches"] == [0]


def test_blank_batch_window_is_reported_empty_not_failed(tmp_path, monkeypatch):
    data = _make_multi_page_pdf([["First page, fine."], []])
    events = _collect_batch_events(data, monkeypatch, tmp_path)
    empty_events = [e for e in events if e["type"] == "batch_empty"]
    assert [e["batch_index"] for e in empty_events] == [1]
    summary = events[-1]
    assert summary["completed_batches"] == [0, 1]
    assert summary["failed_batches"] == []
    assert summary["status"] == "done"


# ── duplicate-processing guard surfaces end to end ──────────────────────

def test_no_batch_index_is_ever_resolved_twice(tmp_path, monkeypatch):
    data = _make_multi_page_pdf([
        ["First batch, fine."], ["FAILME this one fails."], ["Third batch, fine."],
    ])
    events = _collect_batch_events(
        data, monkeypatch, tmp_path, fail_on=lambda t: "FAILME" in t,
    )
    resolutions = [e["batch_index"] for e in events
                   if e["type"] in ("batch_done", "batch_failed", "batch_empty")]
    assert len(resolutions) == len(set(resolutions)) == 3
