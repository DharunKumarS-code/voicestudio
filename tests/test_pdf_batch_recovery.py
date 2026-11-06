"""Recovery reconciliation for an interrupted PDF-batch job (Stage 3):
verifying persisted "done" batch outputs before trusting them, and
resetting the stale/invalid ones back to pending. Real ffprobe, real files
on disk — no server, no TTS, no GPU."""
from __future__ import annotations

import asyncio
import subprocess

import pytest

from services.ffmpeg_utils import find_ffmpeg, find_ffprobe
from services.pdf_batch_queue import BatchQueue
from services.pdf_batch_recovery import reconcile_queue_for_resume, verify_batch_output
from services.pdf_batch_splitter import PdfBatch

pytestmark = pytest.mark.skipif(
    find_ffprobe() is None or find_ffmpeg() is None,
    reason="ffmpeg/ffprobe required for output validation",
)


def _batches(n: int) -> list[PdfBatch]:
    return [
        PdfBatch(index=i, start_page=i * 10 + 1, end_page=i * 10 + 10,
                 script=f"# Batch {i}\nSome text.", pages=())
        for i in range(n)
    ]


def _write_real_wav(path, seconds: float = 0.5, sr: int = 24000) -> None:
    import wave
    import struct
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        n = int(seconds * sr)
        w.writeframes(struct.pack("<%dh" % n, *([0] * n)))


def _write_real_m4b(path, seconds: float = 2.0) -> None:
    """A genuine, ffprobe-valid M4B — the actual container Stage 2 produces
    (not WAV: WAV's simple RIFF chunking means ffprobe often still reports a
    duration from a truncated file's readable prefix, which doesn't exercise
    what a torn *mux output* really looks like)."""
    ffmpeg = find_ffmpeg()
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:a", "aac", str(path)],
        capture_output=True, check=True, timeout=30,
    )


# ── verify_batch_output ──────────────────────────────────────────────────

def test_verify_missing_file_is_invalid(tmp_path):
    ok = asyncio.run(verify_batch_output("does-not-exist.wav", outputs_dir=str(tmp_path)))
    assert ok is False


def test_verify_none_path_is_invalid(tmp_path):
    ok = asyncio.run(verify_batch_output(None, outputs_dir=str(tmp_path)))
    assert ok is False


def test_verify_empty_file_is_invalid(tmp_path):
    (tmp_path / "empty.wav").write_bytes(b"")
    ok = asyncio.run(verify_batch_output("empty.wav", outputs_dir=str(tmp_path)))
    assert ok is False


def test_verify_garbage_bytes_is_invalid(tmp_path):
    (tmp_path / "garbage.wav").write_bytes(b"not actually audio data at all")
    ok = asyncio.run(verify_batch_output("garbage.wav", outputs_dir=str(tmp_path)))
    assert ok is False


def test_verify_real_audio_file_is_valid(tmp_path):
    _write_real_wav(tmp_path / "real.wav")
    ok = asyncio.run(verify_batch_output("real.wav", outputs_dir=str(tmp_path)))
    assert ok is True


def test_verify_real_m4b_is_valid(tmp_path):
    _write_real_m4b(tmp_path / "real.m4b", seconds=1.0)
    ok = asyncio.run(verify_batch_output("real.m4b", outputs_dir=str(tmp_path)))
    assert ok is True


def test_verify_truncated_m4b_is_invalid(tmp_path):
    """A real M4B (the actual container Stage 2 produces), truncated mid-file
    — the exact power-off-mid-mux shape a crash during ffmpeg's write can
    leave behind a "done" status. ffprobe must not be fooled by a readable
    prefix the way a simple WAV-chunk check could be."""
    _write_real_m4b(tmp_path / "real.m4b", seconds=2.0)
    data = (tmp_path / "real.m4b").read_bytes()
    (tmp_path / "torn.m4b").write_bytes(data[: len(data) // 3])
    ok = asyncio.run(verify_batch_output("torn.m4b", outputs_dir=str(tmp_path)))
    assert ok is False


# ── reconcile_queue_for_resume ───────────────────────────────────────────

def test_a_valid_done_batch_is_preserved(tmp_path):
    _write_real_wav(tmp_path / "out0.wav")
    queue = BatchQueue(_batches(1))
    item = queue.next_pending()
    queue.mark_done(item.index, output="out0.wav", chapters=1, duration_s=0.5)

    report = asyncio.run(reconcile_queue_for_resume(queue, outputs_dir=str(tmp_path)))
    assert report == {"verified": 1, "invalidated": 0}
    assert queue.items[0].status == "done"
    assert queue.items[0].output == "out0.wav"


def test_a_missing_output_file_is_reset_to_pending(tmp_path):
    queue = BatchQueue(_batches(1))
    item = queue.next_pending()
    queue.mark_done(item.index, output="never-written.wav", chapters=1, duration_s=0.5)

    report = asyncio.run(reconcile_queue_for_resume(queue, outputs_dir=str(tmp_path)))
    assert report == {"verified": 0, "invalidated": 1}
    assert queue.items[0].status == "pending"
    assert queue.items[0].output is None
    assert queue.items[0].chapters == 0
    assert queue.items[0].duration_s == 0.0


def test_a_corrupt_output_file_is_reset_to_pending(tmp_path):
    (tmp_path / "corrupt.wav").write_bytes(b"garbage")
    queue = BatchQueue(_batches(1))
    item = queue.next_pending()
    queue.mark_done(item.index, output="corrupt.wav", chapters=1, duration_s=0.5)

    report = asyncio.run(reconcile_queue_for_resume(queue, outputs_dir=str(tmp_path)))
    assert report["invalidated"] == 1
    assert queue.items[0].status == "pending"


def test_reconciliation_leaves_non_done_items_untouched(tmp_path):
    """pending/running/failed/empty items are not this function's job —
    BatchQueue.from_batches_and_state already decided their fate; only
    "done" items need a filesystem re-check."""
    queue = BatchQueue(_batches(3))
    item0 = queue.next_pending()
    queue.mark_failed(item0.index, "boom")
    item1 = queue.next_pending()
    queue.mark_empty(item1.index)
    # item 2 stays pending

    report = asyncio.run(reconcile_queue_for_resume(queue, outputs_dir=str(tmp_path)))
    assert report == {"verified": 0, "invalidated": 0}
    assert [it.status for it in queue.items] == ["failed", "empty", "pending"]


def test_mixed_valid_and_invalid_done_batches(tmp_path):
    _write_real_wav(tmp_path / "good.wav")
    queue = BatchQueue(_batches(2))
    item0 = queue.next_pending()
    queue.mark_done(item0.index, output="good.wav", chapters=1, duration_s=0.5)
    item1 = queue.next_pending()
    queue.mark_done(item1.index, output="missing.wav", chapters=1, duration_s=0.5)

    report = asyncio.run(reconcile_queue_for_resume(queue, outputs_dir=str(tmp_path)))
    assert report == {"verified": 1, "invalidated": 1}
    assert queue.items[0].status == "done"
    assert queue.items[1].status == "pending"
