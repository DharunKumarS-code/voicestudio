"""Recovery/reconciliation for an interrupted PDF-batch audiobook job
(Stage 3).

A backend crash mid-batch leaves :mod:`services.pdf_batch_resume`'s manifest
holding whatever state the last successful write captured — including a
batch still marked ``running`` (the one mid-render when the process died)
and, for any ``done`` batch, an output path that *should* exist but was
never re-verified after the crash: a crash during the ffmpeg mux, or a
half-flushed filesystem, can leave a short/corrupt file behind a ``done``
status (#2279-class, at the batch-output level instead of the chapter-WAV
level ``services.longform_render.wav_is_complete`` already guards).

:func:`reconcile_queue_for_resume` is the single gate between "what the
manifest says" and "what Stage 2's unmodified queue loop will actually
trust": every ``done`` item not freshly re-verified as a complete,
ffprobe-decodable output is reset to ``pending`` so the ordinary loop
retries it. This reuses :func:`services.ffmpeg_utils.probe_duration` — an
existing, already-used-elsewhere async ffprobe wrapper — rather than
inventing a new audio-validity primitive.
"""
from __future__ import annotations

import os

from services.ffmpeg_utils import probe_duration
from services.pdf_batch_queue import BatchQueue


async def verify_batch_output(path: "str | None", *, outputs_dir: str) -> bool:
    """True iff ``path`` (a filename under ``outputs_dir``, as persisted in
    a batch item's ``output`` field) exists and ffprobe can read a positive
    duration from it. False for a missing, empty, or torn/corrupt file —
    never raises."""
    if not path:
        return False
    full = os.path.join(outputs_dir, path)
    if not os.path.isfile(full) or os.path.getsize(full) <= 0:
        return False
    duration = await probe_duration(full, allowed_root=outputs_dir)
    return bool(duration and duration > 0)


async def reconcile_queue_for_resume(queue: BatchQueue, *, outputs_dir: str) -> dict:
    """Re-verify every ``done`` item's output file; demote any that are
    missing, empty, or corrupt back to ``pending`` so the queue loop retries
    them exactly like any other pending batch.

    Mutates ``queue`` in place (the object the caller already holds) and
    returns a small report — ``{"verified": n, "invalidated": n}`` — used
    for the recovery-progress SSE event and the performance measurements in
    the Stage 3 verification report. ``empty`` items are skipped: they have
    no output file to verify by construction.
    """
    verified = 0
    invalidated = 0
    for item in queue.items:
        if item.status != "done":
            continue
        if await verify_batch_output(item.output, outputs_dir=outputs_dir):
            verified += 1
            continue
        invalidated += 1
        item.status = "pending"
        item.output = None
        item.chapters = 0
        item.duration_s = 0.0
        item.error = None
        item.started_at = None
        item.finished_at = None
    return {"verified": verified, "invalidated": invalidated}


__all__ = ["verify_batch_output", "reconcile_queue_for_resume"]
