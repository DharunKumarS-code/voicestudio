"""Ordered, sequential processing queue for PDF audiobook batches (Stage 2 of
the batch PDF-to-audiobook work; Stage 1 is :mod:`services.pdf_batch_splitter`,
the pure splitter this module schedules).

Pure bookkeeping — no torch, no TTS, no filesystem — so it is unit-testable
without a model or a GPU, the same contract :mod:`services.longform_progress`
follows for book-level progress. The router drives one :class:`BatchQueue` per
PDF-batch render: :meth:`BatchQueue.next_pending` is the only way to obtain
the next batch to render, and it hands out exactly one at a time, in order,
each exactly once — the guard that keeps a batch from ever being scheduled
twice within the same active job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, Sequence

if TYPE_CHECKING:
    from services.pdf_batch_splitter import PdfBatch

#: A batch's lifecycle. ``empty`` is not a failure — it's a window with no
#: renderable text (e.g. a run of blank pages) that still needs to be
#: accounted for so page/batch ordering stays complete. ``interrupted`` is
#: never set by the live queue itself — it's a label recovery (Stage 3)
#: reports for a persisted ``running`` item found with no resolution after a
#: crash, before resetting it to ``pending`` for retry (see
#: services.pdf_batch_recovery).
BatchStatus = Literal["pending", "running", "done", "failed", "empty", "interrupted"]


@dataclass
class BatchQueueItem:
    index: int
    batch: "PdfBatch"
    status: BatchStatus = "pending"
    output: str | None = None
    chapters: int = 0
    duration_s: float = 0.0
    error: str | None = None
    #: The rendered-through-_render_longform_sse job id for this one batch
    #: (Stage 3). Persisted so a resumed run reuses it, which transparently
    #: gets chapter-level cache/resume reuse from Stage 1/2's unmodified
    #: renderer for any batch that partially rendered before a crash.
    child_job_id: str | None = None
    #: Wall-clock (time.time()) timestamps — persisted so a crash-restart
    #: can show "when did this batch last make progress", and so recovery's
    #: own duration can be measured against them.
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict:
        """The persisted shape (Stage 3's resume manifest / API surface) —
        a plain dict, not the live ``batch: PdfBatch`` object (whose
        ``.script``/``.pages`` are re-derived on resume by re-splitting the
        saved source PDF, not persisted verbatim — see pdf_batch_resume)."""
        return {
            "index": self.index,
            "start_page": self.batch.start_page,
            "end_page": self.batch.end_page,
            "status": self.status,
            "output": self.output,
            "chapters": self.chapters,
            "duration_s": self.duration_s,
            "error": self.error,
            "child_job_id": self.child_job_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class BatchQueue:
    """Ordered queue over a PDF's batches — exactly one in flight at a time."""

    def __init__(self, batches: Sequence["PdfBatch"], *,
                 clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self.items: list[BatchQueueItem] = [
            BatchQueueItem(index=i, batch=b) for i, b in enumerate(batches)
        ]
        self._stopped = False

    @classmethod
    def from_batches_and_state(
        cls,
        batches: Sequence["PdfBatch"],
        state: "Sequence[dict] | None",
        *,
        clock: Callable[[], float] = time.time,
    ) -> "BatchQueue":
        """Build a queue from freshly-(re-)split ``batches``, overlaying any
        persisted per-item ``state`` (Stage 3 resume) on top.

        Only ``done``/``empty`` items carry their persisted state over —
        every other status (``pending``, ``running``, ``failed``) resets to
        fresh ``pending``, so a resumed run retries anything that didn't
        verifiably finish, the same way an ordinary re-run of a chapter plan
        retries any chapter that never produced a cached WAV (a failed
        chapter isn't permanently given up on either). The caller
        (services.pdf_batch_recovery) separately re-verifies every
        surviving ``done`` item's output file before trusting it — this
        classmethod only restores the *shape* of the prior run.

        A persisted item is trusted only when its page range still matches
        the freshly-split batch at that index — if the source PDF or
        ``batch_size_pages`` changed since the manifest was written, the
        split geometry differs and stale state must never be glued onto a
        different batch.
        """
        queue = cls(batches, clock=clock)
        if not state:
            return queue
        by_index = {s["index"]: s for s in state
                    if isinstance(s, dict) and isinstance(s.get("index"), int)}
        for item in queue.items:
            s = by_index.get(item.index)
            if not s:
                continue
            if (s.get("start_page") != item.batch.start_page
                    or s.get("end_page") != item.batch.end_page):
                continue  # split geometry changed — stale state, do not trust it
            item.child_job_id = s.get("child_job_id")
            if s.get("status") in ("done", "empty"):
                item.status = s["status"]
                item.output = s.get("output")
                item.chapters = s.get("chapters", 0) or 0
                item.duration_s = s.get("duration_s", 0.0) or 0.0
                item.error = s.get("error")
                item.started_at = s.get("started_at")
                item.finished_at = s.get("finished_at")
        return queue

    # ── scheduling ──────────────────────────────────────────────────────

    def next_pending(self) -> BatchQueueItem | None:
        """Claim and return the next ``pending`` item, in order, marking it
        ``running``. Returns ``None`` once nothing is left to schedule.

        Raises ``RuntimeError`` if another item is already ``running`` — the
        caller must finish (``mark_done``/``mark_failed``/``mark_empty``) the
        current batch before asking for the next one. This is what makes
        "process exactly one batch at a time" and "never schedule the same
        batch twice" structural invariants rather than just caller discipline.
        """
        running = [it for it in self.items if it.status == "running"]
        if running:
            raise RuntimeError(
                f"batch {running[0].index} is still running — "
                "finish it before requesting the next batch")
        for item in self.items:
            if item.status == "pending":
                item.status = "running"
                item.started_at = self._clock()
                item.finished_at = None
                return item
        return None

    def mark_done(self, index: int, *, output: str | None, chapters: int,
                  duration_s: float) -> None:
        item = self._require_running(index)
        item.status = "done"
        item.output = output
        item.chapters = chapters
        item.duration_s = duration_s
        item.finished_at = self._clock()

    def mark_empty(self, index: int) -> None:
        """The batch had no renderable text (e.g. all-blank pages) — not a
        failure, just nothing to synthesize."""
        item = self._require_running(index)
        item.status = "empty"
        item.finished_at = self._clock()

    def mark_failed(self, index: int, error: str) -> None:
        item = self._require_running(index)
        item.status = "failed"
        item.error = error
        item.finished_at = self._clock()

    def mark_stopped(self) -> None:
        """The client disconnected mid-queue: stop offering further batches.
        Any item still ``running`` stays as-is (its own render loop already
        settled it as done/failed before the queue sees the disconnect)."""
        self._stopped = True

    def _require_running(self, index: int) -> BatchQueueItem:
        item = self.items[index]
        if item.status != "running":
            raise RuntimeError(
                f"batch {index} is not running (status={item.status!r}); "
                "it may have already been resolved")
        return item

    def has_pending(self) -> bool:
        return any(it.status == "pending" for it in self.items)

    # ── reading ─────────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def completed_batches(self) -> list[int]:
        return [it.index for it in self.items if it.status in ("done", "empty")]

    @property
    def failed_batches(self) -> list[int]:
        return [it.index for it in self.items if it.status == "failed"]

    @property
    def current_batch(self) -> int | None:
        running = [it.index for it in self.items if it.status == "running"]
        return running[0] if running else None

    @property
    def status(self) -> Literal["running", "stopped", "done", "done_with_failures", "failed"]:
        if any(it.status in ("pending", "running") for it in self.items):
            return "stopped" if self._stopped else "running"
        failed = self.failed_batches
        if not failed:
            return "done"
        return "failed" if len(failed) == self.total else "done_with_failures"

    def snapshot(self) -> dict:
        """The additive fields of a batch-progress SSE event."""
        return {
            "total_batches": self.total,
            "current_batch": self.current_batch,
            "completed_batches": self.completed_batches,
            "failed_batches": self.failed_batches,
            "status": self.status,
        }

    def to_item_dicts(self) -> list[dict]:
        """Every item's persisted shape (Stage 3 resume manifest), in order."""
        return [item.to_dict() for item in self.items]

    def restored_outputs(self) -> dict[int, dict]:
        """Per-item ``{output, chapters, duration_s}`` for every item that is
        already ``done``/``empty`` at construction time (Stage 3 resume) —
        what a client needs to show a Download link for a batch this run
        will never itself emit a ``batch_done`` for, because it was
        preserved rather than re-rendered. Included once, in the
        ``batch_started`` event, deliberately kept out of :meth:`snapshot`
        (the lightweight shape every other transition event reuses)."""
        return {
            item.index: {
                "output": item.output,
                "chapters": item.chapters,
                "duration_s": item.duration_s,
            }
            for item in self.items if item.status in ("done", "empty")
        }


__all__ = ["BatchQueue", "BatchQueueItem", "BatchStatus"]
