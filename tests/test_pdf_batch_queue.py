"""Ordered batch-processing queue (Stage 2 orchestration over Stage 1's
splitter) and its Stage 3 persistence shape (timestamps, child_job_id,
to_dict/from_batches_and_state). Pure, torch-free — no PDF, no TTS, no GPU
needed."""
from __future__ import annotations

import pytest

from services.pdf_batch_queue import BatchQueue
from services.pdf_batch_splitter import PdfBatch


def _fake_batches(n: int) -> list[PdfBatch]:
    return [
        PdfBatch(index=i, start_page=i * 10 + 1, end_page=i * 10 + 10,
                 script=f"# Batch {i}\nSome text.", pages=())
        for i in range(n)
    ]


# ── ordering / one-at-a-time ─────────────────────────────────────────────

def test_batches_are_claimed_in_order():
    q = BatchQueue(_fake_batches(3))
    claimed = []
    for _ in range(3):
        item = q.next_pending()
        claimed.append(item.index)
        q.mark_done(item.index, output=f"out{item.index}.m4b", chapters=1, duration_s=1.0)
    assert claimed == [0, 1, 2]
    assert q.next_pending() is None


def test_cannot_claim_a_second_batch_while_one_is_running():
    q = BatchQueue(_fake_batches(2))
    q.next_pending()  # claims batch 0, leaves it running
    with pytest.raises(RuntimeError, match="still running"):
        q.next_pending()


def test_resolving_the_running_batch_unblocks_the_next_claim():
    q = BatchQueue(_fake_batches(2))
    item = q.next_pending()
    q.mark_done(item.index, output="out0.m4b", chapters=2, duration_s=5.0)
    nxt = q.next_pending()
    assert nxt.index == 1


# ── duplicate-processing guard ───────────────────────────────────────────

def test_marking_an_already_resolved_batch_raises():
    q = BatchQueue(_fake_batches(1))
    item = q.next_pending()
    q.mark_done(item.index, output="out0.m4b", chapters=1, duration_s=1.0)
    with pytest.raises(RuntimeError, match="not running"):
        q.mark_done(item.index, output="out0.m4b", chapters=1, duration_s=1.0)


def test_every_batch_claimed_exactly_once_across_a_full_run():
    q = BatchQueue(_fake_batches(5))
    seen: list[int] = []
    while (item := q.next_pending()) is not None:
        seen.append(item.index)
        q.mark_done(item.index, output=f"out{item.index}.m4b", chapters=1, duration_s=1.0)
    assert seen == [0, 1, 2, 3, 4]
    assert len(seen) == len(set(seen))


# ── failure isolation ─────────────────────────────────────────────────

def test_a_failed_batch_does_not_block_the_remaining_queue():
    q = BatchQueue(_fake_batches(3))
    item0 = q.next_pending()
    q.mark_failed(item0.index, "engine crashed")
    item1 = q.next_pending()
    assert item1.index == 1
    q.mark_done(item1.index, output="out1.m4b", chapters=1, duration_s=1.0)
    item2 = q.next_pending()
    assert item2.index == 2
    q.mark_done(item2.index, output="out2.m4b", chapters=1, duration_s=1.0)
    assert q.failed_batches == [0]
    assert q.completed_batches == [1, 2]
    assert q.status == "done_with_failures"


def test_all_batches_failing_reports_failed_status():
    q = BatchQueue(_fake_batches(2))
    for _ in range(2):
        item = q.next_pending()
        q.mark_failed(item.index, "boom")
    assert q.status == "failed"
    assert q.failed_batches == [0, 1]
    assert q.completed_batches == []


def test_empty_batch_counts_as_completed_not_failed():
    q = BatchQueue(_fake_batches(2))
    item0 = q.next_pending()
    q.mark_empty(item0.index)
    item1 = q.next_pending()
    q.mark_done(item1.index, output="out1.m4b", chapters=1, duration_s=1.0)
    assert q.completed_batches == [0, 1]
    assert q.failed_batches == []
    assert q.status == "done"


# ── snapshot / status transitions ───────────────────────────────────────

def test_status_is_running_while_batches_remain():
    q = BatchQueue(_fake_batches(2))
    q.next_pending()
    assert q.status == "running"
    snap = q.snapshot()
    assert snap["total_batches"] == 2
    assert snap["current_batch"] == 0
    assert snap["status"] == "running"


def test_status_is_done_when_everything_succeeds():
    q = BatchQueue(_fake_batches(2))
    for _ in range(2):
        item = q.next_pending()
        q.mark_done(item.index, output=f"out{item.index}.m4b", chapters=1, duration_s=1.0)
    assert q.status == "done"
    assert q.current_batch is None


def test_stopping_mid_queue_is_reported_as_stopped_not_running():
    q = BatchQueue(_fake_batches(3))
    item0 = q.next_pending()
    q.mark_done(item0.index, output="out0.m4b", chapters=1, duration_s=1.0)
    q.mark_stopped()
    assert q.status == "stopped"
    assert q.has_pending()  # batches 1 and 2 were never claimed


def test_single_batch_pdf():
    q = BatchQueue(_fake_batches(1))
    item = q.next_pending()
    assert item.index == 0
    q.mark_done(item.index, output="out0.m4b", chapters=1, duration_s=1.0)
    assert q.next_pending() is None
    assert q.status == "done"


# ── Stage 3: timestamps ──────────────────────────────────────────────────

def test_next_pending_stamps_started_at():
    clock_values = iter([100.0, 105.0])
    q = BatchQueue(_fake_batches(1), clock=lambda: next(clock_values))
    item = q.next_pending()
    assert item.started_at == 100.0
    assert item.finished_at is None


def test_mark_done_stamps_finished_at():
    clock_values = iter([100.0, 107.5])
    q = BatchQueue(_fake_batches(1), clock=lambda: next(clock_values))
    item = q.next_pending()
    q.mark_done(item.index, output="out.m4b", chapters=1, duration_s=1.0)
    assert item.started_at == 100.0
    assert item.finished_at == 107.5


def test_mark_failed_and_mark_empty_stamp_finished_at():
    clock_values = iter([1.0, 2.0, 3.0, 4.0])
    q = BatchQueue(_fake_batches(2), clock=lambda: next(clock_values))
    item0 = q.next_pending()
    q.mark_failed(item0.index, "boom")
    assert q.items[0].finished_at == 2.0
    item1 = q.next_pending()
    q.mark_empty(item1.index)
    assert q.items[1].finished_at == 4.0


# ── Stage 3: persistence (to_dict / to_item_dicts / from_batches_and_state) ─

def test_item_to_dict_shape():
    q = BatchQueue(_fake_batches(1), clock=lambda: 42.0)
    item = q.next_pending()
    q.mark_done(item.index, output="out0.m4b", chapters=2, duration_s=3.5)
    d = q.items[0].to_dict()
    assert d == {
        "index": 0, "start_page": 1, "end_page": 10, "status": "done",
        "output": "out0.m4b", "chapters": 2, "duration_s": 3.5, "error": None,
        "child_job_id": None, "started_at": 42.0, "finished_at": 42.0,
    }


def test_to_item_dicts_matches_item_count_and_order():
    q = BatchQueue(_fake_batches(3))
    dicts = q.to_item_dicts()
    assert [d["index"] for d in dicts] == [0, 1, 2]
    assert all(d["status"] == "pending" for d in dicts)


def test_from_batches_and_state_restores_done_and_empty_items():
    batches = _fake_batches(3)
    state = [
        {"index": 0, "start_page": 1, "end_page": 10, "status": "done",
         "output": "out0.m4b", "chapters": 1, "duration_s": 1.5, "error": None,
         "child_job_id": "parentb0", "started_at": 1.0, "finished_at": 2.0},
        {"index": 1, "start_page": 11, "end_page": 20, "status": "empty",
         "output": None, "chapters": 0, "duration_s": 0.0, "error": None,
         "child_job_id": "parentb1", "started_at": 3.0, "finished_at": 4.0},
    ]
    q = BatchQueue.from_batches_and_state(batches, state)
    assert q.items[0].status == "done"
    assert q.items[0].output == "out0.m4b"
    assert q.items[0].child_job_id == "parentb0"
    assert q.items[1].status == "empty"
    assert q.items[2].status == "pending"  # never had persisted state


def test_from_batches_and_state_resets_running_and_failed_to_pending():
    """The batch that was mid-render when the process died (still "running"
    in the manifest) and any previously-"failed" batch both retry on
    resume — the same way an ordinary re-run retries any chapter that never
    produced a cached WAV, failed or not."""
    batches = _fake_batches(2)
    state = [
        {"index": 0, "start_page": 1, "end_page": 10, "status": "running",
         "output": None, "chapters": 0, "duration_s": 0.0, "error": None,
         "child_job_id": "parentb0", "started_at": 1.0, "finished_at": None},
        {"index": 1, "start_page": 11, "end_page": 20, "status": "failed",
         "output": None, "chapters": 0, "duration_s": 0.0,
         "error": "engine crashed", "child_job_id": "parentb1",
         "started_at": 1.0, "finished_at": 2.0},
    ]
    q = BatchQueue.from_batches_and_state(batches, state)
    assert q.items[0].status == "pending"
    assert q.items[1].status == "pending"
    # child_job_id still carries over for a reset item — reusing it on retry
    # is what gets chapter-level cache/resume reuse for free.
    assert q.items[0].child_job_id == "parentb0"
    assert q.items[1].child_job_id == "parentb1"


def test_from_batches_and_state_ignores_state_for_a_changed_split():
    """If the split geometry changed (different batch_size_pages / source
    PDF) since the manifest was written, persisted state at that index must
    never be glued onto a now-different page range."""
    batches = _fake_batches(1)  # pages 1-10
    state = [
        {"index": 0, "start_page": 1, "end_page": 5, "status": "done",  # stale range
         "output": "stale.m4b", "chapters": 1, "duration_s": 1.0, "error": None,
         "child_job_id": "old", "started_at": 1.0, "finished_at": 2.0},
    ]
    q = BatchQueue.from_batches_and_state(batches, state)
    assert q.items[0].status == "pending"
    assert q.items[0].output is None


def test_from_batches_and_state_with_no_state_is_a_fresh_queue():
    q = BatchQueue.from_batches_and_state(_fake_batches(2), None)
    assert all(it.status == "pending" for it in q.items)
    q2 = BatchQueue.from_batches_and_state(_fake_batches(2), [])
    assert all(it.status == "pending" for it in q2.items)


def test_round_trip_through_to_item_dicts_and_from_batches_and_state():
    batches = _fake_batches(2)
    q1 = BatchQueue(batches, clock=lambda: 10.0)
    item = q1.next_pending()
    item.child_job_id = "parentb0"
    q1.mark_done(item.index, output="out0.m4b", chapters=1, duration_s=1.0)

    q2 = BatchQueue.from_batches_and_state(batches, q1.to_item_dicts())
    assert q2.items[0].status == "done"
    assert q2.items[0].output == "out0.m4b"
    assert q2.items[0].child_job_id == "parentb0"
    assert q2.items[1].status == "pending"


# ── restored_outputs (Fix 1: Download links for restored batches) ──────────

def test_restored_outputs_includes_done_items():
    q = BatchQueue(_fake_batches(2))
    item = q.next_pending()
    q.mark_done(item.index, output="out0.m4b", chapters=2, duration_s=3.5)
    assert q.restored_outputs() == {
        0: {"output": "out0.m4b", "chapters": 2, "duration_s": 3.5},
    }


def test_restored_outputs_includes_empty_items():
    q = BatchQueue(_fake_batches(1))
    item = q.next_pending()
    q.mark_empty(item.index)
    assert q.restored_outputs() == {0: {"output": None, "chapters": 0, "duration_s": 0.0}}


def test_restored_outputs_excludes_pending_running_and_failed_items():
    q = BatchQueue(_fake_batches(3))
    item0 = q.next_pending()
    q.mark_failed(item0.index, "boom")
    q.next_pending()  # item 1 left "running"
    # item 2 stays "pending"
    assert q.restored_outputs() == {}


def test_restored_outputs_is_empty_for_a_fresh_queue():
    q = BatchQueue(_fake_batches(3))
    assert q.restored_outputs() == {}


def test_restored_outputs_reflects_items_restored_via_from_batches_and_state():
    """The exact use case Fix 1 targets: a resumed queue's preserved items
    must surface their output filename even though this run will never emit
    a fresh batch_done for them."""
    batches = _fake_batches(2)
    state = [
        {"index": 0, "start_page": 1, "end_page": 10, "status": "done",
         "output": "preserved.m4b", "chapters": 1, "duration_s": 4.2, "error": None,
         "child_job_id": "parentb0", "started_at": 1.0, "finished_at": 2.0},
    ]
    q = BatchQueue.from_batches_and_state(batches, state)
    assert q.restored_outputs() == {
        0: {"output": "preserved.m4b", "chapters": 1, "duration_s": 4.2},
    }
