import { expect, it } from 'vitest';
import { idlePdfBatchState, reducePdfBatchEvent, type PdfBatchState } from './pdf-batch-session';

function apply(state: PdfBatchState, ...events: Record<string, unknown>[]): PdfBatchState {
  return events.reduce(reducePdfBatchEvent, state);
}

it('initializes one pending item per batch on batch_started', () => {
  const state = apply(idlePdfBatchState, { type: 'batch_started', total_batches: 3 });
  expect(state.phase).toBe('running');
  expect(state.totalBatches).toBe(3);
  expect(state.items.map((i) => i.status)).toEqual(['pending', 'pending', 'pending']);
});

it('tracks batch_progress as the current batch, with its page range', () => {
  const state = apply(
    idlePdfBatchState,
    { type: 'batch_started', total_batches: 2 },
    { type: 'batch_progress', batch_index: 0, start_page: 1, end_page: 10 },
  );
  expect(state.currentBatch).toBe(0);
  expect(state.items[0]).toMatchObject({ status: 'running', startPage: 1, endPage: 10 });
  expect(state.items[1].status).toBe('pending'); // untouched until its own progress event
});

it('records a successful batch_done with its output and duration', () => {
  const state = apply(
    idlePdfBatchState,
    { type: 'batch_started', total_batches: 1 },
    { type: 'batch_progress', batch_index: 0, start_page: 1, end_page: 10 },
    { type: 'batch_done', batch_index: 0, output: 'audiobook_abc.m4b', chapters: 2, duration_s: 12.5 },
  );
  expect(state.items[0]).toMatchObject({
    status: 'done',
    output: 'audiobook_abc.m4b',
    chapters: 2,
    durationS: 12.5,
  });
});

it('records a failed batch without disturbing other batches', () => {
  const state = apply(
    idlePdfBatchState,
    { type: 'batch_started', total_batches: 3 },
    { type: 'batch_progress', batch_index: 0, start_page: 1, end_page: 10 },
    { type: 'batch_done', batch_index: 0, output: 'a.m4b', chapters: 1, duration_s: 1 },
    { type: 'batch_progress', batch_index: 1, start_page: 11, end_page: 20 },
    { type: 'batch_failed', batch_index: 1, error: 'engine crashed' },
    { type: 'batch_progress', batch_index: 2, start_page: 21, end_page: 30 },
    { type: 'batch_done', batch_index: 2, output: 'c.m4b', chapters: 1, duration_s: 1 },
  );
  expect(state.items.map((i) => i.status)).toEqual(['done', 'failed', 'done']);
  expect(state.items[1].error).toBe('engine crashed');
  expect(state.items[0].output).toBe('a.m4b');
  expect(state.items[2].output).toBe('c.m4b');
});

// ── Stage 4 bug fix: a detailed mux-failure reason survives verbatim ──────

it('captures a detailed ffmpeg-mux failure reason verbatim, not a generic placeholder', () => {
  // The exact string the backend's build_failure_event fix now sends for a
  // real ffmpeg assembly failure (confirmed via an actual integration run of
  // _render_pdf_batches_sse against a monkeypatched failing run_ffmpeg).
  const detailed = 'ffmpeg assembly failed (exit code 1)';
  const state = apply(
    idlePdfBatchState,
    { type: 'batch_started', total_batches: 1 },
    { type: 'batch_progress', batch_index: 0, start_page: 1, end_page: 10 },
    { type: 'batch_failed', batch_index: 0, error: detailed },
  );
  expect(state.items[0].status).toBe('failed');
  expect(state.items[0].error).toBe(detailed);
  expect(state.items[0].error).not.toBe('render failed (see backend log)');
});

it('marks an all-blank batch window as empty, not failed', () => {
  const state = apply(
    idlePdfBatchState,
    { type: 'batch_started', total_batches: 1 },
    { type: 'batch_progress', batch_index: 0, start_page: 1, end_page: 1 },
    { type: 'batch_empty', batch_index: 0 },
  );
  expect(state.items[0].status).toBe('empty');
});

it('settles into done on batch_summary and surfaces its overall status', () => {
  const state = apply(
    idlePdfBatchState,
    { type: 'batch_started', total_batches: 1 },
    { type: 'batch_progress', batch_index: 0, start_page: 1, end_page: 1 },
    { type: 'batch_done', batch_index: 0, output: 'a.m4b', chapters: 1, duration_s: 1 },
    { type: 'batch_summary', total_batches: 1, completed_batches: [0], failed_batches: [], status: 'done' },
  );
  expect(state.phase).toBe('done');
  expect(state.currentBatch).toBeNull();
  expect(state.overallStatus).toBe('done');
});

it('surfaces a pre-split error (e.g. a corrupt PDF) without a batch index', () => {
  const state = apply(idlePdfBatchState, { type: 'error', error: 'not a valid PDF file' });
  expect(state.phase).toBe('done');
  expect(state.error).toBe('not a valid PDF file');
});

it('ignores an out-of-range batch index rather than throwing', () => {
  const state = apply(
    idlePdfBatchState,
    { type: 'batch_started', total_batches: 1 },
    { type: 'batch_done', batch_index: 5, output: 'x.m4b', chapters: 1, duration_s: 1 },
  );
  expect(state.items).toHaveLength(1);
  expect(state.items[0].status).toBe('pending');
});

it('ignores unknown/chapter-level passthrough event types', () => {
  const started = apply(idlePdfBatchState, { type: 'batch_started', total_batches: 1 });
  const state = apply(started, { type: 'chapter', batch_index: 0, index: 0, title: 'One' });
  expect(state).toEqual(started);
});

// ── Stage 3: resume / recovery ──────────────────────────────────────────

it('a resumed run\'s batch_started restores completed/failed batches, not all-pending', () => {
  const state = apply(idlePdfBatchState, {
    type: 'batch_started',
    total_batches: 3,
    completed_batches: [0],
    failed_batches: [1],
  });
  expect(state.items.map((i) => i.status)).toEqual(['done', 'failed', 'pending']);
});

it('a fresh (non-resume) batch_started with no completed/failed arrays is unaffected', () => {
  const state = apply(idlePdfBatchState, { type: 'batch_started', total_batches: 2 });
  expect(state.items.map((i) => i.status)).toEqual(['pending', 'pending']);
});

// ── Fix 1: restored_outputs populates a preserved batch's Download link ────

it('batch_started restored_outputs populates output/chapters/duration for a preserved batch', () => {
  const state = apply(idlePdfBatchState, {
    type: 'batch_started',
    total_batches: 3,
    completed_batches: [0],
    failed_batches: [],
    restored_outputs: { 0: { output: 'audiobook_x_b0.m4b', chapters: 2, duration_s: 12.5 } },
  });
  expect(state.items[0]).toMatchObject({
    status: 'done',
    output: 'audiobook_x_b0.m4b',
    chapters: 2,
    durationS: 12.5,
  });
  // Batches 1/2 have no restored_outputs entry — untouched, no stale link.
  expect(state.items[1].output).toBeUndefined();
  expect(state.items[2].output).toBeUndefined();
});

it('a fresh (non-resume) batch_started with no restored_outputs leaves every item without an output', () => {
  const state = apply(idlePdfBatchState, { type: 'batch_started', total_batches: 2 });
  expect(state.items.every((i) => i.output === undefined)).toBe(true);
});

it('restored_outputs is ignored for an item not marked completed (defensive)', () => {
  // A malformed/unexpected payload (restored_outputs naming an index the
  // completed/failed lists don't) must not fabricate a done-looking item.
  const state = apply(idlePdfBatchState, {
    type: 'batch_started',
    total_batches: 2,
    completed_batches: [],
    failed_batches: [],
    restored_outputs: { 1: { output: 'suspicious.m4b', chapters: 1, duration_s: 1 } },
  });
  expect(state.items[1].status).toBe('pending');
});

it('records a recovery_summary without disturbing restored item state', () => {
  const state = apply(
    idlePdfBatchState,
    { type: 'batch_started', total_batches: 2, completed_batches: [0], failed_batches: [] },
    { type: 'recovery_summary', verified: 1, invalidated: 0 },
  );
  expect(state.recovery).toEqual({ verified: 1, invalidated: 0 });
  expect(state.items[0].status).toBe('done');
});

it('a resumed run that retries a preserved-invalid batch updates it via the normal events', () => {
  const state = apply(
    idlePdfBatchState,
    { type: 'batch_started', total_batches: 2, completed_batches: [0], failed_batches: [] },
    { type: 'recovery_summary', verified: 0, invalidated: 1 },
    { type: 'batch_progress', batch_index: 0, start_page: 1, end_page: 10 },
    { type: 'batch_done', batch_index: 0, output: 'regenerated.m4b', chapters: 1, duration_s: 2 },
  );
  expect(state.recovery).toEqual({ verified: 0, invalidated: 1 });
  expect(state.items[0]).toMatchObject({ status: 'done', output: 'regenerated.m4b' });
});

it('recovery stays null across an ordinary fresh (non-resume) run', () => {
  const state = apply(
    idlePdfBatchState,
    { type: 'batch_started', total_batches: 1 },
    { type: 'batch_progress', batch_index: 0, start_page: 1, end_page: 1 },
    { type: 'batch_done', batch_index: 0, output: 'a.m4b', chapters: 1, duration_s: 1 },
  );
  expect(state.recovery).toBeNull();
});

// ── Stage 4: job id capture (for stable per-batch dock playback sources) ──

it('captures job_id from batch_started', () => {
  const state = apply(idlePdfBatchState, {
    type: 'batch_started',
    job_id: 'job-abc123',
    total_batches: 1,
  });
  expect(state.jobId).toBe('job-abc123');
});

it('keeps jobId unchanged across subsequent batch events', () => {
  const state = apply(
    idlePdfBatchState,
    { type: 'batch_started', job_id: 'job-abc123', total_batches: 1 },
    { type: 'batch_progress', batch_index: 0, start_page: 1, end_page: 10 },
    { type: 'batch_done', batch_index: 0, output: 'a.m4b', chapters: 1, duration_s: 1 },
    { type: 'batch_summary', total_batches: 1, completed_batches: [0], failed_batches: [], status: 'done' },
  );
  expect(state.jobId).toBe('job-abc123');
});

it('a resumed run\'s batch_started still reports the (reused) job_id', () => {
  const state = apply(idlePdfBatchState, {
    type: 'batch_started',
    job_id: 'job-abc123',
    total_batches: 2,
    completed_batches: [0],
    failed_batches: [],
    restored_outputs: { 0: { output: 'audiobook_x_b0.m4b', chapters: 2, duration_s: 12.5 } },
  });
  expect(state.jobId).toBe('job-abc123');
  expect(state.items[0]).toMatchObject({ status: 'done', output: 'audiobook_x_b0.m4b' });
});

it('jobId stays null when batch_started omits job_id (defensive)', () => {
  const state = apply(idlePdfBatchState, { type: 'batch_started', total_batches: 1 });
  expect(state.jobId).toBeNull();
});
