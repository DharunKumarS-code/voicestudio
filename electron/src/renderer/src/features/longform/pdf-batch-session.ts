/**
 * Batch PDF audiobook import (Stage 2): splits a large PDF into page-aligned
 * batches on the backend and renders them one at a time, each as its own
 * independent output file. Deliberately self-contained — it does not touch
 * `longform-session.ts`'s draft/session state, so the existing single-script
 * Audiobook flow is unaffected either way.
 *
 * Mirrors `renderLongform`'s fetch + `consumeLongformStream` shape (see
 * longform-session.ts), but against `POST /audiobook/pdf-batches` and its
 * `batch_*` SSE events instead of the single-job `done`/`chapter` events.
 *
 * Stage 3 adds `resume()` (same reducer, same event vocabulary — the backend
 * streams the identical `batch_*` shape for a resumed run, plus one
 * `recovery_summary` event up front) and `fetchResumableJobs()`, mirroring
 * how `longform-session.ts`/`render-recovery.tsx` offer resume for the
 * single-script flow.
 */
import { useCallback, useRef, useState } from 'react';
import { apiFetch, apiJson, describeError } from '@/lib/api/client';
import { consumeLongformStream } from '@shared/utils/longformStream';

export type PdfBatchItemStatus = 'pending' | 'running' | 'done' | 'failed' | 'empty';

export interface PdfBatchItem {
  index: number;
  startPage: number | null;
  endPage: number | null;
  status: PdfBatchItemStatus;
  output?: string;
  chapters?: number;
  durationS?: number;
  error?: string;
}

/** Forwarded verbatim as the `options` JSON part of the multipart request —
 * the same render knobs `renderBody` builds for /audiobook (default_voice,
 * language, format, loudness, cover_path, metadata, voice_map, expressive
 * overrides, …), loosely typed since the backend is the source of truth for
 * its shape (`PdfBatchAudiobookRequest`). */
export type PdfBatchRenderOptions = Record<string, unknown>;

export interface PdfBatchState {
  phase: 'idle' | 'uploading' | 'running' | 'done';
  /** The backend job id from `batch_started`'s `job_id` (Stage 4: namespaces
   * each batch's dock playback source so it stays stable across a resume,
   * which reuses the same job id). Null until the first `batch_started`. */
  jobId: string | null;
  totalBatches: number;
  currentBatch: number | null;
  items: PdfBatchItem[];
  overallStatus: string | null;
  error: string | null;
  /** Set only on a resumed run, from its one-time `recovery_summary` event
   * (Stage 3): how many previously-completed batches were re-verified vs.
   * found missing/corrupt and reset to retry. */
  recovery: { verified: number; invalidated: number } | null;
}

export const idlePdfBatchState: PdfBatchState = {
  phase: 'idle',
  jobId: null,
  totalBatches: 0,
  currentBatch: null,
  items: [],
  overallStatus: null,
  error: null,
  recovery: null,
};

/** One interrupted PDF-batch job the user can resume (GET
 * /audiobook/pdf-batches/jobs). */
export interface ResumablePdfBatchJob {
  jobId: string;
  title: string;
  totalBatches: number;
  completedBatches: number[];
  failedBatches: number[];
}

export async function fetchResumablePdfBatchJobs(): Promise<ResumablePdfBatchJob[]> {
  const data = await apiJson<{
    jobs: Array<{
      job_id: string;
      title?: string;
      total_batches?: number;
      completed_batches?: number[];
      failed_batches?: number[];
    }>;
  }>('/audiobook/pdf-batches/jobs');
  return (data.jobs ?? []).map((j) => ({
    jobId: j.job_id,
    title: j.title ?? '',
    totalBatches: j.total_batches ?? 0,
    completedBatches: j.completed_batches ?? [],
    failedBatches: j.failed_batches ?? [],
  }));
}

function num(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

/** Pure reducer over one parsed `batch_*` SSE event — kept free of React so
 * the state machine (ordering, one-batch-at-a-time display, failure
 * isolation) is directly unit-testable, the same way render-eta.ts's
 * renderStatus()/parseProgressEvent() are. */
export function reducePdfBatchEvent(
  prev: PdfBatchState,
  event: Record<string, unknown>,
): PdfBatchState {
  const type = typeof event.type === 'string' ? event.type : '';
  const index = num(event.batch_index);

  if (type === 'batch_started') {
    const total = num(event.total_batches) ?? 0;
    // A resumed run's batch_started already reflects restored state (some
    // batches preserved as completed) — completed_batches here means
    // "done or empty" (the queue snapshot doesn't distinguish the two), so
    // mark them "done"; a later resume that actually retries one of them
    // would correct it via its own batch_progress/batch_done event.
    const completed = new Set(
      Array.isArray(event.completed_batches) ? (event.completed_batches as number[]) : [],
    );
    const failed = new Set(
      Array.isArray(event.failed_batches) ? (event.failed_batches as number[]) : [],
    );
    // Fix 1: a preserved batch (restored as "done" above) never gets its
    // own batch_done event this run — batch_started's restored_outputs is
    // the only place its filename/chapters/duration ever arrives, which is
    // what makes its Download link show up immediately after a resume
    // instead of only after a page reload.
    const restored =
      event.restored_outputs && typeof event.restored_outputs === 'object'
        ? (event.restored_outputs as Record<
            string,
            { output?: unknown; chapters?: unknown; duration_s?: unknown }
          >)
        : {};
    return {
      ...prev,
      phase: 'running',
      jobId: typeof event.job_id === 'string' ? event.job_id : prev.jobId,
      totalBatches: total,
      currentBatch: null,
      items: Array.from({ length: total }, (_, i) => {
        const r = restored[String(i)];
        return {
          index: i,
          startPage: null,
          endPage: null,
          status: completed.has(i) ? 'done' : failed.has(i) ? 'failed' : 'pending',
          output: r && typeof r.output === 'string' ? r.output : undefined,
          chapters: r ? (num(r.chapters) ?? undefined) : undefined,
          durationS: r ? (num(r.duration_s) ?? undefined) : undefined,
        };
      }),
    };
  }
  if (index === null || index < 0 || index >= prev.items.length) {
    if (type === 'batch_summary') {
      return {
        ...prev,
        phase: 'done',
        currentBatch: null,
        overallStatus: typeof event.status === 'string' ? event.status : prev.overallStatus,
      };
    }
    if (type === 'recovery_summary') {
      return {
        ...prev,
        recovery: { verified: num(event.verified) ?? 0, invalidated: num(event.invalidated) ?? 0 },
      };
    }
    if (type === 'error' && typeof event.error === 'string') {
      return { ...prev, phase: 'done', error: event.error };
    }
    return prev;
  }
  const items = prev.items.slice();
  const current = items[index];
  if (type === 'batch_progress') {
    items[index] = {
      ...current,
      status: 'running',
      startPage: num(event.start_page),
      endPage: num(event.end_page),
    };
    return { ...prev, items, currentBatch: index };
  }
  if (type === 'batch_empty') {
    items[index] = { ...current, status: 'empty' };
    return { ...prev, items };
  }
  if (type === 'batch_done') {
    items[index] = {
      ...current,
      status: 'done',
      output: typeof event.output === 'string' ? event.output : undefined,
      chapters: num(event.chapters) ?? undefined,
      durationS: num(event.duration_s) ?? undefined,
    };
    return { ...prev, items };
  }
  if (type === 'batch_failed') {
    items[index] = {
      ...current,
      status: 'failed',
      error: typeof event.error === 'string' ? event.error : 'failed',
    };
    return { ...prev, items };
  }
  return prev;
}

export function usePdfBatchImport() {
  const [state, setState] = useState<PdfBatchState>(idlePdfBatchState);
  const controllerRef = useRef<AbortController | null>(null);

  const handleEvent = useCallback((event: Record<string, unknown>) => {
    setState((prev) => reducePdfBatchEvent(prev, event));
  }, []);

  const start = useCallback(
    async (file: File, options: PdfBatchRenderOptions, batchSizePages?: number) => {
      if (controllerRef.current) return;
      const controller = new AbortController();
      controllerRef.current = controller;
      setState({ ...idlePdfBatchState, phase: 'uploading' });
      try {
        const form = new FormData();
        form.set('file', file);
        form.set('options', JSON.stringify(options));
        if (batchSizePages) form.set('batch_size_pages', String(batchSizePages));
        const response = await apiFetch('/audiobook/pdf-batches', {
          method: 'POST',
          body: form,
          signal: controller.signal,
        });
        await consumeLongformStream(response, handleEvent, { signal: controller.signal });
      } catch (cause) {
        if (!controller.signal.aborted)
          setState((prev) => ({ ...prev, phase: 'done', error: describeError(cause) }));
      } finally {
        if (controllerRef.current === controller) controllerRef.current = null;
      }
    },
    [handleEvent],
  );

  const resume = useCallback(
    async (jobId: string) => {
      if (controllerRef.current) return;
      const controller = new AbortController();
      controllerRef.current = controller;
      setState({ ...idlePdfBatchState, phase: 'uploading' });
      try {
        const response = await apiFetch(
          `/audiobook/pdf-batches/resume/${encodeURIComponent(jobId)}`,
          { method: 'POST', signal: controller.signal },
        );
        await consumeLongformStream(response, handleEvent, { signal: controller.signal });
      } catch (cause) {
        if (!controller.signal.aborted)
          setState((prev) => ({ ...prev, phase: 'done', error: describeError(cause) }));
      } finally {
        if (controllerRef.current === controller) controllerRef.current = null;
      }
    },
    [handleEvent],
  );

  const stop = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
  }, []);

  const reset = useCallback(() => setState(idlePdfBatchState), []);

  return {
    state,
    start,
    resume,
    stop,
    reset,
    busy: state.phase === 'uploading' || state.phase === 'running',
  };
}
