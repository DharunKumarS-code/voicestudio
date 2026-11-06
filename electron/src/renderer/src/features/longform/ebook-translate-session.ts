/**
 * Ebook Translation: translates the currently open Audiobook project's script
 * chapter-by-chapter via the existing backend service
 * (`services.ebook_translate` / `POST /audiobook/translate`), streaming SSE
 * progress. Deliberately scoped to the project already open in the Audiobook
 * workspace — it reuses that project's id and script instead of asking the
 * user to pick or paste one, mirroring how `PdfBatchPanel` reuses the open
 * project's render options. See `backend/api/routers/audiobook.py`'s
 * `audiobook_translate`/`get_translate_status` docstrings for the exact
 * request/response contract this file follows.
 *
 * Mirrors `pdf-batch-session.ts`'s shape (pure reducer + `apiFetch` +
 * `consumeLongformStream`), but against `POST /audiobook/translate`'s
 * `started`/`chapter`/`chapter_error`/`stopped`/`done`/`error` event
 * vocabulary instead of PDF Batch's `batch_*` events, and a JSON body instead
 * of multipart (matching `renderLongform`'s `/audiobook` call in
 * longform-session.ts).
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch, apiJson, describeError } from '@/lib/api/client';
import { consumeLongformStream } from '@shared/utils/longformStream';

export type EbookTranslateChapterStatus = 'pending' | 'done' | 'failed';

export interface EbookTranslateChapter {
  index: number;
  title: string;
  status: EbookTranslateChapterStatus;
  cached?: boolean;
  error?: string;
}

export interface GlossaryEntry {
  source: string;
  target: string;
  note?: string;
}

export interface EbookTranslateState {
  phase: 'idle' | 'running' | 'done';
  projectId: string | null;
  totalChapters: number;
  chapters: EbookTranslateChapter[];
  /** Set once the stream's terminal `done` event arrives — the same flag
   * `GET /audiobook/translate/{project_id}` exposes as `complete`: false
   * whenever any chapter failed. */
  complete: boolean | null;
  error: string | null;
}

export const idleEbookTranslateState: EbookTranslateState = {
  phase: 'idle',
  projectId: null,
  totalChapters: 0,
  chapters: [],
  complete: null,
  error: null,
};

/** `GET /audiobook/translate/{project_id}` response, as needed here — see the
 * route's docstring in backend/api/routers/audiobook.py for the full shape. */
export interface EbookTranslateStatus {
  project_id: string;
  total_chapters: number;
  translated_chapters: number;
  failed_chapters: number[];
  target_lang: string | null;
  provider: string | null;
  complete: boolean;
  chapters: Array<{
    index: number;
    status: EbookTranslateChapterStatus;
    error: string | null;
    source_title: string | null;
  }>;
}

export async function fetchEbookTranslateStatus(
  projectId: string,
): Promise<EbookTranslateStatus> {
  return apiJson<EbookTranslateStatus>(`/audiobook/translate/${encodeURIComponent(projectId)}`);
}

/** Maps a persisted status response onto the same state shape the live SSE
 * reducer produces, so a project reopened mid-translation (or already fully
 * translated) renders identically to one just finished in this session. */
export function stateFromStatus(status: EbookTranslateStatus): EbookTranslateState {
  const failed = new Set(status.failed_chapters);
  return {
    phase: status.total_chapters > 0 ? 'done' : 'idle',
    projectId: status.project_id,
    totalChapters: status.total_chapters,
    complete: status.total_chapters > 0 ? status.complete : null,
    error: null,
    chapters: status.chapters.map((c) => ({
      index: c.index,
      title: c.source_title || '',
      status: failed.has(c.index) ? 'failed' : c.status,
      error: c.error ?? undefined,
    })),
  };
}

function num(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

/** Pure reducer over one parsed SSE event from `POST /audiobook/translate` —
 * kept free of React, the same way `reducePdfBatchEvent` is. */
export function reduceEbookTranslateEvent(
  prev: EbookTranslateState,
  event: Record<string, unknown>,
): EbookTranslateState {
  const type = typeof event.type === 'string' ? event.type : '';

  if (type === 'started') {
    const total = num(event.chapters) ?? 0;
    return {
      ...prev,
      phase: 'running',
      projectId: typeof event.project_id === 'string' ? event.project_id : prev.projectId,
      totalChapters: total,
      complete: null,
      error: null,
      chapters: Array.from({ length: total }, (_, i) => ({
        index: i,
        title: '',
        status: 'pending',
      })),
    };
  }

  const index = num(event.index);
  if (index !== null && index >= 0 && index < prev.chapters.length) {
    const chapters = prev.chapters.slice();
    if (type === 'chapter') {
      chapters[index] = {
        index,
        title: typeof event.title === 'string' ? event.title : '',
        status: 'done',
        cached: event.cached === true,
      };
      return { ...prev, chapters };
    }
    if (type === 'chapter_error') {
      chapters[index] = {
        index,
        title: typeof event.title === 'string' ? event.title : '',
        status: 'failed',
        error: typeof event.error === 'string' ? event.error : 'failed',
      };
      return { ...prev, chapters };
    }
  }

  if (type === 'stopped') {
    return { ...prev, phase: 'done' };
  }
  if (type === 'done') {
    return { ...prev, phase: 'done', complete: event.complete === true };
  }
  if (type === 'error' && typeof event.error === 'string') {
    return { ...prev, phase: 'done', error: event.error };
  }
  return prev;
}

export interface StartEbookTranslateOptions {
  projectId: string;
  text: string;
  targetLang: string;
  sourceLang?: string;
  provider?: string;
  glossary?: GlossaryEntry[];
  translationInstructions?: string;
}

export function useEbookTranslate() {
  const [state, setState] = useState<EbookTranslateState>(idleEbookTranslateState);
  const controllerRef = useRef<AbortController | null>(null);

  const handleEvent = useCallback((event: Record<string, unknown>) => {
    setState((prev) => reduceEbookTranslateEvent(prev, event));
  }, []);

  const start = useCallback(
    async (options: StartEbookTranslateOptions) => {
      // Mirrors pdf-batch-session.ts's guard: a second click while a run is
      // already in flight is a no-op, not a second concurrent request (the
      // backend itself also rejects a second run for the same project with
      // 409, but this avoids even sending it).
      if (controllerRef.current) return;
      const controller = new AbortController();
      controllerRef.current = controller;
      setState({ ...idleEbookTranslateState, projectId: options.projectId, phase: 'running' });
      try {
        const response = await apiFetch('/audiobook/translate', {
          method: 'POST',
          signal: controller.signal,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            project_id: options.projectId,
            text: options.text,
            target_lang: options.targetLang,
            source_lang: options.sourceLang || undefined,
            provider: options.provider || undefined,
            glossary: options.glossary?.length ? options.glossary : undefined,
            translation_instructions: options.translationInstructions || undefined,
          }),
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

  const stop = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
  }, []);

  const reset = useCallback(() => setState(idleEbookTranslateState), []);

  const loadStatus = useCallback(async (projectId: string) => {
    try {
      const status = await fetchEbookTranslateStatus(projectId);
      setState(stateFromStatus(status));
    } catch {
      // Best-effort: no prior run, or the backend is momentarily unreachable
      // (surfaced elsewhere by EngineNotice) — the panel just starts idle.
    }
  }, []);

  return {
    state,
    start,
    stop,
    reset,
    loadStatus,
    busy: state.phase === 'running',
  };
}

/** Loads a project's persisted translation status once, on mount/project
 * change — lets a reopened project show its prior progress immediately
 * instead of only after the user starts a new run. */
export function useEbookTranslateStatusOnMount(
  projectId: string | null,
  loadStatus: (projectId: string) => Promise<void>,
) {
  useEffect(() => {
    if (!projectId) return;
    let cancelled = false;
    void (async () => {
      if (!cancelled) await loadStatus(projectId);
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);
}
