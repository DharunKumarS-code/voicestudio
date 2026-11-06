/**
 * PDF Voice Reader launcher.
 *
 * The reader is a standalone page the backend serves at `/reader/`
 * (backend/reader/). The desktop app does not embed it: this module makes sure
 * the backend is up, confirms it actually serves the reader, and opens the URL
 * in the user's default browser.
 *
 * The launcher never spawns a process of its own. All lifecycle work goes
 * through the existing BackendSupervisor, which already attaches to a running
 * backend, spawns one when none answers, moves off reserved ports and reports
 * `port_in_use`. Concurrent clicks share one in-flight launch.
 */
import { ipcMain, shell, type IpcMainInvokeEvent } from 'electron';
import type { BackendStage, BackendStatus, ReaderOpenResult } from '../preload/index.d';
import { isTrustedRenderer } from './trusted-renderer';

export type { ReaderFailure, ReaderOpenResult } from '../preload/index.d';

export const READER_CHANNELS = { open: 'reader:open' } as const;

export const READER_PATH = '/reader/';

/** The slice of BackendSupervisor the launcher needs (kept narrow for tests). */
export interface ReaderBackend {
  readonly status: BackendStatus;
  start(): Promise<void>;
  subscribe(listener: (status: BackendStatus) => void): () => void;
  requestHeaders(): Record<string, string>;
}

export interface ReaderLauncherDeps {
  backend: ReaderBackend;
  openExternal: (url: string) => Promise<void>;
  fetch?: typeof fetch;
  /** Upper bound on waiting for the backend. The supervisor applies its own
   * startup budget (300-600 s, longer while the backend keeps logging) and
   * reports `failed` when it runs out; this only stops a click hanging forever. */
  readyTimeoutMs?: number;
}

const DEFAULT_READY_TIMEOUT_MS = 15 * 60 * 1000;
const PAGE_CHECK_TIMEOUT_MS = 10_000;

/** Stages that end a wait without the backend becoming ready. */
const TERMINAL: ReadonlySet<BackendStage> = new Set<BackendStage>([
  'failed',
  'crashed',
  'port_in_use',
  'setup_required',
]);
/** Stages in which the supervisor is already working towards `ready`. */
const IN_PROGRESS: ReadonlySet<BackendStage> = new Set<BackendStage>([
  'attaching',
  'starting',
  'unresponsive',
  'installing',
]);

export function readerUrl(baseUrl: string): string {
  return new URL(READER_PATH, baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`).toString();
}

function failure(status: BackendStatus): ReaderOpenResult {
  const detail = status.message ? ` ${status.message}` : '';
  switch (status.stage) {
    case 'setup_required':
      return {
        ok: false,
        reason: 'setup_required',
        message: 'VoiceStudio has to finish setting up its voice engine before the PDF Voice Reader can open.',
      };
    case 'port_in_use':
      return {
        ok: false,
        reason: 'port_in_use',
        message: `Port ${status.port} is in use by another program, so the VoiceStudio backend could not start. Close that program, or set OMNIVOICE_PORT to a free port, then try again.`,
        port: status.port,
      };
    default:
      return {
        ok: false,
        reason: 'start_failed',
        message: `The VoiceStudio backend could not start.${detail}`,
        ...(status.message ? { detail: status.message } : {}),
      };
  }
}

/** Resolve once the supervisor reports `ready`, or with the stage that stopped it. */
function waitForBackend(
  backend: ReaderBackend,
  timeoutMs: number,
): Promise<BackendStatus | 'timeout'> {
  return new Promise((resolve) => {
    let settled = false;
    let unsubscribe: () => void = () => {};
    const finish = (value: BackendStatus | 'timeout') => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      unsubscribe();
      resolve(value);
    };
    const timer = setTimeout(() => finish('timeout'), timeoutMs);
    const check = (status: BackendStatus) => {
      if (status.stage === 'ready' || TERMINAL.has(status.stage)) finish(status);
    };
    unsubscribe = backend.subscribe(check);
    if (settled) unsubscribe();
    else check(backend.status);
  });
}

export function createReaderLauncher(deps: ReaderLauncherDeps) {
  const doFetch = deps.fetch ?? fetch;
  const readyTimeoutMs = deps.readyTimeoutMs ?? DEFAULT_READY_TIMEOUT_MS;
  let inFlight: Promise<ReaderOpenResult> | null = null;

  async function launch(): Promise<ReaderOpenResult> {
    const { backend } = deps;
    const initial = backend.status;

    // The reader talks to the backend from an ordinary browser tab, which a
    // remote backend's API-key gate rejects. Local backends trust loopback.
    if (initial.remote) {
      return {
        ok: false,
        reason: 'remote',
        message:
          'The PDF Voice Reader runs on this computer, but VoiceStudio is connected to a remote backend. Switch back to the local backend in Settings to use it.',
      };
    }
    if (initial.stage === 'setup_required') return failure(initial);

    let startedBackend = false;
    const startOnce = () => {
      // One start() attaches to a backend that is already answering, or spawns
      // exactly one. Never called while a launch is in progress, and at most
      // once per click, so a second backend is never started.
      startedBackend = true;
      void backend.start().catch(() => {
        /* the supervisor reports failures through its status */
      });
    };
    if (initial.stage !== 'ready' && !IN_PROGRESS.has(initial.stage)) startOnce();

    let settled = await waitForBackend(backend, readyTimeoutMs);
    // The backend died just before or during this click (the supervisor briefly
    // reports `attaching` while it waits for a replacement, then `crashed`, and
    // does not restart on its own). Recover with one start of our own; a start
    // this launch already made is never retried, so a backend that cannot start
    // fails cleanly instead of looping.
    if (
      settled !== 'timeout' &&
      !startedBackend &&
      (settled.stage === 'crashed' || settled.stage === 'failed')
    ) {
      startOnce();
      settled = await waitForBackend(backend, readyTimeoutMs);
    }
    if (settled === 'timeout') {
      return {
        ok: false,
        reason: 'timeout',
        message: 'The VoiceStudio backend is taking too long to start. Try again in a moment.',
      };
    }
    if (settled.stage !== 'ready') return failure(settled);

    const url = readerUrl(settled.baseUrl);
    // Confirm this backend actually serves the reader (an older backend
    // attached on the port might not) before sending the user to a 404.
    try {
      const res = await doFetch(url, {
        headers: backend.requestHeaders(),
        signal: AbortSignal.timeout(PAGE_CHECK_TIMEOUT_MS),
      });
      if (!res.ok) {
        return {
          ok: false,
          reason: 'reader_missing',
          message: `The running VoiceStudio backend does not serve the PDF Voice Reader (HTTP ${res.status}). Restart the backend and try again.`,
          url,
          detail: `HTTP ${res.status}`,
        };
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      return {
        ok: false,
        reason: 'reader_missing',
        message: `Could not reach the PDF Voice Reader page: ${detail}`,
        url,
        detail,
      };
    }

    try {
      await deps.openExternal(url);
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      return {
        ok: false,
        reason: 'browser_failed',
        message: `Could not open your web browser (${detail}). Open ${url} in your browser instead.`,
        url,
        detail,
      };
    }
    return { ok: true, url, startedBackend };
  }

  return {
    /** Open the reader; repeated calls while one is running share its result. */
    open(): Promise<ReaderOpenResult> {
      if (!inFlight) {
        inFlight = launch().finally(() => {
          inFlight = null;
        });
      }
      return inFlight;
    },
  };
}

export function registerReaderLauncher(backend: ReaderBackend): void {
  const launcher = createReaderLauncher({
    backend,
    openExternal: (url) => shell.openExternal(url),
  });
  ipcMain.handle(READER_CHANNELS.open, (event: IpcMainInvokeEvent) => {
    if (!isTrustedRenderer(event.senderFrame?.url ?? '', process.env.ELECTRON_RENDERER_URL)) {
      throw new Error('Untrusted sender');
    }
    return launcher.open();
  });
}
