import { toast } from 'sonner';
import i18n from 'i18next';
import { getBridge } from '@/components/bridge';
import { absoluteApiBase } from '@/lib/api/client';
import type { ReaderOpenResult } from '../../../preload/index.d';

type ReaderFailureResult = Extract<ReaderOpenResult, { ok: false }>;

const TOAST_ID = 'pdf-voice-reader';

let pending: Promise<ReaderOpenResult> | null = null;

/**
 * Where a plain browser tab (the web build, no Electron bridge) finds the reader:
 * next to the API this tab already talks to — the backend's own origin in a
 * deployment, a configured public API base, or the dev server's /api proxy.
 * Never an assumed port.
 */
export function webReaderUrl(): string {
  return `${absoluteApiBase()}/reader/`;
}

/** A translated explanation of why the reader could not open. */
export function describeReaderFailure(result: ReaderFailureResult): string {
  const t = i18n.t.bind(i18n);
  const url = result.url ?? '';
  let text: string;
  switch (result.reason) {
    case 'remote':
      text = t('pdfReader.error_remote');
      break;
    case 'setup_required':
      text = t('pdfReader.error_setup_required');
      break;
    case 'port_in_use':
      text = t('pdfReader.error_port_in_use', { port: result.port ?? '' });
      break;
    case 'timeout':
      text = t('pdfReader.error_timeout');
      break;
    case 'reader_missing':
      text = t('pdfReader.error_reader_missing');
      break;
    case 'browser_failed':
      text = t('pdfReader.error_browser_failed', { url });
      break;
    default:
      text = t('pdfReader.error_start_failed');
  }
  // Technical detail (the backend's own words) stays untranslated, in brackets.
  const withDetail =
    result.detail && (result.reason === 'start_failed' || result.reason === 'reader_missing')
      ? `${text} (${result.detail})`
      : text;
  return withDetail;
}

/**
 * Open the PDF Voice Reader in the user's default browser.
 *
 * In the desktop app, main makes sure the local backend is running (reusing it,
 * or starting it through the existing supervisor) and that it serves /reader/
 * before opening the browser. Repeated clicks while that is happening join the
 * same launch instead of starting another.
 */
export function openPdfReader(): Promise<ReaderOpenResult> {
  if (pending) return pending;
  const t = i18n.t.bind(i18n);
  const bridge = getBridge();

  if (!bridge?.reader) {
    const url = webReaderUrl();
    // No 'noopener' feature: with it window.open always returns null, which would
    // be indistinguishable from a blocked popup. Sever the opener by hand instead.
    const opened = window.open(url, '_blank');
    if (opened) {
      opened.opener = null;
      return Promise.resolve({ ok: true, url, startedBackend: false });
    }
    const failure: ReaderFailureResult = {
      ok: false,
      reason: 'browser_failed',
      message: `Open ${url} in your browser.`,
      url,
    };
    toast.error(t('pdfReader.failed'), { id: TOAST_ID, description: describeReaderFailure(failure) });
    return Promise.resolve(failure);
  }

  toast.loading(t('pdfReader.opening'), { id: TOAST_ID });
  pending = bridge.reader
    .open()
    .catch(
      (error: unknown): ReaderOpenResult => ({
        ok: false,
        reason: 'start_failed',
        message: error instanceof Error ? error.message : String(error),
        detail: error instanceof Error ? error.message : String(error),
      }),
    )
    .then((result) => {
      if (result.ok) toast.success(t('pdfReader.opened'), { id: TOAST_ID });
      else
        toast.error(t('pdfReader.failed'), {
          id: TOAST_ID,
          description: describeReaderFailure(result),
          duration: 12_000,
        });
      return result;
    })
    .finally(() => {
      pending = null;
    });
  return pending;
}
