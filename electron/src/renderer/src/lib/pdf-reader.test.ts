import { afterEach, beforeEach, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  bridge: null as null | { reader: { open: ReturnType<typeof vi.fn> } },
  toast: { loading: vi.fn(), success: vi.fn(), error: vi.fn() },
}));
vi.mock('@/components/bridge', () => ({ getBridge: () => mocks.bridge }));
vi.mock('sonner', () => ({ toast: mocks.toast }));
// Echo the key and its interpolation values, so tests can see what was asked for.
vi.mock('i18next', () => ({
  default: {
    t: (key: string, values?: Record<string, unknown>) =>
      values ? `${key}${JSON.stringify(values)}` : key,
  },
}));

import { describeReaderFailure, openPdfReader, webReaderUrl } from './pdf-reader';

beforeEach(() => {
  vi.clearAllMocks();
  mocks.bridge = null;
});
afterEach(() => vi.restoreAllMocks());

it('asks main to launch the reader and confirms it in a toast', async () => {
  const open = vi.fn(async () => ({ ok: true, url: 'http://127.0.0.1:3900/reader/', startedBackend: false }));
  mocks.bridge = { reader: { open } };
  const result = await openPdfReader();
  expect(result.ok).toBe(true);
  expect(mocks.toast.loading).toHaveBeenCalledWith('pdfReader.opening', expect.anything());
  expect(mocks.toast.success).toHaveBeenCalledWith('pdfReader.opened', expect.anything());
  expect(mocks.toast.error).not.toHaveBeenCalled();
});

it('shows a translated reason, with the port, instead of failing silently', async () => {
  mocks.bridge = {
    reader: {
      open: vi.fn(async () => ({
        ok: false,
        reason: 'port_in_use',
        message: 'Port 4900 is in use',
        port: 4900,
      })),
    },
  };
  await openPdfReader();
  expect(mocks.toast.error).toHaveBeenCalledWith(
    'pdfReader.failed',
    expect.objectContaining({ description: 'pdfReader.error_port_in_use{"port":4900}' }),
  );
});

it.each([
  ['remote', 'pdfReader.error_remote'],
  ['setup_required', 'pdfReader.error_setup_required'],
  ['timeout', 'pdfReader.error_timeout'],
  ['start_failed', 'pdfReader.error_start_failed'],
  ['reader_missing', 'pdfReader.error_reader_missing'],
] as const)('translates the %s failure', (reason, key) => {
  expect(describeReaderFailure({ ok: false, reason, message: 'english' })).toBe(key);
});

it('keeps the backend’s own technical detail next to a start failure', () => {
  expect(
    describeReaderFailure({
      ok: false,
      reason: 'start_failed',
      message: 'english',
      detail: 'uvicorn exited with code 1',
    }),
  ).toBe('pdfReader.error_start_failed (uvicorn exited with code 1)');
});

it('gives the URL to open by hand when the browser cannot be opened', () => {
  expect(
    describeReaderFailure({
      ok: false,
      reason: 'browser_failed',
      message: 'english',
      url: 'http://127.0.0.1:4900/reader/',
    }),
  ).toBe('pdfReader.error_browser_failed{"url":"http://127.0.0.1:4900/reader/"}');
});

it('turns a rejected IPC call into an error toast', async () => {
  mocks.bridge = { reader: { open: vi.fn(async () => Promise.reject(new Error('Untrusted sender'))) } };
  const result = await openPdfReader();
  expect(result).toMatchObject({ ok: false, reason: 'start_failed', detail: 'Untrusted sender' });
  expect(mocks.toast.error).toHaveBeenCalled();
});

it('joins repeated clicks onto the launch already in flight', async () => {
  let finish!: (value: unknown) => void;
  const open = vi.fn(() => new Promise((resolve) => (finish = resolve)));
  mocks.bridge = { reader: { open } };
  const first = openPdfReader();
  const second = openPdfReader();
  expect(second).toBe(first);
  finish({ ok: true, url: 'http://127.0.0.1:3900/reader/', startedBackend: true });
  await first;
  expect(open).toHaveBeenCalledTimes(1);
});

it('derives the web build’s reader URL from the API base, never an assumed port', () => {
  const url = webReaderUrl();
  expect(url.startsWith(window.location.origin)).toBe(true);
  expect(url.endsWith('/reader/')).toBe(true);
  expect(url).not.toContain(':3900');
});

it('opens the reader in a new tab when running as a plain web page', async () => {
  const tab = { opener: {} } as unknown as Window;
  const spy = vi.spyOn(window, 'open').mockReturnValue(tab);
  const result = await openPdfReader();
  expect(spy).toHaveBeenCalledWith(webReaderUrl(), '_blank');
  expect(tab.opener).toBeNull();
  expect(result.ok).toBe(true);
});

it('reports a blocked popup in the web build', async () => {
  vi.spyOn(window, 'open').mockReturnValue(null);
  const result = await openPdfReader();
  expect(result).toMatchObject({ ok: false, reason: 'browser_failed' });
  expect(mocks.toast.error).toHaveBeenCalled();
});
