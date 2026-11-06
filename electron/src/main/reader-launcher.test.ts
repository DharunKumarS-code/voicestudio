// @vitest-environment node
import { EventEmitter } from 'node:events';
import { describe, expect, it, vi } from 'vitest';
import type { BackendStage, BackendStatus } from '../preload/index.d';
import { createReaderLauncher, readerUrl, type ReaderBackend } from './reader-launcher';

vi.mock('electron', () => ({ ipcMain: { handle: vi.fn() }, shell: { openExternal: vi.fn() } }));

function status(stage: BackendStage, extra: Partial<BackendStatus> = {}): BackendStatus {
  return {
    stage,
    baseUrl: 'http://127.0.0.1:3900',
    port: 3900,
    managed: false,
    remote: false,
    elapsedMs: 0,
    logTail: [],
    ...extra,
  } as BackendStatus;
}

/** A supervisor stand-in: start() follows a scripted list of stages. */
class FakeBackend extends EventEmitter implements ReaderBackend {
  current: BackendStatus;
  starts = 0;
  constructor(
    initial: BackendStatus,
    private readonly onStart: (fake: FakeBackend) => void = () => {},
  ) {
    super();
    this.current = initial;
  }
  get status() {
    return this.current;
  }
  set(next: BackendStatus) {
    this.current = next;
    this.emit('status', next);
  }
  start() {
    this.starts += 1;
    // Like BackendSupervisor.start(): the stage flips synchronously.
    this.set(status('attaching'));
    this.onStart(this);
    return Promise.resolve();
  }
  subscribe(listener: (s: BackendStatus) => void) {
    this.on('status', listener);
    return () => this.off('status', listener);
  }
  requestHeaders() {
    return { 'X-VoiceStudio-Desktop-Origin': 'app://voicestudio' };
  }
}

const okFetch = vi.fn(async () => new Response('<html></html>', { status: 200 }));

describe('readerUrl', () => {
  it('appends /reader/ to the backend base URL', () => {
    expect(readerUrl('http://127.0.0.1:3900')).toBe('http://127.0.0.1:3900/reader/');
    expect(readerUrl('http://127.0.0.1:4900/')).toBe('http://127.0.0.1:4900/reader/');
  });
});

describe('createReaderLauncher', () => {
  it('reuses a ready backend without starting another', async () => {
    const backend = new FakeBackend(status('ready'));
    const openExternal = vi.fn(async () => {});
    const result = await createReaderLauncher({ backend, openExternal, fetch: okFetch }).open();
    expect(result).toEqual({ ok: true, url: 'http://127.0.0.1:3900/reader/', startedBackend: false });
    expect(backend.starts).toBe(0);
    expect(openExternal).toHaveBeenCalledWith('http://127.0.0.1:3900/reader/');
  });

  it('starts an idle backend and opens the browser only once it is ready', async () => {
    const order: string[] = [];
    const backend = new FakeBackend(status('idle'), (fake) => {
      setTimeout(() => fake.set(status('starting')), 5);
      setTimeout(() => {
        order.push('ready');
        fake.set(status('ready'));
      }, 10);
    });
    const openExternal = vi.fn(async () => {
      order.push('open');
    });
    const result = await createReaderLauncher({ backend, openExternal, fetch: okFetch }).open();
    expect(result.ok && result.startedBackend).toBe(true);
    expect(backend.starts).toBe(1);
    expect(order).toEqual(['ready', 'open']);
  });

  it('waits on a launch already in progress instead of starting a second backend', async () => {
    const backend = new FakeBackend(status('starting'));
    const launcher = createReaderLauncher({ backend, openExternal: async () => {}, fetch: okFetch });
    const pending = launcher.open();
    setTimeout(() => backend.set(status('ready')), 5);
    expect((await pending).ok).toBe(true);
    expect(backend.starts).toBe(0);
  });

  it('shares one launch between repeated clicks', async () => {
    const backend = new FakeBackend(status('idle'), (fake) => {
      setTimeout(() => fake.set(status('ready')), 10);
    });
    const openExternal = vi.fn(async () => {});
    const launcher = createReaderLauncher({ backend, openExternal, fetch: okFetch });
    const [a, b, c] = await Promise.all([launcher.open(), launcher.open(), launcher.open()]);
    expect(a).toBe(b);
    expect(b).toBe(c);
    expect(backend.starts).toBe(1);
    expect(openExternal).toHaveBeenCalledTimes(1);
    // A later click after the first finished is a fresh launch (backend reused).
    await launcher.open();
    expect(backend.starts).toBe(1);
    expect(openExternal).toHaveBeenCalledTimes(2);
  });

  it('reports a startup failure and never opens the browser', async () => {
    const backend = new FakeBackend(status('crashed'), (fake) => {
      setTimeout(() => fake.set(status('failed', { message: 'uvicorn exited with code 1' })), 5);
    });
    const openExternal = vi.fn(async () => {});
    const result = await createReaderLauncher({ backend, openExternal, fetch: okFetch }).open();
    expect(result).toMatchObject({
      ok: false,
      reason: 'start_failed',
      detail: 'uvicorn exited with code 1',
    });
    expect(!result.ok && result.message).toContain('uvicorn exited with code 1');
    expect(openExternal).not.toHaveBeenCalled();
  });

  it('recovers once when the backend crashes while the click is waiting', async () => {
    // The app's own backend just died: the supervisor reports `attaching` while
    // it waits for a replacement, then `crashed`, and does not restart by itself.
    const backend = new FakeBackend(status('attaching'), (fake) => {
      setTimeout(() => fake.set(status('ready')), 5);
    });
    setTimeout(() => backend.set(status('crashed', { message: 'exit code 1' })), 5);
    const openExternal = vi.fn(async () => {});
    const result = await createReaderLauncher({ backend, openExternal, fetch: okFetch }).open();
    expect(result).toMatchObject({ ok: true, startedBackend: true });
    expect(backend.starts).toBe(1);
    expect(openExternal).toHaveBeenCalledTimes(1);
  });

  it('never retries a start it made itself', async () => {
    const backend = new FakeBackend(status('idle'), (fake) => {
      setTimeout(() => fake.set(status('crashed', { message: 'exit code 1' })), 5);
    });
    const result = await createReaderLauncher({
      backend,
      openExternal: async () => {},
      fetch: okFetch,
    }).open();
    expect(result).toMatchObject({ ok: false, reason: 'start_failed' });
    expect(backend.starts).toBe(1);
  });

  it('reports an occupied port', async () => {
    const backend = new FakeBackend(status('idle'), (fake) => {
      setTimeout(() => fake.set(status('port_in_use')), 5);
    });
    const openExternal = vi.fn(async () => {});
    const result = await createReaderLauncher({ backend, openExternal, fetch: okFetch }).open();
    // The renderer builds the translated message from `reason` + `port`.
    expect(result).toMatchObject({ ok: false, reason: 'port_in_use', port: 3900 });
    expect(!result.ok && result.message).toContain('3900');
    expect(openExternal).not.toHaveBeenCalled();
  });

  it('does not start anything while first-run setup is required', async () => {
    const backend = new FakeBackend(status('setup_required'));
    const result = await createReaderLauncher({
      backend,
      openExternal: async () => {},
      fetch: okFetch,
    }).open();
    expect(result).toMatchObject({ ok: false, reason: 'setup_required' });
    expect(backend.starts).toBe(0);
  });

  it('refuses a remote backend', async () => {
    const backend = new FakeBackend(status('ready', { remote: true, baseUrl: 'https://gpu.example' }));
    const openExternal = vi.fn(async () => {});
    const result = await createReaderLauncher({ backend, openExternal, fetch: okFetch }).open();
    expect(result).toMatchObject({ ok: false, reason: 'remote' });
    expect(openExternal).not.toHaveBeenCalled();
  });

  it('times out instead of hanging when the backend never becomes ready', async () => {
    const backend = new FakeBackend(status('starting'));
    const result = await createReaderLauncher({
      backend,
      openExternal: async () => {},
      fetch: okFetch,
      readyTimeoutMs: 20,
    }).open();
    expect(result).toMatchObject({ ok: false, reason: 'timeout' });
  });

  it('does not send the user to a 404 when the backend lacks the reader', async () => {
    const backend = new FakeBackend(status('ready'));
    const openExternal = vi.fn(async () => {});
    const result = await createReaderLauncher({
      backend,
      openExternal,
      fetch: vi.fn(async () => new Response('Not Found', { status: 404 })),
    }).open();
    expect(result).toMatchObject({ ok: false, reason: 'reader_missing' });
    expect(openExternal).not.toHaveBeenCalled();
  });

  it('reports a browser that fails to open, with the URL to open by hand', async () => {
    const backend = new FakeBackend(status('ready'));
    const result = await createReaderLauncher({
      backend,
      openExternal: async () => {
        throw new Error('no handler for http');
      },
      fetch: okFetch,
    }).open();
    expect(result).toMatchObject({
      ok: false,
      reason: 'browser_failed',
      url: 'http://127.0.0.1:3900/reader/',
    });
    expect(!result.ok && result.message).toContain('http://127.0.0.1:3900/reader/');
  });

  it('opens the reader on the fallback port the supervisor actually chose', async () => {
    const backend = new FakeBackend(status('ready', { baseUrl: 'http://127.0.0.1:4900', port: 4900 }));
    const openExternal = vi.fn(async () => {});
    await createReaderLauncher({ backend, openExternal, fetch: okFetch }).open();
    expect(openExternal).toHaveBeenCalledWith('http://127.0.0.1:4900/reader/');
  });
});
