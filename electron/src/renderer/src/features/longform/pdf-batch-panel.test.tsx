import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi, type Mock } from 'vitest';
import { PdfBatchPanel } from './pdf-batch-panel';
import { usePdfBatchImport, fetchResumablePdfBatchJobs, type PdfBatchState } from './pdf-batch-session';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, opts?: Record<string, unknown>) =>
      opts ? `${key}:${JSON.stringify(opts)}` : key,
  }),
}));
vi.mock('@/lib/api/client', () => ({ apiPath: (p: string) => p }));
vi.mock('@/components/now-playing-dock', () => ({
  PlayInDockButton: ({
    src,
    source,
    title,
    kind,
    duration,
  }: {
    src: string;
    source: string;
    title: string;
    kind: string;
    duration?: number | null;
  }) => (
    <button
      type="button"
      data-testid="play-button"
      data-src={src}
      data-source={source}
      data-title={title}
      data-kind={kind}
      data-duration={duration ?? ''}
    >
      play
    </button>
  ),
}));
vi.mock('./pdf-batch-session', () => ({
  usePdfBatchImport: vi.fn(),
  fetchResumablePdfBatchJobs: vi.fn(),
}));

const baseState: PdfBatchState = {
  phase: 'running',
  jobId: 'job-abc',
  totalBatches: 1,
  currentBatch: null,
  items: [],
  overallStatus: null,
  error: null,
  recovery: null,
};

function setup(state: PdfBatchState) {
  (usePdfBatchImport as unknown as Mock).mockReturnValue({
    state,
    start: vi.fn(),
    resume: vi.fn(),
    stop: vi.fn(),
    reset: vi.fn(),
    busy: false,
  });
  (fetchResumablePdfBatchJobs as unknown as Mock).mockResolvedValue([]);
  return render(<PdfBatchPanel disabled={false} renderOptions={{}} />);
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it('shows Play and Download for a completed batch with an output file', () => {
  setup({
    ...baseState,
    items: [{ index: 0, startPage: 1, endPage: 10, status: 'done', output: 'audiobook_x_b0.m4b', durationS: 12.5 }],
  });
  const play = screen.getByTestId('play-button');
  expect(play).toHaveAttribute('data-src', '/audio/audiobook_x_b0.m4b');
  expect(play).toHaveAttribute('data-source', 'pdf-batch-job-abc-0');
  expect(play).toHaveAttribute('data-kind', 'audiobook');
  expect(play).toHaveAttribute('data-duration', '12.5');
  expect(screen.getByText('audiobook.download')).toBeInTheDocument();
});

it('hides Play for a pending batch', () => {
  setup({ ...baseState, items: [{ index: 0, startPage: null, endPage: null, status: 'pending' }] });
  expect(screen.queryByTestId('play-button')).toBeNull();
  expect(screen.queryByText('audiobook.download')).toBeNull();
});

it('hides Play for a running batch', () => {
  setup({ ...baseState, items: [{ index: 0, startPage: 1, endPage: 10, status: 'running' }] });
  expect(screen.queryByTestId('play-button')).toBeNull();
});

it('hides Play for a failed batch', () => {
  setup({ ...baseState, items: [{ index: 0, startPage: 1, endPage: 10, status: 'failed', error: 'engine crashed' }] });
  expect(screen.queryByTestId('play-button')).toBeNull();
});

// ── Stage 4 bug fix: a failed batch's detailed error is now displayed ────

it('displays the detailed error message for a failed batch', () => {
  setup({
    ...baseState,
    items: [{ index: 0, startPage: 1, endPage: 10, status: 'failed', error: 'ffmpeg assembly failed (exit code 1)' }],
  });
  expect(screen.getByRole('alert')).toHaveTextContent('ffmpeg assembly failed (exit code 1)');
  expect(screen.getByText('audiobook.pdf_batch.status_failed')).toBeInTheDocument();
});

it('does not render an error message for a failed batch with no error (defensive)', () => {
  setup({ ...baseState, items: [{ index: 0, startPage: 1, endPage: 10, status: 'failed' }] });
  expect(screen.queryByRole('alert')).toBeNull();
  expect(screen.getByText('audiobook.pdf_batch.status_failed')).toBeInTheDocument();
});

it('hides Play for an empty batch', () => {
  setup({ ...baseState, items: [{ index: 0, startPage: 1, endPage: 1, status: 'empty' }] });
  expect(screen.queryByTestId('play-button')).toBeNull();
});

it('hides Play when a done batch is missing its output (defensive)', () => {
  setup({ ...baseState, items: [{ index: 0, startPage: 1, endPage: 10, status: 'done' }] });
  expect(screen.queryByTestId('play-button')).toBeNull();
  expect(screen.queryByText('audiobook.download')).toBeNull();
});

it('gives each completed batch a stable, distinct playback source', () => {
  setup({
    ...baseState,
    totalBatches: 2,
    items: [
      { index: 0, startPage: 1, endPage: 10, status: 'done', output: 'a.m4b' },
      { index: 1, startPage: 11, endPage: 20, status: 'done', output: 'b.m4b' },
    ],
  });
  const buttons = screen.getAllByTestId('play-button');
  expect(buttons).toHaveLength(2);
  expect(buttons[0]).toHaveAttribute('data-source', 'pdf-batch-job-abc-0');
  expect(buttons[1]).toHaveAttribute('data-source', 'pdf-batch-job-abc-1');
  expect(buttons[0]).toHaveAttribute('data-src', '/audio/a.m4b');
  expect(buttons[1]).toHaveAttribute('data-src', '/audio/b.m4b');
});

it('a failed batch does not disturb a sibling done batch\'s Play/Download, or pending/running rows', () => {
  setup({
    ...baseState,
    totalBatches: 4,
    items: [
      { index: 0, startPage: 1, endPage: 10, status: 'done', output: 'a.m4b' },
      { index: 1, startPage: 11, endPage: 20, status: 'failed', error: 'ffmpeg assembly failed (exit code 1)' },
      { index: 2, startPage: 21, endPage: 30, status: 'pending' },
      { index: 3, startPage: 31, endPage: 40, status: 'running' },
    ],
  });
  // The completed sibling keeps its Play button and Download link untouched.
  const buttons = screen.getAllByTestId('play-button');
  expect(buttons).toHaveLength(1);
  expect(buttons[0]).toHaveAttribute('data-src', '/audio/a.m4b');
  expect(screen.getByText('audiobook.download')).toBeInTheDocument();
  // The failed batch shows the generic status label AND its detailed error.
  expect(screen.getByText('audiobook.pdf_batch.status_failed')).toBeInTheDocument();
  expect(screen.getByRole('alert')).toHaveTextContent('ffmpeg assembly failed (exit code 1)');
  // Pending/running siblings are unaffected.
  expect(screen.getByText('audiobook.pdf_batch.status_pending')).toBeInTheDocument();
  expect(screen.getByText('audiobook.pdf_batch.status_running')).toBeInTheDocument();
});

it('shows Play for a batch restored as done after a resume (restored_outputs)', () => {
  setup({
    ...baseState,
    items: [
      { index: 0, startPage: null, endPage: null, status: 'done', output: 'audiobook_x_b0.m4b', chapters: 2, durationS: 12.5 },
    ],
  });
  const play = screen.getByTestId('play-button');
  expect(play).toHaveAttribute('data-src', '/audio/audiobook_x_b0.m4b');
});
