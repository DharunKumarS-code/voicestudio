import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FileTextIcon, RotateCcwIcon } from 'lucide-react';
import { Button, buttonVariants } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { PlayInDockButton } from '@/components/now-playing-dock';
import { apiPath } from '@/lib/api/client';
import { cn } from '@/lib/utils';
import {
  fetchResumablePdfBatchJobs,
  usePdfBatchImport,
  type PdfBatchItem,
  type PdfBatchRenderOptions,
  type ResumablePdfBatchJob,
} from './pdf-batch-session';

const STATUS_KEY: Record<PdfBatchItem['status'], string> = {
  pending: 'audiobook.pdf_batch.status_pending',
  running: 'audiobook.pdf_batch.status_running',
  done: 'audiobook.pdf_batch.status_done',
  failed: 'audiobook.pdf_batch.status_failed',
  empty: 'audiobook.pdf_batch.status_empty',
};

/** Large-PDF audiobook import: splits the PDF into page batches on the
 * backend and renders each one, in order, as its own independent output
 * file — for books too large to comfortably render as one job. Deliberately
 * separate from the main script editor/generate flow (see
 * pdf-batch-session.ts's docstring). */
export function PdfBatchPanel({
  disabled,
  renderOptions,
}: {
  disabled: boolean;
  renderOptions: PdfBatchRenderOptions;
}) {
  const { t } = useTranslation();
  const { state, start, resume, stop, busy } = usePdfBatchImport();
  const [file, setFile] = useState<File | null>(null);
  const [batchSizePages, setBatchSizePages] = useState(10);
  const fileInput = useRef<HTMLInputElement>(null);
  const [resumableJobs, setResumableJobs] = useState<ResumablePdfBatchJob[]>([]);

  useEffect(() => {
    let cancelled = false;
    fetchResumablePdfBatchJobs()
      .then((jobs) => {
        if (!cancelled) setResumableJobs(jobs);
      })
      .catch(() => {
        // Best-effort affordance — an unreachable backend here is surfaced
        // elsewhere (EngineNotice); nothing to show for this banner.
      });
    return () => {
      cancelled = true;
    };
    // Re-check once a run settles, so a job that just finished (or whose
    // resume just completed) drops off the list without a manual reload.
  }, [state.phase]);

  const create = () => {
    if (!file) return;
    void start(file, renderOptions, batchSizePages);
  };

  const resumeJob = (jobId: string) => {
    setResumableJobs((jobs) => jobs.filter((j) => j.jobId !== jobId));
    void resume(jobId);
  };

  return (
    <div
      data-slot="pdf-batch-panel"
      className="space-y-3 rounded-xl border border-border/60 bg-muted/20 p-3"
    >
      <h2 className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
        <FileTextIcon className="size-4" aria-hidden="true" />
        {t('audiobook.pdf_batch.title')}
      </h2>
      <p className="text-xs text-muted-foreground">{t('audiobook.pdf_batch.hint')}</p>
      {state.phase === 'idle' &&
        resumableJobs.map((job) => (
          <div
            key={job.jobId}
            className="flex items-center justify-between gap-2 rounded-lg border border-border/60 bg-background/60 px-2.5 py-2 text-xs"
          >
            <span className="flex min-w-0 items-center gap-1.5">
              <RotateCcwIcon className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
              <span className="truncate">
                {t('audiobook.pdf_batch.resumable_hint', {
                  title: job.title || t('audiobook.pdf_batch.title'),
                  completed: job.completedBatches.length,
                  total: job.totalBatches,
                })}
              </span>
            </span>
            <Button
              size="sm"
              variant="outline"
              disabled={disabled || busy}
              onClick={() => resumeJob(job.jobId)}
            >
              {t('audiobook.resume')}
            </Button>
          </div>
        ))}
      <div className="flex flex-wrap items-center gap-2">
        <label className={cn(buttonVariants({ variant: 'outline', size: 'sm' }))}>
          {file ? file.name : t('audiobook.pdf_batch.choose_file')}
          <input
            ref={fileInput}
            type="file"
            accept=".pdf"
            disabled={disabled || busy}
            className="sr-only"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
        <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
          {t('audiobook.pdf_batch.batch_size_label')}
          <Input
            type="number"
            min={1}
            max={200}
            value={batchSizePages}
            disabled={disabled || busy}
            onChange={(e) => setBatchSizePages(Math.max(1, Number(e.target.value) || 1))}
            className="h-7 w-16"
          />
        </label>
        {busy ? (
          <Button size="sm" variant="outline" onClick={stop}>
            {t('audiobook.stop')}
          </Button>
        ) : (
          <Button size="sm" disabled={disabled || !file} onClick={create}>
            {t('audiobook.pdf_batch.create')}
          </Button>
        )}
      </div>
      {state.phase !== 'idle' && (
        <div className="space-y-2">
          {state.recovery && (
            <p className="text-xs text-muted-foreground" role="status" aria-live="polite">
              {t('audiobook.pdf_batch.recovery_hint', {
                verified: state.recovery.verified,
                invalidated: state.recovery.invalidated,
              })}
            </p>
          )}
          {state.totalBatches > 0 && (
            <p className="text-xs text-muted-foreground">
              {t('audiobook.pdf_batch.progress', {
                current: Math.min((state.currentBatch ?? 0) + 1, state.totalBatches),
                total: state.totalBatches,
              })}
            </p>
          )}
          <ul className="space-y-1">
            {state.items.map((item) => {
              const label =
                item.startPage != null && item.endPage != null
                  ? t('audiobook.pdf_batch.pages_label', {
                      start: item.startPage,
                      end: item.endPage,
                    })
                  : t('audiobook.chapter_n', { n: item.index + 1 });
              return (
                <li
                  key={item.index}
                  className="flex flex-col gap-1 rounded-md bg-background/60 px-2 py-1 text-xs"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="flex items-center gap-2">
                      <span
                        className={cn(
                          'size-1.5 shrink-0 rounded-full',
                          item.status === 'done' && 'bg-emerald-500',
                          item.status === 'failed' && 'bg-destructive',
                          item.status === 'running' && 'bg-amber-500',
                          item.status === 'empty' && 'bg-muted-foreground',
                          item.status === 'pending' && 'bg-border',
                        )}
                        aria-hidden="true"
                      />
                      {label}
                    </span>
                    <span className="flex items-center gap-2 text-muted-foreground">
                      {t(STATUS_KEY[item.status])}
                      {item.status === 'done' && item.output && (
                        <>
                          <PlayInDockButton
                            src={apiPath('/audio/' + encodeURIComponent(item.output))}
                            source={`pdf-batch-${state.jobId ?? ''}-${item.index}`}
                            title={label}
                            kind="audiobook"
                            duration={item.durationS ?? null}
                          />
                          <a
                            className="underline"
                            href={apiPath('/audio/' + encodeURIComponent(item.output))}
                            target="_blank"
                            rel="noreferrer"
                          >
                            {t('audiobook.download')}
                          </a>
                        </>
                      )}
                    </span>
                  </div>
                  {item.status === 'failed' && item.error && (
                    <p className="break-words text-[11px] text-destructive" role="alert">
                      {item.error}
                    </p>
                  )}
                </li>
              );
            })}
          </ul>
          {state.overallStatus && (
            <p className="text-xs text-muted-foreground" role="status" aria-live="polite">
              {state.overallStatus === 'done' || state.overallStatus === 'done_with_failures'
                ? t('audiobook.pdf_batch.summary_done', { total: state.totalBatches })
                : null}
              {state.overallStatus === 'failed' || state.overallStatus === 'done_with_failures'
                ? t('audiobook.pdf_batch.summary_failures', {
                    failed: state.items.filter((i) => i.status === 'failed').length,
                    total: state.totalBatches,
                  })
                : null}
            </p>
          )}
          {state.error && (
            <p role="alert" className="text-xs text-destructive">
              {state.error}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
