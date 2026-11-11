import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { LanguagesIcon, PlusIcon, XIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { useTranslationEngines } from '@/features/settings/translation-settings';
import {
  useEbookTranslate,
  useEbookTranslateStatusOnMount,
  type EbookTranslateChapter,
  type GlossaryEntry,
} from './ebook-translate-session';

const STATUS_TONE: Record<EbookTranslateChapter['status'], string> = {
  pending: 'bg-border',
  done: 'bg-emerald-500',
  failed: 'bg-destructive',
};

/** Ebook Translation: translates the Audiobook project currently open in this
 * workspace, chapter-by-chapter, via the existing backend service
 * (`POST /audiobook/translate`). Deliberately scoped to the open project's id
 * and script — see ebook-translate-session.ts's docstring — so there is
 * nothing here to "pick" or "paste"; a project must be saved (have an id)
 * before it can be translated, same requirement the backend route enforces. */
export function EbookTranslatePanel({
  disabled,
  projectId,
  text,
}: {
  disabled: boolean;
  projectId: string | null;
  text: string;
}) {
  const { t } = useTranslation();
  const { state, start, stop, busy, loadStatus } = useEbookTranslate();
  useEbookTranslateStatusOnMount(projectId, loadStatus);
  const { data: engineData } = useTranslationEngines();
  const [targetLang, setTargetLang] = useState('');
  const [sourceLang, setSourceLang] = useState('');
  const [provider, setProvider] = useState('argos');
  const [showGlossary, setShowGlossary] = useState(false);
  const [glossary, setGlossary] = useState<GlossaryEntry[]>([]);
  const [glossarySource, setGlossarySource] = useState('');
  const [glossaryTarget, setGlossaryTarget] = useState('');
  const [instructions, setInstructions] = useState('');

  const canStart =
    !disabled && !busy && !!projectId && text.trim().length > 0 && targetLang.trim().length > 0;

  const create = () => {
    if (!canStart || !projectId) return;
    void start({
      projectId,
      text,
      targetLang: targetLang.trim(),
      sourceLang: sourceLang.trim() || undefined,
      provider,
      glossary: provider === 'openai' ? glossary : undefined,
      translationInstructions: provider === 'openai' ? instructions.trim() : undefined,
    });
  };

  const addGlossaryEntry = () => {
    const source = glossarySource.trim();
    const target = glossaryTarget.trim();
    if (!source || !target) return;
    setGlossary((entries) => [...entries, { source, target }]);
    setGlossarySource('');
    setGlossaryTarget('');
  };

  const removeGlossaryEntry = (index: number) => {
    setGlossary((entries) => entries.filter((_, i) => i !== index));
  };

  const doneCount = state.chapters.filter((c) => c.status === 'done').length;
  const failedCount = state.chapters.filter((c) => c.status === 'failed').length;

  return (
    <div
      data-slot="ebook-translate-panel"
      className="space-y-3 rounded-xl border border-border/60 bg-muted/20 p-3"
    >
      <h2 className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
        <LanguagesIcon className="size-4" aria-hidden="true" />
        {t('audiobook.ebook_translate.title')}
      </h2>
      <p className="text-xs text-muted-foreground">{t('audiobook.ebook_translate.hint')}</p>
      {!projectId && (
        <p className="text-xs text-muted-foreground" role="status">
          {t('audiobook.ebook_translate.save_project_hint')}
        </p>
      )}
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex flex-col gap-1 text-xs text-muted-foreground">
          {t('audiobook.ebook_translate.target_lang_label')}
          <Input
            value={targetLang}
            disabled={disabled || busy}
            placeholder={t('audiobook.ebook_translate.target_lang_placeholder')}
            onChange={(e) => setTargetLang(e.target.value)}
            className="h-7 w-36"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted-foreground">
          {t('audiobook.ebook_translate.source_lang_label')}
          <Input
            value={sourceLang}
            disabled={disabled || busy}
            placeholder={t('audiobook.ebook_translate.source_lang_placeholder')}
            onChange={(e) => setSourceLang(e.target.value)}
            className="h-7 w-36"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted-foreground">
          {t('audiobook.ebook_translate.provider_label')}
          <Select
            items={(engineData?.engines ?? []).map((engine) => ({
              value: engine.id,
              label: engine.display_name,
            }))}
            value={provider}
            onValueChange={(value) => {
              if (value !== null) setProvider(value);
            }}
            disabled={disabled || busy}
          >
            <SelectTrigger className="h-7 w-44" aria-label={t('audiobook.ebook_translate.provider_label')}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(engineData?.engines ?? []).map((engine) => (
                <SelectItem key={engine.id} value={engine.id}>
                  {engine.display_name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>
        {busy ? (
          <Button size="sm" variant="outline" onClick={stop}>
            {t('audiobook.stop')}
          </Button>
        ) : (
          <Button size="sm" disabled={!canStart} onClick={create}>
            {t('audiobook.ebook_translate.start')}
          </Button>
        )}
      </div>
      {provider === 'openai' && (
        <div className="space-y-2 rounded-lg border border-border/60 bg-background/40 p-2">
          <button
            type="button"
            className="text-xs text-muted-foreground underline"
            onClick={() => setShowGlossary((v) => !v)}
          >
            {t('audiobook.ebook_translate.glossary_toggle')}
          </button>
          {showGlossary && (
            <div className="space-y-2">
              <ul className="space-y-1">
                {glossary.map((entry, i) => (
                  <li
                    key={`${entry.source}-${entry.target}-${i}`}
                    className="flex items-center gap-2 text-xs"
                  >
                    <span className="flex-1 truncate">
                      {entry.source} → {entry.target}
                    </span>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="size-5"
                      disabled={disabled || busy}
                      aria-label={t('audiobook.ebook_translate.glossary_remove')}
                      onClick={() => removeGlossaryEntry(i)}
                    >
                      <XIcon className="size-3.5" />
                    </Button>
                  </li>
                ))}
              </ul>
              <div className="flex items-center gap-2">
                <Input
                  value={glossarySource}
                  disabled={disabled || busy}
                  placeholder={t('audiobook.ebook_translate.glossary_source_placeholder')}
                  onChange={(e) => setGlossarySource(e.target.value)}
                  className="h-7 flex-1"
                />
                <Input
                  value={glossaryTarget}
                  disabled={disabled || busy}
                  placeholder={t('audiobook.ebook_translate.glossary_target_placeholder')}
                  onChange={(e) => setGlossaryTarget(e.target.value)}
                  className="h-7 flex-1"
                />
                <Button
                  size="icon"
                  variant="outline"
                  className="size-7 shrink-0"
                  disabled={disabled || busy || !glossarySource.trim() || !glossaryTarget.trim()}
                  aria-label={t('audiobook.ebook_translate.glossary_add')}
                  onClick={addGlossaryEntry}
                >
                  <PlusIcon className="size-3.5" />
                </Button>
              </div>
              <Textarea
                value={instructions}
                disabled={disabled || busy}
                maxLength={5000}
                placeholder={t('audiobook.ebook_translate.instructions_placeholder')}
                onChange={(e) => setInstructions(e.target.value)}
                className="min-h-16 text-xs"
              />
            </div>
          )}
        </div>
      )}
      {state.phase !== 'idle' && (
        <div className="space-y-2">
          {state.totalChapters > 0 && (
            <p className="text-xs text-muted-foreground">
              {t('audiobook.ebook_translate.progress', {
                done: doneCount,
                total: state.totalChapters,
              })}
            </p>
          )}
          <ul className="space-y-1">
            {state.chapters.map((chapter) => (
              <li
                key={chapter.index}
                className="flex flex-col gap-1 rounded-md bg-background/60 px-2 py-1 text-xs"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="flex items-center gap-2">
                    <span
                      className={cn('size-1.5 shrink-0 rounded-full', STATUS_TONE[chapter.status])}
                      aria-hidden="true"
                    />
                    {chapter.title || t('audiobook.chapter_n', { n: chapter.index + 1 })}
                  </span>
                  <span className="text-muted-foreground">
                    {chapter.status === 'done' &&
                      t(
                        chapter.cached
                          ? 'audiobook.ebook_translate.chapter_cached'
                          : 'audiobook.ebook_translate.chapter_done',
                      )}
                    {chapter.status === 'failed' && t('audiobook.pdf_batch.status_failed')}
                    {chapter.status === 'pending' && t('audiobook.pdf_batch.status_pending')}
                  </span>
                </div>
                {chapter.status === 'failed' && chapter.error && (
                  <p className="break-words text-[11px] text-destructive" role="alert">
                    {chapter.error}
                  </p>
                )}
              </li>
            ))}
          </ul>
          {state.phase === 'done' && state.complete !== null && (
            <p className="text-xs text-muted-foreground" role="status" aria-live="polite">
              {state.complete
                ? t('audiobook.ebook_translate.summary_done', { total: state.totalChapters })
                : t('audiobook.ebook_translate.summary_failures', {
                    failed: failedCount,
                    total: state.totalChapters,
                  })}
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
