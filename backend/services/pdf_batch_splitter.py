"""Split a PDF into ordered, page-aligned batches for incremental audiobook
generation.

Stage 1 of the batch-based parallel PDF-to-audiobook investigation (see the
architecture report) — **splitter only**. Nothing here is wired into a queue,
a worker pool, or the player; it is a pure, additive module that produces
batches a later stage can schedule and render. :func:`pdf_to_chapter_script`
is untouched by this module — existing single-script PDF import keeps its
exact current behaviour.

Reuses the exact PDF-opening/decryption/page-cap validation
:func:`services.longform_import.pdf_to_chapter_script` uses (same
``ValueError`` messages for encrypted/invalid/too-many-page/no-text
documents) and the exact same :func:`services.longform_import.
chapterize_plaintext` heading grammar, so a batch's script is script-for-script
compatible with :func:`services.audiobook.parse_audiobook_script` /
:func:`services.longform_parser.parse_script_to_spans` — a batch can be fed
into the existing render pipeline unchanged once a later stage wires it up.

Unlike ``pdf_to_chapter_script`` (which always joins page text with a blank
line, treating *every* page boundary as a paragraph break — an artifact
that's harmless for a single flattened script but would needlessly split a
sentence across two independently-synthesized batches), batches join
consecutive pages' text with a blank line only when the preceding page's text
actually finished a sentence/paragraph; otherwise pages are joined with a
single space so a sentence that happens to cross a PDF page boundary stays
one sentence, and the batch boundary itself extends past its page-count
target when needed so that sentence isn't split across two batches either.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from services.longform_import import _PDF_MAX_PAGES, chapterize_plaintext

#: Default page count per batch, per the investigation's approved default.
DEFAULT_BATCH_SIZE_PAGES = 10

#: A page's extracted text ending in one of these (a sentence terminator,
#: optionally followed by closing quotes/brackets) is treated as finishing a
#: sentence/paragraph, so the next page is free to start a new batch without
#: splitting one. Anything else is treated as a sentence continuing onto the
#: next page. Deliberately simple (page-granularity, not a sentence
#: tokenizer) — see the module docstring for why that's sufficient here.
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)\]]*\s*$")

#: A trailing line that looks like a running page-number footer/header
#: ("Page 12", "12", "Page 12 of 340", "12 / 340", "- 12 -") rather than
#: prose. Real PDFs commonly print one on every page (pypdf's extraction
#: surfaced exactly this during the investigation's own browser testing —
#: "Page 1"/"Page 2" literally appear in extracted text), and without this a
#: page-number line as the last line of a page would defeat the
#: sentence-boundary check on nearly every page of a real book.
_FOOTER_LINE_RE = re.compile(
    r"^\s*-?\s*(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?\s*-?\s*$", re.IGNORECASE)
#: How many trailing footer-like lines to look past when checking whether a
#: page ends a sentence — bounded so this can't skip past real content.
_MAX_FOOTER_LINES_SKIPPED = 2

#: A page whose text (after stripping footer-shaped lines) is this short is
#: almost certainly a chapter-opener/title page ("Chapter 3", "The
#: Crossing"), not running prose truncated by the page boundary — real
#: truncated prose fills most of the page. Treated as ending a
#: paragraph outright, regardless of terminal punctuation, so a heading page
#: doesn't get glued onto the next page's first sentence with no pause.
_SHORT_PAGE_WORD_THRESHOLD = 8

#: Safety bound on how many pages a batch may grow past its target size while
#: hunting for a page that ends a sentence, before it cuts anyway. Keeps a
#: pathological document (e.g. no terminal punctuation at all) from producing
#: one unbounded batch; a cut here is rare and, by construction, no worse
#: than what every page boundary already does in the non-batch import path.
DEFAULT_MAX_BOUNDARY_EXTENSION_PAGES = 5


@dataclass(frozen=True)
class PdfPage:
    """One page's extracted text, 1-indexed, in document order.

    ``text`` is ``""`` when the page had no extractable text (blank page, or
    a page whose extraction failed) — it is kept, not dropped, so page
    numbering and the page-to-text mapping stay intact across the whole
    document regardless of which pages carried text.
    """

    number: int
    text: str


@dataclass(frozen=True)
class PdfBatch:
    """One page-aligned batch, ready to feed into
    :func:`services.audiobook.parse_audiobook_script` once a later stage
    wires batches into the render pipeline."""

    index: int               # 0-based, in document order
    start_page: int          # 1-indexed, inclusive
    end_page: int            # 1-indexed, inclusive
    script: str              # chapterized text for this batch (may be "")
    pages: tuple[PdfPage, ...]

    @property
    def page_count(self) -> int:
        return self.end_page - self.start_page + 1


def _open_pdf_reader(data: bytes, *, max_pages: int):
    """Open and validate a PDF exactly like ``pdf_to_chapter_script`` does —
    same error messages, same empty-password-decrypt attempt, same page cap —
    so callers of either function see identical, familiar errors."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, OSError, ValueError) as e:
        raise ValueError(f"not a valid PDF file: {e}") from e

    if reader.is_encrypted:
        # Many PDFs are encrypted with an empty user password (owner-locked
        # but freely readable). Try that; a real password we can't supply.
        try:
            if reader.decrypt("") == 0:  # 0 == wrong password
                raise ValueError("PDF is password-protected")
        except (NotImplementedError, PdfReadError) as e:
            raise ValueError(f"can't read this encrypted PDF: {e}") from e

    if len(reader.pages) > max_pages:
        raise ValueError(f"PDF has too many pages (max {max_pages})")
    return reader


def extract_pdf_pages(data: bytes, *, max_pages: int = _PDF_MAX_PAGES) -> list[PdfPage]:
    """Extract per-page text in page order.

    Same ``pypdf`` call and the same one-bad-page-doesn't-kill-the-import
    behaviour as ``pdf_to_chapter_script``, but keeps each page's text
    separate instead of flattening it immediately — the shared building
    block both the single-script import and the batch splitter extract from.
    """
    reader = _open_pdf_reader(data, max_pages=max_pages)
    pages: list[PdfPage] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 — one bad page shouldn't kill the import
            text = ""
        pages.append(PdfPage(number=i, text=text))
    return pages


def _ends_sentence(text: str) -> bool:
    """True when ``text`` looks like it finishes a sentence/paragraph, so the
    next page is free to start a new batch without splitting one.

    Checks the last non-blank line, skipping past up to
    ``_MAX_FOOTER_LINES_SKIPPED`` trailing page-number-footer-shaped lines
    first (see ``_FOOTER_LINE_RE``) so a running footer doesn't masquerade as
    mid-sentence prose.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    skipped = 0
    while lines and skipped < _MAX_FOOTER_LINES_SKIPPED and _FOOTER_LINE_RE.match(lines[-1]):
        lines.pop()
        skipped += 1
    if not lines:
        return False
    if sum(len(ln.split()) for ln in lines) <= _SHORT_PAGE_WORD_THRESHOLD:
        return True
    return bool(_SENTENCE_END_RE.search(lines[-1].rstrip()))


def _join_pages(pages: list[PdfPage]) -> str:
    """Join one batch's page texts, inserting a paragraph break only where
    the preceding non-empty page's text actually ended one — unlike
    ``pdf_to_chapter_script``, which always inserts a blank line at every
    page boundary. Empty pages contribute nothing and don't themselves count
    as breaking or continuing a sentence."""
    parts: list[str] = []
    prev_had_text = False
    prev_ended_sentence = True
    for page in pages:
        text = page.text.strip("\n")
        if not text.strip():
            continue
        if prev_had_text:
            parts.append("\n\n" if prev_ended_sentence else " ")
        parts.append(text)
        prev_had_text = True
        prev_ended_sentence = _ends_sentence(text)
    return "".join(parts)


def split_pdf_into_batches(
    data: bytes,
    *,
    batch_size_pages: int = DEFAULT_BATCH_SIZE_PAGES,
    max_pages: int = _PDF_MAX_PAGES,
    max_boundary_extension_pages: int = DEFAULT_MAX_BOUNDARY_EXTENSION_PAGES,
) -> list[PdfBatch]:
    """Split a PDF into ordered, page-aligned batches of ~``batch_size_pages``
    pages each.

    Raises the same ``ValueError`` (same message) as ``pdf_to_chapter_script``
    for an invalid/corrupt PDF, a real-password-encrypted PDF, a document over
    ``max_pages``, or a document with no extractable text anywhere (a scanned
    / image-only PDF) — callers get identical, familiar errors either way.

    Page order and the page-to-text mapping are always preserved: a batch
    never skips, reorders, or merges non-adjacent pages, and ``PdfBatch.pages``
    carries every original page number in the batch's range even when a page
    had no extractable text.

    A batch boundary normally falls every ``batch_size_pages`` pages. If the
    last non-empty page in that window doesn't look like it finished a
    sentence, the boundary extends one page at a time (up to
    ``max_boundary_extension_pages`` beyond the target) until it lands on a
    page that does, or the document runs out, or the extension cap is hit —
    so a sentence that happens to cross a page boundary isn't unnecessarily
    split across two batches. Each batch's text is chapterized with the exact
    same grammar as the single-script import path
    (:func:`services.longform_import.chapterize_plaintext`), so
    ``services.audiobook.parse_audiobook_script`` /
    ``services.longform_parser.parse_script_to_spans`` can consume any batch
    unchanged.

    An all-blank window (e.g. a run of blank pages) yields a batch with
    ``script == ""`` rather than being silently dropped, so page numbering
    and batch ordering stay complete and predictable for a caller that will
    schedule every batch.
    """
    if batch_size_pages < 1:
        raise ValueError("batch_size_pages must be >= 1")

    pages = extract_pdf_pages(data, max_pages=max_pages)
    if not any(p.text.strip() for p in pages):
        raise ValueError(
            "no extractable text — this looks like a scanned or image-only PDF")

    batches: list[PdfBatch] = []
    n = len(pages)
    i = 0  # 0-based index into `pages`, start of the next batch
    while i < n:
        end = min(i + batch_size_pages, n)  # exclusive
        extended = 0
        while end < n and extended < max_boundary_extension_pages:
            last_nonempty = next(
                (p for p in reversed(pages[i:end]) if p.text.strip()), None)
            if last_nonempty is None or _ends_sentence(last_nonempty.text):
                break
            end += 1
            extended += 1
        window = pages[i:end]
        batches.append(PdfBatch(
            index=len(batches),
            start_page=window[0].number,
            end_page=window[-1].number,
            script=chapterize_plaintext(_join_pages(window)),
            pages=tuple(window),
        ))
        i = end

    return batches
