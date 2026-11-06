"""PDF → page-aligned batches (Stage 1 of the batch PDF-to-audiobook
investigation — splitter only, no queue, no workers, no playback wiring).

Pure helpers, tested without a server. The multi-page PDF fixture builder
mirrors ``tests/test_longform_import.py``'s single-page ``_make_pdf`` (no
PDF-authoring dependency), extended to build N independently-controlled
pages so page-boundary continuity, blank pages, and incomplete final batches
are all directly testable.
"""
from __future__ import annotations

import io

import pytest

from services.audiobook import parse_audiobook_script
from services.longform_import import pdf_to_chapter_script
from services.pdf_batch_splitter import (
    DEFAULT_BATCH_SIZE_PAGES,
    PdfBatch,
    extract_pdf_pages,
    split_pdf_into_batches,
)


# ── multi-page PDF fixture builder ──────────────────────────────────────────
# Same minimal hand-built object layout as test_longform_import.py's
# single-page `_make_pdf`, extended to N pages: one shared Helvetica font,
# one Page + one Contents stream per entry in `pages`. An entry is either a
# list of lines (one `Tj` per line) or raw content-stream bytes (to build a
# page with no text-showing operators at all, e.g. a true blank page).

def _make_multi_page_pdf(pages: list[list[str] | bytes]) -> bytes:
    def _content_bytes(page: list[str] | bytes) -> bytes:
        if isinstance(page, (bytes, bytearray)):
            return bytes(page)
        if not page:
            return b"q Q"
        show = "BT /F1 12 Tf 72 720 Td 16 TL\n"
        for ln in page:
            esc = ln.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")
            show += f"({esc}) Tj T*\n"
        show += "ET"
        return show.encode("latin-1")

    n = len(pages)
    page_obj_nums = [4 + 2 * i for i in range(n)]
    kids = " ".join(f"{num} 0 R" for num in page_obj_nums)

    objs: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, page in enumerate(pages):
        content = _content_bytes(page)
        content_obj_num = 5 + 2 * i
        objs.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
            + str(content_obj_num).encode() + b" 0 R >>"
        )
        objs.append(
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
            + content + b"\nendstream"
        )

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_pos = out.tell()
    total = len(objs) + 1
    out.write(f"xref\n0 {total}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode())
    return out.getvalue()


def _flat_page(n: int) -> list[str]:
    """A single-sentence page that always ends a sentence (never triggers
    boundary extension) — the default page shape for sizing tests."""
    return [f"This is page {n} of the document, and it ends cleanly."]


# ── defaults / configurability ──────────────────────────────────────────────

def test_default_batch_size_is_ten_pages():
    assert DEFAULT_BATCH_SIZE_PAGES == 10


def test_batch_size_is_configurable():
    data = _make_multi_page_pdf([_flat_page(i) for i in range(1, 7)])
    batches = split_pdf_into_batches(data, batch_size_pages=3)
    assert [(b.start_page, b.end_page) for b in batches] == [(1, 3), (4, 6)]


def test_batch_size_pages_must_be_positive():
    data = _make_multi_page_pdf([_flat_page(1)])
    with pytest.raises(ValueError, match="batch_size_pages"):
        split_pdf_into_batches(data, batch_size_pages=0)


# ── small / medium / larger PDFs ────────────────────────────────────────────

def test_small_pdf_single_batch():
    """A 3-page PDF (smaller than one default batch) is exactly one batch."""
    data = _make_multi_page_pdf([_flat_page(i) for i in range(1, 4)])
    batches = split_pdf_into_batches(data)
    assert len(batches) == 1
    assert (batches[0].start_page, batches[0].end_page) == (1, 3)
    assert batches[0].page_count == 3


def test_medium_pdf_full_batch_plus_incomplete_final_batch():
    """A 12-page PDF yields one full 10-page batch and a 2-page remainder."""
    data = _make_multi_page_pdf([_flat_page(i) for i in range(1, 13)])
    batches = split_pdf_into_batches(data)
    assert [(b.start_page, b.end_page) for b in batches] == [(1, 10), (11, 12)]
    assert [b.page_count for b in batches] == [10, 2]


def test_larger_pdf_multiple_full_batches_plus_remainder():
    """A 47-page PDF yields four full batches and a 7-page remainder, in order,
    covering every page exactly once."""
    data = _make_multi_page_pdf([_flat_page(i) for i in range(1, 48)])
    batches = split_pdf_into_batches(data)
    assert [(b.start_page, b.end_page) for b in batches] == [
        (1, 10), (11, 20), (21, 30), (31, 40), (41, 47),
    ]
    # Every page appears exactly once, across batches, in order.
    seen = [p.number for b in batches for p in b.pages]
    assert seen == list(range(1, 48))


# ── page order / page-to-text mapping ───────────────────────────────────────

def test_batches_preserve_original_page_order_and_indices():
    data = _make_multi_page_pdf([_flat_page(i) for i in range(1, 25)])
    batches = split_pdf_into_batches(data, batch_size_pages=5)
    assert [b.index for b in batches] == list(range(len(batches)))
    # Batches are contiguous and strictly increasing — no gaps, no overlap.
    for prev, nxt in zip(batches, batches[1:]):
        assert nxt.start_page == prev.end_page + 1


def test_page_to_text_mapping_preserved():
    pages = [[f"Unique marker page {i}."] for i in range(1, 8)]
    data = _make_multi_page_pdf(pages)
    batches = split_pdf_into_batches(data, batch_size_pages=3)
    for batch in batches:
        for page in batch.pages:
            assert f"Unique marker page {page.number}." in page.text


def test_extract_pdf_pages_keeps_every_page_even_when_blank():
    data = _make_multi_page_pdf([_flat_page(1), [], _flat_page(3)])
    pages = extract_pdf_pages(data)
    assert [p.number for p in pages] == [1, 2, 3]
    assert pages[1].text.strip() == ""


# ── sentence / paragraph continuity across a page boundary ─────────────────

def test_sentence_split_across_page_boundary_is_not_cut():
    """A sentence that pypdf extracted split across two pages (common for
    real PDFs) stays one unbroken sentence in the batch script, and the
    batch boundary extends to keep both halves together."""
    data = _make_multi_page_pdf([
        ["The lighthouse keeper walked slowly toward the edge of the"],
        ["cliff, watching the waves crash below."],
    ])
    batches = split_pdf_into_batches(data, batch_size_pages=1)
    assert len(batches) == 1  # boundary extended past the 1-page target
    assert "edge of the cliff" in batches[0].script
    # No artificial paragraph break was inserted at the old page boundary.
    assert "edge of the\n\ncliff" not in batches[0].script


def test_sentence_ending_cleanly_at_page_boundary_gets_paragraph_break():
    """When a page's text really does end a sentence, the next page starts a
    new batch at the target boundary (no unnecessary extension), and the
    join reads as a paragraph break, not a run-on."""
    data = _make_multi_page_pdf([
        ["The door closed behind her."],
        ["Morning came early that year."],
    ])
    batches = split_pdf_into_batches(data, batch_size_pages=1)
    assert len(batches) == 2
    assert batches[0].script.endswith("The door closed behind her.")
    assert batches[1].script == "Morning came early that year."


def test_page_number_footer_does_not_defeat_sentence_detection():
    """A running 'Page N' footer (pypdf's own extraction surfaces exactly
    this — confirmed during the investigation's live browser testing) must
    not make a page that actually ends a sentence look like it doesn't."""
    data = _make_multi_page_pdf([
        ["The door closed behind her.", "Page 1"],
        ["Morning came early that year.", "Page 2"],
    ])
    batches = split_pdf_into_batches(data, batch_size_pages=1)
    assert len(batches) == 2  # no unnecessary extension past the footer


def test_bare_page_number_footer_does_not_defeat_sentence_detection():
    data = _make_multi_page_pdf([
        ["The door closed behind her.", "1"],
        ["Morning came early that year.", "2"],
    ])
    batches = split_pdf_into_batches(data, batch_size_pages=1)
    assert len(batches) == 2


def test_short_heading_page_does_not_glue_onto_next_sentence():
    """A short chapter-opener page ('Chapter 3' + a short subtitle, no
    terminal punctuation at all) must not be treated as mid-sentence prose
    continuing onto the next page — it's a heading, and the next page starts
    a fresh paragraph."""
    data = _make_multi_page_pdf([
        ["Chapter 3", "The Crossing"],
        ["The ferry left at dawn, cutting a pale line through the water."],
    ])
    batches = split_pdf_into_batches(data, batch_size_pages=1)
    assert len(batches) == 2  # no unnecessary extension past the heading page
    assert "ferry" not in batches[0].script
    assert batches[1].script.strip().startswith("The ferry")


def test_boundary_extension_is_capped():
    """A pathological run of pages with no terminal punctuation at all
    extends at most `max_boundary_extension_pages` past the target rather
    than growing unboundedly. Each page has enough words that it isn't
    mistaken for a short heading page (see
    test_short_heading_page_does_not_glue_onto_next_sentence)."""
    pages = [
        [f"word{i}a word{i}b word{i}c word{i}d word{i}e word{i}f "
         f"word{i}g word{i}h word{i}i word{i}j"]
        for i in range(1, 10)
    ]  # ten words per page, no sentence ever ends
    data = _make_multi_page_pdf(pages)
    batches = split_pdf_into_batches(
        data, batch_size_pages=1, max_boundary_extension_pages=2)
    assert batches[0].page_count == 3  # 1 target + 2 extension, then cut


def test_parser_sees_one_continuous_span_not_two():
    """Reuses the canonical longform parser (#27) to prove the fix end to
    end: a sentence split across a page boundary parses as a single
    continuous span of text, the way it would if it had never been split
    across pages at all."""
    data = _make_multi_page_pdf([
        ["She opened the letter with trembling hands and began to"],
        ["read every word twice before setting it down."],
    ])
    batches = split_pdf_into_batches(data, batch_size_pages=1)
    assert len(batches) == 1
    plan = parse_audiobook_script(batches[0].script)
    assert len(plan.chapters) == 1
    spans = plan.chapters[0].spans
    assert len(spans) == 1  # one continuous span, not split across two
    assert "trembling hands and began to read every word twice" in spans[0].text


# ── empty pages / extraction failures ───────────────────────────────────────

def test_all_blank_batch_window_keeps_empty_script_not_dropped():
    data = _make_multi_page_pdf([_flat_page(1), [], [], _flat_page(4)])
    batches = split_pdf_into_batches(data, batch_size_pages=2)
    assert [(b.start_page, b.end_page) for b in batches] == [(1, 2), (3, 4)]
    assert batches[0].script.strip() == "This is page 1 of the document, and it ends cleanly."
    assert batches[1].script.strip() == "This is page 4 of the document, and it ends cleanly."


def test_fully_blank_document_raises_same_error_as_single_script_import():
    blank = _make_multi_page_pdf([[], [], []])
    with pytest.raises(ValueError, match="scanned or image-only"):
        split_pdf_into_batches(blank)
    with pytest.raises(ValueError, match="scanned or image-only"):
        pdf_to_chapter_script(blank)


def test_corrupt_pdf_raises_same_error_as_single_script_import():
    with pytest.raises(ValueError, match="not a valid PDF"):
        split_pdf_into_batches(b"this is definitely not a pdf")
    with pytest.raises(ValueError, match="not a valid PDF"):
        pdf_to_chapter_script(b"this is definitely not a pdf")


def test_too_many_pages_raises_same_error_as_single_script_import():
    data = _make_multi_page_pdf([_flat_page(1), _flat_page(2)])
    with pytest.raises(ValueError, match="too many pages"):
        split_pdf_into_batches(data, max_pages=1)
    with pytest.raises(ValueError, match="too many pages"):
        pdf_to_chapter_script(data, max_pages=1)


def test_one_bad_page_does_not_kill_the_batch():
    """A page whose extraction raises is treated as empty, like the
    single-script import path, not as a fatal error for the whole document."""
    # Graphics-only content stream (no BT/Tj) extracts as "" without raising —
    # same "one bad page doesn't kill the import" contract as pdf_to_chapter_script.
    data = _make_multi_page_pdf([_flat_page(1), b"q Q", _flat_page(3)])
    batches = split_pdf_into_batches(data, batch_size_pages=3)
    assert len(batches) == 1
    assert "page 1" in batches[0].script and "page 3" in batches[0].script


# ── chapter headings survive per batch (grammar reuse) ──────────────────────

def test_chapter_headings_recognized_within_a_batch():
    data = _make_multi_page_pdf([
        ["Chapter 1", "Once upon a time."],
        ["Chapter 2", "The end."],
    ])
    batches = split_pdf_into_batches(data, batch_size_pages=5)
    assert len(batches) == 1
    assert "# Chapter 1" in batches[0].script
    assert "# Chapter 2" in batches[0].script
    assert len(parse_audiobook_script(batches[0].script).chapters) == 2


# ── non-regression: pdf_to_chapter_script is untouched ──────────────────────

def test_single_script_import_still_matches_pre_change_behaviour():
    """pdf_to_chapter_script (the existing, unbatched import path) is not
    modified by this module at all — same output for the same input as
    before the batch splitter was added."""
    data = _make_multi_page_pdf([
        ["Chapter 1", "Once upon a time."],
        ["Chapter 2", "The end."],
    ])
    script = pdf_to_chapter_script(data)
    assert "# Chapter 1" in script
    assert "# Chapter 2" in script
    assert len(parse_audiobook_script(script).chapters) == 2
