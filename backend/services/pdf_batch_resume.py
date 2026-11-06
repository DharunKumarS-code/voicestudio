"""Durable resume for PDF-batch audiobook jobs (Stage 3).

A PDF-batch job's parent row lives in ``core.job_store``'s ``jobs`` table
(created/updated by ``api/routers/audiobook.py``, unchanged since Stage 2).
That table is NOT the durable resume truth: ``job_store.sweep_orphans_on_startup``
unconditionally flips every ``pending``/``running`` row to ``failed`` on every
backend boot — by design, the same fact :mod:`services.longform_resume`'s own
docstring states for the single-script render path ("that table is a UI
history convenience only"). A crash-interrupted batch job needs its own
durable record that survives that sweep, exactly the role ``resume.json``
plays for a single-script render.

This module mirrors :mod:`services.longform_resume`'s contract (manifest
written on start, updated on every transition, cleared on full success,
discovered via ``scan_resumable``) for the one job type that module doesn't
cover: PDF-batch parents, whose resumable unit is an ordered batch array +
the original uploaded PDF (re-split deterministically on resume), not a
chapter/span plan. Kept as a sibling module rather than widening
``longform_resume``'s manifest shape — its ``read_manifest`` requires a
``"plan"`` list, a shape assumption this job type doesn't share, and Stage 1/2
must stay untouched. The path-safety shape (``work_dir``/``_SAFE_ID_RE``/
``_SAFE_SEG_RE``) and the atomic-write primitives (``core.durable_io``) are
the same ones that module already uses.

Work dirs use the ``pdfbatch_<job_id>`` prefix. An earlier version of this
module used ``audiobook_batch_<job_id>`` — which, since it starts with
"audiobook_", was silently also matched by ``longform_resume.scan_resumable``'s
own prefix scan for job_type "audiobook", surfacing a spurious, garbled entry
in the single-script resume list. ``pdfbatch_`` is disjoint from
"audiobook_"/"story_" in both directions. The old prefix stays read-only
supported (``legacy_work_dir``/``_existing_work_dir``) so a job interrupted
before this fix remains discoverable and resumable without migration.

Pure file/JSON I/O except for the OS calls themselves — no torch, no model,
no database — so it is unit-testable without a server.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from typing import Optional

from core.durable_io import flush_dir, flush_fd

MANIFEST_VERSION = 1
_MANIFEST_NAME = "resume.json"
_SOURCE_NAME = "source.pdf"
_JOB_TYPE = "pdfbatch"
# Pre-fix Stage 3 work dirs used this prefix. "audiobook_batch_<id>" starts
# with "audiobook_" — services.longform_resume's own scanner
# (RESUMABLE_TYPES = ("audiobook", "story"), matched by `name.startswith(f"{jt}_")`)
# matches it too, misreading a PDF-batch job as a single-script "audiobook"
# job with a garbled id ("batch_<realid>") and surfacing a spurious, broken
# "Resume an audiobook that didn't finish" entry in GET /audiobook/jobs
# (confirmed in the Stage 3 review — no crash, since longform_resume's own
# manifest-shape check rejects this module's manifest, but real UI noise).
# "pdfbatch_" is a prefix disjoint from "audiobook_"/"story_" in both
# directions. Kept read-compatible with the old prefix (never written fresh)
# so a job interrupted before this fix stays discoverable and resumable —
# see _legacy_work_dir / _existing_work_dir.
_LEGACY_JOB_TYPE = "audiobook_batch"
# Mirrors longform_resume's token/segment shape exactly (see that module for
# the CodeQL py/path-injection rationale) — a job id is a server-generated
# token; the only request-supplied id (the resume URL path param) is used
# solely to *select* a trusted, os.listdir-enumerated entry, never to build
# a path directly.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SAFE_SEG_RE = re.compile(rf"^{_JOB_TYPE}_[A-Za-z0-9_-]{{1,64}}$")
_LEGACY_SAFE_SEG_RE = re.compile(rf"^{_LEGACY_JOB_TYPE}_[A-Za-z0-9_-]{{1,64}}$")


def _resolve_work_dir(job_type: str, seg_re: re.Pattern, job_id: str) -> Optional[str]:
    if not _SAFE_ID_RE.match(job_id or ""):
        return None
    seg = os.path.basename(f"{job_type}_{job_id}")
    if not seg_re.match(seg):
        return None
    from core.config import OUTPUTS_DIR
    root = os.path.realpath(OUTPUTS_DIR)
    path = os.path.realpath(os.path.join(root, seg))
    try:
        if os.path.commonpath([path, root]) != root:
            return None
    except ValueError:  # mixed drives on Windows
        return None
    return path


def work_dir(job_id: str) -> Optional[str]:
    """``OUTPUTS_DIR/pdfbatch_<job_id>``, resolved strictly inside
    OUTPUTS_DIR. Returns None for an unsafe id or any path that would escape
    OUTPUTS_DIR — a crafted job_id can never reach a foreign path. The
    current (post-fix) prefix: every NEW job writes here."""
    return _resolve_work_dir(_JOB_TYPE, _SAFE_SEG_RE, job_id)


def legacy_work_dir(job_id: str) -> Optional[str]:
    """``OUTPUTS_DIR/audiobook_batch_<job_id>`` — the pre-fix prefix. Never
    used for a new job; only so a job interrupted before the prefix-collision
    fix remains readable. See the module-level comment on ``_LEGACY_JOB_TYPE``."""
    return _resolve_work_dir(_LEGACY_JOB_TYPE, _LEGACY_SAFE_SEG_RE, job_id)


def _existing_work_dir(job_id: str) -> Optional[str]:
    """The work dir to use for an ALREADY-STARTED job: the current-prefix
    dir if one exists, else the legacy-prefix dir if THAT exists, else None.
    A job already under the legacy prefix keeps using it for the rest of its
    lifecycle (no migration/rename — simpler and lower-risk than moving a
    live work dir); a brand-new job always lands under the current prefix
    because neither directory exists yet, so callers fall back to
    :func:`work_dir` in that case."""
    d = work_dir(job_id)
    if d and os.path.isdir(d):
        return d
    d = legacy_work_dir(job_id)
    if d and os.path.isdir(d):
        return d
    return None


def manifest_path(job_id: str) -> Optional[str]:
    d = _existing_work_dir(job_id) or work_dir(job_id)
    return os.path.join(d, _MANIFEST_NAME) if d else None


def source_pdf_path(job_id: str) -> Optional[str]:
    d = _existing_work_dir(job_id) or work_dir(job_id)
    return os.path.join(d, _SOURCE_NAME) if d else None


def build_manifest(
    *,
    job_id: str,
    title: str = "",
    source_filename: str = "",
    page_count: int,
    batch_size_pages: int,
    params: dict,
    batches: list[dict],
) -> dict:
    """Assemble the manifest dict. ``batches`` is
    ``BatchQueue.to_item_dicts()``; ``params`` is the render kwargs
    (default_voice / language / fmt / bitrate / loudness / cover_path /
    metadata / lexicon / expressive / voice_map) — same shape
    ``longform_resume.build_manifest`` uses for a single-script render, so
    ``ExpressiveOptions.from_manifest(params.get("expressive"))`` works
    unchanged on resume. Pure — no I/O."""
    return {
        "version": MANIFEST_VERSION,
        "job_id": job_id,
        "title": title or "",
        "source_filename": source_filename or "",
        "page_count": page_count,
        "batch_size_pages": batch_size_pages,
        "total_batches": len(batches),
        "params": params,
        "batches": batches,
    }


def _atomic_write_bytes(path: str, data: bytes) -> bool:
    """Temp file + fsync + os.replace + dir fsync — the same durability
    contract as longform_resume.write_manifest (#2279): a half-written file
    must never be the one a later read sees."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            flush_fd(f.fileno())
        os.replace(tmp, path)
        flush_dir(os.path.dirname(path))
        return True
    except OSError:
        return False


def write_source_pdf(job_id: str, data: bytes) -> Optional[str]:
    """Durably persist the uploaded PDF bytes so an interrupted job can be
    re-split identically (same batches, same page ranges) on resume.
    Best-effort, like write_manifest — a lost source means "not resumable",
    never a corrupted one, since the write itself is atomic."""
    path = source_pdf_path(job_id)
    if path is None:
        return None
    return path if _atomic_write_bytes(path, data) else None


def read_source_pdf(job_id: str) -> Optional[bytes]:
    path = source_pdf_path(job_id)
    if path is None:
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def write_manifest(manifest: dict) -> Optional[str]:
    """Persist the manifest. Best-effort — resume is an enhancement, never
    block the render — returns the path or None on failure/unsafe id."""
    path = manifest_path(manifest.get("job_id", ""))
    if path is None:
        return None
    try:
        payload = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return path if _atomic_write_bytes(path, payload) else None


def read_manifest(job_id: str) -> Optional[dict]:
    """Load a job's resume manifest, or None if absent/unreadable/foreign-
    shape/unsafe id."""
    path = manifest_path(job_id)
    if path is None:
        return None
    return load_manifest_file(path)


def load_manifest_file(path: str) -> Optional[dict]:
    """Load a manifest from a path obtained via :func:`scan_resumable`
    (trusted, os.listdir-derived) — avoids re-deriving a path from a
    request-supplied id."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("version") != MANIFEST_VERSION:
        return None
    if not isinstance(data.get("batches"), list):
        return None
    return data


def clear_manifest(job_id: str) -> None:
    """Remove the whole work dir (manifest + source PDF) once a job fully
    succeeds — no resume need, and no reason to keep a multi-MB source PDF
    around forever. Best-effort; a failed/interrupted job's dir is left in
    place on purpose (that's the resumable state). Resolves whichever prefix
    (current or legacy) the job actually used."""
    d = _existing_work_dir(job_id)
    if d is None:
        return
    shutil.rmtree(d, ignore_errors=True)


def has_manifest(job_id: str) -> bool:
    path = manifest_path(job_id)
    return bool(path) and os.path.isfile(path)


def scan_resumable() -> list[dict]:
    """Enumerate resumable PDF-batch jobs by scanning OUTPUTS_DIR for
    ``pdfbatch_<id>`` work dirs that hold a ``resume.json`` — plus any
    surviving ``audiobook_batch_<id>`` dirs from before the prefix-collision
    fix, so a job interrupted before the fix stays discoverable and
    resumable (see ``_LEGACY_JOB_TYPE``).

    Returns ``[{"job_id", "manifest_path"}, …]`` where every path component
    is sourced from ``os.listdir`` (the trusted filesystem), never from
    request input — mirrors longform_resume.scan_resumable exactly."""
    from core.config import OUTPUTS_DIR
    out: list[dict] = []
    try:
        root = os.path.realpath(OUTPUTS_DIR)
        names = os.listdir(root)
    except OSError:
        return out
    for prefix in (f"{_JOB_TYPE}_", f"{_LEGACY_JOB_TYPE}_"):
        for name in names:
            mpath = os.path.join(root, name, _MANIFEST_NAME)
            if name.startswith(prefix) and os.path.isfile(mpath):
                out.append({"job_id": name[len(prefix):], "manifest_path": mpath})
    return out


__all__ = [
    "MANIFEST_VERSION", "work_dir", "legacy_work_dir", "manifest_path", "source_pdf_path",
    "build_manifest", "write_source_pdf", "read_source_pdf",
    "write_manifest", "read_manifest", "load_manifest_file",
    "clear_manifest", "has_manifest", "scan_resumable",
]
