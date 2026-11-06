"""Durable resume manifest + source-PDF persistence for PDF-batch audiobook
jobs (Stage 3). Pure file/JSON I/O — no server, no TTS, no GPU."""
from __future__ import annotations

import os

import pytest

from services import pdf_batch_resume as resume


def test_write_and_read_source_pdf_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    path = resume.write_source_pdf("job1", b"%PDF-1.4 fake bytes")
    assert path is not None and os.path.isfile(path)
    assert resume.read_source_pdf("job1") == b"%PDF-1.4 fake bytes"


def test_read_source_pdf_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    assert resume.read_source_pdf("never-written") is None


def test_write_and_read_manifest_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    manifest = resume.build_manifest(
        job_id="job1", title="My Book", source_filename="book.pdf",
        page_count=30, batch_size_pages=10, params={"default_voice": None},
        batches=[{"index": 0, "start_page": 1, "end_page": 10, "status": "pending",
                  "output": None, "chapters": 0, "duration_s": 0.0, "error": None,
                  "child_job_id": None, "started_at": None, "finished_at": None}],
    )
    path = resume.write_manifest(manifest)
    assert path is not None and os.path.isfile(path)
    loaded = resume.read_manifest("job1")
    assert loaded is not None
    assert loaded["job_id"] == "job1"
    assert loaded["title"] == "My Book"
    assert loaded["total_batches"] == 1
    assert loaded["batches"][0]["start_page"] == 1


def test_read_manifest_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    assert resume.read_manifest("never-written") is None


def test_read_manifest_rejects_wrong_version(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    manifest = resume.build_manifest(
        job_id="job1", page_count=1, batch_size_pages=10, params={}, batches=[],
    )
    manifest["version"] = 999
    resume.write_manifest(manifest)
    assert resume.read_manifest("job1") is None


def test_read_manifest_rejects_foreign_shape(tmp_path, monkeypatch):
    """A manifest missing the "batches" list (e.g. a stray/corrupted file)
    is rejected rather than crashing the caller."""
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    d = resume.work_dir("job1")
    os.makedirs(d, exist_ok=True)
    with open(resume.manifest_path("job1"), "w", encoding="utf-8") as f:
        f.write('{"version": 1, "job_id": "job1"}')  # no "batches" key
    assert resume.read_manifest("job1") is None


def test_write_manifest_is_atomic_no_partial_file_visible(tmp_path, monkeypatch):
    """A write must never leave a reader seeing a half-written manifest —
    verified here by checking the .tmp file is always cleaned up and the
    published file's content is always complete JSON."""
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    manifest = resume.build_manifest(
        job_id="job1", page_count=1, batch_size_pages=10, params={}, batches=[],
    )
    resume.write_manifest(manifest)
    d = resume.work_dir("job1")
    assert not os.path.exists(os.path.join(d, "resume.json.tmp"))
    assert resume.read_manifest("job1") is not None


def test_has_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    assert resume.has_manifest("job1") is False
    resume.write_manifest(resume.build_manifest(
        job_id="job1", page_count=1, batch_size_pages=10, params={}, batches=[]))
    assert resume.has_manifest("job1") is True


def test_clear_manifest_removes_manifest_and_source(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    resume.write_source_pdf("job1", b"pdf bytes")
    resume.write_manifest(resume.build_manifest(
        job_id="job1", page_count=1, batch_size_pages=10, params={}, batches=[]))
    assert resume.has_manifest("job1")
    resume.clear_manifest("job1")
    assert not resume.has_manifest("job1")
    assert resume.read_source_pdf("job1") is None


def test_clear_manifest_on_never_written_job_is_a_safe_noop(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    resume.clear_manifest("never-written")  # must not raise


# ── scan_resumable ───────────────────────────────────────────────────────

def test_scan_resumable_finds_written_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    resume.write_manifest(resume.build_manifest(
        job_id="jobA", page_count=1, batch_size_pages=10, params={}, batches=[]))
    resume.write_manifest(resume.build_manifest(
        job_id="jobB", page_count=1, batch_size_pages=10, params={}, batches=[]))
    found = {e["job_id"] for e in resume.scan_resumable()}
    assert found == {"jobA", "jobB"}


def test_scan_resumable_ignores_unrelated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    os.makedirs(tmp_path / "audiobook_someotherjob", exist_ok=True)
    (tmp_path / "audiobook_someotherjob" / "resume.json").write_text("{}")
    os.makedirs(tmp_path / "not_a_job_dir", exist_ok=True)
    resume.write_manifest(resume.build_manifest(
        job_id="jobA", page_count=1, batch_size_pages=10, params={}, batches=[]))
    found = [e["job_id"] for e in resume.scan_resumable()]
    assert found == ["jobA"]


def test_scan_resumable_empty_outputs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    assert resume.scan_resumable() == []


def test_load_manifest_file_from_scan_result(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    resume.write_manifest(resume.build_manifest(
        job_id="jobA", title="Title A", page_count=1, batch_size_pages=10,
        params={}, batches=[]))
    entries = resume.scan_resumable()
    assert len(entries) == 1
    loaded = resume.load_manifest_file(entries[0]["manifest_path"])
    assert loaded["title"] == "Title A"


# ── path safety ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_id", [
    "../escape", "..\\escape", "a/b", "a\\b", "", "x" * 65, "a:b", "a b",
])
def test_work_dir_rejects_unsafe_job_ids(tmp_path, monkeypatch, bad_id):
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    assert resume.work_dir(bad_id) is None


def test_work_dir_resolves_strictly_inside_outputs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    d = resume.work_dir("safe-id_123")
    assert d is not None
    root = os.path.realpath(str(tmp_path))
    assert os.path.commonpath([d, root]) == root


# ── Fix 2: directory-prefix collision with services.longform_resume ────────
#
# "audiobook_batch_<id>" starts with "audiobook_" — longform_resume's own
# scanner (RESUMABLE_TYPES = ("audiobook", "story"), matched by
# name.startswith(f"{jt}_")) silently matched it too, surfacing a spurious
# "Resume an audiobook that didn't finish" entry for every interrupted
# PDF-batch job. These tests prove the fix and the backward-compat path for
# jobs already interrupted under the old prefix.

def test_new_jobs_use_the_pdfbatch_prefix(tmp_path, monkeypatch):
    """Requirement 1: new PDF batch jobs use pdfbatch_."""
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    d = resume.write_manifest(resume.build_manifest(
        job_id="freshjob1", page_count=1, batch_size_pages=10, params={}, batches=[]))
    assert d is not None
    assert os.path.basename(os.path.dirname(d)) == "pdfbatch_freshjob1"
    assert (tmp_path / "pdfbatch_freshjob1").is_dir()
    assert not (tmp_path / "audiobook_batch_freshjob1").exists()


def test_normal_audiobook_recovery_does_not_list_pdf_batch_jobs(tmp_path, monkeypatch):
    """Requirement 2: services.longform_resume's own scanner must not
    mistake a (new-prefix) PDF batch manifest for a single-script job."""
    from services import longform_resume

    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    resume.write_manifest(resume.build_manifest(
        job_id="pdfjob1", page_count=3, batch_size_pages=1, params={}, batches=[]))

    assert longform_resume.scan_resumable() == []
    # Sanity: pdf_batch_resume's own scanner DOES see it.
    assert [e["job_id"] for e in resume.scan_resumable()] == ["pdfjob1"]


def test_legacy_audiobook_batch_prefixed_manifest_remains_recoverable(tmp_path, monkeypatch):
    """Requirement 3: a manifest written under the pre-fix
    "audiobook_batch_<id>" prefix (simulating a job interrupted before this
    fix shipped) is still discoverable, readable, and resumable — without
    any migration step."""
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    legacy_dir = tmp_path / "audiobook_batch_oldjob1"
    legacy_dir.mkdir()
    manifest = resume.build_manifest(
        job_id="oldjob1", title="Old Interrupted Book", page_count=5,
        batch_size_pages=1, params={"default_voice": None},
        batches=[{"index": 0, "start_page": 1, "end_page": 1, "status": "done",
                  "output": "audiobook_oldjob1b0.m4b", "chapters": 1, "duration_s": 1.0,
                  "error": None, "child_job_id": "oldjob1b0", "started_at": 1.0,
                  "finished_at": 2.0}],
    )
    (legacy_dir / "resume.json").write_text(
        __import__("json").dumps(manifest), encoding="utf-8")
    (legacy_dir / "source.pdf").write_bytes(b"%PDF-1.4 legacy bytes")

    # Discoverable.
    entries = resume.scan_resumable()
    assert [e["job_id"] for e in entries] == ["oldjob1"]

    # Readable.
    loaded = resume.read_manifest("oldjob1")
    assert loaded is not None
    assert loaded["title"] == "Old Interrupted Book"
    assert resume.read_source_pdf("oldjob1") == b"%PDF-1.4 legacy bytes"

    # Resumable: a write against this job_id updates the SAME legacy dir in
    # place (no migration/rename), and it stays readable afterward.
    manifest["batches"][0]["status"] = "done"
    manifest["total_batches"] = 1
    resume.write_manifest(manifest)
    assert not (tmp_path / "pdfbatch_oldjob1").exists()
    assert legacy_dir.is_dir()
    assert resume.read_manifest("oldjob1")["batches"][0]["status"] == "done"

    # Clearing it (as a normal full-success completion does) removes the
    # legacy dir, not a nonexistent new-prefix one.
    resume.clear_manifest("oldjob1")
    assert not legacy_dir.exists()
    assert resume.has_manifest("oldjob1") is False


def test_ordinary_audiobook_and_story_recovery_still_work(tmp_path, monkeypatch):
    """Requirement 4: longform_resume's own audiobook/story manifests are
    completely unaffected by this module existing alongside them."""
    from services import longform_resume

    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))

    audiobook_manifest = longform_resume.build_manifest(
        job_id="scriptjob1", job_type="audiobook",
        plan_chapters=[{"title": "Ch 1", "spans": []}], params={}, title="My Audiobook",
    )
    longform_resume.write_manifest(audiobook_manifest)
    story_manifest = longform_resume.build_manifest(
        job_id="storyjob1", job_type="story",
        plan_chapters=[{"title": "Scene 1", "spans": []}], params={}, title="My Story",
    )
    longform_resume.write_manifest(story_manifest)
    # A PDF batch job alongside them must not interfere either direction.
    resume.write_manifest(resume.build_manifest(
        job_id="pdfjob2", page_count=1, batch_size_pages=10, params={}, batches=[]))

    found = {(e["job_type"], e["job_id"]) for e in longform_resume.scan_resumable()}
    assert found == {("audiobook", "scriptjob1"), ("story", "storyjob1")}

    loaded_audiobook = longform_resume.read_manifest("audiobook", "scriptjob1")
    assert loaded_audiobook["title"] == "My Audiobook"
    loaded_story = longform_resume.read_manifest("story", "storyjob1")
    assert loaded_story["title"] == "My Story"

    # And the PDF-batch scanner, symmetrically, must not see the single-script jobs.
    assert [e["job_id"] for e in resume.scan_resumable()] == ["pdfjob2"]
