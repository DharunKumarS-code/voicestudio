"""AudioSeal CPU chunk batching (opt-in, batch_size>1) — correctness coverage.

``batch_size`` defaults to 1 everywhere (see test_watermark_chunking_1045.py,
untouched by this file) and no production caller passes it yet — this is
strictly an opt-in fast path for long audio (audiobook chapters). These tests
pin: grouping/padding/crop math is exactly right at batch_size 2/4/8, chunk
order and waveform length are preserved, a batching failure degrades to
sequential embedding for just the failed group (never losing the mark), and
— only where the real AudioSeal checkpoint is already cached locally, never
triggering a download — the real model's batched output is numerically close
to its own sequential output and both still detect identically.
"""
from __future__ import annotations

import math
import os

import pytest
import torch

from services import watermark
from services.watermark import (
    _group_chunks,
    detect_watermark,
    embed_watermark,
)

SR = 16000  # == _AUDIOSEAL_SAMPLE_RATE, so no resampling noise in these fakes


class FakeGenerator:
    """Position-preserving, content-checkable transform (x*2): any padding,
    cropping or ordering bug shows up as a wrong VALUE at a specific index,
    not just a wrong shape — same technique test_watermark_chunking_1045.py
    uses. Records each call's (batch rows, time length, message shape) for
    structural assertions, and can simulate a batched-forward failure."""

    def __init__(self, *, fail_when_batched: bool = False):
        self.calls: list[tuple[int, int, tuple | None]] = []
        self.fail_when_batched = fail_when_batched

    def __call__(self, audio, sample_rate, message=None):
        if self.fail_when_batched and audio.shape[0] > 1:
            raise RuntimeError("simulated batched-forward failure")
        self.calls.append(
            (audio.shape[0], audio.shape[-1], tuple(message.shape) if message is not None else None)
        )
        return audio * 2.0


class FakeDetector:
    def detect_watermark(self, audio, sample_rate, message_threshold=0.5):
        conf = 0.9 if float(audio.abs().max()) > 100.0 else 0.1
        msg = torch.tensor(watermark.OMNI_MESSAGE) if conf > 0.5 else torch.zeros(16)
        return (conf, msg)


@pytest.fixture
def fake_audioseal(monkeypatch):
    def _install(**kw):
        gen = FakeGenerator(**kw)
        det = FakeDetector()
        monkeypatch.setattr(watermark, "_generator", gen)
        monkeypatch.setattr(watermark, "_detector", det)
        monkeypatch.setattr(watermark, "_audioseal_available", True)
        return gen, det

    return _install


# ── _group_chunks itself ─────────────────────────────────────────────────────


def test_group_chunks_batch_size_one_is_all_singletons():
    assert list(_group_chunks([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_group_chunks_preserves_order_and_grouping(batch_size):
    groups = list(_group_chunks(range(10), batch_size))
    assert [x for g in groups for x in g] == list(range(10))
    assert all(len(g) <= batch_size for g in groups)
    assert all(len(g) == batch_size for g in groups[:-1])  # only the last may be short


def test_group_chunks_rejects_batch_size_below_one():
    with pytest.raises(ValueError):
        list(_group_chunks([1], 0))


# ── Default behavior is byte-for-byte unchanged ─────────────────────────────


def test_default_batch_size_never_calls_the_batched_path(fake_audioseal):
    gen, _ = fake_audioseal()
    wave = torch.arange(SR * 95, dtype=torch.float32).unsqueeze(0)
    out = embed_watermark(wave, SR, force=True)  # batch_size defaults to 1
    assert all(b == 1 for b, _, _ in gen.calls)
    assert len(gen.calls) == 4  # unchanged from test_watermark_chunking_1045.py
    assert torch.equal(out, wave * 2.0)


# ── Shape / length / ordering at batch_size 2, 4, 8, across durations ──────


@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize(
    "seconds", [3, 10, 30.5, 95, 600, 1200],  # short, short, tail-fold, #1045 fixture, 10min, 20min
)
def test_batched_output_matches_sequential_exactly(fake_audioseal, batch_size, seconds):
    n = int(SR * seconds)
    wave = torch.arange(n, dtype=torch.float32).unsqueeze(0)

    gen_seq, _ = fake_audioseal()
    out_seq = embed_watermark(wave, SR, force=True, batch_size=1)
    n_chunks = len(gen_seq.calls)

    gen_batch, _ = fake_audioseal()
    out_batch = embed_watermark(wave, SR, force=True, batch_size=batch_size)

    assert out_batch.shape == wave.shape == out_seq.shape
    assert torch.equal(out_batch, out_seq)
    assert torch.equal(out_batch, wave * 2.0)

    expected_batched_calls = math.ceil(n_chunks / batch_size) if n_chunks else 0
    assert len(gen_batch.calls) == expected_batched_calls


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_message_tensor_repeated_to_batch_row_count(fake_audioseal, batch_size):
    gen, _ = fake_audioseal()
    wave = torch.arange(SR * 95, dtype=torch.float32).unsqueeze(0)  # 4 chunks: 30/30/30/5s
    embed_watermark(wave, SR, force=True, batch_size=batch_size)
    for batch_rows, _, msg_shape in gen.calls:
        assert msg_shape == (batch_rows, 16)


# ── Variable-length final chunk (tail-fold) inside a batch group ───────────


def test_variable_length_final_chunk_is_padded_and_cropped_correctly(fake_audioseal):
    gen, _ = fake_audioseal()
    # 95s -> chunks of 30,30,30,5s. batch_size=4 puts ALL of them, including
    # the short final one, into ONE batched call -> forces padding.
    wave = torch.arange(SR * 95, dtype=torch.float32).unsqueeze(0)
    out = embed_watermark(wave, SR, force=True, batch_size=4)
    assert len(gen.calls) == 1
    batch_rows, max_len, _ = gen.calls[0]
    assert batch_rows == 4
    assert max_len == 30 * SR  # padded up to the longest (first three) chunks
    assert out.shape == wave.shape
    assert torch.equal(out, wave * 2.0)  # every row cropped back correctly despite padding


# ── Sequential fallback on a batching failure ───────────────────────────────


def test_batching_failure_falls_back_to_sequential_for_that_group(fake_audioseal, caplog):
    gen, _ = fake_audioseal(fail_when_batched=True)
    wave = torch.arange(SR * 95, dtype=torch.float32).unsqueeze(0)
    out = embed_watermark(wave, SR, force=True, batch_size=4)
    # the batched attempt raised; every chunk must still have been embedded
    # one at a time (batch size 1 in every recorded call) and the result is
    # still fully, correctly watermarked.
    assert all(b == 1 for b, _, _ in gen.calls)
    assert len(gen.calls) == 4
    assert torch.equal(out, wave * 2.0)
    assert "falling back to sequential" in caplog.text


def test_batching_failure_never_returns_unwatermarked_audio(fake_audioseal):
    fake_audioseal(fail_when_batched=True)
    wave = torch.randn(1, SR * 61)  # 2 chunks of ~30s -> one batch group of 2
    out = embed_watermark(wave, SR, force=True, batch_size=2)
    assert not torch.equal(out, wave)  # it WAS watermarked, via the fallback


# ── Headroom retry compatibility ─────────────────────────────────────────────
# The whole-waveform headroom-rescaling retry (embed_watermark's `for _ in
# range(_HEADROOM_PASSES)` loop) re-calls the SAME `_embed` closure on every
# re-embed, so it must regroup chunks into batches identically on every pass
# — not fall back to per-chunk on retry just because the first pass batched.


class _OffsetGenerator:
    """Additive residual, like test_watermark_headroom.py's ``_Offset`` —
    elementwise, so it behaves identically whether called with one row or a
    batched group. Records each call's batch-row count."""

    def __init__(self):
        self.calls: list[int] = []

    def __call__(self, audio, sample_rate, message=None):
        self.calls.append(audio.shape[0])
        return audio + 0.05


def test_headroom_retry_regroups_batches_identically_on_every_pass(monkeypatch):
    monkeypatch.setattr(watermark, "_check_available", lambda: True)
    gen = _OffsetGenerator()
    monkeypatch.setattr(watermark, "_get_generator", lambda *a, **k: gen)

    # 65s @ 16kHz -> chunks of 30, 30, 5s. batch_size=2 groups the first two
    # into ONE batched call and leaves the 5s tail as a singleton, so a
    # single embed_watermark call exercises both the batched and the
    # len(group)==1 branch, and the retry loop must repeat that exact [2, 1]
    # grouping on its re-embed, not just the plain per-chunk path.
    wave = torch.full((1, SR * 65), 0.98)  # near full-scale host -> forces a retry
    marked = watermark.embed_watermark(wave, SR, force=True, batch_size=2)

    assert float(marked.abs().max()) <= 1.0 + 1e-6
    gain = (float(marked[0, 0]) - 0.05) / 0.98
    assert 0.0 < gain < 1.0  # host WAS attenuated (a retry happened)
    assert torch.allclose(marked - wave * gain, torch.full_like(wave, 0.05), atol=1e-5)

    # initial embed + exactly one retry, each regrouped as [2, 1] batch rows
    assert gen.calls == [2, 1, 2, 1]


# ── Real AudioSeal checkpoint — only runs where it's already resolvable
# OFFLINE; never downloads it. Reports the ACTUAL measured numerical
# difference rather than asserting bit-identical output. ───────────────────


def _real_models_load_offline() -> bool:
    """True only if BOTH AudioSeal models build with zero network access.

    Forces HF_HUB_OFFLINE for this probe alone so a missing checkpoint raises
    instead of downloading — the only way to check "is this safely usable"
    without risking the one thing this suite must never do. Checks the
    detector too: an earlier version of this probe checked only the
    generator and the detector half of the test went on to trigger a real
    33 MB download on a machine that had the generator cached but not the
    detector — exactly the incident this function exists to prevent.
    Deliberately avoids services.watermark._get_generator()/_get_detector()
    so a probe here never seeds the module's shared globals as a side effect.
    """
    prior = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from audioseal import AudioSeal

        AudioSeal.load_generator("audioseal_wm_16bits")
        AudioSeal.load_detector("audioseal_detector_16bits")
        return True
    except Exception:
        return False
    finally:
        if prior is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prior


requires_real_checkpoints = pytest.mark.skipif(
    not _real_models_load_offline(),
    reason="real AudioSeal generator/detector do not both resolve offline on "
           "this machine; skipping rather than downloading one (this suite "
           "never downloads models)",
)


@requires_real_checkpoints
@pytest.mark.parametrize("batch_size", [2, 4])
def test_real_audioseal_batched_output_is_numerically_close_to_sequential(
    monkeypatch, batch_size
):
    # Belt-and-suspenders beyond the collection-time probe above: force
    # offline mode for the actual test body too, so this specific test can
    # never download anything even if some path the probe didn't cover
    # (precision, device variant, …) needs a file the probe didn't touch.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    t = torch.arange(SR * 95, dtype=torch.float32) / SR
    wave = (0.05 * torch.sin(2 * math.pi * 220.0 * t)).unsqueeze(0)

    out_seq = embed_watermark(wave.clone(), SR, force=True, batch_size=1)
    out_batch = embed_watermark(wave.clone(), SR, force=True, batch_size=batch_size)

    assert out_batch.shape == out_seq.shape == wave.shape
    max_abs_diff = float((out_batch - out_seq).abs().max())
    print(f"\n[real AudioSeal] batch_size={batch_size}: max_abs_diff={max_abs_diff:.3e}")
    # Not claiming bit-identical — this is a loose sanity bound (watermark
    # residuals are O(1e-2)), not a tuned numerical-equivalence tolerance.
    assert max_abs_diff < 1e-3

    det_seq = detect_watermark(out_seq, SR)
    det_batch = detect_watermark(out_batch, SR)
    assert det_seq["is_watermarked"] is True
    assert det_batch["is_watermarked"] is True
    assert det_seq["is_omnivoice"] is True
    assert det_batch["is_omnivoice"] is True
