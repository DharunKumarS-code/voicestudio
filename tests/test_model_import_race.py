"""The model loader retries only the concurrent-first-import race.

Seen live: the PDF reader's first request, ~1 s after backend startup, imported
omnivoice while a startup task was importing transformers; the request failed
with "cannot import name 'AutoFeatureExtractor' from 'transformers'" and the
same import succeeded 300 ms later. Real environment breakage must still fail
at once with its own error.
"""

import pytest

from services import model_manager as mm


def _sequence(*outcomes):
    calls = {"n": 0}

    def importer():
        outcome = outcomes[min(calls["n"], len(outcomes) - 1)]
        calls["n"] += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return importer, calls


RACE = ImportError(
    "cannot import name 'AutoFeatureExtractor' from 'transformers' "
    "(E:\\venv\\Lib\\site-packages\\transformers\\__init__.py)"
)


def test_a_lost_race_is_retried_and_succeeds():
    importer, calls = _sequence(RACE, "OmniVoiceClass")
    sleeps = []
    assert mm._import_with_race_retry(importer, sleep=sleeps.append) == "OmniVoiceClass"
    assert calls["n"] == 2
    assert sleeps == [mm._IMPORT_RACE_BACKOFF_S]


def test_partially_initialized_module_counts_as_a_race():
    exc = ImportError(
        "cannot import name 'x' from partially initialized module 'transformers.models.auto' "
        "(most likely due to a circular import)"
    )
    importer, calls = _sequence(exc, "ok")
    assert mm._import_with_race_retry(importer, sleep=lambda _s: None) == "ok"
    assert calls["n"] == 2


def test_a_missing_module_is_never_retried():
    missing = ModuleNotFoundError("No module named 'omnivoice'", name="omnivoice")
    importer, calls = _sequence(missing, "never")
    with pytest.raises(ModuleNotFoundError):
        mm._import_with_race_retry(importer, sleep=lambda _s: pytest.fail("slept"))
    assert calls["n"] == 1


def test_other_import_errors_are_not_retried():
    broken = ImportError("DLL load failed while importing _C: The specified module could not be found.")
    importer, calls = _sequence(broken, "never")
    with pytest.raises(ImportError, match="DLL load failed"):
        mm._import_with_race_retry(importer, sleep=lambda _s: pytest.fail("slept"))
    assert calls["n"] == 1


def test_a_race_that_never_settles_still_fails_after_bounded_retries():
    importer, calls = _sequence(RACE)
    with pytest.raises(ImportError, match="AutoFeatureExtractor"):
        mm._import_with_race_retry(importer, retries=3, sleep=lambda _s: None)
    assert calls["n"] == 4  # first try + 3 retries, then the real error


def test_lazy_omnivoice_goes_through_the_retry(monkeypatch):
    monkeypatch.setattr(mm, "_OmniVoice", None)
    importer, calls = _sequence(RACE, "OmniVoiceClass")
    monkeypatch.setattr(mm, "_import_omnivoice_class", importer)
    monkeypatch.setattr(mm.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        "core.windows_rocm.pipelines.prepare_audio_imports", lambda: None, raising=False
    )
    assert mm._lazy_omnivoice() == "OmniVoiceClass"
    assert calls["n"] == 2
