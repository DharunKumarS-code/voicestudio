"""Unit tests for services.ebook_translate — no network, no real models.

Mirrors tests/test_dub_translate.py's style (monkeypatch + MagicMock, nothing
async that isn't awaited via pytest.mark.asyncio) since ebook_translate reuses
the same provider-dispatch shape, minus every dub-timing concept.
"""
from unittest.mock import MagicMock

import pytest

from services.ebook_translate import (
    EbookSegment,
    UnsupportedLanguageError,
    UnsupportedProviderError,
    resolve_source_lang,
    translate_segments,
)


def _segs(*pairs):
    """``_segs(("1", "Hello"), ("2", "World"))`` → EbookSegment list."""
    return [EbookSegment(id=i, text=t) for i, t in pairs]


# ── resolve_source_lang ──────────────────────────────────────────────────────


def test_resolve_source_lang_prefers_explicit_value():
    assert resolve_source_lang(_segs(("1", "你好")), "en") == "en"


def test_resolve_source_lang_sniffs_script_when_unset():
    assert resolve_source_lang(_segs(("1", "你好，世界")), None) == "zh"
    assert resolve_source_lang(_segs(("1", "こんにちは")), None) == "ja"
    assert resolve_source_lang(_segs(("1", "Привет")), None) == "ru"


def test_resolve_source_lang_defaults_to_english():
    assert resolve_source_lang(_segs(("1", "Hello there")), None) == "en"


def test_resolve_source_lang_handles_empty_segments():
    assert resolve_source_lang([], None) == "en"


# ── Unknown / unsupported provider ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_provider_raises_before_any_segment_runs():
    with pytest.raises(UnsupportedProviderError):
        await translate_segments(
            _segs(("1", "Hello")), source_lang="en", target_lang="es",
            provider="not-a-real-engine",
        )


# ── Argos: success, per-segment degrade, missing pack ───────────────────────


@pytest.mark.asyncio
async def test_argos_translates_each_segment(monkeypatch):
    import sys
    import types

    fake_argostranslate = types.ModuleType("argostranslate")
    fake_translate_mod = types.ModuleType("argostranslate.translate")
    fake_translate_mod.translate = lambda text, src, tgt: f"[{tgt}] {text}"
    fake_argostranslate.translate = fake_translate_mod
    monkeypatch.setitem(sys.modules, "argostranslate", fake_argostranslate)
    monkeypatch.setitem(sys.modules, "argostranslate.translate", fake_translate_mod)

    from services import translation_engines
    monkeypatch.setattr(translation_engines, "argos_pack_status", lambda src, tgts: {
        "source_lang": "en", "pairs": [{"source_lang": "en", "target_lang": "es", "installed": True}],
    })
    monkeypatch.setattr(translation_engines, "argos_lang_code", lambda code: code)
    monkeypatch.setattr("core.execstack.ensure_ctranslate2_loadable", lambda: (True, "ready"))

    result = await translate_segments(
        _segs(("1", "Hello"), ("2", "World")), source_lang="en", target_lang="es", provider="argos",
    )
    assert result == [
        {"id": "1", "text": "[es] Hello"},
        {"id": "2", "text": "[es] World"},
    ]


@pytest.mark.asyncio
async def test_argos_degrades_one_failed_segment_without_failing_the_chapter(monkeypatch):
    import sys
    import types

    fake_argostranslate = types.ModuleType("argostranslate")
    fake_translate_mod = types.ModuleType("argostranslate.translate")

    def _translate(text, src, tgt):
        if text == "boom":
            raise RuntimeError("provider exploded")
        return f"[{tgt}] {text}"

    fake_translate_mod.translate = _translate
    fake_argostranslate.translate = fake_translate_mod
    monkeypatch.setitem(sys.modules, "argostranslate", fake_argostranslate)
    monkeypatch.setitem(sys.modules, "argostranslate.translate", fake_translate_mod)

    from services import translation_engines
    monkeypatch.setattr(translation_engines, "argos_pack_status", lambda src, tgts: {
        "source_lang": "en", "pairs": [{"source_lang": "en", "target_lang": "es", "installed": True}],
    })
    monkeypatch.setattr(translation_engines, "argos_lang_code", lambda code: code)
    monkeypatch.setattr("core.execstack.ensure_ctranslate2_loadable", lambda: (True, "ready"))

    result = await translate_segments(
        _segs(("1", "Hello"), ("2", "boom")), source_lang="en", target_lang="es", provider="argos",
    )
    by_id = {r["id"]: r for r in result}
    assert by_id["1"] == {"id": "1", "text": "[es] Hello"}
    assert by_id["2"]["text"] == "boom"  # degraded to source text
    assert "provider exploded" in by_id["2"]["error"]


@pytest.mark.asyncio
async def test_argos_missing_language_pack_raises_before_translating(monkeypatch):
    import sys
    import types

    fake_argostranslate = types.ModuleType("argostranslate")
    fake_translate_mod = types.ModuleType("argostranslate.translate")
    fake_translate_mod.translate = lambda text, src, tgt: text
    fake_argostranslate.translate = fake_translate_mod
    monkeypatch.setitem(sys.modules, "argostranslate", fake_argostranslate)
    monkeypatch.setitem(sys.modules, "argostranslate.translate", fake_translate_mod)

    from services import translation_engines
    monkeypatch.setattr(translation_engines, "argos_pack_status", lambda src, tgts: {
        "source_lang": "en", "pairs": [{"source_lang": "en", "target_lang": "ja", "installed": False}],
    })
    monkeypatch.setattr("core.execstack.ensure_ctranslate2_loadable", lambda: (True, "ready"))

    with pytest.raises(UnsupportedLanguageError):
        await translate_segments(
            _segs(("1", "Hello")), source_lang="en", target_lang="ja", provider="argos",
        )


# ── Chapter-level timeout degrade (new behavior vs. dub_translate) ─────────


@pytest.mark.asyncio
async def test_argos_chapter_timeout_degrades_to_source_text(monkeypatch):
    import asyncio
    import sys
    import types

    import services.ebook_translate as ebook_translate

    monkeypatch.setattr(ebook_translate, "chapter_translate_timeout_s", lambda: 0.05)

    fake_argostranslate = types.ModuleType("argostranslate")
    fake_translate_mod = types.ModuleType("argostranslate.translate")

    def _slow_translate(text, src, tgt):
        import time
        time.sleep(1.0)  # longer than the 0.05s budget above
        return text

    fake_translate_mod.translate = _slow_translate
    fake_argostranslate.translate = fake_translate_mod
    monkeypatch.setitem(sys.modules, "argostranslate", fake_argostranslate)
    monkeypatch.setitem(sys.modules, "argostranslate.translate", fake_translate_mod)

    from services import translation_engines
    monkeypatch.setattr(translation_engines, "argos_pack_status", lambda src, tgts: {
        "source_lang": "en", "pairs": [{"source_lang": "en", "target_lang": "es", "installed": True}],
    })
    monkeypatch.setattr(translation_engines, "argos_lang_code", lambda code: code)
    monkeypatch.setattr("core.execstack.ensure_ctranslate2_loadable", lambda: (True, "ready"))

    result = await translate_segments(
        _segs(("1", "Hello")), source_lang="en", target_lang="es", provider="argos",
    )
    assert result == [{"id": "1", "text": "Hello", "error": "timeout"}]


# ── OpenAI / LLM path: script validation + not-configured degrade ─────────


@pytest.mark.asyncio
async def test_llm_not_configured_raises_actionable_error(monkeypatch):
    from services import llm_skills

    monkeypatch.setattr(llm_skills, "resolve_skill_client", lambda skill_id: None)
    monkeypatch.setattr(
        llm_skills, "resolve_skill",
        lambda skill_id: MagicMock(reason="unconfigured"),
    )

    with pytest.raises(UnsupportedProviderError):
        await translate_segments(
            _segs(("1", "Hello")), source_lang="en", target_lang="es", provider="openai",
        )


@pytest.mark.asyncio
async def test_llm_translates_and_validates_target_script(monkeypatch):
    from services import llm_skills

    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hola"))])
    handle = MagicMock(client=client, model="test-model", timeout=5.0)
    monkeypatch.setattr(llm_skills, "resolve_skill_client", lambda skill_id: handle)

    result = await translate_segments(
        _segs(("1", "Hello")), source_lang="en", target_lang="es", provider="openai",
    )
    assert result == [{"id": "1", "text": "Hola"}]
    assert client.chat.completions.create.call_args.kwargs["temperature"] == 0.2


@pytest.mark.asyncio
async def test_llm_retries_once_on_wrong_script_then_degrades(monkeypatch):
    """Hindi target but the model answers in Latin script both times."""
    from services import llm_skills

    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="namaste"))])
    handle = MagicMock(client=client, model="test-model", timeout=5.0)
    monkeypatch.setattr(llm_skills, "resolve_skill_client", lambda skill_id: handle)

    result = await translate_segments(
        _segs(("1", "Hello")), source_lang="en", target_lang="hi", provider="openai",
    )
    assert client.chat.completions.create.call_count == 2
    assert result[0]["id"] == "1"
    assert result[0]["text"] == "Hello"  # degraded to source
    assert "error" in result[0]
