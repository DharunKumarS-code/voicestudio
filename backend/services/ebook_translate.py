"""Chapter-level translation for the Ebook Translator feature.

Reuses the provider dispatch already proven in ``api.routers.dub_translate``
(NLLB / Argos / OpenAI-compatible LLM / paid APIs / deep_translator), stripped
of everything that exists only to match a video's timing: no duration
planning, no speech-rate "fit"/"condense" pass, no dialect/lip-sync handling,
no per-segment ``slot_seconds``. An ebook chapter has no clock to hit.

The one thing this module adds that ``dub_translate`` does not have: a real
wall-clock budget on every provider call. NLLB (local model) and Argos (local
model) have no timeout in the dub path today — a book-length job multiplies
the odds of hitting a hang, so every call here is bounded by ``timeout_s`` and
degrades to the source text with ``error: "timeout"`` rather than wedging the
whole chapter. Degradation is per-segment where the provider call is itself
per-segment (openai, paid APIs); for the single-call batch engines (NLLB,
Argos) a timeout degrades the whole chapter's segments at once — the
underlying call keeps running in the background exactly like a wedged ASR
transcribe (see ``services.asr_backend.run_transcribe_guarded``'s docstring),
and that capacity is reclaimed only when it eventually returns or the process
is killed. This is intentionally the same trade-off that already ships for
ASR, not a new one invented here.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("omnivoice.ebook_translate")

#: Wall-clock budget for translating one chapter's worth of segments.
#: Per-chapter, not per-segment: a chapter is the unit of resumability
#: (see backend/api/routers/audiobook.py's translate route), so a timeout
#: here degrades at most one chapter, never the whole book.
_DEFAULT_CHAPTER_TIMEOUT_S = 90.0


def chapter_translate_timeout_s() -> float:
    try:
        return max(5.0, float(os.environ.get(
            "OMNIVOICE_EBOOK_TRANSLATE_TIMEOUT_S", str(_DEFAULT_CHAPTER_TIMEOUT_S))))
    except ValueError:
        return _DEFAULT_CHAPTER_TIMEOUT_S


@dataclass
class EbookSegment:
    """One translatable unit (a chapter title or one paragraph of its body).

    Deliberately NOT ``schemas.requests.TranslateSegment`` — that model's
    extra fields (``slot_seconds``, ``start``, ``end``, ``direction``) are all
    dub-timing concepts with no meaning here, and reusing it would let a book
    translation accidentally carry timing data nobody set. ``extra`` round-trips
    anything a caller wants attached to the result row (e.g. a chapter index).
    """

    id: str
    text: str
    extra: dict = field(default_factory=dict)


class UnsupportedProviderError(ValueError):
    """Raised for an unknown or unconfigured provider, before any segment runs."""


class UnsupportedLanguageError(ValueError):
    """Raised when the chosen provider cannot translate into/from a language."""


def _guess_source_lang(segments: "list[EbookSegment]") -> "str | None":
    """Script-sniff a likely source language from the first few segments.

    Same heuristic as ``dub_translate._guess_lang_from_text``, reimplemented
    locally rather than imported: that helper takes dub ``TranslateSegment``
    objects (``.text`` attribute on a pydantic model), and re-typing it for a
    plain dataclass isn't worth a cross-module dependency for four lines.
    """
    import re

    text = " ".join(s.text or "" for s in segments[:8])
    script_codes = (
        (r"[぀-ヿ]", "ja"),
        (r"[가-힣ᄀ-ᇿ]", "ko"),
        (r"[一-鿿]", "zh"),
        (r"[Ѐ-ӿ]", "ru"),
        (r"[؀-ۿ]", "ar"),
    )
    for pattern, code in script_codes:
        if re.search(pattern, text):
            return code
    return None


def resolve_source_lang(segments: "list[EbookSegment]", source_lang: "str | None") -> str:
    """Explicit value wins; else script-sniff; else English.

    Mirrors ``dub_translate._resolve_source_lang`` minus its dub-job-lookup
    branch (there is no dub job here) — the two generic fallbacks are the
    same.
    """
    if source_lang:
        return source_lang
    return _guess_source_lang(segments) or "en"


def _degrade(segments: "list[EbookSegment]", error: str) -> "list[dict]":
    return [{"id": s.id, "text": s.text, "error": error} for s in segments]


async def _run_with_timeout(coro, segments: "list[EbookSegment]", timeout_s: float, *, what: str) -> "list[dict]":
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning(
            "%s translate timed out after %.0fs for %d segment(s); degrading to source text.",
            what, timeout_s, len(segments),
        )
        return _degrade(segments, "timeout")


async def _translate_nllb(
    segments: "list[EbookSegment]", *, source_lang: str, target_lang: str,
) -> "list[dict]":
    from api.routers.dub_translate import (
        _load_nllb_component, _nllb_batch_size, _nllb_hypothesis_budget,
        _nllb_language, _should_unload_nllb, _unload_nllb,
    )
    from api.routers import dub_translate as _dt
    from services.model_acceptance import ensure_accepted
    from services.model_manager import _gpu_pool, release_device_cache

    flores_src = _nllb_language(source_lang)
    flores_tgt = _nllb_language(target_lang)
    unsupported = [c for c, f in (("source", flores_src), ("target", flores_tgt)) if f is None]
    if unsupported:
        raise UnsupportedLanguageError(
            f"NLLB does not support the requested {' and '.join(unsupported)} language.")

    ensure_accepted([_dt._NLLB_REPO_ID])

    def _blocking() -> "list[dict]":
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        try:
            tokenizer = _load_nllb_component(AutoTokenizer)
            model = _load_nllb_component(AutoModelForSeq2SeqLM)
            if device != "cpu":
                try:
                    model = model.to(device)
                except Exception as e:  # noqa: BLE001
                    logger.warning("NLLB %s placement failed, using CPU: %s", device, e)
                    device = "cpu"
        except Exception as e:  # noqa: BLE001
            logger.exception("NLLB model load failed")
            return _degrade(segments, f"model load error: {e}")

        from services.performance_profiles import translation_decode_defaults
        decode_options = translation_decode_defaults()
        beam_count = max(1, int(decode_options.get("num_beams", 1)))
        width = min(_nllb_batch_size(), max(1, _nllb_hypothesis_budget() // beam_count))

        tokenizer.src_lang = flores_src
        results: "list[dict]" = []
        try:
            for start in range(0, len(segments), width):
                batch = segments[start:start + width]
                texts = [s.text for s in batch]
                inputs = tokenizer(texts, return_tensors="pt", padding=True)
                if device != "cpu":
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                forced_bos = tokenizer.convert_tokens_to_ids(flores_tgt)
                tokens = model.generate(
                    **inputs, forced_bos_token_id=forced_bos, max_length=400, **decode_options,
                )
                decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
                for seg, text in zip(batch, decoded):
                    results.append({"id": seg.id, "text": text})
        except Exception as e:  # noqa: BLE001 — one failed batch degrades, doesn't crash the chapter
            logger.warning("NLLB batch failed for this chapter: %s", e)
            release_device_cache(device=device)
            done_ids = {r["id"] for r in results}
            remaining = [s for s in segments if s.id not in done_ids]
            results.extend(_degrade(remaining, str(e)))
        return results

    loop = asyncio.get_running_loop()
    result = await _run_with_timeout(
        loop.run_in_executor(_gpu_pool, _blocking), segments, chapter_translate_timeout_s(), what="NLLB",
    )
    if _should_unload_nllb():
        _unload_nllb()
    return result


async def _translate_argos(
    segments: "list[EbookSegment]", *, source_lang: str, target_lang: str,
) -> "list[dict]":
    from services import translation_engines

    try:
        import argostranslate.translate  # noqa: F401
    except ImportError as e:
        cmd = translation_engines.install_command("argos") or "uv pip install argostranslate"
        raise UnsupportedProviderError(
            f"Argos requires the optional `argostranslate` package. Run `{cmd}` and retry.") from e

    from core.execstack import ensure_ctranslate2_loadable
    try:
        ensure_ctranslate2_loadable()
    except Exception:  # noqa: BLE001 — best-effort repair, the import below is the real check
        pass

    pack_status = translation_engines.argos_pack_status(source_lang, [target_lang])
    missing = [p for p in pack_status["pairs"] if not p["installed"]]
    if missing:
        raise UnsupportedLanguageError(
            f"Install the Argos language pack for {pack_status['source_lang']} -> {target_lang} first.")

    def _blocking() -> "list[dict]":
        import argostranslate.translate as argos_translate

        from_code = pack_status["source_lang"]
        to_code = translation_engines.argos_lang_code(target_lang)
        results = []
        for seg in segments:
            if not seg.text or not seg.text.strip():
                results.append({"id": seg.id, "text": seg.text})
                continue
            try:
                text = seg.text if from_code == to_code else argos_translate.translate(seg.text, from_code, to_code)
                results.append({"id": seg.id, "text": text})
            except Exception as e:  # noqa: BLE001 — degrade this segment, keep going
                results.append({"id": seg.id, "text": seg.text, "error": str(e)})
        return results

    from services.model_manager import _cpu_pool
    loop = asyncio.get_running_loop()
    return await _run_with_timeout(
        loop.run_in_executor(_cpu_pool, _blocking), segments, chapter_translate_timeout_s(), what="Argos",
    )


async def _translate_llm(
    segments: "list[EbookSegment]", *, source_lang: str, target_lang: str,
    glossary: "list[dict] | None", instructions: "str | None",
) -> "list[dict]":
    from api.routers.dub_translate import LANG_NAMES, LANG_REQUIRED_SCRIPT
    from services import llm_skills
    from services.translator import _glossary_text, _looks_like_target_script
    from services.model_manager import _cpu_pool

    handle = llm_skills.resolve_skill_client("ebook_translation")
    if handle is None:
        reason = None
        try:
            reason = llm_skills.resolve_skill("ebook_translation").reason
        except Exception:  # noqa: BLE001
            pass
        if reason == "disabled":
            raise UnsupportedProviderError(
                "The LLM translation engine is turned off — enable the 'Ebook translation' "
                "skill in Settings -> LLM Skills, or pick another provider.")
        raise UnsupportedProviderError(
            "The LLM translation engine has no provider configured. Add and test one in "
            "Settings -> LLM Providers, or pick another provider.")
    client, model_name, llm_timeout = handle.client, handle.model, handle.timeout

    src_name = LANG_NAMES.get(source_lang, source_lang)
    tgt_name = LANG_NAMES.get(target_lang, target_lang)
    script_clause = ""
    script_info = LANG_REQUIRED_SCRIPT.get(target_lang)
    if script_info:
        script_name, _ranges = script_info
        script_clause = (
            f" The output MUST be written in {script_name} script only — do not use "
            f"Latin/Roman letters, do not transliterate, do not output any other language.")
    glossary_clause = _glossary_text(glossary)
    instructions_clause = f"\n\n{instructions.strip()}" if instructions and instructions.strip() else ""
    system_msg = (
        f"You are translating a book chapter. Translate the user's text from {src_name} "
        f"into {tgt_name}.{script_clause} Preserve paragraph meaning and tone; this is prose, "
        f"not dialogue to be shortened. Reply ONLY with the translated {tgt_name} text — no "
        f"quotes, notes, headers, or commentary."
        + (f"\n\n{glossary_clause}" if glossary_clause else "")
        + instructions_clause
    )

    def _one(seg: EbookSegment) -> dict:
        if not seg.text or not seg.text.strip():
            return {"id": seg.id, "text": seg.text}
        last_err = None
        for attempt in range(2):
            sys_for_attempt = system_msg
            if attempt == 1:
                sys_for_attempt += (
                    f" Your previous attempt produced output in the wrong language or "
                    f"script. Output ONLY the {tgt_name} translation.")
            try:
                res = client.chat.completions.create(
                    model=model_name, temperature=0.2, timeout=llm_timeout,
                    messages=[
                        {"role": "system", "content": sys_for_attempt},
                        {"role": "user", "content": seg.text},
                    ],
                )
                from services.llm_backend import _strip_reasoning
                out_text = _strip_reasoning(
                    res.choices[0].message.content or "", prompt=f"{sys_for_attempt}\n{seg.text}")
                if not out_text:
                    last_err = "empty LLM response"
                    continue
                if not _looks_like_target_script(out_text, target_lang):
                    last_err = f"wrong script for {target_lang}"
                    continue
                return {"id": seg.id, "text": out_text}
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
        from core.scrub import scrub_provider_error
        return {"id": seg.id, "text": seg.text, "error": scrub_provider_error(last_err, "") or "llm-failed"}

    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(_cpu_pool, _one, seg) for seg in segments]
    return await _run_with_timeout(asyncio.gather(*tasks), segments, chapter_translate_timeout_s(), what="LLM")


async def _translate_api(
    segments: "list[EbookSegment]", *, source_lang: str, target_lang: str, provider: str,
) -> "list[dict]":
    from api.routers.dub_translate import TRANSLATE_CODES, _translation_output_error
    from services.model_manager import _cpu_pool
    from services.translation_apis import PAID_PROVIDERS, Translator as ApiTranslator
    from services.translation_engines import install_command

    if provider not in PAID_PROVIDERS:
        try:
            import deep_translator  # noqa: F401
        except ImportError as e:
            cmd = install_command(provider) or "uv pip install deep_translator"
            raise UnsupportedProviderError(
                f"The '{provider}' engine needs the optional `deep_translator` package. "
                f"Run `{cmd}` and retry.") from e

    if provider == "amazon":
        from services.translation_apis import AmazonConfigurationError, validate_amazon_configuration
        try:
            await asyncio.to_thread(validate_amazon_configuration)
        except AmazonConfigurationError as e:
            raise UnsupportedProviderError(e.public_message) from e

    src_arg = TRANSLATE_CODES.get(source_lang, source_lang) or "auto"
    tgt_arg = TRANSLATE_CODES.get(target_lang, target_lang)
    direct_proxies = {"http": None, "https": None}

    def _build_translator(src, tgt):
        if provider in PAID_PROVIDERS:
            return ApiTranslator(provider, src, tgt)
        if provider == "mymemory":
            from deep_translator import MyMemoryTranslator
            return MyMemoryTranslator(source=src, target=tgt, proxies=direct_proxies)
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source=src, target=tgt, proxies=direct_proxies)

    def _one(seg: EbookSegment) -> dict:
        if not seg.text or not seg.text.strip():
            return {"id": seg.id, "text": seg.text}
        import time
        last_err = None
        for attempt, src in enumerate([src_arg, src_arg, "auto"]):
            try:
                out = _build_translator(src, tgt_arg).translate(seg.text)
                output_error = _translation_output_error(out)
                if output_error is None:
                    return {"id": seg.id, "text": out}
                last_err = output_error
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(0.25 * (attempt + 1))
        from core.scrub import scrub_provider_error
        return {"id": seg.id, "text": seg.text, "error": scrub_provider_error(last_err, "") or "unknown"}

    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(_cpu_pool, _one, seg) for seg in segments]
    return await _run_with_timeout(asyncio.gather(*tasks), segments, chapter_translate_timeout_s(), what=provider)


async def translate_segments(
    segments: "list[EbookSegment]",
    *,
    source_lang: "str | None",
    target_lang: str,
    provider: str,
    glossary: "list[dict] | None" = None,
    instructions: "str | None" = None,
) -> "list[dict]":
    """Translate one chapter's segments (title + paragraphs) as plain prose.

    Returns one ``{"id", "text", "error"?}`` dict per input segment, same
    shape ``dub_translate`` returns, same degrade-and-continue contract: a
    segment that fails (provider error OR timeout) keeps its SOURCE text with
    an ``error`` field — this function never raises for a per-segment
    failure, only for a request-level setup problem (unknown provider,
    unsupported language, missing optional dependency), which the caller
    (the ``/audiobook/translate`` route) maps to a 4xx before any segment runs.

    No video-timing concept reaches this function: no ``slot_seconds``, no
    duration planning, no speech-rate fit, no dialect/lip-sync handling. If a
    caller needs any of those, it needs ``dub_translate``, not this.
    """
    if not segments:
        return []
    from services import translation_engines

    provider = (provider or "argos").lower()
    if not translation_engines.get_engine(provider):
        raise UnsupportedProviderError(f"Unknown translation provider: {provider!r}")

    resolved_source = resolve_source_lang(segments, source_lang)

    if provider == "nllb":
        return await _translate_nllb(segments, source_lang=resolved_source, target_lang=target_lang)
    if provider == "openai":
        return await _translate_llm(
            segments, source_lang=resolved_source, target_lang=target_lang,
            glossary=glossary, instructions=instructions,
        )
    if provider in ("argos", "libretranslate"):
        return await _translate_argos(segments, source_lang=resolved_source, target_lang=target_lang)
    return await _translate_api(segments, source_lang=resolved_source, target_lang=target_lang, provider=provider)
