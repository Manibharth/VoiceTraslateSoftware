"""
Multi-language Translation Pipeline.
Primary: Google Translate (free, no key) via deep-translator.
Optional: Gemini API for casual slang style (if quota available).
Supports: Tamil, English, Japanese, Chinese (Simplified), Thai.
"""

import os
import time
import logging
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# ── Language Config ────────────────────────────────────────────────────────────
LANGUAGES = {
    "ta": {"name": "Tamil",            "gtts": "ta",    "display": "தமிழ்",          "flag": "🇮🇳"},
    "en": {"name": "English",          "gtts": "en",    "display": "English",         "flag": "🇬🇧"},
    "ja": {"name": "Japanese",         "gtts": "ja",    "display": "日本語",           "flag": "🇯🇵"},
    "zh-CN": {"name": "Chinese",       "gtts": "zh-CN", "display": "中文 (简体)",      "flag": "🇨🇳"},
    "th": {"name": "Thai",             "gtts": "th",    "display": "ภาษาไทย",         "flag": "🇹🇭"},
}

# Slang/casual suffixes and filler words per target language
CASUAL_HINTS = {
    "ta": "Use casual Tamil slang: add 'da', 'di', 'pa', 'macha', 'aiyyo', 'dei' appropriately. Spoken street style, NOT formal.",
    "ja": "Use casual Japanese: add 'ne', 'yo', 'na', 'ze' appropriately. Informal spoken style, NOT formal.",
    "zh-CN": "Use casual Mandarin: informal spoken style, not formal written Chinese.",
    "th": "Use casual Thai: add 'นะ', 'ครับ'/'ค่ะ', 'เลย' appropriately. Informal spoken style.",
    "en": "Use casual spoken English. Short, punchy sentences.",
}


def translate_text(text: str, source_lang: str = "en", target_lang: str = "ta") -> str:
    """
    Translates text from source to target language.
    Tries Gemini first (casual slang), falls back to Google Translate.
    """
    if not text or not text.strip():
        return ""
    if source_lang == target_lang:
        return text

    # Try Gemini for natural slang style
    result = _try_gemini(text, source_lang, target_lang)
    if result and result != text:
        return result

    # Google Translate (always works, free)
    return _google_translate(text, source_lang, target_lang)


def _try_gemini(text: str, src: str, tgt: str) -> str:
    """Attempt Gemini translation with casual slang style."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return ""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        tgt_name = LANGUAGES.get(tgt, {}).get("name", tgt)
        src_name = LANGUAGES.get(src, {}).get("name", src)
        casual   = CASUAL_HINTS.get(tgt, "")

        prompt = (
            f"Translate from {src_name} to {tgt_name}.\n"
            f"{casual}\n"
            f"Return ONLY the translation, no explanation.\n"
            f'Text: "{text}"'
        )

        for model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"]:
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(max_output_tokens=256, temperature=0.6),
                )
                result = r.text.strip()
                if result:
                    logger.debug(f"Gemini [{model}] {src}→{tgt}: '{text[:30]}' → '{result[:30]}'")
                    return result
            except Exception as e:
                err = str(e)
                if "429" in err or "503" in err:
                    continue
                break
    except Exception:
        pass
    return ""


def _google_translate(text: str, src: str, tgt: str) -> str:
    """Google Translate via deep-translator (free, no key needed)."""
    try:
        from deep_translator import GoogleTranslator
        # deep-translator source 'auto' for auto-detect
        translator = GoogleTranslator(source=src if src != "auto" else "auto", target=tgt)
        result = translator.translate(text)
        logger.debug(f"GoogleTranslate {src}→{tgt}: '{text[:30]}' → '{(result or '')[:30]}'")
        return result or text
    except Exception as e:
        logger.error(f"Google Translate failed: {e}")
        return text


def translate_all_segments(segments: list, speaker_profiles: dict,
                            source_lang: str = "auto", target_lang: str = "ta") -> list:
    """
    Translates all segments from source to target language.
    Returns segments enriched with 'translated_text' and 'target_lang'.
    """
    enriched = []
    total = len(segments)

    for i, seg in enumerate(segments):
        text    = seg.get("original_text", "").strip()
        src     = seg.get("language", source_lang) if source_lang == "auto" else source_lang

        if text:
            translated = translate_text(text, source_lang=src, target_lang=target_lang)
            # Small delay to avoid rate limiting
            if (i + 1) % 5 == 0:
                time.sleep(0.3)
        else:
            translated = ""

        logger.info(f"[{i+1}/{total}] {src}→{target_lang}: {text[:35]} → {translated[:35]}")
        enriched.append({**seg, "translated_text": translated, "target_lang": target_lang})

    return enriched
