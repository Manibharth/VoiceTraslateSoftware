"""
Speech-to-Text using OpenAI Whisper (offline).
Each speaker segment is transcribed individually for accuracy.
"""

import os
import logging

logger = logging.getLogger(__name__)

_model_cache = {}


def get_whisper_model(model_size="base"):
    if model_size not in _model_cache:
        try:
            import whisper
            import torch
        except ImportError:
            raise ImportError(
                "openai-whisper is not installed in this Python environment. "
                "Run: venv/bin/pip install openai-whisper"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper '{model_size}' on {device}...")
        _model_cache[model_size] = whisper.load_model(model_size, device=device)
    return _model_cache[model_size]


def transcribe_segment(audio_path, model_size="base", language=None):
    """
    Transcribes a single audio segment.
    Returns {"text": str, "language": str, "confidence": float}
    """
    import torch
    model = get_whisper_model(model_size)

    options = {"task": "transcribe", "fp16": torch.cuda.is_available()}
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)
    text = result.get("text", "").strip()
    detected_lang = result.get("language", "en")
    segments = result.get("segments", [])
    if segments:
        avg_confidence = 1.0 - sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
    else:
        avg_confidence = 0.0 if not text else 0.8

    return {"text": text, "language": detected_lang, "confidence": round(avg_confidence, 3)}


def transcribe_all_segments(segments, model_size="base"):
    """
    Transcribes all speaker segments.
    Returns segments enriched with 'original_text', 'language', 'confidence'.
    """
    enriched = []
    for seg in segments:
        audio_file = seg.get("audio_file")
        if not audio_file or not os.path.exists(audio_file):
            enriched.append({**seg, "original_text": "", "language": "en", "confidence": 0.0})
            continue
        try:
            result = transcribe_segment(audio_file, model_size=model_size)
            enriched.append({**seg, **result, "original_text": result["text"]})
            logger.debug(f"Seg {seg.get('index','?')}: [{seg['speaker']}] {result['text'][:60]}")
        except Exception as e:
            logger.error(f"Transcription failed for segment {seg.get('index')}: {e}")
            enriched.append({**seg, "original_text": "", "language": "en", "confidence": 0.0})
    return enriched
