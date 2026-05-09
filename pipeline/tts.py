"""
Text-to-Speech — gTTS (Google) as primary for Tamil clarity.
Coqui TTS used only for non-Tamil fallback.
Applies per-profile pitch/speed with librosa for voice matching.
"""

import os
import logging
import subprocess

logger = logging.getLogger(__name__)

VOICE_PARAMS = {
    "Hero Male":      {"speed": 0.95, "pitch_shift": -3, "energy": 1.2},
    "Heroine Female": {"speed": 1.0,  "pitch_shift":  2, "energy": 1.0},
    "Child Voice":    {"speed": 1.05, "pitch_shift":  5, "energy": 1.0},
    "Comedian Voice": {"speed": 1.08, "pitch_shift":  1, "energy": 1.15},
}


def synthesize_segment(text: str, output_path: str, voice_profile: str = "Hero Male",
                        language: str = "ta", speaker_wav: str = None) -> str:
    if not text or not text.strip():
        _generate_silence(output_path)
        return output_path

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    params = VOICE_PARAMS.get(voice_profile, VOICE_PARAMS["Hero Male"])

    # gTTS is primary for Tamil — best free quality
    result = _gtts_synthesize(text, output_path, params, lang=language)
    return result


def _gtts_synthesize(text: str, output_path: str, params: dict, lang: str = "ta") -> str:
    try:
        from gtts import gTTS
        tmp_mp3 = output_path.replace(".wav", "_raw.mp3")
        tmp_wav = output_path.replace(".wav", "_raw.wav")

        gTTS(text=text, lang=lang, slow=False).save(tmp_mp3)

        # mp3 → 44100 Hz stereo WAV for high quality
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3, "-ar", "44100", "-ac", "2", tmp_wav],
            capture_output=True
        )
        for f in [tmp_mp3]:
            if os.path.exists(f): os.remove(f)

        _apply_voice_effects(tmp_wav, output_path, params)
        if os.path.exists(tmp_wav) and tmp_wav != output_path:
            os.remove(tmp_wav)

        return output_path
    except Exception as e:
        logger.error(f"gTTS failed: {e}")
        _generate_silence(output_path)
        return output_path


def _apply_voice_effects(input_path: str, output_path: str, params: dict):
    """Apply pitch shift and speed change cleanly via ffmpeg + librosa."""
    try:
        import librosa
        import soundfile as sf
        import numpy as np

        y, sr = librosa.load(input_path, sr=None, mono=False)
        if y.ndim > 1:
            y = y.mean(axis=0)  # stereo → mono

        speed = params.get("speed", 1.0)
        pitch = params.get("pitch_shift", 0)
        energy = params.get("energy", 1.0)

        if abs(speed - 1.0) > 0.02:
            y = librosa.effects.time_stretch(y, rate=speed)
        if pitch != 0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
        if abs(energy - 1.0) > 0.02:
            y = np.clip(y * energy, -1.0, 1.0)

        # Normalize to -3 dBFS for clarity
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.707

        sf.write(output_path, y, sr)
    except Exception as e:
        logger.warning(f"Voice effects skipped, copying raw: {e}")
        import shutil
        shutil.copy(input_path, output_path)


def _generate_silence(output_path: str, duration: float = 0.3, sr: int = 22050):
    try:
        import numpy as np
        import soundfile as sf
        sf.write(output_path, np.zeros(int(sr * duration)), sr)
    except Exception:
        pass


def synthesize_all_segments(segments: list, speaker_profiles: dict,
                             output_dir: str, original_audio_dir: str = None,
                             target_lang: str = "ta") -> list:
    os.makedirs(output_dir, exist_ok=True)
    enriched = []
    total = len(segments)
    for i, seg in enumerate(segments):
        text          = seg.get("translated_text", "").strip()
        speaker       = seg.get("speaker", "SPEAKER_00")
        idx           = seg.get("index", 0)
        profile       = speaker_profiles.get(speaker, {})
        voice_profile = seg.get("override_voice") or profile.get("voice_profile", "Hero Male")
        lang          = seg.get("target_lang", target_lang)
        out_path      = os.path.join(output_dir, f"tts_{idx:04d}_{speaker}.wav")

        if text:
            synthesize_segment(text=text, output_path=out_path,
                               voice_profile=voice_profile, language=lang)
            logger.info(f"[{i+1}/{total}] TTS [{voice_profile}] [{lang}]: {text[:50]}")
        else:
            _generate_silence(out_path)

        enriched.append({**seg, "tts_audio_path": out_path})
    return enriched
