"""
Speaker Analysis — classifies gender, age type, and tone from audio features.
All heavy imports are lazy (inside functions) to avoid startup crashes.
"""

import os
import logging

logger = logging.getLogger(__name__)

PITCH_CHILD_MIN  = 200
PITCH_FEMALE_MIN = 165

VOICE_PROFILES = {
    ("Male",   "Adult", "Neutral"):  "Hero Male",
    ("Male",   "Adult", "Happy"):    "Comedian Voice",
    ("Male",   "Adult", "Angry"):    "Hero Male",
    ("Male",   "Adult", "Sad"):      "Hero Male",
    ("Female", "Adult", "Neutral"):  "Heroine Female",
    ("Female", "Adult", "Happy"):    "Heroine Female",
    ("Female", "Adult", "Angry"):    "Heroine Female",
    ("Female", "Adult", "Sad"):      "Heroine Female",
    ("Male",   "Child", "Neutral"):  "Child Voice",
    ("Female", "Child", "Neutral"):  "Child Voice",
    ("Male",   "Child", "Happy"):    "Child Voice",
    ("Female", "Child", "Happy"):    "Child Voice",
}

VOICE_MODEL_MAP = {
    "Hero Male":      "tts_models/en/ljspeech/tacotron2-DDC",
    "Heroine Female": "tts_models/en/ljspeech/tacotron2-DDC",
    "Child Voice":    "tts_models/en/ljspeech/tacotron2-DDC",
    "Comedian Voice": "tts_models/en/ljspeech/tacotron2-DDC",
}


def analyze_speaker(audio_path: str) -> dict:
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        logger.error(f"Could not load {audio_path}: {e}")
        return _default_analysis()

    import numpy as np
    import librosa

    pitch_mean, pitch_std = _get_pitch_stats(y, sr, librosa, np)
    speech_rate           = _estimate_speech_rate(y, sr, librosa)
    gender, age_type      = _classify_speaker(pitch_mean, pitch_std)
    tone                  = _detect_tone(y, sr, librosa, np)

    key = (gender, age_type, tone)
    voice_profile = VOICE_PROFILES.get(key) or VOICE_PROFILES.get((gender, age_type, "Neutral"), "Hero Male")
    voice_model   = VOICE_MODEL_MAP.get(voice_profile, VOICE_MODEL_MAP["Hero Male"])

    return {
        "gender":        gender,
        "age_type":      age_type,
        "tone":          tone,
        "pitch_mean":    round(float(pitch_mean), 2),
        "pitch_std":     round(float(pitch_std), 2),
        "speech_rate":   round(float(speech_rate), 3),
        "voice_profile": voice_profile,
        "voice_model":   voice_model,
    }


def _get_pitch_stats(y, sr, librosa, np):
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
        if len(voiced_f0) == 0:
            return 150.0, 20.0
        return float(np.mean(voiced_f0)), float(np.std(voiced_f0))
    except Exception:
        return 150.0, 20.0


def _estimate_speech_rate(y, sr, librosa):
    try:
        onsets   = librosa.onset.onset_detect(y=y, sr=sr)
        duration = len(y) / sr
        return len(onsets) / duration if duration > 0.1 else 3.0
    except Exception:
        return 3.0


def _classify_speaker(pitch_mean, pitch_std):
    if pitch_mean >= PITCH_CHILD_MIN and pitch_std > 25:
        return ("Female" if pitch_mean > 270 else "Male"), "Child"
    elif pitch_mean >= PITCH_FEMALE_MIN:
        return "Female", "Adult"
    return "Male", "Adult"


def _detect_tone(y, sr, librosa, np):
    try:
        rms              = float(np.mean(librosa.feature.rms(y=y)))
        zcr              = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        contrast_mean    = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
        if rms > 0.08 and contrast_mean > 20: return "Angry"
        if rms > 0.05 and zcr > 0.08:         return "Happy"
        if rms < 0.02:                         return "Sad"
        return "Neutral"
    except Exception:
        return "Neutral"


def _default_analysis() -> dict:
    return {
        "gender": "Male", "age_type": "Adult", "tone": "Neutral",
        "pitch_mean": 150.0, "pitch_std": 20.0, "speech_rate": 3.0,
        "voice_profile": "Hero Male", "voice_model": VOICE_MODEL_MAP["Hero Male"],
    }


def analyze_all_speakers(segments: list) -> dict:
    import tempfile
    from pydub import AudioSegment

    speaker_audio = {}
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_00")
        if seg.get("audio_file") and os.path.exists(seg["audio_file"]):
            speaker_audio.setdefault(spk, []).append(seg["audio_file"])

    results = {}
    for speaker, files in speaker_audio.items():
        combined = AudioSegment.empty()
        for f in files[:6]:
            combined += AudioSegment.from_wav(f)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            combined.export(tmp.name, format="wav")
            analysis = analyze_speaker(tmp.name)
            os.unlink(tmp.name)
        results[speaker] = analysis
        logger.info(f"{speaker}: {analysis['gender']} {analysis['age_type']} [{analysis['tone']}] → {analysis['voice_profile']}")
    return results
