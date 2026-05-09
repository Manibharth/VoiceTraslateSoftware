"""
Speaker Diarization — detects and segments different speakers in audio.
Primary: pyannote.audio (offline)
Fallback: energy-based VAD segmentation
All heavy imports are lazy (inside functions) to avoid startup crashes.
"""

import os
import logging

logger = logging.getLogger(__name__)


def run_diarization(audio_path: str, hf_token: str = None) -> list:
    try:
        return _pyannote_diarize(audio_path, hf_token)
    except Exception as e:
        logger.warning(f"pyannote failed ({e}), trying Resemblyzer...")
        try:
            return _resemblyzer_diarize(audio_path)
        except Exception as e2:
            logger.warning(f"Resemblyzer failed ({e2}), using VAD fallback...")
            return _vad_fallback(audio_path)


def _pyannote_diarize(audio_path: str, hf_token: str = None) -> list:
    from pyannote.audio import Pipeline
    import torch

    token = hf_token or os.getenv("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(torch.device(device))
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3)
        })
    logger.info(f"pyannote found {len(set(s['speaker'] for s in segments))} speakers")
    return segments


def _resemblyzer_diarize(audio_path: str) -> list:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from sklearn.cluster import AgglomerativeClustering
    import librosa
    import numpy as np

    encoder = VoiceEncoder()
    wav, sr = librosa.load(audio_path, sr=16000)
    wav = preprocess_wav(wav)

    window = int(1.5 * 16000)
    hop = int(0.75 * 16000)
    frames, timestamps = [], []
    for i in range(0, len(wav) - window, hop):
        frames.append(wav[i:i + window])
        timestamps.append(i / 16000)

    if not frames:
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": len(wav) / sr}]

    embeds = encoder.embed_batch(frames)
    n_speakers = min(4, len(frames) // 2 or 1)
    labels = AgglomerativeClustering(n_clusters=n_speakers).fit_predict(embeds)

    segments, prev_label, seg_start = [], labels[0], timestamps[0]
    for label, ts in zip(labels[1:], timestamps[1:]):
        if label != prev_label:
            segments.append({"speaker": f"SPEAKER_{prev_label:02d}", "start": round(seg_start, 3), "end": round(ts, 3)})
            prev_label, seg_start = label, ts
    segments.append({"speaker": f"SPEAKER_{prev_label:02d}", "start": round(seg_start, 3), "end": round(len(wav) / 16000, 3)})
    logger.info(f"Resemblyzer found {n_speakers} speakers")
    return segments


def _vad_fallback(audio_path: str) -> list:
    import librosa
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    segments, t = [], 0.0
    while t < duration:
        end = min(t + 3.0, duration)
        segments.append({"speaker": "SPEAKER_00", "start": round(t, 3), "end": round(end, 3)})
        t = end
    logger.warning("VAD fallback: treating all audio as single speaker")
    return segments


def extract_speaker_audio(audio_path: str, segments: list, output_dir: str) -> list:
    from pydub import AudioSegment
    os.makedirs(output_dir, exist_ok=True)
    audio = AudioSegment.from_file(audio_path)
    enriched = []
    for i, seg in enumerate(segments):
        chunk = audio[int(seg["start"] * 1000):int(seg["end"] * 1000)]
        seg_path = os.path.join(output_dir, f"seg_{i:04d}_{seg['speaker']}.wav")
        chunk.export(seg_path, format="wav")
        enriched.append({**seg, "audio_file": seg_path, "index": i})
    return enriched
