"""
Audio/Video Merger — high-quality mixing with Tamil subtitle burn-in.
"""

import os
import logging
import subprocess

logger = logging.getLogger(__name__)

TAMIL_FONT = "/System/Library/Fonts/Supplemental/Tamil Sangam MN.ttc"
FALLBACK_FONT = "/System/Library/Fonts/Helvetica.ttc"


def merge_audio_segments(segments: list, original_audio_path: str,
                          output_audio_path: str, preserve_background: bool = True) -> str:
    import numpy as np
    import librosa
    import soundfile as sf

    TARGET_SR = 44100
    orig_y, sr = librosa.load(original_audio_path, sr=TARGET_SR, mono=True)
    output_y = np.zeros(len(orig_y), dtype=np.float32)

    # Very subtle background ambience
    if preserve_background:
        output_y += orig_y * 0.05

    for seg in segments:
        tts_path = seg.get("tts_audio_path")
        if not tts_path or not os.path.exists(tts_path):
            continue

        start_sec = seg.get("start", 0.0)
        end_sec   = seg.get("end", start_sec + 1.0)
        orig_dur  = end_sec - start_sec

        try:
            tts_y, tts_sr = librosa.load(tts_path, sr=TARGET_SR, mono=True)
        except Exception as e:
            logger.warning(f"Cannot load {tts_path}: {e}")
            continue

        if len(tts_y) == 0:
            continue

        tts_dur = len(tts_y) / TARGET_SR

        # Only time-stretch if significantly off (>30%) and not too extreme
        if orig_dur > 0.2:
            ratio = tts_dur / orig_dur
            if ratio > 1.3:
                # TTS is longer — compress slightly, allow overflow
                try:
                    tts_y = librosa.effects.time_stretch(tts_y, rate=min(ratio, 1.8))
                except Exception:
                    pass

        # Place at timestamp
        start_s = int(start_sec * TARGET_SR)
        chunk   = tts_y
        end_s   = min(start_s + len(chunk), len(output_y))
        avail   = end_s - start_s
        if avail > 0:
            output_y[start_s:end_s] += chunk[:avail] * 0.92

    # Final normalize to -1 dBFS
    peak = np.max(np.abs(output_y))
    if peak > 0:
        output_y = output_y / peak * 0.891

    os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
    sf.write(output_audio_path, output_y, TARGET_SR)
    logger.info(f"Merged audio → {output_audio_path}")
    return output_audio_path


def merge_video_audio(original_video_path: str, translated_audio_path: str,
                       output_video_path: str, subtitle_path: str = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

    if subtitle_path and os.path.exists(subtitle_path):
        font = TAMIL_FONT if os.path.exists(TAMIL_FONT) else FALLBACK_FONT
        # Escape path for ffmpeg filter
        escaped_sub = subtitle_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        vf = (
            f"subtitles='{escaped_sub}'"
            f":force_style='FontName=Tamil Sangam MN,"
            f"FontSize=22,PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,Outline=2,"
            f"Shadow=1,Alignment=2,MarginV=30'"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", original_video_path,
            "-i", translated_audio_path,
            "-vf", vf,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-shortest",
            output_video_path
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", original_video_path,
            "-i", translated_audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-shortest",
            output_video_path
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg stderr: {result.stderr[-400:]}")
        # Retry without subtitles
        if subtitle_path:
            logger.warning("Retrying without subtitle burn-in...")
            return merge_video_audio(original_video_path, translated_audio_path, output_video_path, None)
        raise RuntimeError(f"Video merge failed: {result.stderr[-200:]}")

    logger.info(f"Final video → {output_video_path}")
    return output_video_path


def generate_subtitles(segments: list, output_srt_path: str) -> str:
    """Generate UTF-8 SRT with Tamil text."""
    os.makedirs(os.path.dirname(os.path.abspath(output_srt_path)), exist_ok=True)
    lines = []
    idx = 1
    for seg in segments:
        text = seg.get("translated_text", "").strip()
        if not text:
            continue
        start = _sec_to_srt(seg["start"])
        end   = _sec_to_srt(seg["end"])
        spk   = seg.get("speaker", "")
        lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
        idx += 1

    with open(output_srt_path, "w", encoding="utf-8-sig") as f:  # utf-8-sig adds BOM for compatibility
        f.write("\n".join(lines))

    logger.info(f"Subtitles → {output_srt_path} ({idx-1} entries)")
    return output_srt_path


def generate_ass_subtitles(segments: list, output_ass_path: str) -> str:
    """Generate ASS subtitle file with Tamil font embedded — better rendering than SRT."""
    os.makedirs(os.path.dirname(os.path.abspath(output_ass_path)), exist_ok=True)

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Tamil,Tamil Sangam MN,26,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2.5,1.5,2,20,20,35,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for seg in segments:
        text = seg.get("translated_text", "").strip()
        if not text:
            continue
        start = _sec_to_ass(seg["start"])
        end   = _sec_to_ass(seg["end"])
        # Escape special ASS chars
        safe = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        events.append(f"Dialogue: 0,{start},{end},Tamil,,0,0,0,,{safe}")

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")

    logger.info(f"ASS subtitles → {output_ass_path} ({len(events)} entries)")
    return output_ass_path


def _sec_to_srt(s: float) -> str:
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s%1)*1000):03d}"


def _sec_to_ass(s: float) -> str:
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{int(h):01d}:{int(m):02d}:{s:05.2f}"


def reduce_noise(audio_path: str, output_path: str = None) -> str:
    if output_path is None:
        output_path = audio_path
    try:
        import noisereduce as nr
        import librosa
        import soundfile as sf
        y, sr = librosa.load(audio_path, sr=None)
        reduced = nr.reduce_noise(y=y, sr=sr, y_noise=y[:int(sr * 0.5)])
        sf.write(output_path, reduced, sr)
        logger.info(f"Noise reduced → {output_path}")
    except Exception as e:
        logger.warning(f"Noise reduction skipped: {e}")
    return output_path
