"""
VoiceTranslate — AI-powered Video & Audio Translation System
Flask backend with speaker diarization, STT, Tamil translation, and TTS.
"""

# ── Auto-relaunch with venv Python if packages are missing ────────────────────
import sys, os

_venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")

def _check_deps():
    missing = []
    for pkg in ["flask", "librosa", "whisper", "soundfile", "anthropic"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing

_missing = _check_deps()
if _missing and os.path.exists(_venv_python) and sys.executable != _venv_python:
    print(f"[VoiceTranslate] Wrong Python detected. Re-launching with venv...")
    print(f"  Current Python : {sys.executable}")
    print(f"  Venv Python    : {_venv_python}")
    os.execv(_venv_python, [_venv_python] + sys.argv)
# ─────────────────────────────────────────────────────────────────────────────

import os
import uuid
import json
import logging
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

app.config.update(
    MAX_CONTENT_LENGTH=500 * 1024 * 1024,  # 500 MB
    UPLOAD_FOLDER="static/uploads",
    OUTPUT_FOLDER="outputs",
    TEMP_FOLDER="temp",
    ALLOWED_EXTENSIONS={"mp4", "mkv", "avi", "mov", "mp3", "wav", "m4a", "ogg", "webm"},
)

for folder in [app.config["UPLOAD_FOLDER"], app.config["OUTPUT_FOLDER"], app.config["TEMP_FOLDER"]]:
    os.makedirs(folder, exist_ok=True)

# ── Database ───────────────────────────────────────────────────────────────────
from database.models import init_db, get_session, Job, Speaker, Segment

engine = init_db()

# ── In-memory job status store (augments DB for real-time progress) ────────────
job_progress: dict[str, dict] = {}


def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lstrip(".").lower()
    return ext in app.config["ALLOWED_EXTENSIONS"]


def is_video(filename: str) -> bool:
    return Path(filename).suffix.lstrip(".").lower() in {"mp4", "mkv", "avi", "mov", "webm"}


# ── API Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    """Upload video or audio file. Returns job_id."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type"}), 400

    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    ext = Path(filename).suffix
    stored_name = f"{job_id}{ext}"
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
    file.save(upload_path)

    # Save to DB
    session = get_session(engine)
    try:
        job = Job(job_id=job_id, filename=filename, status="uploaded")
        session.add(job)
        session.commit()
    finally:
        session.close()

    job_progress[job_id] = {"step": "uploaded", "progress": 5, "message": "File uploaded"}

    return jsonify({
        "job_id": job_id,
        "filename": filename,
        "upload_path": upload_path,
        "is_video": is_video(filename),
    })


@app.route("/api/detect_speakers", methods=["POST"])
def detect_speakers():
    """Run speaker diarization on uploaded file."""
    data = request.get_json()
    job_id = data.get("job_id")
    hf_token = data.get("hf_token") or os.getenv("HF_TOKEN")

    session = get_session(engine)
    try:
        job = session.query(Job).filter_by(job_id=job_id).first()
        if not job:
            return jsonify({"error": "Job not found"}), 404

        # Find uploaded file
        upload_path = _find_upload(job_id)
        if not upload_path:
            return jsonify({"error": "Upload file not found"}), 404

        job.status = "diarizing"
        session.commit()
    finally:
        session.close()

    job_progress[job_id] = {"step": "diarizing", "progress": 15, "message": "Detecting speakers..."}

    # Extract audio if video
    audio_path = _extract_audio_if_video(upload_path, job_id)

    # Reduce noise
    from pipeline.merger import reduce_noise
    audio_path = reduce_noise(audio_path)

    # Run diarization
    from pipeline.diarization import run_diarization, extract_speaker_audio
    segments = run_diarization(audio_path, hf_token=hf_token)

    seg_dir = os.path.join(app.config["TEMP_FOLDER"], job_id, "segments")
    segments = extract_speaker_audio(audio_path, segments, seg_dir)

    # Analyze speakers
    from pipeline.speaker_analysis import analyze_all_speakers
    speaker_profiles = analyze_all_speakers(segments)

    # Save to DB
    session = get_session(engine)
    try:
        # Save speaker profiles
        for spk_label, profile in speaker_profiles.items():
            spk = Speaker(
                job_id=job_id,
                speaker_label=spk_label,
                gender=profile["gender"],
                age_type=profile["age_type"],
                tone=profile["tone"],
                pitch_mean=profile["pitch_mean"],
                pitch_std=profile["pitch_std"],
                speech_rate=profile["speech_rate"],
                voice_profile=profile["voice_profile"],
                voice_model=profile["voice_model"],
            )
            session.add(spk)

        # Save segments
        for seg in segments:
            db_seg = Segment(
                job_id=job_id,
                speaker_label=seg["speaker"],
                start_time=seg["start"],
                end_time=seg["end"],
            )
            session.add(db_seg)

        job = session.query(Job).filter_by(job_id=job_id).first()
        job.status = "diarized"
        session.commit()

        # Store segment data in memory for pipeline
        _store_job_segments(job_id, segments)
        _store_job_speaker_profiles(job_id, speaker_profiles)
    finally:
        session.close()

    job_progress[job_id] = {"step": "diarized", "progress": 30, "message": f"Found {len(speaker_profiles)} speakers"}

    return jsonify({
        "job_id": job_id,
        "speakers": [
            {
                "label": lbl,
                "gender": p["gender"],
                "age_type": p["age_type"],
                "tone": p["tone"],
                "voice_profile": p["voice_profile"],
                "pitch_mean": p["pitch_mean"],
            }
            for lbl, p in speaker_profiles.items()
        ],
        "segment_count": len(segments),
    })


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """Transcribe all speaker segments using Whisper."""
    data = request.get_json()
    job_id = data.get("job_id")
    model_size = data.get("model_size", "base")

    segments = _load_job_segments(job_id)
    if not segments:
        return jsonify({"error": "No segments found. Run detect_speakers first."}), 400

    job_progress[job_id] = {"step": "transcribing", "progress": 40, "message": "Transcribing speech..."}

    from pipeline.transcription import transcribe_all_segments
    segments = transcribe_all_segments(segments, model_size=model_size)
    _store_job_segments(job_id, segments)

    # Update DB
    session = get_session(engine)
    try:
        db_segs = session.query(Segment).filter_by(job_id=job_id).all()
        seg_map = {(round(s.start_time, 1), s.speaker_label): s for s in db_segs}
        for seg in segments:
            key = (round(seg["start"], 1), seg["speaker"])
            if key in seg_map:
                seg_map[key].original_text = seg.get("original_text", "")
                seg_map[key].language = seg.get("language", "en")
        job = session.query(Job).filter_by(job_id=job_id).first()
        if job:
            job.status = "transcribed"
        session.commit()
    finally:
        session.close()

    job_progress[job_id] = {"step": "transcribed", "progress": 55, "message": "Transcription complete"}

    preview = [
        {"speaker": s["speaker"], "start": s["start"], "text": s.get("original_text", "")[:100]}
        for s in segments[:10]
    ]
    return jsonify({"job_id": job_id, "total_segments": len(segments), "preview": preview})


@app.route("/api/translate", methods=["POST"])
def translate():
    """Translate all segments to Tamil casual slang."""
    data = request.get_json()
    job_id = data.get("job_id")

    segments = _load_job_segments(job_id)
    speaker_profiles = _load_job_speaker_profiles(job_id)
    if not segments:
        return jsonify({"error": "No segments found. Run transcribe first."}), 400

    job_progress[job_id] = {"step": "translating", "progress": 60, "message": "Translating to Tamil..."}

    from pipeline.translation import translate_all_segments
    segments = translate_all_segments(segments, speaker_profiles)
    _store_job_segments(job_id, segments)

    # Update DB
    session = get_session(engine)
    try:
        db_segs = session.query(Segment).filter_by(job_id=job_id).all()
        seg_map = {(round(s.start_time, 1), s.speaker_label): s for s in db_segs}
        for seg in segments:
            key = (round(seg["start"], 1), seg["speaker"])
            if key in seg_map:
                seg_map[key].translated_text = seg.get("translated_text", "")
        job = session.query(Job).filter_by(job_id=job_id).first()
        if job:
            job.status = "translated"
        session.commit()
    finally:
        session.close()

    job_progress[job_id] = {"step": "translated", "progress": 70, "message": "Translation complete"}

    preview = [
        {
            "speaker": s["speaker"],
            "original": s.get("original_text", "")[:80],
            "tamil": s.get("translated_text", "")[:80],
        }
        for s in segments[:10]
    ]
    return jsonify({"job_id": job_id, "total_segments": len(segments), "preview": preview})


@app.route("/api/update_voice", methods=["POST"])
def update_voice():
    """User overrides voice profile for a speaker."""
    data = request.get_json()
    job_id = data.get("job_id")
    speaker_label = data.get("speaker_label")
    voice_profile = data.get("voice_profile")

    valid_profiles = ["Hero Male", "Heroine Female", "Child Voice", "Comedian Voice"]
    if voice_profile not in valid_profiles:
        return jsonify({"error": f"Invalid voice profile. Choose from: {valid_profiles}"}), 400

    # Update in-memory profiles
    profiles = _load_job_speaker_profiles(job_id)
    if speaker_label in profiles:
        profiles[speaker_label]["voice_profile"] = voice_profile
        _store_job_speaker_profiles(job_id, profiles)

    # Update DB
    session = get_session(engine)
    try:
        spk = session.query(Speaker).filter_by(job_id=job_id, speaker_label=speaker_label).first()
        if spk:
            spk.override_voice = voice_profile
            session.commit()
    finally:
        session.close()

    return jsonify({"status": "ok", "speaker": speaker_label, "new_voice": voice_profile})


@app.route("/api/generate_voice", methods=["POST"])
def generate_voice():
    """Generate TTS audio for all translated segments."""
    data = request.get_json()
    job_id = data.get("job_id")

    segments = _load_job_segments(job_id)
    speaker_profiles = _load_job_speaker_profiles(job_id)
    if not segments:
        return jsonify({"error": "No segments found. Run translate first."}), 400

    job_progress[job_id] = {"step": "synthesizing", "progress": 75, "message": "Generating voices..."}

    tts_dir = os.path.join(app.config["TEMP_FOLDER"], job_id, "tts")
    seg_dir = os.path.join(app.config["TEMP_FOLDER"], job_id, "segments")

    from pipeline.tts import synthesize_all_segments
    segments = synthesize_all_segments(
        segments, speaker_profiles, tts_dir, original_audio_dir=seg_dir
    )
    _store_job_segments(job_id, segments)

    # Update DB segment audio paths
    session = get_session(engine)
    try:
        db_segs = session.query(Segment).filter_by(job_id=job_id).all()
        seg_map = {(round(s.start_time, 1), s.speaker_label): s for s in db_segs}
        for seg in segments:
            key = (round(seg["start"], 1), seg["speaker"])
            if key in seg_map:
                seg_map[key].audio_path = seg.get("tts_audio_path", "")
        job = session.query(Job).filter_by(job_id=job_id).first()
        if job:
            job.status = "voices_generated"
        session.commit()
    finally:
        session.close()

    job_progress[job_id] = {"step": "voices_generated", "progress": 85, "message": "Voices generated"}
    return jsonify({"job_id": job_id, "segments_synthesized": len(segments)})


@app.route("/api/merge", methods=["POST"])
def merge():
    """Merge all TTS segments with video/audio and generate final output."""
    data = request.get_json()
    job_id = data.get("job_id")
    include_subtitles = data.get("include_subtitles", True)

    segments = _load_job_segments(job_id)
    if not segments:
        return jsonify({"error": "No segments found. Complete pipeline first."}), 400

    job_progress[job_id] = {"step": "merging", "progress": 88, "message": "Merging audio..."}

    upload_path = _find_upload(job_id)
    audio_path = _extract_audio_if_video(upload_path, job_id)

    from pipeline.merger import merge_audio_segments, merge_video_audio, generate_subtitles

    # Merge TTS segments into single audio track
    merged_audio = os.path.join(app.config["TEMP_FOLDER"], job_id, "merged_audio.wav")
    merge_audio_segments(segments, audio_path, merged_audio)

    # Generate subtitles
    srt_path = None
    if include_subtitles:
        srt_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_subtitles.srt")
        generate_subtitles(segments, srt_path)

    # Final output
    if is_video(upload_path):
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_translated.mp4")
        merge_video_audio(upload_path, merged_audio, output_path, srt_path)
    else:
        import shutil
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_translated.wav")
        shutil.copy(merged_audio, output_path)

    # Update DB
    session = get_session(engine)
    try:
        job = session.query(Job).filter_by(job_id=job_id).first()
        if job:
            job.status = "done"
            job.output_path = output_path
            job.subtitle_path = srt_path
        session.commit()
    finally:
        session.close()

    job_progress[job_id] = {"step": "done", "progress": 100, "message": "Translation complete!"}

    return jsonify({
        "job_id": job_id,
        "output_path": output_path,
        "subtitle_path": srt_path,
        "download_url": f"/api/download/{job_id}",
        "subtitle_url": f"/api/download_subtitle/{job_id}" if srt_path else None,
    })


@app.route("/api/process_full", methods=["POST"])
def process_full():
    """
    One-shot endpoint: runs the full pipeline in background.
    Returns job_id immediately; poll /api/status/<job_id> for progress.
    """
    data = request.get_json()
    job_id = data.get("job_id")
    model_size = data.get("model_size", "base")
    hf_token = data.get("hf_token") or os.getenv("HF_TOKEN")
    include_subtitles = data.get("include_subtitles", True)
    target_lang = data.get("target_lang", "ta")
    source_lang = data.get("source_lang", "auto")

    def _run():
        try:
            with app.test_request_context():
                _pipeline_run_full(job_id, model_size, hf_token, include_subtitles,
                                   target_lang=target_lang, source_lang=source_lang)
        except Exception as e:
            logger.exception(f"Full pipeline failed for {job_id}: {e}")
            job_progress[job_id] = {"step": "failed", "progress": 0, "message": str(e)}
            session = get_session(engine)
            try:
                job = session.query(Job).filter_by(job_id=job_id).first()
                if job:
                    job.status = "failed"
                    job.error_message = str(e)
                    session.commit()
            finally:
                session.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "processing", "message": "Pipeline started"})


@app.route("/api/status/<job_id>")
def status(job_id):
    """Real-time job progress."""
    progress = job_progress.get(job_id, {})
    session = get_session(engine)
    try:
        job = session.query(Job).filter_by(job_id=job_id).first()
        db_status = job.status if job else "unknown"
    finally:
        session.close()

    return jsonify({
        "job_id": job_id,
        "db_status": db_status,
        **progress
    })


@app.route("/api/speakers/<job_id>")
def get_speakers(job_id):
    """Get detected speakers and their profiles."""
    session = get_session(engine)
    try:
        speakers = session.query(Speaker).filter_by(job_id=job_id).all()
        return jsonify({
            "job_id": job_id,
            "speakers": [
                {
                    "label": s.speaker_label,
                    "gender": s.gender,
                    "age_type": s.age_type,
                    "tone": s.tone,
                    "voice_profile": s.override_voice or s.voice_profile,
                    "pitch_mean": s.pitch_mean,
                }
                for s in speakers
            ]
        })
    finally:
        session.close()


@app.route("/api/segments/<job_id>")
def get_segments(job_id):
    """Get all translated segments for a job."""
    session = get_session(engine)
    try:
        segs = session.query(Segment).filter_by(job_id=job_id).order_by(Segment.start_time).all()
        return jsonify({
            "job_id": job_id,
            "segments": [
                {
                    "speaker": s.speaker_label,
                    "start": s.start_time,
                    "end": s.end_time,
                    "original": s.original_text,
                    "translated": s.translated_text,
                }
                for s in segs
            ]
        })
    finally:
        session.close()


@app.route("/api/preview_voice/<job_id>/<speaker_label>")
def preview_voice(job_id, speaker_label):
    """Returns a sample TTS audio clip for a speaker."""
    segments = _load_job_segments(job_id)
    profiles = _load_job_speaker_profiles(job_id)

    # Find a segment for this speaker
    speaker_seg = next(
        (s for s in segments if s["speaker"] == speaker_label and s.get("tts_audio_path")),
        None
    )
    if speaker_seg and os.path.exists(speaker_seg["tts_audio_path"]):
        return send_file(speaker_seg["tts_audio_path"], mimetype="audio/wav")

    return jsonify({"error": "Preview not available yet"}), 404


@app.route("/api/download/<job_id>")
def download(job_id):
    """Download final translated video/audio."""
    session = get_session(engine)
    try:
        job = session.query(Job).filter_by(job_id=job_id).first()
        if not job or not job.output_path or not os.path.exists(job.output_path):
            return jsonify({"error": "Output not ready"}), 404
        return send_file(job.output_path, as_attachment=True)
    finally:
        session.close()


@app.route("/api/download_subtitle/<job_id>")
def download_subtitle(job_id):
    """Download SRT subtitle file."""
    session = get_session(engine)
    try:
        job = session.query(Job).filter_by(job_id=job_id).first()
        if not job or not job.subtitle_path or not os.path.exists(job.subtitle_path):
            return jsonify({"error": "Subtitles not ready"}), 404
        return send_file(job.subtitle_path, as_attachment=True)
    finally:
        session.close()


@app.route("/api/languages")
def languages():
    """List supported source and target languages."""
    from pipeline.translation import LANGUAGES
    return jsonify({
        "target_languages": [
            {"code": code, "name": info["name"], "display": info["display"], "flag": info["flag"]}
            for code, info in LANGUAGES.items()
        ],
        "source_languages": [
            {"code": "auto",  "name": "Auto-detect", "flag": "🔍"},
            {"code": "en",    "name": "English",      "flag": "🇬🇧"},
            {"code": "ja",    "name": "Japanese",     "flag": "🇯🇵"},
            {"code": "zh-CN", "name": "Chinese",      "flag": "🇨🇳"},
            {"code": "th",    "name": "Thai",         "flag": "🇹🇭"},
            {"code": "ta",    "name": "Tamil",        "flag": "🇮🇳"},
        ]
    })


@app.route("/api/voice_profiles")
def voice_profiles():
    """List available voice profiles."""
    return jsonify({
        "profiles": [
            {"id": "Hero Male", "description": "Deep strong male voice"},
            {"id": "Heroine Female", "description": "Soft female voice"},
            {"id": "Child Voice", "description": "High-pitched child voice"},
            {"id": "Comedian Voice", "description": "Energetic comedian voice"},
        ]
    })


# ── Internal helpers ───────────────────────────────────────────────────────────

# Simple file-based segment/profile cache (avoids re-doing expensive steps)
_job_segments: dict[str, list] = {}
_job_speaker_profiles: dict[str, dict] = {}


def _store_job_segments(job_id: str, segments: list):
    _job_segments[job_id] = segments
    cache_path = os.path.join(app.config["TEMP_FOLDER"], job_id, "segments.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(segments, f)


def _load_job_segments(job_id: str) -> list:
    if job_id in _job_segments:
        return _job_segments[job_id]
    cache_path = os.path.join(app.config["TEMP_FOLDER"], job_id, "segments.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            _job_segments[job_id] = json.load(f)
        return _job_segments[job_id]
    return []


def _store_job_speaker_profiles(job_id: str, profiles: dict):
    _job_speaker_profiles[job_id] = profiles
    cache_path = os.path.join(app.config["TEMP_FOLDER"], job_id, "profiles.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(profiles, f)


def _load_job_speaker_profiles(job_id: str) -> dict:
    if job_id in _job_speaker_profiles:
        return _job_speaker_profiles[job_id]
    cache_path = os.path.join(app.config["TEMP_FOLDER"], job_id, "profiles.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            _job_speaker_profiles[job_id] = json.load(f)
        return _job_speaker_profiles[job_id]
    return {}


def _find_upload(job_id: str) -> str | None:
    upload_dir = app.config["UPLOAD_FOLDER"]
    for f in os.listdir(upload_dir):
        if f.startswith(job_id):
            return os.path.join(upload_dir, f)
    return None


def _extract_audio_if_video(file_path: str, job_id: str) -> str:
    """Extract audio from video using ffmpeg. Returns audio path."""
    if not is_video(file_path):
        return file_path

    audio_dir = os.path.join(app.config["TEMP_FOLDER"], job_id)
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, "audio.wav")

    if os.path.exists(audio_path):
        return audio_path

    import subprocess
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", file_path, "-vn", "-ar", "16000", "-ac", "1", audio_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr[:200]}")

    return audio_path


def _pipeline_run_full(job_id, model_size, hf_token, include_subtitles,
                       target_lang="ta", source_lang="auto"):
    """Runs full pipeline sequentially (used in background thread)."""
    from pipeline.diarization import run_diarization, extract_speaker_audio
    from pipeline.transcription import transcribe_all_segments
    from pipeline.speaker_analysis import analyze_all_speakers
    from pipeline.translation import translate_all_segments
    from pipeline.tts import synthesize_all_segments
    from pipeline.merger import (merge_audio_segments, merge_video_audio,
                                  generate_subtitles, generate_ass_subtitles, reduce_noise)

    upload_path = _find_upload(job_id)
    audio_path = _extract_audio_if_video(upload_path, job_id)
    audio_path = reduce_noise(audio_path)

    job_progress[job_id] = {"step": "diarizing", "progress": 10, "message": "Detecting speakers..."}
    segments = run_diarization(audio_path, hf_token=hf_token)
    seg_dir = os.path.join(app.config["TEMP_FOLDER"], job_id, "segments")
    segments = extract_speaker_audio(audio_path, segments, seg_dir)

    job_progress[job_id] = {"step": "analyzing", "progress": 25, "message": "Analyzing speakers..."}
    speaker_profiles = analyze_all_speakers(segments)
    _store_job_speaker_profiles(job_id, speaker_profiles)

    job_progress[job_id] = {"step": "transcribing", "progress": 40, "message": "Transcribing speech..."}
    segments = transcribe_all_segments(segments, model_size=model_size)

    from pipeline.translation import LANGUAGES
    lang_name = LANGUAGES.get(target_lang, {}).get("name", target_lang)
    job_progress[job_id] = {"step": "translating", "progress": 58, "message": f"Translating to {lang_name}..."}
    segments = translate_all_segments(segments, speaker_profiles,
                                      source_lang=source_lang, target_lang=target_lang)

    job_progress[job_id] = {"step": "synthesizing", "progress": 72, "message": f"Generating {lang_name} voices..."}
    tts_dir = os.path.join(app.config["TEMP_FOLDER"], job_id, "tts")
    segments = synthesize_all_segments(segments, speaker_profiles, tts_dir,
                                       original_audio_dir=seg_dir, target_lang=target_lang)
    _store_job_segments(job_id, segments)

    job_progress[job_id] = {"step": "merging", "progress": 86, "message": "Mixing audio tracks..."}
    merged_audio = os.path.join(app.config["TEMP_FOLDER"], job_id, "merged_audio.wav")
    merge_audio_segments(segments, audio_path, merged_audio)

    # Generate both SRT and ASS subtitles (ASS has better Tamil rendering)
    srt_path = None
    ass_path = None
    if include_subtitles:
        srt_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_subtitles.srt")
        generate_subtitles(segments, srt_path)
        ass_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_subtitles.ass")
        generate_ass_subtitles(segments, ass_path)

    job_progress[job_id] = {"step": "merging", "progress": 94, "message": "Rendering final video..."}
    if is_video(upload_path):
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_translated.mp4")
        # Prefer ASS for better Tamil subtitle rendering
        subtitle_file = ass_path if ass_path and os.path.exists(ass_path) else srt_path
        merge_video_audio(upload_path, merged_audio, output_path, subtitle_file)
    else:
        import shutil
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_translated.wav")
        shutil.copy(merged_audio, output_path)

    session = get_session(engine)
    try:
        job = session.query(Job).filter_by(job_id=job_id).first()
        if job:
            job.status = "done"
            job.output_path = output_path
            job.subtitle_path = srt_path
        session.commit()
    finally:
        session.close()

    job_progress[job_id] = {"step": "done", "progress": 100, "message": "Translation complete!"}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
