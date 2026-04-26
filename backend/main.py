"""
QSS OnCall Demo — FastAPI Backend
Analyzes airline customer service audio conversations.
"""

import os
import certifi

# Fix SSL cert path corrupted by PostgreSQL installation
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

import torch
import torch.serialization

# Fix for PyTorch 2.6+ breaking pyannote model loading
try:
    import torch.torch_version
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
except Exception:
    pass

# Allow all globals as fallback (pyannote models are trusted)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import uuid
import json
import shutil
import tempfile
import httpx
from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from typing import Optional

from diarizer import diarize
from transcriber import transcribe_segments, build_transcript
from emotion import analyze_tone_per_speaker
from analyzer import analyze_conversation

load_dotenv()

# ROOT_PATH lets the app live behind a reverse-proxy prefix (e.g. nginx routing
# qssdemo.shortyhub.com/api -> uvicorn on :8000). Leave empty for local dev.
app = FastAPI(
    title="QSS OnCall Analyzer",
    version="2.0.0",
    root_path=os.getenv("ROOT_PATH", ""),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4"}

# Mount the bundled sample audio folder (if it exists) so the frontend can list/play them
TEST_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "test_audio")
if os.path.isdir(TEST_AUDIO_DIR):
    app.mount("/samples", StaticFiles(directory=TEST_AUDIO_DIR), name="samples")


# ── Friendly metadata for each sample file ─────────────────────────────────
SAMPLE_LABELS = {
    # English
    "excellent_callcenter":       ("Call Center — Excellent",        "Empathetic agent, satisfied customer", "Excellent"),
    "good_checkin_luggage":       ("Check-in — Luggage Weighing",    "Passport, ticket and overweight bag handled professionally", "Good"),
    "good_boarding_gate":         ("Boarding Gate — ID Check",       "Passport and boarding pass scan", "Good"),
    "needs_improvement_luggage":  ("Check-in — Rude Agent",          "Agent dismissive about overweight bag", "Needs Improvement"),
    "poor_callcenter":            ("Call Center — Poor",             "Rude agent, cancelled flight, escalation", "Poor"),
    # Arabic
    "excellent_checkin":          ("تسجيل الدخول — ممتاز",            "موظفة ودودة ومسافر سعيد", "Excellent"),
    "poor_luggage":               ("الأمتعة — سيء",                  "موظف غير لطيف بسبب وزن الحقيبة", "Poor"),
}


@app.get("/samples-index")
def samples_index():
    """List bundled sample audio files grouped by language for the frontend picker."""
    if not os.path.isdir(TEST_AUDIO_DIR):
        return {"languages": {}}

    languages = {}
    for lang in sorted(os.listdir(TEST_AUDIO_DIR)):
        lang_dir = os.path.join(TEST_AUDIO_DIR, lang)
        if not os.path.isdir(lang_dir):
            continue

        files = []
        for fname in sorted(os.listdir(lang_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_FORMATS:
                continue
            stem = os.path.splitext(fname)[0]
            label, description, expected = SAMPLE_LABELS.get(
                stem, (stem.replace("_", " ").title(), "", "")
            )
            files.append({
                "filename": fname,
                "url": f"/samples/{lang}/{fname}",
                "label": label,
                "description": description,
                "expected_verdict": expected,
            })

        if files:
            languages[lang] = files

    return {"languages": languages}


def _verify_api_key(x_api_key: Optional[str]):
    """Reject requests that don't carry a valid API key (if one is configured)."""
    required = os.getenv("API_KEY", "")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


def _build_response(call_id: str, analyzed_at: str, metadata: dict, analysis: dict,
                    transcript: str, labeled_segments: list, tone_per_speaker: dict) -> dict:
    """
    Build a fully resolved, integration-ready response.

    Employee and customer are resolved into nested objects so Laravel never
    needs to cross-reference SPEAKER_XX keys manually.
    """
    emp_id  = analysis.get("employee_speaker", "")
    cust_id = analysis.get("customer_speaker", "")

    emp_tone_data  = tone_per_speaker.get(emp_id, {})
    cust_tone_data = tone_per_speaker.get(cust_id, {})

    employee = {
        "speaker_id": emp_id,
        "identification_reason": analysis.get("employee_identification_reason", ""),
        "tone": analysis.get("employee_tone") or emp_tone_data.get("dominant_tone", "Unknown"),
        "tone_confidence": emp_tone_data.get("confidence", None),
        "top_emotions": emp_tone_data.get("top_emotions", []),
    }

    customer = {
        "speaker_id": cust_id,
        "tone": analysis.get("customer_tone") or cust_tone_data.get("dominant_tone", "Unknown"),
        "tone_confidence": cust_tone_data.get("confidence", None),
        "top_emotions": cust_tone_data.get("top_emotions", []),
    }

    return {
        "success": True,
        "call_id": call_id,
        "analyzed_at": analyzed_at,
        "metadata": metadata,
        "score": analysis.get("score"),
        "verdict": analysis.get("verdict"),
        "summary": analysis.get("summary", ""),
        "employee": employee,
        "customer": customer,
        "strengths": analysis.get("strengths", []),
        "improvements": analysis.get("improvements", []),
        "key_moments": analysis.get("key_moments", []),
        "transcript": transcript,
        "labeled_segments": labeled_segments,
        "tone_per_speaker": tone_per_speaker,
    }


async def _fire_webhook(webhook_url: str, payload: dict):
    """POST result payload to the Laravel webhook URL — best-effort, never blocks the response."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(webhook_url, json=payload)
    except Exception:
        pass


@app.get("/health")
def health():
    return {"status": "ok", "service": "QSS OnCall Analyzer", "version": "2.0.0"}


@app.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    call_id: Optional[str] = Form(None),
    webhook_url: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),  # JSON string e.g. {"agent_id": 42, "call_date": "..."}
    x_api_key: Optional[str] = Header(None),
):
    _verify_api_key(x_api_key)

    call_id     = call_id or str(uuid.uuid4())
    analyzed_at = datetime.now(timezone.utc).isoformat()

    try:
        meta = json.loads(metadata) if metadata else {}
    except (json.JSONDecodeError, TypeError):
        meta = {}

    filename = audio.filename or "audio"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported: {', '.join(SUPPORTED_FORMATS)}",
        )

    groq_api_key = os.getenv("GROQ_API_KEY", "")
    hf_token     = os.getenv("HF_TOKEN", "")

    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in environment.")
    if not hf_token:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set in environment.")

    tmp_dir    = os.path.join(tempfile.gettempdir(), f"qss_{call_id}")
    audio_path = os.path.join(tmp_dir, f"input{ext}")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        # Step 1: Speaker diarization
        try:
            segments = diarize(audio_path, hf_token)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Diarization failed: {str(e)}")

        if not segments:
            raise HTTPException(status_code=422, detail="No speech detected in the audio file.")

        unique_speakers = list({s["speaker"] for s in segments})
        if len(unique_speakers) < 2:
            raise HTTPException(
                status_code=422,
                detail="Only one speaker detected. Please provide a conversation with at least two speakers.",
            )

        # Step 2: Transcription
        segments_dir = os.path.join(tmp_dir, "segments")
        try:
            labeled_segments = transcribe_segments(audio_path, segments, segments_dir)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

        if not labeled_segments:
            raise HTTPException(status_code=422, detail="Could not transcribe any speech from the audio.")

        transcript = build_transcript(labeled_segments)

        # Step 3: Tone / emotion analysis (non-critical — failure is tolerated)
        emotion_dir = os.path.join(tmp_dir, "emotion")
        try:
            tone_per_speaker = analyze_tone_per_speaker(audio_path, labeled_segments, emotion_dir)
        except Exception:
            tone_per_speaker = {}

        # Step 4: Service quality analysis
        try:
            analysis = analyze_conversation(transcript, tone_per_speaker, groq_api_key)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

        response_payload = _build_response(
            call_id=call_id,
            analyzed_at=analyzed_at,
            metadata=meta,
            analysis=analysis,
            transcript=transcript,
            labeled_segments=labeled_segments,
            tone_per_speaker=tone_per_speaker,
        )

        if webhook_url:
            await _fire_webhook(webhook_url, response_payload)

        return JSONResponse(response_payload)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
