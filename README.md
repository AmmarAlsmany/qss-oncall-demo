# QSS OnCall — Airline Customer Service Quality Analyzer

AI-powered analyzer for airline customer service audio recordings. Uploads a call,
identifies the agent and customer, evaluates the agent's performance, and returns a
structured QA report.

## Architecture

```
frontend/index.html        Static HTML/JS UI — upload audio or pick a sample
backend/                   FastAPI service
  main.py                  /analyze, /samples-index endpoints
  diarizer.py              Speaker diarization
  transcriber.py           Speech-to-text
  emotion.py               Per-speaker tone analysis
  analyzer.py              LLM-based QA scoring
  generate_test_audio.py   Generates English + Arabic sample calls
  test_audio/              Generated sample calls (English + Arabic)
```

## Quick start

```bash
# 1. Install Python 3.10+ and ffmpeg
# 2. Create a virtualenv and install deps
python -m venv venv
venv/Scripts/activate            # Windows
# source venv/bin/activate       # Linux / macOS

pip install -r backend/requirements.txt
pip install httpx pyttsx3 gTTS pydub audioop-lts edge-tts   # extras for TTS sample generator

# 3. Set environment variables in backend/.env
cp backend/.env.example backend/.env
# Edit .env and fill in:
#   GROQ_API_KEY  — get one at https://console.groq.com
#   HF_TOKEN      — get one at https://huggingface.co/settings/tokens
#   API_KEY       — optional shared secret for Laravel / external integrations

# 4. (Optional) Generate sample audio for the frontend picker
cd backend
python generate_test_audio.py

# 5. Run the API
uvicorn main:app --host 0.0.0.0 --port 8000

# 6. Open frontend/index.html in a browser
```

## Integration

The `/analyze` endpoint returns a flat, fully-resolved JSON payload designed for
direct ingestion by external dashboards (Laravel, etc.). See `backend/main.py` for
the response schema and supported form fields:

| Field | Type | Description |
|-------|------|-------------|
| `audio` | file | Audio file (mp3/wav/m4a/ogg/flac/mp4) |
| `call_id` | string | Optional caller-supplied call ID, echoed back |
| `webhook_url` | string | Optional URL — receives the result as POST after analysis |
| `metadata` | JSON string | Optional pass-through metadata |
| `X-API-Key` | header | Required if `API_KEY` is set in `.env` |
