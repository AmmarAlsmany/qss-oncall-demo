"""
Transcription using Groq Whisper API.
Transcribes the full audio in one call, then maps words back to diarized speaker segments.
"""

import os
import librosa
import soundfile as sf
import numpy as np
from groq import Groq


def _to_wav(audio_path: str, tmp_path: str) -> str:
    """Convert any audio format to 16kHz mono WAV for Whisper."""
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    sf.write(tmp_path, audio, sr)
    return tmp_path


def _assign_speaker(start: float, end: float, segments: list[dict]) -> str:
    """Return the speaker label with the most overlap in [start, end]."""
    best_speaker = segments[0]["speaker"] if segments else "SPEAKER_00"
    best_overlap = 0.0
    for seg in segments:
        overlap = max(0.0, min(end, seg["end"]) - max(start, seg["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]
    return best_speaker


def transcribe_segments(audio_path: str, segments: list[dict], tmp_dir: str) -> list[dict]:
    """
    Transcribe audio with Groq Whisper, then merge words into speaker-labeled utterances.

    Returns:
        List of {"speaker", "start", "end", "text"} dicts.
    """
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    client = Groq(api_key=groq_api_key)

    os.makedirs(tmp_dir, exist_ok=True)
    wav_path = os.path.join(tmp_dir, "input_16k.wav")
    _to_wav(audio_path, wav_path)

    with open(wav_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

    try:
        os.remove(wav_path)
    except OSError:
        pass

    words = getattr(response, "words", None) or []

    if not words:
        # Fallback: no word timestamps — assign whole transcript to dominant speaker
        text = (response.text or "").strip()
        if not text:
            return []
        mid = (segments[0]["start"] + segments[-1]["end"]) / 2
        speaker = _assign_speaker(mid, mid, segments)
        return [{"speaker": speaker, "start": segments[0]["start"], "end": segments[-1]["end"], "text": text}]

    # Group consecutive words that belong to the same speaker into utterances
    enriched = []
    current_speaker = None
    current_words = []
    current_start = None

    def _wf(w, key, default=None):
        # Groq returns words as dicts, but be defensive in case the SDK ever returns objects.
        if isinstance(w, dict):
            return w.get(key, default)
        return getattr(w, key, default)

    for w in words:
        w_start = _wf(w, "start", 0.0)
        w_end   = _wf(w, "end",   w_start)
        w_word  = _wf(w, "word",  "")
        speaker = _assign_speaker(w_start, w_end, segments)

        if speaker != current_speaker:
            if current_words and current_speaker is not None:
                enriched.append({
                    "speaker": current_speaker,
                    "start": round(current_start, 3),
                    "end": round(w_start, 3),
                    "text": " ".join(current_words).strip(),
                })
            current_speaker = speaker
            current_words = [w_word]
            current_start = w_start
        else:
            current_words.append(w_word)

    if current_words and current_speaker is not None:
        last_end = _wf(words[-1], "end", current_start)
        enriched.append({
            "speaker": current_speaker,
            "start": round(current_start, 3),
            "end": round(last_end, 3),
            "text": " ".join(current_words).strip(),
        })

    return [e for e in enriched if e["text"]]


def build_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        lines.append(f"{seg['speaker']} [{seg['start']}s - {seg['end']}s]: {seg['text']}")
    return "\n".join(lines)
