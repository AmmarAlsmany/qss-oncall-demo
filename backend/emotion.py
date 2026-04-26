"""
Tone / emotion analysis per speaker using a wav2vec2-based emotion recognition model.
Model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
Labels: angry, calm, disgust, fearful, happy, neutral, sad, surprised
"""

import os
import numpy as np
import librosa
import torch
from transformers import pipeline as hf_pipeline
from collections import defaultdict


_classifier = None

EMOTION_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

# Map raw model labels to user-friendly tone descriptions
TONE_MAP = {
    "angry":     "Sharp / Aggressive",
    "calm":      "Calm / Gentle",
    "disgust":   "Irritated",
    "fearful":   "Nervous / Anxious",
    "happy":     "Friendly / Positive",
    "neutral":   "Neutral / Professional",
    "sad":       "Sad / Subdued",
    "surprised": "Surprised",
}


def load_classifier():
    global _classifier
    if _classifier is None:
        device = 0 if torch.cuda.is_available() else -1
        _classifier = hf_pipeline(
            "audio-classification",
            model=EMOTION_MODEL,
            device=device,
        )
    return _classifier


def analyze_emotion_for_file(audio_path: str) -> dict:
    """
    Run emotion classification on an audio file.
    Returns the top label and its score.
    """
    classifier = load_classifier()

    # Load and resample to 16kHz
    audio, sr = librosa.load(audio_path, sr=16000)

    # Model expects numpy array at 16kHz
    result = classifier({"array": audio, "sampling_rate": sr}, top_k=3)
    # result: [{"label": "neutral", "score": 0.8}, ...]
    return result


def analyze_tone_per_speaker(
    audio_path: str,
    segments: list[dict],
    tmp_dir: str,
) -> dict[str, dict]:
    """
    For each speaker, concatenate their audio segments and run emotion analysis.

    Returns:
        {
            "SPEAKER_00": {
                "dominant_tone": "Neutral / Professional",
                "raw_label": "neutral",
                "confidence": 0.82,
                "top_emotions": [{"label": "neutral", "score": 0.82}, ...]
            },
            ...
        }
    """
    import soundfile as sf

    os.makedirs(tmp_dir, exist_ok=True)

    # Group segments by speaker
    speaker_segments = defaultdict(list)
    for seg in segments:
        speaker_segments[seg["speaker"]].append(seg)

    tone_results = {}

    for speaker, segs in speaker_segments.items():
        # Concatenate all audio chunks for this speaker
        chunks = []
        for seg in segs:
            duration = seg["end"] - seg["start"]
            if duration < 0.3:
                continue
            chunk, _ = librosa.load(
                audio_path, sr=16000,
                offset=seg["start"],
                duration=duration,
            )
            chunks.append(chunk)

        if not chunks:
            continue

        combined = np.concatenate(chunks)
        speaker_audio_path = os.path.join(tmp_dir, f"{speaker}_combined.wav")
        sf.write(speaker_audio_path, combined, 16000)

        try:
            top_emotions = analyze_emotion_for_file(speaker_audio_path)
            top = top_emotions[0]
            raw_label = top["label"].lower()
            tone_results[speaker] = {
                "dominant_tone": TONE_MAP.get(raw_label, raw_label.capitalize()),
                "raw_label": raw_label,
                "confidence": round(top["score"], 3),
                "top_emotions": [
                    {
                        "label": TONE_MAP.get(e["label"].lower(), e["label"]),
                        "score": round(e["score"], 3),
                    }
                    for e in top_emotions
                ],
            }
        except Exception as e:
            tone_results[speaker] = {
                "dominant_tone": "Unknown",
                "raw_label": "unknown",
                "confidence": 0.0,
                "top_emotions": [],
                "error": str(e),
            }
        finally:
            try:
                os.remove(speaker_audio_path)
            except OSError:
                pass

    return tone_results
