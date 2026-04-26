"""
Speaker diarization using pyannote.audio.
Splits audio into segments labeled by speaker (SPEAKER_00, SPEAKER_01, etc.)
"""

import os
import sys
import types

# SpeechBrain 1.x uses a lazy-module system for optional integrations.
# When inspect.stack() traverses sys.modules it triggers hasattr() on lazy modules,
# which causes them to actually import — failing on Windows where k2/nlp/huggingface
# deps are absent. Pre-load the real modules that exist, and alias the missing ones
# to speechbrain.inference (the new home of speechbrain.pretrained).
import importlib as _importlib

# Load real speechbrain.inference first so aliases point to something valid
_importlib.import_module("speechbrain.inference")

# Alias deprecated speechbrain.pretrained → speechbrain.inference
if "speechbrain.pretrained" not in sys.modules:
    _sb_inference = sys.modules["speechbrain.inference"]
    sys.modules["speechbrain.pretrained"] = _sb_inference

# Stub out Windows-unavailable optional integration submodules
_MISSING_SB_MODS = [
    "speechbrain.integrations.k2_fsa",
    "speechbrain.k2_integration",
    "speechbrain.integrations.nlp",
    "speechbrain.integrations.huggingface",
    "speechbrain.integrations.huggingface.wordemb",
]
for _mod in _MISSING_SB_MODS:
    if _mod not in sys.modules:
        _stub = types.ModuleType(_mod)
        _stub.__file__ = None
        _stub.__path__ = []
        _stub.__spec__ = None
        sys.modules[_mod] = _stub

import torch
import torch.serialization
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Specifications
from pyannote.audio.core.model import Model

try:
    torch.serialization.add_safe_globals([Specifications, Model])
except Exception:
    pass


_pipeline = None


def load_pipeline(hf_token: str) -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _original_load = torch.load
        torch.load = lambda *a, **kw: _original_load(*a, **{**kw, "weights_only": False})
        try:
            _pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
        finally:
            torch.load = _original_load
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _pipeline = _pipeline.to(device)
    return _pipeline


def diarize(audio_path: str, hf_token: str) -> list[dict]:
    """
    Run speaker diarization on an audio file.

    Returns a list of segments:
        [{"speaker": "SPEAKER_00", "start": 0.5, "end": 3.2}, ...]
    """
    pipeline = load_pipeline(hf_token)
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
        })

    # Sort by start time
    segments.sort(key=lambda s: s["start"])
    return segments
