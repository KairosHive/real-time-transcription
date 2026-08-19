import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "_speech.wav"

SAPI = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SetOutputToWaveFile("{path}")
$s.Speak("The quick brown fox jumps over the lazy dog. \
Multichannel transcription is working correctly on channel one.")
$s.Dispose()
"""


def _synthesize(path):
    """Generate the speech fixture with Windows SAPI. Returns True on success."""
    if sys.platform != "win32":
        return False
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        SAPI.format(path=path)],
                       check=True, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    return path.exists() and path.stat().st_size > 1000


@pytest.fixture(scope="session")
def speech():
    """Mono float32 speech at its native rate, as (samples, samplerate)."""
    if not FIXTURE.exists() and not _synthesize(FIXTURE):
        pytest.skip("no speech fixture and Windows SAPI unavailable")
    with wave.open(str(FIXTURE), "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


@pytest.fixture(scope="session")
def model():
    """Shared WhisperModel. Skips the test if no CUDA device is present."""
    ct2 = pytest.importorskip("ctranslate2")
    if ct2.get_cuda_device_count() < 1:
        pytest.skip("no CUDA device")
    from faster_whisper import WhisperModel

    from scarlett_live_whisper.transcribe import DEFAULT_MODEL
    return WhisperModel(DEFAULT_MODEL, device="cuda", compute_type="float16",
                        num_workers=3)
