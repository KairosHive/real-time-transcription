"""Real-time parallel Whisper transcription across multi-channel audio interfaces."""

__version__ = "0.1.0"

from .segmenter import FRAME, FRAME_MS, TARGET_SR, Segmenter, clean

__all__ = ["Segmenter", "clean", "TARGET_SR", "FRAME", "FRAME_MS", "__version__"]
