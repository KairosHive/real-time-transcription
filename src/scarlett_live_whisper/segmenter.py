"""Utterance segmentation for one mono 16 kHz stream.

Deliberately free of sounddevice / faster-whisper imports so the segmentation
logic can be tested without an audio device or a GPU.
"""
from collections import deque

import numpy as np

TARGET_SR = 16000
FRAME_MS = 20
FRAME = TARGET_SR * FRAME_MS // 1000  # 320 samples

# large-v3-turbo emits these over silence and room noise. Dropped when they
# are the entire utterance.
HALLUCINATIONS = {
    "thank you", "thanks for watching", "you", "bye", "okay", "",
    "thank you for watching", "please subscribe", "merci",
    "sous-titres réalisés par la communauté d'amara.org",
    "sous-titrage société radio-canada",
    "abonnez-vous", "à bientôt", "音楽",
}


class Segmenter:
    """Energy-gated utterance segmenter for one mono 16 kHz stream.

    Feed it arbitrary-sized blocks; it returns complete utterances only. A
    rolling pre-roll keeps the attack of the first word, and an adaptive noise
    floor keeps a quiet channel from latching open.
    """

    def __init__(self, threshold_db=-42, hangover_s=0.7, min_speech_s=0.35,
                 max_utt_s=20.0, preroll_s=0.3, split_lookback_s=2.0):
        self.thresh = 10 ** (threshold_db / 20)
        self.hangover = hangover_s
        self.min_speech = min_speech_s
        self.max_utt = max_utt_s
        self.lookback = max(1, int(split_lookback_s * 1000 / FRAME_MS))
        self.preroll = deque(maxlen=max(1, int(preroll_s * 1000 / FRAME_MS)))
        self.tail = np.zeros(0, dtype=np.float32)
        self.noise = 1e-4
        self.active = False
        self.buf = []
        self.levels = []      # per-frame rms, parallel to buf
        self.silence = 0.0
        self.speech = 0.0
        self.pos = 0          # absolute sample index at 16 kHz
        self.start = 0        # sample index where current utterance began

    def _emit(self, out):
        if self.speech >= self.min_speech:
            out.append((self.start, np.concatenate(self.buf)))
        self.active = False
        self.buf = []
        self.levels = []
        self.silence = 0.0
        self.speech = 0.0
        self.preroll.clear()

    def _force_split(self, out):
        """Cut a too-long utterance at its quietest recent frame.

        Cutting at an arbitrary sample lands mid-word: Whisper invents an
        ending for the first half and the second half starts mid-syllable, so
        every seam corrupts a few words. Splitting at the quietest frame in
        the recent past lands the cut in a pause instead, and the remainder is
        carried forward rather than dropped, so no audio is ever lost.
        """
        n = len(self.buf)
        lo = max(1, n - self.lookback)
        idx = min(range(lo, n), key=self.levels.__getitem__)
        head = self.buf[:idx]
        if head and self.speech >= self.min_speech:
            out.append((self.start, np.concatenate(head)))
        self.start += idx * FRAME
        self.buf = self.buf[idx:]
        self.levels = self.levels[idx:]
        self.speech = len(self.buf) * FRAME_MS / 1000
        self.silence = 0.0

    def feed(self, x):
        """Push mono float32 samples. Returns list of (start_sample, audio)."""
        out = []
        data = np.concatenate((self.tail, x)) if self.tail.size else x
        n = data.size // FRAME
        for i in range(n):
            f = data[i * FRAME:(i + 1) * FRAME]
            rms = float(np.sqrt(np.mean(f * f)) + 1e-12)
            is_speech = rms > max(self.thresh, self.noise * 3.0)
            if not is_speech:
                self.noise = 0.97 * self.noise + 0.03 * rms

            if not self.active:
                self.preroll.append((f, rms))
                if is_speech:
                    self.active = True
                    # levels must stay parallel to buf, pre-roll included,
                    # or the quiet-point search indexes past its end
                    self.buf = [p[0] for p in self.preroll]
                    self.levels = [p[1] for p in self.preroll]
                    self.start = self.pos - (len(self.buf) - 1) * FRAME
                    self.speech = FRAME_MS / 1000
                    self.silence = 0.0
                    self.preroll.clear()
            else:
                self.buf.append(f)
                self.levels.append(rms)
                if is_speech:
                    self.speech += FRAME_MS / 1000
                    self.silence = 0.0
                else:
                    self.silence += FRAME_MS / 1000
                dur = len(self.buf) * FRAME / TARGET_SR
                if self.silence >= self.hangover:
                    self._emit(out)
                elif dur >= self.max_utt:
                    self._force_split(out)
            self.pos += FRAME

        self.tail = data[n * FRAME:].copy()
        return out


def clean(text):
    """Strip a transcript, returning '' if it is a known hallucination."""
    t = text.strip()
    if t.lower().strip(" .!?,…") in HALLUCINATIONS:
        return ""
    return t
