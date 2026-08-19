"""Push real speech through the exact live path (soxr stream -> Segmenter ->
model) on several channels at once. Needs a GPU, but no sound card."""
import numpy as np
import pytest
import soxr

from scarlett_live_whisper.segmenter import TARGET_SR, Segmenter, clean

pytestmark = pytest.mark.gpu

DEV_SR = 44100          # what the Scarlett actually runs at
CHANNELS = 8
PLACEMENT = {0: 0.5, 2: 3.0, 5: 1.8}   # channel index -> onset (s)


@pytest.fixture(scope="module")
def board(speech):
    """8-channel 44.1 kHz board with the same utterance on 3 channels."""
    src, sr = speech
    x = soxr.resample(src, sr, DEV_SR, quality="VHQ")
    total = int(DEV_SR * (max(PLACEMENT.values()) + x.size / DEV_SR + 2))
    rng = np.random.default_rng(0)
    buf = rng.normal(0, 3e-5, (total, CHANNELS)).astype(np.float32)
    for ch, t in PLACEMENT.items():
        i = int(t * DEV_SR)
        buf[i:i + x.size, ch] += x
    return buf


@pytest.fixture(scope="module")
def utterances(board):
    """Exactly what transcribe.resample_loop() does, minus the sound card."""
    rs = soxr.ResampleStream(DEV_SR, TARGET_SR, CHANNELS, dtype="float32",
                             quality="VHQ")
    segs = {c: Segmenter() for c in range(CHANNELS)}
    jobs, block = [], 2048
    for i in range(0, board.shape[0] - block, block):
        y = rs.resample_chunk(board[i:i + block])
        if y.shape[0] == 0:
            continue
        for c in range(CHANNELS):
            for start, audio in segs[c].feed(y[:, c]):
                jobs.append((c, start, audio))
    return jobs


def test_only_active_channels_fire(utterances):
    fired = {c for c, _, _ in utterances}
    assert fired == set(PLACEMENT), "a silent channel produced an utterance"


def test_onsets_align(utterances):
    first = {}
    for c, start, _ in sorted(utterances, key=lambda j: j[1]):
        first.setdefault(c, start / TARGET_SR)
    for ch, t in first.items():
        assert abs(t - PLACEMENT[ch]) < 0.5, f"ch{ch + 1} drift {t - PLACEMENT[ch]:+.2f}s"


def test_transcribes_every_channel(utterances, model):
    """SAPI pauses between sentences, so each channel yields 2 utterances."""
    by_ch = {}
    for ch, start, audio in sorted(utterances, key=lambda j: j[1]):
        segments, _ = model.transcribe(
            audio, language="en", beam_size=1, vad_filter=True,
            condition_on_previous_text=False, without_timestamps=True)
        by_ch.setdefault(ch, []).append(clean(" ".join(s.text for s in segments)))

    assert set(by_ch) == set(PLACEMENT)
    for ch, texts in by_ch.items():
        joined = " ".join(texts).lower()
        assert "quick brown fox" in joined, f"ch{ch + 1} missing sentence 1"
        assert "transcription is working" in joined, f"ch{ch + 1} missing sentence 2"


def test_silence_yields_no_text(model):
    """The VAD filter plus the blocklist must keep silence from hallucinating."""
    segments, _ = model.transcribe(
        np.zeros(3 * TARGET_SR, dtype=np.float32), language="en", beam_size=1,
        vad_filter=True, without_timestamps=True)
    assert clean(" ".join(s.text for s in segments)) == ""
