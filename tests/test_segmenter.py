"""Pure-logic tests: no audio device, no GPU, no model."""
import numpy as np
import pytest

from scarlett_live_whisper.segmenter import FRAME, TARGET_SR, Segmenter, clean


def _feed(seg, sig, block):
    """Push a signal through in fixed-size blocks, collecting utterances."""
    out = []
    for i in range(0, sig.size - block, block):
        out += seg.feed(sig[i:i + block])
    return out


@pytest.fixture
def bursts():
    """1s silence, 2s speech, 1.5s silence, 1s speech, 1s silence."""
    rng = np.random.default_rng(0)
    sil = lambda n: rng.normal(0, 1e-4, int(n * TARGET_SR)).astype(np.float32)
    spk = lambda n: rng.normal(0, 0.05, int(n * TARGET_SR)).astype(np.float32)
    return np.concatenate([sil(1), spk(2), sil(1.5), spk(1), sil(1)])


@pytest.mark.parametrize("block", [743, 1024, 2048, FRAME])
def test_splits_into_two_utterances(bursts, block):
    """Block size must not change segmentation - ragged sizes included."""
    got = _feed(Segmenter(), bursts, block)
    assert len(got) == 2


def test_onset_includes_preroll(bursts):
    got = _feed(Segmenter(preroll_s=0.3), bursts, 1024)
    start = got[0][0] / TARGET_SR
    # speech begins at 1.0s; pre-roll pulls the mark ~0.3s earlier
    assert 0.6 < start < 1.05


def test_duration_covers_speech_plus_hangover(bursts):
    got = _feed(Segmenter(hangover_s=0.7, preroll_s=0.3), bursts, 1024)
    dur = got[0][1].size / TARGET_SR
    assert 2.5 < dur < 3.2  # 0.3 pre-roll + 2.0 speech + 0.7 hangover


def test_silence_never_fires():
    rng = np.random.default_rng(3)
    quiet = rng.normal(0, 3e-5, 10 * TARGET_SR).astype(np.float32)
    assert _feed(Segmenter(), quiet, 2048) == []


def test_short_blip_rejected():
    """A 0.1s tap is below --min-speech and must not produce an utterance."""
    rng = np.random.default_rng(4)
    sig = np.concatenate([
        rng.normal(0, 1e-4, TARGET_SR).astype(np.float32),
        rng.normal(0, 0.05, TARGET_SR // 10).astype(np.float32),
        rng.normal(0, 1e-4, 2 * TARGET_SR).astype(np.float32),
    ])
    assert _feed(Segmenter(min_speech_s=0.35), sig, 1024) == []


def test_max_utt_force_splits():
    rng = np.random.default_rng(1)
    sig = rng.normal(0, 0.05, 12 * TARGET_SR).astype(np.float32)
    got = _feed(Segmenter(max_utt_s=4.0), sig, 1024)
    assert len(got) >= 2
    assert all(a.size / TARGET_SR <= 4.1 for _, a in got)


def test_timestamps_are_monotonic(bursts):
    got = _feed(Segmenter(), bursts, 1024)
    starts = [s for s, _ in got]
    assert starts == sorted(starts)


@pytest.mark.parametrize("text,expected", [
    ("  Thank you. ", ""),
    ("Merci", ""),
    ("thanks for watching!", ""),
    ("Hello there", "Hello there"),
    ("  spaced  ", "spaced"),
])
def test_clean(text, expected):
    assert clean(text) == expected
