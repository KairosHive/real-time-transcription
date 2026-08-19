# scarlett-live-whisper

Real-time parallel Whisper transcription across **every input channel** of a
multi-channel audio interface. Each mic gets its own speaker-separated,
timestamped transcript, live, on a local GPU.

Built and measured on a Focusrite Scarlett 18i20 + RTX 4090 under Windows 11,
but nothing is Focusrite-specific beyond the device auto-pick.

```
[14:22:07] Alice    so if we push the release on Thursday
[14:22:09] Bob      that clashes with the audit
[14:22:11] Room     ...
```

## How it works

```
InputStream (N ch @ 44.1 kHz)
  -> audio callback: copy into a bounded queue, never blocks
  -> resampler thread: soxr N-ch stream -> 16 kHz, split per channel
  -> per-channel energy VAD: pre-roll, hangover, max-length force-split
  -> job queue -> K worker threads sharing ONE WhisperModel (num_workers=K)
  -> colour-coded console line + JSONL record
```

Two design choices carry the whole thing:

**Only complete utterances are transcribed.** A per-channel energy gate with
pre-roll and a silence hangover emits an utterance when someone *stops*
talking. Silent channels cost nothing, which is what makes 8 parallel channels
cheap. The common alternative — re-running Whisper over a rolling buffer every
few seconds — duplicates output and burns GPU on empty channels.

**Timestamps come from an absolute 16 kHz sample counter**, not from wall-clock
time at transcription. Records from different channels therefore sort into one
correct timeline no matter what order the workers finish in.

`vad_filter=True` plus a hallucination blocklist suppress turbo's habit of
emitting "Thank you." / "Sous-titres…" over near-silence.

## Install

Requires Python ≥3.10, an NVIDIA GPU, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/USERNAME/scarlett-live-whisper
cd scarlett-live-whisper
uv sync --extra cuda        # drop --extra cuda if CUDA/cuDNN are system-wide
```

> **Do not build the venv from a conda Python.** ctranslate2 loads alongside
> conda's OpenMP/MKL DLLs and segfaults instantly inside `WhisperModel(...)` —
> on CPU as well as GPU, so it looks like a CUDA problem and isn't. `uv sync`
> uses a standalone CPython and avoids this. If you must use conda, run
> `uv venv --python-preference only-managed`.

## Use

```bash
uv run scarlett-probe                          # list input devices
uv run scarlett-probe --device 2 --seconds 8   # level-check each mic

uv run scarlett-transcribe --channels 8 --language en
```

Named channels, French, only the mics that matter:

```bash
uv run scarlett-transcribe --channels 8 --only 1,2,5 \
    --names Alice,Bob,,,Room --language fr
```

Output streams to the console and appends to `transcript.jsonl`:

```json
{"channel": 1, "name": "Alice", "t": 84.32, "wall": "14:22:07", "dur": 2.9,
 "text": "so if we push the release on Thursday"}
```

Merge into one chronological log afterwards:

```bash
uv run scarlett-merge transcript.jsonl
uv run scarlett-merge transcript.jsonl --channel 2   # just Bob
```

Start with `scarlett-probe` — it prints per-channel peak levels so you can set
`--threshold-db` a few dB above your quietest talking channel.

## Tuning

| Flag | Default | Raise it when |
|---|---|---|
| `--threshold-db` | -42 | noisy room or mic bleed triggers phantom utterances |
| `--hangover` | 0.7 | sentences get chopped mid-thought (costs latency) |
| `--min-speech` | 0.35 | coughs and chair scrapes get transcribed |
| `--max-utt` | 20 | someone monologues; forces a split |
| `--workers` | 3 | measured optimum on a 4090; 6 was slower |
| `--blocksize` | 2048 | the console shows `[audio] input overflow` |

End-to-end latency is roughly `--hangover` + 0.3 s. If you want it snappier,
cut the hangover before you shrink the model.

## Measured performance

RTX 4090, `large-v3-turbo`, float16, 5-second utterances:

| Workers | Throughput | Median latency |
|---|---|---|
| 1 | 43× realtime | 0.09 s |
| **3** | **59× realtime** | 0.25 s |
| 6 | 53× realtime | 0.53 s |

59× realtime means the GPU is nowhere near the bottleneck for 8 channels — the
0.7 s silence hangover dominates. A far smaller GPU will keep up.

## Channel count on Windows

The Focusrite Windows WDM driver exposes **one 8-channel endpoint** (analogue
1–8). On an 18i20, ADAT (9–16) and S/PDIF (17–18) are reachable **only over
ASIO**, and the PortAudio bundled with `sounddevice` is built *without* an ASIO
host API — it offers MME, DirectSound, WASAPI and WDM-KS only. Getting all 18
needs a PortAudio built against the Steinberg ASIO SDK, or routing ADAT/SPDIF
into channels 1–8 in Focusrite Control.

Host API notes, measured:

| Host API | Result |
|---|---|
| MME | opens 8 ch at 44.1 and 48 kHz — the default pick |
| DirectSound | opens 8 ch, also fine |
| WDM-KS | advertises 8 ch, **rejects float32** |
| WASAPI | exposes the endpoint as stereo only |

Device selection tries candidates in that order and falls through on failure,
so a bad first guess is not fatal.

### Sample rate

Whisper wants 16 kHz; the Scarlett is locked to whatever Focusrite Control is
set to (44.1 kHz here) and **cannot be opened at 16 kHz**. Capture runs at the
card's rate and `soxr` resamples. Asking PortAudio for `rate=16000` directly
either fails or gets silently mangled — pass `--samplerate` to match your card.

## Tests

```bash
uv run pytest              # everything
uv run pytest -m "not gpu" # logic only: no GPU, no sound card, runs in CI
```

`tests/test_endtoend.py` places a synthesized utterance on channels 1, 3 and 6
of an 8-channel board at staggered onsets and pushes it through the real
resample → segment → transcribe path. It asserts every active channel is
transcribed, onset timestamps land within 0.5 s, and the five silent channels
never fire. The speech fixture is generated on demand with Windows SAPI.

## License

MIT — see [LICENSE](LICENSE).
