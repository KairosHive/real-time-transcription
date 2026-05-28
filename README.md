# Real Time Whisper Transcription

![Demo GIF](images/demo.gif)

This is a demo of real time speech to text with OpenAI's Whisper model. It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings.

### Environment Setup

Create and activate a Conda environment:

```
conda create -n whisper_env python=3.10 -y
conda activate whisper_env
```

Install PyTorch

Then install the remaining Python dependencies:

```
pip install -r requirements.txt
```

**Note**: Real-time transcription on CPU is practical only with smaller models (`tiny`, `base`, or `small`). Larger models may be too slow without GPU acceleration.

## Usage

Ensure you are in your virtual environment then run:

    python real-time-transcription.py

Transcriptions will be saved in the transcripts/ directory with filenames transcription_{index:02d}.txt. You can customise options:

- --model: choose model (tiny, base, small, medium, large, turbo). By default the English-only (`.en`) variant is used for the smaller models since it is more accurate for English; pass --non_english to keep the multilingual model.
- --language: set the spoken language code (e.g. `en`, `fr`) to skip auto-detection and improve accuracy.
- --initial_prompt: bias the model toward domain vocabulary, names, or spelling (e.g. `"Discussion about Whisper, OSC, and Scarlett 18i20."`).
- --vad_aggressiveness: webrtcvad aggressiveness 0-3 (default 2). Lower values keep more soft/quiet speech.
- --osc_ip and --osc_port: configure OSC output.
- --initial_energy_threshold, --initial_record_timeout to tune mic sensitivity and how often the live partial is refreshed.

#### OSC output and utterance pacing

Two streams are sent on the same client:

- `/transcription <text>` — sent on every refresh (~`--initial_record_timeout` seconds) with the evolving partial text, plus `/trigger 0`. This is the live "I'm listening" feedback; it does **not** mean the phrase is done.
- `/transcription <text>` followed by `/trigger 1` — sent once when an utterance is *finalized*. This is the paced signal for the downstream LLM → txt2img step.

Finalization is **adaptive to the speaker's rhythm** rather than a fixed timeout. The silence needed to end an utterance is learned from the pauses the speaker takes mid-sentence (an EWMA), so a fast talker's short gaps don't chop a thought in two while a slow, deliberate speaker still gets long enough windows. This keeps image changes fluid without flipping too fast. Tuning:

- --min_pause / --initial_phrase_timeout: floor and ceiling (seconds) for the learned end-of-utterance silence.
- --pause_margin: how much longer than a typical mid-sentence pause a silence must be to count as "done" (default 1.6×).
- --min_utterance_duration: minimum spoken seconds before a finalized utterance is allowed to fire `/trigger 1`; shorter fragments keep accumulating instead of changing the image.

Each utterance's audio is buffered and re-transcribed as a whole (rather than stitching together independently transcribed chunks), so Whisper keeps full context and words are not cut at chunk boundaries.

#### Latency

Live partials decode **greedily** (no beam search) so feedback stays snappy; the single final pass that produces the prompt uses beam search for accuracy. If feedback still lags:

- Lower `--initial_record_timeout` for more frequent partial refreshes (more CPU).
- On CPU, use a smaller `--model` (`base` or `small`); `turbo`/`large` need a GPU for real-time partials.
- `--final_beam_size 1` makes the final pass greedy too, cutting GPU load when the trigger feels slow.

#### Backend: faster-whisper

When the GPU is shared with other models (e.g. an LLM, Stable Diffusion, pose detection), use the **faster-whisper** (CTranslate2) backend — it runs the *same* Whisper weights (including `turbo`) noticeably faster and with less VRAM:

```
pip install faster-whisper
```

It is auto-detected (`--backend auto`, the default uses it when installed). Force it with `--backend faster`, or stay on the reference implementation with `--backend openai`. To shrink VRAM further at a small accuracy cost, pass `--compute_type int8_float16` (default is `float16`).

### System Dependencies

Whisper requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

For more information on Whisper please see https://github.com/openai/whisper

The code in this repository is public domain.
