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

Transcriptions will be saved with filenames transcription_{index:02d}.txt. You can customise options:

- `--model`: choose model (tiny, base, small, medium, large, turbo).
- `--osc_ip` and `--osc_port`: configure OSC output.
- `--initial_energy_threshold`, `--initial_record_timeout`, `--initial_phrase_timeout`: tune detection sensitivity.

### Multi-Microphone Support

You can run transcription on multiple microphones simultaneously, with each microphone running in its own process.

**List available microphones:**

    python real-time-transcription.py --list_microphones

This will show all available microphones with their indices:

    Available microphone devices are:
      [0] "Built-in Microphone"
      [1] "USB Audio Device"
      [2] "External Mic"

**Run with multiple microphones:**

Use the `--microphones` argument to specify which microphones to use (by index or partial name match):

    # Using indices
    python real-time-transcription.py --microphones 0 1 2

    # Using names (partial match)
    python real-time-transcription.py --microphones "USB" "Built-in"

    # Mixed
    python real-time-transcription.py --microphones 0 "USB"

Each microphone will:
- Run in a separate process
- Write to its own transcription file (e.g., transcription_00.txt, transcription_01.txt)
- Send OSC messages to incremented ports (e.g., 9000, 9001, 9002)

### System Dependencies

Whisper requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

For more information on Whisper please see https://github.com/openai/whisper

The code in this repository is public domain.
