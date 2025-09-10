# Real Time Whisper Transcription

![Demo GIF](images/demo.gif)

This is a demo of real time speech to text with OpenAI's Whisper model. It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings.

### Environment Setup

Create and activate a Conda environment:

```
conda create -n whisper_env python=3.9 -y
conda activate whisper_env
```

Install PyTorch (CPU version) via Conda:

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Then install the remaining Python dependencies:

```
pip install -r requirements.txt
```

**Note**: Real-time transcription on CPU is practical only with smaller models (`tiny`, `base`, or `small`). Larger models may be too slow without GPU acceleration.

## Usage

Ensure you are in your virtual environment then run:

    python real-time-transcription.py

Transcriptions will be saved in the transcripts/ directory with filenames transcription_{index:02d}.txt. You can customise options:

- --model: choose model (tiny, base, small, medium, large, turbo).
- --osc_ip and --osc_port: configure OSC output.
- --initial_energy_threshold, --initial_record_timeout, --initial_phrase_timeout to tune detection sensitivity.

### System Dependencies

Whisper requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

For more information on Whisper please see https://github.com/openai/whisper

The code in this repository is public domain.
