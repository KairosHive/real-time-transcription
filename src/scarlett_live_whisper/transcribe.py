"""Real-time parallel Whisper transcription of every input channel of a
multi-channel interface (built for a Focusrite Scarlett 18i20 on Windows).

Pipeline:
    InputStream (N ch @ device rate)
      -> audio callback (copy only, never blocks)
      -> resampler thread: soxr N-ch -> 16 kHz, split into per-channel streams
      -> per-channel energy VAD segmenter -> complete utterances
      -> job queue -> K worker threads sharing one faster-whisper model
      -> stamped console output + JSONL log
"""
import argparse
import json
import queue
import sys
import threading
import time

import soxr

from .devices import describe, device_candidates, open_stream
from .segmenter import TARGET_SR, Segmenter, clean

DEFAULT_MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"

COLORS = ["\033[36m", "\033[32m", "\033[33m", "\033[35m",
          "\033[34m", "\033[91m", "\033[92m", "\033[95m"]
RESET = "\033[0m"


def build_parser():
    p = argparse.ArgumentParser(
        prog="scarlett-transcribe",
        description="Transcribe every input channel of a multi-channel "
                    "interface in real time.")
    p.add_argument("--device", default=None,
                   help="device index or name substring (default: auto)")
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--samplerate", type=float, default=44100,
                   help="must match the rate set in Focusrite Control")
    p.add_argument("--blocksize", type=int, default=2048,
                   help="raise this if you see input overflow")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--language", default="en", help="'en', 'fr', ... or 'auto'")
    p.add_argument("--names", default=None,
                   help="comma-separated channel labels, e.g. Alice,Bob,Room")
    p.add_argument("--only", default=None,
                   help="comma-separated 1-based channels to transcribe")
    p.add_argument("--threshold-db", type=float, default=-42,
                   help="speech gate, dBFS RMS per 20 ms frame")
    p.add_argument("--hangover", type=float, default=0.7,
                   help="silence needed to close an utterance (s)")
    p.add_argument("--min-speech", type=float, default=0.35)
    p.add_argument("--max-utt", type=float, default=20.0)
    p.add_argument("--preroll", type=float, default=0.3)
    p.add_argument("--out", default="transcript.jsonl")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)

    cands = device_candidates(a.device, a.channels)
    active = ([int(c) - 1 for c in a.only.split(",")] if a.only
              else list(range(a.channels)))
    names = (a.names.split(",") if a.names
             else [f"ch{c + 1}" for c in range(a.channels)])
    names += [f"ch{c + 1}" for c in range(len(names), a.channels)]
    names = [n if n.strip() else f"ch{i + 1}" for i, n in enumerate(names)]

    from faster_whisper import WhisperModel
    print(f"loading {a.model} ...", flush=True)
    model = WhisperModel(a.model, device="cuda", compute_type=a.compute_type,
                         num_workers=a.workers)

    jobs = queue.Queue()
    blocks = queue.Queue(maxsize=64)
    out_lock = threading.Lock()
    log = open(a.out, "a", encoding="utf-8")
    t0 = time.time()
    stop = threading.Event()

    def capture(indata, frames, tinfo, status):
        if status:
            print(f"  [audio] {status}", file=sys.stderr)
        try:
            blocks.put_nowait(indata.copy())
        except queue.Full:
            pass  # drop rather than stall the audio thread

    def resample_loop():
        rs = soxr.ResampleStream(a.samplerate, TARGET_SR, a.channels,
                                 dtype="float32", quality="VHQ")
        segs = {c: Segmenter(a.threshold_db, a.hangover, a.min_speech,
                             a.max_utt, a.preroll) for c in active}
        while not stop.is_set():
            try:
                blk = blocks.get(timeout=0.2)
            except queue.Empty:
                continue
            y = rs.resample_chunk(blk)
            if y.shape[0] == 0:
                continue
            for c in active:
                for start, audio in segs[c].feed(y[:, c]):
                    jobs.put((c, start, audio))

    def worker():
        while not stop.is_set():
            try:
                ch, start, audio = jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                segments, _ = model.transcribe(
                    audio,
                    language=None if a.language == "auto" else a.language,
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                )
                text = clean(" ".join(s.text for s in segments))
            except Exception as e:
                print(f"  [worker] {e}", file=sys.stderr)
                continue
            if not text:
                continue
            rec = {
                "channel": ch + 1,
                "name": names[ch],
                "t": round(start / TARGET_SR, 2),
                "wall": time.strftime("%H:%M:%S",
                                      time.localtime(t0 + start / TARGET_SR)),
                "dur": round(audio.size / TARGET_SR, 2),
                "text": text,
            }
            col = COLORS[ch % len(COLORS)]
            with out_lock:
                print(f"{col}[{rec['wall']}] {rec['name']:<8}{RESET} {text}",
                      flush=True)
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log.flush()

    threads = [threading.Thread(target=resample_loop, daemon=True)]
    threads += [threading.Thread(target=worker, daemon=True)
                for _ in range(a.workers)]
    for t in threads:
        t.start()

    stream, dev = open_stream(cands, a.channels, a.samplerate, a.blocksize,
                              capture)
    print(describe(dev))
    print(f"{a.channels} ch @ {a.samplerate:g} Hz -> {TARGET_SR} Hz | "
          f"transcribing {[c + 1 for c in active]} | {a.workers} workers")
    print("listening -- Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        stream.stop()
        stream.close()
        stop.set()
        for t in threads:
            t.join(timeout=2)
        log.close()
        print(f"transcript -> {a.out}")


if __name__ == "__main__":
    main()
