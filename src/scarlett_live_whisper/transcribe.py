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
import warnings

import numpy as np
import soxr

from .config import db, resolve_channels
from .segmenter import TARGET_SR, Segmenter, clean
from .ui import COLORS, DIM, RED, RESET, Console, level_bar

DEFAULT_MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"


def build_parser():
    p = argparse.ArgumentParser(
        prog="scarlett-transcribe",
        description="Transcribe every input channel of a multi-channel "
                    "interface in real time.")
    p.add_argument("--device", default=None,
                   help="device index or name substring (default: auto)")
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--samplerate", type=float, default=None,
                   help="capture rate. ASIO adopts the interface's current "
                        "clock unless you set this (which changes the clock "
                        "for every other app too); WDM defaults to 44100")
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
    p.add_argument("--calibrate", type=float, default=3.0,
                   help="seconds of level measurement before listening (0=off)")
    p.add_argument("--no-meter", action="store_true",
                   help="disable the live level meter")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="log every utterance detected, queued and filtered")
    p.add_argument("--asio", action="store_true",
                   help="capture through the ASIO driver directly, reaching "
                        "every input. The WDM endpoint exposes only 1+2.")
    p.add_argument("--asio-driver", default="focusrite usb",
                   help="ASIO driver name or substring (default: %(default)s)")
    p.add_argument("--asio-buffer", type=int, default=None,
                   help="ASIO buffer size in frames (default: driver's own)")
    return p


def calibration_report(con, peak, rms, channels, active, threshold_db, secs):
    """Show what every channel is actually receiving, selected or not."""
    con.line(f"\n{DIM}--- levels over {secs:.1f}s "
             f"(all {channels} channels) ---{RESET}")
    quiet = []
    for c in range(channels):
        pk, rm = db(peak[c]), db(rms[c])
        sel = "transcribing" if c in active else "ignored"
        if pk < -85:
            verdict, colour = "NO SIGNAL", RED
            quiet.append(c + 1)
        elif rm < threshold_db:
            verdict, colour = "below gate", DIM
            if c in active:
                quiet.append(c + 1)
        else:
            verdict, colour = "above gate", ""
        con.line(f"  ch{c + 1:<2} {level_bar(pk)} peak {pk:>6.1f}  "
                 f"rms {rm:>6.1f} dBFS  {colour}{verdict:<10}{RESET} "
                 f"{DIM}{sel}{RESET}")
    con.line(f"{DIM}gate = {threshold_db:.0f} dBFS rms{RESET}")

    # A driver that advertises more channels than it has pads the extras with
    # an identical silent stream. Real converters never match to 0.01 dB.
    groups = {}
    for c in range(channels):
        if db(peak[c]) < -80:
            groups.setdefault(round(db(rms[c]), 1), []).append(c + 1)
    padding = max(groups.values(), key=len) if groups else []
    if len(padding) >= 3:
        con.warn(f"channels {padding} carry an identical silent stream "
                 f"({round(db(rms[padding[0] - 1]), 1)} dBFS on every one).")
        con.warn("that is driver padding, not real inputs -- this endpoint "
                 "has fewer channels than it advertises. Try --asio.")

    live = [c + 1 for c in active if c + 1 not in quiet]
    if not live:
        con.warn("none of the selected channels showed speech-level signal.")
        con.warn("if you were not talking, this is expected -- otherwise check "
                 "that --only matches the physical inputs, and check gain.")
    elif quiet:
        sel_quiet = [c for c in quiet if c - 1 in active]
        if sel_quiet:
            con.warn(f"selected channels {sel_quiet} were quiet during "
                     f"calibration; only {live} showed signal.")
    con.line()


def main(argv=None):
    a = build_parser().parse_args(argv)
    warnings.filterwarnings("ignore", category=FutureWarning,
                            module="huggingface_hub.*")

    try:
        active, names = resolve_channels(a.only, a.channels, a.names)
    except ValueError as e:
        sys.exit(f"error: {e}")

    con = Console(meter_enabled=not a.no_meter)

    # Two capture backends. ASIO talks to one named driver directly; the WDM
    # path goes through PortAudio, which would instantiate every ASIO driver
    # on the machine and can be brought down by any one of them.
    cap = cands = describe = open_stream = None
    if a.asio:
        from .asio import AsioCapture, find_driver
        try:
            drv = find_driver(a.asio_driver)
            cap = AsioCapture(drv, a.channels, samplerate=a.samplerate,
                              buffer_size=a.asio_buffer).open()
        except RuntimeError as e:
            sys.exit(f"error: {e}")
        stream_rate = cap.samplerate
        if cap.changed_rate:
            con.warn(f"changed the interface clock from "
                     f"{cap.original_rate:g} to {cap.samplerate:g} Hz -- "
                     f"other apps using this card will be affected")
        if not a.names:  # the driver knows what its inputs are called
            names = [cap.channel_names[c] or names[c]
                     for c in range(a.channels)]
    else:
        from .devices import describe, device_candidates, open_stream
        cands = device_candidates(a.device, a.channels)
        stream_rate = a.samplerate or 44100
        if a.verbose:
            con.line(f"{DIM}device candidates, best first: {cands}{RESET}")

    from faster_whisper import WhisperModel
    con.line(f"loading {a.model} ...")
    t_load = time.time()
    model = WhisperModel(a.model, device="cuda", compute_type=a.compute_type,
                         num_workers=a.workers)
    con.line(f"{DIM}model ready in {time.time() - t_load:.1f}s{RESET}")

    jobs = queue.Queue()
    blocks = cap.blocks if cap else queue.Queue(maxsize=64)
    log = open(a.out, "a", encoding="utf-8")
    t0 = time.time()
    # The log is appended across runs and `t` restarts at zero every time, so
    # records need an absolute clock and a session tag to stay orderable.
    session = time.strftime("%Y%m%dT%H%M%S", time.localtime(t0))
    stop = threading.Event()

    peak = np.zeros(a.channels, dtype=np.float64)
    rms = np.zeros(a.channels, dtype=np.float64)
    stats = {"blocks": 0, "dropped": 0, "utts": 0, "done": 0, "filtered": 0,
             "errors": 0, "audio_s": 0.0}
    seg_active = {}

    def capture(indata, frames, tinfo, status):
        if status:
            con.warn(f"audio status: {status}")
        try:
            blocks.put_nowait(indata.copy())
        except queue.Full:
            stats["dropped"] += 1  # drop rather than stall the audio thread

    def resample_loop():
        rs = soxr.ResampleStream(stream_rate, TARGET_SR, a.channels,
                                 dtype="float32", quality="VHQ")
        segs = {c: Segmenter(a.threshold_db, a.hangover, a.min_speech,
                             a.max_utt, a.preroll) for c in active}
        seg_active.update({c: False for c in active})
        cal_peak = np.zeros(a.channels)
        cal_sumsq = np.zeros(a.channels)
        cal_n = 0
        reported = a.calibrate <= 0

        while not stop.is_set():
            try:
                blk = blocks.get(timeout=0.2)
            except queue.Empty:
                continue
            stats["blocks"] += 1
            if cap:
                stats["dropped"] = cap.dropped
            y = rs.resample_chunk(blk)
            if y.shape[0] == 0:
                continue
            stats["audio_s"] += y.shape[0] / TARGET_SR

            # levels for every channel, including ones we do not transcribe
            block_peak = np.abs(y).max(axis=0)
            block_rms = np.sqrt((y.astype(np.float64) ** 2).mean(axis=0))
            np.maximum(peak, block_peak, out=peak)
            rms[:] = 0.8 * rms + 0.2 * block_rms

            if not reported:
                np.maximum(cal_peak, block_peak, out=cal_peak)
                cal_sumsq += block_rms ** 2 * y.shape[0]
                cal_n += y.shape[0]
                if cal_n >= a.calibrate * TARGET_SR:
                    calibration_report(con, cal_peak,
                                       np.sqrt(cal_sumsq / cal_n), a.channels,
                                       active, a.threshold_db, a.calibrate)
                    reported = True

            for c in active:
                for start, audio in segs[c].feed(y[:, c]):
                    stats["utts"] += 1
                    jobs.put((c, start, audio))
                    if a.verbose:
                        con.line(f"{DIM}[seg] ch{c + 1} utterance "
                                 f"{audio.size / TARGET_SR:.1f}s at "
                                 f"{start / TARGET_SR:.1f}s -> queued "
                                 f"(depth {jobs.qsize()}){RESET}")
                seg_active[c] = segs[c].active

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
                stats["errors"] += 1
                con.line(f"{RED}[worker] ch{ch + 1}: {e}{RESET}")
                continue
            stats["done"] += 1
            if not text:
                stats["filtered"] += 1
                if a.verbose:
                    con.line(f"{DIM}[drop] ch{ch + 1} "
                             f"{audio.size / TARGET_SR:.1f}s -> no speech "
                             f"(VAD or hallucination filter){RESET}")
                continue
            epoch = t0 + start / TARGET_SR
            rec = {
                "session": session,
                "channel": ch + 1,
                "name": names[ch],
                "epoch": round(epoch, 3),
                "t": round(start / TARGET_SR, 2),
                "wall": time.strftime("%Y-%m-%d %H:%M:%S",
                                      time.localtime(epoch)),
                "dur": round(audio.size / TARGET_SR, 2),
                "text": text,
            }
            col = COLORS[ch % len(COLORS)]
            clock = rec["wall"].split()[1]
            con.line(f"{col}[{clock}] {rec['name']:<8}{RESET} {text}")
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()

    def monitor():
        warned_silent = False
        while not stop.is_set():
            time.sleep(0.4)
            if stats["blocks"] == 0:
                if time.time() - t0 > 4 and not warned_silent:
                    con.warn("no audio blocks received yet -- the device "
                             "opened but is delivering nothing.")
                    warned_silent = True
                continue
            wide = len(active) > 4
            parts = []
            for c in active:
                d = db(rms[c])
                flag = "*" if seg_active.get(c) else (
                    "+" if d > a.threshold_db else " ")
                col = COLORS[c % len(COLORS)] if seg_active.get(c) else ""
                cell = (f"{c + 1}:{d:>4.0f}{flag}" if wide
                        else f"{c + 1}{level_bar(d, width=4)}{d:>4.0f}{flag}")
                parts.append(f"{col}{cell}{RESET}" if col else cell)
            tail = (f"q{jobs.qsize()} ok{stats['done'] - stats['filtered']} "
                    f"skip{stats['filtered']}")
            if stats["dropped"]:
                tail += f" {RED}drop{stats['dropped']}{RESET}"
            con.meter(f"{DIM}live{RESET} " + " ".join(parts) +
                      f" {DIM}|{RESET} {tail}")

    threads = [threading.Thread(target=resample_loop, daemon=True),
               threading.Thread(target=monitor, daemon=True)]
    threads += [threading.Thread(target=worker, daemon=True)
                for _ in range(a.workers)]
    for t in threads:
        t.start()

    stream = None
    if cap:
        cap.start()
        con.line(f"\nASIO [{drv}] -- {cap.max_inputs} in / "
                 f"{cap.max_outputs} out available")
        con.line(f"  opened {a.channels} ch @ {cap.samplerate:g} Hz "
                 f"-> {TARGET_SR} Hz, buffer {cap.buffer_size} frames")
    else:
        stream, dev = open_stream(cands, a.channels, stream_rate,
                                  a.blocksize, capture)
        con.line(f"\n{describe(dev)}")
        con.line(f"{DIM}  (MME truncates device names to 31 chars; indices "
                 f"shift when other devices appear){RESET}")
        con.line(f"  opened {stream.channels} ch @ {stream.samplerate:g} Hz "
                 f"-> {TARGET_SR} Hz, blocksize {a.blocksize}")
        con.warn("WDM exposes only Analogue 1+2 on a Scarlett; higher "
                 "channels are padding. Use --asio for the rest.")
    con.line(f"  transcribing {[c + 1 for c in active]} "
             f"({', '.join(names[c] for c in active)}) | "
             f"{a.workers} workers | gate {a.threshold_db:g} dBFS")
    con.line(f"{DIM}  meter: '*' = utterance open, '+' = above gate{RESET}")
    con.line("listening -- Ctrl+C to stop")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if cap:
            cap.stop()
        else:
            stream.stop()
            stream.close()
        for t in threads:
            t.join(timeout=3)
        con.close()
        con.line(f"\n{DIM}--- session summary ---{RESET}")
        con.line(f"  audio processed   {stats['audio_s']:.1f}s")
        con.line(f"  utterances found  {stats['utts']}")
        con.line(f"  transcribed       {stats['done'] - stats['filtered']}")
        con.line(f"  filtered as noise {stats['filtered']}")
        if stats["dropped"]:
            knob = "--asio-buffer" if cap else "--blocksize"
            con.warn(f"{stats['dropped']} audio blocks dropped -- that audio "
                     f"was never transcribed. Raise {knob}.")
        if stats["errors"]:
            con.warn(f"{stats['errors']} worker errors")
        if stats["utts"] == 0 and stats["audio_s"] > 0:
            con.warn("no utterances detected. If people were talking, lower "
                     "--threshold-db (try -50) or check the level report above.")
        log.close()
        con.line(f"transcript -> {a.out}")


if __name__ == "__main__":
    main()
