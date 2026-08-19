"""List input devices and test-open one, showing per-channel peak levels."""
import argparse
import sys

import numpy as np


def level_test(device, channels, samplerate, seconds):
    import sounddevice as sd

    from .devices import describe
    print(describe(device))
    print(f"opening {channels} ch @ {samplerate:g} Hz ...")
    peaks = np.zeros(channels, dtype=np.float32)

    def cb(indata, frames, t, status):
        if status:
            print("  status:", status, file=sys.stderr)
        np.maximum(peaks, np.abs(indata).max(axis=0), out=peaks)

    with sd.InputStream(device=device, channels=channels,
                        samplerate=samplerate, dtype="float32", callback=cb):
        print(f"recording {seconds:g}s -- speak or tap into each mic now")
        sd.sleep(int(seconds * 1000))

    print("\nper-channel peak level:")
    for c in range(channels):
        db = 20 * np.log10(max(peaks[c], 1e-9))
        bar = "#" * int(max(0, (db + 60) / 60 * 40))
        print(f"  ch {c + 1:>2}  {db:>7.1f} dBFS  {bar}")
    dead = [c + 1 for c in range(channels) if peaks[c] < 1e-5]
    if dead:
        print(f"\nsilent channels: {dead}  (nothing connected, or gain down)")
    print("\npick --threshold-db a few dB above the quietest talking channel.")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="scarlett-probe",
        description="List audio inputs, and optionally level-check a device.")
    p.add_argument("--device", type=int, help="device index to test-open")
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--samplerate", type=float, default=44100)
    p.add_argument("--seconds", type=float, default=8)
    p.add_argument("--asio", action="store_true",
                   help="load the ASIO-enabled PortAudio build")
    a = p.parse_args(argv)

    if a.asio:
        from .devices import enable_asio
        enable_asio()
    from .devices import list_inputs

    list_inputs()
    if a.device is not None:
        print()
        level_test(a.device, a.channels, a.samplerate, a.seconds)


if __name__ == "__main__":
    main()
