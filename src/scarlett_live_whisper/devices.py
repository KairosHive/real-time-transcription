"""Input device discovery and stream opening (PortAudio / WDM backend)."""
import sys

import sounddevice as sd

# ASIO first when it is available: it is the only host API that exposes the
# full channel count of an 18i20. WDM-KS advertises the Focusrite endpoint but
# refuses float32, so it goes last.
API_ORDER = ("ASIO", "MME", "Windows DirectSound", "Windows WASAPI",
             "Windows WDM-KS")


def list_inputs():
    """Print every input device with its host API and channel count."""
    ha = sd.query_hostapis()
    print(f"{'idx':>4} {'host api':<20} {'ch':>3} {'rate':>7}  name")
    print("-" * 78)
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(
                f"{i:>4} {ha[d['hostapi']]['name']:<20} "
                f"{d['max_input_channels']:>3} "
                f"{int(d['default_samplerate']):>7}  {d['name'][:40]}"
            )
    names = [h["name"] for h in ha]
    print()
    print("host APIs available:", ", ".join(names))
    if not any("ASIO" in n for n in names):
        print("NOTE: ASIO is not loaded. The WDM endpoint exposes only 8 of")
        print("      an 18i20's inputs. Re-run with --asio for all 20.")


def device_candidates(spec, need_channels):
    """Ordered list of device indices worth trying.

    `spec` is a device index, a name substring, or None for automatic choice.
    """
    devs = sd.query_devices()
    ha = sd.query_hostapis()
    usable = [i for i, d in enumerate(devs)
              if d["max_input_channels"] >= need_channels]
    if spec is not None:
        try:
            return [int(spec)]
        except ValueError:
            hits = [i for i in usable if spec.lower() in devs[i]["name"].lower()]
            if not hits:
                sys.exit(f"no input device matching {spec!r} with "
                         f">= {need_channels} channels")
            return hits

    def rank(i):
        api = ha[devs[i]["hostapi"]]["name"]
        name = devs[i]["name"].lower()
        is_focusrite = "focusrite" in name or "analogue" in name
        return (0 if is_focusrite else 1,
                API_ORDER.index(api) if api in API_ORDER else len(API_ORDER))

    if not usable:
        sys.exit(f"no input device with >= {need_channels} channels -- is the "
                 f"interface powered on? run scarlett-probe to see what is there")
    return sorted(usable, key=rank)


def open_stream(candidates, channels, samplerate, blocksize, callback):
    """Try each candidate until one opens. Returns (started_stream, index)."""
    errors = []
    for dev in candidates:
        try:
            stream = sd.InputStream(device=dev, channels=channels,
                                    samplerate=samplerate, dtype="float32",
                                    blocksize=blocksize, callback=callback)
            stream.start()
            return stream, dev
        except Exception as e:
            errors.append(f"  dev {dev}: {str(e).splitlines()[0]}")
    sys.exit("could not open any input device:\n" + "\n".join(errors))


def describe(dev):
    """Human-readable 'idx [host api] name' for a device index."""
    info = sd.query_devices(dev)
    api = sd.query_hostapis(info["hostapi"])["name"]
    return f"device {dev} [{api}] {info['name']}"
