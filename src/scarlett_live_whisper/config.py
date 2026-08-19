"""Argument resolution that is worth testing on its own."""


def resolve_channels(only, channels, names):
    """Return (active_indices, labels) for --only / --names.

    `only` is a comma-separated 1-based list or None for every channel.
    Blank entries in `names` fall back to chN. Raises ValueError with a
    message meant to be shown to the user.
    """
    if channels < 1:
        raise ValueError("--channels must be at least 1")

    labels = names.split(",") if names else []
    labels += [""] * (channels - len(labels))
    if names and len(names.split(",")) > channels:
        raise ValueError(
            f"--names has {len(names.split(','))} entries but only "
            f"{channels} channels are open")
    labels = [n.strip() if n.strip() else f"ch{i + 1}"
              for i, n in enumerate(labels[:channels])]

    if not only:
        return list(range(channels)), labels

    active = []
    for part in only.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            n = int(part)
        except ValueError:
            raise ValueError(f"--only got {part!r}, expected a channel number")
        if not 1 <= n <= channels:
            raise ValueError(
                f"--only asks for channel {n}, but only {channels} channels "
                f"are open (valid: 1-{channels}). Raise --channels, or pick "
                f"channels within range.")
        if n - 1 not in active:
            active.append(n - 1)
    if not active:
        raise ValueError("--only selected no channels")
    return sorted(active), labels


def db(x):
    """Amplitude to dBFS, floored so silence prints as -inf-ish, not a crash."""
    import math
    return 20 * math.log10(max(float(x), 1e-9))


def enable_portaudio_asio():
    """Make sounddevice load the ASIO-enabled PortAudio DLL.

    Must run before anything imports sounddevice, which is why it lives here
    rather than in devices.py -- importing that module would itself pull
    sounddevice in and make the call a no-op.

    Note this routes through PortAudio, which instantiates EVERY registered
    ASIO driver at startup; one bad driver crashes the process. The asio
    module talks to a single driver directly and avoids that.
    """
    import os
    import sys
    if "sounddevice" in sys.modules:
        raise RuntimeError("enable_portaudio_asio() called too late -- "
                           "sounddevice is already imported")
    os.environ["SD_ENABLE_ASIO"] = "1"
