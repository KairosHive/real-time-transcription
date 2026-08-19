"""Terminal output: a live level meter that transcript lines print over."""
import re
import sys
import threading

ANSI = re.compile(r"\033\[[0-9;]*m")

# Redirected stdout on Windows is cp1252, which cannot encode block glyphs.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def _encodable(text):
    try:
        text.encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


BLOCKS = " ▁▂▃▄▅▆▇█"
if not _encodable(BLOCKS):
    BLOCKS = " .:-=+*#%"

COLORS = ["\033[36m", "\033[32m", "\033[33m", "\033[35m",
          "\033[34m", "\033[91m", "\033[92m", "\033[95m"]
DIM = "\033[2m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def visible_len(s):
    return len(ANSI.sub("", s))


class Console:
    """Prints scrolling lines above a single redrawn meter line."""

    def __init__(self, meter_enabled=True):
        self.lock = threading.Lock()
        self.meter_enabled = meter_enabled and sys.stdout.isatty()
        self._width = 0

    def _erase(self):
        if self._width:
            sys.stdout.write("\r" + " " * self._width + "\r")
            self._width = 0

    def line(self, text=""):
        """Print a permanent line, temporarily clearing the meter."""
        with self.lock:
            self._erase()
            sys.stdout.write(text + "\n")
            sys.stdout.flush()

    def warn(self, text):
        self.line(f"{YELLOW}! {text}{RESET}")

    def meter(self, text):
        if not self.meter_enabled:
            return
        with self.lock:
            self._erase()
            sys.stdout.write(text)
            self._width = visible_len(text)
            sys.stdout.flush()

    def close(self):
        with self.lock:
            self._erase()
            sys.stdout.flush()


def level_bar(dbfs, lo=-60.0, hi=0.0, width=8):
    """Block-character bar for a dBFS value."""
    blocks = BLOCKS
    frac = (dbfs - lo) / (hi - lo)
    filled = max(0.0, min(1.0, frac)) * width
    out = []
    for i in range(width):
        rem = filled - i
        if rem >= 1:
            out.append(blocks[-1])
        elif rem <= 0:
            out.append(blocks[0])
        else:
            out.append(blocks[int(rem * (len(blocks) - 1))])
    return "".join(out)
