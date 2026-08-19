"""Minimal ASIO capture host, talking to ONE driver directly via ctypes.

PortAudio instantiates every ASIO driver registered on the machine when it
initialises, so a single broken driver takes the process down. This module
CoCreateInstances exactly the driver you name and never touches the others,
which is the only way to reach a Focusrite's full channel count on a machine
where some other ASIO driver misbehaves.

Only capture is implemented; output channels are never created.
"""
import ctypes
import queue
import sys
import threading
import winreg

import numpy as np

# ASIOError
ASE_OK = 0
ASE_SUCCESS = 0x3F4847A0

# ASIOSampleType values we know how to convert
ST_INT16_LSB = 16
ST_INT24_LSB = 17
ST_INT32_LSB = 18
ST_FLOAT32_LSB = 19
ST_FLOAT64_LSB = 20
ST_INT32_LSB16 = 21
ST_INT32_LSB18 = 22
ST_INT32_LSB20 = 23
ST_INT32_LSB24 = 24

_SCALE = {
    ST_INT16_LSB: 1 << 15,
    ST_INT32_LSB: 1 << 31,
    ST_INT32_LSB16: 1 << 15,
    ST_INT32_LSB18: 1 << 17,
    ST_INT32_LSB20: 1 << 19,
    ST_INT32_LSB24: 1 << 23,
}


class GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]


class ASIOBufferInfo(ctypes.Structure):
    _fields_ = [("isInput", ctypes.c_long), ("channelNum", ctypes.c_long),
                ("buffers", ctypes.c_void_p * 2)]


class ASIOChannelInfo(ctypes.Structure):
    _fields_ = [("channel", ctypes.c_long), ("isInput", ctypes.c_long),
                ("isActive", ctypes.c_long), ("channelGroup", ctypes.c_long),
                ("type", ctypes.c_long), ("name", ctypes.c_char * 32)]


class ASIOCallbacks(ctypes.Structure):
    _fields_ = [
        ("bufferSwitch", ctypes.CFUNCTYPE(None, ctypes.c_long, ctypes.c_long)),
        ("sampleRateDidChange", ctypes.CFUNCTYPE(None, ctypes.c_double)),
        ("asioMessage", ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_long,
                                         ctypes.c_long, ctypes.c_void_p,
                                         ctypes.POINTER(ctypes.c_double))),
        ("bufferSwitchTimeInfo", ctypes.CFUNCTYPE(ctypes.c_void_p,
                                                  ctypes.c_void_p,
                                                  ctypes.c_long,
                                                  ctypes.c_long)),
    ]


def list_drivers():
    """Every ASIO driver registered on this machine, by name."""
    out = []
    for hive in (r"SOFTWARE\ASIO", r"SOFTWARE\WOW6432Node\ASIO"):
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, hive) as k:
                for i in range(winreg.QueryInfoKey(k)[0]):
                    name = winreg.EnumKey(k, i)
                    if name not in out:
                        out.append(name)
        except OSError:
            continue
    return out


def _clsid(name):
    for hive in (r"SOFTWARE\ASIO", r"SOFTWARE\WOW6432Node\ASIO"):
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                hive + "\\" + name) as k:
                return winreg.QueryValueEx(k, "CLSID")[0]
        except OSError:
            continue
    return None


def find_driver(substring):
    """Resolve a driver by case-insensitive substring, e.g. 'focusrite usb'."""
    names = list_drivers()
    hits = [n for n in names if substring.lower() in n.lower()]
    if not hits:
        raise RuntimeError(
            f"no ASIO driver matching {substring!r}. Registered: {names}")
    return hits[0]


class AsioCapture:
    """Capture the first `channels` inputs of one ASIO driver.

    Delivers (frames, channels) float32 blocks on `.blocks`, a Queue. The
    driver's realtime thread does one array copy per callback and nothing else.
    """

    def __init__(self, driver_name, channels, samplerate=None,
                 buffer_size=None, queue_size=512):
        if sys.platform != "win32":
            raise RuntimeError("ASIO is Windows-only")
        self.driver_name = driver_name
        self.channels = channels
        self.requested_rate = samplerate
        self.requested_buffer = buffer_size
        self.blocks = queue.Queue(maxsize=queue_size)
        self.dropped = 0
        self._ptr = None
        self._started = False
        self._buffers = None
        self._infos = None
        self._cb = None
        self._lock = threading.Lock()
        self.sample_type = None
        self.buffer_size = None
        self.samplerate = None
        self.original_rate = None
        self.changed_rate = False
        self.channel_names = []

    # -- vtable plumbing -------------------------------------------------
    def _m(self, index, restype, *argtypes):
        vt = ctypes.cast(
            self._ptr,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vt[index])

    def _err(self):
        buf = ctypes.create_string_buffer(128)
        try:
            self._m(6, None, ctypes.c_char_p)(self._ptr, buf)
        except Exception:
            return ""
        return buf.value.decode(errors="replace")

    # -- lifecycle -------------------------------------------------------
    def open(self):
        ole32 = ctypes.windll.ole32
        ole32.CoInitialize(None)
        cls = _clsid(self.driver_name)
        if not cls:
            raise RuntimeError(f"driver {self.driver_name!r} not in registry")
        guid = GUID()
        if ole32.CLSIDFromString(ctypes.c_wchar_p(cls),
                                 ctypes.byref(guid)) != 0:
            raise RuntimeError(f"bad CLSID for {self.driver_name!r}")
        ptr = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(ctypes.byref(guid), None, 1,
                                    ctypes.byref(guid), ctypes.byref(ptr))
        if hr != 0 or not ptr.value:
            raise RuntimeError(
                f"CoCreateInstance failed for {self.driver_name!r}: "
                f"0x{hr & 0xFFFFFFFF:08X}")
        self._ptr = ptr

        if not self._m(3, ctypes.c_long, ctypes.c_void_p)(self._ptr, None):
            msg = self._err()
            self._ptr = None
            raise RuntimeError(f"{self.driver_name}: init failed. {msg}")

        nin, nout = ctypes.c_long(), ctypes.c_long()
        self._m(9, ctypes.c_long, ctypes.POINTER(ctypes.c_long),
                ctypes.POINTER(ctypes.c_long))(
                    self._ptr, ctypes.byref(nin), ctypes.byref(nout))
        self.max_inputs, self.max_outputs = nin.value, nout.value
        if self.channels > self.max_inputs:
            raise RuntimeError(
                f"{self.driver_name} has {self.max_inputs} inputs, "
                f"{self.channels} requested")

        # The clock is shared with every other app using the interface, so
        # never change it implicitly: adopt whatever the driver is already
        # running at unless a rate was explicitly asked for.
        rate = ctypes.c_double()
        get_rate = self._m(13, ctypes.c_long, ctypes.POINTER(ctypes.c_double))
        get_rate(self._ptr, ctypes.byref(rate))
        self.original_rate = rate.value

        if self.requested_rate and \
                float(self.requested_rate) != self.original_rate:
            can = self._m(12, ctypes.c_long, ctypes.c_double)
            if can(self._ptr, float(self.requested_rate)) != ASE_OK:
                raise RuntimeError(
                    f"{self.driver_name} cannot run at "
                    f"{self.requested_rate} Hz (currently "
                    f"{self.original_rate:g} Hz)")
            self._m(14, ctypes.c_long, ctypes.c_double)(
                self._ptr, float(self.requested_rate))
            self.changed_rate = True
            get_rate(self._ptr, ctypes.byref(rate))
        self.samplerate = rate.value

        mn, mx, pref, gran = (ctypes.c_long(), ctypes.c_long(),
                              ctypes.c_long(), ctypes.c_long())
        self._m(11, ctypes.c_long, *([ctypes.POINTER(ctypes.c_long)] * 4))(
            self._ptr, ctypes.byref(mn), ctypes.byref(mx),
            ctypes.byref(pref), ctypes.byref(gran))
        self.buffer_size = int(self.requested_buffer or pref.value)

        get_ci = self._m(18, ctypes.c_long, ctypes.POINTER(ASIOChannelInfo))
        self.channel_names = []
        for c in range(self.channels):
            ci = ASIOChannelInfo(channel=c, isInput=1)
            get_ci(self._ptr, ctypes.byref(ci))
            self.channel_names.append(ci.name.decode(errors="replace"))
            if self.sample_type is None:
                self.sample_type = ci.type
        if self.sample_type not in _SCALE and \
                self.sample_type not in (ST_FLOAT32_LSB, ST_FLOAT64_LSB):
            raise RuntimeError(f"unsupported ASIO sample type "
                               f"{self.sample_type}")
        return self

    def _make_callbacks(self):
        n, size = self.channels, self.buffer_size
        stype = self.sample_type
        infos = self._infos

        if stype == ST_FLOAT32_LSB:
            dtype, scale = np.float32, None
        elif stype == ST_FLOAT64_LSB:
            dtype, scale = np.float64, None
        elif stype == ST_INT16_LSB:
            dtype, scale = np.int16, float(_SCALE[stype])
        elif stype == ST_INT24_LSB:
            dtype, scale = np.uint8, float(1 << 23)
        else:
            dtype, scale = np.int32, float(_SCALE[stype])

        itemsize = 3 if stype == ST_INT24_LSB else np.dtype(dtype).itemsize
        nbytes = size * itemsize

        def buffer_switch(index, direct):
            try:
                out = np.empty((size, n), dtype=np.float32)
                for c in range(n):
                    addr = infos[c].buffers[index]
                    raw = (ctypes.c_char * nbytes).from_address(addr)
                    if stype == ST_INT24_LSB:
                        b = np.frombuffer(raw, dtype=np.uint8).reshape(size, 3)
                        v = (b[:, 0].astype(np.int32)
                             | (b[:, 1].astype(np.int32) << 8)
                             | (b[:, 2].astype(np.int8).astype(np.int32) << 16))
                        out[:, c] = v / scale
                    else:
                        v = np.frombuffer(raw, dtype=dtype, count=size)
                        out[:, c] = v if scale is None else v / scale
                try:
                    self.blocks.put_nowait(out)
                except queue.Full:
                    self.dropped += 1
            except Exception:
                self.dropped += 1  # never raise into the driver's RT thread

        def sample_rate_changed(rate):
            self.samplerate = rate

        def asio_message(selector, value, message, opt):
            # kAsioSelectorSupported=1, kAsioEngineVersion=3
            if selector == 1:
                return 1 if value in (1, 3) else 0
            if selector == 3:
                return 2
            return 0

        def buffer_switch_time_info(params, index, direct):
            buffer_switch(index, direct)
            return None

        cb = ASIOCallbacks()
        cb.bufferSwitch = type(cb.bufferSwitch)(buffer_switch)
        cb.sampleRateDidChange = type(cb.sampleRateDidChange)(
            sample_rate_changed)
        cb.asioMessage = type(cb.asioMessage)(asio_message)
        cb.bufferSwitchTimeInfo = type(cb.bufferSwitchTimeInfo)(
            buffer_switch_time_info)
        return cb

    def start(self):
        infos = (ASIOBufferInfo * self.channels)()
        for c in range(self.channels):
            infos[c].isInput = 1
            infos[c].channelNum = c
        self._infos = infos
        self._cb = self._make_callbacks()

        rc = self._m(19, ctypes.c_long, ctypes.POINTER(ASIOBufferInfo),
                     ctypes.c_long, ctypes.c_long,
                     ctypes.POINTER(ASIOCallbacks))(
            self._ptr, infos, self.channels, self.buffer_size,
            ctypes.byref(self._cb))
        if rc != ASE_OK:
            raise RuntimeError(f"createBuffers failed: {rc} {self._err()}")

        rc = self._m(7, ctypes.c_long)(self._ptr)
        if rc != ASE_OK:
            self._m(20, ctypes.c_long)(self._ptr)
            raise RuntimeError(f"start failed: {rc} {self._err()}")
        self._started = True
        return self

    def stop(self):
        with self._lock:
            if self._ptr is None:
                return
            if self._started:
                try:
                    self._m(8, ctypes.c_long)(self._ptr)      # stop
                    self._m(20, ctypes.c_long)(self._ptr)     # disposeBuffers
                except Exception:
                    pass
                self._started = False
            try:
                self._m(2, ctypes.c_ulong)(self._ptr)         # Release
            except Exception:
                pass
            self._ptr = None
            self._infos = None
            self._cb = None

    def __enter__(self):
        return self.open().start()

    def __exit__(self, *exc):
        self.stop()


def get_samplerate(driver_substring="focusrite usb"):
    """Current clock of an ASIO driver, without disturbing it."""
    cap = AsioCapture(find_driver(driver_substring), channels=1)
    try:
        cap.open()
        return cap.samplerate
    finally:
        cap.stop()


def set_samplerate(rate, driver_substring="focusrite usb"):
    """Set the interface clock. Returns (previous, new).

    The clock is shared by every application using the interface, so changing
    it will disrupt anything currently playing at a different rate. Only ever
    call this when the user explicitly asked for it.
    """
    cap = AsioCapture(find_driver(driver_substring), channels=1)
    try:
        cap.open()
        before = cap.samplerate
        can = cap._m(12, ctypes.c_long, ctypes.c_double)
        if can(cap._ptr, float(rate)) != ASE_OK:
            raise RuntimeError(f"driver cannot run at {rate} Hz")
        cap._m(14, ctypes.c_long, ctypes.c_double)(cap._ptr, float(rate))
        got = ctypes.c_double()
        cap._m(13, ctypes.c_long, ctypes.POINTER(ctypes.c_double))(
            cap._ptr, ctypes.byref(got))
        return before, got.value
    finally:
        cap.stop()
