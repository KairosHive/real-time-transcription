"""ASIO registry/driver-resolution tests. No hardware required."""
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="ASIO is Windows-only")


def test_list_drivers_returns_names():
    from scarlett_live_whisper.asio import list_drivers
    names = list_drivers()
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)


def test_find_driver_is_case_insensitive_substring():
    from scarlett_live_whisper.asio import find_driver, list_drivers
    names = list_drivers()
    if not names:
        pytest.skip("no ASIO drivers registered on this machine")
    target = names[0]
    assert find_driver(target.lower()) == target
    assert find_driver(target[:4].upper()) in names


def test_find_driver_reports_what_is_available():
    from scarlett_live_whisper.asio import find_driver
    with pytest.raises(RuntimeError, match="no ASIO driver matching"):
        find_driver("definitely-not-a-real-driver-xyz")


def test_capture_rejects_more_channels_than_the_driver_has():
    """Guards the createBuffers call, which would otherwise fault."""
    from scarlett_live_whisper.asio import AsioCapture, find_driver
    try:
        drv = find_driver("focusrite usb")
    except RuntimeError:
        pytest.skip("no Focusrite ASIO driver on this machine")
    cap = AsioCapture(drv, channels=999)
    with pytest.raises(RuntimeError, match="inputs"):
        try:
            cap.open()
        finally:
            cap.stop()


def test_open_does_not_change_the_clock_by_default():
    """Claiming the device must never retune the interface behind your back."""
    from scarlett_live_whisper.asio import AsioCapture, find_driver
    try:
        drv = find_driver("focusrite usb")
    except RuntimeError:
        pytest.skip("no Focusrite ASIO driver on this machine")
    cap = AsioCapture(drv, channels=2)
    try:
        cap.open()
    except RuntimeError as e:
        pytest.skip(f"driver unavailable: {e}")
    try:
        assert cap.changed_rate is False
        assert cap.samplerate == cap.original_rate
    finally:
        cap.stop()
