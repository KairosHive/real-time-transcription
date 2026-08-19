import pytest

from scarlett_live_whisper.config import db, resolve_channels


def test_defaults_to_every_channel():
    active, names = resolve_channels(None, 4, None)
    assert active == [0, 1, 2, 3]
    assert names == ["ch1", "ch2", "ch3", "ch4"]


def test_only_selects_and_sorts():
    active, _ = resolve_channels("4,1,3", 8, None)
    assert active == [0, 2, 3]


def test_only_deduplicates():
    active, _ = resolve_channels("2,2,2", 4, None)
    assert active == [1]


def test_blank_names_fall_back():
    """--names Alice,Bob,,,Room must not produce empty labels."""
    _, names = resolve_channels(None, 5, "Alice,Bob,,,Room")
    assert names == ["Alice", "Bob", "ch3", "ch4", "Room"]


def test_short_name_list_is_padded():
    _, names = resolve_channels(None, 4, "Alice")
    assert names == ["Alice", "ch2", "ch3", "ch4"]


def test_out_of_range_channel_is_rejected():
    """The old code silently indexed past the opened channels."""
    with pytest.raises(ValueError, match="only 8 channels are open"):
        resolve_channels("9", 8, None)


def test_zero_channel_is_rejected():
    with pytest.raises(ValueError):
        resolve_channels("0", 8, None)


def test_non_numeric_only_is_rejected():
    with pytest.raises(ValueError, match="expected a channel number"):
        resolve_channels("alice", 8, None)


def test_too_many_names_is_rejected():
    with pytest.raises(ValueError, match="only 2 channels"):
        resolve_channels(None, 2, "a,b,c")


def test_db_floor_does_not_raise_on_silence():
    assert db(0.0) < -150
    assert db(1.0) == pytest.approx(0.0)
