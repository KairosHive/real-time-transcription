"""Ordering rules for a log that is appended across many sessions."""
from scarlett_live_whisper.merge import sort_key


def test_absolute_time_orders_across_sessions():
    """`t` restarts at zero every run, so sorting by it interleaves sessions.
    Records carrying `epoch` must order by real time instead."""
    a = {"session": "B", "epoch": 100.0, "t": 0.0, "channel": 1, "dur": 1}
    b = {"session": "A", "epoch": 50.0, "t": 90.0, "channel": 1, "dur": 1}
    assert sorted([a, b], key=sort_key) == [b, a]


def test_legacy_records_group_by_session():
    """Old rows have no epoch; they must at least not interleave."""
    rows = [
        {"session": "B", "t": 0.0, "channel": 1, "dur": 1},
        {"session": "A", "t": 5.0, "channel": 1, "dur": 1},
        {"session": "A", "t": 0.0, "channel": 1, "dur": 1},
    ]
    got = sorted(rows, key=sort_key)
    assert [r["session"] for r in got] == ["A", "A", "B"]
    assert [r["t"] for r in got] == [0.0, 5.0, 0.0]


def test_untagged_records_do_not_crash():
    rows = [{"t": 1.0, "channel": 1, "dur": 1}, {"t": 0.0, "channel": 1, "dur": 1}]
    assert [r["t"] for r in sorted(rows, key=sort_key)] == [0.0, 1.0]
