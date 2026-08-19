"""Merge a transcript.jsonl into one chronological, human-readable log."""
import argparse
import json


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sort_key(r):
    """Absolute time when present. Older records only carry a per-run `t`,
    which restarts at zero each session, so group those by session first."""
    return (r["epoch"],) if "epoch" in r else (0.0, r.get("session", ""), r["t"])


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="scarlett-merge",
        description="Sort a JSONL transcript into a single timeline.")
    p.add_argument("path", nargs="?", default="transcript.jsonl")
    p.add_argument("--channel", type=int, help="only this 1-based channel")
    p.add_argument("--session", help="only this session tag")
    p.add_argument("--list-sessions", action="store_true",
                   help="show the sessions in the file and exit")
    p.add_argument("--gaps", type=float, metavar="SECONDS",
                   help="flag untranscribed gaps longer than this")
    a = p.parse_args(argv)

    rows = load(a.path)
    if a.list_sessions:
        seen = {}
        for r in rows:
            s = r.get("session", "(untagged)")
            seen[s] = seen.get(s, 0) + 1
        for s, n in sorted(seen.items()):
            print(f"{s}  {n} utterances")
        return
    if a.channel:
        rows = [r for r in rows if r["channel"] == a.channel]
    if a.session:
        rows = [r for r in rows if r.get("session") == a.session]

    prev_end = {}
    for r in sorted(rows, key=sort_key):
        if a.gaps and "epoch" in r:
            last = prev_end.get(r["channel"])
            if last is not None and r["epoch"] - last > a.gaps:
                print(f"    ... {r['epoch'] - last:.1f}s not transcribed on "
                      f"channel {r['channel']} ...")
            prev_end[r["channel"]] = r["epoch"] + r["dur"]
        clock = r["wall"].split()[-1]
        print(f"[{clock}] {r['name']:<10} {r['text']}")


if __name__ == "__main__":
    main()
