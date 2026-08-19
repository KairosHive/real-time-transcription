"""Merge a transcript.jsonl into one chronological, human-readable log."""
import argparse
import json


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="scarlett-merge",
        description="Sort a JSONL transcript into a single timeline.")
    p.add_argument("path", nargs="?", default="transcript.jsonl")
    p.add_argument("--channel", type=int, help="only this 1-based channel")
    a = p.parse_args(argv)

    with open(a.path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if a.channel:
        rows = [r for r in rows if r["channel"] == a.channel]
    for r in sorted(rows, key=lambda r: r["t"]):
        print(f"[{r['wall']}] {r['name']:<10} {r['text']}")


if __name__ == "__main__":
    main()
