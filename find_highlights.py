#!/usr/bin/env python3
"""
Utility script to preview and inspect transcript words between given timestamps.
Usage:
  python find_highlights.py [path_to_transcript.json or project_id]
"""

import argparse
import glob
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def find_transcript(target: str = None) -> Path:
    if target:
        p = Path(target)
        if p.is_file():
            return p
        # Check if project folder exists
        proj_p = Path("projects") / target / "transcripts"
        if proj_p.is_dir():
            found = list(proj_p.glob("*transcript.json"))
            if found:
                return found[0]

    # Look for candidates
    candidates = (
        glob.glob("projects/*/transcripts/*transcript.json")
        + glob.glob("transcripts/*transcript.json")
    )
    if candidates:
        return Path(candidates[0])

    raise FileNotFoundError("No transcript JSON file found. Please provide path or project ID.")


def main():
    parser = argparse.ArgumentParser(description="Inspect candidate highlight words in transcript")
    parser.add_argument("transcript", nargs="?", default=None, help="Path to transcript JSON or project ID")
    parser.add_argument("--start", type=float, default=None, help="Start seconds")
    parser.add_argument("--end", type=float, default=None, help="End seconds")

    args = parser.parse_args()

    try:
        t_path = find_transcript(args.transcript)
        print(f"[*] Reading transcript: {t_path}")
        with open(t_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    words = data.get("words", [])

    def get_words_in_range(t0, t1):
        return [w for w in words if w["start"] >= t0 and w["end"] <= t1]

    if args.start is not None and args.end is not None:
        c = get_words_in_range(args.start, args.end)
        print(f"\n=== Range ({args.start}s to {args.end}s) ===")
        print(" ".join(w["word"] for w in c))
        if c:
            print(f"Start: {c[0]['start']}s, End: {c[-1]['end']}s, Duration: {c[-1]['end'] - c[0]['start']:.2f}s")
        return

    # Default preview ranges
    ranges = [
        ("Clip 1", 354, 366),
        ("Clip 2", 1275, 1292),
        ("Clip 3", 1470, 1488),
    ]

    for label, t0, t1 in ranges:
        c = get_words_in_range(t0, t1)
        print(f"\n=== {label} Candidate words ({t0} to {t1}) ===")
        print(" ".join(w["word"] for w in c))
        if c:
            print(f"Start: {c[0]['start']}s, End: {c[-1]['end']}s, Duration: {c[-1]['end'] - c[0]['start']:.2f}s")


if __name__ == "__main__":
    main()
