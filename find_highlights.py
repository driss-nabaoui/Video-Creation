import json
import glob
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

p = glob.glob("transcripts/*DBMFJZDEhm4*transcript.json")[0]
with open(p, encoding="utf-8") as f:
    data = json.load(f)

words = data.get("words", [])

def get_words_in_range(t0, t1):
    return [w for w in words if w["start"] >= t0 and w["end"] <= t1]

print("=== Clip 1 Candidate words (354 to 366) ===")
c1 = get_words_in_range(354, 366)
print(" ".join(w["word"] for w in c1))
if c1:
    print(f"Start: {c1[0]['start']}s, End: {c1[-1]['end']}s, Duration: {c1[-1]['end'] - c1[0]['start']:.2f}s")

print("\n=== Clip 2 Candidate words (1275 to 1292) ===")
c2 = get_words_in_range(1275, 1292)
print(" ".join(w["word"] for w in c2))
if c2:
    print(f"Start: {c2[0]['start']}s, End: {c2[-1]['end']}s, Duration: {c2[-1]['end'] - c2[0]['start']:.2f}s")

print("\n=== Clip 3 Candidate words (1470 to 1488) ===")
c3 = get_words_in_range(1470, 1488)
print(" ".join(w["word"] for w in c3))
if c3:
    print(f"Start: {c3[0]['start']}s, End: {c3[-1]['end']}s, Duration: {c3[-1]['end'] - c3[0]['start']:.2f}s")
