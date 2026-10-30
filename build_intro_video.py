#!/usr/bin/env python3
"""
Build a professional Teaser / Introduction Video from transcript highlights.
Hybrid Alignment Pipeline:
1. Words Ground Truth: Verified, perfectly spelled Arabic text from the YouTube API transcript.
2. Timing Ground Truth: Exact millisecond acoustic waveforms from audio clip analysis.
3. Typography: Segoe UI, Spacing=0 (100% connected Arabic cursive ligatures).
4. Dynamics: 1-2 words at a time as spoken, hugging background pill box, pop bounce.
5. Transitions: fadewhite flash hit, fadeblack dip-to-black, audio crossfades.
6. Universal Compatibility: yuv420p, high profile 4.0, +faststart.
"""

import difflib
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from faster_whisper import WhisperModel
from dynamic_captions import build_dynamic_ass

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SOURCE_VIDEO = r"c:\Users\AdMin\Downloads\Video\ناصر قنديل بحلقة نارية بعد انتخابات اسرائيل_اعدام نتنياهو واحتلال الكنيست! تفاصيل غزو صنعاء في اسبوع.mp4"
OUTPUT_DIR = Path(r"C:\Users\AdMin\Documents\Video-Creation\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Highlight segments selected from transcript
CLIPS_CONFIG = [
    {
        "id": "clip1",
        "start": 354.7,
        "end": 366.5,
        "title": "📌 المحور الأول: اعتراف نتنياهو بالهزيمة",
        "theme": "dark_pill",
    },
    {
        "id": "clip2",
        "start": 1275.3,
        "end": 1290.5,
        "title": "⚠️ المحور الثاني: سيناريو اقتحام واحتلال الكنيست",
        "theme": "red_alert",
    },
    {
        "id": "clip3",
        "start": 1471.5,
        "end": 1488.0,
        "title": "💥 المحور الثالث: تفاصيل مخطط غزو صنعاء",
        "theme": "yellow_pop",
    },
]


def normalize_ar(text: str) -> str:
    """Normalize Arabic text for robust phonetic / lexical sequence matching."""
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().lower()


def load_api_transcript_words() -> list:
    """Load clean ground-truth words from YouTube API transcript JSON."""
    json_files = glob.glob("transcripts/*DBMFJZDEhm4*transcript.json")
    if not json_files:
        raise FileNotFoundError("Transcript JSON not found in transcripts/")
    with open(json_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("words", [])


def get_api_words_for_clip(all_words: list, start_t: float, end_t: float) -> list:
    """Extract and clean verified ground-truth words for the specified time range."""
    raw = [w["word"].strip().strip(">>").strip() for w in all_words if w["start"] >= start_t and w["end"] <= end_t]
    # Filter out empty or punctuation-only tokens
    clean = [w for w in raw if w and w != "." and w != "،" and w != ">>"]
    # If the interviewer's acknowledgment ("صحيح.") appears at the start of clip 1, skip it
    if clean and clean[0] in ("صحيح", "صحيح."):
        clean = clean[1:]
    return clean


def cut_raw_clip(src_video: str, start: float, duration: float, out_path: Path):
    """Cut clean raw video clip from source file."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", src_video,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        str(out_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def get_audio_timing_words(clip_path: Path, whisper_model: WhisperModel) -> list:
    """Detect precise acoustic word timestamps from the cut clip's audio."""
    segs, _ = whisper_model.transcribe(str(clip_path), language="ar", word_timestamps=True)
    words = []
    for s in segs:
        if s.words:
            for w in s.words:
                txt = w.word.strip().strip(">>").strip()
                if txt:
                    words.append({
                        "word": txt,
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                    })
    return words


def align_api_words_to_audio_timing(api_words: list, whisper_words: list) -> list:
    """
    Hybrid Alignment:
    - Uses API words as GROUND TRUTH for spelling and vocabulary.
    - Uses Whisper acoustic detections as GROUND TRUTH for millisecond timing.
    """
    norm_api = [normalize_ar(w) for w in api_words]
    norm_wsp = [normalize_ar(w["word"]) for w in whisper_words]

    matcher = difflib.SequenceMatcher(None, norm_api, norm_wsp)
    aligned_words = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        api_slice = api_words[i1:i2]
        wsp_slice = whisper_words[j1:j2]

        if not api_slice:
            continue

        # If identical match, attach exact audio timestamp word-by-word
        if tag == "equal":
            for api_w, wsp_w in zip(api_slice, wsp_slice):
                aligned_words.append({
                    "word": api_w,
                    "start": wsp_w["start"],
                    "end": wsp_w["end"],
                })
            continue

        if wsp_slice:
            block_start = wsp_slice[0]["start"]
            block_end = wsp_slice[-1]["end"]
        else:
            prev_end = aligned_words[-1]["end"] if aligned_words else 0.0
            block_start = prev_end
            block_end = prev_end + 0.35 * len(api_slice)

        block_dur = max(0.25, block_end - block_start)
        weights = [max(1, len(w)) for w in api_slice]
        tot_wt = sum(weights)
        curr = block_start

        for w, wt in zip(api_slice, weights):
            dur = (wt / tot_wt) * block_dur
            aligned_words.append({
                "word": w,  # Clean verified API word
                "start": round(curr, 2),
                "end": round(curr + dur, 2),
            })
            curr += dur

    return aligned_words


def style_clip_with_ass(raw_clip: Path, ass_path: Path, out_styled: Path):
    """Burn dynamic ASS subtitles onto the cut clip with universal 4:2:0 format."""
    escaped_ass = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_clip),
        "-vf", f"subtitles='{escaped_ass}',scale=1280:720,setsar=1,fps=25,format=yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "19",
        "-c:a", "copy",
        str(out_styled),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def main():
    print("=" * 60, flush=True)
    print("  CREATING INTRO: HYBRID GROUND-TRUTH WORDS + AUDIO TIMING", flush=True)
    print("=" * 60, flush=True)

    print("[*] Loading ground-truth words from API transcript...", flush=True)
    api_words_all = load_api_transcript_words()

    print("[*] Initializing audio alignment engine on CPU...", flush=True)
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

    clip_paths = []
    durations = []

    for cfg in CLIPS_CONFIG:
        start = cfg["start"]
        end = cfg["end"]
        duration = round(end - start, 2)
        raw_clip = OUTPUT_DIR / f"{cfg['id']}_raw.mp4"
        styled_clip = OUTPUT_DIR / f"{cfg['id']}_styled.mp4"
        ass_file = OUTPUT_DIR / f"{cfg['id']}_dynamic.ass"

        # 1. Cut clean raw clip
        print(f"\n[*] Cutting {cfg['id']} ({start}s -> {end}s, {duration}s)...", flush=True)
        cut_raw_clip(SOURCE_VIDEO, start, duration, raw_clip)

        # 2. Get ground-truth words from API transcript
        api_clip_words = get_api_words_for_clip(api_words_all, start, end)
        print(f"[+] Ground-truth API words ({len(api_clip_words)}): {' '.join(api_clip_words[:6])}...", flush=True)

        # 3. Get exact acoustic timecodes from audio waveform
        wsp_timing_words = get_audio_timing_words(raw_clip, whisper_model)

        # 4. Hybrid Alignment: fuse API spelling + Whisper timing
        aligned_words = align_api_words_to_audio_timing(api_clip_words, wsp_timing_words)
        print(f"[+] Aligned {len(aligned_words)} words with zero-lag millisecond timecodes.", flush=True)

        # 5. Build dynamic ASS subtitle file (Segoe UI, Spacing=0, hugging box, 2 words max)
        build_dynamic_ass(
            words=aligned_words,
            output_ass_path=ass_file,
            words_per_screen=2,
            theme=cfg.get("theme", "dark_pill"),
            position="bottom",
            max_duration=duration,
            pop_animation=True,
            header_badge=cfg["title"],
        )

        # 6. Burn dynamic subtitles onto clip
        print(f"[*] Burning dynamic subtitles into {cfg['id']}...", flush=True)
        style_clip_with_ass(raw_clip, ass_file, styled_clip)

        clip_paths.append(styled_clip)
        durations.append(duration)
        print(f"[✓] {cfg['id']} rendered successfully.", flush=True)

    # 7. Merge all clips with transitions and audio crossfades
    trans_dur = 0.6
    d0, d1, d2 = durations[0], durations[1], durations[2]
    offset_1 = round(d0 - trans_dur, 2)
    offset_2 = round((d0 + d1 - trans_dur) - trans_dur, 2)

    final_video = OUTPUT_DIR / "intro_dynamic_captions_with_effects.mp4"

    filter_complex = (
        f"[0:v][1:v]xfade=transition=fadewhite:duration={trans_dur}:offset={offset_1}[v01]; "
        f"[v01][2:v]xfade=transition=fadeblack:duration={trans_dur}:offset={offset_2},fade=t=in:st=0:d=0.8,fade=t=out:st={offset_2 + d2 - 0.8}:d=0.8,format=yuv420p[vout]; "
        f"[0:a][1:a]acrossfade=d={trans_dur}:c1=tri:c2=tri[a01]; "
        f"[a01][2:a]acrossfade=d={trans_dur}:c1=tri:c2=tri,afade=t=in:st=0:d=0.8,afade=t=out:st={offset_2 + d2 - 0.8}:d=0.8[aout]"
    )

    print(f"\n[*] Merging clips with transitions: fadewhite ({offset_1}s) & fadeblack ({offset_2}s)...", flush=True)

    merge_cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_paths[0]),
        "-i", str(clip_paths[1]),
        "-i", str(clip_paths[2]),
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "19",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high", "-level", "4.0",
        "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        str(final_video),
    ]

    subprocess.run(merge_cmd, check=True)

    print("\n" + "=" * 60, flush=True)
    print(f"[✓] HYBRID ACCURATE INTRO CREATED SUCCESSFULLY!", flush=True)
    print(f"[+] Output Video: {final_video}", flush=True)
    print(f"[+] Total Duration: ~{round(d0 + d1 + d2 - 2 * trans_dur, 1)} seconds", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
