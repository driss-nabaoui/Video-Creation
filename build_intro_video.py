#!/usr/bin/env python3
"""
Build a professional Teaser / Introduction Video from transcript highlights.
Fixed for:
1. Readable Arabic (Segoe UI font, Spacing=0 for cursive ligature connections)
2. Zero lag: precise audio waveform synchronization directly on cut clips
3. Dynamic hugging background boxes (1-2 words at a time as spoken, NO full lines)
4. Smooth transitions (fadewhite flash hit, fadeblack dip-to-black)
5. 100% universal yuv420p / faststart audio & video compatibility
"""

import json
import os
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


def cut_raw_clip(src_video: str, start: float, duration: float, out_path: Path):
    """Cut raw clean clip from source video."""
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


def get_precise_clip_words(clip_path: Path, whisper_model: WhisperModel) -> list:
    """Extract millisecond-accurate word timestamps directly from clip audio to eliminate lag."""
    segs, _ = whisper_model.transcribe(str(clip_path), language="ar", word_timestamps=True)
    words = []
    word_idx = 0
    for s in segs:
        if s.words:
            for w in s.words:
                txt = w.word.strip().strip(">>").strip()
                if txt:
                    words.append({
                        "index": word_idx,
                        "word": txt,
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                    })
                    word_idx += 1
    return words


def style_clip_with_ass(raw_clip: Path, ass_path: Path, out_styled: Path):
    """Burn dynamic ASS subtitles onto the cut clip."""
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
    print("  CREATING INTRO WITH ZERO-LAG ARABIC DYNAMIC CAPTIONS", flush=True)
    print("=" * 60, flush=True)

    print("[*] Loading fast audio alignment model on CPU...", flush=True)
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

        # 1. Cut clean raw clip from source
        print(f"[*] Cutting {cfg['id']} ({start}s -> {end}s, {duration}s)...", flush=True)
        cut_raw_clip(SOURCE_VIDEO, start, duration, raw_clip)

        # 2. Extract 100% exact, zero-lag word timestamps from audio
        print(f"[*] Aligning exact word audio timestamps for {cfg['id']}...", flush=True)
        exact_words = get_precise_clip_words(raw_clip, whisper_model)

        # 3. Build Arabic-safe ASS (Spacing=0, Segoe UI, hugging box, 2 words max)
        build_dynamic_ass(
            words=exact_words,
            output_ass_path=ass_file,
            words_per_screen=2,
            theme=cfg.get("theme", "dark_pill"),
            position="bottom",
            max_duration=duration,
            pop_animation=True,
            header_badge=cfg["title"],
        )

        # 4. Burn subtitles onto clip
        print(f"[*] Burning dynamic subtitles into {cfg['id']}...", flush=True)
        style_clip_with_ass(raw_clip, ass_file, styled_clip)

        clip_paths.append(styled_clip)
        durations.append(duration)
        print(f"[✓] {cfg['id']} complete ({duration}s)", flush=True)

    # 5. Merge all clips with transition effects
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

    print(f"[*] Merging with transitions: fadewhite ({offset_1}s) & fadeblack ({offset_2}s)...", flush=True)

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
    print(f"[✓] ZERO-LAG ARABIC DYNAMIC INTRO CREATED SUCCESSFULLY!", flush=True)
    print(f"[+] Output Video: {final_video}", flush=True)
    print(f"[+] Total Duration: ~{round(d0 + d1 + d2 - 2 * trans_dur, 1)} seconds", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
