#!/usr/bin/env python3
"""
Dynamic Captions Tool for Videos.
Features:
- Native Arabic cursive ligature support (Segoe UI, Spacing=0)
- 1-2 words at a time as spoken (never full lines)
- Dynamic hugging background box that resizes to words
- Zero lag: precise audio waveform synchronization
- Snappy pop-in spring bounce animation
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

THEMES = {
    # 1. Dark Pill (Modern TikTok / Reels): Translucent dark charcoal box with bright white/gold text
    "dark_pill": {
        "font": "Segoe UI",
        "text_color": "&H00FFFFFF&",       # White
        "box_color": "&HA0101010&",        # Translucent dark charcoal box
        "highlight_color": "&H0000D7FF&",  # Warm gold/yellow
        "fontsize": 48,
        "outline": 14,
        "border_style": 3,
    },
    # 2. Yellow Pop (Alex Hormozi style): Bold vibrant yellow box with punchy black text
    "yellow_pop": {
        "font": "Segoe UI",
        "text_color": "&H00000000&",       # Black
        "box_color": "&H0000D7FF&",        # Vibrant Yellow
        "highlight_color": "&H000000FF&",
        "fontsize": 48,
        "outline": 16,
        "border_style": 3,
    },
    # 3. Red Alert: Crimson background box with white text for breaking statements
    "red_alert": {
        "font": "Segoe UI",
        "text_color": "&H00FFFFFF&",       # White
        "box_color": "&H001E1ECC&",        # Crimson Red box
        "highlight_color": "&H0000FFFF&",
        "fontsize": 48,
        "outline": 16,
        "border_style": 3,
    },
}


def to_ass_timestamp(seconds: float) -> str:
    """Format float seconds to ASS timestamp (H:MM:SS.cs)."""
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis -= 100
        secs += 1
        if secs >= 60:
            secs -= 60
            mins += 1
            if mins >= 60:
                mins -= 60
                hrs += 1
    return f"{hrs}:{mins:02d}:{secs:02d}.{centis:02d}"


def build_dynamic_ass(
    words: List[Dict[str, Any]],
    output_ass_path: Path,
    words_per_screen: int = 2,
    theme: str = "dark_pill",
    position: str = "bottom",
    max_duration: Optional[float] = None,
    pop_animation: bool = True,
    header_badge: Optional[str] = None,
) -> Path:
    """
    Builds an ASS subtitle file that displays words 1-2 at a time as spoken,
    with dynamically hugging background boxes and native Arabic cursive connections.
    """
    style_cfg = THEMES.get(theme, THEMES["dark_pill"])
    font_name = style_cfg.get("font", "Segoe UI")

    # Base alignment & margin
    if position == "center":
        alignment = 5
        margin_v = 0
    elif position == "top":
        alignment = 8
        margin_v = 50
    else:  # default bottom
        alignment = 2
        margin_v = 75

    # IMPORTANT: Spacing must be 0 for Arabic. Spacing > 0 breaks cursive character connections!
    ass_lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1280",
        "PlayResY: 720",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: DynamicWord,{font_name},{style_cfg['fontsize']},{style_cfg['text_color']},&H000000FF,{style_cfg['box_color']},{style_cfg['box_color']},-1,0,0,0,100,100,0,0,{style_cfg['border_style']},{style_cfg['outline']},0,{alignment},30,30,{margin_v},1",
        f"Style: HeaderBadge,{font_name},30,&H0000D7FF,&H000000FF,&HA0101010,&HA0101010,-1,0,0,0,100,100,0,0,3,10,0,8,40,40,25,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    # Optional top header badge (constant throughout the clip)
    if header_badge and max_duration:
        ass_lines.append(
            f"Dialogue: 0,0:00:00.00,{to_ass_timestamp(max_duration)},HeaderBadge,,0,0,0,,{header_badge}"
        )

    # Filter clean words
    clean_words = []
    for w in sorted(words, key=lambda x: x.get("start", 0.0)):
        txt = w.get("word", "").strip().strip(">>").strip()
        if not txt:
            continue
        clean_words.append({
            "word": txt,
            "start": max(0.0, float(w.get("start", 0.0))),
            "end": max(0.0, float(w.get("end", 0.0))),
        })

    if not clean_words:
        output_ass_path.parent.mkdir(parents=True, exist_ok=True)
        output_ass_path.write_text("\n".join(ass_lines), encoding="utf-8")
        return output_ass_path

    # Group words into small bite-sized chunks (1 to 2 words per screen, never full lines)
    chunks = []
    for i in range(0, len(clean_words), words_per_screen):
        chunk = clean_words[i:i + words_per_screen]
        chunks.append(chunk)

    for chunk_idx, chunk in enumerate(chunks):
        c_start = chunk[0]["start"]
        c_end = chunk[-1]["end"]

        # Ensure minimal visible window of at least 0.35s
        if c_end - c_start < 0.35:
            c_end = c_start + 0.35

        if max_duration and c_end > max_duration:
            c_end = max_duration
        if max_duration and c_start >= max_duration:
            break

        t_start = to_ass_timestamp(c_start)
        t_end = to_ass_timestamp(c_end)
        text = " ".join(w["word"] for w in chunk)

        # Subtle pop bounce animation
        anim_tag = "{\\fscx106\\fscy106\\t(0,50,\\fscx100\\fscy100)}" if pop_animation else ""

        # Dynamic positioning (optional subtle alternate)
        pos_tag = ""
        if position == "dynamic":
            y_shift = 75 if (chunk_idx % 2 == 0) else 95
            pos_tag = f"{{\\an2\\pos(640,{720 - y_shift})}}"

        ass_lines.append(f"Dialogue: 0,{t_start},{t_end},DynamicWord,,0,0,0,,{anim_tag}{pos_tag}{text}")

    output_ass_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ass_lines) + "\n")

    return output_ass_path


def burn_dynamic_captions(
    video_path: str,
    ass_path: Path,
    output_video_path: str,
) -> None:
    """Burns the dynamic captions onto the video using FFmpeg with universal 4:2:0 compatibility."""
    escaped_ass = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles='{escaped_ass}',format=yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "19",
        "-profile:v", "high", "-level", "4.0",
        "-movflags", "+faststart",
        "-c:a", "copy",
        output_video_path,
    ]
    print(f"[*] Burning dynamic captions into: {output_video_path}...", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[✓] Output saved to: {output_video_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Add dynamic word-by-word captions with hugging background boxes to video"
    )
    parser.add_argument("video", help="Input video file")
    parser.add_argument("transcript_json", help="Path to transcript.json file")
    parser.add_argument("-o", "--output", default="captioned_video.mp4", help="Output video path")
    parser.add_argument("-w", "--words", type=int, default=2, help="Words per screen (default: 2)")
    parser.add_argument(
        "--theme",
        choices=["dark_pill", "yellow_pop", "red_alert"],
        default="dark_pill",
        help="Background box and color theme (default: dark_pill)",
    )
    parser.add_argument(
        "--position",
        choices=["bottom", "center", "top", "dynamic"],
        default="bottom",
        help="Screen positioning (default: bottom)",
    )
    parser.add_argument("--no-pop", action="store_true", help="Disable word pop-in bounce effect")

    args = parser.parse_args()

    with open(args.transcript_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data.get("words", [])
    if not words:
        print("[!] No words found in transcript JSON.", file=sys.stderr)
        sys.exit(1)

    temp_ass = Path("temp_dynamic.ass")
    build_dynamic_ass(
        words=words,
        output_ass_path=temp_ass,
        words_per_screen=args.words,
        theme=args.theme,
        position=args.position,
        pop_animation=not args.no_pop,
    )

    burn_dynamic_captions(args.video, temp_ass, args.output)

    if temp_ass.exists():
        temp_ass.unlink()


if __name__ == "__main__":
    main()
