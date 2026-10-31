#!/usr/bin/env python3
"""
Dynamic Captions Tool for Videos.
Features:
- Native Arabic cursive ligature support (Cairo, Readex Pro, IBM Plex Sans Arabic, Segoe UI)
- Multiple Modern Aesthetic Styles:
  1. floating_karaoke: Boxless floating text with active word karaoke coloring.
  2. stacked_kinetic: Vertical 2-3 word stacks fitting tight lateral gaps.
  3. glassmorphic: Translucent pill with 40% black opacity and clean border.
  4. dark_pill / yellow_pop / red_alert: Classic high-impact social media pills.
- Dynamic Empty Space Placement: Positions captions inside clean background space using coordinates.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

THEMES = {
    # 1. Floating Karaoke: Boxless floating text with active spoken word highlighted
    "floating_karaoke": {
        "font": "Cairo",
        "text_color": "&H00FFFFFF&",       # White
        "box_color": "&H00000000&",
        "highlight_color": "&H0000D7FF&",  # Warm Gold
        "fontsize": 52,
        "outline": 2,
        "shadow": 3,
        "border_style": 1,                 # Text + outline/shadow (No box)
        "karaoke": True,
        "stacked": False,
    },
    # 2. Stacked Kinetic: Vertical word stacks for narrow spaces beside speaker
    "stacked_kinetic": {
        "font": "Cairo",
        "text_color": "&H00FFFFFF&",       # White
        "box_color": "&H00000000&",
        "highlight_color": "&H0000D7FF&",  # Vibrant Gold
        "fontsize": 48,
        "outline": 2,
        "shadow": 3,
        "border_style": 1,
        "karaoke": True,
        "stacked": True,
    },
    # 3. Glassmorphic: 40% opacity frosted-style dark pill
    "glassmorphic": {
        "font": "Readex Pro",
        "text_color": "&H00FFFFFF&",       # White
        "box_color": "&H66000000&",        # 40% opacity black in ASS (&H66 = 102/255)
        "highlight_color": "&H0000D7FF&",  # Gold
        "fontsize": 46,
        "outline": 14,
        "shadow": 0,
        "border_style": 3,                 # Pill box
        "karaoke": True,
        "stacked": False,
    },
    # 4. Classic Dark Pill
    "dark_pill": {
        "font": "Cairo",
        "text_color": "&H00FFFFFF&",       # White
        "box_color": "&HA0101010&",        # Translucent dark charcoal box
        "highlight_color": "&H0000D7FF&",  # Warm gold
        "fontsize": 48,
        "outline": 14,
        "shadow": 0,
        "border_style": 3,
        "karaoke": False,
        "stacked": False,
    },
    # 5. Yellow Pop (Hormozi style)
    "yellow_pop": {
        "font": "Cairo",
        "text_color": "&H00000000&",       # Black
        "box_color": "&H0000D7FF&",        # Vibrant Yellow
        "highlight_color": "&H000000FF&",  # Red
        "fontsize": 48,
        "outline": 16,
        "shadow": 0,
        "border_style": 3,
        "karaoke": False,
        "stacked": False,
    },
    # 6. Red Alert
    "red_alert": {
        "font": "Cairo",
        "text_color": "&H00FFFFFF&",       # White
        "box_color": "&H001E1ECC&",        # Crimson Red
        "highlight_color": "&H0000FFFF&",  # Yellow
        "fontsize": 48,
        "outline": 16,
        "shadow": 0,
        "border_style": 3,
        "karaoke": False,
        "stacked": False,
    },
    # 7. Broadcast Investigative Dark (Default documentary / broadcast editorial)
    "broadcast_investigative_dark": {
        "font": "Cairo",
        "text_color": "&H00FFFFFF&",       # Crisp White
        "box_color": "&H00000000&",
        "highlight_color": "&H0000D7FF&",  # Warm Gold
        "fontsize": 52,
        "outline": 3,
        "shadow": 3,
        "border_style": 1,
        "karaoke": True,
        "stacked": False,
    },
    # 8. Urgent Red (Breaking news, existential stakes, civil clash)
    "urgent_red": {
        "font": "Cairo",
        "text_color": "&H00FFFFFF&",       # White
        "box_color": "&H00000000&",
        "highlight_color": "&H002020FF&",  # Vivid Signal Red / Crimson
        "fontsize": 52,
        "outline": 3,
        "shadow": 3,
        "border_style": 1,
        "karaoke": True,
        "stacked": False,
    },
    # 9. Suspense Amber (Dramatic questions, rising tension, ultimatum)
    "suspense_amber": {
        "font": "Cairo",
        "text_color": "&H00FFFFFF&",       # White
        "box_color": "&H00000000&",
        "highlight_color": "&H000099FF&",  # Electric Amber / Vibrant Orange
        "fontsize": 52,
        "outline": 3,
        "shadow": 3,
        "border_style": 1,
        "karaoke": True,
        "stacked": False,
    },
    # 10. Investigative Blue (Diplomatic leaks, intelligence, Foreign Affairs)
    "investigative_blue": {
        "font": "Cairo",
        "text_color": "&H00FFFFFF&",       # White
        "box_color": "&H00000000&",
        "highlight_color": "&H00FFD000&",  # Electric Cyan / Vivid Sky Blue
        "fontsize": 52,
        "outline": 3,
        "shadow": 3,
        "border_style": 1,
        "karaoke": True,
        "stacked": False,
    },
    # 11. Highlight Gold (Decisive revelations, counter-intuitive facts)
    "highlight_gold": {
        "font": "Cairo",
        "text_color": "&H00FFFFFF&",       # White
        "box_color": "&H00000000&",
        "highlight_color": "&H0000D7FF&",  # Luminous Champagne Gold
        "fontsize": 52,
        "outline": 3,
        "shadow": 3,
        "border_style": 1,
        "karaoke": True,
        "stacked": False,
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


def is_arabic(text: str) -> bool:
    """Check if string contains Arabic characters."""
    return any(
        "\u0600" <= ch <= "\u06FF"
        or "\u0750" <= ch <= "\u077F"
        or "\u08A0" <= ch <= "\u08FF"
        for ch in text
    )


def chunk_words_by_phrase(
    words: List[Dict[str, Any]],
    target_words: int = 5,
    max_words: int = 7,
    max_pause_sec: float = 0.40,
) -> List[List[Dict[str, Any]]]:
    """
    Groups words into natural, readable phrases (default 4-6 words) based on
    speech pauses and punctuation boundaries, avoiding rigid small cuts.
    """
    if not words:
        return []

    chunks = []
    current_chunk = []

    for i, w in enumerate(words):
        current_chunk.append(w)
        should_split = False

        # Punctuation at end of word
        txt = w.get("word", "")
        if any(txt.endswith(p) for p in (".", "؟", "!", "،", "؛", ":")):
            should_split = True

        # Acoustic pause before next word
        if i + 1 < len(words):
            next_start = words[i + 1].get("start", 0.0)
            curr_end = w.get("end", 0.0)
            if next_start - curr_end > max_pause_sec:
                should_split = True

        # Target phrase size reached
        if len(current_chunk) >= target_words:
            should_split = True

        # Hard limit
        if len(current_chunk) >= max_words:
            should_split = True

        if should_split and i + 1 < len(words):
            chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        # Merge trailing tiny dangling 1-2 words into previous chunk if available
        if len(current_chunk) < 3 and len(chunks) > 0 and len(chunks[-1]) + len(current_chunk) <= max_words + 1:
            chunks[-1].extend(current_chunk)
        else:
            chunks.append(current_chunk)

    return chunks


def build_dynamic_ass(
    words: List[Dict[str, Any]],
    output_ass_path: Path,
    words_per_screen: int = 5,
    theme: str = "floating_karaoke",
    position: str = "bottom",
    target_box: Optional[Tuple[int, int, int, int]] = None,
    max_duration: Optional[float] = None,
    pop_animation: bool = True,
    header_badge: Optional[str] = None,
    font: Optional[str] = None,
    highlight_color: Optional[str] = None,
) -> Path:
    """
    Builds an ASS subtitle file supporting Floating Karaoke, Stacked Kinetic Blocks,
    Glassmorphic Pill, and empty space coordinate injection.
    """
    style_cfg = THEMES.get(theme, THEMES["floating_karaoke"])
    font_name = font or style_cfg.get("font", "Cairo")
    active_color = highlight_color or style_cfg.get("highlight_color", "&H0000D7FF&")
    primary_color = style_cfg.get("text_color", "&H00FFFFFF&")
    is_karaoke = style_cfg.get("karaoke", False)
    is_stacked = style_cfg.get("stacked", False)

    # Base alignment & positioning
    pos_tag = ""
    alignment = 2
    margin_v = 75

    if target_box:
        # Place directly inside the detected empty space bounding box
        bx, by, bw, bh = target_box
        # Use center of the detected empty space
        cx = bx + bw // 2
        cy = by + bh // 2
        alignment = 5  # Center alignment
        margin_v = 0
        pos_tag = f"{{\\an5\\pos({cx},{cy})}}"
    else:
        if position == "center":
            alignment = 5
            margin_v = 0
        elif position == "top":
            alignment = 8
            margin_v = 50
        elif position == "left":
            alignment = 4
            pos_tag = "{\\an4\\pos(120,360)}"
        elif position == "right":
            alignment = 6
            pos_tag = "{\\an6\\pos(1160,360)}"
        else:  # default bottom
            alignment = 2
            margin_v = 75

    shadow_val = style_cfg.get("shadow", 0)
    blur_tag = "\\blur3" if shadow_val > 0 else ""

    ass_lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1280",
        "PlayResY: 720",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: DynamicWord,{font_name},{style_cfg['fontsize']},{primary_color},&H000000FF,{style_cfg['box_color']},{style_cfg['box_color']},-1,0,0,0,100,100,0,0,{style_cfg['border_style']},{style_cfg['outline']},{shadow_val},{alignment},30,30,{margin_v},1",
    ]

    if header_badge:
        ass_lines.append(
            f"Style: HeaderBadge,{font_name},28,&H0000D7FF,&H000000FF,&HA0101010,&HA0101010,-1,0,0,0,100,100,0,0,3,10,0,8,40,40,25,1"
        )

    ass_lines.extend([
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ])

    # Optional top header badge
    if header_badge and max_duration:
        ass_lines.append(
            f"Dialogue: 0,0:00:00.00,{to_ass_timestamp(max_duration)},HeaderBadge,,0,0,0,,{header_badge}"
        )

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

    # Group words into natural phrases (4-6 words for horizontal, 2-3 for stacked)
    if is_stacked:
        chunk_size = 3
        chunks = [clean_words[i : i + chunk_size] for i in range(0, len(clean_words), chunk_size)]
    else:
        chunks = chunk_words_by_phrase(
            clean_words,
            target_words=words_per_screen,
            max_words=max(words_per_screen + 2, 7),
        )

    for chunk_idx, chunk in enumerate(chunks):
        c_start = chunk[0]["start"]
        c_end = chunk[-1]["end"]

        # Clamp overlaps
        if chunk_idx + 1 < len(chunks):
            next_start = chunks[chunk_idx + 1][0]["start"]
            if c_end > next_start:
                c_end = next_start
            elif c_end - c_start < 0.25 and next_start > c_end:
                c_end = min(c_start + 0.30, next_start)
        else:
            if c_end - c_start < 0.30:
                c_end = c_start + 0.30

        if max_duration and c_end > max_duration:
            c_end = max_duration
        if max_duration and c_start >= max_duration:
            break

        # Subtle pop bounce animation
        anim_tag = "{\\fscx105\\fscy105\\t(0,60,\\fscx100\\fscy100)}" if pop_animation else ""

        if is_karaoke:
            # Word-by-word active highlight lines within this chunk's window
            for active_idx, active_w in enumerate(chunk):
                # Ensure contiguous timing with no gaps between words
                w_start = c_start if active_idx == 0 else max(c_start, active_w["start"])
                if active_idx + 1 < len(chunk):
                    next_st = chunk[active_idx + 1]["start"]
                    w_end = next_st if next_st > w_start else max(active_w["end"], w_start + 0.20)
                else:
                    w_end = c_end

                if w_end <= w_start:
                    w_end = w_start + 0.25

                t_w_start = to_ass_timestamp(w_start)
                t_w_end = to_ass_timestamp(w_end)

                # Pop entrance bounce only when chunk first appears to keep subsequent words rock-solid
                anim = anim_tag if (pop_animation and active_idx == 0) else ""

                if is_stacked:
                    # Stacked Kinetic: Vertical lines (Top to Bottom)
                    line_parts = []
                    for idx, w in enumerate(chunk):
                        word_str = w["word"]
                        if idx == active_idx:
                            line_parts.append(f"{{\\c{active_color}}}{word_str}{{\\c{primary_color}}}")
                        else:
                            line_parts.append(word_str)
                    text_line = "\\N".join(line_parts)
                else:
                    # Horizontal karaoke highlight
                    is_ar = any(is_arabic(w["word"]) for w in chunk)
                    if is_ar:
                        # Arabic RTL: Words are spoken in chronological order chunk[0], chunk[1], ...
                        # In Arabic reading layout, reading is Right-to-Left, so the first spoken
                        # word chunk[0] appears on the visual FAR RIGHT, and the last spoken word
                        # appears on the visual FAR LEFT.
                        # Therefore, visual order from Left to Right is reversed(chunk).
                        # We isolate each word with LRM markers around spaces (LRM space LRM) to ensure
                        # FriBidi keeps each word pinned at its exact visual position without collapsing
                        # adjacent untagged words into varying multi-word RTL runs.
                        LRM = "\u200E"
                        sep = f"{LRM} {LRM}"
                        disp_items = [(len(chunk) - 1 - i, chunk[len(chunk) - 1 - i]) for i in range(len(chunk))]
                        line_parts = []
                        for orig_idx, w in disp_items:
                            word_str = w["word"]
                            if orig_idx == active_idx:
                                line_parts.append(f"{{\\c{active_color}}}{word_str}{{\\c{primary_color}}}")
                            else:
                                line_parts.append(word_str)
                        text_line = sep.join(line_parts)
                    else:
                        line_parts = []
                        for idx, w in enumerate(chunk):
                            word_str = w["word"]
                            if idx == active_idx:
                                line_parts.append(f"{{\\c{active_color}}}{word_str}{{\\c{primary_color}}}")
                            else:
                                line_parts.append(word_str)
                        text_line = " ".join(line_parts)

                ass_lines.append(
                    f"Dialogue: 0,{t_w_start},{t_w_end},DynamicWord,,0,0,0,,{anim}{pos_tag}{text_line}"
                )
        else:
            # Standard single block per chunk
            t_start = to_ass_timestamp(c_start)
            t_end = to_ass_timestamp(c_end)
            anim = anim_tag if pop_animation else ""
            if is_stacked:
                text_line = "\\N".join(w["word"] for w in chunk)
            else:
                is_ar = any(is_arabic(w["word"]) for w in chunk)
                if is_ar:
                    LRM = "\u200E"
                    sep = f"{LRM} {LRM}"
                    text_line = sep.join(w["word"] for w in reversed(chunk))
                else:
                    text_line = " ".join(w["word"] for w in chunk)

            ass_lines.append(
                f"Dialogue: 0,{t_start},{t_end},DynamicWord,,0,0,0,,{anim}{pos_tag}{text_line}"
            )

    output_ass_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ass_lines) + "\n")

    return output_ass_path


def burn_dynamic_captions(
    video_path: str,
    ass_path: Path,
    output_video_path: str,
) -> None:
    """Burns dynamic captions onto video using FFmpeg with universal 4:2:0 format."""
    escaped_ass = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles='{escaped_ass}',scale=1280:720,setsar=1,fps=25,format=yuv420p",
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
        description="Add modern dynamic captions with empty space placement and typography styles"
    )
    parser.add_argument("video", help="Input video file")
    parser.add_argument("transcript_json", help="Path to transcript.json file")
    parser.add_argument("-o", "--output", default="captioned_video.mp4", help="Output video path")
    parser.add_argument(
        "--theme",
        choices=["floating_karaoke", "stacked_kinetic", "glassmorphic", "dark_pill", "yellow_pop", "red_alert"],
        default="floating_karaoke",
        help="Caption typography & placement theme (default: floating_karaoke)",
    )
    parser.add_argument(
        "--font",
        default="Cairo",
        help="Font family (Cairo, Readex Pro, IBM Plex Sans Arabic, Segoe UI)",
    )
    parser.add_argument("-w", "--words", type=int, default=2, help="Words per screen (default: 2)")
    parser.add_argument(
        "--position",
        choices=["bottom", "center", "top", "left", "right", "dynamic"],
        default="bottom",
        help="Screen positioning",
    )
    parser.add_argument("--no-pop", action="store_true", help="Disable word bounce effect")

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
        font=args.font,
    )

    burn_dynamic_captions(args.video, temp_ass, args.output)

    if temp_ass.exists():
        temp_ass.unlink()


if __name__ == "__main__":
    main()
