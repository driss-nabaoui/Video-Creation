#!/usr/bin/env python3
r"""
Depth Layering & Glassmorphic Video Compositor
==============================================
Renders high-end editorial caption effects that require pixel-level frame manipulation:

1. Depth-Layered Text:
   - Positions large bold text inside clean negative space.
   - Segments the speaker via MediaPipe.
   - Masks the speaker IN FRONT of the text (Head/Shoulders overlap the typography).

2. True Glassmorphism:
   - Samples video background behind the caption box.
   - Applies true Gaussian blur (frosted glass effect).
   - Blends with 40% dark tint and rounded highlight border.

Features:
- Piped directly to FFmpeg libx264 for fast universal MP4 export.
- Preserves original audio seamlessly.
- Supports Cairo, Readex Pro, IBM Plex Sans Arabic.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detect_empty_space import EmptySpaceDetector

FONT_MAP = {
    "cairo": "fonts/Cairo-Bold.ttf",
    "readex pro": "fonts/ReadexPro-Bold.ttf",
    "ibm plex sans arabic": "fonts/IBMPlexSansArabic-Bold.ttf",
}


def get_font_path(font_name: str) -> str:
    key = font_name.lower().strip()
    if key in FONT_MAP and Path(FONT_MAP[key]).is_file():
        return FONT_MAP[key]
    for p in Path("fonts").glob("*.ttf"):
        if key in p.name.lower():
            return str(p)
    return "/usr/share/fonts/noto/NotoKufiArabic-Bold.ttf"


class DepthVideoCompositor:
    def __init__(
        self,
        style: str = "depth_layered",
        font_name: str = "Cairo",
        highlight_color: Tuple[int, int, int] = (255, 215, 0),  # Gold (RGB)
        primary_color: Tuple[int, int, int] = (255, 255, 255),    # White (RGB)
    ):
        self.style = style
        self.font_path = get_font_path(font_name)
        self.highlight_color = highlight_color
        self.primary_color = primary_color
        self.detector = EmptySpaceDetector(quality="medium")

    def render_clip(
        self,
        input_video_path: str,
        words: List[Dict[str, Any]],
        output_video_path: str,
        header_badge: Optional[str] = None,
        max_duration: Optional[float] = None,
    ) -> Path:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open input video: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = Path(output_video_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temp_video = out_path.parent / f"temp_{out_path.name}"

        # Setup FFmpeg rawvideo pipe for video stream
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", f"{fps:.2f}",
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "19",
            "-pix_fmt", "yuv420p",
            str(temp_video),
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Pre-filter clean words
        clean_words = []
        for w in sorted(words, key=lambda x: x.get("start", 0.0)):
            txt = w.get("word", "").strip().strip(">>").strip()
            if txt:
                clean_words.append({
                    "word": txt,
                    "start": max(0.0, float(w.get("start", 0.0))),
                    "end": max(0.0, float(w.get("end", 0.0))),
                })

        # Font sizing based on frame height
        if self.style == "depth_layered":
            font_size = max(38, int(height * 0.058))  # Large bold editorial font
        else:
            font_size = max(32, int(height * 0.045))

        try:
            pil_font = ImageFont.truetype(self.font_path, font_size)
            badge_font = ImageFont.truetype(self.font_path, max(22, int(height * 0.026)))
        except Exception:
            pil_font = ImageFont.load_default()
            badge_font = ImageFont.load_default()

        # Pre-chunk into natural phrases (4-6 words)
        from dynamic_captions import chunk_words_by_phrase
        phrase_chunks = chunk_words_by_phrase(clean_words, target_words=5)

        frame_idx = 0
        t0 = time.time()
        print(f"[*] Compositing '{self.style}' clip ({total_frames} frames)...", flush=True)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                curr_time = frame_idx / fps
                if max_duration and curr_time > max_duration:
                    break

                # 1. Detect person mask & empty space
                analysis = self.detector.process_frame(frame)
                person_mask = analysis["person_mask"]
                ex, ey, ew, eh = analysis["empty_rect"]

                # 2. Find active phrase at current timecode
                active_chunk = []
                for c in phrase_chunks:
                    if c[0]["start"] <= curr_time <= c[-1]["end"] + 0.30:
                        active_chunk = c
                        break

                # Render caption if active
                if active_chunk:
                    text_str = " ".join(w["word"] for w in active_chunk)
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)

                    # Compute text dimensions with RTL support
                    bbox = draw.textbbox((0, 0), text_str, font=pil_font, direction="rtl")
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]

                    if self.style == "depth_layered":
                        # Anchor text in upper-mid space overlapping the subject boundary
                        tx = ex + (ew - tw) // 2
                        ty = ey + (eh - th) // 2
                        tx = max(30, min(width - tw - 30, tx))
                        ty = max(40, min(height - th - 50, ty))

                        # Draw bold text with soft drop shadow and RTL shaping
                        draw.text((tx + 3, ty + 3), text_str, fill=(0, 0, 0), font=pil_font, direction="rtl")
                        draw.text((tx, ty), text_str, fill=self.highlight_color, font=pil_font, direction="rtl")

                        # Top badge
                        if header_badge:
                            draw.text((ex + 20, ey + 15), header_badge, fill=(255, 255, 255), font=badge_font, direction="rtl")

                        text_layer = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                        # Depth Compositing: Person in FRONT of text!
                        # Person pixels (mask==1) retain the original frame; background shows the text!
                        p_mask_3d = person_mask[:, :, None]
                        final_frame = np.where(p_mask_3d == 1, frame, text_layer)

                    elif self.style == "glassmorphic":
                        # True frosted glass: Blur background under pill box
                        pad_x = 28
                        pad_y = 16
                        pw = tw + pad_x * 2
                        ph = th + pad_y * 2
                        px = max(20, min(width - pw - 20, ex + (ew - pw) // 2))
                        py = max(20, min(height - ph - 20, ey + (eh - ph) // 2))

                        # 1. Background Gaussian blur
                        blurred = cv2.GaussianBlur(frame, (35, 35), 0)
                        roi = blurred[py : py + ph, px : px + pw]

                        # 2. 40% dark charcoal tint
                        tint = np.full_like(roi, (18, 18, 18))
                        frosted_pill = cv2.addWeighted(roi, 0.60, tint, 0.40, 0)
                        frame[py : py + ph, px : px + pw] = frosted_pill

                        # 3. Draw rounded 1px highlight border & text
                        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        p_draw = ImageDraw.Draw(pil_frame)
                        p_draw.rounded_rectangle(
                            [px, py, px + pw, py + ph],
                            radius=min(22, ph // 2),
                            outline=(255, 255, 255, 130),
                            width=2,
                        )

                        # Text positioning inside pill
                        tx = px + (pw - tw) // 2
                        ty = py + (ph - th) // 2 - bbox[1]
                        p_draw.text((tx + 2, ty + 2), text_str, fill=(0, 0, 0, 200), font=pil_font, direction="rtl")
                        p_draw.text((tx, ty), text_str, fill=self.primary_color, font=pil_font, direction="rtl")

                        if header_badge:
                            p_draw.text((px, py - 35), header_badge, fill=(255, 215, 0), font=badge_font, direction="rtl")

                        final_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
                    else:
                        final_frame = frame
                else:
                    final_frame = frame

                proc.stdin.write(final_frame.tobytes())
                frame_idx += 1

        finally:
            cap.release()
            if proc and proc.stdin:
                proc.stdin.close()
                proc.wait()

        # Merge audio from source video with generated video stream
        print(f"[*] Muxing original audio into {out_path}...", flush=True)
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_video),
            "-i", input_video_path,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            "-movflags", "+faststart",
            str(out_path),
        ]
        subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        if temp_video.exists():
            temp_video.unlink()

        elapsed = time.time() - t0
        print(f"[✓] Clip rendered in {elapsed:.1f}s ({frame_idx / elapsed:.1f} FPS) -> {out_path}", flush=True)
        return out_path


def main():
    parser = argparse.ArgumentParser(description="Depth Layering & Glassmorphic Video Compositor")
    parser.add_argument("video", help="Input video clip path")
    parser.add_argument("transcript_json", help="Path to transcript.json")
    parser.add_argument("-o", "--output", default="depth_styled_clip.mp4", help="Output video path")
    parser.add_argument(
        "--style",
        choices=["depth_layered", "glassmorphic"],
        default="depth_layered",
        help="Compositing style (default: depth_layered)",
    )
    parser.add_argument("--font", default="Cairo", help="Font family (Cairo, Readex Pro, IBM Plex Sans Arabic)")

    args = parser.parse_args()

    with open(args.transcript_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data.get("words", [])
    compositor = DepthVideoCompositor(style=args.style, font_name=args.font)
    compositor.render_clip(
        input_video_path=args.video,
        words=words,
        output_video_path=args.output,
    )


if __name__ == "__main__":
    main()
