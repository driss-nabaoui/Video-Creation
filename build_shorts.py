#!/usr/bin/env python3
"""
Automated 9:16 Vertical Shorts Pipeline (TikTok, Instagram Reels, YouTube Shorts).
Builds mobile vertical videos from existing clips.json project files using:
1. Speaker-Centered 9:16 Cropping via MediaPipe Person Segmentation.
2. Native RTL Arabic Dynamic Subtitles (safe-zone positioning above mobile UI).
3. Persistent Top Hook Banner (safe-zone top placement, auto-extracted from title).
4. Dual Output Modes:
   - shorts_batch: Exports each clip as an individual standalone vertical Short.
   - short_stitched: Stitches all clips into one fast-paced vertical reel with transitions.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project directory to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Import battle-tested existing components
from build_intro_video import (
    has_nvenc,
    sanitize_project_name,
    find_transcript_words,
    get_clip_words_with_timing,
    cut_raw_clip,
    get_video_dimensions,
    load_alignment_model,
    merge_clips_with_transitions,
)
from detect_empty_space import (
    EmptySpaceDetector,
    get_speaker_crop_x,
    generate_916_crop_filter,
)
from dynamic_captions import (
    build_dynamic_ass,
    extract_hook_banner_text,
    THEMES,
)


VERTICAL_RESOLUTIONS = {
    "720p": (720, 1280),
    "1080p": (1080, 1920),
    "high": (1080, 1920),
    "medium": (720, 1280),
    "low": (480, 854),
}


def render_916_short_clip(
    raw_clip: Path,
    ass_path: Path,
    out_short: Path,
    orig_w: int,
    orig_h: int,
    center_x: Optional[int] = None,
    crop_mode: str = "speaker_track",
    target_w: int = 720,
    target_h: int = 1280,
    crf: str = "19",
    preset: str = "fast",
) -> None:
    """
    Renders a 9:16 vertical short clip with burned ASS captions and top hook banner in a single FFmpeg pass.
    Utilizes NVIDIA NVENC (h264_nvenc) when available with automatic CPU fallback.
    """
    raw_clip = Path(raw_clip).resolve()
    out_short = Path(out_short).resolve()
    out_short.parent.mkdir(parents=True, exist_ok=True)

    escaped_ass = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")

    if crop_mode == "blur_pad":
        filter_complex = (
            f"[0:v]split[fg_raw][bg_raw];"
            f"[bg_raw]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=25:5,colorchannelmixer=aa=0.75[bg_blur];"
            f"[fg_raw]scale={target_w}:-2[fg_fit];"
            f"[bg_blur][fg_fit]overlay=(W-w)/2:(H-h)/2,setsar=1,subtitles='{escaped_ass}',format=yuv420p[vout]"
        )
    else:
        crop_filter = generate_916_crop_filter(
            src_w=orig_w,
            src_h=orig_h,
            target_w=target_w,
            target_h=target_h,
            center_x=center_x,
            mode=crop_mode,
        )
        filter_complex = f"[0:v]{crop_filter},subtitles='{escaped_ass}',format=yuv420p[vout]"

    v_enc = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(crf)] if has_nvenc() else ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]

    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_clip),
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "0:a?",
        *v_enc,
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart",
        str(out_short),
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        if has_nvenc():
            # CPU fallback
            cmd_cpu = [
                "ffmpeg", "-y",
                "-i", str(raw_clip),
                "-filter_complex", filter_complex,
                "-map", "[vout]", "-map", "0:a?",
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                "-movflags", "+faststart",
                str(out_short),
            ]
            subprocess.run(cmd_cpu, check=True)
        else:
            raise


def build_shorts(
    config_path: str,
    stitched: bool = False,
    crop_mode_override: Optional[str] = None,
    quality_override: Optional[str] = None,
    style_override: Optional[str] = None,
    font_override: Optional[str] = None,
    words_per_screen_override: Optional[int] = None,
    show_hook_banner_override: Optional[bool] = None,
    hook_duration_override: Optional[float] = None,
    output_dir_override: Optional[str] = None,
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Main orchestration function to build vertical 9:16 Shorts from clips.json.
    """
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    settings = config_data.get("settings", {})
    source_video = config_data.get("source_video") or os.environ.get("SOURCE_VIDEO")
    if not source_video or not Path(source_video).is_file():
        raise FileNotFoundError(f"Source video file not found or not specified: '{source_video}'")

    project_id = config_data.get("project_id") or sanitize_project_name(Path(source_video).stem)
    transcript_path = config_data.get("transcript_path")
    clips_config = config_data.get("clips", [])

    if not clips_config:
        raise ValueError("No 'clips' specified in configuration JSON.")

    # Determine mode: CLI flag takes precedence over config
    is_stitched_mode = stitched or (settings.get("mode") == "short_stitched")

    # Project directories
    base_project_dir = Path(output_dir_override) if output_dir_override else Path("projects") / project_id
    raw_clips_dir = base_project_dir / "clips"
    shorts_dir = base_project_dir / "shorts"
    raw_clips_dir.mkdir(parents=True, exist_ok=True)
    shorts_dir.mkdir(parents=True, exist_ok=True)

    # Resolution & Framing options
    quality_key = (quality_override or settings.get("quality", "720p")).lower()
    target_w, target_h = VERTICAL_RESOLUTIONS.get(quality_key, VERTICAL_RESOLUTIONS["720p"])
    default_crop_mode = (crop_mode_override or settings.get("crop_mode", "speaker_track")).lower()
    font_name = font_override or settings.get("font", "Cairo")
    theme_default = style_override or settings.get("theme_default", "floating_karaoke")
    words_per_screen = words_per_screen_override or settings.get("words_per_screen", 3)
    pop_animation = settings.get("pop_animation", True)
    trans_dur = settings.get("transition_duration", 0.5)
    transitions = settings.get("transitions", ["fadewhite", "fadeblack", "dissolve"])

    # Hook banner options
    show_hook_banner = show_hook_banner_override if show_hook_banner_override is not None else settings.get("show_hook_banner", True)
    default_hook_duration = hook_duration_override if hook_duration_override is not None else settings.get("hook_duration")

    print("\n" + "=" * 65, flush=True)
    print(f"  AUTOMATED 9:16 VERTICAL SHORTS PIPELINE", flush=True)
    print(f"  Project ID:       [{project_id}]", flush=True)
    print(f"  Mode:             {'Stitched Single Reel (--stitched)' if is_stitched_mode else 'Batch Standalone Shorts (1 clip = 1 Short)'}", flush=True)
    print(f"  Resolution:       {target_w}x{target_h} (Aspect Ratio: 9:16)", flush=True)
    print(f"  Crop Mode:        {default_crop_mode}", flush=True)
    print(f"  Theme / Font:     {theme_default} / {font_name}", flush=True)
    print(f"  Hook Banner:      {'Enabled' if show_hook_banner else 'Disabled'}", flush=True)
    print(f"  GPU NVENC:        {'Enabled (h264_nvenc)' if has_nvenc() else 'Disabled (libx264 CPU)'}", flush=True)
    print(f"  Clips to Process: {len(clips_config)}", flush=True)
    print("=" * 65 + "\n", flush=True)

    # Initialize MediaPipe detector for speaker tracking
    detector = EmptySpaceDetector(quality="medium")

    # Load transcript words
    api_words_all = find_transcript_words(transcript_path=transcript_path, project_id=project_id)
    whisper_model = None

    exported_shorts = []
    durations = []

    for idx, cfg in enumerate(clips_config, 1):
        clip_id = cfg.get("id", f"clip_{idx:02d}")
        start = float(cfg["start"])
        end = float(cfg["end"])
        duration = round(end - start, 2)
        title = cfg.get("title", "")
        clip_theme = cfg.get("theme", style_override or theme_default)
        clip_crop_mode = cfg.get("crop_mode", default_crop_mode)

        raw_clip = raw_clips_dir / f"{clip_id}_raw.mp4"
        short_clip = shorts_dir / f"{clip_id}.mp4"
        ass_file = shorts_dir / f"{clip_id}_916.ass"

        print(f"[*] Processing [{clip_id}]: {start:.1f}s -> {end:.1f}s ({duration:.1f}s)", flush=True)

        # 1. Cut raw horizontal clip
        cut_raw_clip(source_video, start, duration, raw_clip)
        orig_w, orig_h = get_video_dimensions(raw_clip)

        # 2. Determine Speaker Horizontal Center (X) for 9:16 framing
        speaker_x = None
        if clip_crop_mode == "speaker_track":
            speaker_x = get_speaker_crop_x(
                video_path=raw_clip,
                start_time=0.0,
                end_time=duration,
                sample_frames=5,
                detector=detector,
            )
            print(f"    -> Speaker tracked at X={speaker_x} (Frame: {orig_w}x{orig_h})", flush=True)
        elif clip_crop_mode == "center":
            speaker_x = orig_w // 2
            print(f"    -> Center crop mode at X={speaker_x}", flush=True)
        elif clip_crop_mode == "blur_pad":
            print(f"    -> Blur-pad background framing mode", flush=True)

        # 3. Load acoustic word timestamps
        has_timestamps = (
            api_words_all
            and isinstance(api_words_all[0], dict)
            and "start" in api_words_all[0]
            and "end" in api_words_all[0]
        )
        if not has_timestamps and whisper_model is None:
            whisper_model = load_alignment_model("tiny", device=device)

        aligned_words = get_clip_words_with_timing(
            all_words=api_words_all,
            start_t=start,
            end_t=end,
            whisper_model=whisper_model,
            raw_clip_path=raw_clip,
        )

        # 4. Resolve Hook Banner Text
        banner_text = None
        if show_hook_banner:
            banner_text = cfg.get("hook_banner") or extract_hook_banner_text(title)
            if banner_text:
                print(f"    -> Hook Banner: \"{banner_text}\"", flush=True)

        clip_hook_dur = cfg.get("hook_duration", default_hook_duration)

        # 5. Build Vertical ASS Subtitles (9:16 layout)
        build_dynamic_ass(
            words=aligned_words,
            output_ass_path=ass_file,
            words_per_screen=words_per_screen,
            theme=clip_theme,
            position="bottom",
            target_box=None,
            max_duration=duration,
            pop_animation=pop_animation,
            font=font_name,
            aspect_ratio="9:16",
            target_resolution=(target_w, target_h),
            hook_banner=banner_text,
            hook_duration=clip_hook_dur,
        )

        # 6. Render 9:16 vertical short in single pass
        print(f"    -> Slicing & burning 9:16 Short ({target_w}x{target_h})...", flush=True)
        render_916_short_clip(
            raw_clip=raw_clip,
            ass_path=ass_file,
            out_short=short_clip,
            orig_w=orig_w,
            orig_h=orig_h,
            center_x=speaker_x,
            crop_mode=clip_crop_mode,
            target_w=target_w,
            target_h=target_h,
            crf="19",
            preset="fast",
        )

        exported_shorts.append(short_clip)
        durations.append(duration)
        print(f"[✓] Completed Short: {short_clip.name} ({short_clip.stat().st_size / (1024*1024):.2f} MB)\n", flush=True)

    # 7. If stitched mode requested, merge all clips into one vertical reel
    stitched_output = None
    if is_stitched_mode and len(exported_shorts) > 0:
        stitched_output = shorts_dir / f"{project_id}_stitched_short.mp4"
        print(f"[*] Merging {len(exported_shorts)} clips into single vertical reel...", flush=True)
        merge_clips_with_transitions(
            clip_paths=exported_shorts,
            durations=durations,
            output_video_path=stitched_output,
            trans_dur=trans_dur,
            transitions=transitions,
        )
        total_time = round(sum(durations) - max(0, len(durations) - 1) * trans_dur, 1)
        print(f"[✓] STITCHED REEL EXPORTED: {stitched_output} (~{total_time}s)", flush=True)

    print("=" * 65, flush=True)
    print(f"  SHORTS GENERATION COMPLETE", flush=True)
    print(f"  Output Folder: {shorts_dir.resolve()}", flush=True)
    if is_stitched_mode and stitched_output:
        print(f"  Final Stitched Reel: {stitched_output.resolve()}", flush=True)
    else:
        print(f"  Standalone Shorts:   {len(exported_shorts)} files exported", flush=True)
    print("=" * 65 + "\n", flush=True)

    return {
        "project_id": project_id,
        "mode": "short_stitched" if is_stitched_mode else "shorts_batch",
        "shorts_dir": str(shorts_dir),
        "exported_shorts": [str(p) for p in exported_shorts],
        "stitched_reel": str(stitched_output) if stitched_output else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated 9:16 Vertical Shorts Pipeline (TikTok / Reels / YouTube Shorts)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to project clips.json configuration file",
    )
    parser.add_argument(
        "--stitched",
        action="store_true",
        help="Merge all clips into ONE continuous vertical Short reel with transitions",
    )
    parser.add_argument(
        "--crop-mode",
        type=str,
        default=None,
        choices=["speaker_track", "blur_pad", "center"],
        help="Vertical framing mode: 'speaker_track' (MediaPipe speaker centering), 'blur_pad' (blurred background), or 'center'",
    )
    parser.add_argument(
        "--quality",
        type=str,
        default=None,
        choices=["720p", "1080p"],
        help="Output vertical resolution: 720p (720x1280) or 1080p (1080x1920)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        choices=list(THEMES.keys()),
        help="Override caption theme (e.g. urgent_red, floating_karaoke, suspense_amber, glassmorphic)",
    )
    parser.add_argument(
        "--font",
        type=str,
        default=None,
        help="Override typography font name (e.g. Cairo, Readex Pro, IBM Plex Sans Arabic)",
    )
    parser.add_argument(
        "--words-per-screen",
        type=int,
        default=None,
        help="Number of words per subtitle screen (default: 3 for 9:16 vertical)",
    )
    parser.add_argument(
        "--no-hook-banner",
        action="store_true",
        help="Disable rendering top hook banner badge",
    )
    parser.add_argument(
        "--hook-duration",
        type=float,
        default=None,
        help="Duration in seconds to display the top hook banner (default: full clip duration)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Hardware accelerator to use for acoustic alignment",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    build_shorts(
        config_path=args.config,
        stitched=args.stitched,
        crop_mode_override=args.crop_mode,
        quality_override=args.quality,
        style_override=args.style,
        font_override=args.font,
        words_per_screen_override=args.words_per_screen,
        show_hook_banner_override=False if args.no_hook_banner else None,
        hook_duration_override=args.hook_duration,
        output_dir_override=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
