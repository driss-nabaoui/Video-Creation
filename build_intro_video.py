#!/usr/bin/env python3
"""
Build a professional Teaser / Introduction Video from transcript highlights.
Hybrid Alignment Pipeline:
1. Words Ground Truth: Verified, perfectly spelled Arabic text from the transcript.
2. Timing Ground Truth: Exact millisecond acoustic waveforms from audio clip analysis.
3. Typography: Configurable font (Segoe UI, Noto Sans Arabic, etc.), Spacing=0 (connected ligatures).
4. Dynamics: 1-2 words at a time as spoken, hugging background pill box, pop bounce.
5. Transitions: Dynamic multi-clip crossfades (fadewhite, fadeblack, audio acrossfades).
6. Project Architecture: Isolated project directory (projects/<project_id>/) with clips/ and output/ subfolders.
"""

import argparse
import difflib
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Compatibility fix: PyAV 19 removed metadata_errors argument
try:
    import av
    _orig_av_open = av.open
    def _patched_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)
    av.open = _patched_av_open
except Exception:
    pass

import cv2
import ctranslate2
from faster_whisper import WhisperModel
from dynamic_captions import build_dynamic_ass, THEMES
from detect_empty_space import EmptySpaceDetector, QUALITY_PRESETS
from depth_compositor import DepthVideoCompositor


_HAS_NVENC = None


def has_nvenc() -> bool:
    """Check if FFmpeg supports NVIDIA NVENC hardware encoder."""
    global _HAS_NVENC
    if _HAS_NVENC is not None:
        return _HAS_NVENC
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        _HAS_NVENC = "h264_nvenc" in (res.stdout or "")
    except Exception:
        _HAS_NVENC = False
    return _HAS_NVENC


def load_alignment_model(model_name: str = "tiny", device: str = "auto") -> WhisperModel:
    """Load Whisper model with automatic GPU (CUDA) acceleration and CPU fallback."""
    has_cuda = False
    try:
        has_cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        has_cuda = False

    target_device = device.lower() if device else "auto"

    if target_device in ("auto", "cuda") and has_cuda:
        try:
            print(f"[*] Initializing audio alignment engine on GPU (CUDA, float16)...", flush=True)
            return WhisperModel(model_name, device="cuda", compute_type="float16")
        except Exception as e:
            if target_device == "cuda":
                raise RuntimeError(f"CUDA was explicitly requested but failed: {e}")
            print(f"[!] GPU execution failed: {e}. Falling back to CPU...", flush=True)

    threads = os.cpu_count() or 4
    print(f"[*] Initializing audio alignment engine on CPU ({threads} threads, int8)...", flush=True)
    return WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads)


def sanitize_project_name(name: str) -> str:
    """Sanitize string for directory naming."""
    clean = re.sub(r'[\\/*?:"<>|]', "_", name).strip()
    return clean.replace(" ", "_") if clean else "default_project"


def normalize_ar(text: str) -> str:
    """Normalize Arabic text for robust phonetic / lexical sequence matching."""
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().lower()


def find_transcript_words(transcript_path: Optional[str] = None, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load clean ground-truth words from transcript JSON."""
    selected_path = None
    if transcript_path:
        p = Path(transcript_path).resolve()
        if p.is_file():
            selected_path = p
        else:
            print(f"[!] Warning: Specified transcript path does not exist: {p}", file=sys.stderr)

    if not selected_path:
        candidates = []
        if project_id:
            candidates.extend(glob.glob(f"projects/{project_id}/transcripts/*transcript.json"))
            candidates.extend(glob.glob(f"projects/{project_id}/*transcript.json"))
        candidates.extend(glob.glob("transcripts/*transcript.json"))
        candidates.extend(glob.glob("projects/*/*/*transcript.json"))

        if not candidates:
            raise FileNotFoundError(
                "No transcript JSON found! Please specify 'transcript_path' in your config JSON or via --transcript."
            )
        selected_path = Path(candidates[0])
        print(f"[*] Auto-detected transcript JSON: {selected_path}", flush=True)

    with open(selected_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data.get("words", [])
    if not words:
        raise ValueError(f"No 'words' found in transcript: {selected_path}")
    return words


def get_clip_words_with_timing(
    all_words: list,
    start_t: float,
    end_t: float,
    whisper_model: Optional[WhisperModel] = None,
    raw_clip_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts words for the clip with exact acoustic timing.
    - If transcript has valid word timestamps, computes exact relative offsets: (t - start_t).
    - Filters filler words and boundary artifacts.
    - Falls back to Whisper acoustic alignment if timestamps are missing.
    """
    clip_dur = round(end_t - start_t, 2)
    has_timestamps = (
        all_words
        and isinstance(all_words[0], dict)
        and "start" in all_words[0]
        and "end" in all_words[0]
    )

    if has_timestamps:
        extracted = []
        for w in all_words:
            w_st = float(w.get("start", 0.0))
            w_en = float(w.get("end", 0.0))
            # Skip words that barely end at the start of clip (< 0.15s overlap)
            if w_st < start_t and (w_en - start_t) < 0.15:
                continue
            # Skip words that start after the clip ends
            if w_st >= end_t - 0.10:
                continue
            if w_en >= start_t and w_st <= end_t:
                txt = str(w.get("word", "")).strip().strip(">>").strip()
                if not txt or txt in (".", "،", ">>"):
                    continue
                rel_st = max(0.0, round(w_st - start_t, 2))
                rel_en = min(clip_dur, round(w_en - start_t, 2))
                if rel_en <= rel_st:
                    rel_en = round(rel_st + 0.25, 2)
                extracted.append({
                    "word": txt,
                    "start": rel_st,
                    "end": rel_en,
                })
        # If leading word is interviewer acknowledgment, skip it
        if extracted and extracted[0]["word"] in ("صحيح", "صحيح."):
            extracted = extracted[1:]
        if extracted:
            return extracted

    # Fallback to Whisper if transcript had no timestamps
    if whisper_model and raw_clip_path:
        wsp_words = get_audio_timing_words(raw_clip_path, whisper_model)
        return wsp_words

    return []


def cut_raw_clip(src_video: str, start: float, duration: float, out_path: Path):
    """Cut clean raw video clip from source file, using GPU (NVENC/CUDA) when available."""
    if has_nvenc():
        try:
            cmd = [
                "ffmpeg", "-y",
                "-hwaccel", "cuda",
                "-ss", str(start),
                "-i", str(src_video),
                "-t", str(duration),
                "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18",
                "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                str(out_path),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return
        except subprocess.CalledProcessError:
            pass

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(src_video),
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
                "word": w,
                "start": round(curr, 2),
                "end": round(curr + dur, 2),
            })
            curr += dur

    return aligned_words


def get_video_dimensions(video_path: Path) -> Tuple[int, int]:
    """Get video width and height using cv2."""
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (w or 1920, h or 1080)


def detect_clip_empty_box(
    clip_path: Path,
    detector: EmptySpaceDetector,
    sample_count: int = 5,
) -> Tuple[int, int, int, int, str, float]:
    """
    Samples frames across a clip to detect stable clean empty negative space.
    Returns: (x, y, w, h, quadrant, mean_lum)
    """
    cap = cv2.VideoCapture(str(clip_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    if not cap.isOpened() or total_frames <= 0:
        if cap.isOpened():
            cap.release()
        return (int(w * 0.55), int(h * 0.25), int(w * 0.38), int(h * 0.45), "Mid-Right", 140.0)

    step = max(1, total_frames // (sample_count + 1))
    sample_indices = [step * (i + 1) for i in range(sample_count)]

    rects = []
    quads = []
    lums = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            res = detector.process_frame(frame)
            rects.append(res["empty_rect"])
            quads.append(res["quadrant"])
            lums.append(res["mean_lum"])

    cap.release()

    if not rects:
        return (int(w * 0.55), int(h * 0.25), int(w * 0.38), int(h * 0.45), "Mid-Right", 140.0)

    avg_x = int(sum(r[0] for r in rects) / len(rects))
    avg_y = int(sum(r[1] for r in rects) / len(rects))
    avg_w = int(sum(r[2] for r in rects) / len(rects))
    avg_h = int(sum(r[3] for r in rects) / len(rects))
    most_common_quad = max(set(quads), key=quads.count) if quads else "Mid-Right"
    avg_lum = float(sum(lums) / len(lums)) if lums else 140.0

    return (avg_x, avg_y, avg_w, avg_h, most_common_quad, avg_lum)


def scale_box_to_ass(
    box: Tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    ass_w: int = 1280,
    ass_h: int = 720,
) -> Tuple[int, int, int, int]:
    """Convert bounding box from native video resolution to ASS canvas resolution."""
    bx, by, bw, bh = box
    sx = ass_w / frame_w
    sy = ass_h / frame_h
    return (int(bx * sx), int(by * sy), int(bw * sx), int(bh * sy))


def style_clip_with_ass(
    raw_clip: Path,
    ass_path: Path,
    out_styled: Path,
    crf: str = "19",
    preset: str = "fast",
):
    """Burn dynamic ASS subtitles onto the cut clip with universal 4:2:0 format, using GPU (NVENC) when available."""
    escaped_ass = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
    if has_nvenc():
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(raw_clip),
                "-vf", f"subtitles='{escaped_ass}',scale=1280:720,setsar=1,fps=25,format=yuv420p",
                "-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(crf),
                "-c:a", "copy",
                str(out_styled),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return
        except subprocess.CalledProcessError:
            pass

    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_clip),
        "-vf", f"subtitles='{escaped_ass}',scale=1280:720,setsar=1,fps=25,format=yuv420p",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-c:a", "copy",
        str(out_styled),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


VALID_XFADE_TRANSITIONS = {
    "fade", "wipeleft", "wiperight", "wipeup", "wipedown",
    "slideleft", "slideright", "slideup", "slidedown",
    "circlecrop", "rectcrop", "distance", "fadeblack", "fadewhite",
    "radial", "smoothleft", "smoothright", "smoothup", "smoothdown",
    "circleopen", "circleclose", "vertopen", "vertclose",
    "horzopen", "horzclose", "dissolve", "pixelize",
    "diagtl", "diagtr", "diagbl", "diagbr",
    "hlslice", "hrslice", "vuslice", "vdslice",
    "hblur", "fadegrays", "wipetl", "wipetr", "wipebl", "wipebr",
    "squeezeh", "squeezev", "zoomin", "fadefast", "fadeslow",
    "hlwind", "hrwind", "vuwind", "vdwind",
    "coverleft", "coverright", "coverup", "coverdown",
    "revealleft", "revealright", "revealup", "revealdown",
}

TRANSITION_ALIASES = {
    "flash_cut": "fadewhite",
    "dip_to_black": "fadeblack",
    "cross_dissolve": "dissolve",
    "glitch_cut": "pixelize",
    "audio_j_cut": "fadefast",
    "j_cut": "fadefast",
    "l_cut": "fadefast",
    "cut": "fadefast",
    "white_flash": "fadewhite",
    "black_dip": "fadeblack",
    "blur": "hblur",
    "zoom": "zoomin",
}


def resolve_xfade_transition(name: str) -> str:
    """Normalize and resolve user transition name to valid FFmpeg xfade transition."""
    clean = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if clean in VALID_XFADE_TRANSITIONS:
        return clean
    if clean in TRANSITION_ALIASES:
        return TRANSITION_ALIASES[clean]
    print(f"[!] Warning: Unknown xfade transition '{name}', falling back to 'fadeblack'", flush=True)
    return "fadeblack"


def merge_clips_with_transitions(
    clip_paths: List[Path],
    durations: List[float],
    output_video_path: Path,
    trans_dur: float = 0.6,
    transitions: Optional[List[str]] = None,
) -> None:
    """Dynamically merges N clips using xfade and acrossfade transitions, with GPU (NVENC) acceleration."""
    n = len(clip_paths)
    if n == 0:
        raise ValueError("No clips to merge.")

    v_enc = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19"] if has_nvenc() else ["-c:v", "libx264", "-preset", "fast", "-crf", "19"]

    if n == 1:
        # Single clip: simply apply subtle in/out fade
        d = durations[0]
        fade_out_st = max(0.0, d - 0.8)
        filter_complex = (
            f"[0:v]fade=t=in:st=0:d=0.8,fade=t=out:st={fade_out_st:.2f}:d=0.8,format=yuv420p[vout]; "
            f"[0:a]afade=t=in:st=0:d=0.8,afade=t=out:st={fade_out_st:.2f}:d=0.8[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(clip_paths[0]),
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
            *v_enc,
            "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-level", "4.0",
            "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            str(output_video_path),
        ]
        print(f"[*] Exporting single styled clip...", flush=True)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            if has_nvenc():
                cmd[cmd.index("h264_nvenc")] = "libx264"
                subprocess.run(cmd, check=True)
        return

    if not transitions:
        transitions = ["fadewhite", "fadeblack"]

    v_chains = []
    a_chains = []

    curr_offset = durations[0] - trans_dur
    total_effective_dur = durations[0]

    last_v = "[0:v]"
    last_a = "[0:a]"

    resolved_trans_names = []
    for i in range(1, n):
        raw_trans = transitions[(i - 1) % len(transitions)]
        trans = resolve_xfade_transition(raw_trans)
        resolved_trans_names.append(trans)
        next_v = f"[{i}:v]"
        next_a = f"[{i}:a]"
        out_v = f"[v{i}]" if i < n - 1 else "[v_pre]"
        out_a = f"[a{i}]" if i < n - 1 else "[a_pre]"

        v_chains.append(f"{last_v}{next_v}xfade=transition={trans}:duration={trans_dur}:offset={curr_offset:.2f}{out_v}")
        a_chains.append(f"{last_a}{next_a}acrossfade=d={trans_dur}:c1=tri:c2=tri{out_a}")

        last_v = out_v
        last_a = out_a

        total_effective_dur += durations[i] - trans_dur
        if i < n - 1:
            curr_offset += durations[i] - trans_dur

    fade_out_st = max(0.0, total_effective_dur - 0.6)
    v_chains.append(f"[v_pre]fade=t=in:st=0:d=0.25,fade=t=out:st={fade_out_st:.2f}:d=0.6,format=yuv420p[vout]")
    a_chains.append(f"[a_pre]afade=t=in:st=0:d=0.25,afade=t=out:st={fade_out_st:.2f}:d=0.6[aout]")

    filter_complex = "; ".join(v_chains + a_chains)

    cmd = ["ffmpeg", "-y"]
    for c in clip_paths:
        cmd.extend(["-i", str(c)])
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        *v_enc,
        "-pix_fmt", "yuv420p",
        "-profile:v", "high", "-level", "4.0",
        "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        str(output_video_path),
    ])

    print(f"\n[*] Merging {n} clips with transitions ({', '.join(resolved_trans_names)})...", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if has_nvenc():
            print("[!] NVENC merge failed, retrying with CPU libx264...", flush=True)
            cmd = ["ffmpeg", "-y"]
            for c in clip_paths:
                cmd.extend(["-i", str(c)])
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[vout]", "-map", "[aout]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                "-pix_fmt", "yuv420p",
                "-profile:v", "high", "-level", "4.0",
                "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                str(output_video_path),
            ])
            subprocess.run(cmd, check=True)


def build_intro_video(
    config_path: str,
    source_video_override: Optional[str] = None,
    transcript_override: Optional[str] = None,
    project_id_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    font_override: Optional[str] = None,
    style_override: Optional[str] = None,
    quality_override: Optional[str] = None,
    placement_override: Optional[str] = None,
    device: str = "auto",
    show_title: bool = False,
):
    """Main workflow to execute intro build from a JSON project configuration."""
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    # Resolve settings
    settings = config_data.get("settings", {})
    source_video = source_video_override or config_data.get("source_video") or os.environ.get("SOURCE_VIDEO")
    if not source_video or not Path(source_video).is_file():
        raise FileNotFoundError(f"Source video file not found or not specified: '{source_video}'")

    project_id = project_id_override or config_data.get("project_id") or sanitize_project_name(Path(source_video).stem)
    transcript_path = transcript_override or config_data.get("transcript_path")
    clips_config = config_data.get("clips", [])

    if not clips_config:
        raise ValueError("No 'clips' specified in configuration JSON.")

    # Project directories
    base_project_dir = Path(output_dir_override) if output_dir_override else Path("projects") / project_id
    clips_dir = base_project_dir / "clips"
    output_dir = base_project_dir / "output"
    clips_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save snapshot of config used for this build
    with open(base_project_dir / "project_config.json", "w", encoding="utf-8") as f_cfg:
        json.dump(config_data, f_cfg, indent=2, ensure_ascii=False)

    font_name = font_override or settings.get("font", "Cairo")
    theme_default = style_override or settings.get("theme_default", "floating_karaoke")
    quality = (quality_override or settings.get("quality", "high")).lower()
    placement = (placement_override or settings.get("placement", "auto")).lower()
    quality_cfg = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["high"])
    words_per_screen = settings.get("words_per_screen", 5)
    pop_animation = settings.get("pop_animation", True)
    trans_dur = settings.get("transition_duration", 0.6)
    transitions = settings.get("transitions", ["fadewhite", "fadeblack"])

    print("=" * 60, flush=True)
    print(f"  CREATING INTRO PROJECT: [{project_id}]", flush=True)
    print(f"  Project Directory:      {base_project_dir.resolve()}", flush=True)
    print(f"  Source Video:           {source_video}", flush=True)
    print(f"  Clips to process:       {len(clips_config)}", flush=True)
    print(f"  Quality Preset:         {quality} ({quality_cfg['desc']})", flush=True)
    print(f"  Placement Mode:         {placement}", flush=True)
    print(f"  Typography Style:       {theme_default} (Font: {font_name})", flush=True)
    print(f"  GPU Acceleration:       {'Enabled (h264_nvenc)' if has_nvenc() else 'Disabled (libx264 CPU)'}", flush=True)
    print(f"  Show Title in Frame:    {show_title}", flush=True)
    print("=" * 60, flush=True)

    # Initialize empty space detector
    detector = EmptySpaceDetector(quality=quality)

    # Load words
    api_words_all = find_transcript_words(transcript_path=transcript_path, project_id=project_id)
    whisper_model = None  # Lazy-load only if transcript timestamps are missing

    clip_paths = []
    durations = []

    for idx, cfg in enumerate(clips_config, 1):
        clip_id = cfg.get("id", f"clip{idx}")
        start = float(cfg["start"])
        end = float(cfg["end"])
        duration = round(end - start, 2)
        title = cfg.get("title", "")
        clip_theme = cfg.get("theme", style_override or theme_default)

        raw_clip = clips_dir / f"{clip_id}_raw.mp4"
        styled_clip = clips_dir / f"{clip_id}_styled.mp4"
        ass_file = clips_dir / f"{clip_id}_dynamic.ass"

        print(f"\n[*] Processing [{clip_id}]: {start}s -> {end}s ({duration}s)" + (f" | '{title}'" if title else ""), flush=True)

        # 1. Cut clean raw clip
        cut_raw_clip(source_video, start, duration, raw_clip)
        orig_w, orig_h = get_video_dimensions(raw_clip)

        # 2. Empty Space Detection
        ass_box = None
        if placement in ("auto", "empty_space"):
            ex, ey, ew, eh, quad, lum = detect_clip_empty_box(raw_clip, detector)
            ass_box = scale_box_to_ass((ex, ey, ew, eh), orig_w, orig_h)
            print(f"[+] Detected clean empty space: {quad} [{ex}, {ey}, {ew}x{eh}] | Lum: {lum:.1f}", flush=True)

        # 3. Get words with exact acoustic timestamps
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
        print(f"[+] Loaded {len(aligned_words)} words with acoustic timestamps: {' '.join(w['word'] for w in aligned_words[:6])}...", flush=True)

        badge_to_draw = title if (show_title and title) else None

        # 6. Apply transcription typography style
        if clip_theme == "depth_layered":
            print(f"[*] Rendering depth-layered text (speaker masked in front)...", flush=True)
            compositor = DepthVideoCompositor(style="depth_layered", font_name=font_name)
            compositor.render_clip(
                input_video_path=str(raw_clip),
                words=aligned_words,
                output_video_path=str(styled_clip),
                header_badge=badge_to_draw,
                max_duration=duration,
            )
        else:
            # ASS-based styles: floating_karaoke, stacked_kinetic, glassmorphic, dark_pill, yellow_pop, red_alert
            build_dynamic_ass(
                words=aligned_words,
                output_ass_path=ass_file,
                words_per_screen=words_per_screen,
                theme=clip_theme,
                position="bottom",
                target_box=ass_box,
                max_duration=duration,
                pop_animation=pop_animation,
                header_badge=badge_to_draw,
                font=font_name,
            )
            print(f"[*] Burning styled subtitles into {clip_id} ({clip_theme})...", flush=True)
            style_clip_with_ass(
                raw_clip,
                ass_file,
                styled_clip,
                crf=quality_cfg["crf"],
                preset=quality_cfg["preset"],
            )

        clip_paths.append(styled_clip)
        durations.append(duration)
        print(f"[✓] {clip_id} completed.", flush=True)

    # 7. Merge all clips with dynamic transitions
    final_video = output_dir / f"{project_id}_intro.mp4"
    merge_clips_with_transitions(
        clip_paths=clip_paths,
        durations=durations,
        output_video_path=final_video,
        trans_dur=trans_dur,
        transitions=transitions,
    )

    total_time = round(sum(durations) - max(0, len(durations) - 1) * trans_dur, 1)
    print("\n" + "=" * 60, flush=True)
    print(f"[✓] INTRO VIDEO CREATED SUCCESSFULLY!", flush=True)
    print(f"[+] Final Video: {final_video}", flush=True)
    print(f"[+] Total Duration: ~{total_time}s", flush=True)
    print(f"[+] Clips & Assets: {clips_dir}", flush=True)
    print("=" * 60, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build Intro / Teaser Video from external JSON configuration with unique Project IDs"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="clips.json",
        help="Path to clips config JSON file (default: clips.json)",
    )
    parser.add_argument("--video", default=None, help="Override source video path")
    parser.add_argument("--transcript", default=None, help="Override transcript JSON path")
    parser.add_argument("--project-id", default=None, help="Override project ID (creates projects/<project_id>/)")
    parser.add_argument("-o", "--output-dir", default=None, help="Override output directory")
    parser.add_argument("--font", default=None, help="Override subtitle font (e.g. 'Cairo', 'Readex Pro', 'IBM Plex Sans Arabic')")
    parser.add_argument(
        "--style", "--theme",
        dest="style",
        default=None,
        choices=[
            "floating_karaoke", "stacked_kinetic", "glassmorphic", "depth_layered",
            "dark_pill", "yellow_pop", "red_alert", "broadcast_investigative_dark",
            "urgent_red", "suspense_amber", "investigative_blue", "highlight_gold",
        ],
        help="Subtitle typography theme/style (default: from config or floating_karaoke)",
    )
    parser.add_argument(
        "--quality",
        default=None,
        choices=["high", "medium", "low", "1080p", "720p", "480p"],
        help="Output quality preset: high/1080p (CRF 18), medium/720p (CRF 22), low/480p (CRF 28)",
    )
    parser.add_argument(
        "--placement",
        default=None,
        choices=["auto", "empty_space", "bottom", "center", "bottom_center"],
        help="Subtitle placement mode: auto (dynamic empty space) or fixed position (bottom, center, bottom_center)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Whisper alignment device: auto (default), cuda, or cpu",
    )
    parser.add_argument(
        "--show-title",
        action="store_true",
        default=False,
        help="Burn the clip title as a top badge into the frame (default: False/disabled)",
    )

    args = parser.parse_args()

    # If default clips.json does not exist, warn and create example
    if not Path(args.config).is_file():
        example_path = Path("clips.example.json")
        print(
            f"[!] Config file '{args.config}' not found!\n"
            f"[!] You can create one or refer to the template '{example_path}'.\n"
            f"Usage: python build_intro_video.py path/to/your_config.json",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        build_intro_video(
            config_path=args.config,
            source_video_override=args.video,
            transcript_override=args.transcript,
            project_id_override=args.project_id,
            output_dir_override=args.output_dir,
            font_override=args.font,
            style_override=args.style,
            quality_override=args.quality,
            placement_override=args.placement,
            device=args.device,
            show_title=args.show_title,
        )
    except KeyboardInterrupt:
        print("\nExited by user.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
