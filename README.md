# Video Creation & Automated Teaser Pipeline

An end-to-end toolkit to transcribe long-form videos, detect clean background negative space, generate RTL Arabic dynamic captions, and build teaser / intro videos with GPU-accelerated rendering.

---

## ⚡ Highlights

- **Hardware Acceleration**: NVIDIA NVENC + CUDA (`h264_nvenc`) encoding & decoding (~**3.4x faster**, rendering transitions at **218+ FPS**), with automatic CPU fallback.
- **High-Tension Scene Detector (`find_tension.py`)**: Automated parliamentary & debate analyzer detecting shouting, gavel bangs, cross-talk, and hostility via multi-signal fusion (Acoustic 50% + Overlap 30% + Hostility 20%), exporting ranked video clips and timestamps.
- **Vertical 9:16 Shorts Pipeline (`build_shorts.py`)**: Automated TikTok/Reels/Shorts generator using MediaPipe speaker tracking, mobile safe-zone RTL captions, and top hook banners (batch or stitched).
- **Teaser Video Generator (`build_intro_video.py`)**: Cuts soundbites, applies dynamic typography, and stitches scenes with smooth transitions from a simple `clips.json`.
- **Broadcast Editorial Themes**: `urgent_red`, `suspense_amber`, `investigative_blue`, `highlight_gold`, `floating_karaoke`, `glassmorphic`, `stacked_kinetic`.
- **RTL Arabic Stability**: Zero word jumping or permutation using bidirectional Unicode LRM isolation and native ligature shaping (Cairo, Readex Pro).
- **Clean Negative Space Detection (`detect_empty_space.py`)**: MediaPipe person segmentation + Largest Empty Rectangle (LER) algorithm to position captions collision-free.
- **Live Video Matting (`rvm_matting_live.py`)**: Robust Video Matting (RVM) in PyTorch FP16 for real-time depth-layered typography (speaker masked in front of text).
- **Fast Word-Level Transcription**: Instant YouTube transcripts via API (3s) or local Whisper (`faster-whisper` GPU/CPU).

---

## 🚀 Quick Start

### 1. Detect High-Tension Parliamentary Debate Scenes
```bash
# Analyze full parliamentary video and export ranked tension clips + timestamps.json
python find_tension.py --input debate_video.mp4 --threshold 60.0

# Use pre-existing transcript (bypasses Whisper GPU/CPU inference completely):
python find_tension.py --input debate_video.mp4 --transcript projects/teaser/transcripts/video_transcript.json

# Optional flags (lossless cutting, custom padding, HuggingFace Pyannote token):
python find_tension.py --input debate_video.mp4 --threshold 55.0 --padding-before 4.0 --padding-after 3.0 --merge-gap 8.0 --lossless
```

### 2. Build an Intro / Teaser Video (GPU Accelerated - 16:9)
Configure your soundbites in `clips.json` and build the final teaser:

```bash
# Build teaser video from clips.json (runs on NVENC GPU)
python build_intro_video.py clips.json

# Optional overrides:
python build_intro_video.py clips.json --font "Cairo" --style urgent_red --quality 1080p
```

### 3. Generate Vertical 9:16 Shorts (TikTok / Reels / YouTube Shorts)
Generate vertical mobile videos from the same `clips.json` with speaker-centered tracking and top hook banners:

```bash
# Export every clip in clips.json as an individual standalone 9:16 Short (Default)
python build_shorts.py clips.json

# Export all clips stitched together with transitions into ONE vertical reel:
python build_shorts.py clips.json --stitched

# Optional framing & style overrides:
python build_shorts.py clips.json --crop-mode speaker_track --quality 720p --style urgent_red --words-per-screen 3
```

### 4. Transcription with Word Timestamps
```bash
# Instant transcript from YouTube URL (3 seconds)
python youtube_api.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Or transcribe local file using Faster-Whisper (CUDA / CPU)
python transcribe.py "video.mp4" --model base
```

### 5. Live Robust Video Matting (RVM) Preview
```bash
# Interactive live preview with depth-layered typography
python rvm_matting_live.py "video.mp4" --text "عنوان المشهد"
```

### 6. Interactive Empty Space Detector
```bash
# Live preview of speaker segmentation and empty background space
python detect_empty_space.py "video.mp4" --quality medium
```

---

## 🎨 Editorial Subtitle Themes

| Theme | Best Used For | Text / Highlight Colors |
| :--- | :--- | :--- |
| `urgent_red` | Breaking news, high tension, military clash | White + Vivid Crimson Red |
| `suspense_amber` | Dramatic questions, rising stakes, ultimatum | White + Electric Golden Amber |
| `investigative_blue` | Diplomatic leaks, intelligence, Foreign Affairs | White + Electric Cyan / Sky Blue |
| `highlight_gold` | Key statements, counter-intuitive facts | White + Luminous Gold |
| `floating_karaoke` | Default clean boxless floating subtitle | White + Warm Gold (Drop Shadow) |
| `stacked_kinetic` | Tight lateral spaces beside speaker | 2-3 word vertical stack |
| `glassmorphic` | Modern frosted translucent dark pill | Translucent 40% Charcoal Box |

---

## ⚙️ Configuration (`clips.json`)

```json
{
  "project_id": "teaser_01",
  "source_video": "/path/to/video.mp4",
  "transcript_path": "projects/teaser_01/transcripts/video_transcript.json",
  "settings": {
    "theme_default": "floating_karaoke",
    "font": "Cairo-Bold",
    "quality": "1080p",
    "placement": "auto",
    "words_per_screen": 4,
    "transition_duration": 0.4,
    "transitions": ["flash_cut", "dip_to_black", "cross_dissolve", "glitch_cut"]
  },
  "clips": [
    {
      "id": "clip_01",
      "start": 120.5,
      "end": 132.0,
      "title": "Scene 1",
      "theme": "urgent_red"
    }
  ]
}
```

> **Supported Transitions**: `flash_cut` (`fadewhite`), `dip_to_black` (`fadeblack`), `cross_dissolve` (`dissolve`), `glitch_cut` (`pixelize`), `audio_J_cut` (`fadefast`), and all 57 standard FFmpeg `xfade` filters.

---

## 📁 Repository Structure

- [`build_shorts.py`](build_shorts.py) - Automated 9:16 vertical Shorts pipeline (speaker tracking, safe-zone captions, hook banners, batch/stitched).
- [`find_tension.py`](find_tension.py) - CLI pipeline for parliamentary debate tension detection, scene ranking, and cutting.
- [`tension_engine/`](tension_engine/) - Modular multi-signal tension engine (acoustic strain, overlap detection, transcript hostility, scene merging, fast video slicing).
- [`build_intro_video.py`](build_intro_video.py) - Master teaser builder (cutting, ASS styling, transition merging with NVENC).
- [`dynamic_captions.py`](dynamic_captions.py) - ASS subtitle generator with Arabic RTL stabilization & karaoke animation.
- [`detect_empty_space.py`](detect_empty_space.py) - MediaPipe human segmentation & Largest Empty Rectangle (LER) analyzer.
- [`rvm_matting_live.py`](rvm_matting_live.py) - Robust Video Matting (RVM) GPU matting and depth-layered text compositor.
- [`depth_compositor.py`](depth_compositor.py) - Background/foreground video layer compositor.
- [`transcribe.py`](transcribe.py) - Faster-Whisper acoustic word timestamp extractor.
- [`youtube_api.py`](youtube_api.py) - High-speed YouTube transcript fetcher.
- [`video_tools.py`](video_tools.py) - Core video clipping and utility helpers.
