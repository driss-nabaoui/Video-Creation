# Video Transcription with Word-Level Timestamps (CPU-Optimized)

A Python tool designed to transcribe video/audio files with **exact word-level timestamps (in seconds)**, specially optimized for machines **without a dedicated GPU**.

---

## 🚀 Hardware Acceleration (Auto GPU & CPU)

This tool automatically detects and leverages your hardware:
1. **NVIDIA GPU (CUDA)**: Automatically detected and used with `float16` precision for blazing fast transcription (typically **15x-30x real-time**).
2. **CPU Fallback / Execution**: If no GPU is present (or if `--device cpu` is passed), it utilizes **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** with **CTranslate2**, `int8` quantization, and multi-threading across all CPU cores.
3. **Voice Activity Detection (VAD) Filtering**: Automatically skips silence.

---

## 📁 Files Included

- [transcribe.py](transcribe.py): The main CLI transcription tool (auto-detects local files or YouTube URLs).
- [youtube_api.py](youtube_api.py): Instant YouTube transcript API fetcher (3 seconds for 50+ min videos).
- [dynamic_captions.py](dynamic_captions.py): Creates dynamic word-by-word captions with hugging background boxes, pop animation, and empty space placement.
- [detect_empty_space.py](detect_empty_space.py): MediaPipe segmentation & Largest Empty Rectangle (LER) detector for clean background caption placement.
- [depth_compositor.py](depth_compositor.py): Depth-layered video compositor placing text behind the speaker.
- [build_intro_video.py](build_intro_video.py): Builds a teaser/introduction video with scene transitions and dynamic captions.
- [video_tools.py](video_tools.py): Video editing helper functions (clipping, searching words, burning animated subtitles).

---

## 🛠️ Usage

### 1. Instant Transcription via YouTube URL (⚡ 3 Seconds)

No CPU or GPU wait time! Fetches the entire transcript with word-level timestamps in seconds:

```powershell
# Using transcribe.py (auto-detects YouTube URL)
.venv\Scripts\python transcribe.py "https://www.youtube.com/watch?v=DBMFJZDEhm4"

# Or using the dedicated youtube_api.py:
.venv\Scripts\python youtube_api.py "https://www.youtube.com/watch?v=DBMFJZDEhm4"
```

### 3. Dynamic Word-by-Word Captions (TikTok / Reels Style)

Burn dynamic short-chunk captions with hugging background boxes and pop animations (no full lines):

```powershell
.venv\Scripts\python dynamic_captions.py "input_video.mp4" "transcripts/my_transcript.json" -o "output.mp4" --words 2 --theme dark_pill --position bottom
```

Options:
- `-w, --words`: Words per screen (default: `2`). Set to `1` for single-word burst or `3` for triples.
- `--theme`: `dark_pill` (semi-transparent charcoal pill), `yellow_pop` (Alex Hormozi vibrant yellow), `red_alert` (crimson box), `neon_cyan` (deep navy/cyan).
- `--position`: `bottom` (lower-third), `center` (middle screen), `dynamic` (subtle alternating vertical bounce).
- `--no-pop`: Disable the 15% spring pop bounce when words change.

### 5. Clean Empty Space Detection & Person Highlighting (Live Preview)

Run live segment detection to find the cleanest unobstructed negative space in the background and highlight the speaker with neon contours:

```bash
# Basic run with video input (defaults to medium quality):
python detect_empty_space.py "video.mp4"

# Adjust output & processing quality (high | medium | low):
python detect_empty_space.py "video.mp4" --quality high

# Save a processed demo video:
python detect_empty_space.py "video.mp4" --quality medium --save
```

#### Quality Presets:
- `--quality high`: Full resolution (1080p), fine LER grid, visually lossless H.264 (CRF 18).
- `--quality medium` (default): 720p scaling, balanced LER grid, crisp H.264 (CRF 22).
- `--quality low`: 480p scaling, fast LER grid, high-speed H.264 (CRF 28).

#### Interactive Keyboard Controls:
- <kbd>P</kbd>: Toggle Person Highlight (neon outline + soft silhouette tint)
- <kbd>E</kbd>: Toggle Empty Space Bounding Box
- <kbd>C</kbd>: Toggle Visual Clutter / Canny edge density overlay
- <kbd>B</kbd>: Toggle Raw Binary Mask overlay
- <kbd>H</kbd>: Toggle Real-time Telemetry HUD
- <kbd>Space</kbd>: Pause / Resume playback
- <kbd>S</kbd>: Save snapshot image (PNG)
- <kbd>Q</kbd>: Quit

---

### 4. Create Highlights Introduction Video with Dynamic Empty Space Typography

Stitch selected soundbites into a teaser with transition effects (`fadewhite`, `fadeblack`), dynamic empty space placement, and modern typography styles. All clip timings and settings are stored in an external JSON configuration file (e.g. `clips.json` or `clips.example.json`):

```bash
# Run with default clips.json (Empty Space Detection + Floating Karaoke):
python build_intro_video.py clips.json

# Choose a typography style for all clips:
python build_intro_video.py clips.json --style floating_karaoke
python build_intro_video.py clips.json --style stacked_kinetic
python build_intro_video.py clips.json --style glassmorphic
python build_intro_video.py clips.json --style depth_layered

# Modern Arabic Typography (Cairo, Readex Pro, IBM Plex Sans Arabic):
python build_intro_video.py clips.json --font "Cairo" --style floating_karaoke

# Configure Quality Preset (high | medium | low):
python build_intro_video.py clips.json --quality high

# Override source video or project ID on the fly:
python build_intro_video.py clips.json --video input.mp4 --project-id my_custom_project
```

#### 🎨 4 Alternative Typography Styles for Empty Space:
1. **Floating Karaoke Highlight** (`floating_karaoke`): Boxless floating bold white text with soft drop shadow; active word dynamically highlights in gold/red.
2. **Stacked Kinetic Blocks** (`stacked_kinetic`): Stacks 2 to 3 words vertically, key word enlarged; fits tight lateral gaps beside a speaker's head.
3. **Glassmorphic Pill** (`glassmorphic`): Rounded pill with 40% black opacity and subtle background blur / border highlight.
4. **Depth-Layered Text** (`depth_layered`): Bold typography in the upper negative space with the speaker's head and shoulders masked *in front* of the text.

#### Example `clips.json` Config:
```json
{
  "project_id": "nasser_kandil",
  "source_video": "/path/to/video.mp4",
  "transcript_path": "projects/nasser_kandil/transcripts/DBMFJZDEhm4_transcript.json",
  "settings": {
    "theme_default": "floating_karaoke",
    "font": "Cairo",
    "quality": "high",
    "placement": "auto",
    "words_per_screen": 2,
    "pop_animation": true,
    "transition_duration": 0.6,
    "transitions": ["fadewhite", "fadeblack"]
  },
  "clips": [
    {
      "id": "clip1",
      "start": 354.7,
      "end": 366.5,
      "title": "📌 المحور الأول: اعتراف نتنياهو بالهزيمة",
      "theme": "floating_karaoke"
    },
    {
      "id": "clip2",
      "start": 1275.3,
      "end": 1290.5,
      "title": "⚠️ المحور الثاني: سيناريو اقتحام واحتلال الكنيست",
      "theme": "stacked_kinetic"
    },
    {
      "id": "clip3",
      "start": 1471.5,
      "end": 1488.0,
      "title": "💥 المحور الثالث: تفاصيل مخطط غزو صنعاء",
      "theme": "glassmorphic"
    }
  ]
}
```

---

## 🗂️ Unique Project Directory Structure

Every video or teaser project is isolated in its own unique directory under `projects/<project_id>/` to prevent file conflicts:

```text
projects/
└── <project_id>/
    ├── project_config.json        # Snapshot of settings used for the build
    ├── transcripts/               # Transcripts (JSON, CSV, SRT, TXT)
    │   ├── <name>_transcript.json
    │   ├── <name>_words.csv
    │   ├── <name>_words.srt
    │   └── <name>_subtitles.srt
    ├── clips/                     # Intermediate cut & styled segments
    │   ├── clip1_raw.mp4
    │   ├── clip1_dynamic.ass
    │   ├── clip1_styled.mp4
    │   └── ...
    └── output/                    # Final rendered intro video
        └── <project_id>_intro.mp4
```

### 2. Speed / Model Selection (CPU Guide)

You can choose the Whisper model according to your CPU speed and accuracy requirements:

| Model | Speed on CPU | Accuracy | Recommended For |
|---|---|---|---|
| `tiny` / `tiny.en` | ⚡ Fastest (~5-10x real-time) | Good | Quick drafts, testing, clean audio |
| `base` / `base.en` | ⚖️ Balanced (~2-4x real-time) | Better | **Default recommended** |
| `small` / `small.en` | 🐢 Slower (~1x real-time) | High | Difficult accents, background noise |

```powershell
# For maximum speed on CPU:
python transcribe.py "video.mp4" --model tiny

# If your video is in English only (faster and more accurate):
python transcribe.py "video.mp4" --model base.en

# Specify an output directory:
python transcribe.py "video.mp4" --output-dir "./output"
```

---

## 📊 Generated Output Files

For every video (e.g. `interview.mp4`), five output files are automatically generated:

1. **`interview_transcript.json`**  
   Contains the complete structured JSON with exact start/end seconds and confidence probability for every single word. **This is what you use for video editing scripts.**
   ```json
   {
     "words": [
       { "index": 0, "word": "Welcome", "start": 0.42, "end": 0.86, "duration": 0.44, "probability": 0.98 },
       { "index": 1, "word": "back",    "start": 0.88, "end": 1.12, "duration": 0.24, "probability": 0.99 }
     ]
   }
   ```
2. **`interview_words.csv`**  
   A spreadsheet table of all words with `start_seconds`, `end_seconds`, `duration_seconds`, and `probability`. Easy to review in Excel.
3. **`interview_words.srt`**  
   Word-by-word subtitle file (useful for fast-paced short-form video editors).
4. **`interview_subtitles.srt`**  
   Standard segment-level subtitles.
5. **`interview_transcript.txt`**  
   Clean plain text transcript.

---

## 🎬 How to Use the Timestamps for Video Editing

Use [video_tools.py](video_tools.py) to edit videos using your transcript:

### Example 1: Cut a clip between two timestamps
```python
from video_tools import cut_video_segment

# Cut video between 12.4 seconds and 25.8 seconds
cut_video_segment(
    input_video="video.mp4",
    output_video="clip.mp4",
    start_seconds=12.4,
    end_seconds=25.8,
    accurate=True
)
```

### Example 2: Search for a word and clip where it was spoken
```python
from video_tools import load_transcript, find_word_occurrences, cut_video_segment

data = load_transcript("video_transcript.json")

# Find when the speaker said "amazing"
matches = find_word_occurrences(data, "amazing")
for m in matches:
    print(f"Said '{m['word']}' at {m['start']}s -> {m['end']}s")

# Cut a 5-second buffer around the first occurrence
first = matches[0]
cut_video_segment("video.mp4", "highlight.mp4", first["start"] - 1.0, first["end"] + 4.0)
```

### Example 3: Create TikTok / Shorts Style Word-Highlight Subtitles
```python
from video_tools import load_transcript, create_highlighted_ass_subtitles, burn_subtitles

data = load_transcript("video_transcript.json")

# 1. Generate ASS subtitles with animated word highlighting
create_highlighted_ass_subtitles(data, "styled_subs.ass", words_per_subtitle=4)

# 2. Burn them directly into the video using FFmpeg
burn_subtitles("video.mp4", "styled_subs.ass", "video_with_captions.mp4")
```
