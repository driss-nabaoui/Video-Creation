# Video Transcription with Word-Level Timestamps (CPU-Optimized)

A Python tool designed to transcribe video/audio files with **exact word-level timestamps (in seconds)**, specially optimized for machines **without a dedicated GPU**.

---

## 🚀 Why this is fast on CPU

Standard Whisper (PyTorch) can be very slow on CPU. This script uses:
1. **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** powered by **CTranslate2**, which is up to **4x faster than vanilla Whisper** on CPU with a much smaller memory footprint.
2. **`int8` Quantization**: Runs weights in 8-bit precision, reducing CPU load while preserving transcription accuracy.
3. **Voice Activity Detection (VAD) Filtering**: Automatically skips silent segments, eliminating unnecessary processing time.
4. **Multi-threaded CPU execution**: Automatically uses all available CPU cores.

---

## 📁 Files Included

- [transcribe.py](file:///C:/Users/AdMin/Documents/Video-Creation/transcribe.py): The main CLI transcription tool (auto-detects local files or YouTube URLs).
- [youtube_api.py](file:///C:/Users/AdMin/Documents/Video-Creation/youtube_api.py): Instant YouTube transcript API fetcher (3 seconds for 50+ min videos).
- [dynamic_captions.py](file:///C:/Users/AdMin/Documents/Video-Creation/dynamic_captions.py): Creates dynamic word-by-word captions with hugging background boxes, pop animation, and custom positions (no full lines).
- [build_intro_video.py](file:///C:/Users/AdMin/Documents/Video-Creation/build_intro_video.py): Builds a teaser/introduction video with scene transitions and dynamic captions.
- [video_tools.py](file:///C:/Users/AdMin/Documents/Video-Creation/video_tools.py): Video editing helper functions (clipping, searching words, burning animated subtitles).

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

### 4. Create Highlights Introduction Video with Transitions & Hybrid Alignment

Stitch the best soundbites into a teaser with transition effects (`fadewhite`, `fadeblack`) and dynamic captions. It uses a **Hybrid Alignment Engine**:
- **Words Ground Truth**: Human-verified spelling from the YouTube API transcript.
- **Timing Ground Truth**: Exact millisecond acoustic waveforms from audio clip analysis.
- **Arabic Ligatures**: Native connected cursive text (`Segoe UI`, `Spacing=0`).
- **No Overlaps & Zero Lag**: Word chunks appear synchronously with speech.

```powershell
.venv\Scripts\python build_intro_video.py
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

Use [video_tools.py](file:///C:/Users/AdMin/Documents/Video-Creation/video_tools.py) to edit videos using your transcript:

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
