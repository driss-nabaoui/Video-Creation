#!/usr/bin/env python3
"""
Video Transcription Tool with Word-Level Timestamps (CPU Optimized).
Features:
- Progressive real-time saving (files update on disk live as segments are transcribed).
- Interruption/Cancellation resilience (saves everything up to the cancellation point).
- Time range clipping (--start, --duration, --end) for working on specific video portions.
- Fast FFmpeg audio pre-extraction.
- Live progress bar with percentage, speed, and ETA.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Force UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from faster_whisper import WhisperModel


def format_timestamp(seconds: float, srt_format: bool = True) -> str:
    """Convert seconds to SRT (HH:MM:SS,mmm) or VTT (HH:MM:SS.mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))

    if millis >= 1000:
        millis -= 1000
        secs += 1
        if secs >= 60:
            secs -= 60
            mins += 1
            if mins >= 60:
                mins -= 60
                hrs += 1

    delimiter = "," if srt_format else "."
    return f"{hrs:02d}:{mins:02d}:{secs:02d}{delimiter}{millis:03d}"


def parse_time_str(time_str: Optional[str]) -> Optional[float]:
    """Parse time string like '120', '02:30', or '01:15:30' into seconds."""
    if not time_str:
        return None
    try:
        return float(time_str)
    except ValueError:
        parts = [float(p) for p in time_str.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        raise ValueError(f"Invalid time format: {time_str}")


def extract_audio_fast(
    input_path: Path,
    start_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
) -> Path:
    """Pre-extract 16kHz mono WAV using FFmpeg for ultra-fast, robust audio loading."""
    temp_wav = Path(tempfile.gettempdir()) / f"whisper_audio_{os.getpid()}_{int(time.time())}.wav"
    cmd = ["ffmpeg", "-y"]
    if start_sec is not None and start_sec > 0:
        cmd.extend(["-ss", f"{start_sec:.2f}"])
    cmd.extend(["-i", str(input_path)])
    if duration_sec is not None and duration_sec > 0:
        cmd.extend(["-t", f"{duration_sec:.2f}"])
    cmd.extend(["-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(temp_wav)])

    print(f"[*] Pre-extracting 16kHz audio with FFmpeg...", flush=True)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return temp_wav


def transcribe(
    media_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    output_dir: Optional[str] = None,
    vad_filter: bool = True,
    threads: Optional[int] = None,
    start_time: Optional[str] = None,
    duration: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribes media file with real-time progressive disk streaming and word timestamps."""
    path = Path(media_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Media file not found: {path}")

    target_dir = Path(output_dir).resolve() if output_dir else path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    start_sec = parse_time_str(start_time)
    duration_sec = parse_time_str(duration)

    # Output file paths
    json_path = target_dir / f"{stem}_transcript.json"
    txt_path = target_dir / f"{stem}_transcript.txt"
    csv_path = target_dir / f"{stem}_words.csv"
    srt_path = target_dir / f"{stem}_subtitles.srt"
    words_srt_path = target_dir / f"{stem}_words.srt"

    # Pre-extract audio using FFmpeg for optimal speed
    temp_audio = extract_audio_fast(path, start_sec, duration_sec)

    num_threads = threads or os.cpu_count() or 4
    print(f"[*] Initializing faster-whisper '{model_size}' model on CPU ({num_threads} threads, int8)...", flush=True)
    model = WhisperModel(
        model_size_or_path=model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=num_threads,
        num_workers=1,
    )

    print(f"[*] Starting transcription (progressive live saving enabled)...", flush=True)
    print(f"[+] Files will be updated live in: {target_dir}", flush=True)

    start_wall_clock = time.time()
    segments_generator, info = model.transcribe(
        str(temp_audio),
        language=language,
        word_timestamps=True,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    total_duration = info.duration or 1.0
    time_offset = start_sec or 0.0

    print(f"[*] Audio length: {total_duration:.1f}s ({total_duration / 60:.1f} mins) | Detected language: '{info.language}' ({info.language_probability * 100:.0f}%)", flush=True)

    all_words: List[Dict[str, Any]] = []
    all_segments: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []
    word_index = 0
    seg_index = 0

    # Initialize live output files
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["index", "word", "start_seconds", "end_seconds", "duration_seconds", "probability"])

    # Clear/create srt, words_srt, and txt files
    open(srt_path, "w", encoding="utf-8").close()
    open(words_srt_path, "w", encoding="utf-8").close()
    open(txt_path, "w", encoding="utf-8").close()

    def save_checkpoint(is_final: bool = False):
        """Save JSON checkpoint immediately to disk."""
        elapsed = time.time() - start_wall_clock
        speed = (all_segments[-1]["end"] / elapsed) if all_segments and elapsed > 0 else 0.0
        data = {
            "file": str(path),
            "filename": path.name,
            "duration_seconds": round(total_duration, 3),
            "time_offset_seconds": time_offset,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "is_complete": is_final,
            "total_words": len(all_words),
            "total_segments": len(all_segments),
            "processing_time_seconds": round(elapsed, 2),
            "speed_factor": round(speed, 2),
            "text": " ".join(full_text_parts),
            "words": all_words,
            "segments": all_segments,
        }
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(data, f_json, indent=2, ensure_ascii=False)

    try:
        for segment in segments_generator:
            seg_index += 1
            # Adjust timestamp with offset if partial clip was transcribed
            seg_start = round(segment.start + time_offset, 3)
            seg_end = round(segment.end + time_offset, 3)
            seg_text = segment.text.strip()
            full_text_parts.append(seg_text)

            seg_data: Dict[str, Any] = {
                "id": seg_index,
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
                "words": [],
            }

            # 1. Append to segment SRT live
            with open(srt_path, "a", encoding="utf-8") as f_srt:
                f_srt.write(f"{seg_index}\n")
                f_srt.write(f"{format_timestamp(seg_start)} --> {format_timestamp(seg_end)}\n")
                f_srt.write(f"{seg_text}\n\n")

            # 2. Append to plain text live
            with open(txt_path, "a", encoding="utf-8") as f_txt:
                f_txt.write(seg_text + " ")

            # 3. Process words and append to CSV + words.srt live
            if segment.words:
                with open(csv_path, "a", newline="", encoding="utf-8") as f_csv, \
                     open(words_srt_path, "a", encoding="utf-8") as f_wsrt:
                    csv_writer = csv.writer(f_csv)

                    for w in segment.words:
                        clean_w = w.word.strip()
                        if not clean_w:
                            continue
                        w_start = round(w.start + time_offset, 3)
                        w_end = round(w.end + time_offset, 3)
                        w_dur = round(w_end - w_start, 3)
                        w_prob = round(w.probability, 3)

                        w_info = {
                            "index": word_index,
                            "word": clean_w,
                            "start": w_start,
                            "end": w_end,
                            "duration": w_dur,
                            "probability": w_prob,
                        }
                        seg_data["words"].append(w_info)
                        all_words.append(w_info)

                        # Write to CSV
                        csv_writer.writerow([word_index, clean_w, w_start, w_end, w_dur, w_prob])

                        # Write to words.srt
                        f_wsrt.write(f"{word_index + 1}\n")
                        f_wsrt.write(f"{format_timestamp(w_start)} --> {format_timestamp(w_end)}\n")
                        f_wsrt.write(f"{clean_w}\n\n")

                        word_index += 1

            all_segments.append(seg_data)

            # Compute progress
            current_sec = segment.end
            percent = min(100.0, (current_sec / total_duration) * 100)
            elapsed = time.time() - start_wall_clock
            speed = (current_sec / elapsed) if elapsed > 0 else 1.0
            remaining_sec = max(0.0, (total_duration - current_sec) / speed) if speed > 0 else 0.0

            # Print live progress
            print(
                f"[{percent:5.1f}% | {format_timestamp(seg_start, False)} -> {format_timestamp(seg_end, False)}] "
                f"(Speed: {speed:.1f}x, ETA: {int(remaining_sec // 60)}m {int(remaining_sec % 60):02d}s, Words: {word_index}) : {seg_text}",
                flush=True
            )

            # Periodically update JSON checkpoint every 5 segments
            if seg_index % 5 == 0:
                save_checkpoint(is_final=False)

    except (KeyboardInterrupt, BaseException) as e:
        print(f"\n[!] Process stopped/interrupted. Saving all progress up to {all_segments[-1]['end'] if all_segments else 0}s...", flush=True)
        save_checkpoint(is_final=False)
        raise e
    finally:
        # Clean up temporary audio file
        if temp_audio.exists():
            try:
                temp_audio.unlink()
            except Exception:
                pass

    # Save final JSON
    save_checkpoint(is_final=True)
    total_elapsed = time.time() - start_wall_clock
    final_speed = (total_duration / total_elapsed) if total_elapsed > 0 else 0.0

    print("\n" + "=" * 60, flush=True)
    print(f"[✓] TRANSCRIPTION COMPLETE in {total_elapsed:.1f}s ({final_speed:.1f}x real-time)", flush=True)
    print(f"[✓] Total words transcribed: {len(all_words)}", flush=True)
    print(f"[+] Output JSON:      {json_path}", flush=True)
    print(f"[+] Output CSV:       {csv_path}", flush=True)
    print(f"[+] Output Words SRT: {words_srt_path}", flush=True)
    print(f"[+] Output Seg SRT:   {srt_path}", flush=True)
    print(f"[+] Output Text:      {txt_path}", flush=True)
    print("=" * 60, flush=True)

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="CPU-Optimized Video Transcription with Real-time Progressive File Saving"
    )
    parser.add_argument("media", help="Path to video/audio file OR YouTube URL/ID")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "large-v3"],
        help="Whisper model size (default: base). Use 'tiny' for maximum speed on long videos.",
    )
    parser.add_argument("--language", default=None, help="Language code (e.g. 'ar', 'en')")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads to use")
    parser.add_argument("--start", default=None, help="Start time offset, e.g. '00:05:00' or '300'")
    parser.add_argument("--duration", default=None, help="Duration to transcribe in seconds or 'MM:SS'")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD silence filter")
    parser.add_argument("--api-token", default=None, help="Custom Bearer token for youtube-transcript.io API")

    args = parser.parse_args()

    # Check if input is a YouTube URL or YouTube video ID
    from youtube_api import extract_video_id, get_transcript_from_url
    if args.media.startswith("http://") or args.media.startswith("https://") or (len(args.media) == 11 and not Path(args.media).exists()):
        vid = extract_video_id(args.media)
        if vid:
            print(f"[*] Detected YouTube input ({vid}). Fetching via YouTube Transcript API...", flush=True)
            try:
                get_transcript_from_url(
                    video_url_or_id=args.media,
                    output_dir=args.output_dir,
                    api_token=args.api_token,
                )
                return
            except Exception as e:
                print(f"[!] API fetch failed: {e}. Falling back to local transcription if file exists...", flush=True)

    try:
        transcribe(
            media_path=args.media,
            model_size=args.model,
            language=args.language,
            output_dir=args.output_dir,
            vad_filter=not args.no_vad,
            threads=args.threads,
            start_time=args.start,
            duration=args.duration,
        )
    except KeyboardInterrupt:
        print("\nExited by user.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
