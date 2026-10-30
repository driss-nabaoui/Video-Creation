#!/usr/bin/env python3
"""
YouTube Transcript API Fetcher with Word-Level Timestamps.
Fetches transcripts instantly via youtube-transcript.io API and generates:
- Word-level JSON (with exact start/end seconds per word)
- Word-level CSV
- Word-level SRT
- Standard Subtitle SRT
- Plain Text Transcript
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_API_URL = "https://www.youtube-transcript.io/api/transcripts/v2"
DEFAULT_TOKEN = (
    "eyJhbGciOiJSUzI1NiIsImtpZCI6IjkwNGIwZjkxYmU2ZWIwODY3ZGFiYmFhYjU4N2RmYzJlMzNmMDBhMjQiLCJ0eXAiOiJKV1QifQ."
    "eyJwcm92aWRlcl9pZCI6ImFub255bW91cyIsImlzcyI6Imh0dHBzOi8vc2VjdXJldG9rZW4uZ29vZ2xlLmNvbS90cmFuc2NyaXB0LWZjNTgx"
    "IiwiYXVkIjoidHJhbnNjcmlwdC1mYzU4MSIsImF1dGhfdGltZSI6MTc5MzM3NjY4MCwidXNlcl9pZCI6IlV0N1NqYlJsYnZkSFdvUDhrQjh3"
    "eWpZRVcwSDIiLCJzdWIiOiJVdDdTamJSbGJ2ZEhXb1A4a0I4d3lqWUVXMEgyIiwiaWF0IjoxNzkzMzc2NjgwLCJleHAiOjE3OTMzODAyODAs"
    "ImZpcmViYXNlIjp7ImlkZW50aXRpZXMiOnt9LCJzaWduX2luX3Byb3ZpZGVyIjoiYW5vbnltb3VzIn19."
    "WU9KDGtBMATzpYqJP_6Aq7MoVmsU4jz1qCTLuT15PSIyrmmyHuJ3WTEdb7Q1jxgPleL2vyWcp6U1BFbKKXgaDRaBHwHhuxU-1ifB1urs8F2BW9"
    "QKc6--EPh02ZeqEVzJlA0DqwMozfbzfeiqCXvlCzNG77Ceil8jjmrj7RgwxNu5UCTHFBOrICL4cWxlD7dLrWiK_ghuYlrKqqaoC5N9pu_7hWMcl"
    "AbjJaLONjiKyyfl5H7Ea2wBDeLEb0GjAAMn1Cz6rqG1Ydav_b7ObhAVnmX3xlkJJsmWtx6mVNe2COs2l3oKUmn7DxCVAMEy8Z9MWKK-Vl1yQpAm3Zmb18ElFg"
)


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Extract 11-character YouTube video ID from various URL formats or raw ID."""
    clean_str = url_or_id.strip()
    if re.match(r"^[0-9A-Za-z_-]{11}$", clean_str):
        return clean_str

    patterns = [
        r"(?:v=|/v/|embed/|shorts/|youtu\.be/|[?&]id=)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean_str)
        if match:
            return match.group(1)
    return None


def format_timestamp(seconds: float, srt_format: bool = True) -> str:
    """Format seconds into SRT (HH:MM:SS,mmm) or VTT (HH:MM:SS.mmm)."""
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


def sanitize_filename(name: str) -> str:
    """Remove invalid filename characters for cross-platform file saving."""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def fetch_transcript_from_api(
    video_id: str,
    api_token: Optional[str] = None,
    api_url: str = DEFAULT_API_URL,
) -> Dict[str, Any]:
    """Call the youtube-transcript.io API to retrieve video transcript tracks."""
    token = api_token or os.environ.get("YOUTUBE_TRANSCRIPT_API_TOKEN") or DEFAULT_TOKEN

    headers = {
        "accept": "*/*",
        "accept-language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        "authorization": f"Bearer {token}",
        "content-type": "application/json",
        "priority": "u=1, i",
        "x-request-channel": "9527-c",
        "Referer": f"https://www.youtube-transcript.io/videos?id={video_id}",
    }

    payload = {"ids": [video_id], "source": "singleVideoUI"}

    print(f"[*] Querying youtube-transcript.io API for Video ID: {video_id}...", flush=True)
    res = requests.post(api_url, headers=headers, json=payload, timeout=20)

    if res.status_code != 200:
        raise RuntimeError(f"API request failed with HTTP {res.status_code}: {res.text[:300]}")

    data = res.json()
    if not data.get("success"):
        failed = data.get("failed", [])
        raise RuntimeError(f"API could not fetch transcript: {failed or data}")

    return data["success"][0]


def process_and_save_transcript(
    api_item: Dict[str, Any],
    output_dir: Optional[str] = None,
    custom_stem: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parses transcript segments, calculates word-level timestamps in seconds,
    and writes JSON, CSV, SRT, words SRT, and TXT files.
    """
    video_id = api_item.get("id", "unknown_video")
    title = api_item.get("title", f"youtube_{video_id}")
    stem = custom_stem or sanitize_filename(f"{title}_{video_id}")

    target_dir = Path(output_dir).resolve() if output_dir else Path.cwd() / "transcripts"
    target_dir.mkdir(parents=True, exist_ok=True)

    tracks = api_item.get("tracks", [])
    if not tracks or not tracks[0].get("transcript"):
        raise ValueError(f"No transcript track found for video {video_id}")

    # Use first available track
    active_track = tracks[0]
    raw_segments = active_track.get("transcript", [])
    language = active_track.get("language", "auto")

    print(f"[+] Found {len(raw_segments)} segments for '{title}' (Language: {language})", flush=True)

    all_words: List[Dict[str, Any]] = []
    all_segments: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []
    word_index = 0

    # Calculate word timestamps
    for seg_idx, item in enumerate(raw_segments):
        raw_text = item.get("text", "").strip()
        if not raw_text:
            continue

        try:
            start_sec = round(float(item.get("start", 0.0)), 3)
            dur_sec = round(float(item.get("dur", 0.0)), 3)
        except (ValueError, TypeError):
            continue

        end_sec = round(start_sec + dur_sec, 3)
        words_in_seg = [w.strip() for w in raw_text.split() if w.strip()]
        if not words_in_seg:
            continue

        full_text_parts.append(raw_text)

        seg_data: Dict[str, Any] = {
            "id": seg_idx + 1,
            "start": start_sec,
            "end": end_sec,
            "duration": dur_sec,
            "text": raw_text,
            "words": [],
        }

        # Distribute seconds across words proportional to character length
        weights = [max(1, len(w)) for w in words_in_seg]
        total_weight = sum(weights)
        current_word_time = start_sec

        for w_idx, (word, weight) in enumerate(zip(words_in_seg, weights)):
            # Proportional duration
            w_dur = round((weight / total_weight) * dur_sec, 3)
            w_start = round(current_word_time, 3)
            w_end = round(w_start + w_dur, 3)
            current_word_time += w_dur

            w_info = {
                "index": word_index,
                "word": word,
                "start": w_start,
                "end": w_end,
                "duration": w_dur,
                "probability": 1.0,
            }
            seg_data["words"].append(w_info)
            all_words.append(w_info)
            word_index += 1

        all_segments.append(seg_data)

    total_duration = all_segments[-1]["end"] if all_segments else 0.0

    result_data = {
        "source": "youtube_api",
        "video_id": video_id,
        "title": title,
        "duration_seconds": round(total_duration, 3),
        "language": language,
        "total_words": len(all_words),
        "total_segments": len(all_segments),
        "text": " ".join(full_text_parts),
        "words": all_words,
        "segments": all_segments,
    }

    # 1. Detailed JSON (used for video editing)
    json_path = target_dir / f"{stem}_transcript.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    # 2. Plain Text
    txt_path = target_dir / f"{stem}_transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result_data["text"] + "\n")

    # 3. CSV with word-level seconds
    csv_path = target_dir / f"{stem}_words.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "word", "start_seconds", "end_seconds", "duration_seconds", "probability"])
        for w in all_words:
            writer.writerow([w["index"], w["word"], w["start"], w["end"], w["duration"], w["probability"]])

    # 4. Standard Segment SRT
    srt_path = target_dir / f"{stem}_subtitles.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for seg in all_segments:
            f.write(f"{seg['id']}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")

    # 5. Word-by-word SRT
    words_srt_path = target_dir / f"{stem}_words.srt"
    with open(words_srt_path, "w", encoding="utf-8") as f:
        for w in all_words:
            f.write(f"{w['index'] + 1}\n")
            f.write(f"{format_timestamp(w['start'])} --> {format_timestamp(w['end'])}\n")
            f.write(f"{w['word']}\n\n")

    print("\n" + "=" * 60, flush=True)
    print(f"[✓] TRANSCRIPT READY IN SECONDS!", flush=True)
    print(f"[✓] Title:            {title}", flush=True)
    print(f"[✓] Video Duration:   {total_duration:.1f}s ({total_duration / 60:.1f} mins)", flush=True)
    print(f"[✓] Total Words:      {len(all_words)}", flush=True)
    print(f"[+] Output JSON:      {json_path}", flush=True)
    print(f"[+] Output CSV:       {csv_path}", flush=True)
    print(f"[+] Output Words SRT: {words_srt_path}", flush=True)
    print(f"[+] Output Seg SRT:   {srt_path}", flush=True)
    print(f"[+] Output Text:      {txt_path}", flush=True)
    print("=" * 60, flush=True)

    return result_data


def get_transcript_from_url(
    video_url_or_id: str,
    output_dir: Optional[str] = None,
    api_token: Optional[str] = None,
) -> Dict[str, Any]:
    """End-to-end helper: takes any YouTube URL or ID and generates all transcript files."""
    video_id = extract_video_id(video_url_or_id)
    if not video_id:
        raise ValueError(f"Could not extract a valid 11-character YouTube video ID from: {video_url_or_id}")

    api_item = fetch_transcript_from_api(video_id, api_token=api_token)
    return process_and_save_transcript(api_item, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Instant YouTube Transcript Fetcher with Word-Level Timestamps"
    )
    parser.add_argument("url", help="YouTube video URL or Video ID (e.g. https://www.youtube.com/watch?v=DBMFJZDEhm4)")
    parser.add_argument("-o", "--output-dir", default=None, help="Directory to save transcript files")
    parser.add_argument("--token", default=None, help="Custom Bearer token for youtube-transcript.io API")

    args = parser.parse_args()

    try:
        get_transcript_from_url(
            video_url_or_id=args.url,
            output_dir=args.output_dir,
            api_token=args.token,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
