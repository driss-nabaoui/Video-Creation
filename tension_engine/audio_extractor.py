"""
Audio Extraction Utility using FFmpeg.
Extracts 16kHz mono audio optimized for acoustic analysis, VAD, and transcription.
"""

import json
import subprocess
from pathlib import Path
from typing import Tuple


def get_media_duration(file_path: Path) -> float:
    """Extract total media duration in seconds via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(file_path),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(res.stdout)
        return float(data.get("format", {}).get("duration", 0.0))
    except Exception as e:
        # Fallback to cv2 or av if ffprobe fails
        try:
            import av
            with av.open(str(file_path)) as container:
                stream = container.streams.video[0] if container.streams.video else container.streams.audio[0]
                if stream.duration and stream.time_base:
                    return float(stream.duration * stream.time_base)
        except Exception:
            pass
        raise RuntimeError(f"Failed to determine media duration for '{file_path}': {e}")


def extract_audio_from_video(video_path: Path, output_wav_path: Path) -> Tuple[Path, float]:
    """
    Extracts high-fidelity 16kHz 16-bit PCM mono WAV from the video.
    Returns: (output_wav_path, duration_seconds)
    """
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    output_wav_path = Path(output_wav_path).resolve()
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    duration = get_media_duration(video_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    return output_wav_path, duration
