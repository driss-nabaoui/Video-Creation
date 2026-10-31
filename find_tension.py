#!/usr/bin/env python3
"""
Parliament & Debate High-Tension Scene Detection CLI.
Detects arguments, shouting, interruptions, gavel banging, and heated debates
from long video files using a multi-signal scoring engine.

Architecture:
1. Acoustic Intensity & Vocal Strain (35% weight): RMS volume, Spectral Centroid, Spectral Flux.
2. Overlapping Speech & Interruptions (40% weight): Pyannote neural OSD or Harmonic Dispersion fallback.
3. Transcript Sentiment & Hostility (25% weight): Faster-Whisper WPS acceleration, multilingual conflict lexicon.
4. Scene Cutting: Merges outbursts within 8s, adds 4s lead-in / 3s lead-out context padding,
   and rapidly slices clips via FFmpeg.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tension_engine.audio_extractor import extract_audio_from_video, get_media_duration
from tension_engine.acoustic_analyzer import AcousticAnalyzer
from tension_engine.overlap_detector import OverlapDetector
from tension_engine.transcript_analyzer import TranscriptAnalyzer
from tension_engine.scene_processor import SceneProcessor
from tension_engine.video_cutter import VideoCutter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated High-Tension Parliamentary Debate Scene Detector & Video Slicer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input parliamentary debate video file (.mp4, .mkv, etc.)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output_tension",
        help="Directory to save exported clips, timestamps.json report, and timeline plot",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=60.0,
        help="Tension detection threshold (0-100). Higher = only extreme arguments; Lower = captures mild debates",
    )
    parser.add_argument(
        "--padding-before",
        type=float,
        default=4.0,
        help="Context lead-in padding in seconds before tension outburst begins",
    )
    parser.add_argument(
        "--padding-after",
        type=float,
        default=3.0,
        help="Context lead-out padding in seconds after tension subsides",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=8.0,
        help="Maximum gap in seconds between spikes to merge into a single continuous scene",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="Minimum scene duration in seconds (after padding)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Hardware accelerator to use for neural models",
    )
    parser.add_argument(
        "--transcript",
        type=str,
        default=None,
        help="Path to pre-computed transcript JSON file (bypasses Whisper model execution entirely)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="Faster-Whisper model size (e.g. tiny, base, small, medium, large-v3, turbo)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for Pyannote overlapped speech detection (optional; uses acoustic fallback if omitted)",
    )
    parser.add_argument(
        "--lossless",
        action="store_true",
        help="Use FFmpeg stream copy (-c copy) for instantaneous slicing without re-encoding",
    )
    parser.add_argument(
        "--no-clips",
        action="store_true",
        help="Skip video slicing, only generate timestamps.json and timeline plot",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable generating tension_timeline.png visualization",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Retain temporary 16kHz mono audio WAV file after completion",
    )

    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    secs = int(round(seconds))
    hrs = secs // 3600
    mins = (secs % 3600) // 60
    rem = secs % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{rem:02d}"
    return f"{mins:02d}:{rem:02d}"


def print_banner():
    banner = """
================================================================================
   PARLIAMENTARY DEBATE HIGH-TENSION SCENE DETECTOR
   Multi-Signal Engine: Acoustic (35%) + Overlap (40%) + Transcript (25%)
================================================================================
"""
    print(banner, flush=True)


def main():
    start_total_time = time.time()
    print_banner()
    args = parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        print(f"[!] Error: Input video file not found: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_audio_path = output_dir / "temp_audio_16k.wav"

    print(f"[*] Input Video: {input_path}")
    print(f"[*] Output Directory: {output_dir}")
    print(f"[*] Config: Threshold={args.threshold} | MergeGap={args.merge_gap}s | Padding=(-{args.padding_before}s, +{args.padding_after}s)")
    print(f"[*] Device: {args.device} | Whisper Model: {args.whisper_model}")

    # Step 1: Extract Audio & Get Total Duration
    print("\n" + "=" * 60)
    print("STEP 1: Extracting Audio & Media Metadata")
    print("=" * 60)
    total_duration = get_media_duration(input_path)
    print(f"[+] Total Video Duration: {total_duration:.2f}s ({format_seconds(total_duration)})")

    extract_audio_from_video(input_path, temp_audio_path)
    print(f"[+] 16kHz Mono Audio Extracted to: {temp_audio_path.name}")

    # Step 2: Signal 1 - Acoustic Intensity & Vocal Strain (35%)
    print("\n" + "=" * 60)
    print("STEP 2: Analyzing Acoustic Intensity & Vocal Strain (35% weight)")
    print("=" * 60)
    acoustic_analyzer = AcousticAnalyzer(target_sr=16000)
    acoustic_res = acoustic_analyzer.analyze_audio(temp_audio_path, total_duration=total_duration)
    acoustic_scores = acoustic_res["scores"]
    print(f"[+] Acoustic analysis complete: {len(acoustic_scores)} seconds processed.")

    # Step 3: Signal 2 - Overlapping Speech & Interruptions (40%)
    print("\n" + "=" * 60)
    print("STEP 3: Detecting Overlapping Speech & Cross-Talk (40% weight)")
    print("=" * 60)
    overlap_detector = OverlapDetector(hf_token=args.hf_token, device=args.device)
    overlap_res = overlap_detector.detect_overlap(temp_audio_path, total_duration=total_duration)
    overlap_scores = overlap_res["scores"]
    print(f"[+] Overlap detection complete: Engine='{overlap_res.get('engine')}', {len(overlap_scores)} seconds processed.")

    # Step 4: Signal 3 - Transcript Sentiment, Speech Rate & Hostility (25%)
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating Lexical Hostility & Speech Rate (25% weight)")
    print("=" * 60)
    transcript_file = None
    if args.transcript:
        candidate = Path(args.transcript).resolve()
        if candidate.is_file():
            transcript_file = candidate
            print(f"[+] Using specified transcript: {transcript_file}")
        else:
            print(f"[!] Warning: Specified transcript '{args.transcript}' not found. Falling back to Whisper.")

    # Auto-detection if not explicitly passed
    if not transcript_file:
        import re
        vid_id_match = re.search(r"\[([a-zA-Z0-9_-]{11})\]", input_path.stem)
        check_paths = []
        if vid_id_match:
            vid_id = vid_id_match.group(1)
            check_paths.append(ROOT_DIR / f"projects/{vid_id}/transcripts/{vid_id}_transcript.json")
            check_paths.append(ROOT_DIR / f"transcripts/{vid_id}_transcript.json")
        check_paths.append(input_path.with_suffix(".json"))
        check_paths.append(input_path.parent / f"{input_path.stem}_transcript.json")
        check_paths.append(ROOT_DIR / f"transcripts/{input_path.stem}_transcript.json")

        for cp in check_paths:
            if cp.is_file():
                transcript_file = cp
                print(f"[+] Auto-detected pre-existing transcript: {transcript_file}")
                print("[*] Skipping Whisper GPU/CPU inference — using ready transcript!")
                break

    transcript_analyzer = TranscriptAnalyzer(model_size=args.whisper_model, device=args.device)
    transcript_res = transcript_analyzer.analyze(
        audio_path=temp_audio_path if not transcript_file else None,
        total_duration=total_duration,
        precomputed_transcript_path=transcript_file,
    )
    transcript_scores = transcript_res["scores"]
    whisper_segments = transcript_res.get("segments", [])
    print(f"[+] Transcript analysis complete: Language detected='{transcript_res.get('language')}', {len(whisper_segments)} segments evaluated.")

    # Step 5: Multi-Signal Fusion, Smoothing, Merging & Scene Ranking
    print("\n" + "=" * 60)
    print("STEP 5: Multi-Signal Fusion, Smoothing & Scene Ranking")
    print("=" * 60)
    scene_processor = SceneProcessor(
        weight_acoustic=0.35,
        weight_overlap=0.40,
        weight_sentiment=0.25,
        default_threshold=args.threshold,
        merge_gap=args.merge_gap,
        padding_before=args.padding_before,
        padding_after=args.padding_after,
        min_scene_duration=args.min_duration,
    )
    fusion_res = scene_processor.process(
        acoustic_scores=acoustic_scores,
        overlap_scores=overlap_scores,
        sentiment_scores=transcript_scores,
        total_duration=total_duration,
        whisper_segments=whisper_segments,
        threshold_override=args.threshold,
    )

    scenes = fusion_res["scenes"]
    print(f"[+] Found {len(scenes)} high-tension scenes exceeding threshold ({args.threshold}).")

    # Step 6: Export Deliverables (Video Slicing, JSON Report, Timeline Plot)
    print("\n" + "=" * 60)
    print("STEP 6: Exporting Deliverables")
    print("=" * 60)
    cutter = VideoCutter(prefer_nvenc=True)
    clips_dir = output_dir / "output_clips"

    if not args.no_clips and len(scenes) > 0:
        scenes = cutter.cut_all_scenes(
            video_path=input_path,
            scenes=scenes,
            output_clips_dir=clips_dir,
            lossless=args.lossless,
        )
    elif args.no_clips:
        print("[*] Slicing skipped (--no-clips specified).")
    else:
        print("[!] No scenes met the tension threshold. Skipping clip export.")

    # Export timestamps.json
    metadata = {
        "input_video": str(input_path),
        "total_duration_seconds": round(total_duration, 2),
        "total_duration_formatted": format_seconds(total_duration),
        "threshold": args.threshold,
        "weights": {
            "acoustic_strain": 0.35,
            "overlapping_speech": 0.40,
            "transcript_hostility": 0.25,
        },
        "padding_seconds": {
            "lead_in_before": args.padding_before,
            "lead_out_after": args.padding_after,
        },
        "merge_gap_seconds": args.merge_gap,
        "language_detected": transcript_res.get("language", "unknown"),
        "overlap_engine_used": overlap_res.get("engine", "acoustic_heuristic"),
    }
    json_path = output_dir / "timestamps.json"
    cutter.export_timestamps_json(metadata=metadata, scenes=scenes, output_json_path=json_path)

    # Generate timeline plot
    if not args.no_plot:
        plot_path = output_dir / "tension_timeline.png"
        cutter.generate_timeline_plot(
            acoustic_scores=acoustic_scores,
            overlap_scores=overlap_scores,
            sentiment_scores=transcript_scores,
            smoothed_scores=fusion_res["smoothed_scores"],
            scenes=scenes,
            threshold=args.threshold,
            output_plot_path=plot_path,
        )

    # Cleanup temporary WAV if requested
    if not args.keep_audio and temp_audio_path.exists():
        try:
            temp_audio_path.unlink()
            print(f"[+] Cleaned up temporary audio: {temp_audio_path.name}")
        except Exception:
            pass

    # Step 7: Print Summary Table
    total_elapsed = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f" PIPELINE COMPLETE — DETECTED {len(scenes)} HIGH-TENSION SCENES ({total_elapsed:.1f}s total)")
    print("=" * 80)

    if len(scenes) > 0:
        print(f"{'Rank':<5} {'Clip Filename':<26} {'Time Span':<18} {'Dur':<6} {'Peak':<6} {'Acoustic':<9} {'Overlap':<9} {'Hostility':<9}")
        print("-" * 92)
        for sc in scenes:
            span_str = f"{format_seconds(sc['start'])} - {format_seconds(sc['end'])}"
            fname = sc.get("filename", f"scene_{sc['rank']:02d}.mp4")
            dur_str = f"{sc['duration']:.1f}s"
            peak_str = f"{sc['peak_tension_score']:.0f}"
            ac_str = f"{sc['signals']['peak_acoustic']:.0f}"
            ov_str = f"{sc['signals']['peak_overlap']:.0f}"
            st_str = f"{sc['signals']['peak_sentiment']:.0f}"
            print(f"#{sc['rank']:<4} {fname:<26} {span_str:<18} {dur_str:<6} {peak_str:<6} {ac_str:<9} {ov_str:<9} {st_str:<9}")
            if sc.get("transcript_snippet"):
                snip = sc["transcript_snippet"][:75] + ("..." if len(sc["transcript_snippet"]) > 75 else "")
                print(f"      Transcript: \"{snip}\"")
        print("-" * 92)
        print(f"[+] Output Clips: {clips_dir}")
        print(f"[+] Timestamps:   {json_path}")
        if not args.no_plot:
            print(f"[+] Visual Plot:  {output_dir / 'tension_timeline.png'}")
    else:
        print(f"[*] No scenes exceeded tension threshold {args.threshold}. Try lowering threshold (e.g. --threshold 45).")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
