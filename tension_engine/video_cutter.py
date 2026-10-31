"""
Video Cutter & Report Generator.
Rapidly slices detected high-tension scenes using FFmpeg (lossless stream-copy or hardware NVENC),
exports structured timestamps.json, and generates timeline visualizations.
"""

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class VideoCutter:
    """
    Rapidly slices video clips using FFmpeg and produces metadata reports and visual timeline plots.
    """

    def __init__(self, prefer_nvenc: bool = True):
        self.has_nvenc = prefer_nvenc and self._check_nvenc_support()

    @staticmethod
    def _check_nvenc_support() -> bool:
        """Check if FFmpeg supports h264_nvenc encoding."""
        try:
            res = subprocess.run(
                ["ffmpeg", "-encoders"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return "h264_nvenc" in res.stdout
        except Exception:
            return False

    def cut_clip(
        self,
        video_path: Path,
        start_time: float,
        duration: float,
        output_path: Path,
        lossless: bool = False,
    ) -> bool:
        """
        Extracts a single clip from video_path.
        Uses input-seeking (-ss before -i) for instantaneous seeking without decoding from frame 0.
        """
        video_path = Path(video_path).resolve()
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if duration <= 0.1:
            print(f"[!] Invalid clip duration: {duration}s for {output_path.name}")
            return False

        # Base seeking options
        start_str = f"{max(0.0, start_time):.3f}"
        dur_str = f"{max(0.1, duration):.3f}"

        if lossless:
            # Stream copy: ultra-fast, zero quality loss (cuts at nearest keyframes)
            cmd = [
                "ffmpeg", "-y",
                "-ss", start_str,
                "-i", str(video_path),
                "-t", dur_str,
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(output_path),
            ]
            try:
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                    return True
            except Exception as e:
                print(f"[!] Lossless stream copy failed ({e}), falling back to re-encoding...")

        # Re-encode mode: frame-accurate cuts
        # Turing / Ampere / Ada: use h264_nvenc if available, else libx264
        enc_v = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"] if self.has_nvenc else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        enc_a = ["-c:a", "aac", "-b:a", "192k"]

        cmd = [
            "ffmpeg", "-y",
            "-ss", start_str,
            "-i", str(video_path),
            "-t", dur_str,
            *enc_v,
            *enc_a,
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0 and self.has_nvenc:
                # If NVENC failed, retry with CPU libx264
                print(f"[!] NVENC encoding failed, retrying with libx264...")
                cmd_fallback = [
                    "ffmpeg", "-y",
                    "-ss", start_str,
                    "-i", str(video_path),
                    "-t", dur_str,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "192k",
                    "-avoid_negative_ts", "make_zero",
                    str(output_path),
                ]
                res = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                return True
            else:
                print(f"[!] FFmpeg cut failed for {output_path.name}: {res.stderr[-300:]}")
                return False
        except Exception as e:
            print(f"[!] FFmpeg error cutting {output_path.name}: {e}")
            return False

    def cut_all_scenes(
        self,
        video_path: Path,
        scenes: List[Dict[str, Any]],
        output_clips_dir: Path,
        lossless: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Slices all detected scenes into output_clips/scene_01_score88.mp4, etc.
        Updates each scene dictionary with filename and clip_path.
        """
        output_clips_dir = Path(output_clips_dir).resolve()
        output_clips_dir.mkdir(parents=True, exist_ok=True)

        mode_name = "lossless stream copy" if lossless else ("GPU NVENC (h264_nvenc)" if self.has_nvenc else "CPU (libx264)")
        print(f"[*] Slicing {len(scenes)} tension scenes using {mode_name}...", flush=True)

        for sc in scenes:
            rank = sc.get("rank", 1)
            score = int(round(sc.get("peak_tension_score", 0.0)))
            filename = f"scene_{rank:02d}_score{score}.mp4"
            output_file = output_clips_dir / filename

            start = sc["start"]
            duration = sc["duration"]

            print(f"    -> Exporting Scene #{rank:02d} [{start:.1f}s - {sc['end']:.1f}s | {duration:.1f}s | Peak: {score}]: {filename}...", flush=True)
            success = self.cut_clip(
                video_path=video_path,
                start_time=start,
                duration=duration,
                output_path=output_file,
                lossless=lossless,
            )
            sc["filename"] = filename
            sc["clip_path"] = str(output_file) if success else None
            sc["cut_status"] = "success" if success else "failed"

        print(f"[+] All scenes exported to {output_clips_dir}", flush=True)
        return scenes

    @staticmethod
    def export_timestamps_json(
        metadata: Dict[str, Any],
        scenes: List[Dict[str, Any]],
        output_json_path: Path,
    ) -> Path:
        """
        Saves structured report to timestamps.json.
        """
        output_json_path = Path(output_json_path).resolve()
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "metadata": {
                **metadata,
                "exported_at": datetime.now().isoformat(),
                "total_scenes_detected": len(scenes),
            },
            "scenes": scenes,
        }

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"[+] Timestamps report exported to: {output_json_path}", flush=True)
        return output_json_path

    @staticmethod
    def generate_timeline_plot(
        acoustic_scores: List[float],
        overlap_scores: List[float],
        sentiment_scores: List[float],
        smoothed_scores: List[float],
        scenes: List[Dict[str, Any]],
        threshold: float,
        output_plot_path: Path,
    ) -> Optional[Path]:
        """
        Generates a 4-panel multi-signal timeline plot showing:
        1. Acoustic Strain & Energy (35%)
        2. Overlapping Speech & Interruption (40%)
        3. Transcript Sentiment & Hostility (25%)
        4. Unified Smoothed Tension Score (0-100) with highlighted scene spans.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            n_sec = max(len(acoustic_scores), len(overlap_scores), len(sentiment_scores), len(smoothed_scores))
            if n_sec == 0:
                return None

            time_axis = np.arange(n_sec) / 60.0  # Time in minutes

            def pad_sig(lst, n):
                arr = np.array(lst, dtype=np.float32)
                if len(arr) < n:
                    arr = np.pad(arr, (0, n - len(arr)), mode="edge")
                return arr[:n]

            ac = pad_sig(acoustic_scores, n_sec)
            ov = pad_sig(overlap_scores, n_sec)
            st = pad_sig(sentiment_scores, n_sec)
            sm = pad_sig(smoothed_scores, n_sec)

            fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
            plt.subplots_adjust(hspace=0.25)

            # Panel 1: Acoustic
            axes[0].plot(time_axis, ac, color="#ff9800", linewidth=1.2, label="Acoustic Strain (35% weight)")
            axes[0].fill_between(time_axis, ac, color="#ff9800", alpha=0.2)
            axes[0].set_ylabel("Acoustic (0-100)", fontsize=10, fontweight="bold")
            axes[0].set_ylim(-2, 105)
            axes[0].grid(True, linestyle="--", alpha=0.5)
            axes[0].legend(loc="upper right", framealpha=0.8)
            axes[0].set_title("Multi-Signal Tension Detection Timeline", fontsize=14, fontweight="bold")

            # Panel 2: Overlap
            axes[1].plot(time_axis, ov, color="#00bcd4", linewidth=1.2, label="Overlap & Cross-Talk (40% weight)")
            axes[1].fill_between(time_axis, ov, color="#00bcd4", alpha=0.2)
            axes[1].set_ylabel("Overlap (0-100)", fontsize=10, fontweight="bold")
            axes[1].set_ylim(-2, 105)
            axes[1].grid(True, linestyle="--", alpha=0.5)
            axes[1].legend(loc="upper right", framealpha=0.8)

            # Panel 3: Hostility
            axes[2].plot(time_axis, st, color="#9c27b0", linewidth=1.2, label="Lexical Hostility & Rate (25% weight)")
            axes[2].fill_between(time_axis, st, color="#9c27b0", alpha=0.2)
            axes[2].set_ylabel("Hostility (0-100)", fontsize=10, fontweight="bold")
            axes[2].set_ylim(-2, 105)
            axes[2].grid(True, linestyle="--", alpha=0.5)
            axes[2].legend(loc="upper right", framealpha=0.8)

            # Panel 4: Unified Smoothed Tension Score & Scenes
            axes[3].plot(time_axis, sm, color="#d32f2f", linewidth=1.8, label="Fused Smoothed Tension Score")
            axes[3].fill_between(time_axis, sm, color="#d32f2f", alpha=0.25)
            axes[3].axhline(y=threshold, color="#b71c1c", linestyle="--", linewidth=1.4, label=f"Threshold ({threshold:.0f})")

            # Highlight detected scenes
            for sc in scenes:
                st_m = sc["start"] / 60.0
                en_m = sc["end"] / 60.0
                rank = sc.get("rank", 1)
                peak = int(round(sc.get("peak_tension_score", 0.0)))
                axes[3].axvspan(st_m, en_m, color="#d32f2f", alpha=0.35)
                # Label above scene
                mid_m = (st_m + en_m) / 2.0
                axes[3].text(
                    mid_m, min(100, peak + 3),
                    f"#{rank}\n({peak})",
                    color="#b71c1c",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            axes[3].set_ylabel("Unified Tension", fontsize=10, fontweight="bold")
            axes[3].set_ylim(-2, 115)
            axes[3].set_xlabel("Timeline (minutes)", fontsize=11, fontweight="bold")
            axes[3].grid(True, linestyle="--", alpha=0.5)
            axes[3].legend(loc="upper right", framealpha=0.8)

            output_plot_path = Path(output_plot_path).resolve()
            output_plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_plot_path), dpi=150, bbox_inches="tight")
            plt.close(fig)

            print(f"[+] Tension timeline plot exported to: {output_plot_path}", flush=True)
            return output_plot_path
        except Exception as e:
            print(f"[!] Warning: Failed to generate timeline plot ({e})")
            return None
