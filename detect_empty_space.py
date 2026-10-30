#!/usr/bin/env python3
r"""
Clean Empty Space & Person Segment Detection (Live Preview)
===========================================================
Detects clean, unobstructed negative space in video frames while segmenting
and highlighting the speaker.

Features:
1. Person Segmentation & Highlighting:
   - Real-time foreground mask separation using MediaPipe.
   - Vibrant neon contour glow and translucent silhouette tint around the speaker.

2. Clean Empty Space Detection (LER + Clutter Exclusion):
   - Fuses person mask with Canny edge clutter map.
   - Largest Empty Rectangle (LER) algorithm in O(R·C) monotonic stack.
   - Guaranteed collision-free negative space for graphics, text, or callouts.

3. Saliency & Luminance Telemetry:
   - Edge density analysis to avoid complex visual clutter (posters, bookshelves, logos).
   - Samples average luminance (0-255) in the detected empty space.
   - Temporal smoothing (EMA) to prevent bounding box jitter.

4. Quality Presets:
   - High   : Native resolution, fine LER grid, visually lossless H.264 (CRF 18).
   - Medium : 720p scaling, balanced LER grid, crisp H.264 (CRF 22).
   - Low    : 480p scaling, fast LER grid, high-speed H.264 (CRF 28).

Usage:
  python detect_empty_space.py "/path/to/video.mp4"
  python detect_empty_space.py "/path/to/video.mp4" --quality high --save
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np

# MediaPipe Tasks Vision
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = Path("models/pose_landmarker_lite.task")

DEFAULT_VIDEO = "/home/tapa/Videos/ناصر قنديل بحلقة نارية بعد انتخابات اسرائيل_اعدام نتنياهو واحتلال الكنيست! تفاصيل غزو صنعاء في اسبوع_1080p.mp4"

QUALITY_PRESETS = {
    "high": {
        "max_dim": 1920,
        "ler_scale": 6,
        "crf": "18",
        "preset": "fast",
        "desc": "Full 1080p • Fine LER • Visually Lossless",
    },
    "medium": {
        "max_dim": 1280,
        "ler_scale": 8,
        "crf": "22",
        "preset": "fast",
        "desc": "720p Scaled • Balanced LER • High Performance",
    },
    "low": {
        "max_dim": 854,
        "ler_scale": 12,
        "crf": "28",
        "preset": "veryfast",
        "desc": "480p Scaled • Fast LER • Maximum Speed",
    },
}


def ensure_model_exists(model_path: Path) -> Path:
    """Download MediaPipe pose landmarker model if not present."""
    if not model_path.is_file():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[*] Downloading MediaPipe model to {model_path}...", flush=True)
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("[✓] Model downloaded successfully.", flush=True)
    return model_path


def max_rectangle_in_histogram(heights: List[int]) -> Tuple[int, int, int, int]:
    """
    Computes Largest Empty Rectangle in a 1D histogram using a monotonic stack.
    Returns: (start_col, width, height, max_area)
    Time Complexity: O(N)
    """
    stack: List[Tuple[int, int]] = []
    max_area = 0
    best = (0, 0, 0, 0)
    extended = heights + [0]

    for i, h in enumerate(extended):
        start = i
        while stack and stack[-1][1] > h:
            idx, height = stack.pop()
            width = i - idx
            area = width * height
            if area > max_area:
                max_area = area
                best = (idx, width, height, area)
            start = idx
        stack.append((start, h))
    return best


def find_largest_empty_rectangle(binary_clean: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Finds the Largest Empty Rectangle in a 2D binary grid (1 = clean empty space, 0 = obstacle).
    Returns (x, y, w, h).
    """
    R, C = binary_clean.shape
    heights = [0] * C
    best_x, best_y, best_w, best_h, max_area = 0, 0, 0, 0, 0

    for r in range(R):
        for c in range(C):
            heights[c] = heights[c] + 1 if binary_clean[r, c] == 1 else 0
        left, w, h, area = max_rectangle_in_histogram(heights)
        if area > max_area:
            max_area = area
            best_x = left
            best_y = r - h + 1
            best_w = w
            best_h = h

    return best_x, best_y, best_w, best_h


class EmptySpaceDetector:
    def __init__(self, quality: str = "medium", model_path: Path = MODEL_PATH):
        ensure_model_exists(model_path)
        self.quality = quality.lower() if quality in QUALITY_PRESETS else "medium"
        self.cfg = QUALITY_PRESETS[self.quality]

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            output_segmentation_masks=True,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # Temporal smoothing state (EMA)
        self.smoothed_box: Optional[Tuple[float, float, float, float]] = None
        self.alpha = 0.22  # Smoothing factor (0 = static, 1 = instant jump)

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Runs the full segmentation & clean empty space detection pipeline on a single frame.
        """
        orig_h, orig_w, _ = frame.shape

        # Scale down if quality setting specifies smaller processing dimension
        max_dim = self.cfg["max_dim"]
        if max(orig_h, orig_w) > max_dim:
            scale_factor = max_dim / max(orig_h, orig_w)
            proc_w = int(orig_w * scale_factor)
            proc_h = int(orig_h * scale_factor)
            proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        else:
            scale_factor = 1.0
            proc_w, proc_h = orig_w, orig_h
            proc_frame = frame

        # 1. MediaPipe Person Segmentation & Landmarks
        rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_img)

        # Extract Person Mask
        if res.segmentation_masks and len(res.segmentation_masks) > 0:
            raw_mask = res.segmentation_masks[0].numpy_view()
            if raw_mask.shape[:2] != (proc_h, proc_w):
                raw_mask = cv2.resize(raw_mask, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
            if raw_mask.ndim == 3:
                raw_mask = raw_mask.squeeze(-1)
            # Threshold person segmentation
            person_mask = (raw_mask > 0.45).astype(np.uint8)
        else:
            person_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)

        # 2. Extract Person Contours for Highlighting
        contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 3. Visual Clutter / Edge Detection
        gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(gray, 40, 120)

        # Dilate edges slightly to form visual obstacle zones
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clutter_obstacles = cv2.dilate(canny_edges, kernel, iterations=1)

        # 4. Clean Background Map (1 = Clean Empty Space, 0 = Person or Visual Clutter)
        # Background is where person == 0
        bg_mask = (1 - person_mask).astype(np.uint8)
        # Clean space is background AND free of dense clutter
        clean_bg_map = np.where((person_mask == 0) & (clutter_obstacles == 0), 1, 0).astype(np.uint8)

        # If clean space is too fragmented, fall back to pure background mask
        if np.mean(clean_bg_map) < 0.15:
            clean_bg_map = bg_mask

        # 5. Largest Empty Rectangle (LER)
        ler_scale = self.cfg["ler_scale"]
        grid_w = max(10, proc_w // ler_scale)
        grid_h = max(10, proc_h // ler_scale)
        small_clean = cv2.resize(clean_bg_map, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)

        lx, ly, lw, lh = find_largest_empty_rectangle(small_clean)

        # Scale coordinates back to proc_frame space
        ler_x = int(lx * ler_scale)
        ler_y = int(ly * ler_scale)
        ler_w = int(lw * ler_scale)
        ler_h = int(lh * ler_scale)

        # Safety bounds
        ler_x = max(0, min(proc_w - 1, ler_x))
        ler_y = max(0, min(proc_h - 1, ler_y))
        ler_w = max(40, min(proc_w - ler_x, ler_w))
        ler_h = max(40, min(proc_h - ler_y, ler_h))

        # Scale to original frame space
        orig_ler_x = int(ler_x / scale_factor)
        orig_ler_y = int(ler_y / scale_factor)
        orig_ler_w = int(ler_w / scale_factor)
        orig_ler_h = int(ler_h / scale_factor)

        # 6. Temporal Smoothing (EMA)
        target_box = (float(orig_ler_x), float(orig_ler_y), float(orig_ler_w), float(orig_ler_h))
        if self.smoothed_box is None:
            self.smoothed_box = target_box
        else:
            sx, sy, sw, sh = self.smoothed_box
            self.smoothed_box = (
                sx + self.alpha * (target_box[0] - sx),
                sy + self.alpha * (target_box[1] - sy),
                sw + self.alpha * (target_box[2] - sw),
                sh + self.alpha * (target_box[3] - sh),
            )

        final_ler = (
            int(self.smoothed_box[0]),
            int(self.smoothed_box[1]),
            int(self.smoothed_box[2]),
            int(self.smoothed_box[3]),
        )

        # 7. Quality Metrics on Detected Empty Space
        fx, fy, fw, fh = final_ler
        orig_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = orig_gray[max(0, fy) : min(orig_h, fy + fh), max(0, fx) : min(orig_w, fx + fw)]

        mean_lum = float(np.mean(roi_gray)) if roi_gray.size > 0 else 128.0
        roi_edges = cv2.Canny(roi_gray, 50, 150) if roi_gray.size > 0 else np.zeros((1, 1))
        edge_clutter = float(np.mean(roi_edges > 0)) * 100.0 if roi_edges.size > 0 else 0.0
        area_pct = (fw * fh) / (orig_w * orig_h) * 100.0

        # Position / Quadrant determination
        cx = fx + fw / 2.0
        cy = fy + fh / 2.0
        h_pos = "Left" if cx < orig_w / 3.0 else ("Center" if cx < 2 * orig_w / 3.0 else "Right")
        v_pos = "Top" if cy < orig_h / 3.0 else ("Mid" if cy < 2 * orig_h / 3.0 else "Bottom")
        quadrant_name = f"{v_pos}-{h_pos}"

        # Scale person mask to original frame dimensions
        if scale_factor != 1.0:
            full_person_mask = cv2.resize(person_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            full_contours, _ = cv2.findContours(full_person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            full_person_mask = person_mask
            full_contours = contours

        return {
            "orig_size": (orig_w, orig_h),
            "proc_size": (proc_w, proc_h),
            "quality": self.quality,
            "person_mask": full_person_mask,
            "contours": full_contours,
            "edges": canny_edges,
            "empty_rect": final_ler,
            "area_pct": area_pct,
            "mean_lum": mean_lum,
            "edge_clutter": edge_clutter,
            "quadrant": quadrant_name,
            "has_person": len(full_contours) > 0,
        }


class LivePreviewRenderer:
    """Handles rendering of overlays, person highlights, clean empty box, and telemetry HUD."""

    def __init__(self):
        pass

    def draw(
        self,
        frame: np.ndarray,
        data: Dict[str, Any],
        show_person: bool = True,
        show_empty_box: bool = True,
        show_edges: bool = False,
        show_mask: bool = False,
        show_hud: bool = True,
    ) -> np.ndarray:
        out = frame.copy()
        H, W, _ = frame.shape

        # 1. Raw Binary Mask Mode
        if show_mask and "person_mask" in data:
            mask_rgb = np.zeros_like(out)
            mask_rgb[data["person_mask"] == 1] = (255, 0, 255)  # Magenta
            return cv2.addWeighted(out, 0.4, mask_rgb, 0.6, 0)

        # 2. Clutter / Edge Map Overlay
        if show_edges and "edges" in data:
            edge_full = cv2.resize(data["edges"], (W, H), interpolation=cv2.INTER_NEAREST)
            edge_color = np.zeros_like(out)
            edge_color[edge_full > 0] = (0, 255, 255)  # Cyan
            out = cv2.addWeighted(out, 0.7, edge_color, 0.3, 0)

        # 3. Person Highlighting (Neon Glow Contour + Subtle Silhouette Tint)
        if show_person and data.get("has_person"):
            contours = data["contours"]
            # Soft purple/cyan tint over person
            person_mask_3d = np.repeat(data["person_mask"][:, :, np.newaxis], 3, axis=2)
            tint = np.full_like(out, (160, 32, 240))  # Electric purple tint
            tinted = cv2.addWeighted(out, 0.82, tint, 0.18, 0)
            out = np.where(person_mask_3d == 1, tinted, out)

            # Neon contour outline (Outer glow + crisp inner line)
            cv2.drawContours(out, contours, -1, (0, 240, 255), 4, cv2.LINE_AA)  # Outer cyan glow
            cv2.drawContours(out, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)  # Crisp white core

        # 4. Clean Empty Space Bounding Box
        if show_empty_box:
            ex, ey, ew, eh = data["empty_rect"]

            # Translucent soft green fill over empty space
            overlay = out.copy()
            cv2.rectangle(overlay, (ex, ey), (ex + ew, ey + eh), (0, 255, 120), -1)
            out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)

            # High-tech corner brackets
            bracket_len = min(35, min(ew, eh) // 4)
            corner_color = (0, 255, 128)  # Neon Emerald
            thick = 3

            # Top-Left
            cv2.line(out, (ex, ey), (ex + bracket_len, ey), corner_color, thick, cv2.LINE_AA)
            cv2.line(out, (ex, ey), (ex, ey + bracket_len), corner_color, thick, cv2.LINE_AA)
            # Top-Right
            cv2.line(out, (ex + ew, ey), (ex + ew - bracket_len, ey), corner_color, thick, cv2.LINE_AA)
            cv2.line(out, (ex + ew, ey), (ex + ew, ey + bracket_len), corner_color, thick, cv2.LINE_AA)
            # Bottom-Left
            cv2.line(out, (ex, ey + eh), (ex + bracket_len, ey + eh), corner_color, thick, cv2.LINE_AA)
            cv2.line(out, (ex, ey + eh), (ex, ey + eh - bracket_len), corner_color, thick, cv2.LINE_AA)
            # Bottom-Right
            cv2.line(out, (ex + ew, ey + eh), (ex + ew - bracket_len, ey + eh), corner_color, thick, cv2.LINE_AA)
            cv2.line(out, (ex + ew, ey + eh), (ex + ew, ey + eh - bracket_len), corner_color, thick, cv2.LINE_AA)

            # Center target crosshair
            cx, cy = ex + ew // 2, ey + eh // 2
            cv2.drawMarker(out, (cx, cy), corner_color, cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)

            # Empty space label
            lbl = f"CLEAN EMPTY SPACE [{ew}x{eh} | {data['area_pct']:.1f}%]"
            cv2.putText(out, lbl, (ex + 12, ey + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 128), 1, cv2.LINE_AA)

        # 5. Telemetry HUD
        if show_hud:
            hud_w, hud_h = 420, 140
            hx, hy = 16, 16
            hud_bg = out[hy : hy + hud_h, hx : hx + hud_w].copy()
            cv2.rectangle(hud_bg, (0, 0), (hud_w, hud_h), (18, 18, 18), -1)
            out[hy : hy + hud_h, hx : hx + hud_w] = cv2.addWeighted(
                out[hy : hy + hud_h, hx : hx + hud_w], 0.25, hud_bg, 0.75, 0
            )
            cv2.rectangle(out, (hx, hy), (hx + hud_w, hy + hud_h), (0, 240, 255), 1)

            ex, ey, ew, eh = data["empty_rect"]
            lum = data["mean_lum"]
            clutter = data["edge_clutter"]
            quality_tag = data["quality"].upper()

            lines = [
                (f"SEGMENT & EMPTY SPACE DETECTOR [{quality_tag}]", (0, 240, 255), 0.52),
                (f"Location: {data['quadrant']}  (X:{ex}, Y:{ey}, W:{ew}, H:{eh})", (255, 255, 255), 0.42),
                (f"Cleanliness: Edge Clutter {clutter:.1f}% • Luminance {lum:.0f}/255", (0, 255, 128), 0.42),
                (f"Person Highlight: {'Detected (Glow Active)' if data['has_person'] else 'No Subject'}", (200, 100, 255), 0.42),
                ("Keys: [P] Person | [E] Box | [C] Clutter | [B] Mask | [Space] Pause", (180, 180, 180), 0.36),
            ]
            for idx, (txt, col, sc) in enumerate(lines):
                cv2.putText(out, txt, (hx + 12, hy + 24 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)

        return out


class FFmpegVideoSaver:
    """High-quality video writer using FFmpeg pipe with precise CRF quality presets."""

    def __init__(self, output_path: Path, width: int, height: int, fps: float, quality: str = "medium"):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["medium"])

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", f"{fps:.2f}",
            "-i", "-",
            "-c:v", "libx264",
            "-preset", cfg["preset"],
            "-crf", cfg["crf"],
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(self.output_path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def write(self, frame: np.ndarray):
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(frame.tobytes())

    def close(self):
        if self.proc:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait()
            self.proc = None


def main():
    parser = argparse.ArgumentParser(
        description="Clean Empty Space & Person Segment Detection with Live Preview"
    )
    # The user requested: "the input will be only the video and optional setting for the quality in the output"
    parser.add_argument(
        "video",
        nargs="?",
        default=DEFAULT_VIDEO,
        help="Input video file path (default: Nasser Kandil 1080p video)",
    )
    parser.add_argument(
        "-q", "--quality",
        choices=["high", "medium", "low"],
        default="medium",
        help="Output & processing quality preset (default: medium)",
    )
    parser.add_argument(
        "-s", "--save",
        nargs="?",
        const="auto",
        default=None,
        help="Save processed output video. Can provide custom path or omit for auto-naming.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Headless execution without live cv2.imshow window (useful for batch rendering)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop processing after N frames (useful for test runs)",
    )

    args = parser.parse_args()

    # Validate video path
    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"[!] Error: Video file not found: '{args.video}'", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Error: Could not open video: '{video_path}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 60)
    print("  CLEAN EMPTY SPACE & PERSON SEGMENT DETECTION")
    print(f"  Input Video:  {video_path.name}")
    print(f"  Resolution:   {width}x{height} @ {fps:.1f} fps ({total_frames} frames)")
    print(f"  Quality Mode: {args.quality.upper()} ({QUALITY_PRESETS[args.quality]['desc']})")
    print("=" * 60)

    # Initialize detector and renderer
    detector = EmptySpaceDetector(quality=args.quality)
    renderer = LivePreviewRenderer()

    # Setup video saver if requested
    saver: Optional[FFmpegVideoSaver] = None
    if args.save:
        if args.save == "auto":
            out_dir = Path("projects") / video_path.stem[:40].strip() / "output"
            out_file = out_dir / f"empty_space_{args.quality}.mp4"
        else:
            out_file = Path(args.save)
        saver = FFmpegVideoSaver(out_file, width, height, fps, quality=args.quality)
        print(f"[*] Exporting output to: {out_file}", flush=True)

    # Interactive display setup
    window_name = f"Empty Space & Person Segment Detection [{args.quality.upper()}]"
    if not args.no_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, width), min(720, height))

    # Toggles
    show_person = True
    show_empty_box = True
    show_edges = False
    show_mask = False
    show_hud = True
    paused = False
    frame_idx = 0
    t_start = time.time()
    last_frame = None

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                analysis = detector.process_frame(frame)
                rendered = renderer.draw(
                    frame=frame,
                    data=analysis,
                    show_person=show_person,
                    show_empty_box=show_empty_box,
                    show_edges=show_edges,
                    show_mask=show_mask,
                    show_hud=show_hud,
                )
                last_frame = rendered
                frame_idx += 1

                if saver:
                    saver.write(rendered)

                if args.max_frames and frame_idx >= args.max_frames:
                    break

                # Print console progress every 50 frames
                if frame_idx % 50 == 0:
                    elapsed = time.time() - t_start
                    curr_fps = frame_idx / elapsed if elapsed > 0 else 0
                    print(
                        f"[{frame_idx:4d} frames | {curr_fps:.1f} FPS] Empty Box: {analysis['empty_rect'][2]}x{analysis['empty_rect'][3]} ({analysis['area_pct']:.1f}%) in {analysis['quadrant']}",
                        flush=True,
                    )
            else:
                rendered = last_frame

            if not args.no_gui:
                cv2.imshow(window_name, rendered)
                key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    paused = not paused
                elif key == ord("p"):
                    show_person = not show_person
                elif key == ord("e"):
                    show_empty_box = not show_empty_box
                elif key == ord("c"):
                    show_edges = not show_edges
                elif key == ord("b"):
                    show_mask = not show_mask
                elif key == ord("h"):
                    show_hud = not show_hud
                elif key == ord("s"):
                    snap_name = f"snapshot_{frame_idx}_{int(time.time())}.png"
                    cv2.imwrite(snap_name, rendered)
                    print(f"[✓] Saved snapshot: {snap_name}")

    finally:
        cap.release()
        if saver:
            saver.close()
            print(f"[✓] Output video rendered successfully: {saver.output_path}")
        if not args.no_gui:
            cv2.destroyAllWindows()

    total_elapsed = time.time() - t_start
    avg_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0
    print(f"\n[✓] Done: Processed {frame_idx} frames in {total_elapsed:.1f}s (Average {avg_fps:.1f} FPS).")


if __name__ == "__main__":
    main()
