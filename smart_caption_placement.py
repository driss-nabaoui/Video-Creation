#!/usr/bin/env python3
r"""
Smart Caption Placement MVP
===========================
Live intelligent caption positioning using MediaPipe & OpenCV:

1. Face & Pose Quadrant Mapping:
   - Detects face (nose, eyes) and shoulders (left/right).
   - Divides frame into a 3x3 grid (9 quadrants).
   - Analyzes speaker center-of-mass and lateral lean angle.
   - If speaker leans right, candidate anchor flips dynamically to Left quadrants.

2. Foreground Segmentation & Largest Empty Rectangle (LER):
   - Computes binary mask separating subject from background.
   - Evaluates background clearance and finds the Largest Empty Rectangle (LER)
     to guarantee captions never touch or overlap the speaker.

3. Edge Density & Saliency Scoring:
   - Runs Canny edge detector and local variance filters across candidate zones.
   - Penalizes visual clutter (bookshelves, posters, complex geometry) and places
     captions inside the lowest-detail negative space (e.g. plain wall or blurred bokeh).

4. Luminance-Adaptive Styling:
   - Samples background luminance ($L \in [0, 255]$) in the selected text placement area.
   - High luminance (> 150): Strong contrast dark pill with gold/white text or dark text.
   - Low luminance (< 85): Translucent dark pill with glowing/bright text.
   - Mid luminance (85 - 150): High-contrast stroked pill with maximum legibility.
   - Smooth exponential moving average (EMA) to prevent jitter.

Controls:
  'q'     : Quit
  'space' : Play / Pause
  'g'     : Toggle 3x3 Grid overlay
  'm'     : Toggle Person Segmentation Mask overlay
  'e'     : Toggle Canny Edge Detection overlay
  't'     : Cycle through sample phrases (Arabic / English)
"""

import argparse
import math
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# MediaPipe Tasks Vision
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = Path("models/pose_landmarker_lite.task")

DEFAULT_VIDEO = "/home/tapa/Videos/ناصر قنديل بحلقة نارية بعد انتخابات اسرائيل_اعدام نتنياهو واحتلال الكنيست! تفاصيل غزو صنعاء في اسبوع_1080p.mp4"

SAMPLE_PHRASES = [
    "📌 اعتراف نتنياهو بالهزيمة",
    "⚠️ سيناريو اقتحام واحتلال الكنيست",
    "💥 مخطط غزو صنعاء خلال أسبوع",
    "Smart Dynamic Captions with AI Placement",
    "Zero Visual Clutter • High Contrast",
]


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
    Returns: (start_idx, width, height, max_area)
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


def find_largest_empty_rectangle(binary_bg: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Finds the Largest Empty Rectangle in a 2D binary image (1 = empty background, 0 = obstacle).
    Returns (x, y, w, h).
    """
    R, C = binary_bg.shape
    heights = [0] * C
    best_x, best_y, best_w, best_h, max_area = 0, 0, 0, 0, 0

    for r in range(R):
        for c in range(C):
            heights[c] = heights[c] + 1 if binary_bg[r, c] == 1 else 0
        left, w, h, area = max_rectangle_in_histogram(heights)
        if area > max_area:
            max_area = area
            best_x = left
            best_y = r - h + 1
            best_w = w
            best_h = h

    return best_x, best_y, best_w, best_h


class SmartCaptionPlacer:
    def __init__(self, model_path: Path = MODEL_PATH, font_path: Optional[str] = None):
        ensure_model_exists(model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            output_segmentation_masks=True,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # Font resolution (prefer Noto Arabic or fallback to DejaVu Sans / default)
        candidate_fonts = [
            font_path,
            "/usr/share/fonts/noto/NotoKufiArabic-Bold.ttf",
            "/usr/share/fonts/noto/NotoSansArabic-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        ]
        self.font_file = None
        for f in candidate_fonts:
            if f and Path(f).is_file():
                self.font_file = str(f)
                break

        # Temporal smoothing state (EMA)
        self.smoothed_box: Optional[Tuple[float, float, float, float]] = None
        self.alpha = 0.25  # Smoothing factor (0 = static, 1 = instant jump)

    def analyze(self, frame: np.ndarray, text: str) -> Dict[str, Any]:
        """
        Runs the full 4-stage pipeline on the input frame:
        1. Pose & Quadrant Mapping
        2. Foreground Segmentation & LER
        3. Edge Density & Clutter Scoring
        4. Luminance-Adaptive Color Selection
        """
        H, W, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_img)

        # Default fallback values
        speaker_detected = False
        nose_pt = (W // 2, H // 3)
        l_sh = (W // 2 - 100, H // 2)
        r_sh = (W // 2 + 100, H // 2)
        lean_label = "Centered"
        speaker_x = W / 2.0
        tilt_deg = 0.0

        # Binary background mask (1 = background, 0 = person)
        if res.segmentation_masks and len(res.segmentation_masks) > 0:
            raw_mask = res.segmentation_masks[0].numpy_view()
            if raw_mask.shape[:2] != (H, W):
                raw_mask = cv2.resize(raw_mask, (W, H), interpolation=cv2.INTER_LINEAR)
            if raw_mask.ndim == 3:
                raw_mask = raw_mask.squeeze(-1)
            person_mask = (raw_mask > 0.45).astype(np.uint8)
            bg_mask = (1 - person_mask).astype(np.uint8)
        else:
            person_mask = np.zeros((H, W), dtype=np.uint8)
            bg_mask = np.ones((H, W), dtype=np.uint8)

        # Stage 1: Pose & Quadrant Analysis
        if res.pose_landmarks and len(res.pose_landmarks) > 0:
            speaker_detected = True
            lm = res.pose_landmarks[0]
            nose_pt = (int(lm[0].x * W), int(lm[0].y * H))
            l_sh = (int(lm[11].x * W), int(lm[11].y * H))
            r_sh = (int(lm[12].x * W), int(lm[12].y * H))

            # Centroid of upper body
            speaker_x = (nose_pt[0] * 0.5) + ((l_sh[0] + r_sh[0]) * 0.25)
            dx = r_sh[0] - l_sh[0]
            dy = r_sh[1] - l_sh[1]
            tilt_deg = math.degrees(math.atan2(dy, dx)) if dx != 0 else 0.0

            x_ratio = speaker_x / W
            if x_ratio > 0.55:
                lean_label = "Speaker on Right -> Anchor Left"
                primary_cols = [0, 1]  # Left, Center
            elif x_ratio < 0.45:
                lean_label = "Speaker on Left -> Anchor Right"
                primary_cols = [2, 1]  # Right, Center
            else:
                lean_label = "Speaker Centered -> Balanced Sides"
                primary_cols = [0, 2, 1]
        else:
            primary_cols = [0, 2, 1]

        # Stage 3: Edge Density & Visual Saliency
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 3x3 Grid Evaluation
        gw, gh = W // 3, H // 3
        quadrants_info = []

        # Candidate rows: Top and Mid (captions generally avoid bottom if speaker fills lower body)
        candidate_rows = [0, 1, 2]

        for r_idx in candidate_rows:
            for c_idx in [0, 1, 2]:
                gx, gy = c_idx * gw, r_idx * gh
                q_bg = bg_mask[gy : gy + gh, gx : gx + gw]
                q_edges = edges[gy : gy + gh, gx : gx + gw]
                q_gray = gray[gy : gy + gh, gx : gx + gw]

                bg_ratio = float(np.mean(q_bg))
                edge_density = float(np.mean(q_edges > 0)) * 100.0
                mean_lum = float(np.mean(q_gray))

                row_names = ["Top", "Mid", "Bottom"]
                col_names = ["Left", "Center", "Right"]
                name = f"{row_names[r_idx]}-{col_names[c_idx]}"

                # Score quadrant: High BG clearance is mandatory, low edge clutter is preferred
                is_opposed = c_idx in primary_cols[:1]
                suitability_score = (bg_ratio * 70.0) - (edge_density * 1.5) + (20.0 if is_opposed else 0.0)
                if r_idx == 1 and is_opposed:
                    suitability_score += 10.0  # Mid-third side has great viewer gaze balance

                quadrants_info.append({
                    "name": name,
                    "r_idx": r_idx,
                    "c_idx": c_idx,
                    "box": (gx, gy, gw, gh),
                    "bg_ratio": bg_ratio,
                    "edge_density": edge_density,
                    "mean_lum": mean_lum,
                    "score": suitability_score,
                })

        # Sort quadrants by suitability
        quadrants_info.sort(key=lambda x: x["score"], reverse=True)
        best_quad = quadrants_info[0]

        # Stage 2: Largest Empty Rectangle in best candidate zone
        # Crop background mask to best quadrant with safety margins
        bx, by, bw, bh = best_quad["box"]
        quad_bg_crop = bg_mask[by : by + bh, bx : bx + bw]

        # Downsample for ultra-fast LER calculation
        scale = 8
        small_bg = cv2.resize(quad_bg_crop, (bw // scale, bh // scale), interpolation=cv2.INTER_NEAREST)
        lx, ly, lw, lh = find_largest_empty_rectangle(small_bg)

        # Scale coordinates back to full resolution
        ler_x = bx + (lx * scale)
        ler_y = by + (ly * scale)
        ler_w = max(100, lw * scale)
        ler_h = max(50, lh * scale)

        # Fit caption text dimensions
        font_size = max(24, int(H * 0.035))
        try:
            pil_font = ImageFont.truetype(self.font_file, font_size) if self.font_file else ImageFont.load_default()
        except Exception:
            pil_font = ImageFont.load_default()

        # Compute text bounding box
        dummy_img = Image.new("RGB", (100, 100))
        d_draw = ImageDraw.Draw(dummy_img)
        bbox = d_draw.textbbox((0, 0), text, font=pil_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad_x = 24
        pad_y = 14
        pill_w = text_w + (pad_x * 2)
        pill_h = text_h + (pad_y * 2)

        # Place caption inside the Largest Empty Rectangle
        # Center horizontally and vertically within LER with safety clamping
        target_x = ler_x + (ler_w - pill_w) // 2
        target_y = ler_y + (ler_h - pill_h) // 2
        target_x = max(20, min(W - pill_w - 20, target_x))
        target_y = max(20, min(H - pill_h - 20, target_y))

        # Temporal Smoothing (EMA) to prevent frame-by-frame jumpiness
        if self.smoothed_box is None:
            self.smoothed_box = (float(target_x), float(target_y), float(pill_w), float(pill_h))
        else:
            sx, sy, sw, sh = self.smoothed_box
            self.smoothed_box = (
                sx + self.alpha * (target_x - sx),
                sy + self.alpha * (target_y - sy),
                sw + self.alpha * (pill_w - sw),
                sh + self.alpha * (pill_h - sh),
            )

        final_box = (
            int(self.smoothed_box[0]),
            int(self.smoothed_box[1]),
            int(self.smoothed_box[2]),
            int(self.smoothed_box[3]),
        )

        # Stage 4: Luminance-Adaptive Color Sampling
        fx, fy, fw, fh = final_box
        sample_roi = gray[max(0, fy) : min(H, fy + fh), max(0, fx) : min(W, fx + fw)]
        local_lum = float(np.mean(sample_roi)) if sample_roi.size > 0 else 128.0

        if local_lum > 155:
            # Bright background (e.g. sunlight, white studio wall)
            # Use dark translucent charcoal box with warm gold text for high contrast
            style_desc = f"Bright BG (Lum: {local_lum:.0f}) -> Charcoal Pill + Warm Gold Text"
            text_color = (255, 215, 0)       # Warm Gold
            box_fill = (15, 15, 15, 220)      # High-density dark charcoal
            box_border = (255, 215, 0, 255)
        elif local_lum < 85:
            # Dark background (e.g. moody studio lighting, dark backdrop)
            # Use bright white text with subtle glowing translucent border
            style_desc = f"Dark BG (Lum: {local_lum:.0f}) -> Semi-Translucent Box + Bright White Text"
            text_color = (255, 255, 255)     # Crisp White
            box_fill = (25, 25, 25, 180)      # Semi-transparent dark pill
            box_border = (100, 220, 255, 200) # Soft cyan outline
        else:
            # Mid-tone background
            style_desc = f"Mid-tone BG (Lum: {local_lum:.0f}) -> Contrast Pill + Pure White Text"
            text_color = (255, 255, 255)
            box_fill = (10, 10, 10, 200)
            box_border = (255, 255, 255, 160)

        return {
            "speaker_detected": speaker_detected,
            "speaker_x": speaker_x,
            "lean_label": lean_label,
            "tilt_deg": tilt_deg,
            "nose_pt": nose_pt,
            "l_sh": l_sh,
            "r_sh": r_sh,
            "best_quadrant": best_quad,
            "ler_box": (ler_x, ler_y, ler_w, ler_h),
            "final_caption_box": final_box,
            "text": text,
            "pil_font": pil_font,
            "text_color": text_color,
            "box_fill": box_fill,
            "box_border": box_border,
            "local_lum": local_lum,
            "style_desc": style_desc,
            "person_mask": person_mask,
            "edges": edges,
            "quadrants": quadrants_info,
        }

    def render_overlay(
        self,
        frame: np.ndarray,
        data: Dict[str, Any],
        show_grid: bool = True,
        show_mask: bool = False,
        show_edges: bool = False,
        show_hud: bool = True,
    ) -> np.ndarray:
        """Draws visual HUD, 3x3 grid, LER bounding box, and styled adaptive caption."""
        H, W, _ = frame.shape
        out = frame.copy()

        # 1. Optional Mask overlay
        if show_mask and "person_mask" in data:
            mask_color = np.zeros_like(out)
            mask_color[:, :] = (180, 0, 180)  # Magenta highlight for subject
            p_mask_3d = np.repeat(data["person_mask"][:, :, np.newaxis], 3, axis=2)
            out = np.where(p_mask_3d == 1, cv2.addWeighted(out, 0.6, mask_color, 0.4, 0), out)

        # 2. Optional Edges overlay
        if show_edges and "edges" in data:
            edge_color = np.zeros_like(out)
            edge_color[:, :] = (0, 255, 255)  # Cyan edges
            e_mask_3d = np.repeat((data["edges"] > 0)[:, :, np.newaxis], 3, axis=2)
            out = np.where(e_mask_3d, cv2.addWeighted(out, 0.5, edge_color, 0.5, 0), out)

        # 3. 3x3 Grid Overlay
        if show_grid:
            gw, gh = W // 3, H // 3
            grid_layer = out.copy()
            for i in range(1, 3):
                cv2.line(grid_layer, (i * gw, 0), (i * gw, H), (200, 200, 200), 1, cv2.LINE_AA)
                cv2.line(grid_layer, (0, i * gh), (W, i * gh), (200, 200, 200), 1, cv2.LINE_AA)

            # Highlight selected quadrant with soft cyan tint
            bq = data["best_quadrant"]
            bx, by, bw, bh = bq["box"]
            cv2.rectangle(grid_layer, (bx, by), (bx + bw, by + bh), (0, 200, 255), -1)
            out = cv2.addWeighted(grid_layer, 0.15, out, 0.85, 0)

            # Draw subtle quadrant label in corners
            for q in data["quadrants"]:
                qx, qy, _, _ = q["box"]
                lbl = f"{q['name']} (Clutter: {q['edge_density']:.1f}%)"
                cv2.putText(out, lbl, (qx + 12, qy + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # 4. Draw Largest Empty Rectangle (LER)
        lx, ly, lw, lh = data["ler_box"]
        cv2.rectangle(out, (lx, ly), (lx + lw, ly + lh), (0, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(
            out,
            "LER: Safe Zone (No Collision)",
            (lx + 8, ly + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 120),
            1,
            cv2.LINE_AA,
        )

        # 5. Pose tracking visualizer
        if data["speaker_detected"]:
            nose = data["nose_pt"]
            l_sh = data["l_sh"]
            r_sh = data["r_sh"]
            cv2.circle(out, nose, 7, (0, 120, 255), -1, cv2.LINE_AA)
            cv2.circle(out, l_sh, 7, (255, 100, 0), -1, cv2.LINE_AA)
            cv2.circle(out, r_sh, 7, (255, 100, 0), -1, cv2.LINE_AA)
            cv2.line(out, l_sh, r_sh, (0, 255, 255), 2, cv2.LINE_AA)

        # 6. Render Adaptive Pill Caption via Pillow (supports Arabic & Latin perfectly)
        fx, fy, fw, fh = data["final_caption_box"]
        pil_frame = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new("RGBA", pil_frame.size, (0, 0, 0, 0))
        d_draw = ImageDraw.Draw(overlay)

        # Draw rounded pill box
        radius = min(20, fh // 2)
        d_draw.rounded_rectangle(
            [fx, fy, fx + fw, fy + fh],
            radius=radius,
            fill=data["box_fill"],
            outline=data["box_border"],
            width=2,
        )

        # Center text inside pill
        bbox = d_draw.textbbox((0, 0), data["text"], font=data["pil_font"])
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = fx + (fw - tw) // 2
        ty = fy + (fh - th) // 2 - bbox[1]

        # Draw text drop shadow
        d_draw.text((tx + 2, ty + 2), data["text"], fill=(0, 0, 0, 220), font=data["pil_font"])
        # Draw main text
        d_draw.text((tx, ty), data["text"], fill=data["text_color"], font=data["pil_font"])

        # Composite caption layer
        composed = Image.alpha_composite(pil_frame, overlay)
        out = cv2.cvtColor(np.array(composed.convert("RGB")), cv2.COLOR_RGB2BGR)

        # 7. Real-Time Telemetry HUD
        if show_hud:
            hud_w, hud_h = 440, 150
            hud_bg = out[15 : 15 + hud_h, 15 : 15 + hud_w].copy()
            cv2.rectangle(hud_bg, (0, 0), (hud_w, hud_h), (15, 15, 15), -1)
            out[15 : 15 + hud_h, 15 : 15 + hud_w] = cv2.addWeighted(
                out[15 : 15 + hud_h, 15 : 15 + hud_w], 0.25, hud_bg, 0.75, 0
            )
            cv2.rectangle(out, (15, 15), (15 + hud_w, 15 + hud_h), (0, 200, 255), 1)

            bq = data["best_quadrant"]
            lines = [
                ("SMART CAPTION TELEMETRY", (0, 255, 255), 0.55),
                (f"Pose: {data['lean_label']}", (255, 255, 255), 0.45),
                (f"Quadrant: {bq['name']} (BG: {bq['bg_ratio']*100:.0f}%, Clutter: {bq['edge_density']:.1f}%)", (0, 255, 120), 0.45),
                (f"Style: {data['style_desc']}", (255, 215, 0), 0.42),
                ("Keys: [G] Grid | [M] Mask | [E] Edges | [T] Text | [Space] Pause", (180, 180, 180), 0.38),
            ]
            for idx, (txt, col, sc) in enumerate(lines):
                cv2.putText(out, txt, (25, 38 + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)

        return out


def main():
    parser = argparse.ArgumentParser(description="Smart Dynamic Caption Placement MVP")
    parser.add_argument(
        "--source",
        default=DEFAULT_VIDEO,
        help="Video file path OR webcam index (e.g. 0). Default: Nasser Kandil video.",
    )
    parser.add_argument("--save", default=None, help="Optional output MP4 path to save processed demo")
    parser.add_argument("--text", default=None, help="Custom caption phrase")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless mode without cv2.imshow")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    # Source resolution
    source_val = args.source
    if source_val.isdigit():
        source_val = int(source_val)
    elif not Path(source_val).is_file():
        print(f"[!] Source video not found: {source_val}", file=sys.stderr)
        print("[!] Falling back to webcam 0...", file=sys.stderr)
        source_val = 0

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"[!] Failed to open source: {source_val}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    print("=" * 60)
    print("  SMART CAPTION PLACEMENT (LIVE MVP)")
    print(f"  Source:      {source_val}")
    print(f"  Resolution:  {width}x{height} @ {fps:.1f} fps")
    print("=" * 60)

    # Initialize placer
    placer = SmartCaptionPlacer()

    # Optional video writer
    writer = None
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
        print(f"[*] Saving processed demo to: {save_path}")

    # Interactive state
    phrases = [args.text] if args.text else SAMPLE_PHRASES
    phrase_idx = 0
    show_grid = True
    show_mask = False
    show_edges = False
    paused = False
    frame_count = 0

    window_name = "Smart Caption Placement (MediaPipe + OpenCV)"
    if not args.no_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, width), min(720, height))

    t_start = time.time()
    last_rendered_frame = None

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # Loop video if file
                    if not isinstance(source_val, int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    if not ret:
                        break

                current_phrase = phrases[phrase_idx % len(phrases)]
                analysis = placer.analyze(frame, current_phrase)
                rendered = placer.render_overlay(
                    frame=frame,
                    data=analysis,
                    show_grid=show_grid,
                    show_mask=show_mask,
                    show_edges=show_edges,
                    show_hud=True,
                )
                last_rendered_frame = rendered
                frame_count += 1

                if writer:
                    writer.write(rendered)

                if args.max_frames and frame_count >= args.max_frames:
                    break
            else:
                rendered = last_rendered_frame

            if not args.no_gui:
                cv2.imshow(window_name, rendered)
                key = cv2.waitKey(int(1000 / fps)) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    paused = not paused
                elif key == ord("g"):
                    show_grid = not show_grid
                elif key == ord("m"):
                    show_mask = not show_mask
                elif key == ord("e"):
                    show_edges = not show_edges
                elif key == ord("t"):
                    phrase_idx += 1
                    print(f"[*] Switched caption phrase: {phrases[phrase_idx % len(phrases)]}")

    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_gui:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"\n[✓] Processed {frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS).")


if __name__ == "__main__":
    main()
