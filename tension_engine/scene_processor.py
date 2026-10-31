"""
Scene Processor: Multi-Signal Fusion, Temporal Smoothing, Spike Merging & Context Padding.
Combines Acoustic (35%) + Overlap (40%) + Hostility (25%) into a unified 0-100 tension score.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d


class SceneProcessor:
    """
    Fuses multi-modal tension signals, applies anti-spurious smoothing, merges nearby outbursts,
    and applies lead-in/lead-out context padding.
    """

    def __init__(
        self,
        weight_acoustic: float = 0.35,
        weight_overlap: float = 0.40,
        weight_sentiment: float = 0.25,
        default_threshold: float = 60.0,
        merge_gap: float = 8.0,
        padding_before: float = 4.0,
        padding_after: float = 3.0,
        min_scene_duration: float = 3.0,
    ):
        self.w_acoustic = weight_acoustic
        self.w_overlap = weight_overlap
        self.w_sentiment = weight_sentiment
        self.threshold = default_threshold
        self.merge_gap = merge_gap
        self.padding_before = padding_before
        self.padding_after = padding_after
        self.min_scene_duration = min_scene_duration

    def process(
        self,
        acoustic_scores: List[float],
        overlap_scores: List[float],
        sentiment_scores: List[float],
        total_duration: float,
        whisper_segments: Optional[List[Dict[str, Any]]] = None,
        threshold_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Combines signals and extracts high-tension scenes.
        Returns:
        - "unified_scores": List[float] per-second tension scores (0-100)
        - "smoothed_scores": List[float] smoothed scores
        - "scenes": List[Dict] ranked high-tension scenes
        """
        length = max(len(acoustic_scores), len(overlap_scores), len(sentiment_scores))
        if length == 0:
            return {"unified_scores": [], "smoothed_scores": [], "scenes": []}

        # Pad all signals to same length
        def pad_arr(lst: List[float], n: int) -> np.ndarray:
            arr = np.array(lst, dtype=np.float32)
            if len(arr) < n:
                arr = np.pad(arr, (0, n - len(arr)), mode="edge")
            return arr[:n]

        ac = pad_arr(acoustic_scores, length)
        ov = pad_arr(overlap_scores, length)
        st = pad_arr(sentiment_scores, length)

        # 1. Multi-signal weighted fusion (0 - 100)
        raw_unified = (self.w_acoustic * ac) + (self.w_overlap * ov) + (self.w_sentiment * st)
        raw_unified = np.clip(raw_unified, 0.0, 100.0)

        # 2. Temporal Smoothing:
        # Step A: 3-second median filter rejects isolated 1-second transient spikes (coughs, mic bumps)
        med_filtered = median_filter(raw_unified, size=3, mode="reflect")
        # Step B: Gaussian filter (sigma=1.0) produces clean, natural tension contours
        smoothed = gaussian_filter1d(med_filtered, sigma=1.0, mode="reflect")
        smoothed = np.clip(smoothed, 0.0, 100.0)

        # 3. Identify intervals exceeding threshold
        active_threshold = threshold_override if threshold_override is not None else self.threshold
        is_high_tension = smoothed >= active_threshold

        candidate_spans = []
        in_span = False
        span_start = 0

        for sec in range(length):
            if is_high_tension[sec] and not in_span:
                in_span = True
                span_start = sec
            elif not is_high_tension[sec] and in_span:
                in_span = False
                candidate_spans.append((float(span_start), float(sec)))

        if in_span:
            candidate_spans.append((float(span_start), float(length)))

        # 4. Spike Merging: Merge outbursts within merge_gap seconds (e.g. 8.0s)
        merged_spans = []
        for span in candidate_spans:
            if not merged_spans:
                merged_spans.append(span)
            else:
                prev_start, prev_end = merged_spans[-1]
                cur_start, cur_end = span
                if (cur_start - prev_end) <= self.merge_gap:
                    # Merge with previous span
                    merged_spans[-1] = (prev_start, max(prev_end, cur_end))
                else:
                    merged_spans.append(span)

        # 5. Context Padding: Add lead-in and lead-out padding
        padded_scenes = []
        for s_idx, (st_t, en_t) in enumerate(merged_spans):
            padded_start = max(0.0, round(st_t - self.padding_before, 2))
            padded_end = min(total_duration, round(en_t + self.padding_after, 2))
            padded_scenes.append({
                "raw_start": round(st_t, 2),
                "raw_end": round(en_t, 2),
                "start": padded_start,
                "end": padded_end,
                "duration": round(padded_end - padded_start, 2),
            })

        # Merge any scenes that overlap after padding
        consolidated = []
        for sc in padded_scenes:
            if not consolidated:
                consolidated.append(sc)
            else:
                prev = consolidated[-1]
                if sc["start"] <= prev["end"]:
                    # Overlap after padding -> combine
                    prev["end"] = max(prev["end"], sc["end"])
                    prev["raw_end"] = max(prev["raw_end"], sc["raw_end"])
                    prev["duration"] = round(prev["end"] - prev["start"], 2)
                else:
                    consolidated.append(sc)

        # Filter out very short scenes (< min_scene_duration)
        final_scenes = [sc for sc in consolidated if sc["duration"] >= self.min_scene_duration]

        # 6. Score Telemetry & Ranking
        for sc in final_scenes:
            st_idx = int(np.floor(sc["start"]))
            en_idx = int(np.ceil(sc["end"]))
            scene_slice = smoothed[st_idx:en_idx] if st_idx < len(smoothed) else np.array([0.0])

            sc["peak_tension_score"] = round(float(np.max(scene_slice)), 1) if len(scene_slice) > 0 else 0.0
            sc["avg_tension_score"] = round(float(np.mean(scene_slice)), 1) if len(scene_slice) > 0 else 0.0

            # Component peak telemetry
            ac_slice = ac[st_idx:en_idx] if st_idx < len(ac) else [0.0]
            ov_slice = ov[st_idx:en_idx] if st_idx < len(ov) else [0.0]
            st_slice = st[st_idx:en_idx] if st_idx < len(st) else [0.0]
            sc["signals"] = {
                "peak_acoustic": round(float(np.max(ac_slice)), 1),
                "peak_overlap": round(float(np.max(ov_slice)), 1),
                "peak_sentiment": round(float(np.max(st_slice)), 1),
            }

            # Associate transcript text snippet if available
            if whisper_segments:
                matching_texts = []
                for seg in whisper_segments:
                    seg_st = float(seg.get("start", 0.0))
                    seg_en = float(seg.get("end", 0.0))
                    if seg_en >= sc["start"] and seg_st <= sc["end"]:
                        matching_texts.append(seg.get("text", ""))
                sc["transcript_snippet"] = " ".join(matching_texts[:4]).strip()
            else:
                sc["transcript_snippet"] = ""

        # Rank by peak tension score (primary) and avg score (secondary)
        final_scenes.sort(key=lambda x: (x["peak_tension_score"], x["avg_tension_score"]), reverse=True)
        for rank_idx, sc in enumerate(final_scenes, 1):
            sc["rank"] = rank_idx
            sc["scene_id"] = f"scene_{rank_idx:02d}"

        return {
            "unified_scores": [round(float(s), 1) for s in raw_unified],
            "smoothed_scores": [round(float(s), 1) for s in smoothed],
            "scenes": final_scenes,
            "threshold_used": active_threshold,
        }
