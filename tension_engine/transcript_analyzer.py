"""
Transcript Sentiment, Hostility & Speech Rate Analyzer (25% Weight).
Uses faster-whisper with word-level timestamps to detect:
1. Words-Per-Second (WPS) acceleration spikes.
2. Multilingual parliamentary conflict lexicon (Arabic, English, French).
3. Exclamation and arousal punctuation density (!, ؟, ?!).
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


# Multilingual Parliamentary Conflict & High-Arousal Lexicon
CONFLICT_LEXICON = {
    # Arabic Parliamentary / Heated Debate terms
    "ar": [
        "نقطة نظام", "اعتراض", "غير صحيح", "كذب", "كاذب", "كذاب", "اسكت", "اخرس",
        "احترم", "جلسة", "عيب", "فضيحة", "خيانة", "مهزلة", "سرقة", "إرهاب",
        "ساقط", "بلطجة", "دستور", "لا يحق لك", "انسحاب", "قاطع", "فوضى", "صراخ",
        "استفزاز", "ما يمكنش", "بزاف", "كفى", "أوقف", "غير دستوري", "استهتار",
        "تضليل", "تهريج", "وقاحة", "صمت", "حشومة", "كفانا", "ارحل", "سكت",
    ],
    # English Parliamentary terms (UK, US, Commonwealth)
    "en": [
        "point of order", "out of order", "shut up", "sit down", "lie", "liar",
        "shame", "disgrace", "resign", "resignation", "nonsense", "order",
        "unacceptable", "scandal", "coward", "illegal", "withdraw", "silence",
        "shouting", "rubbish", "apologize", "disorder", "corrupt", "corruption",
    ],
    # French Parliamentary terms
    "fr": [
        "rappel au règlement", "mensonge", "menteur", "honte", "taire", "silence",
        "scandale", "démission", "inacceptable", "arrêtez", "hors la loi", "bruit",
        "scandaleux", "respectez", "provocation", "inadmissible", "voleur",
    ],
}


class TranscriptAnalyzer:
    """
    Analyzes spoken text, speech tempo, and lexical hostility using faster-whisper.
    """

    def __init__(self, model_size: str = "base", device: str = "auto", compute_type: str = "auto"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _init_whisper(self):
        """Lazy-loads Faster-Whisper on GPU (CUDA) with CPU fallback."""
        if self._model is not None:
            return

        import os
        import glob
        import ctypes
        import torch

        # Preload nvidia virtualenv libraries for ctranslate2 (libcublas.so.12, libcudnn, etc.)
        try:
            site_packages = os.path.dirname(torch.__file__)
            venv_lib = os.path.dirname(site_packages)
            nvidia_libs = glob.glob(os.path.join(venv_lib, "nvidia", "*", "lib"))
            for p in nvidia_libs:
                for so in sorted(glob.glob(os.path.join(p, "*.so*"))):
                    try:
                        ctypes.CDLL(so)
                    except Exception:
                        pass
        except Exception:
            pass

        from faster_whisper import WhisperModel

        has_cuda = torch.cuda.is_available() and self.device in ("auto", "cuda")
        target_dev = "cuda" if has_cuda else "cpu"
        comp_type = "float16" if has_cuda else "int8"

        print(f"[*] Initializing Faster-Whisper ({self.model_size}) on {target_dev.upper()} ({comp_type})...", flush=True)
        try:
            self._model = WhisperModel(self.model_size, device=target_dev, compute_type=comp_type)
        except Exception as e:
            if target_dev == "cuda":
                print(f"[!] CUDA Whisper init failed ({e}), falling back to CPU (int8)...", flush=True)
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            else:
                raise e

    def transcribe_audio(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        """
        Transcribes audio with word-level acoustic timestamps.
        Returns: (segments, all_words, detected_language)
        """
        self._init_whisper()
        audio_path = Path(audio_path).resolve()

        print(f"[*] Transcribing audio with word-level timestamps...", flush=True)
        # Load audio via soundfile or librosa to bypass PyAV 19 metadata_errors incompatibility
        try:
            import soundfile as sf
            y, sr = sf.read(str(audio_path), dtype="float32")
        except Exception:
            import librosa
            y, sr = librosa.load(str(audio_path), sr=16000, mono=True)

        segs_gen, info = self._model.transcribe(
            y,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
        )

        segments = []
        words = []

        for s in segs_gen:
            seg_dict = {
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": s.text.strip(),
            }
            if s.words:
                seg_dict["words"] = [
                    {"word": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2), "prob": round(w.probability, 2)}
                    for w in s.words if w.word.strip()
                ]
                words.extend(seg_dict["words"])
            segments.append(seg_dict)

        print(f"[+] Transcription complete: {len(segments)} segments, {len(words)} words (Language: {info.language}).", flush=True)
        return segments, words, info.language

    @staticmethod
    def load_transcript_file(transcript_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        """
        Loads pre-computed transcript from JSON file (supports Whisper output, YouTube API, etc.).
        Returns: (segments, words, language)
        """
        import json
        transcript_path = Path(transcript_path).resolve()
        if not transcript_path.is_file():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

        print(f"[*] Ingesting pre-existing transcript from: {transcript_path.name}...", flush=True)
        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        language = "ar"
        segments = []
        words = []

        if isinstance(data, dict):
            language = data.get("language", "ar")
            if "words" in data and isinstance(data["words"], list) and len(data["words"]) > 0:
                words = data["words"]
            if "segments" in data and isinstance(data["segments"], list) and len(data["segments"]) > 0:
                segments = data["segments"]
        elif isinstance(data, list):
            segments = data

        # If we have segments but no standalone words list, extract words from segments
        if not words and segments:
            for seg in segments:
                if "words" in seg and isinstance(seg["words"], list):
                    words.extend(seg["words"])
                else:
                    # Synthesize approximate word timestamps from segment
                    text_words = seg.get("text", "").split()
                    if text_words:
                        st = float(seg.get("start", 0.0))
                        en = float(seg.get("end", st + 1.0))
                        dur = max(0.1, en - st) / len(text_words)
                        for i, w_str in enumerate(text_words):
                            w_st = round(st + i * dur, 2)
                            words.append({
                                "word": w_str,
                                "start": w_st,
                                "end": round(w_st + dur, 2),
                                "probability": 1.0,
                            })

        # If we have words but segments are empty (e.g. YouTube API transcript), group words into phrases
        if words and not segments:
            cur_seg_words = []
            cur_seg_st = None
            for w in words:
                w_st = float(w.get("start", 0.0))
                w_en = float(w.get("end", w_st + 0.3))
                if cur_seg_st is None:
                    cur_seg_st = w_st

                cur_seg_words.append(w.get("word", ""))
                # Break segment on pause > 1.2s or after 12 words
                if len(cur_seg_words) >= 12 or (len(cur_seg_words) > 3 and w.get("word", "").endswith((".", "!", "؟", "?"))):
                    segments.append({
                        "start": round(cur_seg_st, 2),
                        "end": round(w_en, 2),
                        "text": " ".join(cur_seg_words).strip(),
                    })
                    cur_seg_words = []
                    cur_seg_st = None

            if cur_seg_words:
                segments.append({
                    "start": round(cur_seg_st or 0.0, 2),
                    "end": round(float(words[-1].get("end", (cur_seg_st or 0.0) + 1.0)), 2),
                    "text": " ".join(cur_seg_words).strip(),
                })

        print(f"[+] Loaded pre-computed transcript: {len(segments)} segments, {len(words)} words (Language: {language}).", flush=True)
        return segments, words, language

    def analyze(
        self,
        audio_path: Optional[Path] = None,
        total_duration: float = 0.0,
        language: Optional[str] = None,
        precomputed_transcript_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Transcribes and analyzes hostility, speech acceleration, and conflict lexicon.
        If precomputed_transcript_path is provided, skips Whisper model execution completely!
        Returns:
        - "scores": List[float] per-second hostility scores (0-100)
        - "segments": List[Dict] with timestamps and text
        - "words": List[Dict] word-level timestamps
        - "wps": List[float] words-per-second per second
        - "language": detected language code
        """
        if precomputed_transcript_path is not None and Path(precomputed_transcript_path).is_file():
            segments, words, detected_lang = self.load_transcript_file(Path(precomputed_transcript_path))
        else:
            if audio_path is None:
                raise ValueError("Either audio_path or precomputed_transcript_path must be provided.")
            segments, words, detected_lang = self.transcribe_audio(audio_path, language=language)

        hostility_data = self.analyze_hostility(words, segments, total_duration)
        return {
            "scores": hostility_data["scores"],
            "wps": hostility_data["wps"],
            "segments": segments,
            "words": words,
            "language": detected_lang,
        }

    def analyze_hostility(
        self,
        words: List[Dict[str, Any]],
        segments: List[Dict[str, Any]],
        total_duration: float,
    ) -> Dict[str, Any]:
        """
        Calculates per-second Sentiment & Hostility score (0-100) based on:
        - Words-per-second (WPS) acceleration spikes.
        - High-arousal parliamentary hostility lexicon matches.
        - Punctuation & exclamation mark density.
        """
        total_seconds = int(np.ceil(total_duration))
        if total_seconds <= 0:
            return {"scores": [], "wps": []}

        sec_word_counts = np.zeros(total_seconds, dtype=np.float32)
        sec_lexicon_hits = np.zeros(total_seconds, dtype=np.float32)
        sec_exclamations = np.zeros(total_seconds, dtype=np.float32)

        # 1. Map words to seconds for Words-Per-Second (WPS)
        for w in words:
            st = max(0.0, float(w.get("start", 0.0)))
            s_idx = int(np.floor(st))
            if 0 <= s_idx < total_seconds:
                sec_word_counts[s_idx] += 1.0

        # Rolling 3-second window for smoothed Words Per Second
        wps = np.zeros(total_seconds, dtype=np.float32)
        for s in range(total_seconds):
            window_slice = sec_word_counts[max(0, s - 1) : min(total_seconds, s + 2)]
            wps[s] = np.sum(window_slice) / float(len(window_slice) if len(window_slice) > 0 else 1)

        # 2. Check for parliamentary hostility lexicon and exclamations in segments
        all_terms = []
        for lang_terms in CONFLICT_LEXICON.values():
            all_terms.extend(lang_terms)

        for seg in segments:
            text = seg.get("text", "")
            seg_st = int(np.floor(seg.get("start", 0.0)))
            seg_en = int(np.ceil(seg.get("end", 0.0)))

            # Exclamation marks (!, ؟, ?!)
            excl_count = len(re.findall(r"[!؟?]{1,}", text))

            # Conflict terms
            hits = 0
            for term in all_terms:
                if term in text:
                    hits += 1

            for s in range(max(0, seg_st), min(total_seconds, seg_en)):
                sec_lexicon_hits[s] += hits
                sec_exclamations[s] += excl_count

        # 3. Compute component Z-scores
        def compute_zscore(arr: np.ndarray) -> np.ndarray:
            mean = np.mean(arr)
            std = np.std(arr)
            if std < 1e-6:
                return np.zeros_like(arr)
            return (arr - mean) / std

        wps_z = compute_zscore(wps)
        lex_z = compute_zscore(sec_lexicon_hits)
        excl_z = compute_zscore(sec_exclamations)

        # 4. Fusion of speech rate acceleration + verbal hostility
        # Normal speech: wps_z ~ 0, lex_z = 0.
        # Heated speech: rapid speech rate + frequent objections/interruptions.
        combined_hostility_z = (0.50 * wps_z) + (0.35 * lex_z) + (0.15 * excl_z)

        # Sigmoid calibration to [0, 100]
        scores = 100.0 / (1.0 + np.exp(-1.3 * (combined_hostility_z - 0.5)))
        scores = np.clip(scores, 0.0, 100.0)

        return {
            "scores": [round(float(s), 2) for s in scores],
            "wps": [round(float(w), 2) for w in wps],
        }
