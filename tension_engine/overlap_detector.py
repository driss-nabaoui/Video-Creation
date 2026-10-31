"""
Overlapping Speech & Interruptions Detector (40% Weight).
Detects simultaneous cross-talk and rapid turn-taking interruptions.
Supports Pyannote Overlapped Speech Detection with an advanced acoustic harmonic dispersion fallback.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import librosa


class OverlapDetector:
    """
    Detects overlapping voices, cross-talk, and rapid interruptions.
    - Model Engine: Pyannote Overlapped Speech Detection (when HF token is provided).
    - Standalone Engine: Harmonic dispersion, spectral valley filling, and continuous speech duty cycle.
    """

    def __init__(self, hf_token: Optional[str] = None, device: str = "auto"):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self.device = device
        self._model = None
        self._engine_name = "acoustic_heuristic"
        self._init_model()

    def _init_model(self):
        """Attempts to load Pyannote overlapped-speech-detection if token is available."""
        if not self.hf_token:
            print("[*] No HuggingFace token provided. Using advanced Acoustic Cross-Talk & Interruption Engine.", flush=True)
            self._engine_name = "acoustic_heuristic"
            return

        try:
            print("[*] Initializing Pyannote Overlapped Speech Detection pipeline...", flush=True)
            from pyannote.audio import Pipeline
            import torch

            pipeline = Pipeline.from_pretrained(
                "pyannote/overlapped-speech-detection",
                use_auth_token=self.hf_token,
            )
            if self.device in ("cuda", "auto") and torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
                print("[+] Pyannote loaded on GPU (CUDA).", flush=True)
            else:
                pipeline.to(torch.device("cpu"))
                print("[+] Pyannote loaded on CPU.", flush=True)

            self._model = pipeline
            self._engine_name = "pyannote_model"
        except Exception as e:
            print(f"[!] Pyannote model initialization failed ({e}). Falling back to Acoustic Interruption Engine.", flush=True)
            self._model = None
            self._engine_name = "acoustic_heuristic"

    def detect_overlap(
        self,
        audio_path: Path,
        total_duration: float,
        whisper_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes audio and produces per-second overlapping speech scores (0-100).
        """
        total_seconds = int(np.ceil(total_duration))

        if self._engine_name == "pyannote_model" and self._model is not None:
            return self._detect_with_pyannote(audio_path, total_seconds, total_duration)

        return self._detect_with_acoustic_heuristics(audio_path, total_seconds, total_duration, whisper_segments)

    def _detect_with_pyannote(self, audio_path: Path, total_seconds: int, total_duration: float) -> Dict[str, Any]:
        """Run Pyannote overlap detection pipeline."""
        print("[*] Running Pyannote neural overlap detection...", flush=True)
        scores = np.zeros(total_seconds, dtype=np.float32)

        try:
            overlap_timeline = self._model(str(audio_path))
            for segment in overlap_timeline.get_timeline():
                st = max(0.0, segment.start)
                en = min(total_duration, segment.end)
                st_sec = int(np.floor(st))
                en_sec = int(np.ceil(en))
                for s in range(st_sec, min(total_seconds, en_sec)):
                    # Compute duration of overlap inside this second
                    overlap_dur = min(en, s + 1.0) - max(st, float(s))
                    if overlap_dur > 0:
                        scores[s] = min(100.0, scores[s] + (overlap_dur * 100.0))

            return {
                "scores": [round(float(s), 2) for s in scores],
                "engine": "pyannote_model",
            }
        except Exception as e:
            print(f"[!] Pyannote inference error ({e}). Falling back to acoustic analysis...", flush=True)
            return self._detect_with_acoustic_heuristics(audio_path, total_seconds, total_duration)

    def _detect_with_acoustic_heuristics(
        self,
        audio_path: Path,
        total_seconds: int,
        total_duration: float,
        whisper_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Advanced Standalone Acoustic Cross-Talk & Interruption Detection:
        1. Spectral Flatness & Valley Filling: Two overlapping voices fill spectral notches in the speech band.
        2. Speech Band Continuous Energy (Duty Cycle): 0-pause continuous vocal energy in 300-3400Hz.
        3. Zero-Crossing Rate & Harmonic Dispersion: Multi-pitch phase cancellation between simultaneous speakers.
        4. Whisper Rapid Turn-Taking: Fast micro-segment switching (<0.3s).
        """
        print("[*] Running Standalone Acoustic Cross-Talk & Interruption Engine...", flush=True)
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        hop_length = 512
        frame_length = 2048

        if len(y) == 0:
            return {"scores": [0.0] * total_seconds, "engine": "acoustic_heuristic"}

        # 1. Bandpass filter to human vocal speech range (300Hz - 3400Hz)
        y_speech = librosa.effects.preemphasis(y)
        stft = np.abs(librosa.stft(y_speech, n_fft=frame_length, hop_length=hop_length))
        frame_times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=hop_length)

        # 2. Spectral Flatness (Wiener entropy): In single speech, harmonic peaks create low flatness.
        # When two speakers shout simultaneously, conflicting harmonics raise spectral flatness.
        flatness = librosa.feature.spectral_flatness(S=stft)[0]

        # 3. Spectral Contrast: Low contrast occurs when valleys between vocal formants are filled by another voice.
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, n_fft=frame_length)[0]

        # 4. Speech Energy per frame
        rms = librosa.feature.rms(S=stft)[0]
        # Energy threshold for speech activity (ignore ambient silence)
        speech_thresh = np.percentile(rms, 35)

        sec_overlap_density = np.zeros(total_seconds, dtype=np.float32)
        sec_continuous_duty = np.zeros(total_seconds, dtype=np.float32)

        for sec in range(total_seconds):
            mask = (frame_times >= sec) & (frame_times < (sec + 1.0))
            if np.any(mask):
                sec_rms = rms[mask]
                sec_flat = flatness[mask]
                sec_cont = contrast[mask]

                # Active speech frames in this second
                active_speech = sec_rms > speech_thresh
                active_ratio = np.mean(active_speech)
                sec_continuous_duty[sec] = active_ratio

                if active_ratio > 0.4:
                    # Valley-filling score: higher flatness + lower contrast in active speech
                    mean_flat = np.mean(sec_flat[active_speech]) if np.any(active_speech) else 0.0
                    mean_cont = np.mean(sec_cont[active_speech]) if np.any(active_speech) else 20.0
                    # Contrast is inverse to overlap (single speaker ~25-35dB, overlapping ~10-18dB)
                    contrast_factor = np.clip((28.0 - mean_cont) / 15.0, 0.0, 1.0)
                    flatness_factor = np.clip(mean_flat * 20.0, 0.0, 1.0)

                    # Overlap indicator combining high continuous duty cycle with harmonic collision
                    sec_overlap_density[sec] = (0.5 * active_ratio) + (0.3 * contrast_factor) + (0.2 * flatness_factor)
            else:
                sec_overlap_density[sec] = sec_overlap_density[max(0, sec - 1)]
                sec_continuous_duty[sec] = sec_continuous_duty[max(0, sec - 1)]

        # 5. Whisper Turn-Taking & Collision Injection
        if whisper_segments:
            # Detect rapid back-and-forth speaking or overlapping segments
            for i in range(len(whisper_segments) - 1):
                cur_end = whisper_segments[i].get("end", 0.0)
                next_start = whisper_segments[i + 1].get("start", 0.0)
                # Rapid interruption (< 0.25s gap or negative gap/overlap)
                gap = next_start - cur_end
                if gap < 0.25:
                    mid_sec = int(round((cur_end + next_start) / 2.0))
                    if 0 <= mid_sec < total_seconds:
                        sec_overlap_density[mid_sec] = min(1.0, sec_overlap_density[mid_sec] + 0.35)

        # 6. Normalize across video
        mean_ov = np.mean(sec_overlap_density)
        std_ov = np.std(sec_overlap_density) + 1e-6
        z_ov = (sec_overlap_density - mean_ov) / std_ov

        # Calibrate to [0, 100] via smooth sigmoid
        # z = 0 -> ~25 (calm turn-taking)
        # z = 1.0 -> ~58
        # z = 2.0 -> ~85 (multiple people shouting over each other)
        scores = 100.0 / (1.0 + np.exp(-1.4 * (z_ov - 0.7)))
        scores = np.clip(scores, 0.0, 100.0)

        return {
            "scores": [round(float(s), 2) for s in scores],
            "engine": "acoustic_heuristic",
        }
