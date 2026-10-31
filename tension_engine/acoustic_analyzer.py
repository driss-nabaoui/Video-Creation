"""
Acoustic Intensity & Vocal Strain Analyzer (35% Weight).
Extracts RMS volume, Spectral Centroid (pitch strain), and Spectral Flux.
Normalizes features across the video using dynamic Z-scores over 1-second windows.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import librosa


class AcousticAnalyzer:
    """
    Extracts acoustic energy and vocal strain signals:
    - RMS Energy: Volume amplitude spikes.
    - Spectral Centroid: Vocal cord tension / high-frequency formant shift when shouting.
    - Spectral Flux: Sudden transient energy bursts (gavel bangs, table strikes, abrupt shouting).
    """

    def __init__(self, target_sr: int = 16000, hop_length: int = 512, frame_length: int = 2048):
        self.target_sr = target_sr
        self.hop_length = hop_length
        self.frame_length = frame_length

    def analyze_audio(
        self,
        audio_path: Path,
        total_duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Processes audio file and generates per-second acoustic tension scores (0-100).
        Returns a dict containing:
        - "scores": List[float] of tension scores per second [0, 1, ..., T-1]
        - "rms_z": List[float] Z-scores for RMS energy
        - "centroid_z": List[float] Z-scores for Spectral Centroid
        - "flux_z": List[float] Z-scores for Spectral Flux
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"[*] Extracting acoustic features with Librosa (sr={self.target_sr})...", flush=True)
        y, sr = librosa.load(str(audio_path), sr=self.target_sr, mono=True)
        duration_sec = total_duration or (len(y) / sr)
        total_seconds = int(np.ceil(duration_sec))

        if len(y) == 0:
            return {
                "scores": [0.0] * total_seconds,
                "rms_z": [0.0] * total_seconds,
                "centroid_z": [0.0] * total_seconds,
                "flux_z": [0.0] * total_seconds,
            }

        # 1. Compute frame-level features
        rms_frames = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        centroid_frames = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.frame_length, hop_length=self.hop_length
        )[0]
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)

        # Time corresponding to each STFT frame
        frame_times = librosa.frames_to_time(np.arange(len(rms_frames)), sr=sr, hop_length=self.hop_length)

        # 2. Aggregate into 1-second rolling windows [t, t+1)
        sec_rms = np.zeros(total_seconds, dtype=np.float32)
        sec_centroid = np.zeros(total_seconds, dtype=np.float32)
        sec_flux = np.zeros(total_seconds, dtype=np.float32)

        for sec in range(total_seconds):
            mask = (frame_times >= sec) & (frame_times < (sec + 1.0))
            if np.any(mask):
                # 90th percentile of RMS in the second captures vocal peaks over background noise
                sec_rms[sec] = np.percentile(rms_frames[mask], 85)
                sec_centroid[sec] = np.mean(centroid_frames[mask])
                # Max onset/flux captures sharp strikes or shouting bursts
                sec_flux[sec] = np.max(onset_env[mask[: len(onset_env)]]) if len(onset_env) > 0 else 0.0
            else:
                # Boundary fallback
                sec_rms[sec] = sec_rms[max(0, sec - 1)]
                sec_centroid[sec] = sec_centroid[max(0, sec - 1)]
                sec_flux[sec] = sec_flux[max(0, sec - 1)]

        # 3. Z-score normalization across the entire video
        def compute_zscore(arr: np.ndarray) -> np.ndarray:
            mean = np.mean(arr)
            std = np.std(arr)
            if std < 1e-6:
                return np.zeros_like(arr)
            return (arr - mean) / std

        rms_z = compute_zscore(sec_rms)
        centroid_z = compute_zscore(sec_centroid)
        flux_z = compute_zscore(sec_flux)

        # 4. Multi-feature acoustic fusion
        # Shouting voices: high RMS + high centroid (higher pitch/formant resonance).
        # Abrupt clashes/gavel: high flux + high RMS.
        combined_z = (0.45 * rms_z) + (0.35 * centroid_z) + (0.20 * flux_z)

        # 5. Sigmoid calibration to [0, 100]
        # Center at z=0.6, scale=1.3:
        # z = 0.0 (calm average) -> ~32
        # z = 1.0 (elevated voice) -> ~63
        # z = 2.0 (shouting argument) -> ~86
        # z >= 3.0 (peak uproar / gavel banging) -> 95-100
        sigmoid_val = 1.0 / (1.0 + np.exp(-1.3 * (combined_z - 0.6)))
        scores = np.clip(sigmoid_val * 100.0, 0.0, 100.0)

        return {
            "scores": [round(float(s), 2) for s in scores],
            "rms_z": [round(float(z), 2) for z in rms_z],
            "centroid_z": [round(float(z), 2) for z in centroid_z],
            "flux_z": [round(float(z), 2) for z in flux_z],
            "duration": duration_sec,
        }
