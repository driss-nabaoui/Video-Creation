"""
Tension Engine: Multi-Signal High-Tension Scene Detection Pipeline for Parliamentary & Heated Debates.
"""

from .acoustic_analyzer import AcousticAnalyzer
from .overlap_detector import OverlapDetector
from .transcript_analyzer import TranscriptAnalyzer
from .scene_processor import SceneProcessor
from .video_cutter import VideoCutter
from .audio_extractor import extract_audio_from_video

__all__ = [
    "AcousticAnalyzer",
    "OverlapDetector",
    "TranscriptAnalyzer",
    "SceneProcessor",
    "VideoCutter",
    "extract_audio_from_video",
]
