"""
Live2D Anime Generation Package

A Python package for converting facial landmarks to Live2D model parameters,
enabling video-to-animation conversion and real-time motion capture.
"""

__version__ = "1.0.0"

from .core.input_type import InputType
from .core.types import Live2DParameters
from .detectors.mediapipe_detector import MediaPipeDetector
from .mappers.face_mapper import FaceMapper
from .processors.data_loader import DataLoader
from .processors.input_utils import detect_input_type
from .processors.smoother import ParameterSmoother
from .processors.data_exporter import DataExporter
from .processors.video_reader import VideoReader
from .processors.video_writer import VideoWriter
from .renderers.live2d_renderer import Live2DRenderer

__all__ = [
    "DataExporter",
    "DataLoader",
    "FaceMapper",
    "InputType",
    "MediaPipeDetector",
    "Live2DParameters",
    "Live2DRenderer",
    "ParameterSmoother",
    "VideoReader",
    "VideoWriter",
    "detect_input_type",
]