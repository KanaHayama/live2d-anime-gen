"""
Live2D Anime Generation Package

A Python package for converting facial landmarks to Live2D model parameters,
enabling video-to-animation conversion and real-time motion capture.
"""

__version__ = "0.1.0"

from .core.types import Live2DParameters
from .detectors.insightface_detector import InsightFaceDetector
from .mappers.face_mapper import FaceMapper
from .processors.smoother import ParameterSmoother
from .processors.video_io import VideoReader, VideoWriter
from .processors.pipeline import Pipeline, DataCollector
from .processors.data_io import DataExporter, DataLoader, detect_input_type, InputType
from .renderers.live2d_renderer import Live2DRenderer

__all__ = [
    "Live2DParameters",
    "InsightFaceDetector",
    "FaceMapper",
    "ParameterSmoother",
    "VideoReader",
    "VideoWriter",
    "Pipeline",
    "DataCollector",
    "InputType",
    "DataExporter",
    "DataLoader",
    "detect_input_type",
    "Live2DRenderer",
]