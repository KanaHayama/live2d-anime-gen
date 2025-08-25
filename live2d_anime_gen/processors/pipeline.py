"""Processing pipeline components."""

from typing import List, Optional, Iterator, Tuple
import torch
import numpy as np
import cv2
from dataclasses import dataclass

from ..core.base_detector import BaseDetector
from ..core.base_mapper import BaseLandmarkMapper
from ..core.types import Live2DParameters
from ..renderers.live2d_renderer import Live2DRenderer
from .smoother import ParameterSmoother


@dataclass
class FrameData:
    """Data for a single frame in the pipeline."""
    frame_idx: int
    image: Optional[np.ndarray] = None
    landmarks: Optional[torch.Tensor] = None
    parameters: Optional[Live2DParameters] = None
    rendered: Optional[np.ndarray] = None


class Pipeline:
    """
    Modular processing pipeline for video to Live2D conversion.
    """
    
    def __init__(self):
        """Initialize empty pipeline."""
        self.detector: Optional[BaseDetector] = None
        self.mapper: Optional[BaseLandmarkMapper] = None
        self.smoother: Optional[ParameterSmoother] = None
        self.renderer: Optional[Live2DRenderer] = None
    
    def detect_landmarks(self, 
                        frames: Iterator[np.ndarray],
                        detector: BaseDetector) -> Iterator[Tuple[np.ndarray, Optional[torch.Tensor]]]:
        """
        Detect landmarks in frames.
        
        Args:
            frames: Iterator of video frames
            detector: Landmark detector
            
        Yields:
            Tuples of (frame, landmarks)
        """
        for frame in frames:
            landmarks = detector.detect(frame)
            yield frame, landmarks
    
    def map_parameters(self,
                       landmarks_data: Iterator[Tuple[np.ndarray, Optional[torch.Tensor]]],
                       mapper: BaseLandmarkMapper) -> Iterator[Tuple[np.ndarray, Optional[torch.Tensor], Optional[Live2DParameters]]]:
        """
        Map landmarks to Live2D parameters.
        
        Args:
            landmarks_data: Iterator of (frame, landmarks) tuples
            mapper: Landmark to parameter mapper
            
        Yields:
            Tuples of (frame, landmarks, parameters)
        """
        for frame, landmarks in landmarks_data:
            if landmarks is not None:
                height, width = frame.shape[:2]
                parameters = mapper.map(landmarks, (height, width))
            else:
                parameters = None
            yield frame, landmarks, parameters
    
    def smooth_parameters(self,
                         params_data: Iterator[Tuple[np.ndarray, Optional[torch.Tensor], Optional[Live2DParameters]]],
                         smoother: ParameterSmoother) -> Iterator[Tuple[np.ndarray, Optional[torch.Tensor], Optional[Live2DParameters]]]:
        """
        Apply temporal smoothing to parameters.
        
        Args:
            params_data: Iterator of (frame, landmarks, parameters) tuples
            smoother: Parameter smoother
            
        Yields:
            Tuples of (frame, landmarks, smoothed_parameters)
        """
        for frame, landmarks, parameters in params_data:
            if parameters is not None:
                parameters = smoother.smooth(parameters)
            yield frame, landmarks, parameters
    
    def render_live2d(self,
                     params_data: Iterator[Tuple[np.ndarray, Optional[torch.Tensor], Optional[Live2DParameters]]],
                     renderer: Live2DRenderer) -> Iterator[Tuple[np.ndarray, Optional[torch.Tensor], Optional[Live2DParameters], Optional[np.ndarray]]]:
        """
        Render Live2D frames.
        
        Args:
            params_data: Iterator of (frame, landmarks, parameters) tuples
            renderer: Live2D renderer
            
        Yields:
            Tuples of (frame, landmarks, parameters, rendered_frame)
        """
        for frame, landmarks, parameters in params_data:
            if parameters is not None:
                rendered = renderer.render(parameters)
                # Convert RGB to BGR for OpenCV
                rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            else:
                rendered = None
            yield frame, landmarks, parameters, rendered


class DataCollector:
    """
    Collect intermediate data from pipeline.
    """
    
    def __init__(self):
        """Initialize data collector."""
        self.frames: List[np.ndarray] = []
        self.landmarks: List[Optional[torch.Tensor]] = []
        self.parameters: List[Optional[Live2DParameters]] = []
        self.rendered: List[Optional[np.ndarray]] = []
    
    def collect(self, 
               data_stream: Iterator[Tuple[np.ndarray, Optional[torch.Tensor], Optional[Live2DParameters], Optional[np.ndarray]]],
               collect_frames: bool = False,
               collect_landmarks: bool = False,
               collect_parameters: bool = False,
               collect_rendered: bool = False) -> Iterator[Tuple[np.ndarray, Optional[torch.Tensor], Optional[Live2DParameters], Optional[np.ndarray]]]:
        """
        Collect data from pipeline stream.
        
        Args:
            data_stream: Pipeline data stream
            collect_frames: Collect original frames
            collect_landmarks: Collect landmarks
            collect_parameters: Collect Live2D parameters
            collect_rendered: Collect rendered frames
            
        Yields:
            Same data stream (passthrough)
        """
        for frame, landmarks, parameters, rendered in data_stream:
            if collect_frames:
                self.frames.append(frame.copy() if frame is not None else None)
            if collect_landmarks:
                self.landmarks.append(landmarks.clone() if landmarks is not None else None)
            if collect_parameters:
                self.parameters.append(parameters)
            if collect_rendered:
                self.rendered.append(rendered.copy() if rendered is not None else None)
            
            yield frame, landmarks, parameters, rendered
    
    def clear(self):
        """Clear collected data."""
        self.frames.clear()
        self.landmarks.clear()
        self.parameters.clear()
        self.rendered.clear()