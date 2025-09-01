"""Processing pipeline components."""

from typing import List, Optional, Iterator, Union, Any
import torch
import numpy as np
import cv2
from dataclasses import dataclass

from ..core.base_detector import BaseDetector
from ..core.base_mapper import BaseLandmarkMapper
from ..core.types import Live2DParameters
from ..renderers.live2d_renderer import Live2DRenderer
from .smoother import ParameterSmoother
from .stream_utils import (
    ensure_iterator,
    collect_stream,
    enumerate_stream
)


@dataclass
class FrameData:
    """Data for a single frame in the pipeline."""
    frame_idx: int
    image: Optional[np.ndarray] = None
    landmarks: Optional[torch.Tensor] = None
    parameters: Optional[Live2DParameters] = None
    rendered: Optional[torch.Tensor] = None


class Pipeline:
    """
    Unified processing pipeline supporting both batch and streaming modes.
    
    The pipeline automatically adapts to input type:
    - Iterator/Generator input → streaming mode (frame-by-frame processing)
    - List input → can be processed as batch or stream
    - Single item input → single processing
    """
    
    def __init__(self, 
                 detector: Optional[BaseDetector] = None,
                 mapper: Optional[BaseLandmarkMapper] = None,
                 smoother: Optional[ParameterSmoother] = None,
                 renderer: Optional[Live2DRenderer] = None):
        """
        Initialize pipeline with optional components.
        
        Args:
            detector: Face detector for landmark extraction
            mapper: Mapper from landmarks to Live2D parameters
            smoother: Parameter smoother for temporal consistency
            renderer: Live2D renderer for final output
        """
        self.detector = detector
        self.mapper = mapper
        self.smoother = smoother
        self.renderer = renderer
    
    def process(self, 
               input_data: Union[np.ndarray, Iterator[np.ndarray], List[np.ndarray]],
               output_sink: Optional[Any] = None,
               buffer_size: Optional[int] = 1) -> Union[Iterator[Any], List[Any], Any]:
        """
        Unified processing interface supporting both batch and streaming modes.
        
        Args:
            input_data: Input frames (single, list, or iterator)
            output_sink: Optional output handler (e.g., VideoWriter)
            buffer_size: Processing mode control:
                        - 1: Pure streaming (frame-by-frame)
                        - >1: Batched streaming
                        - None: Full batch processing
                        
        Returns:
            - If single input: single result
            - If batch mode (buffer_size=None): complete results list
            - If streaming: iterator of results
        """
        # Handle single frame input
        if isinstance(input_data, np.ndarray):
            return self._process_single_frame(input_data)
        
        # Convert to stream for unified processing
        stream = ensure_iterator(input_data)
        
        # Process through pipeline stages
        processed_stream = self._process_stream(stream)
        
        # Handle output
        if output_sink is not None:
            # Stream to output sink
            return self._stream_to_sink(processed_stream, output_sink)
        else:
            # Return based on buffer_size
            return collect_stream(processed_stream, buffer_size)
    
    def _process_single_frame(self, frame: np.ndarray) -> Any:
        """
        Process a single frame through all pipeline stages.
        
        Args:
            frame: Input frame
            
        Returns:
            Final processed result
        """
        result = FrameData(frame_idx=0, image=frame)
        
        # Detection stage
        if self.detector:
            result.landmarks = self.detector.detect(frame)
        
        # Mapping stage
        if self.mapper and result.landmarks is not None:
            height, width = frame.shape[:2]
            result.parameters = self.mapper.map(result.landmarks, (height, width))
        
        # Smoothing stage
        if self.smoother and result.parameters is not None:
            result.parameters = self.smoother.smooth(result.parameters)
        
        # Rendering stage
        if self.renderer and result.parameters is not None:
            result.rendered = self.renderer.render(result.parameters)
        
        return result
    
    def _process_stream(self, frames: Iterator[np.ndarray]) -> Iterator[FrameData]:
        """
        Process frame stream through all pipeline stages.
        
        Args:
            frames: Iterator of input frames
            
        Yields:
            FrameData objects with processed results
        """
        for frame_idx, frame in enumerate_stream(frames):
            result = FrameData(frame_idx=frame_idx, image=frame)
            
            # Detection stage
            if self.detector:
                result.landmarks = self.detector.detect(frame)
            
            # Mapping stage
            if self.mapper and result.landmarks is not None:
                height, width = frame.shape[:2]
                result.parameters = self.mapper.map(result.landmarks, (height, width))
            
            # Smoothing stage
            if self.smoother and result.parameters is not None:
                result.parameters = self.smoother.smooth(result.parameters)
            
            # Rendering stage
            if self.renderer and result.parameters is not None:
                result.rendered = self.renderer.render(result.parameters)
            
            yield result
    
    def _stream_to_sink(self, stream: Iterator[FrameData], output_sink: Any) -> Iterator[FrameData]:
        """
        Stream processed data to output sink while passing through.
        
        Args:
            stream: Processed frame stream
            output_sink: Output handler (e.g., VideoWriter)
            
        Yields:
            FrameData objects (passthrough)
        """
        for frame_data in stream:
            # Write to sink if applicable
            if hasattr(output_sink, 'write') and frame_data.rendered is not None:
                output_sink.write(frame_data.rendered)
            elif hasattr(output_sink, 'write_frame') and frame_data.rendered is not None:
                output_sink.write_frame(frame_data.rendered)
            
            yield frame_data


class DataCollector:
    """
    Streaming-compatible data collector for pipeline results.
    
    Supports both streaming collection (passthrough) and batch collection.
    """
    
    def __init__(self):
        """Initialize data collector."""
        self.frames: List[np.ndarray] = []
        self.landmarks: List[Optional[torch.Tensor]] = []
        self.parameters: List[Optional[Live2DParameters]] = []
        self.rendered: List[Optional[torch.Tensor]] = []
    
    def collect(self, 
               data_stream: Iterator[FrameData],
               collect_frames: bool = False,
               collect_landmarks: bool = False,
               collect_parameters: bool = False,
               collect_rendered: bool = False) -> Iterator[FrameData]:
        """
        Collect data from pipeline stream with passthrough.
        
        Args:
            data_stream: Pipeline data stream (FrameData objects)
            collect_frames: Collect original frames
            collect_landmarks: Collect landmarks
            collect_parameters: Collect Live2D parameters
            collect_rendered: Collect rendered frames
            
        Yields:
            Same data stream (passthrough)
        """
        for frame_data in data_stream:
            # Collect requested data
            if collect_frames and frame_data.image is not None:
                self.frames.append(frame_data.image.copy())
            if collect_landmarks and frame_data.landmarks is not None:
                self.landmarks.append(frame_data.landmarks.clone())
            if collect_parameters:
                self.parameters.append(frame_data.parameters)
            if collect_rendered and frame_data.rendered is not None:
                self.rendered.append(frame_data.rendered.clone())
            
            yield frame_data
    
    def clear(self):
        """Clear collected data."""
        self.frames.clear()
        self.landmarks.clear()
        self.parameters.clear()
        self.rendered.clear()