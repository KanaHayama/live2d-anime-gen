"""Video I/O utilities for reading and writing video files."""

from typing import Iterator, Tuple, Optional, Union, List
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm


class VideoReader:
    """
    Read video frames from a file.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def read_frames(self, show_progress: bool = True) -> Iterator[np.ndarray]:
        """
        Iterate over video frames.
        
        Args:
            show_progress: Show progress bar
            
        Yields:
            Video frames as numpy arrays (BGR format)
        """
        progress_bar = tqdm(total=self.frame_count, desc="Reading frames") if show_progress else None
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                yield frame
                
                if progress_bar:
                    progress_bar.update(1)
        finally:
            if progress_bar:
                progress_bar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()


class VideoWriter:
    """
    Write frames to a video file.
    """
    
    def __init__(self, 
                 output_path: str,
                 fps: float,
                 frame_size: Tuple[int, int],
                 codec: str = 'mp4v'):
        """
        Initialize video writer.
        
        Args:
            output_path: Path to output video file
            fps: Frames per second
            frame_size: Frame size (width, height)
            codec: Video codec (default: mp4v)
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            frame_size
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output_path}")
        
        self.frame_count = 0
    
    def write(self, input_data: Union[torch.Tensor, Iterator[torch.Tensor], List[torch.Tensor]], 
             show_progress: bool = True) -> Optional[Iterator[torch.Tensor]]:
        """
        Unified write interface supporting single frame, batch, and streaming modes.
        
        Args:
            input_data: Input frames - single array, list, or iterator
            show_progress: Show progress bar for batch/streaming modes
            
        Returns:
            - Single frame: None
            - Iterator input: Iterator (passthrough for chaining)
            - List input: None
        """
        from .stream_utils import is_iterator
        
        # Handle single frame
        if isinstance(input_data, torch.Tensor):
            self.write_frame(input_data)
            return None
        
        # Handle iterator/generator (streaming mode)
        elif is_iterator(input_data):
            return self._write_stream(input_data, show_progress)
        
        # Handle list (batch mode)
        else:
            self._write_batch(input_data, show_progress)
            return None
    
    def write_frame(self, frame: torch.Tensor):
        """
        Write a single frame.
        
        Args:
            frame: Frame to write (BGR format)
        """
        # Convert torch tensor to numpy for OpenCV
        frame_np = frame.cpu().numpy()
        
        if frame_np.dtype != np.uint8:
            frame_np = (frame_np * 255).astype(np.uint8)
        
        self.writer.write(frame_np)
        self.frame_count += 1
    
    def _write_stream(self, frames: Iterator[torch.Tensor], show_progress: bool) -> Iterator[torch.Tensor]:
        """
        Write frames from iterator with passthrough.
        
        Args:
            frames: Iterator of video frames
            show_progress: Show progress bar
            
        Yields:
            Same frames (passthrough for chaining)
        """
        progress_bar = tqdm(desc="Writing frames", unit="frames") if show_progress else None
        
        try:
            for frame in frames:
                self.write_frame(frame)
                
                if progress_bar is not None:
                    progress_bar.update(1)
                
                yield frame
        finally:
            if progress_bar is not None:
                progress_bar.close()
    
    def _write_batch(self, frames: List[torch.Tensor], show_progress: bool):
        """
        Write frames from list.
        
        Args:
            frames: List of video frames
            show_progress: Show progress bar
        """
        progress_bar = tqdm(total=len(frames), desc="Writing frames") if show_progress else None
        
        try:
            for frame in frames:
                self.write_frame(frame)
                
                if progress_bar:
                    progress_bar.update(1)
        finally:
            if progress_bar:
                progress_bar.close()
    
    # Legacy method for backward compatibility
    def write_frames(self, 
                    frames: Iterator[torch.Tensor],
                    total: Optional[int] = None,
                    show_progress: bool = True):
        """
        Legacy method: Write multiple frames.
        
        Args:
            frames: Iterator of frames
            total: Total number of frames (for progress bar)
            show_progress: Show progress bar
        """
        list(self._write_stream(frames, show_progress))
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Release video writer."""
        if self.writer:
            self.writer.release()