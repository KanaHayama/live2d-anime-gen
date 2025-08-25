"""Video I/O utilities for reading and writing video files."""

from typing import Iterator, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np
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
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a single frame.
        
        Args:
            frame: Frame to write (BGR format)
        """
        self.writer.write(frame)
        self.frame_count += 1
    
    def write_frames(self, 
                    frames: Iterator[np.ndarray],
                    total: Optional[int] = None,
                    show_progress: bool = True):
        """
        Write multiple frames.
        
        Args:
            frames: Iterator of frames
            total: Total number of frames (for progress bar)
            show_progress: Show progress bar
        """
        progress_bar = tqdm(total=total, desc="Writing frames") if show_progress and total else None
        
        try:
            for frame in frames:
                self.write_frame(frame)
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
        """Release video writer."""
        if self.writer:
            self.writer.release()