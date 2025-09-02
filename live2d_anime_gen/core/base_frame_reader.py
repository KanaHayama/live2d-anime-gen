"""Base frame reader interface for video input sources."""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Any
import torch


class BaseFrameReader(ABC):
    """
    Abstract base class for frame readers (video files, cameras, etc.).
    
    All frame readers must implement the standardized interface and return
    frames in the format: (H, W, 3) RGB uint8 on CUDA device.
    """
    
    def __init__(self) -> None:
        """Initialize base frame reader."""
        self.fps: float = 30.0
        self.width: int = 0
        self.height: int = 0
        self.frame_count: Optional[int] = None
    
    @abstractmethod
    def read_frames(self) -> Iterator[torch.Tensor]:
        """
        Read frames from the input source.
        
        Yields:
            Video frames as torch tensors (RGB format, H x W x 3, uint8, CUDA)
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close reader and clean up resources."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        """Context manager exit with automatic cleanup."""
        self.close()