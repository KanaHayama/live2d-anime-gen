"""Base class for landmark detection systems."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch


class BaseDetector(ABC):
    """Abstract base class for facial/body landmark detectors."""
    
    @abstractmethod
    def detect(self, image: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Detect landmarks in the given image.
        
        Args:
            image: Input image as torch tensor (H, W, C) in RGB format on CUDA
            
        Returns:
            Landmarks as torch tensor (N, 2) or (N, 3) on GPU, or None if no detection
        """
        pass
    
    
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image before detection.
        Default implementation returns image as-is.
        
        Args:
            image: Input image tensor on CUDA
            
        Returns:
            Preprocessed image tensor on CUDA
        """
        return image
    
    def postprocess_landmarks(self, landmarks: torch.Tensor, image_shape: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """
        Postprocess detected landmarks.
        Default implementation ensures tensor is on GPU and normalizes to [0,1] range.
        
        Args:
            landmarks: Raw landmarks tensor (in pixel coordinates), assumed on CUDA
            image_shape: Image shape (H, W, C) for normalization
            
        Returns:
            Processed landmarks as torch tensor on GPU, normalized to [0,1] range if image_shape provided
        """
        # Assume tensor is already on CUDA
        landmarks = landmarks.float()
        
        # Normalize to [0,1] range if image shape is provided
        if image_shape is not None:
            height, width = image_shape[:2]
            landmarks = landmarks.clone()  # Don't modify original
            landmarks[:, 0] = landmarks[:, 0] / width   # x coordinates
            landmarks[:, 1] = landmarks[:, 1] / height  # y coordinates
        
        return landmarks