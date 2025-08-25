"""Base class for landmark detection systems."""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union
import numpy as np
import torch


class BaseDetector(ABC):
    """Abstract base class for facial/body landmark detectors."""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """
        Detect landmarks in the given image.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR or RGB format
            
        Returns:
            Landmarks as torch tensor (N, 2) or (N, 3) on GPU, or None if no detection
        """
        pass
    
    @abstractmethod
    def get_num_landmarks(self) -> int:
        """Return the number of landmarks this detector provides."""
        pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image before detection.
        Default implementation returns image as-is.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        return image
    
    def postprocess_landmarks(self, landmarks: Union[np.ndarray, torch.Tensor], image_shape: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """
        Postprocess detected landmarks.
        Default implementation converts to torch tensor on GPU and normalizes to [0,1] range.
        
        Args:
            landmarks: Raw landmarks from detector (in pixel coordinates)
            image_shape: Image shape (H, W, C) for normalization, optional for backward compatibility
            
        Returns:
            Processed landmarks as torch tensor on GPU, normalized to [0,1] range if image_shape provided
        """
        if isinstance(landmarks, np.ndarray):
            landmarks = torch.from_numpy(landmarks)
        
        if torch.cuda.is_available():
            landmarks = landmarks.cuda()
        
        landmarks = landmarks.float()
        
        # Normalize to [0,1] range if image shape is provided
        if image_shape is not None:
            height, width = image_shape[:2]
            landmarks = landmarks.clone()  # Don't modify original
            landmarks[:, 0] = landmarks[:, 0] / width   # x coordinates
            landmarks[:, 1] = landmarks[:, 1] / height  # y coordinates
        
        return landmarks