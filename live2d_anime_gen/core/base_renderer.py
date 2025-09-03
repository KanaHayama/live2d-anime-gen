"""Base renderer interface for Live2D model rendering."""

from abc import ABC, abstractmethod
from typing import Union, Iterator, List, Tuple
import torch

from .types import Live2DParameters


class BaseRenderer(ABC):
    """
    Abstract base class for Live2D model renderers.
    
    All renderers must implement the core rendering interface and return
    tensors in the standardized format: (H, W, 3) RGB uint8 on CUDA device.
    """
    
    def __init__(self):
        """Initialize base renderer."""
        pass
    
    @abstractmethod
    def render(self, input_data: Union[Live2DParameters, Iterator[Live2DParameters], List[Live2DParameters]]) -> Union[torch.Tensor, Iterator[torch.Tensor], List[torch.Tensor]]:
        """
        Render Live2D model with given parameters.
        
        Args:
            input_data: Input parameters - single object, list, or iterator of Live2DParameters
            
        Returns:
            Rendered image(s) as torch tensor(s) in RGB format (H, W, 3) uint8 on CUDA device
            - Single input: torch.Tensor
            - Multiple inputs: Iterator or List of torch.Tensor
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close renderer and clean up resources."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.close()