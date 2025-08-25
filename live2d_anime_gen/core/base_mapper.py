"""Base class for landmark to Live2D parameter mapping."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch

from .types import Live2DParameters


class BaseLandmarkMapper(ABC):
    """Abstract base class for mapping landmarks to Live2D parameters."""
    
    def __init__(self, smooth_factor: float = 0.5):
        """
        Initialize the mapper.
        
        Args:
            smooth_factor: Smoothing factor for temporal filtering (0-1)
                          0 = no smoothing, 1 = maximum smoothing
        """
        self.smooth_factor = smooth_factor
        self.prev_params: Optional[Live2DParameters] = None
    
    @abstractmethod
    def map(self, 
            landmarks: torch.Tensor,
            image_shape: Tuple[int, int]) -> Live2DParameters:
        """
        Map facial landmarks to Live2D parameters.
        
        Args:
            landmarks: Landmark coordinates as torch tensor (N, 2) or (N, 3)
            image_shape: Image dimensions as (height, width)
            
        Returns:
            Live2D parameters
        """
        pass
    
    def preprocess_landmarks(self, 
                           landmarks: torch.Tensor,
                           image_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Preprocess landmarks before mapping.
        Converts [0, 1] normalized coordinates to [-1, 1] range for mapping algorithms.
        
        Args:
            landmarks: Landmark coordinates in [0, 1] range
            image_shape: Image dimensions (height, width) - not used but kept for compatibility
            
        Returns:
            Landmarks in [-1, 1] range for mapping algorithms
        """
        normalized = landmarks.clone()
        
        # Convert [0, 1] to [-1, 1] range
        normalized[:, 0] = landmarks[:, 0] * 2 - 1  # x coordinates
        normalized[:, 1] = landmarks[:, 1] * 2 - 1  # y coordinates
        
        # Keep z coordinate as-is if present
        
        return normalized
    
    def postprocess_parameters(self, params: Live2DParameters) -> Live2DParameters:
        """
        Postprocess Live2D parameters after mapping.
        Applies clamping and smoothing.
        
        Args:
            params: Raw parameters from mapping
            
        Returns:
            Processed parameters
        """
        # Apply value clamping
        params = self._clamp_parameters(params)
        
        # Apply temporal smoothing
        if self.prev_params is not None and self.smooth_factor > 0:
            params = self._smooth_parameters(params, self.prev_params)
        
        # Store for next frame
        self.prev_params = params.clone()
        
        return params
    
    def _clamp_parameters(self, params: Live2DParameters) -> Live2DParameters:
        """Clamp parameter values to valid ranges."""
        # Clamp eye openness to [0, 1]
        params.eye_l_open = torch.clamp(params.eye_l_open, 0, 1)
        params.eye_r_open = torch.clamp(params.eye_r_open, 0, 1)
        
        # Clamp mouth parameters to Live2D model ranges
        params.mouth_open_y = torch.clamp(params.mouth_open_y, 0, 2)  # Testing 0-2 range
        params.mouth_form = torch.clamp(params.mouth_form, 0, 2)  # Updated range for Live2D compatibility
        
        # Clamp head rotation to [-30, 30] degrees
        params.angle_x = torch.clamp(params.angle_x, -30, 30)
        params.angle_y = torch.clamp(params.angle_y, -30, 30)
        params.angle_z = torch.clamp(params.angle_z, -30, 30)
        
        # Clamp eye ball position to [-1, 1]
        params.eye_ball_x = torch.clamp(params.eye_ball_x, -1, 1)
        params.eye_ball_y = torch.clamp(params.eye_ball_y, -1, 1)
        
        # Clamp optional parameters if present
        if params.brow_l_y is not None:
            params.brow_l_y = torch.clamp(params.brow_l_y, -1, 1)
        if params.brow_r_y is not None:
            params.brow_r_y = torch.clamp(params.brow_r_y, -1, 1)
        
        if params.body_angle_x is not None:
            params.body_angle_x = torch.clamp(params.body_angle_x, -10, 10)
        if params.body_angle_y is not None:
            params.body_angle_y = torch.clamp(params.body_angle_y, -10, 10)
        if params.body_angle_z is not None:
            params.body_angle_z = torch.clamp(params.body_angle_z, -10, 10)
        
        return params
    
    def _smooth_parameters(self, 
                         current: Live2DParameters,
                         previous: Live2DParameters) -> Live2DParameters:
        """
        Apply exponential moving average smoothing.
        
        Args:
            current: Current frame parameters
            previous: Previous frame parameters
            
        Returns:
            Smoothed parameters
        """
        alpha = self.smooth_factor
        
        # Use parameter-specific smoothing factors for better results
        from .constants import SMOOTHING_FACTORS
        
        # Eye parameters need minimal smoothing for natural blinking
        eye_alpha = SMOOTHING_FACTORS.get("eye", alpha)
        current.eye_l_open = eye_alpha * previous.eye_l_open + (1 - eye_alpha) * current.eye_l_open
        current.eye_r_open = eye_alpha * previous.eye_r_open + (1 - eye_alpha) * current.eye_r_open
        
        # Mouth parameters need moderate smoothing
        mouth_alpha = SMOOTHING_FACTORS.get("mouth", alpha) 
        current.mouth_open_y = mouth_alpha * previous.mouth_open_y + (1 - mouth_alpha) * current.mouth_open_y
        current.mouth_form = mouth_alpha * previous.mouth_form + (1 - mouth_alpha) * current.mouth_form
        
        # Head rotation parameters use default smoothing
        current.angle_x = alpha * previous.angle_x + (1 - alpha) * current.angle_x
        current.angle_y = alpha * previous.angle_y + (1 - alpha) * current.angle_y
        current.angle_z = alpha * previous.angle_z + (1 - alpha) * current.angle_z
        
        # Eye ball movement needs more smoothing to reduce jitter
        eye_ball_alpha = SMOOTHING_FACTORS.get("eye_ball", alpha)
        current.eye_ball_x = eye_ball_alpha * previous.eye_ball_x + (1 - eye_ball_alpha) * current.eye_ball_x
        current.eye_ball_y = eye_ball_alpha * previous.eye_ball_y + (1 - eye_ball_alpha) * current.eye_ball_y
        
        # Smooth optional parameters if present in both
        if current.brow_l_y is not None and previous.brow_l_y is not None:
            current.brow_l_y = alpha * previous.brow_l_y + (1 - alpha) * current.brow_l_y
        if current.brow_r_y is not None and previous.brow_r_y is not None:
            current.brow_r_y = alpha * previous.brow_r_y + (1 - alpha) * current.brow_r_y
            
        if current.body_angle_x is not None and previous.body_angle_x is not None:
            current.body_angle_x = alpha * previous.body_angle_x + (1 - alpha) * current.body_angle_x
        if current.body_angle_y is not None and previous.body_angle_y is not None:
            current.body_angle_y = alpha * previous.body_angle_y + (1 - alpha) * current.body_angle_y
        if current.body_angle_z is not None and previous.body_angle_z is not None:
            current.body_angle_z = alpha * previous.body_angle_z + (1 - alpha) * current.body_angle_z
        
        return current
    
    def reset(self):
        """Reset the mapper state (clear previous parameters)."""
        self.prev_params = None