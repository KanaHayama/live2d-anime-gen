"""Type definitions for Live2D parameters and data structures."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Union
import torch


@dataclass
class Live2DParameters:
    """
    Container for Live2D model parameters.
    All values are torch tensors for GPU acceleration and batch processing.
    """
    
    # Face parameters (required)
    eye_l_open: torch.Tensor
    eye_r_open: torch.Tensor
    mouth_open_y: torch.Tensor
    mouth_form: torch.Tensor
    angle_x: torch.Tensor  # Yaw (left-right head rotation)
    angle_y: torch.Tensor  # Pitch (up-down head rotation)
    angle_z: torch.Tensor  # Roll (head tilt)
    eye_ball_x: torch.Tensor
    eye_ball_y: torch.Tensor
    
    # Optional face parameters
    brow_l_y: Optional[torch.Tensor] = None
    brow_r_y: Optional[torch.Tensor] = None
    
    # Body parameters (for future extension)
    body_angle_x: Optional[torch.Tensor] = None
    body_angle_y: Optional[torch.Tensor] = None
    body_angle_z: Optional[torch.Tensor] = None
    
    # Additional custom parameters
    custom_params: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    @classmethod
    def create_default(cls, device: str = 'cuda') -> 'Live2DParameters':
        """
        Create Live2D parameters with neutral/default values.
        
        Args:
            device: Device to create tensors on
            
        Returns:
            Live2DParameters with neutral default values
        """
        return cls(
            eye_l_open=torch.tensor(1.0, device=device),      # Eyes fully open
            eye_r_open=torch.tensor(1.0, device=device),      # Eyes fully open
            mouth_open_y=torch.tensor(0.0, device=device),    # Mouth closed
            mouth_form=torch.tensor(0.0, device=device),      # Neutral mouth shape
            angle_x=torch.tensor(0.0, device=device),         # No yaw rotation
            angle_y=torch.tensor(0.0, device=device),         # No pitch rotation
            angle_z=torch.tensor(0.0, device=device),         # No roll rotation
            eye_ball_x=torch.tensor(0.0, device=device),      # Eyes looking forward
            eye_ball_y=torch.tensor(0.0, device=device),      # Eyes looking forward
            brow_l_y=torch.tensor(0.0, device=device),        # Neutral eyebrow
            brow_r_y=torch.tensor(0.0, device=device),        # Neutral eyebrow
        )
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert parameters to dictionary format for Live2D API.
        Tensor values are converted to Python floats.
        """
        result = {}
        
        # Convert standard parameters
        for field_name in self.__dataclass_fields__:
            if field_name == "custom_params":
                continue
            value = getattr(self, field_name)
            if value is not None:
                # Convert tensor to float
                param_name = self._field_to_live2d_name(field_name)
                result[param_name] = value.item() if value.numel() == 1 else value.cpu().numpy().tolist()
        
        # Add custom parameters
        for key, value in self.custom_params.items():
            result[key] = value.item() if value.numel() == 1 else value.cpu().numpy().tolist()
        
        return result
    
    def _field_to_live2d_name(self, field_name: str) -> str:
        """Convert field name to Live2D parameter name format."""
        # Map internal names to Live2D v3 standard parameter names
        mapping = {
            "eye_l_open": "ParamEyeLOpen",
            "eye_r_open": "ParamEyeROpen",
            "mouth_open_y": "ParamMouthOpenY",
            "mouth_form": "ParamMouthForm",
            "angle_x": "ParamAngleX",
            "angle_y": "ParamAngleY",
            "angle_z": "ParamAngleZ",
            "eye_ball_x": "ParamEyeBallX",
            "eye_ball_y": "ParamEyeBallY",
            "brow_l_y": "ParamBrowLY",
            "brow_r_y": "ParamBrowRY",
            "body_angle_x": "ParamBodyAngleX",
            "body_angle_y": "ParamBodyAngleY",
            "body_angle_z": "ParamBodyAngleZ",
        }
        return mapping.get(field_name, field_name)
    
    def clone(self) -> "Live2DParameters":
        """Create a deep copy of the parameters."""
        kwargs = {}
        for field_name in self.__dataclass_fields__:
            if field_name == "custom_params":
                kwargs[field_name] = {k: v.clone() for k, v in self.custom_params.items()}
            else:
                value = getattr(self, field_name)
                kwargs[field_name] = value.clone() if value is not None else None
        return Live2DParameters(**kwargs)
    
    def to(self, device: Union[str, torch.device]) -> "Live2DParameters":
        """Move all tensors to specified device."""
        kwargs = {}
        for field_name in self.__dataclass_fields__:
            if field_name == "custom_params":
                kwargs[field_name] = {k: v.to(device) for k, v in self.custom_params.items()}
            else:
                value = getattr(self, field_name)
                kwargs[field_name] = value.to(device) if value is not None else None
        return Live2DParameters(**kwargs)