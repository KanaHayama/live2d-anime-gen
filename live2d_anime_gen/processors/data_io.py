"""Data I/O for intermediate results."""

import json
import pickle
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import torch
import numpy as np

from ..core.types import Live2DParameters


class DataExporter:
    """Export intermediate data to files."""
    
    @staticmethod
    def export_landmarks(landmarks: List[Optional[torch.Tensor]], 
                        output_path: str,
                        format: str = 'json',
                        video_width: int = None,
                        video_height: int = None):
        """
        Export landmarks sequence to file.
        
        Args:
            landmarks: List of landmark tensors (in original pixel coordinates)
            output_path: Output file path
            format: Export format ('json' or 'pickle')
            video_width: Original video width for coordinate normalization
            video_height: Original video height for coordinate normalization
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            # Convert to JSON-serializable format
            data = {
                'format': 'insightface_106_landmarks_normalized',
                'frame_count': len(landmarks),
                'video_width': video_width,
                'video_height': video_height,
                'landmarks': []
            }
            
            for frame_landmarks in landmarks:
                if frame_landmarks is not None:
                    # Convert tensor to numpy for processing
                    if isinstance(frame_landmarks, torch.Tensor):
                        landmarks_np = frame_landmarks.cpu().numpy()
                    else:
                        landmarks_np = np.array(frame_landmarks)
                    
                    # Landmarks are already normalized to [0, 1] range by detector
                    data['landmarks'].append(landmarks_np.tolist())
                else:
                    data['landmarks'].append(None)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format}. Only JSON format is supported.")
    
    @staticmethod
    def export_parameters(parameters: List[Optional[Live2DParameters]], 
                         output_path: str,
                         format: str = 'json'):
        """
        Export Live2D parameters sequence to file.
        
        Args:
            parameters: List of Live2D parameters
            output_path: Output file path
            format: Export format ('json' or 'pickle')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            # Convert to JSON-serializable format
            data = {
                'format': 'live2d_v3_parameters',
                'frame_count': len(parameters),
                'parameters': []
            }
            
            for frame_params in parameters:
                if frame_params is not None:
                    param_dict = {}
                    # Convert each parameter to scalar
                    for field_name in frame_params.__dataclass_fields__:
                        field_value = getattr(frame_params, field_name)
                        if isinstance(field_value, torch.Tensor):
                            param_dict[field_name] = field_value.cpu().item()
                        else:
                            param_dict[field_name] = field_value
                    data['parameters'].append(param_dict)
                else:
                    data['parameters'].append(None)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'pickle':
            # Save as pickle for faster loading
            with open(output_path, 'wb') as f:
                pickle.dump(parameters, f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")


class DataLoader:
    """Load intermediate data from files."""
    
    @staticmethod
    def load_landmarks(input_path: str, 
                      device: str = 'cuda') -> Tuple[List[Optional[torch.Tensor]], Optional[Tuple[int, int]]]:
        """
        Load landmarks sequence from file.
        
        Args:
            input_path: Input file path
            device: Device to load tensors to
            
        Returns:
            Tuple of (landmarks list, video dimensions as (width, height) or None)
        """
        input_path = Path(input_path)
        
        if input_path.suffix == '.json':
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            landmarks = []
            for frame_landmarks in data['landmarks']:
                if frame_landmarks is not None:
                    # Convert list to tensor
                    tensor = torch.tensor(frame_landmarks, dtype=torch.float32, device=device)
                    landmarks.append(tensor)
                else:
                    landmarks.append(None)
            
            # Extract video dimensions if available
            video_dims = None
            if 'video_width' in data and 'video_height' in data:
                video_dims = (data['video_width'], data['video_height'])
            
            return landmarks, video_dims
        
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}. Only JSON format is supported.")
    
    @staticmethod
    def load_parameters(input_path: str,
                       device: str = 'cuda') -> List[Optional[Live2DParameters]]:
        """
        Load Live2D parameters sequence from file.
        
        Args:
            input_path: Input file path
            device: Device to load tensors to
            
        Returns:
            List of Live2D parameters
        """
        input_path = Path(input_path)
        
        if input_path.suffix == '.json':
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            parameters = []
            for frame_params in data['parameters']:
                if frame_params is not None:
                    # Convert dict to Live2DParameters
                    param_dict = {}
                    for key, value in frame_params.items():
                        param_dict[key] = torch.tensor(value, dtype=torch.float32, device=device)
                    parameters.append(Live2DParameters(**param_dict))
                else:
                    parameters.append(None)
            
            return parameters
        
        elif input_path.suffix in ['.pkl', '.pickle']:
            with open(input_path, 'rb') as f:
                parameters = pickle.load(f)
            
            # Move tensors to specified device
            for i, params in enumerate(parameters):
                if params is not None:
                    for field_name in params.__dataclass_fields__:
                        field_value = getattr(params, field_name)
                        if isinstance(field_value, torch.Tensor):
                            setattr(params, field_name, field_value.to(device))
            
            return parameters
        
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")


def detect_input_type(input_path: str) -> str:
    """
    Detect the type of input file.
    
    Args:
        input_path: Path to input file
        
    Returns:
        Input type: 'video', 'landmarks', or 'parameters'
    """
    input_path = Path(input_path)
    
    # Check file extension
    if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return 'video'
    
    # Check if it's a JSON data file
    if input_path.suffix == '.json':
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            if 'format' in data:
                if 'landmarks' in data['format'].lower():
                    return 'landmarks'
                elif 'parameters' in data['format'].lower():
                    return 'parameters'
        except:
            pass
    
    # Default to video if unsure
    return 'video'