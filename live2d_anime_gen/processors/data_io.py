"""Data I/O for intermediate results."""

import json
from enum import Enum
from typing import List, Optional, Tuple
from pathlib import Path
import torch
import numpy as np

from ..core.types import Live2DParameters


class InputType(Enum):
    """Enum for different input file types."""
    VIDEO = "video"
    LANDMARKS = "landmarks"  
    PARAMETERS = "parameters"


class DataExporter:
    """Export intermediate data to files."""
    
    @staticmethod
    def export_landmarks(landmarks: List[Optional[torch.Tensor]], 
                        output_path: str,
                        fps: float,
                        width: int,
                        height: int):
        """
        Export landmarks sequence to JSON file.
        
        Args:
            landmarks: List of landmark tensors (in original pixel coordinates)
            output_path: Output file path
            fps: Video fps
            width: Video width
            height: Video height
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        data = {
            'frame_count': len(landmarks),
            'fps': fps,
            'video_width': width,
            'video_height': height,
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
    
    @staticmethod
    def export_parameters(parameters: List[Optional[Live2DParameters]], 
                         output_path: str,
                         fps: float):
        """
        Export Live2D parameters sequence to JSON file.
        
        Args:
            parameters: List of Live2D parameters
            output_path: Output file path
            fps: Video fps
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        data = {
            'frame_count': len(parameters),
            'fps': fps,
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


class DataLoader:
    """Load intermediate data from files."""
    
    @staticmethod
    def load_landmarks(input_path: str, 
                      device: str = 'cuda') -> Tuple[List[Optional[torch.Tensor]], Optional[dict]]:
        """
        Load landmarks sequence from file.
        
        Args:
            input_path: Input file path
            device: Device to load tensors to
            
        Returns:
            Tuple of (landmarks list, metadata dict with video dimensions and fps or None)
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
            
            # Extract metadata
            metadata = {}
            if 'video_width' in data and 'video_height' in data:
                metadata['width'] = data['video_width']
                metadata['height'] = data['video_height']
            if 'fps' in data:
                metadata['fps'] = data['fps']
            
            return landmarks, metadata if metadata else None
        
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}. Only JSON format is supported.")
    
    @staticmethod
    def load_parameters(input_path: str,
                       device: str = 'cuda') -> List[Optional[Live2DParameters]]:
        """
        Load Live2D parameters sequence from JSON file.
        
        Args:
            input_path: Input file path
            device: Device to load tensors to
            
        Returns:
            List of Live2D parameters
        """
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
    
    @staticmethod
    def get_metadata_from_parameters(input_path: str) -> dict:
        """
        Get metadata from parameters JSON file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Dictionary with metadata including fps
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fps = data.get('fps')
        if fps is None:
            raise ValueError(f"Parameters file {input_path} is missing required fps metadata. Please regenerate the parameters file.")
        
        return {
            'fps': fps,
            'frame_count': data.get('frame_count')
        }
        


def detect_input_type(input_path: str) -> InputType:
    """
    Detect the type of input file.
    
    Args:
        input_path: Path to input file
        
    Returns:
        Input type enum value
    """
    input_path = Path(input_path)
    
    # Check file extension
    if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return InputType.VIDEO
    
    # Check if it's a JSON data file
    if input_path.suffix == '.json':
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Check content structure to determine type
            if 'landmarks' in data:
                return InputType.LANDMARKS
            elif 'parameters' in data:
                return InputType.PARAMETERS
        except:
            pass
    
    # Default to video if unsure
    return InputType.VIDEO