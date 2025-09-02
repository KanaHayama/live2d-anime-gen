"""Load intermediate data from files."""

from typing import Tuple, Iterator, Optional, Dict, Any
import json
from pathlib import Path
import torch

from .streaming_json_reader import StreamingJSONReader
from ..core.types import Live2DParameters


class DataLoader:
    """Load intermediate data from files."""
    
    @staticmethod
    def load_landmarks(input_path: str, 
                      device: str = 'cuda') -> Tuple[Iterator[Optional[torch.Tensor]], Dict[str, Any]]:
        """
        Load landmarks sequence from JSON file using streaming.
        
        Args:
            input_path: Input JSON file path
            device: Device to load tensors to
            
        Returns:
            Tuple of (landmarks iterator, metadata dict with video dimensions and fps)
        """
        input_path_obj: Path = Path(input_path)
        
        # Get metadata separately using ijson
        with StreamingJSONReader(input_path_obj) as reader:
            data = reader.get_metadata()
        
        metadata: Dict[str, Any] = {
            'width': int(data['width']),
            'height': int(data['height']),
            'fps': float(data['fps']),
            'frame_count': int(data['frame_count']) if data['frame_count'] is not None else None
        }
        
        return DataLoader._create_landmarks_iterator(input_path_obj, device), metadata
    
    @staticmethod
    def _create_landmarks_iterator(input_path: Path, device: str) -> Iterator[Optional[torch.Tensor]]:
        """Create streaming iterator for landmarks."""
        with StreamingJSONReader(input_path) as reader:
            for frame_landmarks in reader.read_items():
                if frame_landmarks is not None:
                    # Convert list to tensor
                    tensor: torch.Tensor = torch.tensor(frame_landmarks, dtype=torch.float32, device=device)
                    yield tensor
                else:
                    yield None
    
    @staticmethod
    def load_parameters(input_path: str,
                       device: str = 'cuda') -> Iterator[Optional[Live2DParameters]]:
        """
        Load Live2D parameters sequence from JSON file using streaming.
        
        Args:
            input_path: Input JSON file path
            device: Device to load tensors to
            
        Returns:
            Iterator of Live2D parameters
        """
        return DataLoader._create_parameters_iterator(input_path, device)
    
    @staticmethod
    def _create_parameters_iterator(input_path: str, device: str) -> Iterator[Optional[Live2DParameters]]:
        """Create streaming iterator for Live2D parameters."""
        with StreamingJSONReader(input_path) as reader:
            for frame_params in reader.read_items():
                if frame_params is not None:
                    # Convert dict to Live2DParameters
                    param_dict = {}
                    for key, value in frame_params.items():
                        if key == 'custom_params':
                            param_dict[key] = value  # Keep as dict
                        elif value is not None:
                            param_dict[key] = torch.tensor(value, dtype=torch.float32, device=device)
                        else:
                            param_dict[key] = None
                    yield Live2DParameters(**param_dict)
                else:
                    yield None
    
    @staticmethod
    def get_metadata_from_parameters(input_path: str) -> Dict[str, Any]:
        """
        Get metadata from parameters JSON file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Dictionary with metadata including fps
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            'fps': data['fps'],
            'frame_count': data['frame_count']
        }