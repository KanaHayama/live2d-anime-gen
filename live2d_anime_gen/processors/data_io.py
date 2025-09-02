"""Data I/O for intermediate results."""

import json
from enum import Enum
from typing import List, Optional, Tuple, Union, Iterator
from pathlib import Path
import torch
import numpy as np
import ijson

from ..core.types import Live2DParameters
from .stream_utils import is_iterator


class InputType(Enum):
    """Enum for different input file types."""
    VIDEO = "video"
    LANDMARKS = "landmarks"  
    PARAMETERS = "parameters"


class StreamingJSONWriter:
    """Streaming JSON writer for large datasets."""
    
    def __init__(self, output_path: str, metadata: dict = None):
        """Initialize streaming JSON writer.
        
        Args:
            output_path: Output file path
            metadata: Optional metadata to include
        """
        self.output_path = Path(output_path)
        self.file = None
        self.metadata = metadata or {}
        self.first_item = True
        self.frame_count = 0
        
    def __enter__(self):
        self.file = open(self.output_path, 'w', encoding='utf-8')
        # Write metadata and start array
        self.file.write('{\n')
        
        # Write metadata first (except frame_count which will be added later)
        for key, value in self.metadata.items():
            if key != 'frame_count':  # Skip frame_count, will add it later
                self.file.write(f'  "{key}": {json.dumps(value)},\n')
        
        # Write placeholder for frame_count - we'll update this later
        self.frame_count_position = self.file.tell()
        self.file.write('  "frame_count": null,\n')  # null placeholder
        
        # Start data array
        self.file.write('  "data": [\n')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            # Close data array and object
            self.file.write('\n  ]\n}\n')
            
            # Update frame_count in place
            self._update_frame_count()
            
            self.file.close()
            
    def write_item(self, item):
        """Write a single item to the stream."""
        if not self.first_item:
            self.file.write(',\n')
        else:
            self.first_item = False
            
        # Indent the item
        item_json = json.dumps(item, indent=2)
        indented = '\n'.join('    ' + line for line in item_json.split('\n'))
        self.file.write(indented)
        
        # Count frames
        self.frame_count += 1
        
        # Flush to disk periodically
        self.file.flush()
    
    def _update_frame_count(self):
        """Update frame_count placeholder with actual count."""
        current_pos = self.file.tell()
        
        # Go back to frame_count position
        self.file.seek(self.frame_count_position)
        
        # Simply replace "null" with actual frame count
        self.file.write(f'  "frame_count": {self.frame_count}')
        
        # Go back to end of file
        self.file.seek(current_pos)


class DataExporter:
    """Export intermediate data to files."""
    
    @staticmethod
    def export_landmarks(landmarks, 
                        output_path: str,
                        fps: float,
                        width: int,
                        height: int):
        """
        Export landmarks sequence to JSON file (supports both list and iterator).
        
        Args:
            landmarks: List or iterator of landmark tensors (in original pixel coordinates)
            output_path: Output file path
            fps: Video fps
            width: Video width
            height: Video height
        """
        from .stream_utils import is_iterator
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if is_iterator(landmarks):
            # Use streaming JSON writer for iterators
            metadata = {
                'fps': fps,
                'width': width,
                'height': height
            }
            
            with StreamingJSONWriter(output_path, metadata) as writer:
                frame_count = 0
                for frame_landmarks in landmarks:
                    if frame_landmarks is not None:
                        # Convert tensor to numpy for processing
                        if isinstance(frame_landmarks, torch.Tensor):
                            landmarks_np = frame_landmarks.cpu().numpy()
                        else:
                            landmarks_np = np.array(frame_landmarks)
                        
                        # Landmarks are already normalized to [0, 1] range by detector
                        writer.write_item(landmarks_np.tolist())
                    else:
                        writer.write_item(None)
                    frame_count += 1
        else:
            # Original batch processing for lists
            # Convert to JSON-serializable format
            data = {
                'frame_count': len(landmarks),
                'fps': fps,
                'width': width,
                'height': height,
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
    def export_parameters(parameters, 
                         output_path: str,
                         fps: float):
        """
        Export Live2D parameters sequence to JSON file (supports both list and iterator).
        
        Args:
            parameters: List or iterator of Live2D parameters
            output_path: Output file path
            fps: Video fps
        """
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if is_iterator(parameters):
            # Use streaming JSON writer for iterators
            metadata = {
                'fps': fps
            }
            
            with StreamingJSONWriter(output_path, metadata) as writer:
                frame_count = 0
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
                        writer.write_item(param_dict)
                    else:
                        writer.write_item(None)
                    frame_count += 1
        else:
            # Original batch processing for lists
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


class StreamingJSONReader:
    """True streaming JSON reader using ijson for token-based parsing."""
    
    def __init__(self, input_path: str):
        """
        Initialize streaming JSON reader.
        
        Args:
            input_path: Input file path
        """
        self.input_path = Path(input_path)
        self.file = None
        self.metadata = {}
        
    def __enter__(self):
        self.file = open(self.input_path, 'rb')  # Open in binary mode for ijson
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
            
    def read_items(self):
        """Read items from the stream using ijson."""
        # Reset file pointer
        self.file.seek(0)
        
        # Parse data items from the 'data' array
        for item in ijson.items(self.file, 'data.item'):
            yield item
            
    def get_metadata(self):
        """Get metadata from the file using ijson."""
        # Reset file pointer
        self.file.seek(0)
        
        metadata = {}
        
        # Parse all top-level keys except 'data'
        parser = ijson.parse(self.file)
        for prefix, event, value in parser:
            if event == 'string' or event == 'number':
                # Get the top-level key name
                key = prefix.split('.')[0] if '.' in prefix else prefix
                if key != 'data' and '.' not in prefix:  # Only top-level metadata
                    # Convert Decimal to float for compatibility
                    if hasattr(value, '__float__'):
                        metadata[key] = float(value)
                    else:
                        metadata[key] = value
            elif event == 'start_array' and prefix == 'data':
                # Stop when we reach the data array
                break
                
        return metadata


class DataLoader:
    """Load intermediate data from files."""
    
    @staticmethod
    def load_landmarks(input_path: str, 
                      device: str = 'cuda') -> Tuple[Iterator[Optional[torch.Tensor]], dict]:
        """
        Load landmarks sequence from JSON file using streaming.
        
        Args:
            input_path: Input JSON file path
            device: Device to load tensors to
            
        Returns:
            Tuple of (landmarks iterator, metadata dict with video dimensions and fps)
        """
        input_path = Path(input_path)
        
        # Get metadata separately using ijson
        with StreamingJSONReader(input_path) as reader:
            data = reader.get_metadata()
        
        metadata = {
            'width': int(data['width']),
            'height': int(data['height']),
            'fps': float(data['fps']),
            'frame_count': int(data['frame_count']) if data['frame_count'] is not None else None
        }
        
        return DataLoader._create_landmarks_iterator(input_path, device), metadata
    
    @staticmethod
    def _create_landmarks_iterator(input_path: Path, device: str) -> Iterator[Optional[torch.Tensor]]:
        """Create streaming iterator for landmarks."""
        with StreamingJSONReader(input_path) as reader:
            for frame_landmarks in reader.read_items():
                if frame_landmarks is not None:
                    # Convert list to tensor
                    tensor = torch.tensor(frame_landmarks, dtype=torch.float32, device=device)
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
        
        return {
            'fps': data['fps'],
            'frame_count': data['frame_count']
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
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Determine type based on presence of video dimensions
        if 'width' in data:
            return InputType.LANDMARKS
        else:
            return InputType.PARAMETERS
    
    # Default to video if unsure
    return InputType.VIDEO