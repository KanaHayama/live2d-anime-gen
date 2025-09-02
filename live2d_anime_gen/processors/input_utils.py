"""Utilities for detecting and handling input types."""

import json
from pathlib import Path

from ..core.input_type import InputType


def detect_input_type(input_path: str) -> InputType:
    """
    Detect the type of input file.
    
    Args:
        input_path: Path to input file
        
    Returns:
        Input type enum value
    """
    input_path_obj: Path = Path(input_path)
    
    # Check file extension
    if input_path_obj.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return InputType.VIDEO
    
    # Check if it's a JSON data file
    if input_path_obj.suffix == '.json':
        with open(input_path_obj, 'r') as f:
            data = json.load(f)
        
        # Determine type based on presence of video dimensions
        if 'width' in data:
            return InputType.LANDMARKS
        else:
            return InputType.PARAMETERS
    
    # Default to video if unsure
    return InputType.VIDEO