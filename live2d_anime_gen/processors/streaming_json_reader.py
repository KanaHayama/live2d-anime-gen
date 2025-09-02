"""Streaming JSON reader using ijson for token-based parsing."""

from typing import Iterator, Dict, Any, Optional, Union
import ijson
from pathlib import Path


class StreamingJSONReader:
    """True streaming JSON reader using ijson for token-based parsing."""
    
    def __init__(self, input_path: str) -> None:
        """
        Initialize streaming JSON reader.
        
        Args:
            input_path: Input file path
        """
        self.input_path: Path = Path(input_path)
        self.file: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        
    def __enter__(self) -> 'StreamingJSONReader':
        self.file = open(self.input_path, 'rb')  # Open in binary mode for ijson
        return self
        
    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        if self.file:
            self.file.close()
            
    def read_items(self) -> Iterator[Any]:
        """Read items from the stream using ijson."""
        # Reset file pointer
        if self.file:
            self.file.seek(0)
            
            # Parse data items from the 'data' array
            for item in ijson.items(self.file, 'data.item'):
                yield item
            
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from the file using ijson."""
        # Reset file pointer
        if self.file:
            self.file.seek(0)
            
            metadata: Dict[str, Any] = {}
            
            # Parse all top-level keys except 'data'
            parser = ijson.parse(self.file)
            for prefix, event, value in parser:
                if event == 'string' or event == 'number':
                    # Get the top-level key name
                    key: str = prefix.split('.')[0] if '.' in prefix else prefix
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
        return {}