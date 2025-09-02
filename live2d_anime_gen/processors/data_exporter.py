"""Data exporter for large datasets."""

from typing import Dict, Any, Optional, TextIO
import json
from pathlib import Path


class DataExporter:
    """Data exporter for large datasets."""
    
    def __init__(self, output_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Initialize data exporter.
        
        Args:
            output_path: Output file path
            metadata: Optional metadata to include
        """
        self.output_path: Path = Path(output_path)
        self.file: Optional[TextIO] = None
        self.metadata: Dict[str, Any] = metadata or {}
        self.first_item: bool = True
        self.frame_count: int = 0
        self.frame_count_position: int = 0
        
    def __enter__(self) -> 'DataExporter':
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
        
    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        if self.file:
            # Close data array and object
            self.file.write('\n  ]\n}\n')
            
            # Update frame_count in place
            self._update_frame_count()
            
            self.file.close()
            
    def write_item(self, item: Any) -> None:
        """Write a single item to the stream."""
        if self.file:
            if not self.first_item:
                self.file.write(',\n')
            else:
                self.first_item = False
                
            # Indent the item
            item_json: str = json.dumps(item, indent=2)
            indented: str = '\n'.join('    ' + line for line in item_json.split('\n'))
            self.file.write(indented)
            
            # Count frames
            self.frame_count += 1
            
            # Flush to disk periodically
            self.file.flush()
    
    def _update_frame_count(self) -> None:
        """Update frame_count placeholder with actual count."""
        if self.file:
            current_pos: int = self.file.tell()
            
            # Go back to frame_count position
            self.file.seek(self.frame_count_position)
            
            # Simply replace "null" with actual frame count
            self.file.write(f'  "frame_count": {self.frame_count}')
            
            # Go back to end of file
            self.file.seek(current_pos)