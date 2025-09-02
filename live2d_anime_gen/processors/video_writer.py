"""Video writer for writing video files using NVENC hardware acceleration."""

from typing import Tuple, Optional, Any
from pathlib import Path
import torch
import PyNvVideoCodec as nvc


class VideoWriter:
    """
    Write frames to a video file using NVENC hardware acceleration.
    """
    
    def __init__(self, 
                 output_path: str,
                 fps: float,
                 frame_size: Tuple[int, int]) -> None:
        """
        Initialize NVENC video writer with HEVC encoding.
        
        Args:
            output_path: Path to output video file
            fps: Frames per second
            frame_size: Frame size (width, height)
        """
        self.output_path: Path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Assert and ensure frame_size types are correct
        assert len(frame_size) == 2, f"frame_size must be (width, height), got {frame_size}"
        self.width: int = int(frame_size[0])
        self.height: int = int(frame_size[1])
        assert isinstance(self.width, int) and isinstance(self.height, int), f"Width and height must be integers, got width={type(self.width)}, height={type(self.height)}"
        assert isinstance(fps, (int, float)), f"FPS must be numeric, got {type(fps)}"
        
        # Create NVENC encoder with ARGB input format and HEVC codec
        try:
            self.encoder = nvc.CreateEncoder(
                width=self.width,
                height=self.height,
                fmt="ARGB",  # ARGB 8-bit input format
                usecpuinputbuffer=False,   # Use GPU memory
                codec="hevc",  # Specify HEVC codec
                fps=fps   # Set frame rate
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create NVENC encoder: {e}")
        
        # Open output file for writing bitstream
        self.output_file = open(self.output_path, 'wb')
        self.frame_count: int = 0
    
    def write_frame(self, frame: torch.Tensor) -> None:
        """
        Write a single frame.
        
        Args:
            frame: Frame to write (RGB format, H x W x 3, uint8, on CUDA)
        """
        # Assert frame is on CUDA and has correct properties
        assert frame.device.type == 'cuda', f"Frame must be on CUDA, got {frame.device}"
        assert frame.dtype == torch.uint8, f"Frame must be uint8, got {frame.dtype}"
        assert frame.ndim == 3 and frame.shape[2] == 3, f"Frame must be H x W x 3, got {frame.shape}"
        
        height: int = frame.shape[0]
        width: int = frame.shape[1]
        
        # Create BGRA tensor with proper alignment
        bgra: torch.Tensor = torch.empty((height, width, 4), dtype=torch.uint8, device='cuda')
        
        # Copy RGB channels to BGRA format
        bgra[:, :, 0] = frame[:, :, 2]  # B <- R
        bgra[:, :, 1] = frame[:, :, 1]  # G <- G  
        bgra[:, :, 2] = frame[:, :, 0]  # R <- B
        bgra[:, :, 3] = 255             # A = 255
        
        # Synchronize CUDA operations to ensure all tensor operations are complete
        # before passing to NVENC encoder to prevent race conditions
        torch.cuda.synchronize()
        
        # Encode frame
        bitstream = self.encoder.Encode(bgra)
        
        # Write bitstream to file
        if bitstream:
            self.output_file.write(bitstream)
        
        # Always increment frame count regardless of bitstream availability
        # NVENC may buffer initial frames before outputting bitstream
        self.frame_count += 1
    
    def __enter__(self) -> 'VideoWriter':
        return self
    
    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        self.close()
    
    def close(self) -> None:
        """Release NVENC encoder and close output file."""
        # Flush any remaining frames
        if hasattr(self, 'encoder'):
            remaining_bitstream = self.encoder.EndEncode()
            if remaining_bitstream:
                self.output_file.write(remaining_bitstream)
        
        # Close output file
        if hasattr(self, 'output_file'):
            self.output_file.close()
        
        # Clean up encoder
        if hasattr(self, 'encoder'):
            del self.encoder