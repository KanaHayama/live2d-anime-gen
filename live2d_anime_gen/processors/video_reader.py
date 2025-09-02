"""Video reader for reading video files using NVDEC hardware acceleration."""

from typing import Iterator, Optional, Any
from pathlib import Path
import torch
import PyNvVideoCodec as nvc

from ..core.base_frame_reader import BaseFrameReader


class VideoReader(BaseFrameReader):
    """
    Read video frames from a file using NVDEC hardware acceleration.
    """
    
    def __init__(self, video_path: str) -> None:
        """
        Initialize NVDEC video reader.
        
        Args:
            video_path: Path to the video file
        """
        super().__init__()
        
        self.video_path: Path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create demuxer for extracting encoded packets
        self.demuxer = nvc.CreateDemuxer(str(self.video_path))
        
        # Create decoder with RGB output on GPU
        self.decoder = nvc.CreateDecoder(
            gpuid=0,
            codec=self.demuxer.GetNvCodecId(),
            usedevicememory=True,
            outputColorType=nvc.OutputColorType.RGB  # Direct RGB output
        )
        
        # Video properties
        self.fps: float = self.demuxer.FrameRate()
        self.frame_count: Optional[int] = None  # PyNvVideoCodec doesn't provide frame count directly
        self.width: int = self.demuxer.Width()
        self.height: int = self.demuxer.Height()
    
    def read_frames(self) -> Iterator[torch.Tensor]:
        """
        Iterate over video frames.
        
        Yields:
            Video frames as torch tensors (RGB format, H x W x 3, uint8, CUDA)
        """
        # Decode frames from demuxed packets
        for packet in self.demuxer:
            for frame in self.decoder.Decode(packet):
                # Convert DecodedFrame to CUDA tensor using DLPack (zero-copy)
                # Frame is already RGB format from decoder
                frame_tensor: torch.Tensor = torch.from_dlpack(frame)
                yield frame_tensor
    
    def close(self) -> None:
        """Release decoder and demuxer resources."""
        # PyNvVideoCodec handles cleanup automatically
        pass