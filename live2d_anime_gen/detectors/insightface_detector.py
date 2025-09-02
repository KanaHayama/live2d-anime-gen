"""InsightFace detector wrapper for 106-point landmark detection."""

from typing import Optional, Tuple, Union, Iterator, List
import numpy as np
import torch
from insightface.app import FaceAnalysis

from ..core.base_detector import BaseDetector
from ..processors.stream_utils import is_iterator, apply_to_stream


class InsightFaceDetector(BaseDetector):
    """
    InsightFace detector wrapper for 106-point facial landmark detection.
    """
    
    def __init__(self, 
                 det_size: Tuple[int, int] = (640, 640),
                 det_thresh: float = 0.5):
        """
        Initialize InsightFace detector.
        
        Args:
            det_size: Detection input size (width, height)
            det_thresh: Detection confidence threshold
        """
        self.det_size = det_size
        self.det_thresh = det_thresh
        
        # Initialize FaceAnalysis with CUDA only
        providers = ['CUDAExecutionProvider']
        self.app = FaceAnalysis(providers=providers, allowed_modules=['detection', 'landmark_2d_106'])
        self.app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
    
    def detect(self, input_data: Union[torch.Tensor, Iterator[torch.Tensor], List[torch.Tensor]]) -> Union[Optional[torch.Tensor], Iterator[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """
        Unified detection interface supporting single frame, batch, and streaming modes.
        
        Args:
            input_data: Input image(s) - single tensor, list, or iterator (RGB format, H x W x 3, uint8, CUDA)
            
        Returns:
            - Single frame: Optional[torch.Tensor]
            - Multiple frames: Iterator or List of Optional[torch.Tensor]
        """
        # Handle single frame
        if isinstance(input_data, torch.Tensor):
            return self._detect_single(input_data)
        
        # Handle iterator/generator (streaming mode)
        elif is_iterator(input_data):
            return apply_to_stream(input_data, self._detect_single, preserve_none=True)
        
        # Handle list (batch mode)
        else:
            return [self._detect_single(image) for image in input_data]
    
    def _detect_single(self, image: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Detect 106 facial landmarks in a single image.
        
        Args:
            image: Input image as torch tensor (RGB format, H x W x 3, uint8, CUDA)
            
        Returns:
            106 landmarks as torch tensor (106, 2) on GPU, or None if no face detected
        """
        # Convert RGB to BGR on GPU for InsightFace
        image_bgr = torch.empty_like(image)
        image_bgr[..., 0] = image[..., 2]  # B = R
        image_bgr[..., 1] = image[..., 1]  # G = G
        image_bgr[..., 2] = image[..., 0]  # R = B
        
        # Convert to numpy for InsightFace API
        image_np = image_bgr.cpu().numpy()
        
        # Detect faces
        faces = self.app.get(image_np)
        
        if not faces:
            return None
        
        # Get the first (most prominent) face
        face = faces[0]
        
        # Extract 106 landmarks
        if hasattr(face, 'landmark_2d_106'):
            landmarks = face.landmark_2d_106
        elif hasattr(face, 'kps') and len(face.kps) == 106:
            landmarks = face.kps
        else:
            # Try to find 106-point landmarks in face attributes
            for key in face.keys():
                if 'landmark' in key and '106' in key:
                    landmarks = face[key]
                    break
            else:
                print(f"Warning: No 106-point landmarks found. Available keys: {face.keys() if hasattr(face, 'keys') else 'N/A'}")
                return None
        
        # Ensure we have 106 landmarks
        if landmarks.shape[0] != 106:
            print(f"Warning: Expected 106 landmarks, got {landmarks.shape[0]}")
            return None
        
        # Convert numpy landmarks to torch tensor on CUDA, then postprocess
        landmarks_tensor = torch.from_numpy(landmarks).cuda()
        return self.postprocess_landmarks(landmarks_tensor, image_np.shape)
    
