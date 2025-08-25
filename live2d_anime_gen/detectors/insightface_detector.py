"""InsightFace detector wrapper for 106-point landmark detection."""

from typing import Optional, Tuple
import numpy as np
import torch
from insightface.app import FaceAnalysis

from ..core.base_detector import BaseDetector


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
        
        self._num_landmarks = 106
    
    def detect(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """
        Detect 106 facial landmarks in the image.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            
        Returns:
            106 landmarks as torch tensor (106, 2) on GPU, or None if no face detected
        """
        # Detect faces
        faces = self.app.get(image)
        
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
        
        # Convert to torch tensor and move to GPU, normalize to [0,1]
        return self.postprocess_landmarks(landmarks, image.shape)
    
    def get_num_landmarks(self) -> int:
        """Return the number of landmarks (106)."""
        return self._num_landmarks
    
    def detect_multiple(self, image: np.ndarray, max_faces: int = 1) -> list[Optional[torch.Tensor]]:
        """
        Detect landmarks for multiple faces.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            max_faces: Maximum number of faces to detect
            
        Returns:
            List of landmark tensors, one per detected face
        """
        faces = self.app.get(image, max_num=max_faces)
        
        if not faces:
            return []
        
        results = []
        for face in faces:
            # Extract 106 landmarks for each face
            if hasattr(face, 'landmark_2d_106'):
                landmarks = face.landmark_2d_106
            elif hasattr(face, 'kps') and len(face.kps) == 106:
                landmarks = face.kps
            else:
                continue
            
            if landmarks.shape[0] == 106:
                results.append(self.postprocess_landmarks(landmarks, image.shape))
        
        return results