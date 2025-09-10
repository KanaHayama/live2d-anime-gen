"""MediaPipe detector wrapper for 478-point 3D landmark detection with iris tracking."""

from typing import Optional, Tuple, Union, Iterator, List
from pathlib import Path
import requests
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

from ..core.base_detector import BaseDetector
from ..processors.stream_utils import is_iterator, apply_to_stream


class MediaPipeDetector(BaseDetector):
    """
    MediaPipe FaceLandmarker detector using new Tasks API for 478-point 3D facial landmark detection.
    
    Provides 468 face landmarks + 10 iris landmarks (5 per eye) with 3D coordinates.
    All landmarks include (x, y, z) where:
    - x, y are normalized to [0, 1] relative to image dimensions
    - z represents depth with head center as origin (smaller = closer to camera)
    
    Uses the new MediaPipe Tasks API with RunningMode.VIDEO for optimal streaming performance.
    """
    
    def __init__(self,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe FaceLandmarker detector with auto-download.
        
        Args:
            max_num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Frame counter for timestamp calculation
        self.frame_counter = 0
        self.frame_time_ms = 40  # Default 25 fps (1000ms / 25 = 40ms per frame)
        
        # Ensure model exists and get path
        model_path = self._ensure_model_exists()
        
        # Initialize MediaPipe FaceLandmarker with new Tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
    
    def _ensure_model_exists(self) -> str:
        """
        Ensure model file exists in current directory, download if necessary.
        
        Returns:
            Path to the model file
        """
        model_path = Path.cwd() / "face_landmarker.task"
        
        if not model_path.exists():
            print("MediaPipe FaceLandmarker model not found. Downloading...")
            model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            
            try:
                # Download with progress bar
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(model_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading model") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                print(f"Model downloaded successfully to: {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download MediaPipe model: {e}")
        else:
            print(f"Using existing MediaPipe model: {model_path}")
        
        return str(model_path)
    
    def detect(self, input_data: Union[torch.Tensor, Iterator[torch.Tensor], List[torch.Tensor]]) -> Union[Optional[torch.Tensor], Iterator[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """
        Unified detection interface supporting single frame, batch, and streaming modes.
        
        Args:
            input_data: Input image(s) - single tensor, list, or iterator (RGB format, H x W x 3, uint8, CUDA)
            
        Returns:
            - Single frame: Optional[torch.Tensor] (478, 3) with 3D coordinates
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
        Detect 478 3D facial landmarks in a single image using new Tasks API.
        
        Args:
            image: Input image as torch tensor (RGB format, H x W x 3, uint8, CUDA)
            
        Returns:
            478 3D landmarks as torch tensor (478, 3) on GPU, or None if no face detected
        """
        # Get image dimensions
        height, width = image.shape[0], image.shape[1]
        
        # Convert to numpy for MediaPipe
        image_np = image.cpu().numpy()
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        
        # Calculate timestamp for this frame
        timestamp_ms = int(self.frame_counter * self.frame_time_ms)
        self.frame_counter += 1
        
        # Process with MediaPipe FaceLandmarker (VIDEO mode)
        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if not detection_result.face_landmarks:
            return None
        
        # Get the first (most prominent) face
        face_landmarks = detection_result.face_landmarks[0]
        
        # Extract all 478 landmarks with 3D coordinates
        landmarks = []
        for landmark in face_landmarks:
            landmarks.append([
                landmark.x,  # Normalized x [0, 1]
                landmark.y,  # Normalized y [0, 1]
                landmark.z   # Depth (relative to head center)
            ])
        
        # Convert to numpy array
        landmarks_np = np.array(landmarks, dtype=np.float32)
        
        # Verify we have 478 landmarks (468 face + 10 iris)
        if landmarks_np.shape[0] != 478:
            print(f"Warning: Expected 478 landmarks, got {landmarks_np.shape[0]}")
            # Fallback: pad with zeros if less than 478
            if landmarks_np.shape[0] < 478:
                padding = np.zeros((478 - landmarks_np.shape[0], 3), dtype=np.float32)
                landmarks_np = np.vstack([landmarks_np, padding])
        
        # Convert to torch tensor on CUDA
        landmarks_tensor = torch.from_numpy(landmarks_np).cuda()
        
        # Apply postprocessing
        return self.postprocess_landmarks(landmarks_tensor, (height, width))
    
    def postprocess_landmarks(self, landmarks: torch.Tensor, image_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Postprocess landmarks with normalization documentation.
        
        MediaPipe landmark coordinate system:
        - x: Normalized [0, 1] relative to image width (0=left, 1=right)
        - y: Normalized [0, 1] relative to image height (0=top, 1=bottom)
        - z: Normalized approximately [-0.3, 0.3] relative to head width
              (negative=closer to camera, positive=farther from camera, 0=head center)
        
        Args:
            landmarks: Raw landmarks tensor (478, 3) with normalized coordinates
            image_shape: Original image dimensions (height, width)
            
        Returns:
            Processed landmarks tensor (478, 3) with:
            - x, y: Kept normalized [0, 1] for pipeline consistency
            - z: Kept in original normalized scale (roughly -0.3 to 0.3)
        """
        # Keep all coordinates as-is from MediaPipe
        # x, y are already normalized [0, 1]
        # z is normalized relative to head width (approximately -0.3 to 0.3)
        return landmarks
    
    def reset_frame_counter(self):
        """Reset frame counter for new video stream."""
        self.frame_counter = 0
    
    def set_fps(self, fps: float):
        """
        Set the frame rate for timestamp calculation.
        
        Args:
            fps: Frames per second of the video
        """
        self.frame_time_ms = 1000.0 / fps
    
    def close(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()