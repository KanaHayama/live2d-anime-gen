"""Face landmark to Live2D parameter mapper with 3D coordinate support and iris tracking."""

from typing import Tuple, Optional, Union, Iterator, List
import torch
import numpy as np

from ..core.base_mapper import BaseLandmarkMapper
from ..core.types import Live2DParameters
from ..core.constants import LANDMARK_INDICES, LANDMARK_POINTS
from ..processors.stream_utils import is_iterator, apply_to_stream


class FaceMapper(BaseLandmarkMapper):
    """
    Maps 478 3D facial landmarks from MediaPipe to Live2D parameters.
    
    Utilizes 3D coordinates (x, y, z) for improved head pose estimation.
    Includes iris tracking using MediaPipe's dedicated iris landmarks (468-477).
    
    Key design principle: Each parameter group (eyes, mouth, head pose, eye gaze)
    is calculated independently to avoid unwanted coupling.
    """
    
    # Bias values calculated from input video analysis to center head pose
    YAW_BIAS = -0.32    # Left-right rotation bias (degrees)
    PITCH_BIAS = -27.74 # Up-down rotation bias (degrees) - corrects excessive downward tilt
    
    def __init__(self, smooth_factor: float):
        """
        Initialize the face mapper.
        
        Args:
            smooth_factor: Temporal smoothing factor (0-1)
        """
        super().__init__(smooth_factor)
        # Store reference face size for normalization
        self._reference_face_size = None
        self._face_size_alpha = 0.95  # Smoothing for face size
    
    def map(self, 
            input_data: Union[torch.Tensor, Iterator[Optional[torch.Tensor]], List[Optional[torch.Tensor]]],
            image_shape: Optional[Tuple[int, int]] = None) -> Union[Live2DParameters, Iterator[Optional[Live2DParameters]], List[Optional[Live2DParameters]]]:
        """
        Unified mapping interface supporting single landmarks, batch, and streaming modes.
        
        Args:
            input_data: Input 3D landmarks - single tensor (478, 3), list, or iterator
            image_shape: Original image shape (height, width) - required for single input
            
        Returns:
            - Single input: Live2DParameters
            - Multiple inputs: Iterator or List of Optional[Live2DParameters]
        """
        # Handle single landmarks tensor
        if isinstance(input_data, torch.Tensor):
            if image_shape is None:
                raise ValueError("image_shape is required for single tensor input")
            return self._map_single(input_data, image_shape)
        
        # Handle iterator/generator (streaming mode)
        elif is_iterator(input_data):
            return apply_to_stream(input_data, lambda lm: self._map_single(lm, image_shape) if lm is not None else None, preserve_none=True)
        
        # Handle list (batch mode) 
        else:
            return [self._map_single(lm, image_shape) if lm is not None else None for lm in input_data]
    
    def _map_single(self, 
                   landmarks: torch.Tensor,
                   image_shape: Tuple[int, int]) -> Live2DParameters:
        """
        Map 478 3D facial landmarks to Live2D parameters for a single frame.
        
        Each parameter is calculated independently:
        - Eye openness: Based on eye aspect ratio
        - Mouth: Based on mouth shape  
        - Head pose: Using 3D coordinates for accurate estimation
        - Eye gaze: Using dedicated iris landmarks (468-477)
        
        Args:
            landmarks: 478 3D landmark points as torch tensor (478, 3) with normalized coords
            image_shape: Image dimensions (height, width)
            
        Returns:
            Live2D parameters
        """
        # MediaPipe landmarks are already normalized [0, 1] for x, y
        # Extract 2D and 3D components
        landmarks_2d = landmarks[:, :2]  # Just x, y for 2D calculations
        landmarks_3d = landmarks  # Full x, y, z for 3D calculations
        
        # Convert to pixel space for certain calculations
        landmarks_pixel = landmarks_2d.clone()
        landmarks_pixel[:, 0] *= image_shape[1]  # width
        landmarks_pixel[:, 1] *= image_shape[0]  # height
        
        # Use normalized 2D landmarks for most calculations
        landmarks_norm = landmarks_2d.clone()
        
        # Calculate face size for normalization (but don't transform landmarks)
        face_size = self._calculate_face_size(landmarks_norm)
        
        # Update reference face size with smoothing
        if self._reference_face_size is None:
            self._reference_face_size = face_size
        else:
            self._reference_face_size = (
                self._face_size_alpha * self._reference_face_size + 
                (1 - self._face_size_alpha) * face_size
            )
        
        # Calculate each parameter independently
        params = Live2DParameters(
            # Eye openness - only depends on eye shape
            eye_l_open=self._calculate_eye_openness(landmarks_norm, "left"),
            eye_r_open=self._calculate_eye_openness(landmarks_norm, "right"),
            
            # Mouth - only depends on mouth shape
            mouth_open_y=self._calculate_mouth_openness(landmarks_norm),
            mouth_form=self._calculate_mouth_form(landmarks_norm),
            
            # Head pose - using 3D coordinates for better accuracy
            angle_x=self._calculate_head_yaw_3d(landmarks_3d),
            angle_y=self._calculate_head_pitch_3d(landmarks_3d),
            angle_z=self._calculate_head_roll_3d(landmarks_3d),
            
            # Eye gaze - using dedicated iris landmarks from MediaPipe
            eye_ball_x=self._calculate_eye_ball_x_iris(landmarks_3d),
            eye_ball_y=self._calculate_eye_ball_y_iris(landmarks_3d),
            
            # Eyebrows - only depends on brow position relative to eyes
            brow_l_y=self._calculate_eyebrow_height(landmarks_norm, "left"),
            brow_r_y=self._calculate_eyebrow_height(landmarks_norm, "right"),
        )
        
        # Apply postprocessing (clamping and smoothing)
        return self.postprocess_parameters(params)
    
    def _calculate_face_size(self, landmarks: torch.Tensor) -> float:
        """
        Calculate normalized face size for reference.
        
        Args:
            landmarks: Normalized landmarks [0, 1]
            
        Returns:
            Face size metric
        """
        # Use inter-ocular distance as stable reference
        left_eye = landmarks[LANDMARK_POINTS["left_eye_corners"]].mean(dim=0)
        right_eye = landmarks[LANDMARK_POINTS["right_eye_corners"]].mean(dim=0)
        inter_ocular = torch.norm(right_eye - left_eye).item()
        
        return inter_ocular
    
    def _calculate_eye_openness(self, landmarks: torch.Tensor, side: str) -> torch.Tensor:
        """
        Calculate eye openness using eye aspect ratio.
        Independent of head pose and other features.
        
        Args:
            landmarks: Normalized landmarks [0, 1]
            side: "left" or "right"
            
        Returns:
            Eye openness value (0-1)
        """
        if side == "left":
            corners = LANDMARK_POINTS["left_eye_corners"]
            top_indices = LANDMARK_POINTS["left_eye_top_points"]
            bottom_indices = LANDMARK_POINTS["left_eye_bottom_points"]
        else:
            corners = LANDMARK_POINTS["right_eye_corners"]
            top_indices = LANDMARK_POINTS["right_eye_top_points"]
            bottom_indices = LANDMARK_POINTS["right_eye_bottom_points"]
        
        # Calculate eye width
        inner_corner = landmarks[corners[0]]
        outer_corner = landmarks[corners[1]]
        eye_width = torch.norm(outer_corner - inner_corner)
        
        # Calculate eye height (average of multiple points)
        top_points = landmarks[top_indices]
        bottom_points = landmarks[bottom_indices]
        
        heights = []
        min_len = min(len(top_points), len(bottom_points))
        for i in range(min_len):
            heights.append(torch.abs(top_points[i, 1] - bottom_points[i, 1]))
        
        eye_height = torch.mean(torch.stack(heights)) if heights else torch.tensor(0.0)
        
        # Eye Aspect Ratio (height/width)
        ear = eye_height / (eye_width + 1e-6)
        
        # Normalize by face size for consistency across distances
        scale_factor = self._reference_face_size / (self._calculate_face_size(landmarks) + 1e-6)
        ear_normalized = ear * scale_factor
        
        # Map to [0, 2] with calibrated thresholds for full eye range
        # 0=closed, 1=normal open, 2=large open (for surprised/shocked expressions)
        closed_threshold = 0.25    # Lowered from 0.32 for better closure detection
        normal_threshold = 0.35    # Lowered from 0.45 to reach 1.0 more easily
        large_open_threshold = 0.55    # Lowered from 0.65 to use 1-2 range
        
        if ear_normalized < closed_threshold:
            # Closed range: 0 to 0.2
            openness = ear_normalized / closed_threshold * 0.2
        elif ear_normalized < normal_threshold:
            # Transition to normal: 0.2 to 1.0
            t = (ear_normalized - closed_threshold) / (normal_threshold - closed_threshold)
            openness = 0.2 + t * 0.8
        elif ear_normalized < large_open_threshold:
            # Normal open range: 1.0 to 1.2
            t = (ear_normalized - normal_threshold) / (large_open_threshold - normal_threshold)
            openness = 1.0 + t * 0.2
        else:
            # Large open eyes range: 1.2 to 2.0 for very open/surprised expressions
            t = min((ear_normalized - large_open_threshold) / (large_open_threshold * 0.5), 1.0)
            openness = 1.2 + t * 0.8
        
        # Ensure openness is a tensor
        if not isinstance(openness, torch.Tensor):
            openness = torch.tensor(openness, device=landmarks.device)
        
        return torch.clamp(openness, 0, 2)
    
    def _calculate_mouth_openness(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate mouth openness using mouth aspect ratio.
        Independent of head pose and smile.
        
        Args:
            landmarks: Normalized landmarks [0, 1]
            
        Returns:
            Mouth openness value (0-1)
        """
        corners = LANDMARK_POINTS["mouth_corners"]
        centers = LANDMARK_POINTS["mouth_center_points"]
        
        # Mouth dimensions
        left_corner = landmarks[corners[0]]
        right_corner = landmarks[corners[1]]
        upper_lip = landmarks[centers[0]]
        lower_lip = landmarks[centers[1]]
        
        mouth_width = torch.norm(right_corner - left_corner)
        mouth_height = torch.abs(lower_lip[1] - upper_lip[1])
        
        # Mouth Aspect Ratio
        mar = mouth_height / (mouth_width + 1e-6)
        
        # Normalize by face size
        scale_factor = self._reference_face_size / (self._calculate_face_size(landmarks) + 1e-6)
        mar_normalized = mar * scale_factor
        
        # Map to [0, 1] with adjusted thresholds for thicker lips
        closed_threshold = 0.80  # Even higher threshold for perfect closure
        open_threshold = 1.04   # Keep same open threshold
        
        if mar_normalized <= closed_threshold:
            openness = torch.tensor(0.0, device=landmarks.device)
        elif mar_normalized >= open_threshold:
            openness = torch.tensor(1.0, device=landmarks.device)
        else:
            t = (mar_normalized - closed_threshold) / (open_threshold - closed_threshold)
            openness = torch.pow(t, 0.8) if isinstance(t, torch.Tensor) else t ** 0.8
        
        # Ensure openness is a tensor
        if not isinstance(openness, torch.Tensor):
            openness = torch.tensor(openness, device=landmarks.device)
        
        return torch.clamp(openness, 0, 1)
    
    def _calculate_mouth_form(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate mouth form (smile/frown).
        Independent of mouth openness and head pose.
        
        Args:
            landmarks: Normalized landmarks [0, 1]
            
        Returns:
            Mouth form value (0 to 2, 1.0 = neutral)
        """
        corners = LANDMARK_POINTS["mouth_corners"]
        centers = LANDMARK_POINTS["mouth_center_points"]
        
        left_corner = landmarks[corners[0]]
        right_corner = landmarks[corners[1]]
        upper_lip = landmarks[centers[0]]
        lower_lip = landmarks[centers[1]]
        
        # Mouth center Y position
        mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
        
        # Average corner height relative to mouth center
        left_offset = left_corner[1] - mouth_center_y
        right_offset = right_corner[1] - mouth_center_y
        avg_offset = (left_offset + right_offset) / 2
        
        # Normalize by mouth width for scale invariance
        mouth_width = torch.norm(right_corner - left_corner)
        normalized_offset = avg_offset / (mouth_width + 1e-6)
        
        # Map to form parameter
        # Negative offset (corners up) = smile
        # Positive offset (corners down) = frown
        form = -normalized_offset * 10.0 + 1.0
        
        # Ensure form is a tensor
        if not isinstance(form, torch.Tensor):
            form = torch.tensor(form, device=landmarks.device)
        
        return torch.clamp(form, 0, 2)
    
    def _calculate_head_yaw(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate head yaw (left-right rotation).
        Based on face asymmetry, independent of eye/mouth state.
        
        Args:
            landmarks: Normalized landmarks [0, 1]
            
        Returns:
            Yaw angle in degrees
        """
        # Use nose position relative to face center
        nose_tip = landmarks[LANDMARK_POINTS["nose_tip"]]
        
        # Face center from eye corners
        left_eye = landmarks[LANDMARK_POINTS["left_eye_corners"]].mean(dim=0)
        right_eye = landmarks[LANDMARK_POINTS["right_eye_corners"]].mean(dim=0)
        face_center_x = (left_eye[0] + right_eye[0]) / 2
        
        # Nose deviation from center (normalized by inter-ocular distance)
        inter_ocular = torch.norm(right_eye - left_eye)
        nose_offset = (nose_tip[0] - face_center_x) / (inter_ocular + 1e-6)
        
        # Face contour asymmetry
        left_face = landmarks[LANDMARK_POINTS["face_left_points"]]
        right_face = landmarks[LANDMARK_POINTS["face_right_points"]]
        
        # Average distance from center to each side
        left_dist = torch.abs(left_face[:, 0] - face_center_x).mean()
        right_dist = torch.abs(face_center_x - right_face[:, 0]).mean()
        
        # Asymmetry ratio
        asymmetry = (right_dist - left_dist) / (right_dist + left_dist + 1e-6)
        
        # Combine cues with enhanced scaling for more visible movement
        yaw = nose_offset * 90.0 + asymmetry * 54.0  # ~3.6x increase
        
        return torch.clamp(yaw, -30, 30)
    
    def _calculate_head_pitch(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate head pitch (up-down rotation).
        Based on nose position relative to eyes, independent of expression.
        
        Args:
            landmarks: Normalized landmarks [0, 1]
            
        Returns:
            Pitch angle in degrees
        """
        # Nose tip position
        nose_tip = landmarks[LANDMARK_POINTS["nose_tip"]]
        
        # Eye line Y position
        left_eye = landmarks[LANDMARK_POINTS["left_eye_corners"]].mean(dim=0)
        right_eye = landmarks[LANDMARK_POINTS["right_eye_corners"]].mean(dim=0)
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        
        # Vertical offset normalized by face height
        face_points = landmarks[:33]  # Face contour
        face_height = face_points[:, 1].max() - face_points[:, 1].min()
        
        vertical_offset = (nose_tip[1] - eye_center_y) / (face_height + 1e-6)
        
        # Map to pitch angle with moderate scaling for more visible movement  
        pitch = vertical_offset * 150.0  # ~2.5x increase from 60.0
        
        return torch.clamp(pitch, -30, 30)
    
    def _calculate_head_roll(self, landmarks_pixel: torch.Tensor, image_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Calculate head roll (tilt).
        Based on eye line angle, independent of other features.
        
        Args:
            landmarks_pixel: Landmarks in pixel coordinates
            image_shape: (height, width)
            
        Returns:
            Roll angle in degrees
        """
        # Use pixel coordinates for accurate angle calculation
        left_eye = landmarks_pixel[LANDMARK_POINTS["left_eye_corners"]].mean(dim=0)
        right_eye = landmarks_pixel[LANDMARK_POINTS["right_eye_corners"]].mean(dim=0)
        
        # Calculate angle from horizontal
        delta_y = right_eye[1] - left_eye[1]
        delta_x = right_eye[0] - left_eye[0]
        
        roll = torch.atan2(delta_y, delta_x) * 180.0 / torch.pi
        
        return torch.clamp(roll, -15, 15)
    
    def _calculate_eye_ball_x(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate horizontal eye gaze.
        Only based on iris position if available, otherwise neutral.
        
        Args:
            landmarks: Normalized landmarks [0, 1]
            
        Returns:
            Eye ball X position (-1 to 1)
        """
        # Check for iris landmarks
        left_iris = LANDMARK_POINTS.get("left_iris", [])
        right_iris = LANDMARK_POINTS.get("right_iris", [])
        
        # If no iris data, return neutral position
        if not left_iris or not all(idx < len(landmarks) for idx in left_iris + right_iris):
            return torch.tensor(0.0, device=landmarks.device)
        
        # Calculate iris offset from eye center
        left_iris_center = landmarks[left_iris].mean(dim=0)
        right_iris_center = landmarks[right_iris].mean(dim=0)
        
        left_eye_center = landmarks[LANDMARK_POINTS["left_eye_corners"]].mean(dim=0)
        right_eye_center = landmarks[LANDMARK_POINTS["right_eye_corners"]].mean(dim=0)
        
        # Normalize by eye width
        left_eye_width = torch.norm(
            landmarks[LANDMARK_POINTS["left_eye_corners"][1]] - 
            landmarks[LANDMARK_POINTS["left_eye_corners"][0]]
        )
        right_eye_width = torch.norm(
            landmarks[LANDMARK_POINTS["right_eye_corners"][1]] - 
            landmarks[LANDMARK_POINTS["right_eye_corners"][0]]
        )
        
        left_offset = (left_iris_center[0] - left_eye_center[0]) / (left_eye_width + 1e-6)
        right_offset = (right_iris_center[0] - right_eye_center[0]) / (right_eye_width + 1e-6)
        
        # Average both eyes - need large multiplier for visible movement
        eye_ball_x = (left_offset + right_offset) * 12.0
        
        return torch.clamp(eye_ball_x, -1, 1)
    
    def _calculate_eye_ball_y(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate vertical eye gaze using stable eye corner reference.
        Avoids eyelid-dependent measurements that change when looking down.
        
        Args:
            landmarks: Normalized landmarks [0, 1]
            
        Returns:
            Eye ball Y position (-1 to 1)
        """
        # Check for iris landmarks
        left_iris = LANDMARK_POINTS.get("left_iris", [])
        right_iris = LANDMARK_POINTS.get("right_iris", [])
        
        # If no iris data, return neutral position
        if not left_iris or not all(idx < len(landmarks) for idx in left_iris + right_iris):
            return torch.tensor(0.0, device=landmarks.device)
        
        # Calculate iris centers
        left_iris_center = landmarks[left_iris].mean(dim=0)
        right_iris_center = landmarks[right_iris].mean(dim=0)
        
        # Use stable eye corner Y positions as reference (unaffected by eyelid closure)
        left_eye_corners = LANDMARK_POINTS["left_eye_corners"]  # [39, 35]
        right_eye_corners = LANDMARK_POINTS["right_eye_corners"]  # [89, 93]
        
        # Eye corner Y average as stable baseline
        left_corner_y = landmarks[left_eye_corners][:, 1].mean()
        right_corner_y = landmarks[right_eye_corners][:, 1].mean()
        
        # Calculate iris offset from stable corner baseline
        left_offset = left_iris_center[1] - left_corner_y
        right_offset = right_iris_center[1] - right_corner_y
        
        # Average both eyes
        avg_offset = (left_offset + right_offset) / 2
        
        # Normalize by inter-corner distance for scale invariance
        left_eye_width = torch.norm(landmarks[left_eye_corners[1]] - landmarks[left_eye_corners[0]])
        right_eye_width = torch.norm(landmarks[right_eye_corners[1]] - landmarks[right_eye_corners[0]])
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        
        normalized_offset = avg_offset / (avg_eye_width + 1e-6)
        
        # Map to Live2D coordinates: CV (Y down) → Live2D (Y up)
        # Use 8x multiplier for more visible eye movement
        eye_ball_y = -normalized_offset * 8.0
        
        return torch.clamp(eye_ball_y, -1, 1)
    
    def _calculate_eyebrow_height(self, landmarks: torch.Tensor, side: str) -> torch.Tensor:
        """
        Calculate eyebrow height relative to eye.
        Independent of head pose and other expressions.
        
        Args:
            landmarks: Normalized landmarks [0, 1]
            side: "left" or "right"
            
        Returns:
            Eyebrow height (-1 to 1)
        """
        if side == "left":
            brow_indices = LANDMARK_INDICES["left_eyebrow"]
            eye_indices = LANDMARK_INDICES["left_eye"]
        else:
            brow_indices = LANDMARK_INDICES["right_eyebrow"]
            eye_indices = LANDMARK_INDICES["right_eye"]
        
        # Average positions
        brow_center = landmarks[brow_indices].mean(dim=0)
        eye_center = landmarks[eye_indices].mean(dim=0)
        
        # Vertical distance normalized by eye height
        eye_points = landmarks[eye_indices]
        eye_height = eye_points[:, 1].max() - eye_points[:, 1].min()
        
        vertical_dist = (eye_center[1] - brow_center[1]) / (eye_height + 1e-6)
        
        # Map to [-1, 1]
        brow_height = torch.clamp(vertical_dist * 3.0 - 0.5, -1, 1)
        
        return brow_height
    
    # New 3D methods for MediaPipe landmarks
    def _calculate_head_yaw_3d(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate head yaw (left-right rotation) using comprehensive 3D analysis.
        
        Uses multiple 3D cues:
        1. Face contour depth asymmetry (primary indicator)
        2. Nose tip deviation from face center
        3. Eye separation changes during rotation
        4. Temple depth differences
        
        Args:
            landmarks: 3D landmarks tensor (478, 3) with x, y, z coordinates
            
        Returns:
            Yaw angle in degrees (not clamped to allow Live2D renderer to handle limits)
        """
        # Use correct MediaPipe face contour landmarks
        face_oval_indices = LANDMARK_INDICES["face_oval"]
        face_points = landmarks[face_oval_indices]
        
        # Split face contour into left and right sides based on x-coordinate
        face_center_x = face_points[:, 0].mean()
        left_face_mask = face_points[:, 0] < face_center_x
        right_face_mask = face_points[:, 0] > face_center_x
        
        left_face_points = face_points[left_face_mask]
        right_face_points = face_points[right_face_mask]
        
        # Feature 1: Face contour depth asymmetry (strongest indicator)
        if len(left_face_points) > 0 and len(right_face_points) > 0:
            left_depth = left_face_points[:, 2].mean()
            right_depth = right_face_points[:, 2].mean()
            
            # When turning left: right side closer (smaller z), left side farther (larger z)
            # When turning right: left side closer (smaller z), right side farther (larger z)
            depth_asymmetry = right_depth - left_depth
        else:
            depth_asymmetry = torch.tensor(0.0, device=landmarks.device)
        
        # Feature 2: Nose tip horizontal deviation
        nose_tip = landmarks[LANDMARK_POINTS["nose_tip"]]
        # Use eye centers for more stable face center calculation
        left_eye_center = landmarks[LANDMARK_POINTS["left_eye_corners"]].mean(dim=0)
        right_eye_center = landmarks[LANDMARK_POINTS["right_eye_corners"]].mean(dim=0)
        stable_face_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
        
        nose_deviation = nose_tip[0] - stable_face_center_x
        
        # Feature 3: Eye separation changes (perspective effect)
        inter_eye_distance = torch.norm(right_eye_center - left_eye_center)
        # Normalize by expected distance (this varies with head rotation)
        # During yaw rotation, eye separation appears to change due to perspective
        
        # Feature 4: Temple depth difference for additional validation
        left_temple = landmarks[LANDMARK_POINTS["left_temple"]]
        right_temple = landmarks[LANDMARK_POINTS["right_temple"]]
        temple_depth_diff = right_temple[2] - left_temple[2]
        
        # Feature 5: Face width asymmetry
        left_face_width = torch.abs(left_face_points[:, 0] - stable_face_center_x).mean()
        right_face_width = torch.abs(right_face_points[:, 0] - stable_face_center_x).mean()
        width_asymmetry = (right_face_width - left_face_width) / (right_face_width + left_face_width + 1e-6)
        
        # Combine all features with calibrated weights
        # Primary: depth asymmetry (most reliable for 3D)
        # Secondary: nose deviation (visible in 2D projection)
        # Tertiary: temple depth and width asymmetry (validation cues)
        
        yaw_angle = (
            depth_asymmetry * 180.0 +      # Primary 3D cue - increased sensitivity
            nose_deviation * 120.0 +       # Secondary 2D projection cue  
            temple_depth_diff * 60.0 +     # Additional 3D validation
            width_asymmetry * 40.0         # Geometric asymmetry
        )
        
        # Apply adaptive scaling based on detected confidence
        # If multiple cues agree, trust the result more
        cue_agreement = torch.abs(depth_asymmetry * 180.0) + torch.abs(nose_deviation * 120.0)
        confidence_boost = 1.0 + (cue_agreement / 30.0) * 0.3  # Up to 30% boost for strong signals
        
        yaw_angle = yaw_angle * confidence_boost
        
        # Apply bias correction to center the head pose
        yaw_angle = yaw_angle - self.YAW_BIAS
        
        # DO NOT clamp - let Live2D renderer handle the limits
        # This allows for more expressive head movements beyond ±30 degrees
        return yaw_angle
    
    def _calculate_head_pitch_3d(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate head pitch (up-down rotation) using advanced 3D analysis.
        
        Uses multiple 3D depth cues:
        1. Forehead to chin depth gradient
        2. Nose tip relative depth position  
        3. Eye to mouth vertical depth relationship
        4. Face profile curvature analysis
        
        Args:
            landmarks: 3D landmarks tensor (478, 3) with x, y, z coordinates
            
        Returns:
            Pitch angle in degrees (not clamped to allow Live2D renderer to handle limits)
        """
        # Get key facial reference points
        forehead = landmarks[LANDMARK_POINTS["forehead_center"]]
        nose_tip = landmarks[LANDMARK_POINTS["nose_tip"]]
        chin = landmarks[LANDMARK_POINTS["chin_tip"]]
        
        # Get eye centers for stable reference
        left_eye_center = landmarks[LANDMARK_POINTS["left_eye_corners"]].mean(dim=0)
        right_eye_center = landmarks[LANDMARK_POINTS["right_eye_corners"]].mean(dim=0)
        eye_level = (left_eye_center + right_eye_center) / 2
        
        # Get mouth center for additional reference
        mouth_corners = landmarks[LANDMARK_POINTS["mouth_corners"]]
        mouth_center = mouth_corners.mean(dim=0)
        
        # Feature 1: Forehead-to-chin depth gradient (primary indicator)
        # When looking up: chin closer (smaller z), forehead farther (larger z)
        # When looking down: forehead closer (smaller z), chin farther (larger z)
        vertical_depth_gradient = chin[2] - forehead[2]
        
        # Feature 2: Nose tip depth relative to eye level
        # Nose should be approximately at same depth as eyes in neutral position
        nose_to_eye_depth = nose_tip[2] - eye_level[2]
        
        # Feature 3: Eye-to-mouth depth relationship
        eye_to_mouth_depth = mouth_center[2] - eye_level[2]
        
        # Feature 4: Vertical position changes (2D projection cues)
        # Calculate face height for normalization
        face_height = torch.abs(forehead[1] - chin[1])
        
        # Nose vertical position relative to expected neutral
        eye_to_chin_midpoint_y = (eye_level[1] + chin[1]) / 2
        nose_vertical_deviation = (nose_tip[1] - eye_to_chin_midpoint_y) / (face_height + 1e-6)
        
        # Feature 5: Face profile curvature (advanced 3D analysis)
        # Sample points along vertical face profile
        profile_points = torch.stack([forehead, eye_level, nose_tip, mouth_center, chin])
        
        # Calculate curvature changes in z-direction along y-axis
        # Sort by y-coordinate for proper ordering
        y_coords = profile_points[:, 1]
        z_coords = profile_points[:, 2]
        
        # Approximate face profile curvature
        if len(profile_points) >= 3:
            # Calculate second derivative approximation (curvature)
            z_diff1 = z_coords[1:] - z_coords[:-1]
            if len(z_diff1) >= 2:
                z_diff2 = z_diff1[1:] - z_diff1[:-1]
                profile_curvature = z_diff2.mean()
            else:
                profile_curvature = torch.tensor(0.0, device=landmarks.device)
        else:
            profile_curvature = torch.tensor(0.0, device=landmarks.device)
        
        # Combine all features with optimized weights
        pitch_angle = (
            vertical_depth_gradient * 200.0 +    # Primary 3D depth gradient  
            nose_to_eye_depth * 150.0 +          # Nose depth deviation
            eye_to_mouth_depth * 80.0 +          # Mouth depth relationship
            nose_vertical_deviation * 60.0 +     # 2D projection cue
            profile_curvature * 100.0            # Face curvature analysis
        )
        
        # Apply confidence-based scaling
        # Strong depth gradients get higher confidence
        confidence_indicators = torch.abs(vertical_depth_gradient * 200.0) + torch.abs(nose_to_eye_depth * 150.0)
        confidence_boost = 1.0 + (confidence_indicators / 25.0) * 0.25  # Up to 25% boost
        
        pitch_angle = pitch_angle * confidence_boost
        
        # Apply bias correction to center the head pose
        pitch_angle = pitch_angle - self.PITCH_BIAS
        
        # DO NOT clamp - let Live2D renderer handle the limits
        return pitch_angle
    
    def _calculate_head_roll_3d(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate head roll (left-right tilt) using enhanced 3D analysis.
        
        Uses multiple geometric cues:
        1. 3D eye line vector analysis
        2. Face contour symmetry changes during roll
        3. Depth-corrected angle calculations  
        4. Temple and cheek depth relationships
        
        Args:
            landmarks: 3D landmarks tensor (478, 3) with x, y, z coordinates
            
        Returns:
            Roll angle in degrees (not clamped to allow Live2D renderer to handle limits)
        """
        # Get eye corner landmarks for precise eye line calculation
        left_eye_corners = landmarks[LANDMARK_POINTS["left_eye_corners"]]
        right_eye_corners = landmarks[LANDMARK_POINTS["right_eye_corners"]]
        
        # Calculate eye centers with full 3D information
        left_eye_center = left_eye_corners.mean(dim=0)
        right_eye_center = right_eye_corners.mean(dim=0)
        
        # Feature 1: Primary 3D eye line vector
        eye_vector_3d = right_eye_center - left_eye_center
        
        # Calculate roll angle using 2D projection (x, y plane)
        # This is the primary roll indicator
        primary_roll = torch.atan2(eye_vector_3d[1], eye_vector_3d[0])
        primary_roll_deg = torch.rad2deg(primary_roll)
        
        # Feature 2: 3D depth correction for perspective effects
        # When head is tilted, the eye depths change relative to each other
        eye_depth_difference = right_eye_center[2] - left_eye_center[2]
        
        # Feature 3: Face contour symmetry analysis
        face_oval_indices = LANDMARK_INDICES["face_oval"]
        face_points = landmarks[face_oval_indices]
        
        # Split face contour into upper and lower halves
        face_center_y = face_points[:, 1].mean()
        upper_face_mask = face_points[:, 1] < face_center_y
        lower_face_mask = face_points[:, 1] > face_center_y
        
        upper_face_points = face_points[upper_face_mask]
        lower_face_points = face_points[lower_face_mask]
        
        # Calculate horizontal asymmetry in upper vs lower face
        if len(upper_face_points) > 0 and len(lower_face_points) > 0:
            upper_x_range = upper_face_points[:, 0].max() - upper_face_points[:, 0].min()
            lower_x_range = lower_face_points[:, 0].max() - lower_face_points[:, 0].min()
            face_asymmetry = (upper_x_range - lower_x_range) / (upper_x_range + lower_x_range + 1e-6)
        else:
            face_asymmetry = torch.tensor(0.0, device=landmarks.device)
        
        # Feature 4: Temple depth relationship
        left_temple = landmarks[LANDMARK_POINTS["left_temple"]]
        right_temple = landmarks[LANDMARK_POINTS["right_temple"]]
        
        # During roll, temples move to different depths
        temple_depth_asymmetry = (right_temple[2] - left_temple[2])
        
        # Feature 5: Eyebrow line angle as secondary validation
        left_eyebrow_indices = LANDMARK_INDICES["left_eyebrow"]
        right_eyebrow_indices = LANDMARK_INDICES["right_eyebrow"]
        
        left_eyebrow_center = landmarks[left_eyebrow_indices].mean(dim=0)
        right_eyebrow_center = landmarks[right_eyebrow_indices].mean(dim=0)
        
        eyebrow_vector = right_eyebrow_center - left_eyebrow_center
        eyebrow_roll = torch.atan2(eyebrow_vector[1], eyebrow_vector[0])
        eyebrow_roll_deg = torch.rad2deg(eyebrow_roll)
        
        # Combine all features with appropriate weights
        # Primary: eye line angle (most reliable)
        # Secondary: depth and asymmetry cues for validation
        roll_angle = (
            primary_roll_deg * 1.0 +              # Primary 2D angle measurement
            eyebrow_roll_deg * 0.3 +              # Eyebrow line validation
            eye_depth_difference * 20.0 +         # 3D depth correction
            temple_depth_asymmetry * 15.0 +       # Temple depth cue
            face_asymmetry * 8.0                  # Face contour asymmetry
        )
        
        # Apply stability filtering - roll changes should be smooth
        # For roll, we can be more conservative as it's typically smaller movement
        confidence_indicators = torch.abs(primary_roll_deg) + torch.abs(eyebrow_roll_deg * 0.3)
        confidence_scaling = 1.0 + (confidence_indicators / 20.0) * 0.2  # Up to 20% boost
        
        roll_angle = roll_angle * confidence_scaling
        
        # Apply correct sign convention for Live2D
        # Negative for clockwise tilt (right shoulder down)
        # Positive for counter-clockwise tilt (left shoulder down)
        roll_angle = -roll_angle  # Invert for Live2D coordinate system
        
        # DO NOT clamp - let Live2D renderer handle the limits
        return roll_angle
    
    def _calculate_eye_ball_x_iris(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate horizontal eye gaze using MediaPipe iris landmarks.
        Uses dedicated iris center points (468, 473) for accurate tracking.
        
        Args:
            landmarks: 3D landmarks tensor (478, 3) with iris landmarks
            
        Returns:
            Eye ball X position (-1 to 1)
        """
        # MediaPipe iris landmarks: 468-472 (left), 473-477 (right)
        # First point in each group is the iris center
        left_iris_center = landmarks[468]  # Left iris center
        right_iris_center = landmarks[473]  # Right iris center
        
        # Get eye corners for reference
        left_eye_corners = landmarks[LANDMARK_POINTS["left_eye_corners"]]
        right_eye_corners = landmarks[LANDMARK_POINTS["right_eye_corners"]]
        
        # Calculate eye centers
        left_eye_center = left_eye_corners.mean(dim=0)
        right_eye_center = right_eye_corners.mean(dim=0)
        
        # Calculate eye widths for normalization
        left_eye_width = torch.norm(left_eye_corners[1] - left_eye_corners[0])
        right_eye_width = torch.norm(right_eye_corners[1] - right_eye_corners[0])
        
        # Calculate normalized iris offset from eye center
        left_offset = (left_iris_center[0] - left_eye_center[0]) / (left_eye_width + 1e-6)
        right_offset = (right_iris_center[0] - right_eye_center[0]) / (right_eye_width + 1e-6)
        
        # Average both eyes with scaling for visibility
        eye_ball_x = (left_offset + right_offset) * 4.0
        
        return torch.clamp(eye_ball_x, -1, 1)
    
    def _calculate_eye_ball_y_iris(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate vertical eye gaze using MediaPipe iris landmarks.
        Uses dedicated iris center points for accurate tracking.
        
        Args:
            landmarks: 3D landmarks tensor (478, 3) with iris landmarks
            
        Returns:
            Eye ball Y position (-1 to 1)
        """
        # MediaPipe iris landmarks: 468-472 (left), 473-477 (right)
        left_iris_center = landmarks[468]  # Left iris center
        right_iris_center = landmarks[473]  # Right iris center
        
        # Use eye corners as stable reference
        left_eye_corners = landmarks[LANDMARK_POINTS["left_eye_corners"]]
        right_eye_corners = landmarks[LANDMARK_POINTS["right_eye_corners"]]
        
        # Eye corner Y average as baseline
        left_corner_y = left_eye_corners[:, 1].mean()
        right_corner_y = right_eye_corners[:, 1].mean()
        
        # Calculate iris offset from corner baseline
        left_offset = left_iris_center[1] - left_corner_y
        right_offset = right_iris_center[1] - right_corner_y
        
        # Average both eyes
        avg_offset = (left_offset + right_offset) / 2
        
        # Normalize by eye height
        left_eye_width = torch.norm(left_eye_corners[1] - left_eye_corners[0])
        right_eye_width = torch.norm(right_eye_corners[1] - right_eye_corners[0])
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        
        normalized_offset = avg_offset / (avg_eye_width * 0.6 + 1e-6)  # Eye height ~0.6 * width
        
        # Map to Live2D coordinates with appropriate scaling
        eye_ball_y = -normalized_offset * 3.0  # Negative for Y-up coordinate system
        
        return torch.clamp(eye_ball_y, -1, 1)