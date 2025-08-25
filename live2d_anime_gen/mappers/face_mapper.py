"""Face landmark to Live2D parameter mapper for 106-point InsightFace model."""

from typing import Tuple
import torch
import torch.nn.functional as F

from ..core.base_mapper import BaseLandmarkMapper
from ..core.types import Live2DParameters
from ..core.constants import LANDMARK_INDICES, LANDMARK_POINTS


class FaceMapper(BaseLandmarkMapper):
    """
    Maps 106 facial landmarks from InsightFace to Live2D parameters.
    
    NOTE: All parameter mappings and thresholds in this class have been calibrated
    specifically for user 'kana' -> Live2D model 'haru_greeter_pro_jp'.
    The calibration is based on analysis of kana's facial expressions and movements
    to ensure optimal animation quality for the haru model.
    
    If using different users or models, recalibration may be needed.
    """
    
    def __init__(self, smooth_factor: float = 0.5):
        """
        Initialize the face mapper.
        
        Args:
            smooth_factor: Temporal smoothing factor (0-1)
        """
        super().__init__(smooth_factor)
    
    def map(self, 
            landmarks: torch.Tensor,
            image_shape: Tuple[int, int]) -> Live2DParameters:
        """
        Map 106 facial landmarks to Live2D parameters.
        
        Args:
            landmarks: 106 landmark points as torch tensor (106, 2)
            image_shape: Image dimensions (height, width)
            
        Returns:
            Live2D parameters
        """
        # Normalize landmarks to [-1, 1]
        landmarks = self.preprocess_landmarks(landmarks, image_shape)
        
        # Calculate all parameters
        params = Live2DParameters(
            eye_l_open=self._calculate_eye_openness(landmarks, "left"),
            eye_r_open=self._calculate_eye_openness(landmarks, "right"),
            mouth_open_y=self._calculate_mouth_openness(landmarks),
            mouth_form=self._calculate_mouth_form(landmarks),
            angle_x=self._calculate_head_yaw(landmarks),
            angle_y=self._calculate_head_pitch(landmarks),
            angle_z=self._calculate_head_roll(landmarks),
            eye_ball_x=self._calculate_eye_ball_x(landmarks),
            eye_ball_y=self._calculate_eye_ball_y(landmarks),
            brow_l_y=self._calculate_eyebrow_height(landmarks, "left"),
            brow_r_y=self._calculate_eyebrow_height(landmarks, "right"),
        )
        
        # Apply postprocessing (clamping and smoothing)
        return self.postprocess_parameters(params)
    
    def _calculate_eye_openness(self, landmarks: torch.Tensor, side: str) -> torch.Tensor:
        """
        Calculate eye openness using aspect ratio (similar to live2d-py method).
        
        Args:
            landmarks: Normalized landmarks
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
        
        # Get corner points for horizontal distance
        inner_corner = landmarks[corners[0]]
        outer_corner = landmarks[corners[1]]
        horizontal_dist = torch.norm(outer_corner - inner_corner)
        
        # Get top and bottom eyelid points
        top_points = landmarks[top_indices]
        bottom_points = landmarks[bottom_indices]
        
        # Calculate vertical distances at multiple points
        vertical_distances = []
        min_len = min(len(top_points), len(bottom_points))
        for i in range(min_len):
            vertical_distances.append(torch.norm(top_points[i] - bottom_points[i]))
        
        if vertical_distances:
            # Average vertical distance
            vertical_dist = torch.mean(torch.stack(vertical_distances))
        else:
            vertical_dist = torch.tensor(0.01, device=landmarks.device)
        
        # Eye Aspect Ratio (EAR)
        ear = vertical_dist / (horizontal_dist + 1e-6)
        
        # Calibrate EAR to binary-like [0,1] range for proper blinking
        # Live2D models expect 0=closed, 1=open for natural blinking
        # Closed eye: EAR ~0.05-0.1, Open eye: EAR ~0.15-0.25
        threshold = 0.12  # Blinking threshold
        if ear < threshold:
            # Eye is closed or nearly closed
            openness = torch.clamp((ear - 0.05) / (threshold - 0.05), 0, 1) * 0.1
        else:
            # Eye is open - use steeper curve to approach 1.0
            openness = 0.1 + torch.clamp((ear - threshold) / 0.08, 0, 1) * 0.9
        
        # Ensure values are close to 0 or 1 for natural blinking
        openness = torch.clamp(openness, 0, 1)
        
        return openness
    
    def _calculate_mouth_openness(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate mouth vertical openness using aspect ratio (similar to live2d-py method).
        
        Args:
            landmarks: Normalized landmarks
            
        Returns:
            Mouth openness value (0-1)
        """
        # Get mouth corner and center points
        corners = LANDMARK_POINTS["mouth_corners"]
        centers = LANDMARK_POINTS["mouth_center_points"]
        
        left_corner = landmarks[corners[0]]
        right_corner = landmarks[corners[1]]
        upper_lip_center = landmarks[centers[0]]
        lower_lip_center = landmarks[centers[1]]
        
        # Calculate horizontal distance (mouth width)
        horizontal_dist = torch.norm(right_corner - left_corner)
        
        # Calculate vertical distance (mouth height)
        vertical_dist = torch.abs(lower_lip_center[1] - upper_lip_center[1])
        
        # Mouth Aspect Ratio (MAR) - similar to Eye Aspect Ratio
        mar = vertical_dist / (horizontal_dist + 1e-6)
        
        # Calibrate MAR to [0,1.0] range for more dramatic mouth opening
        # Based on complete video data analysis:
        # Closed mouth: MAR ≤ 0.600 (25th percentile), Max opening: MAR = 2.205
        # Increased range to 0-1.0 for more visible mouth movement
        threshold = 0.60  # 25th percentile - ensures mouth closes during quiet moments
        max_mar = 2.21    # Slightly above observed maximum for safety margin
        
        if mar <= threshold:
            openness = torch.tensor(0.0, device=landmarks.device)  # Mouth is fully closed
        else:
            # Scale from threshold to maximum observed value to 0-2.0 range for testing
            # (mar - 0.60) / (2.21 - 0.60) * 2.0 - trying larger range to see if model responds
            openness = torch.clamp((mar - threshold) / (max_mar - threshold) * 2.0, 0, 2.0)
        
        return openness
    
    def _calculate_mouth_form(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate mouth form (smile/frown) using corner relative position.
        
        Args:
            landmarks: Normalized landmarks
            
        Returns:
            Mouth form value (-1 to 1, negative=frown, positive=smile)
        """
        # Get mouth corner and center points
        corners = LANDMARK_POINTS["mouth_corners"]
        centers = LANDMARK_POINTS["mouth_center_points"]
        
        left_corner = landmarks[corners[0]]
        right_corner = landmarks[corners[1]]
        upper_lip_center = landmarks[centers[0]]
        lower_lip_center = landmarks[centers[1]]
        mouth_center = (upper_lip_center + lower_lip_center) / 2
        
        # Calculate average corner height relative to mouth center
        left_corner_offset = left_corner[1] - mouth_center[1]
        right_corner_offset = right_corner[1] - mouth_center[1]
        corner_height = (left_corner_offset + right_corner_offset) / 2
        
        # In normalized coordinates, negative Y is up
        # Smile: corners go up (negative Y), Frown: corners go down (positive Y)
        raw_form = -corner_height * 5.0
        
        # Bias correction based on data analysis: mean form = 0.021
        # This centers the neutral expression around 0
        bias_corrected_form = raw_form - 0.021
        
        # Enhanced mapping based on actual data distribution:
        # Strong frown: <= -0.037, Strong smile: >= 0.137
        # Using percentile-based thresholds for better expression recognition
        
        # Define expression thresholds from data analysis
        strong_frown_threshold = -0.037  # From data: minimum observed
        mild_frown_threshold = 0.005     # 25th percentile
        mild_smile_threshold = 0.032     # 75th percentile  
        strong_smile_threshold = 0.137   # Strong smile threshold from data
        
        if bias_corrected_form <= strong_frown_threshold:
            # Strong frown: map to Live2D range [0.0, 0.3]
            intensity = torch.clamp((bias_corrected_form - strong_frown_threshold) / -0.02, 0, 1)
            form = 0.3 * (1 - intensity)  # 0.0 to 0.3
        elif bias_corrected_form <= mild_frown_threshold:
            # Mild frown: map to [0.3, 0.8]
            range_size = mild_frown_threshold - strong_frown_threshold
            if range_size > 0:
                intensity = (bias_corrected_form - strong_frown_threshold) / range_size
                form = 0.3 + intensity * 0.5  # 0.3 to 0.8
            else:
                form = 0.55  # fallback
        elif bias_corrected_form <= mild_smile_threshold:
            # Neutral range: map to [0.8, 1.2] with 1.0 as neutral
            range_size = mild_smile_threshold - mild_frown_threshold
            if range_size > 0:
                intensity = (bias_corrected_form - mild_frown_threshold) / range_size
                form = 0.8 + intensity * 0.4  # 0.8 to 1.2
            else:
                form = 1.0  # neutral fallback
        elif bias_corrected_form <= strong_smile_threshold:
            # Mild smile: map to [1.2, 1.7]
            range_size = strong_smile_threshold - mild_smile_threshold
            if range_size > 0:
                intensity = (bias_corrected_form - mild_smile_threshold) / range_size
                form = 1.2 + intensity * 0.5  # 1.2 to 1.7
            else:
                form = 1.45  # fallback
        else:
            # Strong smile: map to [1.7, 2.0]
            intensity = torch.clamp((bias_corrected_form - strong_smile_threshold) / 0.05, 0, 1)
            form = 1.7 + intensity * 0.3  # 1.7 to 2.0
        
        return torch.clamp(form, 0, 2)
    
    def _calculate_head_yaw(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate head yaw (left-right rotation) using nose and face asymmetry.
        
        Args:
            landmarks: Normalized landmarks
            
        Returns:
            Yaw angle in degrees
        """
        # Get nose tip for reference
        nose_tip = landmarks[LANDMARK_POINTS["nose_tip"]]
        
        # Get face contour points for asymmetry calculation
        left_face_indices = LANDMARK_POINTS["face_left_points"]
        right_face_indices = LANDMARK_POINTS["face_right_points"]
        
        left_face = landmarks[left_face_indices]
        right_face = landmarks[right_face_indices]
        
        # Calculate perpendicular distances from nose to left and right sides
        left_distances = torch.norm(left_face - nose_tip.unsqueeze(0), dim=1)
        right_distances = torch.norm(right_face - nose_tip.unsqueeze(0), dim=1)
        
        perp_left = torch.mean(left_distances)
        perp_right = torch.mean(right_distances)
        
        # Calculate yaw angle using asymmetry ratio (similar to live2d-py)
        # When face turns left, perp_left decreases, perp_right increases
        yaw_ratio = (perp_right - perp_left) / (perp_right + perp_left + 1e-6)
        
        # Convert ratio to degrees with appropriate scaling
        yaw = yaw_ratio * 30.0  # Scale to ±30 degrees range
        
        return torch.clamp(yaw, -30, 30)
    
    def _calculate_head_pitch(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate head pitch (up-down rotation).
        
        Args:
            landmarks: Normalized landmarks
            
        Returns:
            Pitch angle in degrees
        """
        # Use vertical position of nose relative to eyes
        nose_tip = landmarks[LANDMARK_POINTS["nose_tip"]]
        
        # Get eye centers using corner points
        left_eye_corners = LANDMARK_POINTS["left_eye_corners"]
        right_eye_corners = LANDMARK_POINTS["right_eye_corners"]
        
        left_eye_center = (landmarks[left_eye_corners[0]] + landmarks[left_eye_corners[1]]) / 2
        right_eye_center = (landmarks[right_eye_corners[0]] + landmarks[right_eye_corners[1]]) / 2
        eye_center = (left_eye_center + right_eye_center) / 2
        
        # Calculate vertical offset
        vertical_offset = nose_tip[1] - eye_center[1]
        
        # Convert to angle (calibrated approximation)
        # Vertical offset is typically -0.2 to 0.2 in normalized coordinates
        pitch = vertical_offset * 60.0
        
        return torch.clamp(pitch, -30, 30)
    
    def _calculate_head_roll(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate head roll (tilt).
        
        Args:
            landmarks: Normalized landmarks
            
        Returns:
            Roll angle in degrees
        """
        # Use eye line angle
        left_eye_corners = LANDMARK_POINTS["left_eye_corners"]
        right_eye_corners = LANDMARK_POINTS["right_eye_corners"]
        
        left_eye_center = (landmarks[left_eye_corners[0]] + landmarks[left_eye_corners[1]]) / 2
        right_eye_center = (landmarks[right_eye_corners[0]] + landmarks[right_eye_corners[1]]) / 2
        
        # Calculate angle from horizontal
        delta_y = right_eye_center[1] - left_eye_center[1]
        delta_x = right_eye_center[0] - left_eye_center[0]
        
        # Angle in radians, then convert to degrees
        roll = torch.atan2(delta_y, delta_x) * 180.0 / torch.pi
        
        # The roll should be much smaller, typically within ±15 degrees
        return torch.clamp(roll, -15, 15)
    
    def _calculate_eye_ball_x(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate horizontal eye ball position.
        
        Args:
            landmarks: Normalized landmarks
            
        Returns:
            Eye ball X position (-1 to 1)
        """
        # Check if iris landmarks are available
        left_iris_indices = LANDMARK_POINTS["left_iris"]
        right_iris_indices = LANDMARK_POINTS["right_iris"]
        
        if all(idx < len(landmarks) for idx in left_iris_indices + right_iris_indices):
            # Calculate average position of iris points
            left_iris = (landmarks[left_iris_indices[0]] + landmarks[left_iris_indices[1]]) / 2
            right_iris = (landmarks[right_iris_indices[0]] + landmarks[right_iris_indices[1]]) / 2
            
            # Get eye centers using corner points
            left_eye_corners = LANDMARK_POINTS["left_eye_corners"]
            right_eye_corners = LANDMARK_POINTS["right_eye_corners"]
            
            left_eye_center = (landmarks[left_eye_corners[0]] + landmarks[left_eye_corners[1]]) / 2
            right_eye_center = (landmarks[right_eye_corners[0]] + landmarks[right_eye_corners[1]]) / 2
            
            # Calculate iris offset from eye center
            left_offset = (left_iris[0] - left_eye_center[0]) * 3.0
            right_offset = (right_iris[0] - right_eye_center[0]) * 3.0
            
            # Average both eyes
            eye_ball_x = (left_offset + right_offset) / 2
        else:
            # Fallback: use head yaw as approximation
            eye_ball_x = self._calculate_head_yaw(landmarks) / 30.0
        
        return torch.clamp(eye_ball_x, -1, 1)
    
    def _calculate_eye_ball_y(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate vertical eye ball position.
        
        Args:
            landmarks: Normalized landmarks
            
        Returns:
            Eye ball Y position (-1 to 1)
        """
        # Check if iris landmarks are available
        left_iris_indices = LANDMARK_POINTS["left_iris"]
        right_iris_indices = LANDMARK_POINTS["right_iris"]
        
        if all(idx < len(landmarks) for idx in left_iris_indices + right_iris_indices):
            # Calculate average position of iris points
            left_iris = (landmarks[left_iris_indices[0]] + landmarks[left_iris_indices[1]]) / 2
            right_iris = (landmarks[right_iris_indices[0]] + landmarks[right_iris_indices[1]]) / 2
            
            # Get eye centers using corner points
            left_eye_corners = LANDMARK_POINTS["left_eye_corners"]
            right_eye_corners = LANDMARK_POINTS["right_eye_corners"]
            
            left_eye_center = (landmarks[left_eye_corners[0]] + landmarks[left_eye_corners[1]]) / 2
            right_eye_center = (landmarks[right_eye_corners[0]] + landmarks[right_eye_corners[1]]) / 2
            
            # Calculate iris offset from eye center
            left_offset = (left_iris[1] - left_eye_center[1]) * 3.0
            right_offset = (right_iris[1] - right_eye_center[1]) * 3.0
            
            # Average both eyes
            eye_ball_y = (left_offset + right_offset) / 2
        else:
            # Fallback: use head pitch as approximation
            eye_ball_y = -self._calculate_head_pitch(landmarks) / 30.0
        
        return torch.clamp(eye_ball_y, -1, 1)
    
    def _calculate_eyebrow_height(self, landmarks: torch.Tensor, side: str) -> torch.Tensor:
        """
        Calculate eyebrow height/position.
        
        Args:
            landmarks: Normalized landmarks
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
        
        # Get average positions
        brow_center = torch.mean(landmarks[brow_indices], dim=0)
        eye_center = torch.mean(landmarks[eye_indices], dim=0)
        
        # Calculate vertical distance (normalized)
        vertical_dist = (eye_center[1] - brow_center[1]) * 5.0
        
        # Map to [-1, 1] where 0 is neutral
        return torch.clamp(vertical_dist - 0.5, -1, 1)