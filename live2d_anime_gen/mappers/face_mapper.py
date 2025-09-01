"""Face landmark to Live2D parameter mapper with independent parameter calculation."""

from typing import Tuple
import torch

from ..core.base_mapper import BaseLandmarkMapper
from ..core.types import Live2DParameters
from ..core.constants import LANDMARK_INDICES, LANDMARK_POINTS


class FaceMapper(BaseLandmarkMapper):
    """
    Maps 106 facial landmarks from InsightFace to Live2D parameters.
    
    Key design principle: Each parameter group (eyes, mouth, head pose, eye gaze)
    is calculated independently to avoid unwanted coupling.
    
    NOTE: All parameter mappings and thresholds in this class have been calibrated
    specifically for user 'kana' -> Live2D model 'haru_greeter_pro_jp'.
    The calibration is based on analysis of kana's facial expressions and movements
    to ensure optimal animation quality for the haru model.
    """
    
    def __init__(self, smooth_factor: float = 0.5):
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
            landmarks: torch.Tensor,
            image_shape: Tuple[int, int]) -> Live2DParameters:
        """
        Map 106 facial landmarks to Live2D parameters.
        
        Each parameter is calculated independently:
        - Eye openness: Based on eye aspect ratio only
        - Mouth: Based on mouth shape only  
        - Head pose: Based on face geometry
        - Eye gaze: Based on iris position (if available)
        
        Args:
            landmarks: 106 landmark points as torch tensor (106, 2) in pixel coords
            image_shape: Image dimensions (height, width)
            
        Returns:
            Live2D parameters
        """
        # Landmarks from InsightFace are already normalized to [0, 1]
        # Keep original landmarks for pixel-space calculations (roll angle)
        landmarks_pixel = landmarks.clone() * torch.tensor([image_shape[1], image_shape[0]], device=landmarks.device)
        
        # Use normalized landmarks directly
        landmarks_norm = landmarks.clone()
        
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
            
            # Head pose - calculated from face geometry
            angle_x=self._calculate_head_yaw(landmarks_norm),
            angle_y=self._calculate_head_pitch(landmarks_norm),
            angle_z=self._calculate_head_roll(landmarks_pixel, image_shape),
            
            # Eye gaze - only depends on iris position if available
            eye_ball_x=self._calculate_eye_ball_x(landmarks_norm),
            eye_ball_y=self._calculate_eye_ball_y(landmarks_norm),
            
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
            openness = torch.pow(t, 0.8)
        
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