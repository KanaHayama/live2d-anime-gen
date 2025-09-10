"""Constants and default values for Live2D parameters."""

# Live2D v3 standard parameter names
PARAM_ANGLE_X = "ParamAngleX"
PARAM_ANGLE_Y = "ParamAngleY"
PARAM_ANGLE_Z = "ParamAngleZ"
PARAM_EYE_L_OPEN = "ParamEyeLOpen"
PARAM_EYE_R_OPEN = "ParamEyeROpen"
PARAM_EYE_BALL_X = "ParamEyeBallX"
PARAM_EYE_BALL_Y = "ParamEyeBallY"
PARAM_BROW_L_Y = "ParamBrowLY"
PARAM_BROW_R_Y = "ParamBrowRY"
PARAM_MOUTH_OPEN_Y = "ParamMouthOpenY"
PARAM_MOUTH_FORM = "ParamMouthForm"
PARAM_BODY_ANGLE_X = "ParamBodyAngleX"
PARAM_BODY_ANGLE_Y = "ParamBodyAngleY"
PARAM_BODY_ANGLE_Z = "ParamBodyAngleZ"

# Parameter value ranges
PARAM_RANGES = {
    PARAM_ANGLE_X: (-30.0, 30.0),
    PARAM_ANGLE_Y: (-30.0, 30.0),
    PARAM_ANGLE_Z: (-30.0, 30.0),
    PARAM_EYE_L_OPEN: (0.0, 1.0),
    PARAM_EYE_R_OPEN: (0.0, 1.0),
    PARAM_EYE_BALL_X: (-1.0, 1.0),
    PARAM_EYE_BALL_Y: (-1.0, 1.0),
    PARAM_BROW_L_Y: (-1.0, 1.0),
    PARAM_BROW_R_Y: (-1.0, 1.0),
    PARAM_MOUTH_OPEN_Y: (0.0, 1.0),
    PARAM_MOUTH_FORM: (-1.0, 1.0),
    PARAM_BODY_ANGLE_X: (-10.0, 10.0),
    PARAM_BODY_ANGLE_Y: (-10.0, 10.0),
    PARAM_BODY_ANGLE_Z: (-10.0, 10.0),
}

# Default parameter values
DEFAULT_VALUES = {
    PARAM_ANGLE_X: 0.0,
    PARAM_ANGLE_Y: 0.0,
    PARAM_ANGLE_Z: 0.0,
    PARAM_EYE_L_OPEN: 1.0,
    PARAM_EYE_R_OPEN: 1.0,
    PARAM_EYE_BALL_X: 0.0,
    PARAM_EYE_BALL_Y: 0.0,
    PARAM_BROW_L_Y: 0.0,
    PARAM_BROW_R_Y: 0.0,
    PARAM_MOUTH_OPEN_Y: 0.0,
    PARAM_MOUTH_FORM: 0.0,
    PARAM_BODY_ANGLE_X: 0.0,
    PARAM_BODY_ANGLE_Y: 0.0,
    PARAM_BODY_ANGLE_Z: 0.0,
}

# Smoothing factors for different parameter types
SMOOTHING_FACTORS = {
    "eye": 0.03,
    "eye_ball": 0.05,
    "mouth": 0.05,
    "head": 0.05,
    "body": 0.4,
}

# MediaPipe 478-point landmark indices (468 face + 10 iris)
# Based on MediaPipe Face Mesh with refine_landmarks=True
# Indices from official MediaPipe face_mesh_connections.py
LANDMARK_INDICES = {
    "face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],  # Complete 36-point face boundary
    "left_eyebrow": [46, 52, 53, 63, 65, 66, 70, 105, 107],  # FACEMESH_RIGHT_EYEBROW (right from viewer's perspective)
    "right_eyebrow": [276, 282, 283, 293, 295, 296, 300, 334, 336],  # FACEMESH_LEFT_EYEBROW (left from viewer's perspective)
    "left_eye": [7, 33, 133, 144, 145, 153, 154, 155, 161, 163, 246],  # FACEMESH_RIGHT_EYE (right from viewer's perspective)
    "right_eye": [249, 262, 263, 373, 374, 380, 381, 382, 386, 387, 388, 390, 466],  # FACEMESH_LEFT_EYE (left from viewer's perspective)
    "lips": [
        # Upper outer lip
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        # Lower outer lip
        146, 91, 181, 84, 17, 314, 405, 321, 375,
        # Upper inner lip
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
        # Lower inner lip
        95, 88, 178, 87, 14, 317, 402, 318, 324
    ],  # Complete 40-point lips (outer + inner contours)
    "left_iris": list(range(468, 473)),           # Left iris: center + 4 contour points
    "right_iris": list(range(473, 478)),          # Right iris: center + 4 contour points
}

# Specific landmark points for calculations
# All indices refer to absolute positions in the 478-point landmark array
LANDMARK_POINTS = {
    # Eye points for MediaPipe (using key eye contour points)
    "left_eye_corners": [33, 133],              # [inner_corner, outer_corner] - Left eye
    "left_eye_top_points": [159, 145, 158],     # Upper eyelid key points
    "left_eye_bottom_points": [144, 163, 7],    # Lower eyelid key points
    
    "right_eye_corners": [362, 263],            # [inner_corner, outer_corner] - Right eye
    "right_eye_top_points": [386, 374, 385],    # Upper eyelid key points
    "right_eye_bottom_points": [373, 390, 249], # Lower eyelid key points
    
    # Mouth points
    "mouth_corners": [61, 291],                 # [left_corner, right_corner]
    "mouth_center_points": [13, 14],            # [upper_lip_center, lower_lip_center]
    
    # Face structure points for 3D pose estimation
    "nose_tip": 1,                              # Nose tip (kept for 3D pose calculations)
    "face_left_points": [10, 21, 54, 58, 93, 127, 132, 136, 148, 149, 150],  # Left side of face oval
    "face_right_points": [356, 361, 365, 377, 378, 379, 400, 288, 297, 323, 332, 338, 350],  # Right side of face oval
    
    # Iris points for pupil tracking (MediaPipe specific)
    "left_iris": list(range(468, 473)),         # Left iris: [center, right, top, left, bottom]
    "right_iris": list(range(473, 478)),        # Right iris: [center, right, top, left, bottom]
    
    # Additional 3D reference points for head pose
    "forehead_center": 9,                       # Forehead center point
    "chin_tip": 152,                            # Chin bottom point
    "left_temple": 54,                          # Left temple
    "right_temple": 284,                        # Right temple
}