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
    "eye": 0.1,      # Very quick response for eye blinking
    "eye_ball": 0.6, # More smoothing for eye ball movement to reduce jitter
    "mouth": 0.2,    # Quick response for mouth (reduced from 0.3)
    "head": 0.5,     # More smoothing for head rotation
    "body": 0.6,     # Most smoothing for body movements
}

# InsightFace 106-point landmark indices  
# Based on actual observation from generated landmark visualization
LANDMARK_INDICES = {
    "jaw": list(range(0, 33)),        # Face contour (0-32, 33 points)
    "right_eyebrow": list(range(97, 106)),  # Right eyebrow (97-105, 9 points)
    "left_eyebrow": list(range(43, 52)),    # Left eyebrow (43-51, 9 points)  
    "nose": list(range(72, 87)),            # Nose (72-86, 15 points)
    "right_eye": list(range(87, 97)),       # Right eye (87-96, 10 points)
    "left_eye": list(range(33, 43)),        # Left eye (33-42, 10 points)
    "mouth": list(range(52, 72)),           # Mouth (52-71, 20 points)
    "left_iris": [34, 38],                  # Left iris (34 and 38)
    "right_iris": [88, 92],                 # Right iris (88 and 92)
    "extra": [],                            # No extra points
}

# Specific landmark points for calculations
# All indices refer to absolute positions in the 106-point landmark array
LANDMARK_POINTS = {
    # Eye points
    "left_eye_corners": [39, 35],           # [inner_corner, outer_corner]
    "left_eye_top_points": [42, 40, 41], # Upper eyelid points, inner to outer
    "left_eye_bottom_points": [37, 33, 36], # Lower eyelid points, inner to outer
    
    "right_eye_corners": [89, 93],          # [inner_corner, outer_corner]
    "right_eye_top_points": [95, 94, 96], # Upper eyelid points, inner to outer
    "right_eye_bottom_points": [90, 87, 91], # Lower eyelid points, inner to outer
    
    # Mouth points
    "mouth_corners": [52, 61],              # [left_corner, right_corner]
    "mouth_center_points": [71, 53],        # [top_center, bottom_center]
    
    # Face structure points
    "nose_tip": 86,                         # Nose tip
    "face_left_points": [1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0],   # Left face contour for yaw, top to bottom
    "face_right_points": [17, 25, 26, 27, 28, 29, 30, 31, 32, 18, 19, 20, 21, 22, 23, 24, 0], # Right face contour for yaw, top to bottom
    
    # Iris points (if available)
    "left_iris": [34, 38],                  # Left iris (34 and 38)
    "right_iris": [88, 92],                 # Right iris (88 and 92)
}