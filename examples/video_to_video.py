#!/usr/bin/env python3
"""
video_to_video.py - Modular Live2D Anime Generation Pipeline

This script demonstrates the modular pipeline for converting face videos 
to Live2D animations, with support for intermediate data export and loading.

Usage:
    python video_to_video.py [options]

Example:
    python video_to_video.py --input input.mp4 --output output.mp4
    python video_to_video.py --input input.mp4 --save-landmarks landmarks.json
    python video_to_video.py --input landmarks.json --output output.mp4
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Iterator, Tuple, Union
import torch
import numpy as np
import cv2
from tqdm import tqdm

# Import our package
sys.path.append(str(Path(__file__).parent.parent))
from live2d_anime_gen import (
    InsightFaceDetector,
    FaceMapper,
    Live2DRenderer,
    ParameterSmoother,
    VideoReader,
    VideoWriter,
    InputType,
    DataLoader,
    detect_input_type,
    DataExporter,
    Live2DParameters
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Modular pipeline for face video to Live2D conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input Types:
  The script automatically detects input type:
  - Video files (.mp4, .avi, etc.) → full pipeline
  - Landmarks file (.json, .pkl) → skip detection, start from mapping
  - Parameters file (.json, .pkl) → skip detection and mapping, only render

Examples:
  # Full pipeline
  python video_to_video.py --input input.mp4 --output output.mp4
  
  # Save intermediate data
  python video_to_video.py --input input.mp4 --save-landmarks landmarks.json --save-parameters params.json
  
  # Resume from landmarks
  python video_to_video.py --input landmarks.json --output output.mp4
  
  # Resume from parameters
  python video_to_video.py --input params.json --output output.mp4
  
  # Create landmark visualization for debugging
  python video_to_video.py --input input.mp4 --save-landmark-video landmarks.mp4
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input file (video, landmarks, or parameters)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output Live2D rendered video file path"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="live2d_models/haru_greeter_pro_jp/haru_greeter_t05.model3.json",
        help="Live2D model file path"
    )
    
    # Processing parameters
    parser.add_argument(
        "--smoothing", "-s",
        type=float,
        default=0.5,
        help="Parameter smoothing factor (0.0-1.0)"
    )
    
    parser.add_argument(
        "--det-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("WIDTH", "HEIGHT"),
        help="Face detection input size"
    )
    
    # Intermediate data saving
    parser.add_argument(
        "--save-landmarks",
        type=str,
        help="Save landmarks to file (json or pkl)"
    )
    
    parser.add_argument(
        "--save-parameters",
        type=str,
        help="Save Live2D parameters to file (json or pkl)"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug information"
    )
    
    # Output configuration
    parser.add_argument(
        "--render-resolution",
        type=int,
        nargs=2,
        default=[1280, 720],
        metavar=("WIDTH", "HEIGHT"),
        help="Live2D render resolution"
    )
    
    parser.add_argument(
        "--save-landmark-video",
        type=str,
        help="Save landmark visualization video"
    )
    
    parser.add_argument(
        "--show-rendering-window",
        action="store_true",
        help="Show Live2D rendering window during processing"
    )
    
    return parser.parse_args()












def main() -> None:
    """Main execution function - unified streaming processing."""
    args = parse_args()
    
    # Detect input type
    input_type = detect_input_type(args.input)
    print(f"Detected input type: {input_type}")
    
    show_progress = not args.no_progress
    
    try:
        # Unified processing pipeline for all input types
        process_pipeline(args, input_type, show_progress)
            
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    print("Pipeline completed successfully!")


def create_landmark_frame(landmarks: Optional[torch.Tensor], canvas_size: Tuple[int, int]) -> torch.Tensor:
    """Create a single landmark visualization frame."""
    
    height, width = canvas_size[1], canvas_size[0]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    if landmarks is not None:
        # Draw landmarks on frame
        lm_np = landmarks.cpu().numpy()  # Convert to numpy for OpenCV
        
        # Convert [0, 1] normalized coordinates to pixel coordinates
        lm_pixels = np.copy(lm_np)
        lm_pixels[:, 0] = lm_np[:, 0] * width   # x coordinates
        lm_pixels[:, 1] = lm_np[:, 1] * height  # y coordinates
        
        # Draw landmarks as colored numbers
        colors = {
            'jaw': (255, 255, 255),        # White
            'left_eye': (0, 255, 0),       # Green
            'right_eye': (0, 255, 0),      # Green
            'nose': (255, 0, 0),           # Blue
            'mouth': (0, 0, 255),          # Red
            'left_eyebrow': (255, 255, 0), # Cyan
            'right_eyebrow': (255, 255, 0), # Cyan
            'left_iris': (255, 0, 255),    # Magenta
            'right_iris': (255, 0, 255),   # Magenta
            'extra': (128, 128, 128),      # Gray
        }
        
        # Draw landmarks as colored numbers (no circles)
        from live2d_anime_gen.core.constants import LANDMARK_INDICES
        
        # First, determine color for each point
        point_colors = {}
        for part, indices in LANDMARK_INDICES.items():
            if part in colors:
                color = colors[part]
                for idx in indices:
                    point_colors[idx] = color
        
        # Draw all landmarks as colored numbers
        for idx in range(len(lm_pixels)):
            x, y = int(lm_pixels[idx, 0]), int(lm_pixels[idx, 1])
            if 0 <= x < width and 0 <= y < height:
                # Get color for this point, default to green if not assigned
                text_color = point_colors.get(idx, (0, 255, 0))
                
                # Draw the index number with colored text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 2  # Thicker text for better visibility
                
                # Get text size to center it on the landmark point
                text = str(idx)
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                
                # Center text on the landmark point
                text_x = x - text_width // 2
                text_y = y + text_height // 2
                
                # Make sure text stays within frame bounds
                if text_x < 0:
                    text_x = 0
                elif text_x + text_width > width:
                    text_x = width - text_width
                if text_y - text_height < 0:
                    text_y = text_height
                elif text_y > height:
                    text_y = height
                    
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    
    return torch.from_numpy(frame).cuda()


def process_pipeline(args: argparse.Namespace, input_type: InputType, show_progress: bool) -> None:
    """Unified processing pipeline for all input types."""
    # Initialize components
    detector = InsightFaceDetector(det_size=tuple(args.det_size))
    mapper = FaceMapper(smooth_factor=args.smoothing)
    smoother = ParameterSmoother(method="ema", alpha=args.smoothing)
    print("Initialized detector, mapper, and smoother")
    
    # Initialize renderer if output specified
    renderer: Optional[Live2DRenderer] = None
    if args.output:
        model_path = Path(args.model)
        if not model_path.exists():
            raise ValueError(f"Live2D model not found: {args.model}")
        
        renderer = Live2DRenderer(
            model_path=str(model_path),
            canvas_size=(args.render_resolution[0], args.render_resolution[1]),
            show=args.show_rendering_window
        )
        print("Initialized Live2D renderer")
    
    # Get input metadata and create data stream
    if input_type == InputType.VIDEO:
        # Video input - create frame stream
        reader = VideoReader(args.input)
        video_metadata = {
            'fps': reader.fps,
            'width': reader.width,
            'height': reader.height,
            'frame_count': reader.frame_count
        }
        
        def input_stream() -> Iterator[Tuple[Optional[torch.Tensor], Optional[Tuple[int, int]]]]:
            for frame in reader.read_frames():
                # Detection
                landmarks = detector.detect(frame)
                yield landmarks, (frame.shape[0], frame.shape[1])  # (landmarks, image_shape)
                
    elif input_type == InputType.LANDMARKS:
        # Landmarks input - create landmarks stream
        landmarks_iterator, video_metadata = DataLoader.load_landmarks(args.input)
        
        def input_stream() -> Iterator[Tuple[Optional[torch.Tensor], Optional[Tuple[int, int]]]]:
            for landmarks in landmarks_iterator:
                image_shape = (video_metadata['height'], video_metadata['width'])
                yield landmarks, image_shape
                
    elif input_type == InputType.PARAMETERS:
        # Parameters input - create parameters stream (skip mapping)
        parameters_iterator = DataLoader.load_parameters(args.input)
        video_metadata = DataLoader.get_metadata_from_parameters(args.input)
        # Parameters don't have video dimensions, use default resolution
        video_metadata['width'] = args.render_resolution[0]
        video_metadata['height'] = args.render_resolution[1]
        
        def input_stream() -> Iterator[Tuple[Union[Optional[torch.Tensor], Optional[Live2DParameters]], Optional[Tuple[int, int]]]]:
            for parameters in parameters_iterator:
                yield parameters, None  # Parameters don't need image_shape
    else:
        raise ValueError(f"Unsupported input type: {input_type}")
    
    print(f"Starting processing pipeline...")
    if input_type == InputType.VIDEO:
        print(f"  Video: {video_metadata['width']}x{video_metadata['height']}, {video_metadata['fps']} fps, {video_metadata['frame_count']} frames")
    else:
        print(f"  Input: {video_metadata['width']}x{video_metadata['height']}, {video_metadata['fps']} fps, {video_metadata['frame_count']} frames")
    
    # Create all output writers
    output_writer = VideoWriter(args.output, video_metadata['fps'], args.render_resolution) if args.output else None
    
    # For landmark video, use original video dimensions
    landmark_resolution = (video_metadata['width'], video_metadata['height'])
    landmark_writer = VideoWriter(args.save_landmark_video, video_metadata['fps'], landmark_resolution) if args.save_landmark_video else None
    
    # Create JSON writers for data export
    landmarks_json_writer = None
    parameters_json_writer = None
    
    if args.save_landmarks:
        landmarks_metadata = {
            'fps': video_metadata['fps'],
            'width': video_metadata['width'],
            'height': video_metadata['height']
        }
        landmarks_json_writer = DataExporter(args.save_landmarks, landmarks_metadata)
        landmarks_json_writer.__enter__()
        print(f"Writing landmarks to: {args.save_landmarks}")
    
    if args.save_parameters:
        parameters_metadata = {
            'fps': video_metadata['fps']
        }
        parameters_json_writer = DataExporter(args.save_parameters, parameters_metadata)
        parameters_json_writer.__enter__()
        print(f"Writing parameters to: {args.save_parameters}")
    
    # Process frames in streaming mode
    frame_count = 0
    total_frames = video_metadata['frame_count']
    # Create progress bar - shows processing speed even without total
    if show_progress:
        if total_frames is not None:
            progress_bar = tqdm(total=total_frames, desc="Processing frames", unit="frames")
        else:
            # Create tqdm without total - will show count and processing speed
            progress_bar = tqdm(desc="Processing", unit=" frames")
    else:
        progress_bar = None
    
    # Initialize with default parameters to maintain consistent frame count
    last_valid_parameters = Live2DParameters.create_default()
    
    try:
        for data_item, image_shape in input_stream():
            if input_type == InputType.PARAMETERS:
                # For parameters input, data_item is already parameters
                landmarks = None
                parameters = data_item
            else:
                # For video/landmarks input, data_item is landmarks
                landmarks = data_item
                parameters = None
                
                # Write landmarks to JSON
                if landmarks_json_writer and landmarks is not None:
                    if isinstance(landmarks, torch.Tensor):
                        landmarks_np = landmarks.cpu().numpy()
                    else:
                        landmarks_np = np.array(landmarks)
                    landmarks_json_writer.write_item(landmarks_np.tolist())
                elif landmarks_json_writer:
                    landmarks_json_writer.write_item(None)
                
                # Mapping and smoothing (only for video/landmarks input)
                if landmarks is not None and image_shape is not None:
                    parameters = mapper.map(landmarks, image_shape)
                    if parameters is not None:
                        parameters = smoother.smooth(parameters)
                        last_valid_parameters = parameters
            
            # Write parameters to JSON
            if parameters_json_writer:
                if parameters is not None:
                    param_dict = {}
                    for field_name in parameters.__dataclass_fields__:
                        field_value = getattr(parameters, field_name)
                        if isinstance(field_value, torch.Tensor):
                            param_dict[field_name] = field_value.cpu().item()
                        else:
                            param_dict[field_name] = field_value
                    parameters_json_writer.write_item(param_dict)
                else:
                    parameters_json_writer.write_item(None)
            
            # Render Live2D output (always render to maintain frame count)
            if output_writer:
                # Use current parameters if available, otherwise use last valid parameters
                render_params = parameters if parameters is not None else last_valid_parameters
                rendered_frame = renderer.render(render_params)
                output_writer.write_frame(rendered_frame)
            
            # Render landmark visualization (always render to maintain frame count)
            if landmark_writer:
                landmark_frame = create_landmark_frame(landmarks, landmark_resolution)
                landmark_writer.write_frame(landmark_frame)
            
            frame_count += 1
            if progress_bar is not None:
                progress_bar.update(1)
                
    finally:
        if input_type == InputType.VIDEO:
            reader.close()
        if output_writer:
            output_writer.close()
        if landmark_writer:
            landmark_writer.close()
        if landmarks_json_writer:
            landmarks_json_writer.__exit__(None, None, None)
            print("Landmarks saved")
        if parameters_json_writer:
            parameters_json_writer.__exit__(None, None, None)
            print("Parameters saved")
        if progress_bar is not None:
            progress_bar.close()
    
    print(f"Processing complete: {frame_count} frames processed")

if __name__ == "__main__":
    main()