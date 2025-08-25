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
from typing import Optional, List, Iterator, Tuple
import torch
import numpy as np
import cv2

# Import our package
sys.path.append(str(Path(__file__).parent.parent))
from live2d_anime_gen import (
    InsightFaceDetector,
    FaceMapper,
    Live2DRenderer,
    ParameterSmoother,
    VideoReader,
    VideoWriter,
    Pipeline,
    DataCollector,
    DataExporter,
    DataLoader,
    detect_input_type,
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
        help="Output video file path"
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
        default=0.7,
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
        "--fps",
        type=int,
        default=30,
        help="Output video FPS"
    )
    
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[1280, 720],
        metavar=("WIDTH", "HEIGHT"),
        help="Output video resolution"
    )
    
    parser.add_argument(
        "--save-landmark-video",
        type=str,
        help="Save landmark visualization video"
    )
    
    return parser.parse_args()


def process_video_to_landmarks(
    video_path: str,
    detector: InsightFaceDetector,
    show_progress: bool = True
) -> Tuple[List[Optional[torch.Tensor]], VideoReader]:
    """
    Process video to extract landmarks.
    
    Returns:
        Tuple of (landmarks list, video reader for metadata)
    """
    print("📹 Processing video to extract landmarks...")
    
    reader = VideoReader(video_path)
    print(f"  Video: {reader.width}x{reader.height}, {reader.fps} fps, {reader.frame_count} frames")
    
    # Create pipeline and collector
    pipeline = Pipeline()
    collector = DataCollector()
    
    # Process frames
    frames = reader.read_frames(show_progress=show_progress)
    landmarks_stream = pipeline.detect_landmarks(frames, detector)
    
    # Collect landmarks
    collected_stream = collector.collect(
        ((frame, lm, None, None) for frame, lm in landmarks_stream),
        collect_landmarks=True
    )
    
    # Consume the stream
    for _ in collected_stream:
        pass
    
    print(f"✓ Extracted landmarks from {len(collector.landmarks)} frames")
    
    # Count successful detections
    valid_count = sum(1 for lm in collector.landmarks if lm is not None)
    print(f"  Valid detections: {valid_count}/{len(collector.landmarks)} ({valid_count*100/len(collector.landmarks):.1f}%)")
    
    return collector.landmarks, reader


def process_landmarks_to_parameters(
    landmarks: List[Optional[torch.Tensor]],
    mapper: FaceMapper,
    smoother: ParameterSmoother,
    image_shape: Tuple[int, int]
) -> List[Optional[Live2DParameters]]:
    """
    Process landmarks to Live2D parameters.
    """
    print("🎯 Mapping landmarks to Live2D parameters...")
    
    parameters = []
    for lm in landmarks:
        if lm is not None:
            params = mapper.map(lm, image_shape)
            params = smoother.smooth(params)
        else:
            params = None
        parameters.append(params)
    
    valid_count = sum(1 for p in parameters if p is not None)
    print(f"✓ Generated parameters for {valid_count} frames")
    
    return parameters


def process_parameters_to_video(
    parameters: List[Optional[Live2DParameters]],
    renderer: Live2DRenderer,
    output_path: str,
    fps: float,
    show_progress: bool = True
):
    """
    Render Live2D parameters to video.
    """
    print("🎨 Rendering Live2D animation...")
    
    # Initialize renderer
    renderer.initialize()
    
    # Create video writer
    width, height = renderer.canvas_size
    writer = VideoWriter(output_path, fps, (width, height))
    
    # Render frames
    def render_frames():
        for i, params in enumerate(parameters):
            if params is not None:
                frame = renderer.render(params)
                # Convert RGB to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                # Create black frame for missing parameters
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            yield frame
    
    writer.write_frames(
        render_frames(),
        total=len(parameters),
        show_progress=show_progress
    )
    
    writer.close()
    print(f"✓ Saved video to: {output_path}")


def process_landmarks_to_video(
    landmarks: List[Optional[torch.Tensor]],
    output_path: str,
    fps: float,
    canvas_size: Tuple[int, int] = (1280, 720),
    show_progress: bool = True
):
    """
    Render landmarks visualization video on black background.
    Expects landmarks in [0,1] normalized coordinates.
    """
    # Create video writer
    writer = VideoWriter(output_path, fps, canvas_size)
    
    def render_landmark_frames():
        for i, lm in enumerate(landmarks):
            # Create black frame
            height, width = canvas_size[1], canvas_size[0]
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            if lm is not None:
                # Draw landmarks on frame
                lm_np = lm.cpu().numpy()  # Convert to numpy for OpenCV
                
                # Convert [0, 1] normalized coordinates to pixel coordinates
                lm_pixels = np.copy(lm_np)
                lm_pixels[:, 0] = lm_np[:, 0] * width   # x coordinates
                lm_pixels[:, 1] = lm_np[:, 1] * height  # y coordinates
                
                # Draw different facial parts in different colors
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
                
            
            yield frame
    
    writer.write_frames(
        render_landmark_frames(),
        total=len(landmarks),
        show_progress=show_progress
    )
    
    writer.close()
    print(f"✓ Saved landmark video to: {output_path}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Detect input type
    input_type = detect_input_type(args.input)
    print(f"🔍 Detected input type: {input_type}")
    
    show_progress = not args.no_progress
    
    # Initialize components as needed
    detector = None
    mapper = None
    smoother = None
    renderer = None
    
    # Process based on input type
    landmarks = None
    parameters = None
    video_metadata = None
    
    try:
        # Step 1: Get landmarks (from video or file)
        if input_type == 'video':
            # Initialize detector
            detector = InsightFaceDetector(det_size=tuple(args.det_size))
            print("✓ Initialized InsightFace detector")
            
            # Process video to landmarks
            landmarks, video_reader = process_video_to_landmarks(
                args.input,
                detector,
                show_progress
            )
            video_metadata = {
                'fps': video_reader.fps,
                'width': video_reader.width,
                'height': video_reader.height
            }
            video_reader.close()
            
        elif input_type == 'landmarks':
            # Load landmarks from file
            print("📁 Loading landmarks from file...")
            landmarks, video_dims = DataLoader.load_landmarks(args.input)
            print(f"✓ Loaded {len(landmarks)} landmark frames")
            
            # Use video dimensions from file if available
            if video_dims:
                width, height = video_dims
                print(f"  Original video dimensions: {width}x{height}")
                video_metadata = {
                    'fps': args.fps,
                    'width': width,
                    'height': height
                }
            else:
                # Fallback to default resolution
                print("  No video dimensions found, using default resolution")
                video_metadata = {
                    'fps': args.fps,
                    'width': args.resolution[0],
                    'height': args.resolution[1]
                }
        
        # Save landmarks if requested
        if landmarks and args.save_landmarks:
            print(f"💾 Saving landmarks to: {args.save_landmarks}")
            DataExporter.export_landmarks(
                landmarks, 
                args.save_landmarks, 
                'json',
                video_metadata.get('width') if video_metadata else None,
                video_metadata.get('height') if video_metadata else None
            )
            print("✓ Landmarks saved")
        
        # Create landmark visualization video if requested (right after landmarks are ready)
        if landmarks and args.save_landmark_video:
            print(f"🎯 Creating landmark visualization video...")
            # Use original video dimensions for landmark visualization
            canvas_width = video_metadata.get('width', args.resolution[0])
            canvas_height = video_metadata.get('height', args.resolution[1])
            
            # Landmarks are always in [0,1] range at this point:
            # - From video: normalized by detector during detection  
            # - From JSON: loaded as [0,1] normalized coordinates
            process_landmarks_to_video(
                landmarks,
                args.save_landmark_video,
                video_metadata.get('fps', args.fps),
                (canvas_width, canvas_height),
                show_progress
            )
        
        # Step 2: Get parameters (from landmarks or file)
        if input_type in ['video', 'landmarks']:
            # Initialize mapper and smoother
            mapper = FaceMapper(smooth_factor=args.smoothing)
            smoother = ParameterSmoother(method="ema", alpha=args.smoothing)
            print("✓ Initialized mapper and smoother")
            
            # Process landmarks to parameters
            parameters = process_landmarks_to_parameters(
                landmarks,
                mapper,
                smoother,
                (video_metadata['height'], video_metadata['width'])
            )
            
        elif input_type == 'parameters':
            # Load parameters from file
            print("📁 Loading parameters from file...")
            parameters = DataLoader.load_parameters(args.input)
            print(f"✓ Loaded {len(parameters)} parameter frames")
            
            # Use provided FPS for output
            video_metadata = {'fps': args.fps}
        
        # Save parameters if requested
        if parameters and args.save_parameters:
            print(f"💾 Saving parameters to: {args.save_parameters}")
            format = 'pickle' if args.save_parameters.endswith('.pkl') else 'json'
            DataExporter.export_parameters(parameters, args.save_parameters, format)
            print("✓ Parameters saved")
        
        # Step 3: Render to video if output specified
        if args.output and parameters:
            # Initialize renderer
            model_path = Path(args.model)
            if not model_path.exists():
                print(f"❌ Error: Live2D model not found: {args.model}")
                sys.exit(1)
            
            renderer = Live2DRenderer(
                model_path=str(model_path),
                canvas_size=(args.resolution[0], args.resolution[1])
            )
            print("✓ Initialized Live2D renderer")
            
            # Render to video
            process_parameters_to_video(
                parameters,
                renderer,
                args.output,
                video_metadata.get('fps', args.fps),
                show_progress
            )
        
        # Debug information
        if args.debug and parameters:
            print("\n📊 Debug Information:")
            
            # Sample first valid parameter frame
            first_params = next((p for p in parameters if p is not None), None)
            if first_params:
                print("\nFirst valid parameter frame:")
                for field_name in first_params.__dataclass_fields__:
                    value = getattr(first_params, field_name)
                    if isinstance(value, torch.Tensor):
                        print(f"  {field_name}: {value.item():.4f}")
            
            # Statistics
            valid_params = [p for p in parameters if p is not None]
            if valid_params:
                print(f"\nParameter statistics ({len(valid_params)} valid frames):")
                
                # Collect values for each parameter
                param_stats = {}
                for field_name in valid_params[0].__dataclass_fields__:
                    values = []
                    for p in valid_params:
                        val = getattr(p, field_name)
                        if isinstance(val, torch.Tensor):
                            values.append(val.item())
                    
                    if values:
                        values_tensor = torch.tensor(values)
                        param_stats[field_name] = {
                            'min': values_tensor.min().item(),
                            'max': values_tensor.max().item(),
                            'mean': values_tensor.mean().item(),
                            'std': values_tensor.std().item() if len(values) > 1 else 0
                        }
                
                for param_name, stats in param_stats.items():
                    print(f"  {param_name}:")
                    print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                    print(f"    Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
        
        print("\n🎉 Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup
        if renderer:
            renderer.cleanup()


if __name__ == "__main__":
    main()