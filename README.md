# Live2D Anime Generation

A Python package for generating Live2D animations from multiple input sources. Currently supports face video processing with InsightFace 106-point detection. Future versions will include audio-to-animation generation using deep learning models.

## Environment Setup

### Prerequisites
- Python 3.11
- CUDA-compatible GPU (required)
- Conda package manager
- Git

### Installation Steps

1. Create conda environment with Python 3.11
2. Install PyTorch with CUDA support
3. Install core dependencies (insightface, onnxruntime-gpu)
4. Install Live2D rendering dependencies (live2d-py, pygame, PyOpenGL)

## Dependencies

### Core Package Dependencies
The following are top-level dependencies that need to be manually installed.

- **torch** (pip): GPU tensor operations
- **insightface** (pip): 106-point facial landmark detection (CUDA only)
- **onnxruntime-gpu** (pip): GPU-accelerated inference for InsightFace models
- **ijson** (pip): Streaming JSON parser for memory-efficient data I/O
- **PyNvVideoCodec** (pip): NVIDIA hardware-accelerated video encoding/decoding with NVENC
- **live2d-py** (pip): Live2D v3 model runtime and rendering
- **pygame** (pip): Window management for Live2D display
- **opencv-python** (pip): Video I/O and image processing

### Example Dependencies
Additional top-level dependencies required only for running the example scripts, beyond the core dependencies listed above.

- None

## Quick Start

```python
from live2d_anime_gen import InsightFaceDetector, FaceMapper, Live2DRenderer, ParameterSmoother, VideoReader, VideoWriter

# Initialize components (CUDA required)
detector = InsightFaceDetector(det_size=(640, 640))
mapper = FaceMapper(smooth_factor=0.7)
smoother = ParameterSmoother(method="ema", alpha=0.7)
renderer = Live2DRenderer("path/to/model.model3.json", canvas_size=(1280, 720))

# Process video frame by frame
reader = VideoReader("input.mp4")
writer = VideoWriter("output.mp4", reader.fps, (1280, 720))

for frame in reader.read_frames():
    landmarks = detector.detect(frame)
    if landmarks is not None:
        parameters = mapper.map(landmarks, (frame.shape[0], frame.shape[1]))
        if parameters is not None:
            parameters = smoother.smooth(parameters)
            rendered_frame = renderer.render(parameters)
            writer.write_frame(rendered_frame)

reader.close()
writer.close()
```

## Project Structure

```
./
├── live2d_anime_gen/      # Main package
│   ├── core/              # Base classes and type definitions
│   ├── detectors/         # Face detection implementations
│   ├── mappers/           # Landmark-to-parameter mapping
│   ├── processors/        # Video processing and parameter smoothing
│   └── renderers/         # Live2D rendering components
└── examples/              # Example scripts and Live2D models
```
