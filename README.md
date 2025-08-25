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
- **torch** (pip): GPU tensor operations and future ML model inference
- **insightface** (pip): 106-point facial landmark detection (CUDA only)
- **onnxruntime-gpu** (pip): GPU-accelerated inference for InsightFace models

### Example Dependencies (for Live2D rendering)
- **live2d-py** (pip): Live2D v3 model runtime and rendering
- **pygame** (pip): Window management for Live2D display
- **PyOpenGL** (pip): OpenGL bindings required by live2d-py

## Quick Start

```python
from live2d_anime_gen import InsightFaceDetector, FaceMapper, Live2DRenderer, VideoProcessor

# Initialize components (CUDA required)
detector = InsightFaceDetector()
mapper = FaceMapper(smooth_factor=0.5)
renderer = Live2DRenderer("path/to/model.model3.json")

# Process video
processor = VideoProcessor(detector, mapper, renderer)
processor.process_video(
    input_path="input.mp4",
    output_path="output.mp4"
)
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

## Troubleshooting

### Common Issues

1. **ONNX Runtime GPU Installation**
   ```bash
   # Uninstall CPU version first
   pip uninstall onnxruntime
   # Install GPU version
   pip install onnxruntime-gpu
   ```

2. **Live2D Model Compatibility**
   - Only Live2D v3 models (.model3.json) are supported
   - Ensure all model files (textures, physics, etc.) are in the same directory

3. **OpenGL Issues on Linux**
   ```bash
   sudo apt-get install freeglut3-dev
   ```

4. **InsightFace Model Download**
   ```python
   import insightface
   app = insightface.app.FaceAnalysis()
   app.prepare(ctx_id=0)  # This will download models automatically
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.