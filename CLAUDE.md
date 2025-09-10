# Live2D Anime Generation - Claude Development Guidelines

## Project Overview

This project is a Python package (`live2d-anime-gen`) that converts facial landmarks to Live2D v3 model parameters for animation generation. Current implementation processes face videos using MediaPipe 478-point 3D landmark detection with iris tracking. Future development will include audio-to-video generation using deep learning models.

## Key Requirements

### Development Environment
- **Conda Environment**: `paraverbal`
- **Python Version**: 3.11
- **Package Distribution**: Wheel file format for distribution
- **Language**: All code and documentation in English

### Core Functionality
1. **Video Processing**: Process face videos → Live2D animation videos (current)
2. **Audio Processing**: Process audio → Live2D animation videos (future ML model)
3. **Parameter Mapping**: Convert 3D landmarks to Live2D v3 model parameters using PyTorch
4. **3D Head Pose**: Utilize depth information for accurate head rotation estimation
5. **Iris Tracking**: Dedicated iris landmarks for precise eye gaze control
6. **Extensible Architecture**: Base class design for different input modalities

### Technical Standards

#### Code Quality
- **Type Annotations**: Strict typing for all function parameters and return values
- **Modern Python Features**: Utilize Python 3.11+ capabilities where beneficial
- **Object-Oriented Design**: Clear class hierarchies with proper encapsulation
- **Access Control**: Explicit public/private member distinctions

#### Package Structure
- **Base Classes**: Abstract interfaces for different input modalities (face detection, audio processing)
- **Face Mapping**: InsightFace 106-point to Live2D parameter mapping
- **Audio Pipeline**: Future deep learning model for audio-to-landmark generation
- **CUDA-Only**: All processing optimized for GPU acceleration

### Dependencies
- Dependencies are documented in README.md
- Core package requires: torch, mediapipe, opencv-python
- Example demos require additional packages for Live2D rendering

### Deliverables
1. **Python Package**: Core library with base classes and sample implementations
2. **Example Scripts**: Complete video-to-Live2D conversion pipeline
3. **Documentation**: README.md with dependency installation instructions
4. **Distribution**: Wheel package for easy installation

## Development Commands

When implementing changes:
- Always run type checking and linting before committing
- Test the complete pipeline with sample data
- Ensure all dependencies are properly documented

## Key Findings from Codebase Analysis

### MediaPipe Integration
- **Face Detection**: Uses Face Mesh with refine_landmarks=True for iris tracking
- **Landmark Format**: Returns 478 3D landmarks as (478, 3) PyTorch tensors on GPU
- **3D Coordinates**: x, y normalized to [0,1], z represents depth relative to head center
- **Iris Support**: Dedicated iris landmarks (468-477) for eye gaze tracking

### Live2D-py Integration
- **Model Support**: Only Live2D v3 models (.model3.json format)
- **Parameter System**: Uses StandardParams class for parameter names
- **Core Parameters**: ParamEyeLOpen, ParamEyeROpen, ParamMouthOpenY, ParamMouthForm, ParamAngleX/Y/Z, ParamEyeBallX
- **Rendering Pipeline**: LAppModel class handles model loading and parameter updates
- **Smoothing**: Parameter smoothing implemented via exponential moving average

### Mapping Strategy (from live2d-py examples)
- **Eye Openness**: Calculated using aspect ratio of eye landmark points
- **Mouth**: Both openness and form calculated from lip landmarks
- **Head Pose**: Derived from facial geometry and landmark positions
- **Eye Tracking**: Iris position mapped to eye ball parameters
- **Smoothing Factor**: Configurable (0-1) for reducing jitter

## Architecture Notes
- **Current Phase**: Face video → Live2D video processing only
- **Future Phase**: Audio → Live2D video using deep learning models
- **GPU-First Design**: All components optimized for CUDA acceleration
- **Extensible Pipeline**: Base classes support different input modalities
- **PyTorch Integration**: All tensors on GPU for ML model compatibility