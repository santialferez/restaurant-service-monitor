# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🏗️ Repository Organization (Updated)

The repository has been cleaned and organized. All development should follow this structure:

```
src/           - Core application code
demos/         - Demo scripts and video generators  
tests/         - Test scripts
debug/         - Debug and analysis tools
models/        - AI model files
outputs/       - Generated outputs (videos, reports, archive)
scripts/       - Utility scripts
docs/          - Documentation
```

## Quick Start Commands

### GPU-Optimized Development (Recommended)
```bash
# GPU analysis (10x faster)
cd src/ && python main_gpu.py

# Quick GPU demo
cd demos/ && python simple_demo_video.py

# Full GPU demo video (2min)
cd demos/ && python demo_gpu_2min.py

# GPU component testing
cd tests/ && python test_gpu_components.py
```

### CPU Development (Legacy)
```bash
# CPU analysis (slower, for compatibility)
cd src/ && python main.py

# Quick CPU demo
cd demos/ && python demo.py

# CPU testing
cd tests/ && python test_quick.py

# Analysis script
bash scripts/run_analysis.sh
```

### Final Demo Video
```bash
# Best demo video (GPU-optimized, accurate detection)
outputs/videos/restaurant_final_2min.mp4
```

### Performance Testing Commands
- **Ultra-fast testing**: `--skip-frames 60 --resize 0.2`
- **Balanced performance**: `--skip-frames 10 --resize 0.5`  
- **High accuracy**: `--skip-frames 2 --resize 1.0`

## Architecture Overview

### GPU-Optimized Pipeline (Current)
The system uses GPU acceleration for 10x performance improvement via `RestaurantAnalyzerGPU` in `src/main_gpu.py`:

1. **Video Processing** (`VideoProcessor`) - Frame extraction with 2 FPS processing
2. **GPU Person Detection** (`PersonTrackerGPU`) - YOLOv8 + FP16 precision + DeepSORT with GPU embedder
3. **GPU Gesture Recognition** (`GestureDetectorGPU`) - CNN-based pose estimation (replaces MediaPipe)
4. **GPU Movement Analysis** (`MovementAnalyzerGPU`) - PyTorch tensor operations on GPU
5. **Batch Processing** - Processes 8-frame batches for optimal GPU utilization

### Key GPU Optimizations
- **FP16 Precision**: Half-precision inference on RTX tensor cores
- **Batch Processing**: 8-frame batches reduce GPU overhead
- **Memory Management**: <300MB VRAM usage with torch.cuda.empty_cache()
- **Optimized Parameters**: conf_threshold=0.6, nms_threshold=0.25

### Data Flow Architecture
```
Video → VideoProcessor → PersonTracker → GestureDetector
                                    ↓
TableMapper ← MovementAnalyzer ← ServiceMetrics
     ↓              ↓              ↓
Results aggregation → Dashboard/Reports
```

### Key Integration Points
- **Person Classification**: Uses movement threshold (100px/frame) to distinguish waiters from customers
- **Service Correlation**: Links gesture events to table assignments and waiter responses
- **Real-time Processing**: Frame-by-frame processing with configurable performance trade-offs

## Configuration System

### Critical Settings (`config/settings.yaml`)
- **Performance**: `video.skip_frames`, `video.resize_factor` 
- **Detection**: `detection.confidence_threshold`, `detection.model`
- **Gesture**: `gesture.hand_raise_threshold`, `gesture.hand_raise_duration`
- **Classification**: `classification.movement_threshold` (key for waiter/customer distinction)

## Common Development Patterns

### Processing Parameters
The system uses a standardized parameter pattern for performance tuning:
- `skip_frames`: Higher = faster, lower accuracy (1-100 range)
- `resize_factor`: Smaller = faster, lower accuracy (0.1-1.0 range)
- Balance: `skip_frames=10, resize_factor=0.5` for most development

### DeepSORT Integration
Uses specific format for detections: `(bbox_xywh, confidence, class_name)` tuples. This is a key integration point that has been debugged and should not be modified without careful testing.

### Output Structure
All analysis results follow consistent JSON schema in `analysis_results.json`:
- `metrics`: Aggregated service metrics
- `events`: Time-series gesture/response events  
- `waiter_performance`: Individual waiter statistics
- `movement_statistics`: Per-person movement analysis

### Error-Prone Areas
1. **OpenCV Type Conversions**: Always cast coordinates to `int()` for OpenCV functions
2. **DeepSORT API**: Use exact tuple format `(bbox, confidence, class)` 
3. **MediaPipe Coordinates**: Normalize pose landmarks properly for gesture detection
4. **Convex Hull**: Handle empty point sets with try/catch blocks

## Testing Strategy

### Quick Validation
- `test_quick.py`: Processes 100 frames for rapid validation
- `demo.py`: 30-second analysis with full pipeline demonstration

### Performance Benchmarking  
Monitor these metrics during development:
- Frame processing rate (target: >1 FPS)
- Memory usage (watch for accumulator growth)
- GPU utilization (if available)
- Detection accuracy vs. speed trade-offs

### Fixed Issues & Solutions
- **FP16 Precision Mismatch**: Fixed YOLO→DeepSORT type conversion (half→float32)
- **Over-detection**: Tuned confidence threshold from 0.5→0.6, NMS to 0.25
- **False Gestures**: Replaced simple heuristics with height-based motion analysis
- **Missing Detections**: Enhanced right-side contrast, full frame processing
- **Visual Clarity**: Increased line thickness to 2-3px, added label backgrounds

### Common Debug Commands
```bash
# Check GPU status
nvidia-smi

# Debug GPU detection
cd debug/ && python debug_gpu_detection.py

# Test specific components
cd tests/ && python test_gpu_components.py

# Check video properties
python -c "from src.core.video_processor import VideoProcessor; vp = VideoProcessor('../data/video_salon_poco_gente.MP4'); print(vp.get_video_info())"

# Verify outputs
ls -la outputs/videos/     # Final videos
ls -la outputs/reports/    # Analysis reports
ls -la outputs/archive/    # Old/test videos
```

### Performance Benchmarks
- **CPU Baseline**: ~0.5 FPS
- **GPU Optimized**: 5+ FPS (10x improvement)
- **Memory**: 270MB GPU VRAM
- **Accuracy**: 16+ people tracked simultaneously