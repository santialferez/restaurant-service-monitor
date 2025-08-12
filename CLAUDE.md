# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

### Development and Testing
```bash
# Quick 30-second demo
python demo.py

# Fast testing (sparse frame sampling)
python test_quick.py

# Full analysis with default settings
bash run_analysis.sh

# Full analysis with custom video
bash run_analysis.sh path/to/video.mp4

# Manual execution with performance options
python src/main.py data/video_salon_poco_gente.MP4 --skip-frames 10 --resize 0.3
```

### Dashboard and Visualization
```bash
# Launch interactive dashboard
streamlit run src/visualization/dashboard.py

# Check GPU availability for performance
nvidia-smi
```

### Performance Testing Commands
- **Ultra-fast testing**: `--skip-frames 60 --resize 0.2`
- **Balanced performance**: `--skip-frames 10 --resize 0.5`  
- **High accuracy**: `--skip-frames 2 --resize 1.0`

## Architecture Overview

### Core Processing Pipeline
The system follows a multi-stage computer vision pipeline orchestrated by `RestaurantAnalyzer` in `src/main.py`:

1. **Video Processing** (`VideoProcessor`) - Handles frame extraction with configurable skip rates and resizing
2. **Person Detection** (`PersonTracker`) - YOLOv8 detection + DeepSORT tracking with automatic waiter/customer classification based on movement patterns  
3. **Gesture Recognition** (`GestureDetector`) - MediaPipe pose estimation to detect hand-raising gestures
4. **Spatial Analysis** (`TableMapper`) - Table detection/mapping and visit tracking
5. **Analytics** (`ServiceMetricsCalculator`, `MovementAnalyzer`) - Service metrics calculation and movement pattern analysis

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

### Common Debug Commands
```bash
# Check video properties
python -c "from src.core.video_processor import VideoProcessor; vp = VideoProcessor('data/video_salon_poco_gente.MP4'); print(vp.get_video_info())"

# Test DeepSORT format
python -c "from deep_sort_realtime.deepsort_tracker import DeepSort; print('API compatible')"

# Verify output generation
ls -la data/demo_output/  # Should show JSON, CSV, images
```