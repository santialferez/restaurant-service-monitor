# 🏗️ Repository Structure

This document describes the organized structure of the Restaurant Service Monitor project.

## 📁 Directory Overview

```
restaurant-service-monitor/
├── README.md                    # Main project documentation
├── CLAUDE.md                    # Claude AI development notes
├── requirements.txt             # CPU dependencies
├── requirements-gpu.txt         # GPU dependencies
├── requirements-gpu-essential.txt # Essential GPU dependencies
│
├── config/                      # Configuration files
│   └── settings.yaml           # Application settings
│
├── src/                        # Core source code
│   ├── main.py                 # CPU version entry point
│   ├── main_gpu.py             # GPU version entry point
│   ├── core/                   # Core modules
│   │   ├── video_processor.py
│   │   ├── person_tracker.py       # CPU version
│   │   ├── person_tracker_gpu.py   # GPU optimized version
│   │   ├── gesture_detector.py     # CPU version
│   │   ├── gesture_detector_gpu.py # GPU optimized version
│   │   └── table_mapper.py
│   ├── analytics/              # Analysis modules
│   │   ├── movement_analyzer.py    # CPU version
│   │   ├── movement_analyzer_gpu.py # GPU version
│   │   └── service_metrics.py
│   └── visualization/          # Visualization components
│       └── dashboard.py
│
├── data/                       # Data files
│   ├── video_salon_poco_gente.MP4  # Main video file
│   └── demo_output/            # Analysis outputs
│       ├── analysis_results.json
│       ├── movement_data.csv
│       ├── movement_heatmap.jpg
│       └── service_metrics.xlsx
│
├── models/                     # AI model files
│   ├── yolov8m.pt             # YOLOv8 model weights
│   └── yolov8m.onnx           # ONNX export
│
├── demos/                      # Demo scripts
│   ├── demo.py                 # Basic CPU demo
│   ├── demo_gpu.py             # Basic GPU demo
│   ├── demo_gpu_2min.py        # 2-minute GPU demo
│   ├── demo_gpu_quick.py       # Quick GPU test
│   ├── simple_demo_video.py    # Simple video demo
│   ├── improved_demo_video.py  # Improved video demo
│   └── generate_demo_video.py  # Full video generator
│
├── tests/                      # Test scripts
│   ├── test_gpu_components.py  # GPU component tests
│   ├── test_optimized_gpu.py   # GPU optimization tests
│   └── test_quick.py           # Quick functionality tests
│
├── debug/                      # Debug and analysis tools
│   ├── debug_detection.py      # Debug person detection
│   ├── debug_gpu_detection.py  # Debug GPU detection
│   └── debug_overdetection.py  # Debug over-detection issues
│
├── scripts/                    # Utility scripts
│   └── run_analysis.sh         # Analysis runner script
│
├── outputs/                    # Generated outputs
│   ├── videos/                 # Final demo videos
│   │   └── restaurant_final_2min.mp4  # Best demo video
│   ├── reports/                # Analysis reports
│   │   ├── final_demo_summary.json
│   │   └── improved_demo_summary.json
│   └── archive/                # Archived/old files
│       ├── restaurant_gpu_demo_1min.mp4
│       ├── restaurant_gpu_demo_2min.mp4
│       └── restaurant_improved_2min.mp4
│
└── docs/                       # Documentation
    └── REPOSITORY_STRUCTURE.md # This file
```

## 🚀 Quick Start

### Running the System

**CPU Version:**
```bash
cd src/
python main.py
```

**GPU Version (Recommended):**
```bash
cd src/
python main_gpu.py
```

### Running Demos

**Quick Video Demo:**
```bash
cd demos/
python simple_demo_video.py
```

**Full 2-Minute Demo:**
```bash
cd demos/
python demo_gpu_2min.py
```

### Running Tests

```bash
cd tests/
python test_gpu_components.py
```

## 📊 Key Files

### Main Applications
- `src/main_gpu.py` - GPU-optimized restaurant service analyzer
- `src/main.py` - CPU version (for reference)

### Best Demo Video
- `outputs/videos/restaurant_final_2min.mp4` - Final optimized demo with:
  - Accurate person detection and tracking
  - Real hand-raising gesture detection
  - Clean visualization with thick lines
  - GPU acceleration (5+ FPS)

### Core Modules
- `src/core/person_tracker_gpu.py` - GPU-optimized person detection & tracking
- `src/core/gesture_detector_gpu.py` - GPU-based gesture detection
- `src/analytics/movement_analyzer_gpu.py` - GPU-accelerated movement analysis

## 🔧 Development

### Adding New Features
1. Place core functionality in `src/core/`
2. Add analysis features in `src/analytics/`
3. Create demos in `demos/`
4. Add tests in `tests/`

### GPU vs CPU Versions
- GPU files are suffixed with `_gpu.py`
- GPU versions provide 8-10x performance improvement
- CPU versions maintained for compatibility

## 📈 Performance

### GPU Optimizations
- **YOLOv8 with FP16 precision** - 2x faster inference
- **Batch processing** - Process multiple frames together
- **DeepSORT GPU tracking** - Accelerated multi-object tracking
- **PyTorch GPU tensors** - All computations on GPU

### Results
- **CPU Performance**: ~0.5 FPS
- **GPU Performance**: 5+ FPS (10x improvement)
- **Memory Usage**: ~270MB GPU VRAM
- **Accuracy**: 16+ people tracked simultaneously

## 🎯 Best Practices

### File Organization
1. **Core code** goes in `src/`
2. **Experiments** go in `demos/`
3. **Debug tools** go in `debug/`
4. **Final outputs** go in `outputs/videos/`
5. **Test code** goes in `tests/`

### Naming Conventions
- `*_gpu.py` - GPU-optimized versions
- `*_demo.py` - Demo/example scripts
- `test_*.py` - Test scripts
- `debug_*.py` - Debug/analysis tools

This organized structure makes the codebase easy to navigate, maintain, and extend! 🚀