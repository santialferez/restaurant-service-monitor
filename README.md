# Restaurant Service Video Analysis System

A comprehensive computer vision system for analyzing restaurant surveillance footage to track customer service metrics and operational efficiency.

## Features

### 🙋 Customer Request Detection
- Hand-raising gesture recognition using MediaPipe
- Automatic timestamp and table association
- Confidence scoring for gesture detection

### 🚶 Waiter Movement Analysis
- Real-time person tracking with YOLOv8 and DeepSORT
- Movement path visualization and heatmaps
- Speed and distance calculations
- Common route pattern identification

### ⏱️ Service Response Time Metrics
- Automated response time measurement
- Statistical analysis (average, min, max, median)
- Service delay pattern identification
- Efficiency scoring

### 🪑 Table Attention Frequency Analysis
- Automatic table visit tracking
- Service interval calculations
- Underserved table alerts
- Table-specific service metrics

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)
- 4GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd rest
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the analysis with default settings:
```bash
bash run_analysis.sh
```

Or specify a custom video:
```bash
bash run_analysis.sh path/to/your/video.mp4
```

### Manual Execution

```bash
python src/main.py data/video_salon_poco_gente.MP4 \
    --config config/settings.yaml \
    --output data/outputs \
    --skip-frames 2 \
    --resize 0.5
```

### Command Line Options

- `video`: Path to input video file
- `--config`: Path to configuration file (optional)
- `--output`: Output directory for results (default: data/outputs)
- `--skip-frames`: Process every Nth frame (default: 2)
- `--resize`: Resize factor for video (default: 0.5)

## Dashboard

Launch the interactive dashboard to visualize results:

```bash
streamlit run src/visualization/dashboard.py
```

The dashboard provides:
- Real-time metrics overview
- Service performance graphs
- Movement heatmaps
- Table status monitoring
- Waiter performance analysis
- Video playback with annotations

## Project Structure

```
rest/
├── data/
│   ├── video_salon_poco_gente.MP4  # Input video
│   └── outputs/                    # Analysis results
├── src/
│   ├── core/                       # Core modules
│   │   ├── video_processor.py      # Video handling
│   │   ├── person_tracker.py       # Person detection/tracking
│   │   ├── gesture_detector.py     # Gesture recognition
│   │   └── table_mapper.py         # Table management
│   ├── analytics/                  # Analysis modules
│   │   ├── service_metrics.py      # Service metrics calculation
│   │   └── movement_analyzer.py    # Movement pattern analysis
│   ├── visualization/              # Visualization tools
│   │   └── dashboard.py           # Streamlit dashboard
│   └── main.py                    # Main analysis script
├── config/
│   └── settings.yaml              # Configuration settings
├── requirements.txt               # Python dependencies
├── run_analysis.sh               # Run script
└── README.md                     # Documentation
```

## Configuration

Edit `config/settings.yaml` to customize:

- **Video Processing**: Frame skip rate, resize factor
- **Detection**: Model selection, confidence thresholds
- **Gesture Recognition**: Detection sensitivity, duration thresholds
- **Service Metrics**: Target response times, efficiency thresholds
- **Visualization**: Colors, heatmap settings, annotation options

## Output Files

After analysis, the following files are generated in `data/outputs/`:

- `analysis_results.json`: Complete analysis data
- `analysis_report.txt`: Human-readable summary report
- `service_metrics.xlsx`: Detailed metrics in Excel format
- `movement_data.csv`: Movement tracking data
- `movement_heatmap.jpg`: Visual heatmap of movement patterns
- `flow_field.jpg`: Movement flow visualization
- `movement_statistics.png`: Statistical plots
- `frame_XXXXXX.jpg`: Annotated video frames
- `table_config.json`: Table configuration data

## Key Metrics Explained

### Response Time
Time between customer hand-raise and waiter arrival at table

### Response Rate
Percentage of customer requests that received a response

### Efficiency Score
Combined metric (0-100) based on:
- Response time performance
- Response rate
- Table visit frequency

### Table Visit Interval
Average time between consecutive visits to each table

## Performance Optimization

For faster processing:

1. **Increase frame skip**: Set `--skip-frames 3` or higher
2. **Reduce resolution**: Set `--resize 0.3` for smaller frames
3. **Use GPU**: Ensure CUDA is installed for GPU acceleration
4. **Adjust detection confidence**: Higher thresholds = faster but less accurate

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce video resolution with `--resize`
   - Increase frame skip rate
   - Process shorter video segments

2. **Slow Processing**
   - Check GPU availability: `nvidia-smi`
   - Reduce model complexity in settings
   - Use lighter YOLO model (yolov8n.pt)

3. **Poor Detection Accuracy**
   - Adjust confidence thresholds in config
   - Ensure good video quality and lighting
   - Calibrate table positions manually

## API Usage

```python
from src.main import RestaurantAnalyzer

# Initialize analyzer
analyzer = RestaurantAnalyzer(
    video_path="path/to/video.mp4",
    config_path="config/settings.yaml",
    output_dir="outputs"
)

# Run analysis
analyzer.run()

# Access results
results = analyzer.analysis_results
metrics = results['metrics']
print(f"Average response time: {metrics['avg_response_time']:.1f}s")
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- YOLOv8 by Ultralytics for object detection
- MediaPipe by Google for pose estimation
- DeepSORT for multi-object tracking
- Streamlit for dashboard framework

## Contact

For questions or support, please open an issue on GitHub.