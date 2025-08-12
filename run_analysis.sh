#!/bin/bash

# Restaurant Service Video Analysis System
# Run script for analyzing restaurant surveillance footage

echo "================================================"
echo "Restaurant Service Video Analysis System"
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Default video path
VIDEO_PATH="${1:-data/video_salon_poco_gente.MP4}"

# Check if video file exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found: $VIDEO_PATH"
    exit 1
fi

echo ""
echo "Starting analysis..."
echo "Video: $VIDEO_PATH"
echo ""

# Run the analysis
python src/main.py "$VIDEO_PATH" \
    --config config/settings.yaml \
    --output data/outputs \
    --skip-frames 2 \
    --resize 0.5

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "Analysis complete!"
    echo "Results saved to: data/outputs/"
    echo "================================================"
    
    # Ask if user wants to launch dashboard
    echo ""
    read -p "Launch dashboard? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Launching dashboard..."
        streamlit run src/visualization/dashboard.py
    fi
else
    echo ""
    echo "Analysis failed. Check the logs for details."
    exit 1
fi