#!/usr/bin/env python3
"""Debug detection issues in GPU version"""

import sys
import os
import warnings
import torch
import cv2
import numpy as np
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test both CPU and GPU versions
from src.main import RestaurantAnalyzer
from src.main_gpu import RestaurantAnalyzerGPU

print("🔍 DEBUGGING DETECTION ISSUES")
print("="*50)

# Test single frame detection comparison
print("\n1. Testing single frame - CPU vs GPU detection...")

# Create both analyzers
print("   Creating CPU analyzer...")
cpu_analyzer = RestaurantAnalyzer(
    video_path="../data/video_salon_poco_gente.MP4",
    output_dir="../data/debug_cpu",
    skip_frames=15,
    resize_factor=0.5
)

print("   Creating GPU analyzer...")  
gpu_analyzer = RestaurantAnalyzerGPU(
    video_path="../data/video_salon_poco_gente.MP4",
    output_dir="../data/debug_gpu", 
    skip_frames=15,
    resize_factor=0.5,
    batch_size=1,
    use_tensorrt=False,
    use_half_precision=True
)

# Get the same frame for both
frame_num = 100
frame = cpu_analyzer.video_processor.get_frame(frame_num)
if frame is None:
    print("❌ Could not get frame")
    exit(1)

print(f"   Testing frame {frame_num}, shape: {frame.shape}")

# Test CPU detection
print("\n2. CPU Detection Test:")
cpu_persons = cpu_analyzer.person_tracker.update_tracks(frame, frame_num)
print(f"   CPU detected {len(cpu_persons)} persons")

for person_id, person in cpu_persons.items():
    print(f"     Person {person_id}: {person.person_type}, confidence: {person.confidence:.2f}")

# Test GPU detection
print("\n3. GPU Detection Test:")
gpu_persons = gpu_analyzer.person_tracker.update_tracks(frame, frame_num) 
print(f"   GPU detected {len(gpu_persons)} persons")

for person_id, person in gpu_persons.items():
    print(f"     Person {person_id}: {person.person_type}, confidence: {person.confidence:.2f}")

# Test raw YOLO detection
print("\n4. Raw YOLO Detection Test:")
print("   CPU YOLO:")
cpu_results = cpu_analyzer.person_tracker.yolo(frame, verbose=False)
cpu_detections = []
for result in cpu_results:
    if hasattr(result, 'boxes') and result.boxes is not None:
        boxes = result.boxes
        print(f"     Raw detections: {len(boxes)}")
        for i in range(len(boxes)):
            conf = boxes.conf[i].item()
            cls = boxes.cls[i].item()
            if cls == 0:  # person class
                cpu_detections.append(conf)
        print(f"     Person detections: {len(cpu_detections)} with confidences: {cpu_detections}")

print("   GPU YOLO:")
gpu_results = gpu_analyzer.person_tracker.yolo(frame, verbose=False)
gpu_detections = []
for result in gpu_results:
    if hasattr(result, 'boxes') and result.boxes is not None:
        boxes = result.boxes  
        print(f"     Raw detections: {len(boxes)}")
        for i in range(len(boxes)):
            conf = boxes.conf[i].item()
            cls = boxes.cls[i].item()
            if cls == 0:  # person class
                gpu_detections.append(conf)
        print(f"     Person detections: {len(gpu_detections)} with confidences: {gpu_detections}")

# Test confidence thresholds
print(f"\n5. Confidence Threshold Analysis:")
print(f"   CPU confidence threshold: {cpu_analyzer.person_tracker.conf_threshold}")
print(f"   GPU confidence threshold: {gpu_analyzer.person_tracker.conf_threshold}")

# Test movement threshold
print(f"   CPU movement threshold: {cpu_analyzer.person_tracker.movement_threshold}")
print(f"   GPU movement threshold: {gpu_analyzer.person_tracker.movement_threshold}")

# Test different frames to see if there are people in this video section
print(f"\n6. Multi-frame detection test:")
test_frames = [50, 100, 150, 200, 250, 300]
for test_frame in test_frames:
    frame = cpu_analyzer.video_processor.get_frame(test_frame)
    if frame is not None:
        # Quick CPU test
        cpu_persons = cpu_analyzer.person_tracker.update_tracks(frame, test_frame)
        print(f"   Frame {test_frame}: {len(cpu_persons)} persons detected")
        
        if len(cpu_persons) > 0:
            print(f"     ✅ People found at frame {test_frame}!")
            break
else:
    print("   ⚠️  No people detected in any test frames")

# Save debug frame for visual inspection
print(f"\n7. Saving debug frame for visual inspection...")
debug_frame = cpu_analyzer.video_processor.get_frame(150)
if debug_frame is not None:
    cv2.imwrite("debug_frame.jpg", debug_frame)
    print("   Saved debug_frame.jpg for visual inspection")

print(f"\n🔍 Detection debugging complete!")
print(f"   Check debug_frame.jpg to see if there are visible people in the video")