#!/usr/bin/env python3
"""Test optimized GPU tracker settings"""

import sys
import os
import warnings
import torch
import cv2
import numpy as np
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.person_tracker_gpu import PersonTrackerGPU
from src.core.video_processor import VideoProcessor

print("🧪 TESTING OPTIMIZED GPU TRACKER")
print("="*40)

# Create video processor
video_processor = VideoProcessor(
    video_path="../data/video_salon_poco_gente.MP4",
    skip_frames=15,
    resize_factor=0.5
)

# Create optimized GPU tracker (using new default parameters)
print("Creating optimized GPU tracker...")
tracker = PersonTrackerGPU(
    model_size='yolov8m.pt',
    batch_size=1,
    use_tensorrt=False,
    use_half_precision=True
)

print(f"✅ Optimized settings:")
print(f"   Confidence threshold: {tracker.conf_threshold}")
print(f"   Movement threshold: {tracker.movement_threshold}")
print(f"   Max age: 45 (DeepSORT parameter)")

# Test on 30 frames (1 minute at 2fps)
print(f"\n🎬 Testing on 30 frames (1 minute)...")

unique_ids = set()
frame_counts = []

for i in range(30):
    frame_num = i * 30  # Every 30th frame
    frame = video_processor.get_frame(frame_num)
    
    if frame is None:
        continue
    
    persons = tracker.update_tracks(frame, frame_num)
    unique_ids.update(persons.keys())
    frame_counts.append(len(persons))
    
    if i % 5 == 0:
        print(f"   Frame {frame_num:4d}: {len(persons)} persons")

print(f"\n📊 Results:")
print(f"   Total unique IDs: {len(unique_ids)}")
print(f"   Average persons per frame: {np.mean(frame_counts):.1f}")
print(f"   Max persons in frame: {max(frame_counts) if frame_counts else 0}")
print(f"   Min persons in frame: {min(frame_counts) if frame_counts else 0}")

# Classification breakdown
waiters = [p for p in tracker.persons.values() if p.person_type == 'waiter']
customers = [p for p in tracker.persons.values() if p.person_type == 'customer']
unknown = [p for p in tracker.persons.values() if p.person_type == 'unknown']

print(f"\n👥 Classification:")
print(f"   Waiters: {len(waiters)}")
print(f"   Customers: {len(customers)}")
print(f"   Unknown: {len(unknown)}")

print(f"\n✨ Optimized GPU tracker test complete!")
print(f"🎯 Target: ~17 people | Detected: {len(unique_ids)} unique IDs")

# Cleanup
del tracker
if torch.cuda.is_available():
    torch.cuda.empty_cache()