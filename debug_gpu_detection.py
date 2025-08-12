#!/usr/bin/env python3
"""Debug GPU detection pipeline step by step"""

import sys
import os
import warnings
import torch
import cv2
import numpy as np
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.person_tracker_gpu import PersonTrackerGPU
from src.core.video_processor import VideoProcessor

print("🔍 DEBUGGING GPU DETECTION PIPELINE")
print("="*50)

# Create video processor
video_processor = VideoProcessor(
    video_path="data/video_salon_poco_gente.MP4",
    skip_frames=15,
    resize_factor=0.5
)

# Create GPU tracker
print("1. Creating GPU PersonTracker...")
tracker = PersonTrackerGPU(
    model_size='yolov8m.pt',
    conf_threshold=0.5,
    movement_threshold=3.0,
    batch_size=1,
    use_tensorrt=False,
    use_half_precision=True
)
print(f"   ✅ GPU tracker created")
print(f"   Device: {tracker.device}")
print(f"   Confidence threshold: {tracker.conf_threshold}")

# Test specific frames where CPU version detected people
test_frames = [100, 200, 300, 400, 500]

print(f"\n2. Testing frame-by-frame detection...")

for frame_num in test_frames:
    print(f"\n   Testing frame {frame_num}:")
    
    # Get frame
    frame = video_processor.get_frame(frame_num)
    if frame is None:
        print(f"     ❌ Could not get frame {frame_num}")
        continue
    
    print(f"     Frame shape: {frame.shape}")
    
    # Test raw YOLO detection first
    print(f"     Raw YOLO detection:")
    try:
        results = tracker.yolo(frame, verbose=False)
        total_detections = 0
        person_detections = 0
        person_confidences = []
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                total_detections = len(boxes)
                
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    cls = int(boxes.cls[i])
                    
                    if cls == 0:  # person class
                        person_detections += 1
                        person_confidences.append(conf)
        
        print(f"       Total detections: {total_detections}")
        print(f"       Person detections: {person_detections}")
        if person_confidences:
            print(f"       Person confidences: {[f'{c:.2f}' for c in person_confidences]}")
            print(f"       Above threshold ({tracker.conf_threshold}): {sum(1 for c in person_confidences if c > tracker.conf_threshold)}")
        else:
            print(f"       No person detections found")
            
    except Exception as e:
        print(f"     ❌ YOLO detection failed: {e}")
        continue
    
    # Test full tracking pipeline
    print(f"     Full tracking pipeline:")
    try:
        persons = tracker.update_tracks(frame, frame_num)
        print(f"       Tracked persons: {len(persons)}")
        
        if len(persons) > 0:
            for person_id, person in persons.items():
                print(f"         Person {person_id}: confidence={person.confidence:.2f}, type={person.person_type}")
        else:
            print(f"       No persons tracked")
            
    except Exception as e:
        print(f"     ❌ Tracking failed: {e}")
        continue

# Test batch detection (which is used in the demo)
print(f"\n3. Testing batch detection (demo method)...")
try:
    # Get a small batch of frames
    frames = []
    frame_nums = []
    
    for i, frame_num in enumerate([100, 115, 130]):  # 3 frames
        frame = video_processor.get_frame(frame_num)
        if frame is not None:
            frames.append(frame)
            frame_nums.append(frame_num)
    
    if frames:
        print(f"   Testing batch of {len(frames)} frames...")
        batch_results = tracker.update_tracks_batch(frames, frame_nums)
        
        print(f"   Batch results:")
        for i, (frame_num, persons) in enumerate(zip(frame_nums, batch_results)):
            print(f"     Frame {frame_num}: {len(persons)} persons")
            if len(persons) > 0:
                for pid, person in persons.items():
                    print(f"       Person {pid}: {person.person_type}, conf={person.confidence:.2f}")
    else:
        print(f"   ❌ Could not get frames for batch test")

except Exception as e:
    print(f"   ❌ Batch detection failed: {e}")
    import traceback
    traceback.print_exc()

# Test with different confidence thresholds
print(f"\n4. Testing different confidence thresholds...")
test_frame = video_processor.get_frame(200)
if test_frame is not None:
    for threshold in [0.1, 0.3, 0.5, 0.7]:
        print(f"   Testing threshold {threshold}:")
        tracker.conf_threshold = threshold
        
        try:
            results = tracker.yolo(test_frame, verbose=False)
            person_count = 0
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        conf = float(boxes.conf[i])
                        cls = int(boxes.cls[i])
                        if cls == 0 and conf > threshold:
                            person_count += 1
            
            print(f"     Persons detected: {person_count}")
            
        except Exception as e:
            print(f"     Error: {e}")

# Reset threshold
tracker.conf_threshold = 0.5

print(f"\n5. Checking model and device status...")
print(f"   YOLO model: {tracker.yolo.model}")
print(f"   Model device: {next(tracker.yolo.model.parameters()).device}")
print(f"   Expected device: {tracker.device}")
print(f"   Half precision: {tracker.use_half_precision}")

# Save a test frame for visual inspection
print(f"\n6. Saving test frame for visual inspection...")
test_frame = video_processor.get_frame(200)
if test_frame is not None:
    cv2.imwrite("gpu_debug_frame.jpg", test_frame)
    print(f"   Saved gpu_debug_frame.jpg")

print(f"\n🔍 GPU Detection debugging complete!")

# Cleanup
del tracker
if torch.cuda.is_available():
    torch.cuda.empty_cache()