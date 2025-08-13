#!/usr/bin/env python3
"""Quick analysis video generation - 30 seconds with all improvements"""

import sys
import os
import warnings
import torch
import cv2
import numpy as np
import time
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.person_tracker_gpu import PersonTrackerGPU
from src.core.gesture_detector_gpu import GestureDetectorGPU
from src.core.video_processor import VideoProcessor

def generate_quick_analysis_video():
    """Generate 30-second analysis video with all improvements"""
    
    print("🚀 QUICK ANALYSIS VIDEO - 30 SECONDS")
    print("=" * 45)
    
    # Initialize components with improved parameters
    video_processor = VideoProcessor(
        "../data/video_salon_poco_gente.MP4",
        skip_frames=15,  # Every 15th frame for 2 FPS
        resize_factor=0.6
    )
    
    # Initialize improved tracker
    person_tracker = PersonTrackerGPU(
        model_size='yolov8m.pt',
        conf_threshold=0.4,  # IMPROVED
        max_age=30,          # IMPROVED
        movement_threshold=5.0,
        nms_threshold=0.4    # IMPROVED
    )
    
    # Initialize gesture detector
    gesture_detector = GestureDetectorGPU(device='cuda')
    
    # Video output setup - smaller for speed
    output_path = "../outputs/videos/quick_analysis.mp4"
    sample_frame = video_processor.get_frame(0)
    height, width = sample_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 2.0, (width, height))  # 2 FPS
    
    if not out.isOpened():
        print("❌ Could not open video writer")
        return False
    
    print(f"📹 Output: {output_path}")
    print(f"📐 Size: {width}x{height}")
    
    # Process only 60 frames (30 seconds at 2 FPS)
    total_frames = 60
    gesture_count = 0
    
    try:
        for frame_idx in range(total_frames):
            # Get frame
            frame = video_processor.get_frame(frame_idx * 15)
            if frame is None:
                continue
            
            timestamp = frame_idx * 15 / 30  # Real timestamp
            
            # Track people
            persons = person_tracker.update_tracks(frame, frame_idx + 1)
            
            # Detect gestures
            frame_gestures = gesture_detector.detect_hand_raise_batch(
                frame, persons, frame_idx + 1, timestamp
            )
            gesture_count += len(frame_gestures)
            
            # Draw overlays
            frame = draw_analysis_overlays(frame, persons, frame_gestures, frame_idx, timestamp, height, width)
            
            # Write frame
            out.write(frame)
            
            # Progress
            if (frame_idx + 1) % 15 == 0:
                print(f"  Frame {frame_idx+1:2d}/{total_frames}: {len(persons)} people, {len(frame_gestures)} gestures")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    finally:
        out.release()
        video_processor.release()
    
    # Check output
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"\n✅ QUICK ANALYSIS VIDEO COMPLETE!")
        print(f"📁 File: {output_path} ({file_size/1024/1024:.2f} MB)")
        print(f"🙋 Total gestures: {gesture_count}")
        print(f"⏱️  Duration: 30 seconds")
        
        return True
    else:
        print("❌ Video file not created")
        return False

def draw_analysis_overlays(frame, persons, gestures, frame_idx, timestamp, height, width):
    """Draw simplified analysis overlays"""
    
    overlay_frame = frame.copy()
    
    # Draw person tracking boxes
    for person_id, person in persons.items():
        x1, y1, x2, y2 = person.bbox
        
        # Color by type
        if person.person_type == 'waiter':
            color = (0, 255, 0)  # Green
            type_label = "W"
        elif person.person_type == 'customer':
            color = (255, 100, 0)  # Blue
            type_label = "C"
        else:
            color = (0, 255, 255)  # Yellow
            type_label = "P"
        
        # Draw box
        cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID
        cv2.circle(overlay_frame, (x1+10, y1+10), 12, color, -1)
        cv2.putText(overlay_frame, f"{type_label}{person_id}", (x1+2, y1+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw current gestures
    for gesture in gestures:
        pos_x, pos_y = gesture.position
        cv2.circle(overlay_frame, (int(pos_x), int(pos_y)), 20, (0, 0, 255), 3)
        cv2.putText(overlay_frame, "GESTURE!", (int(pos_x)-30, int(pos_y)-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Statistics panel
    cv2.rectangle(overlay_frame, (10, 10), (400, 80), (0, 0, 0), -1)
    cv2.rectangle(overlay_frame, (10, 10), (400, 80), (255, 255, 255), 2)
    
    stats_text = [
        f"IMPROVED TRACKING - Frame {frame_idx+1:2d}/60",
        f"People: {len(persons):2d} | Gestures: {len(gestures):2d} | Time: {timestamp:4.1f}s"
    ]
    
    for i, text in enumerate(stats_text):
        cv2.putText(overlay_frame, text, (20, 30 + i*20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return overlay_frame

def main():
    print("🎯 QUICK ANALYSIS VIDEO GENERATOR")
    print("30-second demonstration with improved tracking")
    
    success = generate_quick_analysis_video()
    
    if success:
        print(f"\n🎉 Quick video complete!")
    else:
        print(f"\n❌ Video generation failed")

if __name__ == "__main__":
    main()