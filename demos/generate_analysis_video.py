#!/usr/bin/env python3
"""Generate comprehensive analysis video with all detection overlays"""

import sys
import os
import warnings
import torch
import cv2
import numpy as np
import time
import json
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.person_tracker_gpu import PersonTrackerGPU
from src.core.gesture_detector_gpu import GestureDetectorGPU
from src.core.video_processor import VideoProcessor

def generate_analysis_video():
    """Generate comprehensive analysis video with all overlays"""
    
    print("🎬 GENERATING COMPLETE ANALYSIS VIDEO")
    print("=" * 50)
    
    # Initialize components with improved parameters
    video_processor = VideoProcessor(
        "../data/video_salon_poco_gente.MP4",
        skip_frames=10,  # 3 FPS processing
        resize_factor=0.7
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
    
    # Video output setup
    output_path = "../outputs/videos/complete_analysis_2min.mp4"
    sample_frame = video_processor.get_frame(0)
    height, width = sample_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 3.0, (width, height))  # 3 FPS output
    
    if not out.isOpened():
        print("❌ Could not open video writer")
        return False
    
    print(f"📹 Output: {output_path}")
    print(f"📐 Size: {width}x{height}")
    
    # Processing parameters
    total_frames = 360  # 2 minutes at 3 FPS
    
    # Tracking data
    gesture_history = []
    people_stats = []
    
    try:
        for frame_idx in range(total_frames):
            # Get frame
            frame = video_processor.get_frame(frame_idx * 10)
            if frame is None:
                continue
            
            timestamp = frame_idx * 10 / 30  # Real timestamp
            
            # Track people
            persons = person_tracker.update_tracks(frame, frame_idx + 1)
            
            # Detect gestures
            frame_gestures = gesture_detector.detect_hand_raise_batch(
                frame, persons, frame_idx + 1, timestamp
            )
            
            # Store gesture events
            for gesture in frame_gestures:
                gesture_history.append({
                    'timestamp': timestamp,
                    'person_id': gesture.person_id,
                    'type': gesture.gesture_type,
                    'position': gesture.position
                })
            
            # Draw all overlays
            frame = draw_comprehensive_overlays(
                frame, persons, frame_gestures, gesture_history, frame_idx, timestamp, height, width
            )
            
            # Write frame
            out.write(frame)
            
            # Store stats
            people_stats.append({
                'frame': frame_idx,
                'people': len(persons),
                'waiters': sum(1 for p in persons.values() if p.person_type == 'waiter'),
                'customers': sum(1 for p in persons.values() if p.person_type == 'customer'),
                'gestures': len(frame_gestures)
            })
            
            # Progress
            if (frame_idx + 1) % 30 == 0:
                print(f"  Frame {frame_idx+1:3d}/{total_frames}: "
                     f"{len(persons)} people, {len(frame_gestures)} gestures")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    finally:
        out.release()
        video_processor.release()
    
    # Check output
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        avg_people = np.mean([s['people'] for s in people_stats]) if people_stats else 0
        total_gestures = len(gesture_history)
        
        print(f"\n✅ ANALYSIS VIDEO COMPLETE!")
        print(f"📁 File: {output_path} ({file_size/1024/1024:.2f} MB)")
        print(f"👥 Average people per frame: {avg_people:.1f}")
        print(f"🙋 Total gestures detected: {total_gestures}")
        print(f"⏱️  Duration: {total_frames/3:.1f} seconds")
        
        return True
    else:
        print("❌ Video file not created")
        return False

def draw_comprehensive_overlays(frame, persons, gestures, gesture_history, frame_idx, timestamp, height, width):
    """Draw all analysis overlays on frame"""
    
    # Copy frame for drawing
    overlay_frame = frame.copy()
    
    # 1. Draw person tracking boxes
    for person_id, person in persons.items():
        x1, y1, x2, y2 = person.bbox
        
        # Color by type
        if person.person_type == 'waiter':
            color = (0, 255, 0)  # Green
            type_label = "WAITER"
        elif person.person_type == 'customer':
            color = (255, 100, 0)  # Blue
            type_label = "CUSTOMER"
        else:
            color = (0, 255, 255)  # Yellow
            type_label = "PERSON"
        
        # Draw bounding box
        cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID and type label
        label = f"{type_label} {person_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(overlay_frame, (x1, y1-25), (x1+label_size[0]+10, y1), color, -1)
        cv2.putText(overlay_frame, label, (x1+5, y1-8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 2. Draw current gesture events
    for gesture in gestures:
        pos_x, pos_y = gesture.position
        
        # Draw gesture indicator
        cv2.circle(overlay_frame, (int(pos_x), int(pos_y)), 15, (0, 0, 255), 3)
        cv2.putText(overlay_frame, "HAND RAISED!", 
                   (int(pos_x)-40, int(pos_y)-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 3. Draw recent gesture trails (last 5 seconds)
    recent_gestures = [g for g in gesture_history 
                      if timestamp - g['timestamp'] <= 5.0]
    
    for gesture in recent_gestures:
        age = timestamp - gesture['timestamp']
        alpha = max(0.1, 1.0 - (age / 5.0))  # Fade out over 5 seconds
        
        pos_x, pos_y = gesture['position']
        color_intensity = int(255 * alpha)
        cv2.circle(overlay_frame, (int(pos_x), int(pos_y)), 8, 
                  (0, 0, color_intensity), 2)
    
    # 4. Draw statistics panel
    stats_panel_height = 120
    cv2.rectangle(overlay_frame, (10, 10), (500, stats_panel_height), (0, 0, 0), -1)
    cv2.rectangle(overlay_frame, (10, 10), (500, stats_panel_height), (255, 255, 255), 2)
    
    # Statistics text
    stats_lines = [
        f"IMPROVED TRACKING ANALYSIS - Frame {frame_idx+1:3d}",
        f"Time: {timestamp:6.1f}s | People: {len(persons):2d}",
        f"Waiters: {sum(1 for p in persons.values() if p.person_type == 'waiter'):2d} | "
        f"Customers: {sum(1 for p in persons.values() if p.person_type == 'customer'):2d}",
        f"Current Gestures: {len(gestures):2d} | Total: {len(gesture_history):3d}"
    ]
    
    for i, line in enumerate(stats_lines):
        cv2.putText(overlay_frame, line, (20, 35 + i*20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 5. Draw legend
    legend_y = height - 80
    cv2.rectangle(overlay_frame, (10, legend_y), (300, height-10), (0, 0, 0), -1)
    cv2.rectangle(overlay_frame, (10, legend_y), (300, height-10), (255, 255, 255), 1)
    
    legend_items = [
        ("Waiters: Green boxes", (0, 255, 0)),
        ("Customers: Blue boxes", (255, 100, 0)),
        ("Gestures: Red circles", (0, 0, 255))
    ]
    
    for i, (text, color) in enumerate(legend_items):
        cv2.putText(overlay_frame, text, (20, legend_y + 20 + i*15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return overlay_frame

def main():
    print("🎯 COMPLETE ANALYSIS VIDEO GENERATOR")
    print("Shows tracking, classification, and gesture detection")
    
    success = generate_analysis_video()
    
    if success:
        print(f"\n🎉 Video generation successful!")
        print(f"📱 Next: Converting to WhatsApp format...")
    else:
        print(f"\n❌ Video generation failed")

if __name__ == "__main__":
    main()