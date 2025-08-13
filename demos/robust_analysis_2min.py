#!/usr/bin/env python3
"""Robust 2-minute analysis video with simplified overlays and proper error handling"""

import sys
import os
import warnings
import torch
import cv2
import numpy as np
import time
import signal
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.person_tracker_gpu import PersonTrackerGPU
from src.core.gesture_detector_gpu import GestureDetectorGPU
from src.core.video_processor import VideoProcessor

class VideoGenerationHandler:
    def __init__(self):
        self.video_writer = None
        self.video_processor = None
        
    def cleanup(self):
        """Ensure proper cleanup"""
        if self.video_writer:
            print("\n🔄 Finalizing video...")
            self.video_writer.release()
        if self.video_processor:
            self.video_processor.release()
        print("✅ Cleanup complete")

# Global handler for cleanup
cleanup_handler = VideoGenerationHandler()

def signal_handler(signum, frame):
    """Handle interruption gracefully"""
    print(f"\n⚠️  Process interrupted (signal {signum})")
    cleanup_handler.cleanup()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def generate_robust_analysis_video():
    """Generate robust 2-minute analysis video with proper error handling"""
    
    print("🎬 ROBUST 2-MINUTE ANALYSIS VIDEO")
    print("=" * 50)
    
    # Initialize components with optimized parameters
    video_processor = VideoProcessor(
        "../data/video_salon_poco_gente.MP4",
        skip_frames=20,  # Every 20th frame = 1.5 FPS (faster)
        resize_factor=0.5  # Smaller for speed
    )
    cleanup_handler.video_processor = video_processor
    
    # Initialize improved tracker
    person_tracker = PersonTrackerGPU(
        model_size='yolov8m.pt',
        conf_threshold=0.4,  # IMPROVED
        max_age=30,          # IMPROVED
        movement_threshold=5.0,
        nms_threshold=0.4    # IMPROVED
    )
    
    # NO gesture detector for speed - focus on tracking improvements
    print("🚀 Optimized for speed: Tracking only (no gesture detection)")
    
    # Video output setup
    output_path = "../outputs/videos/robust_analysis_2min.mp4"
    sample_frame = video_processor.get_frame(0)
    height, width = sample_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 1.5, (width, height))  # 1.5 FPS
    cleanup_handler.video_writer = out
    
    if not out.isOpened():
        print("❌ Could not open video writer")
        return False
    
    print(f"📹 Output: {output_path}")
    print(f"📐 Size: {width}x{height}")
    print(f"⚡ Processing: 1.5 FPS (180 frames for 2 minutes)")
    
    # Process 180 frames (2 minutes at 1.5 FPS)
    total_frames = 180
    start_time = time.time()
    frames_written = 0
    people_stats = []
    
    try:
        for frame_idx in range(total_frames):
            # Get frame
            frame = video_processor.get_frame(frame_idx * 20)
            if frame is None:
                continue
            
            timestamp = frame_idx * 20 / 30  # Real timestamp
            
            # Track people only (fast)
            persons = person_tracker.update_tracks(frame, frame_idx + 1)
            
            # Draw simplified overlays (minimal processing)
            frame = draw_fast_overlays(frame, persons, frame_idx, timestamp, total_frames)
            
            # Write frame
            out.write(frame)
            frames_written += 1
            
            # Store stats
            people_stats.append({
                'frame': frame_idx,
                'people': len(persons),
                'waiters': sum(1 for p in persons.values() if p.person_type == 'waiter'),
                'customers': sum(1 for p in persons.values() if p.person_type == 'customer')
            })
            
            # Progress with time estimation
            if (frame_idx + 1) % 20 == 0:
                elapsed = time.time() - start_time
                fps = frames_written / elapsed
                eta = (total_frames - frame_idx - 1) / fps if fps > 0 else 0
                print(f"  Frame {frame_idx+1:3d}/{total_frames}: {len(persons)} people "
                     f"| FPS: {fps:.1f} | ETA: {eta:.0f}s")
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return False
    
    finally:
        # CRITICAL: Always release video writer
        print(f"\n🔄 Finalizing video (wrote {frames_written} frames)...")
        out.release()
        video_processor.release()
        cleanup_handler.video_writer = None
        cleanup_handler.video_processor = None
    
    # Verify output
    processing_time = time.time() - start_time
    
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        avg_people = np.mean([s['people'] for s in people_stats]) if people_stats else 0
        max_people = max([s['people'] for s in people_stats]) if people_stats else 0
        
        print(f"\n✅ ROBUST ANALYSIS VIDEO COMPLETE!")
        print(f"📁 File: {output_path} ({file_size/1024/1024:.2f} MB)")
        print(f"⏱️  Processing time: {processing_time:.1f}s")
        print(f"📊 Performance:")
        print(f"   Frames written: {frames_written}/{total_frames}")
        print(f"   Average people: {avg_people:.1f}")
        print(f"   Max people: {max_people}")
        print(f"   Processing FPS: {frames_written/processing_time:.1f}")
        
        # Test video playability
        test_cap = cv2.VideoCapture(output_path)
        if test_cap.isOpened():
            frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            test_cap.release()
            print(f"✅ Video verification: {frame_count} frames readable")
            return True
        else:
            print("❌ Video verification failed")
            return False
    else:
        print("❌ Video file not created")
        return False

def draw_fast_overlays(frame, persons, frame_idx, timestamp, total_frames):
    """Draw minimal overlays for maximum speed"""
    
    # Minimal overlay - just tracking boxes and basic stats
    for person_id, person in persons.items():
        x1, y1, x2, y2 = person.bbox
        
        # Simple color coding
        if person.person_type == 'waiter':
            color = (0, 255, 0)  # Green
        elif person.person_type == 'customer':
            color = (255, 100, 0)  # Blue
        else:
            color = (0, 255, 255)  # Yellow
        
        # Simple box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, str(person_id), (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Minimal stats panel
    cv2.rectangle(frame, (10, 10), (350, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"IMPROVED TRACKING - {frame_idx+1}/{total_frames}", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"People: {len(persons)} | Time: {timestamp:.1f}s", (20, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def main():
    print("🎯 ROBUST 2-MINUTE ANALYSIS VIDEO")
    print("Optimized for reliability and speed")
    
    try:
        success = generate_robust_analysis_video()
        
        if success:
            print(f"\n🎉 Video generation successful!")
            print(f"🔄 Ready for WhatsApp conversion")
        else:
            print(f"\n❌ Video generation failed")
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        cleanup_handler.cleanup()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        cleanup_handler.cleanup()

if __name__ == "__main__":
    main()