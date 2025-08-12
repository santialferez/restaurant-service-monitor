#!/usr/bin/env python3
"""Clean pose visualization with minimal overlays"""

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

from src.core.video_processor import VideoProcessor
from src.core.pose_detector_yolo import PoseDetectorYOLO

class CleanPoseDemo:
    def __init__(self, video_path: str):
        self.video_path = video_path
        
        # Initialize video processor
        self.video_processor = VideoProcessor(
            video_path=video_path,
            skip_frames=15,  # 2 FPS processing
            resize_factor=0.5
        )
        
        # Initialize YOLOv8 pose detector
        print("🎯 Initializing YOLOv8 pose detector...")
        self.pose_detector = PoseDetectorYOLO(
            model_name='yolov8m-pose.pt',
            device='cuda',
            conf_threshold=0.4,  # Lower threshold to catch more people
            batch_size=1
        )
        
        # COCO pose connections for skeleton
        self.pose_connections = [
            # Head connections
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Arm connections  
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
            # Body connections
            (5, 11), (6, 12), (11, 12),
            # Leg connections
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        
        self.stats = {'total_poses': 0, 'frames_saved': 0}
        
    def draw_clean_pose(self, frame: np.ndarray, pose, person_id: int):
        """Draw clean pose overlay with minimal text"""
        points = pose.points
        
        # Draw skeleton connections
        for connection in self.pose_connections:
            pt1_idx, pt2_idx = connection
            if pt1_idx < len(points) and pt2_idx < len(points):
                pt1 = points[pt1_idx]
                pt2 = points[pt2_idx]
                
                if pt1[2] > 0.3 and pt2[2] > 0.3:
                    x1, y1 = int(pt1[0]), int(pt1[1])
                    x2, y2 = int(pt2[0]), int(pt2[1])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Yellow skeleton
        
        # Draw keypoints with specific colors
        for i, point in enumerate(points):
            if point[2] > 0.3:
                x, y = int(point[0]), int(point[1])
                
                if i == 0:  # nose
                    color, radius = (255, 0, 255), 6  # Magenta
                elif i in [5, 6]:  # shoulders
                    color, radius = (0, 255, 0), 7    # Green
                elif i in [9, 10]:  # wrists
                    color, radius = (0, 0, 255), 8    # Red
                else:
                    color, radius = (255, 255, 0), 5  # Cyan
                
                cv2.circle(frame, (x, y), radius, color, -1)
                cv2.circle(frame, (x, y), radius + 1, (255, 255, 255), 1)  # White outline
        
        # Minimal person ID (small, unobtrusive)
        x1, y1, x2, y2 = pose.bbox
        cv2.putText(frame, f"{person_id}", (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame: np.ndarray, frame_num: int):
        """Process frame with clean pose visualization"""
        # Detect all poses in the full frame
        poses = self.pose_detector.detect_poses(frame)
        self.stats['total_poses'] += len(poses)
        
        # Draw all detected poses cleanly
        for i, pose in enumerate(poses):
            frame = self.draw_clean_pose(frame, pose, i + 1)
        
        # Minimal stats in corner (small and unobtrusive)
        cv2.putText(frame, f"Frame {frame_num} | {len(poses)} people", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
        
    def generate_clean_frames(self, num_frames: int = 15):
        """Generate clean pose frames with minimal overlays"""
        print(f"\n🎬 CLEAN POSE VISUALIZATION")
        print("=" * 40)
        print(f"📸 Saving {num_frames} clean frames to ../outputs/videos/clean_pose_frames/")
        print("🎨 Minimal overlays - focus on pose visualization")
        
        # Create output directory
        output_dir = "../outputs/videos/clean_pose_frames"
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        
        try:
            for i in range(num_frames):
                # Get frame (spread across first 2 minutes)
                frame_interval = 120 // num_frames
                original_frame_num = i * frame_interval * 30
                
                frame = self.video_processor.get_frame(original_frame_num)
                if frame is None:
                    print(f"Warning: Could not read frame {original_frame_num}")
                    continue
                
                # Process frame with clean pose visualization
                processed_frame = self.process_frame(frame.copy(), i + 1)
                
                # Save frame as image
                output_path = f"{output_dir}/clean_pose_{i+1:03d}.jpg"
                success = cv2.imwrite(output_path, processed_frame)
                
                if success:
                    self.stats['frames_saved'] += 1
                    poses = self.pose_detector.detect_poses(frame)
                    print(f"  ✅ Frame {i+1:2d}/{num_frames}: {len(poses)} poses → {os.path.basename(output_path)}")
                else:
                    print(f"  ❌ Failed to save frame {i+1}")
                    
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            self.video_processor.release()
        
        total_time = time.time() - start_time
        
        print(f"\n📊 Clean Frame Generation Summary:")
        print(f"   Frames saved: {self.stats['frames_saved']}/{num_frames}")
        print(f"   Total poses detected: {self.stats['total_poses']}")
        print(f"   Average poses per frame: {self.stats['total_poses']/self.stats['frames_saved']:.1f}" if self.stats['frames_saved'] > 0 else "0")
        print(f"   Processing time: {total_time:.1f}s")
        print(f"   Output directory: {output_dir}")
        
        if self.stats['frames_saved'] > 0:
            print(f"\n✅ Clean pose visualization complete!")
            print(f"🎨 Minimal overlays - poses are the main focus")
            print(f"🔍 Color code: Red=Wrists, Green=Shoulders, Magenta=Nose, Yellow=Skeleton")

def main():
    print("🎯 CLEAN YOLOV8 POSE VISUALIZATION")
    print("Minimal overlays - focus on the poses themselves")
    
    demo = CleanPoseDemo("../data/video_salon_poco_gente.MP4")
    demo.generate_clean_frames(num_frames=15)
    
    print(f"\n🎉 Clean Demo Complete!")
    print(f"📸 Pure pose visualization with minimal distractions")

if __name__ == "__main__":
    main()