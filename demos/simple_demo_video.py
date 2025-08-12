#!/usr/bin/env python3
"""Generate simple but comprehensive video demo with GPU-optimized tracking"""

import sys
import os
import warnings
import torch
import cv2
import numpy as np
import time
from collections import defaultdict, deque
import json
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.person_tracker_gpu import PersonTrackerGPU
from src.core.video_processor import VideoProcessor
from src.core.gesture_detector_gpu import GestureDetectorGPU

class SimpleVideoDemo:
    def __init__(self, video_path: str, output_path: str = "../outputs/videos/restaurant_demo_simple.mp4"):
        self.video_path = video_path
        self.output_path = output_path
        
        # Initialize video processor
        self.video_processor = VideoProcessor(
            video_path=video_path,
            skip_frames=15,  # Process every 15th frame for 2 FPS
            resize_factor=0.5
        )
        
        # Initialize GPU tracker WITHOUT TensorRT for faster startup
        print("🚀 Initializing GPU tracker (no TensorRT for faster demo)...")
        self.person_tracker = PersonTrackerGPU(
            model_size='yolov8m.pt',
            conf_threshold=0.6,  # Optimized threshold
            batch_size=1,
            use_tensorrt=False,  # Disable for faster demo
            use_half_precision=True
        )
        
        # Initialize gesture detector
        print("🤖 Initializing gesture detector...")
        self.gesture_detector = GestureDetectorGPU()
        
        # Colors for visualization
        self.colors = {
            'waiter': (0, 255, 0),      # Green
            'customer': (255, 0, 0),    # Blue  
            'unknown': (0, 255, 255),   # Yellow
            'gesture': (255, 0, 255),   # Magenta
            'table': (255, 255, 0),     # Cyan
            'trajectory': (128, 128, 128)  # Gray
        }
        
        # Tracking data
        self.trajectories = defaultdict(lambda: deque(maxlen=30))
        self.gesture_frames = defaultdict(list)
        
        # Simple table positions (hardcoded for demo)
        self.tables = {
            1: (100, 100, 80, 60),   # x, y, w, h
            2: (300, 120, 80, 60),
            3: (500, 140, 80, 60),
            4: (150, 300, 80, 60),
            5: (350, 320, 80, 60),
            6: (550, 340, 80, 60),
        }
        
    def draw_person_info(self, frame: np.ndarray, person, frame_num: int) -> np.ndarray:
        """Draw person bounding box, ID, and info"""
        x1, y1, x2, y2 = person.bbox
        person_type = person.person_type
        color = self.colors.get(person_type, self.colors['unknown'])
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Person info
        label = f"ID{person.id} {person_type.upper()}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Confidence (handle None values)
        confidence = person.confidence if person.confidence is not None else 0.5
        conf_text = f"{confidence:.2f}"
        cv2.putText(frame, conf_text, (x1, y2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def draw_trajectory(self, frame: np.ndarray, person_id: int, center: tuple) -> np.ndarray:
        """Draw tracking trajectory"""
        self.trajectories[person_id].append(center)
        
        if len(self.trajectories[person_id]) > 1:
            points = list(self.trajectories[person_id])
            for i in range(1, len(points)):
                alpha = i / len(points)
                color = tuple(int(c * alpha) for c in self.colors['trajectory'])
                cv2.line(frame, points[i-1], points[i], color, 2)
        
        return frame
    
    def draw_tables(self, frame: np.ndarray) -> np.ndarray:
        """Draw simple table indicators"""
        for table_id, (x, y, w, h) in self.tables.items():
            # Draw table
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['table'], 2)
            
            # Table number
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.putText(frame, f"T{table_id}", (center_x - 15, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['table'], 2)
        
        return frame
    
    def detect_gestures(self, frame: np.ndarray, persons: dict, frame_num: int) -> np.ndarray:
        """Simple gesture detection visualization"""
        for person_id, person in persons.items():
            x1, y1, x2, y2 = person.bbox
            
            # Simple gesture detection based on bounding box aspect ratio and position
            # This is a simplified version for demo purposes
            height = y2 - y1
            width = x2 - x1
            
            # Heuristic: if person is tall (hands up) or in upper area
            if height > width * 1.5 or y1 < frame.shape[0] * 0.3:
                # Mark as potential gesture
                cv2.circle(frame, person.center, 25, self.colors['gesture'], 3)
                cv2.putText(frame, "GESTURE", (x1, y1 - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['gesture'], 2)
                
                self.gesture_frames[person_id].append(frame_num)
        
        return frame
    
    def check_table_interactions(self, frame: np.ndarray, persons: dict) -> np.ndarray:
        """Check and visualize table interactions"""
        for person_id, person in persons.items():
            person_center = person.center
            
            # Check proximity to tables
            for table_id, (tx, ty, tw, th) in self.tables.items():
                table_center = (tx + tw//2, ty + th//2)
                
                distance = np.sqrt((person_center[0] - table_center[0])**2 + 
                                 (person_center[1] - table_center[1])**2)
                
                # If person is close to table (within 100 pixels)
                if distance < 100:
                    # Draw interaction line
                    cv2.line(frame, person_center, table_center, (0, 255, 255), 2)
                    
                    # Interaction indicator
                    mid_point = ((person_center[0] + table_center[0])//2, 
                               (person_center[1] + table_center[1])//2)
                    cv2.circle(frame, mid_point, 8, (0, 255, 255), -1)
                    
                    # Distance text
                    cv2.putText(frame, f"{distance:.0f}px", 
                               (mid_point[0] - 20, mid_point[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_stats_panel(self, frame: np.ndarray, persons: dict, frame_num: int, 
                        processing_time: float) -> np.ndarray:
        """Draw statistics overlay"""
        h, w = frame.shape[:2]
        
        # Stats panel background
        cv2.rectangle(frame, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 180), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "GPU RESTAURANT MONITOR DEMO", (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_pos = 60
        
        # Person counts
        waiters = sum(1 for p in persons.values() if p.person_type == 'waiter')
        customers = sum(1 for p in persons.values() if p.person_type == 'customer')
        
        cv2.putText(frame, f"Total Persons: {len(persons)}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        
        cv2.putText(frame, f"Waiters: {waiters} | Customers: {customers}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        
        # Performance
        fps = 1.0 / processing_time if processing_time > 0 else 0
        cv2.putText(frame, f"Processing FPS: {fps:.1f}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_pos += 20
        
        # GPU Memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() // 1024 // 1024
            cv2.putText(frame, f"GPU Memory: {gpu_mem}MB", (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_pos += 20
        
        # Time
        time_sec = frame_num / 2.0
        minutes = int(time_sec // 60)
        seconds = int(time_sec % 60)
        cv2.putText(frame, f"Time: {minutes:02d}:{seconds:02d}", (250, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def generate_video(self, duration_minutes: int = 2):
        """Generate the demo video"""
        print(f"\n🎬 GENERATING {duration_minutes}-MINUTE GPU DEMO VIDEO")
        print(f"="*50)
        
        frames_to_process = duration_minutes * 60 * 2  # 2 FPS processing
        
        # Video setup
        sample_frame = self.video_processor.get_frame(0)
        if sample_frame is None:
            raise ValueError("Could not read video")
        
        h, w = sample_frame.shape[:2]
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, 10, (w, h))  # 10 FPS output
        
        print(f"📹 Output: {self.output_path}")
        print(f"📐 Resolution: {w}x{h}")
        print(f"🎯 Processing {frames_to_process} frames")
        
        stats_list = []
        start_time = time.time()
        
        try:
            for i in range(frames_to_process):
                frame_start = time.time()
                
                # Get frame (every 15th frame from original video)
                original_frame_num = i * 15
                frame = self.video_processor.get_frame(original_frame_num)
                
                if frame is None:
                    print(f"⚠️  Frame {original_frame_num} not found, skipping...")
                    continue
                
                # Person tracking
                persons = self.person_tracker.update_tracks(frame, i)
                
                # Create annotated frame
                demo_frame = frame.copy()
                
                # Draw tables (background)
                demo_frame = self.draw_tables(demo_frame)
                
                # Draw person tracking
                for person_id, person in persons.items():
                    demo_frame = self.draw_person_info(demo_frame, person, i)
                    demo_frame = self.draw_trajectory(demo_frame, person_id, person.center)
                
                # Draw interactions
                demo_frame = self.check_table_interactions(demo_frame, persons)
                demo_frame = self.detect_gestures(demo_frame, persons, i)
                
                # Calculate processing time
                processing_time = time.time() - frame_start
                
                # Draw stats panel
                demo_frame = self.draw_stats_panel(demo_frame, persons, i, processing_time)
                
                # Write frame multiple times for smooth playback
                for _ in range(5):  # 5 repeats = 10 FPS output from 2 FPS processing
                    out.write(demo_frame)
                
                # Progress update
                if i % 10 == 0 or i == frames_to_process - 1:
                    progress = (i + 1) / frames_to_process * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / (i + 1)) * (frames_to_process - i - 1)
                    
                    print(f"  [{progress:5.1f}%] Frame {i+1:3d}/{frames_to_process} | "
                          f"Persons: {len(persons):2d} | "
                          f"ETA: {eta:.1f}s")
                
                # Record stats
                stats_list.append({
                    'frame': i,
                    'persons': len(persons),
                    'waiters': sum(1 for p in persons.values() if p.person_type == 'waiter'),
                    'customers': sum(1 for p in persons.values() if p.person_type == 'customer'),
                    'processing_time': processing_time
                })
        
        except KeyboardInterrupt:
            print("\n⚠️  Demo generation interrupted!")
        
        finally:
            out.release()
        
        total_time = time.time() - start_time
        
        # Summary
        print(f"\n✅ Demo video complete!")
        print(f"📁 Saved: {self.output_path}")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"🚀 Average FPS: {len(stats_list) / total_time:.1f}")
        
        # Save summary
        summary = {
            'video_info': {
                'output_path': self.output_path,
                'duration_minutes': duration_minutes,
                'frames_processed': len(stats_list),
                'total_processing_time': total_time
            },
            'detection_stats': {
                'unique_persons': len(self.trajectories),
                'avg_persons_per_frame': np.mean([s['persons'] for s in stats_list]),
                'max_persons': max([s['persons'] for s in stats_list]) if stats_list else 0,
                'total_gesture_detections': sum(len(frames) for frames in self.gesture_frames.values())
            }
        }
        
        with open('../outputs/reports/demo_video_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Summary saved: demo_video_summary.json")

def main():
    print("🚀 SIMPLE GPU RESTAURANT DEMO")
    print("🎬 Video Generator with All Visualizations")
    
    # Create and run demo
    demo = SimpleVideoDemo(
        video_path="../data/video_salon_poco_gente.MP4",
        output_path="../outputs/videos/restaurant_gpu_demo_2min.mp4"
    )
    
    # Generate full 2-minute demo
    demo.generate_video(duration_minutes=2)
    
    print(f"\n🎉 GPU Demo Complete!")
    print(f"▶️  Play: restaurant_gpu_demo_2min.mp4")

if __name__ == "__main__":
    main()