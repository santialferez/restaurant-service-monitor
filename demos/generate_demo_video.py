#!/usr/bin/env python3
"""Generate comprehensive video demo with all GPU-optimized detections and interactions"""

import sys
import os
import warnings
import torch
import cv2
import numpy as np
import time
from collections import defaultdict, deque
import json
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main_gpu import RestaurantAnalyzerGPU
from src.core.video_processor import VideoProcessor

class VideoDemo:
    def __init__(self, video_path: str, output_path: str = "demo_output_2min.mp4"):
        self.video_path = video_path
        self.output_path = output_path
        
        # Initialize video processor
        self.video_processor = VideoProcessor(
            video_path=video_path,
            skip_frames=15,  # Process every 15th frame for 2 FPS
            resize_factor=0.5
        )
        
        # Initialize GPU analyzer with TensorRT disabled for faster demo
        self.analyzer = RestaurantAnalyzerGPU(
            video_path=video_path,
            skip_frames=15,
            resize_factor=0.5
        )
        
        # Disable TensorRT for faster demo generation
        if hasattr(self.analyzer.person_tracker, 'tracker'):
            self.analyzer.person_tracker.use_tensorrt = False
        
        # Colors for visualization
        self.colors = {
            'waiter': (0, 255, 0),      # Green
            'customer': (255, 0, 0),    # Blue  
            'unknown': (0, 255, 255),   # Yellow
            'gesture': (255, 0, 255),   # Magenta
            'table': (255, 255, 0),     # Cyan
            'trajectory': (128, 128, 128)  # Gray
        }
        
        # Tracking data for visualization
        self.trajectories = defaultdict(lambda: deque(maxlen=50))
        self.gesture_history = defaultdict(lambda: deque(maxlen=10))
        self.table_interactions = []
        self.frame_stats = []
        
        # Performance tracking
        self.processing_times = []
        self.detection_counts = []
        
    def draw_person_box(self, frame: np.ndarray, person, frame_num: int) -> np.ndarray:
        """Draw bounding box and info for a person"""
        x1, y1, x2, y2 = person.bbox
        person_type = person.person_type
        color = self.colors.get(person_type, self.colors['unknown'])
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw person ID and type
        label = f"ID:{person.id} {person_type.upper()}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background for text
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0] + 10, y1), color, -1)
        
        # Text
        cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw confidence score
        conf_text = f"{person.confidence:.2f}"
        cv2.putText(frame, conf_text, (x1 + 5, y2 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def draw_trajectory(self, frame: np.ndarray, person_id: int, center: Tuple[int, int]) -> np.ndarray:
        """Draw tracking trajectory for a person"""
        self.trajectories[person_id].append(center)
        
        if len(self.trajectories[person_id]) > 1:
            points = list(self.trajectories[person_id])
            for i in range(1, len(points)):
                # Fade trajectory color based on age
                alpha = i / len(points)
                color = tuple(int(c * alpha) for c in self.colors['trajectory'])
                cv2.line(frame, points[i-1], points[i], color, 2)
        
        return frame
    
    def draw_tables(self, frame: np.ndarray) -> np.ndarray:
        """Draw table positions"""
        for table_id, table in self.analyzer.table_mapper.tables.items():
            x, y, w, h = table.area
            
            # Draw table area
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['table'], 2)
            
            # Draw table number
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.putText(frame, f"T{table_id}", (center_x - 10, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['table'], 2)
        
        return frame
    
    def draw_gestures(self, frame: np.ndarray, persons: Dict, frame_num: int) -> np.ndarray:
        """Draw gesture detection indicators"""
        # Process gesture detection for each person
        for person_id, person in persons.items():
            # Get person's region for gesture analysis
            x1, y1, x2, y2 = person.bbox
            person_roi = frame[y1:y2, x1:x2]
            
            if person_roi.size > 0:
                # Use the GPU gesture detector
                gestures = self.analyzer.gesture_detector.detect_gestures_batch([person_roi])
                
                if gestures and gestures[0]:  # If gestures detected
                    # Draw gesture indicator
                    cv2.circle(frame, person.center, 30, self.colors['gesture'], 3)
                    cv2.putText(frame, "GESTURE", (x1, y1 - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['gesture'], 2)
                    
                    # Record gesture
                    self.gesture_history[person_id].append(frame_num)
        
        return frame
    
    def draw_table_interactions(self, frame: np.ndarray, persons: Dict, frame_num: int) -> np.ndarray:
        """Draw table interaction indicators"""
        for person_id, person in persons.items():
            # Check if person is near any table
            person_center = person.center
            
            for table_id, table in self.analyzer.table_mapper.tables.items():
                table_center = (table.area[0] + table.area[2]//2, table.area[1] + table.area[3]//2)
                distance = np.sqrt((person_center[0] - table_center[0])**2 + 
                                 (person_center[1] - table_center[1])**2)
                
                # If person is close to table (within 80 pixels)
                if distance < 80:
                    # Draw connection line
                    cv2.line(frame, person_center, table_center, (0, 255, 255), 2)
                    
                    # Draw interaction indicator
                    mid_point = ((person_center[0] + table_center[0])//2, 
                               (person_center[1] + table_center[1])//2)
                    cv2.circle(frame, mid_point, 8, (0, 255, 255), -1)
                    
                    # Record interaction
                    interaction = {
                        'frame': frame_num,
                        'person_id': person_id,
                        'person_type': person.person_type,
                        'table_id': table_id,
                        'distance': distance
                    }
                    self.table_interactions.append(interaction)
        
        return frame
    
    def draw_statistics_panel(self, frame: np.ndarray, persons: Dict, frame_num: int, 
                            processing_time: float) -> np.ndarray:
        """Draw real-time statistics panel"""
        h, w = frame.shape[:2]
        
        # Statistics background
        panel_h = 200
        cv2.rectangle(frame, (10, 10), (350, panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, panel_h), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "RESTAURANT SERVICE MONITOR - GPU", (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_pos = 60
        line_height = 20
        
        # Person counts
        waiters = sum(1 for p in persons.values() if p.person_type == 'waiter')
        customers = sum(1 for p in persons.values() if p.person_type == 'customer')
        unknown = sum(1 for p in persons.values() if p.person_type == 'unknown')
        
        cv2.putText(frame, f"Total Persons: {len(persons)}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        
        cv2.putText(frame, f"Waiters: {waiters}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += line_height
        
        cv2.putText(frame, f"Customers: {customers}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        y_pos += line_height
        
        cv2.putText(frame, f"Unknown: {unknown}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_pos += line_height
        
        # Performance metrics
        fps = 1.0 / processing_time if processing_time > 0 else 0
        cv2.putText(frame, f"Processing FPS: {fps:.1f}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_pos += line_height
        
        # GPU info
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() // 1024 // 1024  # MB
            cv2.putText(frame, f"GPU Memory: {gpu_memory}MB", (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_pos += line_height
        
        # Table interactions count
        recent_interactions = sum(1 for i in self.table_interactions 
                                if frame_num - i['frame'] < 30)  # Last 1 second
        cv2.putText(frame, f"Table Interactions: {recent_interactions}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        
        # Gesture count
        recent_gestures = sum(1 for person_id, history in self.gesture_history.items() 
                            if any(frame_num - f < 30 for f in history))
        cv2.putText(frame, f"Recent Gestures: {recent_gestures}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Frame info
        time_seconds = frame_num / 2.0  # 2 FPS processing
        minutes = int(time_seconds // 60)
        seconds = int(time_seconds % 60)
        cv2.putText(frame, f"Time: {minutes:02d}:{seconds:02d}", (250, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def process_frame(self, frame: np.ndarray, frame_num: int) -> Tuple[np.ndarray, Dict]:
        """Process a single frame with all visualizations"""
        start_time = time.time()
        
        # Get person detections and tracking
        persons = self.analyzer.person_tracker.update_tracks(frame, frame_num)
        
        # Draw visualizations
        annotated_frame = frame.copy()
        
        # Draw tables first (background)
        annotated_frame = self.draw_tables(annotated_frame)
        
        # Draw person detections and tracking
        for person_id, person in persons.items():
            # Draw bounding box and info
            annotated_frame = self.draw_person_box(annotated_frame, person, frame_num)
            
            # Draw trajectory
            annotated_frame = self.draw_trajectory(annotated_frame, person_id, person.center)
        
        # Draw interactions
        annotated_frame = self.draw_table_interactions(annotated_frame, persons, frame_num)
        
        # Draw gestures
        annotated_frame = self.draw_gestures(annotated_frame, persons, frame_num)
        
        processing_time = time.time() - start_time
        
        # Draw statistics panel
        annotated_frame = self.draw_statistics_panel(annotated_frame, persons, frame_num, processing_time)
        
        # Record statistics
        stats = {
            'frame_num': frame_num,
            'processing_time': processing_time,
            'person_count': len(persons),
            'waiter_count': sum(1 for p in persons.values() if p.person_type == 'waiter'),
            'customer_count': sum(1 for p in persons.values() if p.person_type == 'customer')
        }
        
        return annotated_frame, stats
    
    def generate_video(self, duration_minutes: int = 2):
        """Generate the complete demo video"""
        print(f"🎬 GENERATING COMPREHENSIVE VIDEO DEMO")
        print(f"="*50)
        
        # Video setup
        fps_output = 15  # Output video FPS (smoother playback)
        frames_to_process = int(duration_minutes * 60 * 2)  # 2 FPS processing
        
        # Get video properties
        sample_frame = self.video_processor.get_frame(0)
        if sample_frame is None:
            raise ValueError("Could not read video")
        
        h, w = sample_frame.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps_output, (w, h))
        
        print(f"📹 Output: {self.output_path}")
        print(f"📐 Resolution: {w}x{h}")
        print(f"⏱️  Duration: {duration_minutes} minutes")
        print(f"🎯 Processing {frames_to_process} frames")
        
        # Initialize analyzer
        self.analyzer.calibrate_tables()
        
        print(f"\n🚀 Starting video generation...")
        
        try:
            for i in range(frames_to_process):
                # Calculate frame number in original video (every 15th frame)
                original_frame_num = i * 15
                
                # Get frame
                frame = self.video_processor.get_frame(original_frame_num)
                if frame is None:
                    print(f"⚠️  Could not get frame {original_frame_num}, skipping...")
                    continue
                
                # Process frame with all visualizations
                annotated_frame, stats = self.process_frame(frame, i)
                
                # Write to output video (duplicate frames for smoother playback)
                for _ in range(fps_output // 2):  # Write each frame multiple times
                    out.write(annotated_frame)
                
                # Progress update
                if i % 20 == 0 or i == frames_to_process - 1:
                    progress = (i + 1) / frames_to_process * 100
                    elapsed_time = time.time() - (getattr(self, 'start_time', time.time()))
                    avg_fps = (i + 1) / elapsed_time if elapsed_time > 0 else 0
                    
                    print(f"  [{progress:5.1f}%] Frame {i+1:4d}/{frames_to_process} | "
                          f"Persons: {stats['person_count']:2d} | "
                          f"FPS: {avg_fps:.1f} | "
                          f"Time: {elapsed_time:.1f}s")
                
                self.frame_stats.append(stats)
                
                # Set start time after first frame
                if i == 0:
                    self.start_time = time.time()
        
        finally:
            out.release()
        
        # Generate summary
        self.generate_summary()
        
        print(f"\n✅ Video generation complete!")
        print(f"📁 Output saved: {self.output_path}")
        print(f"📊 Summary saved: demo_summary.json")
    
    def generate_summary(self):
        """Generate analysis summary"""
        if not self.frame_stats:
            return
        
        # Calculate statistics
        total_frames = len(self.frame_stats)
        avg_processing_time = np.mean([s['processing_time'] for s in self.frame_stats])
        avg_person_count = np.mean([s['person_count'] for s in self.frame_stats])
        max_persons = max([s['person_count'] for s in self.frame_stats])
        
        # Count unique persons
        unique_persons = set()
        for person_id in self.trajectories:
            unique_persons.add(person_id)
        
        summary = {
            'video_info': {
                'output_path': self.output_path,
                'duration_minutes': 2,
                'total_frames_processed': total_frames,
                'processing_fps': 2.0
            },
            'performance': {
                'avg_processing_time_per_frame': avg_processing_time,
                'avg_fps': 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
                'total_processing_time': sum([s['processing_time'] for s in self.frame_stats])
            },
            'detection_statistics': {
                'unique_persons_detected': len(unique_persons),
                'avg_persons_per_frame': avg_processing_time,
                'max_persons_in_frame': max_persons,
                'total_table_interactions': len(self.table_interactions),
                'total_gestures_detected': sum(len(history) for history in self.gesture_history.values())
            },
            'interactions': {
                'table_interactions': self.table_interactions[-50:],  # Last 50 interactions
                'gesture_detections': dict(self.gesture_history)
            }
        }
        
        # Save summary
        with open('../outputs/reports/demo_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

def main():
    """Main function to generate video demo"""
    print("🚀 GPU-OPTIMIZED RESTAURANT SERVICE MONITORING")
    print("🎬 COMPREHENSIVE VIDEO DEMO GENERATOR")
    print("="*60)
    
    # Create video demo
    demo = VideoDemo(
        video_path="../data/video_salon_poco_gente.MP4",
        output_path="../outputs/videos/restaurant_demo_gpu_2min.mp4"
    )
    
    # Generate 2-minute video
    demo.generate_video(duration_minutes=2)
    
    print(f"\n🎉 Demo video generation complete!")
    print(f"../outputs/videos/🎬 Video: restaurant_demo_gpu_2min.mp4")
    print(f"📊 Summary: demo_summary.json")

if __name__ == "__main__":
    main()