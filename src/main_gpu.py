#!/usr/bin/env python3
"""
GPU-Optimized Restaurant Analysis System
High-performance version using CUDA acceleration and batch processing
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# GPU-optimized imports
from core.person_tracker_gpu import PersonTrackerGPU
from core.gesture_detector_gpu import GestureDetectorGPU  
from analytics.movement_analyzer_gpu import MovementAnalyzerGPU
from core.video_processor import VideoProcessor
from core.table_mapper import TableMapper
from analytics.service_metrics import ServiceMetricsCalculator

# Regular imports for components not yet GPU-optimized
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RestaurantAnalyzerGPU:
    """GPU-accelerated restaurant service analysis system"""
    
    def __init__(self,
                 video_path: str,
                 output_dir: str = "output",
                 skip_frames: int = 15,
                 resize_factor: float = 0.5,
                 batch_size: int = 8,
                 use_tensorrt: bool = True,
                 use_half_precision: bool = True,
                 device: Optional[str] = None):
        
        # GPU configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"RestaurantAnalyzerGPU initializing on device: {self.device}")
        
        # Configuration
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.skip_frames = skip_frames
        self.resize_factor = resize_factor
        self.batch_size = batch_size
        
        # Performance settings
        self.use_tensorrt = use_tensorrt
        self.use_half_precision = use_half_precision
        
        # Initialize components
        self._initialize_components()
        
        # Analysis results
        self.analysis_results = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_frames_processed': 0,
            'total_processing_time': 0,
            'gpu_memory_usage': [],
            'batch_processing_times': [],
            'component_performance': {}
        }
        
        logger.info("RestaurantAnalyzerGPU initialization complete")
    
    def _initialize_components(self):
        """Initialize all analysis components with GPU optimization"""
        logger.info("Initializing GPU-optimized components...")
        
        # Video processor (CPU-based, but optimized for batch feeding)
        self.video_processor = VideoProcessor(
            self.video_path,
            skip_frames=self.skip_frames,
            resize_factor=self.resize_factor
        )
        
        # GPU-optimized person tracker with TensorRT
        self.person_tracker = PersonTrackerGPU(
            model_size='yolov8m.pt',
            conf_threshold=0.5,
            movement_threshold=3.0,
            batch_size=self.batch_size,
            use_tensorrt=self.use_tensorrt,
            use_half_precision=self.use_half_precision
        )
        
        # GPU-optimized gesture detector
        self.gesture_detector = GestureDetectorGPU(
            device=str(self.device),
            batch_size=self.batch_size,
            min_detection_confidence=0.5,
            hand_raise_threshold=0.7
        )
        
        # GPU-optimized movement analyzer
        frame_shape = (
            int(self.video_processor.height * self.resize_factor),
            int(self.video_processor.width * self.resize_factor)
        )
        self.movement_analyzer = MovementAnalyzerGPU(
            frame_shape=frame_shape,
            device=str(self.device)
        )
        
        # Standard components (will be GPU-optimized in future versions)
        self.table_mapper = TableMapper()
        self.service_metrics = ServiceMetricsCalculator()
        
        logger.info("All components initialized successfully")
    
    def calibrate_tables(self):
        """Set up table configuration"""
        logger.info("Starting table calibration...")
        
        # Auto-detection
        sample_frame = self.video_processor.get_frame(0)
        if sample_frame is not None:
            tables = self.table_mapper.auto_detect_tables(sample_frame)
            logger.info(f"Auto-detected {len(tables)} potential tables")
        
        if len(self.table_mapper.tables) == 0:
            logger.info("Auto-detection failed. Manual calibration required.")
            # Create default table configuration
            frame_width = int(self.video_processor.width * self.resize_factor)
            frame_height = int(self.video_processor.height * self.resize_factor)
            
            # Create a grid of default tables
            from core.table_mapper import Table
            tables_x = 4
            tables_y = 3 
            table_width = frame_width // tables_x
            table_height = frame_height // tables_y
            
            for i in range(tables_y):
                for j in range(tables_x):
                    table_id = i * tables_x + j
                    center_x = j * table_width + table_width // 2
                    center_y = i * table_height + table_height // 2
                    margin = min(table_width, table_height) // 6
                    
                    # Create table with bounding box
                    table = Table(
                        id=table_id,
                        center=(center_x, center_y),
                        bbox=(center_x - margin, center_y - margin, center_x + margin, center_y + margin),
                        polygon=[]
                    )
                    self.table_mapper.tables[table_id] = table
            
            logger.info(f"Created {len(self.table_mapper.tables)} default tables")
        
        # Save table configuration
        table_config_path = self.output_dir / "table_config.json"
        self.table_mapper.save_configuration(str(table_config_path))
        logger.info(f"Table configuration saved to {table_config_path}")
    
    def process_video_batch(self):
        """Process video using GPU-accelerated batch processing"""
        logger.info("Starting GPU-accelerated video processing...")
        
        # Batch processing setup
        frame_batch = []
        frame_number_batch = []
        timestamp_batch = []
        
        total_start_time = time.time()
        
        for frame_num, frame, timestamp in self.video_processor.process_frames():
            # Add frame to batch
            frame_batch.append(frame)
            frame_number_batch.append(frame_num)
            timestamp_batch.append(timestamp)
            
            # Process when batch is full
            if len(frame_batch) >= self.batch_size:
                self._process_frame_batch(frame_batch, frame_number_batch, timestamp_batch)
                
                # Clear batch
                frame_batch.clear()
                frame_number_batch.clear()
                timestamp_batch.clear()
                
                # Update performance metrics
                self.performance_metrics['total_frames_processed'] += self.batch_size
                
                # Memory optimization every 100 batches
                if self.performance_metrics['total_frames_processed'] % (100 * self.batch_size) == 0:
                    self._optimize_gpu_memory()
        
        # Process remaining frames in batch
        if frame_batch:
            self._process_frame_batch(frame_batch, frame_number_batch, timestamp_batch)
            self.performance_metrics['total_frames_processed'] += len(frame_batch)
        
        # Final processing
        self._finalize_processing(timestamp_batch[-1] if timestamp_batch else 0)
        
        total_processing_time = time.time() - total_start_time
        self.performance_metrics['total_processing_time'] = total_processing_time
        
        logger.info(f"Video processing complete in {total_processing_time:.2f}s")
        logger.info(f"Processed {self.performance_metrics['total_frames_processed']} frames")
        logger.info(f"Average FPS: {self.performance_metrics['total_frames_processed'] / total_processing_time:.2f}")
    
    def _process_frame_batch(self, frames: List[np.ndarray], frame_numbers: List[int], timestamps: List[float]):
        """Process a batch of frames with GPU acceleration"""
        batch_start_time = time.time()
        
        # Person detection and tracking (batch processing)
        batch_persons = self.person_tracker.update_tracks_batch(frames, frame_numbers)
        
        # Process each frame's results
        for frame_idx, (frame, frame_num, timestamp, frame_persons) in enumerate(
            zip(frames, frame_numbers, timestamps, batch_persons)
        ):
            # Gesture detection for this frame
            gesture_events = self.gesture_detector.detect_hand_raise_batch(
                frame, frame_persons, frame_num, timestamp
            )
            
            # Process gesture events
            for event in gesture_events:
                if event.position:
                    table_id = self.table_mapper.assign_person_to_table(event.position)
                    event.table_id = table_id
                self.service_metrics.add_request_event(
                    timestamp, event.person_id, event.table_id
                )
            
            # Check for gesture responses
            waiters = self.person_tracker.get_waiters()
            for gesture in self.gesture_detector.gesture_events:
                if not gesture.responded:
                    self.gesture_detector.check_gesture_response(gesture, waiters, timestamp)
            
            # Update movement analysis
            for person_id, person in frame_persons.items():
                self.movement_analyzer.update_position(person_id, person.center, timestamp)
                
                # Update table visits for waiters
                if person.person_type == 'waiter':
                    self.table_mapper.update_table_visits(person.center, timestamp)
        
        # Track batch processing time
        batch_time = time.time() - batch_start_time
        self.performance_metrics['batch_processing_times'].append(batch_time)
        
        # Track GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            self.performance_metrics['gpu_memory_usage'].append(memory_used)
    
    def _finalize_processing(self, final_timestamp: float):
        """Finalize all processing and cleanup"""
        logger.info("Finalizing analysis...")
        
        # Finalize active gestures
        finalized_gestures = self.gesture_detector.finalize_active_gestures(final_timestamp)
        
        # Process finalized gestures
        for gesture in finalized_gestures:
            if gesture.position:
                table_id = self.table_mapper.assign_person_to_table(gesture.position)
                gesture.table_id = table_id
            
            self.service_metrics.add_request_event(
                gesture.timestamp, gesture.person_id, gesture.table_id
            )
            
            # Check for responses
            waiters = self.person_tracker.get_waiters()
            self.gesture_detector.check_gesture_response(gesture, waiters, final_timestamp)
        
        # Flush movement analyzer buffers
        self.movement_analyzer.flush_buffers()
        
        logger.info("Processing finalization complete")
    
    def _optimize_gpu_memory(self):
        """Optimize GPU memory usage during processing"""
        logger.info("Optimizing GPU memory...")
        
        # Clean up old tracking data
        current_frame = self.performance_metrics['total_frames_processed']
        self.person_tracker.clear_old_tracks(current_frame)
        
        # Clean up old gestures
        if self.performance_metrics['batch_processing_times']:
            current_time = time.time()
            self.gesture_detector.clear_old_gestures(current_time)
        
        # Optimize movement analyzer memory
        self.movement_analyzer.optimize_memory()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("GPU memory optimization complete")
    
    def generate_results(self):
        """Generate comprehensive analysis results"""
        logger.info("Generating analysis results...")
        
        # Get component performance stats
        self.performance_metrics['component_performance'] = {
            'person_tracker': self.person_tracker.get_performance_stats(),
            'gesture_detector': self.gesture_detector.get_performance_stats(),
            'gpu_memory': self.movement_analyzer.get_gpu_memory_stats()
        }
        
        # Service metrics
        service_stats = self.service_metrics.calculate_metrics()
        
        # Movement patterns
        movement_patterns = self.movement_analyzer.calculate_movement_patterns_gpu()
        
        # Gesture statistics
        gesture_stats = {
            'total_gestures': len(self.gesture_detector.gesture_events),
            'responded_gestures': sum(1 for g in self.gesture_detector.gesture_events if g.responded),
            'avg_response_time': np.mean([g.response_time for g in self.gesture_detector.gesture_events 
                                        if g.response_time]) if any(g.response_time for g in self.gesture_detector.gesture_events) else 0
        }
        
        # Table attention metrics
        final_timestamp = self.performance_metrics.get('total_processing_time', time.time())
        table_attention = self.table_mapper.get_table_attention_metrics(final_timestamp)
        
        # Personnel statistics
        all_persons = self.person_tracker.persons
        personnel_stats = {
            'total_persons': len(all_persons),
            'waiters_detected': len([p for p in all_persons.values() if p.person_type == 'waiter']),
            'customers_detected': len([p for p in all_persons.values() if p.person_type == 'customer']),
            'unclassified_persons': len([p for p in all_persons.values() if p.person_type == 'unknown'])
        }
        
        # Compile final results
        self.analysis_results = {
            'video_info': {
                'path': self.video_path,
                'duration': self.video_processor.duration,
                'fps': self.video_processor.fps,
                'total_frames': self.video_processor.total_frames,
                'processed_frames': self.performance_metrics['total_frames_processed']
            },
            'performance_metrics': self.performance_metrics,
            'metrics': service_stats,
            'movement_patterns': movement_patterns,
            'gesture_statistics': gesture_stats,
            'table_attention': table_attention,
            'personnel_statistics': personnel_stats,
            'analysis_timestamp': time.time(),
            'gpu_acceleration': {
                'device': str(self.device),
                'tensorrt_enabled': self.use_tensorrt,
                'half_precision': self.use_half_precision,
                'batch_size': self.batch_size
            }
        }
        
        logger.info("Results generation complete")
    
    def save_results(self):
        """Save analysis results to files"""
        logger.info("Saving analysis results...")
        
        # Save JSON results
        results_path = self.output_dir / "analysis_results_gpu.json"
        with open(results_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Save performance report
        self._save_performance_report()
        
        # Export movement data
        self._export_movement_data()
        
        # Generate GPU-optimized visualizations
        self._generate_gpu_visualizations()
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _save_performance_report(self):
        """Save detailed performance analysis report"""
        report_path = self.output_dir / "performance_report_gpu.txt"
        
        with open(report_path, 'w') as f:
            f.write("GPU-ACCELERATED RESTAURANT ANALYSIS PERFORMANCE REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # System info
            f.write(f"GPU Device: {self.device}\n")
            f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"GPU Name: {torch.cuda.get_device_name(0)}\n")
                f.write(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
            f.write(f"Batch Size: {self.batch_size}\n")
            f.write(f"TensorRT Enabled: {self.use_tensorrt}\n")
            f.write(f"Half Precision: {self.use_half_precision}\n\n")
            
            # Processing stats
            pm = self.performance_metrics
            f.write(f"Total Frames Processed: {pm['total_frames_processed']}\n")
            f.write(f"Total Processing Time: {pm['total_processing_time']:.2f}s\n")
            f.write(f"Average FPS: {pm['total_frames_processed'] / pm['total_processing_time']:.2f}\n")
            
            if pm['batch_processing_times']:
                f.write(f"Average Batch Time: {np.mean(pm['batch_processing_times']):.3f}s\n")
                f.write(f"Batch Throughput: {self.batch_size / np.mean(pm['batch_processing_times']):.2f} FPS\n")
            
            if pm['gpu_memory_usage']:
                f.write(f"Peak GPU Memory: {max(pm['gpu_memory_usage']):.1f} MB\n")
                f.write(f"Average GPU Memory: {np.mean(pm['gpu_memory_usage']):.1f} MB\n")
    
    def _export_movement_data(self):
        """Export movement analysis data"""
        # Regular CSV export
        csv_path = self.output_dir / "movement_data_gpu.csv"
        movement_patterns = self.analysis_results['movement_patterns']
        
        with open(csv_path, 'w') as f:
            f.write("person_id,total_distance,avg_speed,max_speed,coverage_area,path_points\n")
            for person_id, data in movement_patterns.items():
                f.write(f"{person_id},{data['total_distance']:.3f},{data['avg_speed']:.3f},"
                       f"{data['max_speed']:.3f},{data['coverage_area']:.1f},{data['path_points']}\n")
    
    def _generate_gpu_visualizations(self):
        """Generate visualizations using GPU-accelerated data"""
        logger.info("Generating GPU-accelerated visualizations...")
        
        # Save GPU-generated heatmap
        heatmap_path = self.output_dir / "movement_heatmap_gpu.png"
        self.movement_analyzer.save_heatmap_gpu(str(heatmap_path))
        
        # Generate flow field visualization
        flow_field = self.movement_analyzer.generate_flow_field_gpu()
        if flow_field.size > 0:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            plt.quiver(flow_field[:, :, 0], flow_field[:, :, 1])
            plt.title('Movement Flow Field (GPU Generated)')
            plt.xlabel('Grid X')
            plt.ylabel('Grid Y')
            plt.tight_layout()
            
            flow_path = self.output_dir / "flow_field_gpu.png"
            plt.savefig(flow_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("GPU visualizations generated")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'video_processor'):
            self.video_processor.release()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU-Accelerated Restaurant Service Analysis')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--output', default='output_gpu', help='Output directory')
    parser.add_argument('--skip-frames', type=int, default=15, help='Frame skip interval')
    parser.add_argument('--resize', type=float, default=0.5, help='Resize factor for processing')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for GPU processing')
    parser.add_argument('--no-tensorrt', action='store_true', help='Disable TensorRT optimization')
    parser.add_argument('--no-half-precision', action='store_true', help='Disable FP16 precision')
    parser.add_argument('--device', default=None, help='GPU device (cuda:0, cuda:1, etc.)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = RestaurantAnalyzerGPU(
        video_path=args.video_path,
        output_dir=args.output,
        skip_frames=args.skip_frames,
        resize_factor=args.resize,
        batch_size=args.batch_size,
        use_tensorrt=not args.no_tensorrt,
        use_half_precision=not args.no_half_precision,
        device=args.device
    )
    
    try:
        # Run analysis
        analyzer.calibrate_tables()
        analyzer.process_video_batch()
        analyzer.generate_results()
        analyzer.save_results()
        
        print("\n" + "="*60)
        print("GPU-ACCELERATED ANALYSIS COMPLETE!")
        print("="*60)
        
        # Print performance summary
        pm = analyzer.performance_metrics
        print(f"Frames processed: {pm['total_frames_processed']}")
        print(f"Processing time: {pm['total_processing_time']:.2f}s")
        print(f"Average FPS: {pm['total_frames_processed'] / pm['total_processing_time']:.2f}")
        
        if pm['gpu_memory_usage']:
            print(f"Peak GPU memory: {max(pm['gpu_memory_usage']):.1f} MB")
        
        print(f"Results saved to: {analyzer.output_dir}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
    finally:
        # Cleanup
        del analyzer


if __name__ == "__main__":
    main()