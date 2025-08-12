#!/usr/bin/env python3
import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.video_processor import VideoProcessor
from src.core.person_tracker import PersonTracker
from src.core.gesture_detector import GestureDetector
from src.core.table_mapper import TableMapper
from src.analytics.service_metrics import ServiceMetricsCalculator
from src.analytics.movement_analyzer import MovementAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RestaurantAnalyzer:
    def __init__(self, video_path: str, config_path: str = None, output_dir: str = "data/outputs",
                 skip_frames: int = 2, resize_factor: float = 0.5):
        self.video_path = video_path
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_frames = skip_frames
        self.resize_factor = resize_factor
        
        # Initialize components
        logger.info("Initializing Restaurant Analyzer...")
        
        # Video processor
        self.video_processor = VideoProcessor(video_path, skip_frames=self.skip_frames, resize_factor=self.resize_factor)
        
        # Get frame dimensions for other components
        frame_shape = (self.video_processor.height, self.video_processor.width)
        
        # Core components
        self.person_tracker = PersonTracker(model_size='yolov8m.pt', conf_threshold=0.5, movement_threshold=3.0)
        self.gesture_detector = GestureDetector(min_detection_confidence=0.5)
        self.table_mapper = TableMapper(config_path)
        
        # Analytics components
        self.service_metrics = ServiceMetricsCalculator()
        self.movement_analyzer = MovementAnalyzer(frame_shape)
        
        # Results storage
        self.analysis_results = {
            'video_info': self.video_processor.get_video_info(),
            'metrics': {},
            'events': [],
            'waiter_performance': {},
            'table_statistics': {}
        }
        
        logger.info("Initialization complete")
    
    def calibrate_tables(self):
        logger.info("Starting table calibration...")
        
        # Get first frame for calibration
        frame = self.video_processor.get_frame(0)
        
        if frame is None:
            logger.error("Could not get frame for calibration")
            return
        
        # Try auto-detection first
        detected_tables = self.table_mapper.auto_detect_tables(frame)
        
        if detected_tables:
            logger.info(f"Auto-detected {len(detected_tables)} tables")
            for table in detected_tables:
                self.table_mapper.tables[table.id] = table
        else:
            logger.info("Auto-detection failed. Manual calibration required.")
            # In a real application, this would open an interactive window
            # For now, we'll create some default tables
            self._create_default_tables(frame.shape)
        
        # Save table configuration
        config_path = self.output_dir / "table_config.json"
        self.table_mapper.save_configuration(str(config_path))
        logger.info(f"Table configuration saved to {config_path}")
    
    def _create_default_tables(self, frame_shape):
        height, width = frame_shape[:2]
        
        # Create a grid of tables
        rows, cols = 3, 4
        margin = 50
        table_width = (width - 2 * margin) // cols
        table_height = (height - 2 * margin) // rows
        
        table_id = 1
        for row in range(rows):
            for col in range(cols):
                x = margin + col * table_width
                y = margin + row * table_height
                
                center = (x + table_width // 2, y + table_height // 2)
                bbox = (x, y, x + table_width - 20, y + table_height - 20)
                
                from src.core.table_mapper import Table
                table = Table(
                    id=table_id,
                    center=center,
                    bbox=bbox,
                    polygon=[]
                )
                
                self.table_mapper.tables[table_id] = table
                table_id += 1
        
        logger.info(f"Created {len(self.table_mapper.tables)} default tables")
    
    def analyze_video(self):
        logger.info("Starting video analysis...")
        
        # Process video frames
        frame_count = 0
        
        for frame_num, frame, timestamp in self.video_processor.process_frames():
            # Track persons
            persons = self.person_tracker.update_tracks(frame, frame_num)
            
            # Get waiters and customers
            waiters = self.person_tracker.get_waiters()
            customers = self.person_tracker.get_customers()
            
            # Detect hand raises
            gesture_events = self.gesture_detector.detect_hand_raise(
                frame, persons, frame_num, timestamp
            )
            
            # Process gesture events
            for event in gesture_events:
                # Assign table to gesture
                if event.position:
                    table_id = self.table_mapper.assign_person_to_table(event.position)
                    event.table_id = table_id
                
                # Add to service metrics
                self.service_metrics.add_request_event(
                    timestamp, event.person_id, event.table_id
                )
                
                # Store event
                self.analysis_results['events'].append({
                    'type': 'hand_raise',
                    'timestamp': timestamp,
                    'frame': frame_num,
                    'person_id': event.person_id,
                    'table_id': event.table_id,
                    'confidence': event.confidence
                })
            
            # Check for gesture responses
            for gesture in self.gesture_detector.gesture_events:
                if not gesture.responded:
                    self.gesture_detector.check_gesture_response(
                        gesture, waiters, timestamp
                    )
                    
                    if gesture.responded:
                        # Add response to metrics
                        self.service_metrics.add_response_event(
                            gesture, gesture.person_id, timestamp
                        )
            
            # Update movement tracking
            for person_id, person in persons.items():
                self.movement_analyzer.update_position(
                    person_id, person.center, timestamp
                )
                
                # Track waiter visits to tables
                if person.person_type == 'waiter':
                    self.table_mapper.update_table_visits(
                        person.center, timestamp
                    )
                    
                    # Find which table the waiter is visiting
                    for table_id, table in self.table_mapper.tables.items():
                        if table.distance_to_point(person.center) < 100:
                            self.service_metrics.add_table_visit(
                                timestamp, table_id, person_id
                            )
            
            # Save annotated frame periodically
            if frame_num % 300 == 0:  # Every 300 frames
                self._save_annotated_frame(frame, persons, frame_num)
            
            frame_count += 1
        
        logger.info(f"Processed {frame_count} frames")
    
    def _save_annotated_frame(self, frame: np.ndarray, persons: dict, frame_num: int):
        # Draw annotations
        annotated = self.person_tracker.draw_tracks(frame, draw_history=True)
        annotated = self.gesture_detector.draw_gestures(annotated, persons)
        annotated = self.table_mapper.draw_tables(annotated)
        
        # Save frame
        output_path = self.output_dir / f"frame_{frame_num:06d}.jpg"
        cv2.imwrite(str(output_path), annotated)
        logger.debug(f"Saved annotated frame to {output_path}")
    
    def generate_results(self):
        logger.info("Generating analysis results...")
        
        # Calculate metrics
        metrics = self.service_metrics.calculate_metrics()
        self.analysis_results['metrics'] = {
            'avg_response_time': metrics.avg_response_time,
            'min_response_time': metrics.min_response_time,
            'max_response_time': metrics.max_response_time,
            'median_response_time': metrics.median_response_time,
            'response_rate': metrics.response_rate,
            'total_requests': metrics.total_requests,
            'total_responses': metrics.total_responses,
            'avg_table_visit_interval': metrics.avg_table_visit_interval,
            'efficiency_score': metrics.efficiency_score
        }
        
        # Get waiter performance
        self.analysis_results['waiter_performance'] = self.service_metrics.get_waiter_metrics()
        
        # Get table statistics
        table_metrics = self.table_mapper.get_table_attention_metrics(
            self.video_processor.duration
        )
        self.analysis_results['table_statistics'] = table_metrics
        
        # Get gesture statistics
        gesture_stats = self.gesture_detector.get_gesture_statistics()
        self.analysis_results['gesture_statistics'] = gesture_stats
        
        # Generate movement statistics
        movement_stats = {}
        for person_id in self.movement_analyzer.movement_paths.keys():
            patterns = self.movement_analyzer.get_movement_patterns(person_id)
            movement_stats[person_id] = patterns
        self.analysis_results['movement_statistics'] = movement_stats
        
        logger.info("Results generation complete")
    
    def save_results(self):
        logger.info("Saving analysis results...")
        
        # Save JSON results
        results_path = self.output_dir / "analysis_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                return obj
            
            json.dump(convert_types(self.analysis_results), f, indent=2)
        logger.info(f"Results saved to {results_path}")
        
        # Generate and save report
        report = self.service_metrics.generate_summary_report()
        report_path = self.output_dir / "analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
        
        # Export metrics to Excel
        excel_path = self.output_dir / "service_metrics.xlsx"
        self.service_metrics.export_metrics(str(excel_path))
        logger.info(f"Metrics exported to {excel_path}")
        
        # Export movement data
        movement_path = self.output_dir / "movement_data.csv"
        self.movement_analyzer.export_movement_data(str(movement_path))
        logger.info(f"Movement data exported to {movement_path}")
        
        # Generate visualizations
        self._generate_visualizations()
    
    def _generate_visualizations(self):
        logger.info("Generating visualizations...")
        
        # Generate movement heatmap
        sample_frame = self.video_processor.get_frame(0)
        if sample_frame is not None:
            heatmap_overlay = self.movement_analyzer.generate_heatmap_overlay(sample_frame)
            heatmap_path = self.output_dir / "movement_heatmap.jpg"
            cv2.imwrite(str(heatmap_path), heatmap_overlay)
            logger.info(f"Heatmap saved to {heatmap_path}")
            
            # Generate flow field visualization
            flow_viz = self.movement_analyzer.visualize_flow_field(sample_frame)
            flow_path = self.output_dir / "flow_field.jpg"
            cv2.imwrite(str(flow_path), flow_viz)
            logger.info(f"Flow field saved to {flow_path}")
        
        # Generate statistics plots
        plot_path = self.output_dir / "movement_statistics.png"
        fig = self.movement_analyzer.generate_statistics_plot(str(plot_path))
        logger.info(f"Statistics plot saved to {plot_path}")
    
    def run(self):
        try:
            # Calibrate tables
            self.calibrate_tables()
            
            # Analyze video
            self.analyze_video()
            
            # Generate results
            self.generate_results()
            
            # Save results
            self.save_results()
            
            # Print summary
            self.print_summary()
            
            logger.info("Analysis complete!")
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise
        
        finally:
            # Cleanup
            self.video_processor.release()
    
    def print_summary(self):
        print("\n" + "="*60)
        print("RESTAURANT SERVICE ANALYSIS SUMMARY")
        print("="*60)
        
        metrics = self.analysis_results['metrics']
        print(f"\nVideo Duration: {self.video_processor.duration:.1f} seconds")
        print(f"Total Frames Processed: {self.video_processor.total_frames}")
        
        print(f"\nService Metrics:")
        print(f"  - Total Requests: {metrics['total_requests']}")
        print(f"  - Total Responses: {metrics['total_responses']}")
        print(f"  - Response Rate: {metrics['response_rate']*100:.1f}%")
        print(f"  - Avg Response Time: {metrics['avg_response_time']:.1f}s")
        print(f"  - Efficiency Score: {metrics['efficiency_score']:.1f}/100")
        
        print(f"\nPersonnel Detected:")
        print(f"  - Waiters: {len([p for p in self.person_tracker.persons.values() if p.person_type == 'waiter'])}")
        print(f"  - Customers: {len([p for p in self.person_tracker.persons.values() if p.person_type == 'customer'])}")
        
        print(f"\nTables Monitored: {len(self.table_mapper.tables)}")
        
        print(f"\nResults saved to: {self.output_dir}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Restaurant Service Video Analysis System")
    parser.add_argument("video", nargs='?', default="data/video_salon_poco_gente.MP4",
                       help="Path to input video file")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", default="data/outputs", 
                       help="Output directory for results")
    parser.add_argument("--skip-frames", type=int, default=2,
                       help="Process every Nth frame (default: 2)")
    parser.add_argument("--resize", type=float, default=0.5,
                       help="Resize factor for video (default: 0.5)")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)
    
    # Run analysis
    analyzer = RestaurantAnalyzer(
        args.video, 
        args.config, 
        args.output,
        skip_frames=args.skip_frames,
        resize_factor=args.resize
    )
    analyzer.run()


if __name__ == "__main__":
    main()