#!/usr/bin/env python3
"""Demo of the restaurant analysis system with comprehensive analysis."""

import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import RestaurantAnalyzer

print("\n" + "="*70)
print("RESTAURANT SERVICE VIDEO ANALYSIS SYSTEM - DEMO")
print("="*70)
print("\nAnalyzing restaurant surveillance footage for service metrics...")
print("This demo will process the first 30 seconds of video.\n")

# Create analyzer with balanced settings for demo
analyzer = RestaurantAnalyzer(
    video_path="../data/video_salon_poco_gente.MP4",
    output_dir="../data/demo_output",
    skip_frames=15,  # Process every 15th frame (2 FPS)
    resize_factor=0.3  # Reasonable quality
)

# Process first 30 seconds of video
MAX_SECONDS = 30
max_frames = int(MAX_SECONDS * analyzer.video_processor.fps / analyzer.skip_frames)

print(f"🎬 Processing {MAX_SECONDS} seconds of video (~{max_frames} frames)")
print(f"📹 Original video: {analyzer.video_processor.fps:.0f} FPS")
print(f"⚡ Processing rate: {analyzer.video_processor.fps/analyzer.skip_frames:.1f} FPS")
print(f"📐 Resolution: {analyzer.video_processor.width}x{analyzer.video_processor.height}\n")

try:
    # Run limited analysis
    print("Starting analysis...\n")
    
    # Calibrate tables
    analyzer.calibrate_tables()
    print(f"✅ Configured {len(analyzer.table_mapper.tables)} tables\n")
    
    # Process frames
    frame_count = 0
    for frame_num, frame, timestamp in analyzer.video_processor.process_frames():
        if frame_count >= max_frames:
            break
            
        # Core processing
        persons = analyzer.person_tracker.update_tracks(frame, frame_num)
        waiters = analyzer.person_tracker.get_waiters()
        customers = analyzer.person_tracker.get_customers()
        
        # Detect gestures
        gesture_events = analyzer.gesture_detector.detect_hand_raise(
            frame, persons, frame_num, timestamp
        )
        
        # Process events
        for event in gesture_events:
            if event.position:
                table_id = analyzer.table_mapper.assign_person_to_table(event.position)
                event.table_id = table_id
            analyzer.service_metrics.add_request_event(
                timestamp, event.person_id, event.table_id
            )
        
        # Check responses
        for gesture in analyzer.gesture_detector.gesture_events:
            if not gesture.responded:
                analyzer.gesture_detector.check_gesture_response(
                    gesture, waiters, timestamp
                )
        
        # Update movement
        for person_id, person in persons.items():
            analyzer.movement_analyzer.update_position(
                person_id, person.center, timestamp
            )
            
            if person.person_type == 'waiter':
                analyzer.table_mapper.update_table_visits(
                    person.center, timestamp
                )
        
        # Progress update
        frame_count += 1
        if frame_count % 20 == 0:
            print(f"  Processed {frame_count}/{max_frames} frames | "
                  f"Persons: {len(persons)} | "
                  f"Waiters: {len(waiters)} | "
                  f"Customers: {len(customers)}")
    
    print(f"\n✅ Processed {frame_count} frames\n")
    
    # Finalize any active gestures
    analyzer.gesture_detector.finalize_active_gestures(timestamp)
    
    # Process any finalized gestures as service requests
    for gesture in analyzer.gesture_detector.gesture_events:
        if gesture.position:
            table_id = analyzer.table_mapper.assign_person_to_table(gesture.position)
            gesture.table_id = table_id
        
        # Add to service metrics if not already processed
        analyzer.service_metrics.add_request_event(
            gesture.timestamp, gesture.person_id, gesture.table_id
        )
        
        # Check for responses from waiters
        waiters = analyzer.person_tracker.get_waiters()
        analyzer.gesture_detector.check_gesture_response(
            gesture, waiters, timestamp
        )
    
    # Generate and save results
    print("📊 Generating analysis results...")
    analyzer.generate_results()
    analyzer.save_results()
    
    # Print detailed summary
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    
    metrics = analyzer.analysis_results.get('metrics', {})
    gesture_stats = analyzer.analysis_results.get('gesture_statistics', {})
    
    print("\n📈 SERVICE METRICS:")
    print(f"  • Total Requests: {metrics.get('total_requests', 0)}")
    print(f"  • Total Responses: {metrics.get('total_responses', 0)}")
    print(f"  • Response Rate: {metrics.get('response_rate', 0)*100:.1f}%")
    print(f"  • Avg Response Time: {metrics.get('avg_response_time', 0):.1f}s")
    print(f"  • Efficiency Score: {metrics.get('efficiency_score', 0):.1f}/100")
    
    print("\n👥 PERSONNEL DETECTED:")
    all_persons = analyzer.person_tracker.persons
    waiters = [p for p in all_persons.values() if p.person_type == 'waiter']
    customers = [p for p in all_persons.values() if p.person_type == 'customer']
    unknown = [p for p in all_persons.values() if p.person_type == 'unknown']
    
    print(f"  • Total Persons: {len(all_persons)}")
    print(f"  • Waiters: {len(waiters)}")
    print(f"  • Customers: {len(customers)}")
    print(f"  • Unclassified: {len(unknown)}")
    
    print("\n🙋 GESTURE DETECTION:")
    print(f"  • Total Gestures: {gesture_stats.get('total_gestures', 0)}")
    print(f"  • Responded: {gesture_stats.get('responded_gestures', 0)}")
    if gesture_stats.get('avg_response_time', 0) > 0:
        print(f"  • Avg Response: {gesture_stats.get('avg_response_time', 0):.1f}s")
    
    print("\n🪑 TABLE SERVICE:")
    table_metrics = analyzer.table_mapper.get_table_attention_metrics(timestamp)
    tables_needing_attention = sum(1 for t in table_metrics.values() if t['needs_attention'])
    print(f"  • Tables Configured: {len(analyzer.table_mapper.tables)}")
    print(f"  • Tables Needing Attention: {tables_needing_attention}")
    
    print("\n📁 OUTPUT FILES:")
    print(f"  • Results: {analyzer.output_dir}/analysis_results.json")
    print(f"  • Report: {analyzer.output_dir}/analysis_report.txt")
    print(f"  • Metrics: {analyzer.output_dir}/service_metrics.xlsx")
    print(f"  • Movement: {analyzer.output_dir}/movement_data.csv")
    
    print("\n" + "="*70)
    print("✨ DEMO COMPLETE!")
    print("="*70)
    print("\nTo view the interactive dashboard, run:")
    print("  streamlit run src/visualization/dashboard.py")
    print("\nFor full video analysis, run:")
    print("  python src/main.py data/video_salon_poco_gente.MP4")
    print("="*70 + "\n")
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    analyzer.video_processor.release()