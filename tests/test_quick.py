#!/usr/bin/env python3
"""Quick test of the restaurant analysis system with minimal processing."""

import sys
import os
import logging

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import RestaurantAnalyzer

# Configure logging to be less verbose
logging.basicConfig(level=logging.INFO, format='%(message)s')

def quick_test():
    print("\n" + "="*60)
    print("QUICK TEST - Restaurant Analysis System")
    print("Processing only first 100 frames for testing")
    print("="*60 + "\n")
    
    # Create analyzer with very aggressive settings for speed
    analyzer = RestaurantAnalyzer(
        video_path="../data/video_salon_poco_gente.MP4",
        output_dir="../data/test_output",
        skip_frames=100,  # Process only every 100th frame
        resize_factor=0.2  # Very small resolution
    )
    
    # Limit processing to first 100 frames only
    max_frames = 100
    frame_count = 0
    
    try:
        # Calibrate tables
        print("📍 Calibrating tables...")
        analyzer.calibrate_tables()
        
        # Process limited frames
        print(f"🎬 Processing first {max_frames} frames...")
        
        for frame_num, frame, timestamp in analyzer.video_processor.process_frames():
            # Track persons
            persons = analyzer.person_tracker.update_tracks(frame, frame_num)
            
            # Update movement
            for person_id, person in persons.items():
                analyzer.movement_analyzer.update_position(
                    person_id, person.center, timestamp
                )
            
            frame_count += 1
            if frame_count >= max_frames:
                print(f"✅ Processed {frame_count} frames successfully")
                break
            
            if frame_count % 10 == 0:
                print(f"   Frame {frame_count}: {len(persons)} persons detected")
        
        # Generate minimal results
        print("\n📊 Generating results...")
        analyzer.generate_results()
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"✅ Frames processed: {frame_count}")
        print(f"✅ Persons detected: {len(analyzer.person_tracker.persons)}")
        print(f"✅ Tables configured: {len(analyzer.table_mapper.tables)}")
        
        metrics = analyzer.analysis_results.get('metrics', {})
        if metrics:
            print(f"✅ Total requests: {metrics.get('total_requests', 0)}")
            print(f"✅ Efficiency score: {metrics.get('efficiency_score', 0):.1f}/100")
        
        print("\n🎉 TEST COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {analyzer.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        analyzer.video_processor.release()

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)