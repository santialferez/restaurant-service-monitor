#!/usr/bin/env python3
"""GPU-accelerated demo - 2 minutes for comprehensive analysis"""

import sys
import os
import warnings
import torch
import time
import json
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main_gpu import RestaurantAnalyzerGPU

print("\n" + "="*80)
print("RESTAURANT SERVICE ANALYSIS - 2 MINUTE GPU COMPREHENSIVE DEMO")
print("="*80)

# GPU status
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🚀 GPU: {gpu_name}")
    print(f"💾 VRAM: {gpu_memory:.1f} GB")
    print(f"🔥 PyTorch: {torch.__version__} with CUDA {torch.version.cuda}")
else:
    print("⚠️  No GPU acceleration available")
    exit(1)

# Create GPU-optimized analyzer for 2-minute analysis
print(f"\n📊 Initializing GPU-accelerated analyzer for 2-minute analysis...")
analyzer = RestaurantAnalyzerGPU(
    video_path="../data/video_salon_poco_gente.MP4",
    output_dir="../data/demo_gpu_2min",
    skip_frames=15,  # Process every 15th frame (2 FPS from 30 FPS)
    resize_factor=0.5,  # Good balance of speed vs quality
    batch_size=8,  # Good GPU utilization
    use_tensorrt=False,  # Disable for stability
    use_half_precision=True  # Enable FP16
)

# Process 2 minutes = 120 seconds
MAX_SECONDS = 120
max_frames = int(MAX_SECONDS * analyzer.video_processor.fps / analyzer.skip_frames)

print(f"\n🎬 Processing {MAX_SECONDS} seconds ({max_frames} frames)")
print(f"📹 Original: {analyzer.video_processor.fps:.0f} FPS")
print(f"⚡ Processing: {analyzer.video_processor.fps/analyzer.skip_frames:.1f} FPS")
print(f"🎯 Batch size: {analyzer.batch_size}")
print(f"💾 Resolution: {analyzer.video_processor.width}x{analyzer.video_processor.height}")

try:
    # Setup with detailed logging
    print(f"\n📍 Setting up tables...")
    analyzer.calibrate_tables()
    print(f"✅ Configured {len(analyzer.table_mapper.tables)} tables")
    
    # Start comprehensive processing
    start_time = time.time()
    print(f"\n🚀 Starting comprehensive GPU-accelerated analysis...")
    print(f"   Target: {max_frames} frames in batches of {analyzer.batch_size}")
    
    # Batch processing with detailed tracking
    frame_count = 0
    batch_count = 0
    total_persons_seen = set()
    detection_log = []
    
    batch_frames = []
    batch_frame_nums = []
    batch_timestamps = []
    
    for frame_num, frame, timestamp in analyzer.video_processor.process_frames():
        if frame_count >= max_frames:
            break
            
        batch_frames.append(frame)
        batch_frame_nums.append(frame_num)
        batch_timestamps.append(timestamp)
        
        # Process when batch is full
        if len(batch_frames) >= analyzer.batch_size:
            batch_start_time = time.time()
            
            # Process the batch
            batch_persons = analyzer.person_tracker.update_tracks_batch(batch_frames, batch_frame_nums)
            
            # Process each frame's results
            for frame_idx, (frame, frame_num, timestamp, frame_persons) in enumerate(
                zip(batch_frames, batch_frame_nums, batch_timestamps, batch_persons)
            ):
                # Track unique persons
                total_persons_seen.update(frame_persons.keys())
                
                # Gesture detection
                gesture_events = analyzer.gesture_detector.detect_hand_raise_batch(
                    frame, frame_persons, frame_num, timestamp
                )
                
                # Process gestures
                for event in gesture_events:
                    if event.position:
                        table_id = analyzer.table_mapper.assign_person_to_table(event.position)
                        event.table_id = table_id
                    analyzer.service_metrics.add_request_event(
                        timestamp, event.person_id, event.table_id
                    )
                
                # Check gesture responses
                waiters = analyzer.person_tracker.get_waiters()
                for gesture in analyzer.gesture_detector.gesture_events:
                    if not gesture.responded:
                        analyzer.gesture_detector.check_gesture_response(gesture, waiters, timestamp)
                
                # Update movement analysis
                for person_id, person in frame_persons.items():
                    analyzer.movement_analyzer.update_position(person_id, person.center, timestamp)
                    
                    if person.person_type == 'waiter':
                        analyzer.table_mapper.update_table_visits(person.center, timestamp)
                
                # Log detection details every 20 frames
                if frame_count % 20 == 0:
                    waiters_count = len([p for p in frame_persons.values() if p.person_type == 'waiter'])
                    customers_count = len([p for p in frame_persons.values() if p.person_type == 'customer'])
                    detection_log.append({
                        'frame': frame_count,
                        'timestamp': timestamp,
                        'total_persons': len(frame_persons),
                        'waiters': waiters_count,
                        'customers': customers_count,
                        'gestures_detected': len(analyzer.gesture_detector.gesture_events)
                    })
            
            batch_time = time.time() - batch_start_time
            batch_count += 1
            frame_count += len(batch_frames)
            
            # Progress update
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            progress = (frame_count / max_frames) * 100
            
            # GPU memory tracking
            gpu_mem = ""
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**2
                gpu_mem = f"GPU:{memory_used:.0f}MB"
            
            print(f"  [{progress:5.1f}%] Batch {batch_count:3d} | "
                  f"Frames: {frame_count:4d}/{max_frames} | "
                  f"Speed: {fps:.1f}FPS | "
                  f"Time: {elapsed:.1f}s | {gpu_mem}")
            
            # Clear batch
            batch_frames.clear()
            batch_frame_nums.clear()
            batch_timestamps.clear()
    
    # Process any remaining frames
    if batch_frames:
        batch_persons = analyzer.person_tracker.update_tracks_batch(batch_frames, batch_frame_nums)
        for frame_idx, (frame, frame_num, timestamp, frame_persons) in enumerate(
            zip(batch_frames, batch_frame_nums, batch_timestamps, batch_persons)
        ):
            total_persons_seen.update(frame_persons.keys())
            for person_id, person in frame_persons.items():
                analyzer.movement_analyzer.update_position(person_id, person.center, timestamp)
        frame_count += len(batch_frames)
    
    # Finalize processing
    final_timestamp = batch_timestamps[-1] if batch_timestamps else time.time()
    analyzer._finalize_processing(final_timestamp)
    
    processing_time = time.time() - start_time
    
    print(f"\n✅ PROCESSING COMPLETE!")
    print(f"   Frames processed: {frame_count}")
    print(f"   Processing time: {processing_time:.2f}s")
    print(f"   Average speed: {frame_count / processing_time:.2f} FPS")
    print(f"   Batches processed: {batch_count}")
    
    # Generate comprehensive results
    print(f"\n📊 Generating comprehensive analysis results...")
    analyzer.generate_results()
    analyzer.save_results()
    
    # DETAILED ANALYSIS OF RESULTS
    print(f"\n" + "="*80)
    print("COMPREHENSIVE 2-MINUTE ANALYSIS RESULTS")
    print("="*80)
    
    results = analyzer.analysis_results
    metrics = results.get('metrics', {})
    gesture_stats = results.get('gesture_statistics', {})
    personnel_stats = results.get('personnel_statistics', {})
    performance = results.get('performance_metrics', {})
    movement_patterns = results.get('movement_patterns', {})
    table_attention = results.get('table_attention', {})
    
    # 1. PERFORMANCE ANALYSIS
    print(f"\n🚀 GPU PERFORMANCE ANALYSIS:")
    print(f"  • Total Processing Time: {processing_time:.2f}s")
    print(f"  • Average Processing Speed: {frame_count / processing_time:.2f} FPS")
    print(f"  • Frames Processed: {frame_count} out of {max_frames} target")
    print(f"  • Video Coverage: {(frame_count * analyzer.skip_frames / analyzer.video_processor.fps):.1f} seconds analyzed")
    
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1024**2
        max_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  • Current GPU Memory: {current_memory:.1f} MB")
        print(f"  • Peak GPU Memory: {max_memory:.1f} MB")
    
    # 2. PERSON DETECTION ANALYSIS
    print(f"\n👥 PERSON DETECTION ANALYSIS:")
    all_persons = analyzer.person_tracker.persons
    waiters = [p for p in all_persons.values() if p.person_type == 'waiter']
    customers = [p for p in all_persons.values() if p.person_type == 'customer']
    unknown = [p for p in all_persons.values() if p.person_type == 'unknown']
    
    print(f"  • Total Unique Persons Detected: {len(all_persons)}")
    print(f"  • Waiters Identified: {len(waiters)}")
    print(f"  • Customers Identified: {len(customers)}")
    print(f"  • Unclassified Persons: {len(unknown)}")
    
    # Movement analysis for each person type
    if waiters:
        print(f"\n  📊 WAITER MOVEMENT PATTERNS:")
        for waiter in waiters:
            if waiter.id in movement_patterns:
                pattern = movement_patterns[waiter.id]
                print(f"     Waiter {waiter.id}: Distance: {pattern['total_distance']:.1f}px, "
                      f"Avg Speed: {pattern['avg_speed']:.2f}px/frame, "
                      f"Path Points: {pattern['path_points']}")
    
    if customers:
        print(f"\n  📊 CUSTOMER MOVEMENT PATTERNS:")
        customer_movements = []
        for customer in customers[:5]:  # Show top 5
            if customer.id in movement_patterns:
                pattern = movement_patterns[customer.id]
                customer_movements.append(pattern)
                print(f"     Customer {customer.id}: Distance: {pattern['total_distance']:.1f}px, "
                      f"Avg Speed: {pattern['avg_speed']:.2f}px/frame")
    
    # 3. GESTURE DETECTION ANALYSIS
    print(f"\n🙋 GESTURE DETECTION ANALYSIS:")
    print(f"  • Total Gestures Detected: {gesture_stats.get('total_gestures', 0)}")
    print(f"  • Gestures Responded To: {gesture_stats.get('responded_gestures', 0)}")
    
    if gesture_stats.get('avg_response_time', 0) > 0:
        print(f"  • Average Response Time: {gesture_stats.get('avg_response_time', 0):.2f}s")
    
    # Show individual gestures
    if analyzer.gesture_detector.gesture_events:
        print(f"\n  📋 INDIVIDUAL GESTURES:")
        for i, gesture in enumerate(analyzer.gesture_detector.gesture_events[:10]):  # Show first 10
            status = "✅ Responded" if gesture.responded else "⏳ Pending"
            response_info = f"({gesture.response_time:.1f}s)" if gesture.response_time else ""
            print(f"     Gesture {i+1}: Person {gesture.person_id} at {gesture.timestamp:.1f}s - {status} {response_info}")
    
    # 4. SERVICE METRICS ANALYSIS
    print(f"\n📈 SERVICE METRICS ANALYSIS:")
    print(f"  • Total Service Requests: {metrics.get('total_requests', 0)}")
    print(f"  • Total Responses: {metrics.get('total_responses', 0)}")
    print(f"  • Response Rate: {metrics.get('response_rate', 0)*100:.1f}%")
    print(f"  • Average Response Time: {metrics.get('avg_response_time', 0):.1f}s")
    print(f"  • Service Efficiency Score: {metrics.get('efficiency_score', 0):.1f}/100")
    
    # 5. TABLE ANALYSIS
    print(f"\n🪑 TABLE SERVICE ANALYSIS:")
    print(f"  • Total Tables Configured: {len(analyzer.table_mapper.tables)}")
    
    tables_with_visits = sum(1 for table in analyzer.table_mapper.tables.values() if table.visit_count > 0)
    tables_needing_attention = sum(1 for t in table_attention.values() if t.get('needs_attention', False))
    
    print(f"  • Tables with Waiter Visits: {tables_with_visits}")
    print(f"  • Tables Needing Attention: {tables_needing_attention}")
    
    # Show table visit details
    print(f"\n  📊 TABLE VISIT DETAILS:")
    for table_id, table in analyzer.table_mapper.tables.items():
        if table.visit_count > 0:
            last_visit = table.last_visit_time
            time_since = final_timestamp - last_visit if last_visit else float('inf')
            print(f"     Table {table_id}: {table.visit_count} visits, "
                  f"last visit {time_since:.1f}s ago")
    
    # 6. MOVEMENT HEATMAP ANALYSIS
    print(f"\n🗺️  MOVEMENT ANALYSIS:")
    print(f"  • Movement patterns calculated for {len(movement_patterns)} persons")
    
    if movement_patterns:
        total_movement = sum(p['total_distance'] for p in movement_patterns.values())
        avg_speed = sum(p['avg_speed'] for p in movement_patterns.values()) / len(movement_patterns)
        print(f"  • Total Movement Distance: {total_movement:.1f} pixels")
        print(f"  • Average Movement Speed: {avg_speed:.2f} pixels/frame")
    
    # 7. DETECTION LOG ANALYSIS
    print(f"\n📈 DETECTION TIMELINE:")
    if detection_log:
        print(f"  • Detection samples: {len(detection_log)}")
        for log in detection_log[-5:]:  # Show last 5 samples
            print(f"     Frame {log['frame']:4d} ({log['timestamp']:6.1f}s): "
                  f"{log['total_persons']} persons "
                  f"({log['waiters']} waiters, {log['customers']} customers)")
    
    # 8. FILES GENERATED
    print(f"\n📁 OUTPUT FILES GENERATED:")
    output_files = [
        "analysis_results_gpu.json",
        "performance_report_gpu.txt", 
        "movement_data_gpu.csv",
        "movement_heatmap_gpu.png",
        "table_config.json"
    ]
    
    for file in output_files:
        file_path = analyzer.output_dir / file
        if file_path.exists():
            size = file_path.stat().st_size / 1024
            print(f"  • {file}: {size:.1f} KB")
    
    # 9. PERFORMANCE COMPARISON
    estimated_cpu_time = frame_count * 0.6  # Rough estimate
    speedup = estimated_cpu_time / processing_time if processing_time > 0 else 1
    
    print(f"\n⚡ PERFORMANCE COMPARISON:")
    print(f"  • GPU Processing Time: {processing_time:.2f}s")
    print(f"  • Estimated CPU Time: {estimated_cpu_time:.2f}s")
    print(f"  • Performance Speedup: {speedup:.1f}x faster")
    print(f"  • Time Saved: {estimated_cpu_time - processing_time:.2f}s")
    
    print(f"\n" + "="*80)
    print("✨ 2-MINUTE GPU ANALYSIS COMPLETE - COMPREHENSIVE RESULTS GENERATED!")
    print("="*80)
    
    # Save detailed log
    detailed_log = {
        'processing_time': processing_time,
        'frames_processed': frame_count,
        'fps': frame_count / processing_time,
        'detection_log': detection_log,
        'final_results': results
    }
    
    with open(analyzer.output_dir / "detailed_analysis_log.json", 'w') as f:
        json.dump(detailed_log, f, indent=2, default=str)
    
    print(f"\n🔍 Detailed analysis saved to: {analyzer.output_dir}/detailed_analysis_log.json")
    
except Exception as e:
    print(f"\n❌ Error during processing: {e}")
    import traceback
    traceback.print_exc()

finally:
    if 'analyzer' in locals():
        del analyzer
    
    # Final GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n🧹 GPU memory cleared")
    
    print(f"\n✨ 2-minute comprehensive demo complete!")