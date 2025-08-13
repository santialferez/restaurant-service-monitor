#!/usr/bin/env python3
"""Complete restaurant analysis for first 2 minutes with improved tracking"""

import sys
import os
import warnings
import torch
import cv2
import numpy as np
import time
import json
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.person_tracker_gpu import PersonTrackerGPU
from src.core.gesture_detector_gpu import GestureDetectorGPU
from src.core.video_processor import VideoProcessor

def complete_analysis_2min():
    """Run complete analysis for first 2 minutes with all improvements"""
    
    print("🚀 COMPLETE RESTAURANT ANALYSIS - IMPROVED TRACKING")
    print("=" * 60)
    
    # Initialize components with improved parameters
    video_processor = VideoProcessor(
        "../data/video_salon_poco_gente.MP4",
        skip_frames=10,  # Process every 10th frame for 3 FPS
        resize_factor=0.7  # Good balance of quality/speed
    )
    
    # Initialize IMPROVED tracker with fixed parameters
    person_tracker = PersonTrackerGPU(
        model_size='yolov8m.pt',
        conf_threshold=0.4,  # IMPROVED - was 0.6
        max_age=30,          # IMPROVED - was 60
        movement_threshold=5.0,
        nms_threshold=0.4    # IMPROVED - was 0.25
    )
    
    # Initialize gesture detector with pose estimation
    gesture_detector = GestureDetectorGPU(device='cuda')
    
    # Analysis parameters
    video_duration = 120  # 2 minutes
    fps = 30
    total_frames_to_process = int((video_duration * fps) / 10)  # Every 10th frame
    
    print(f"📹 Processing {total_frames_to_process} frames over {video_duration} seconds")
    print(f"🎯 Skip frames: 10, Resize: 0.7, Target: 3 FPS processing")
    
    # Storage for results
    all_persons = {}
    gesture_events = []
    movement_data = []
    frame_stats = []
    
    # Process frames
    start_time = time.time()
    
    try:
        for frame_idx in range(total_frames_to_process):
            # Get frame
            frame = video_processor.get_frame(frame_idx * 10)
            if frame is None:
                continue
            
            timestamp = frame_idx * 10 / fps  # Real timestamp
            
            # Track people
            persons = person_tracker.update_tracks(frame, frame_idx + 1)
            
            # Detect gestures for each person
            for person_id, person in persons.items():
                # Store person data
                if person_id not in all_persons:
                    all_persons[person_id] = {
                        'first_seen': timestamp,
                        'last_seen': timestamp,
                        'positions': [],
                        'type': person.person_type,
                        'gestures': []
                    }
                
                all_persons[person_id]['last_seen'] = timestamp
                all_persons[person_id]['positions'].append({
                    'timestamp': timestamp,
                    'bbox': person.bbox,
                    'center': [(person.bbox[0] + person.bbox[2]) / 2,
                              (person.bbox[1] + person.bbox[3]) / 2]
                })
                
            # Gesture detection using the batch method
            frame_gestures = gesture_detector.detect_hand_raise_batch(
                frame, {person_id: person}, frame_idx + 1, timestamp
            )
            
            # Store detected gestures
            for gesture_event in frame_gestures:
                gesture_dict = {
                    'person_id': gesture_event.person_id,
                    'timestamp': gesture_event.timestamp,
                    'type': gesture_event.gesture_type,
                    'confidence': gesture_event.confidence,
                    'person_type': person.person_type,
                    'position': [gesture_event.position[0], gesture_event.position[1]]
                }
                gesture_events.append(gesture_dict)
                all_persons[person_id]['gestures'].append(gesture_dict)
            
            # Frame statistics
            frame_stat = {
                'frame': frame_idx,
                'timestamp': timestamp,
                'people_count': len(persons),
                'waiters': sum(1 for p in persons.values() if p.person_type == 'waiter'),
                'customers': sum(1 for p in persons.values() if p.person_type == 'customer'),
                'unknown': sum(1 for p in persons.values() if p.person_type == 'unknown')
            }
            frame_stats.append(frame_stat)
            
            # Progress update
            if (frame_idx + 1) % 30 == 0:
                elapsed = time.time() - start_time
                remaining = (elapsed / (frame_idx + 1)) * (total_frames_to_process - frame_idx - 1)
                print(f"  Frame {frame_idx+1:3d}/{total_frames_to_process}: "
                     f"{len(persons)} people, {len(gesture_events)} gestures, "
                     f"ETA: {remaining:.1f}s")
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False
    
    finally:
        video_processor.release()
    
    # Calculate metrics
    processing_time = time.time() - start_time
    avg_people = np.mean([s['people_count'] for s in frame_stats]) if frame_stats else 0
    max_people = max([s['people_count'] for s in frame_stats]) if frame_stats else 0
    
    # Service analysis
    service_events = []
    for gesture in gesture_events:
        if gesture['person_type'] == 'customer':
            # Look for waiter responses within 30 seconds
            gesture_time = gesture['timestamp']
            waiter_responses = []
            
            for person_id, person_data in all_persons.items():
                if person_data['type'] == 'waiter':
                    # Check if waiter moved towards customer within 30s
                    for pos in person_data['positions']:
                        if gesture_time <= pos['timestamp'] <= gesture_time + 30:
                            # Calculate distance to customer
                            distance = np.sqrt(
                                (pos['center'][0] - gesture['position'][0])**2 + 
                                (pos['center'][1] - gesture['position'][1])**2
                            )
                            waiter_responses.append({
                                'waiter_id': person_id,
                                'response_time': pos['timestamp'] - gesture_time,
                                'distance': distance
                            })
            
            if waiter_responses:
                best_response = min(waiter_responses, key=lambda x: x['distance'])
                service_events.append({
                    'customer_gesture': gesture,
                    'waiter_response': best_response,
                    'service_time': best_response['response_time']
                })
    
    # Save results
    results = {
        'analysis_info': {
            'video_path': '../data/video_salon_poco_gente.MP4',
            'duration_analyzed': video_duration,
            'frames_processed': len(frame_stats),
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        },
        'tracking_performance': {
            'total_people_detected': len(all_persons),
            'avg_people_per_frame': avg_people,
            'max_people_in_frame': max_people,
            'improvement_note': 'Using improved tracker parameters'
        },
        'gesture_analysis': {
            'total_gestures': len(gesture_events),
            'hand_raises': len([g for g in gesture_events if g['type'] == 'hand_raised']),
            'customer_gestures': len([g for g in gesture_events if g['person_type'] == 'customer']),
            'waiter_gestures': len([g for g in gesture_events if g['person_type'] == 'waiter'])
        },
        'service_analysis': {
            'service_events': len(service_events),
            'avg_response_time': np.mean([s['service_time'] for s in service_events]) if service_events else 0,
            'service_efficiency': f"{len(service_events)}/{len([g for g in gesture_events if g['person_type'] == 'customer'])}"
        },
        'detailed_data': {
            'persons': all_persons,
            'gesture_events': gesture_events,
            'service_events': service_events,
            'frame_statistics': frame_stats
        }
    }
    
    # Save to file
    output_file = "../outputs/reports/complete_analysis_2min.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print(f"\n✅ COMPLETE ANALYSIS FINISHED!")
    print(f"⏱️  Processing time: {processing_time:.1f}s")
    print(f"👥 People tracking:")
    print(f"   Total unique people: {len(all_persons)}")
    print(f"   Average per frame: {avg_people:.1f}")
    print(f"   Maximum in frame: {max_people}")
    print(f"🙋 Gesture detection:")
    print(f"   Total gestures: {len(gesture_events)}")
    print(f"   Customer hand raises: {len([g for g in gesture_events if g['person_type'] == 'customer'])}")
    print(f"🔔 Service analysis:")
    print(f"   Service events: {len(service_events)}")
    if service_events:
        avg_response = np.mean([s['service_time'] for s in service_events])
        print(f"   Average response time: {avg_response:.1f}s")
    print(f"📊 Results saved: {output_file}")
    
    return True

def main():
    print("🎯 COMPLETE 2-MINUTE RESTAURANT ANALYSIS")
    print("Improved tracking + gesture detection + service analysis")
    
    success = complete_analysis_2min()
    
    if success:
        print(f"\n🎉 Analysis complete! Check outputs/reports/ for detailed results")
    else:
        print(f"\n❌ Analysis failed")

if __name__ == "__main__":
    main()