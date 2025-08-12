#!/usr/bin/env python3
"""Debug over-detection and ID fragmentation issues"""

import sys
import os
import warnings
import torch
import cv2
import numpy as np
from collections import defaultdict
import json
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.person_tracker_gpu import PersonTrackerGPU
from src.core.video_processor import VideoProcessor

print("🔍 DEBUGGING OVER-DETECTION AND ID FRAGMENTATION")
print("="*60)

# Create video processor
video_processor = VideoProcessor(
    video_path="../data/video_salon_poco_gente.MP4",
    skip_frames=15,
    resize_factor=0.5
)

# Create GPU tracker with more conservative settings
print("1. Creating GPU PersonTracker with conservative settings...")
tracker = PersonTrackerGPU(
    model_size='yolov8m.pt',
    conf_threshold=0.7,  # Higher confidence threshold
    movement_threshold=5.0,  # Higher movement threshold  
    batch_size=1,
    use_tensorrt=False,
    use_half_precision=True,
    max_age=50  # Keep tracks longer before deletion
)

print(f"   ✅ GPU tracker created with conf_threshold={tracker.conf_threshold}")

# Track persons over 2 minutes (120 frames at 2fps = 60 processed frames)
frames_to_process = 60  # 2 minutes worth
frame_interval = 30  # Process every 30th frame (2fps at 30fps video)

person_tracks = {}
person_centers = defaultdict(list)
confidence_history = defaultdict(list)
first_detection = {}
last_detection = {}

print(f"\n2. Processing {frames_to_process} frames over 2 minutes...")
print("   Tracking person appearance/disappearance patterns...")

for i in range(frames_to_process):
    frame_num = i * frame_interval
    frame = video_processor.get_frame(frame_num)
    
    if frame is None:
        continue
    
    # Update tracking
    persons = tracker.update_tracks(frame, frame_num)
    
    if i % 10 == 0:
        print(f"     Frame {frame_num:4d}: {len(persons)} persons detected")
    
    # Record tracking data
    for person_id, person in persons.items():
        if person_id not in first_detection:
            first_detection[person_id] = frame_num
        last_detection[person_id] = frame_num
        
        person_centers[person_id].append(person.center)
        # Handle None confidence values
        conf_val = person.confidence if person.confidence is not None else 0.5
        confidence_history[person_id].append(conf_val)
        
        person_tracks[person_id] = {
            'type': person.person_type,
            'bbox': person.bbox,
            'confidence': person.confidence,
            'center': person.center
        }

print(f"\n3. Analysis Results:")
print(f"   Total unique IDs detected: {len(person_tracks)}")

# Analyze tracking patterns
print(f"\n4. Detailed tracking analysis:")

# Group by lifetime
short_lived = []  # < 10 frames
medium_lived = []  # 10-30 frames  
long_lived = []   # > 30 frames

for person_id in person_tracks:
    lifetime = last_detection[person_id] - first_detection[person_id]
    frames_seen = len(person_centers[person_id])
    # Handle empty confidence history
    if confidence_history[person_id]:
        avg_confidence = np.mean(confidence_history[person_id])
    else:
        avg_confidence = 0.5
    
    track_info = {
        'id': person_id,
        'lifetime': lifetime,
        'frames_seen': frames_seen,
        'avg_confidence': avg_confidence,
        'first_frame': first_detection[person_id],
        'last_frame': last_detection[person_id],
        'type': person_tracks[person_id]['type']
    }
    
    if frames_seen < 10:
        short_lived.append(track_info)
    elif frames_seen < 30:
        medium_lived.append(track_info)
    else:
        long_lived.append(track_info)

print(f"   Long-lived tracks (>30 frames): {len(long_lived)}")
for track in sorted(long_lived, key=lambda x: x['frames_seen'], reverse=True):
    print(f"     ID {str(track['id']):>3}: {track['frames_seen']:2d} frames, "
          f"conf={track['avg_confidence']:.2f}, type={track['type']}")

print(f"\n   Medium-lived tracks (10-30 frames): {len(medium_lived)}")
for track in sorted(medium_lived, key=lambda x: x['frames_seen'], reverse=True):
    print(f"     ID {str(track['id']):>3}: {track['frames_seen']:2d} frames, "
          f"conf={track['avg_confidence']:.2f}, type={track['type']}")

print(f"\n   Short-lived tracks (<10 frames): {len(short_lived)}")
for track in sorted(short_lived, key=lambda x: x['frames_seen'], reverse=True)[:10]:  # Show top 10
    print(f"     ID {str(track['id']):>3}: {track['frames_seen']:2d} frames, "
          f"conf={track['avg_confidence']:.2f}, type={track['type']}")

if len(short_lived) > 10:
    print(f"     ... and {len(short_lived) - 10} more short-lived tracks")

# Analyze spatial clustering (potential duplicates)
print(f"\n5. Spatial clustering analysis (potential ID fragmentation):")

def calculate_center_distance(centers1, centers2):
    """Calculate average distance between two sets of centers"""
    if not centers1 or not centers2:
        return float('inf')
    
    distances = []
    for c1 in centers1:
        for c2 in centers2:
            dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            distances.append(dist)
    
    return np.mean(distances)

# Check for potential duplicates (tracks with similar spatial patterns)
potential_duplicates = []
all_tracks = list(person_tracks.keys())

for i, track1 in enumerate(all_tracks):
    for track2 in all_tracks[i+1:]:
        avg_distance = calculate_center_distance(
            person_centers[track1], 
            person_centers[track2]
        )
        
        if avg_distance < 50:  # Within 50 pixels average
            potential_duplicates.append((track1, track2, avg_distance))

print(f"   Found {len(potential_duplicates)} potential duplicate pairs:")
for track1, track2, distance in sorted(potential_duplicates, key=lambda x: x[2])[:10]:
    print(f"     IDs {str(track1):>3} & {str(track2):>3}: avg_distance={distance:.1f}px")

# Recommendations
print(f"\n6. Recommendations:")
if len(short_lived) > len(long_lived) * 2:
    print("   ⚠️  Many short-lived tracks suggest tracking instability")
    print("   💡 Consider increasing max_age parameter in DeepSORT")

if len(potential_duplicates) > 5:
    print("   ⚠️  Many potential duplicate tracks suggest ID fragmentation") 
    print("   💡 Consider tuning DeepSORT appearance threshold")

if len(long_lived) > 20:
    print("   ⚠️  Too many long-lived tracks for restaurant setting")
    print("   💡 Consider higher confidence threshold or stricter NMS")

# Estimate realistic count
realistic_estimate = max(len(long_lived), len(medium_lived) + len(long_lived) // 2)
print(f"\n7. Realistic person count estimate: {realistic_estimate}")
print(f"   (Based on {len(long_lived)} long-lived + {len(medium_lived)//2} medium-lived tracks)")

print(f"\n🎯 Your manual count: ~17 people (12 clients + 3 waiters + 2 cooks)")
print(f"🤖 System detected: {len(person_tracks)} unique IDs")
print(f"📊 Realistic estimate: {realistic_estimate} actual people")

# Save detailed analysis
analysis_data = {
    'total_ids': len(person_tracks),
    'long_lived': len(long_lived),
    'medium_lived': len(medium_lived), 
    'short_lived': len(short_lived),
    'potential_duplicates': len(potential_duplicates),
    'realistic_estimate': realistic_estimate,
    'manual_count': 17,
    'tracking_details': {
        'long_lived_tracks': long_lived,
        'medium_lived_tracks': medium_lived,
        'short_lived_tracks': short_lived[:20],  # Top 20
        'potential_duplicate_pairs': potential_duplicates[:20]  # Top 20
    }
}

with open('overdetection_analysis.json', 'w') as f:
    json.dump(analysis_data, f, indent=2)

print(f"\n📝 Detailed analysis saved to overdetection_analysis.json")

# Cleanup
del tracker
if torch.cuda.is_available():
    torch.cuda.empty_cache()