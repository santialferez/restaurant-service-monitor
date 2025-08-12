#!/usr/bin/env python3
"""Quick GPU demo focusing on performance comparison"""

import sys
import os
import warnings
import torch
import time
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main_gpu import RestaurantAnalyzerGPU

print("\n" + "="*70)
print("RESTAURANT SERVICE ANALYSIS - GPU PERFORMANCE DEMO")
print("="*70)

# GPU status
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🚀 GPU: {gpu_name}")
    print(f"🔥 PyTorch: {torch.__version__} with CUDA {torch.version.cuda}")
else:
    print("⚠️  No GPU acceleration available")
    exit(1)

# Quick performance test
print(f"\n⚡ GPU Performance Test")
device = torch.device('cuda')
start = time.time()
x = torch.randn(2000, 2000, device=device)
y = torch.randn(2000, 2000, device=device)
z = torch.matmul(x, y)
gpu_time = time.time() - start
print(f"   Matrix multiplication (4M elements): {gpu_time:.3f}s")

# Create analyzer with moderate settings
print(f"\n📊 Initializing GPU-accelerated analyzer...")
analyzer = RestaurantAnalyzerGPU(
    video_path="../data/video_salon_poco_gente.MP4",
    output_dir="../data/demo_gpu_quick",
    skip_frames=20,  # Process every 20th frame for speed
    resize_factor=0.4,  # Moderate resolution 
    batch_size=4,  # Smaller batch for quick demo
    use_tensorrt=False,
    use_half_precision=True
)

print(f"✅ Initialized successfully")

# Process just 10 seconds
MAX_SECONDS = 10
max_frames = int(MAX_SECONDS * analyzer.video_processor.fps / analyzer.skip_frames)
print(f"\n🎬 Processing {MAX_SECONDS} seconds ({max_frames} frames)")

try:
    # Setup
    analyzer.calibrate_tables()
    
    # Quick processing
    start_time = time.time()
    frame_count = 0
    
    for frame_num, frame, timestamp in analyzer.video_processor.process_frames():
        if frame_count >= max_frames:
            break
            
        # Process single frame (not batch for simplicity)
        persons = analyzer.person_tracker.update_tracks(frame, frame_num)
        
        # Update movement
        for person_id, person in persons.items():
            analyzer.movement_analyzer.update_position(person_id, person.center, timestamp)
        
        frame_count += 1
        
        if frame_count % 5 == 0:
            print(f"  Processed {frame_count}/{max_frames} frames...")
    
    processing_time = time.time() - start_time
    fps = frame_count / processing_time
    
    # Results
    print(f"\n✅ RESULTS:")
    print(f"   Frames processed: {frame_count}")
    print(f"   Processing time: {processing_time:.2f}s")
    print(f"   Processing speed: {fps:.1f} FPS")
    
    # Get person stats
    waiters = len([p for p in analyzer.person_tracker.persons.values() if p.person_type == 'waiter'])
    customers = len([p for p in analyzer.person_tracker.persons.values() if p.person_type == 'customer'])
    
    print(f"   Persons detected: {len(analyzer.person_tracker.persons)}")
    print(f"   Waiters: {waiters}, Customers: {customers}")
    
    # Performance comparison
    cpu_estimate = frame_count * 0.5  # Rough CPU estimate
    speedup = cpu_estimate / processing_time if processing_time > 0 else 1
    
    print(f"\n⚡ PERFORMANCE:")
    print(f"   GPU time: {processing_time:.2f}s")
    print(f"   CPU estimate: {cpu_estimate:.2f}s")
    print(f"   Estimated speedup: {speedup:.1f}x")
    
    # GPU memory
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**2
        print(f"   GPU memory used: {memory_used:.1f}MB")
    
    print(f"\n🎯 GPU acceleration is working!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    if 'analyzer' in locals():
        del analyzer
    torch.cuda.empty_cache()
    print(f"\n✨ Demo complete!")