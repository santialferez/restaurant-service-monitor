#!/usr/bin/env python3
"""Quick test of GPU components functionality"""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🧪 TESTING GPU COMPONENTS")
print("=" * 50)

# Test 1: Basic CUDA functionality
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Test 2: GPU tensor operations
print(f"\n🔥 Testing tensor operations...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
z = torch.matmul(x, y)
print(f"   Matrix multiplication on {device}: ✅")

# Test 3: PersonTrackerGPU initialization
print(f"\n🎯 Testing PersonTrackerGPU...")
try:
    from src.core.person_tracker_gpu import PersonTrackerGPU
    tracker = PersonTrackerGPU(
        batch_size=4,
        use_tensorrt=False,
        use_half_precision=True
    )
    print("   PersonTrackerGPU initialization: ✅")
    del tracker
except Exception as e:
    print(f"   PersonTrackerGPU initialization: ❌ {e}")

# Test 4: MovementAnalyzerGPU
print(f"\n📊 Testing MovementAnalyzerGPU...")
try:
    from src.analytics.movement_analyzer_gpu import MovementAnalyzerGPU
    movement = MovementAnalyzerGPU(frame_shape=(320, 240))
    movement.update_position(1, (100, 100), 1.0)
    movement.flush_buffers()
    print("   MovementAnalyzerGPU: ✅")
    del movement
except Exception as e:
    print(f"   MovementAnalyzerGPU: ❌ {e}")

# Test 5: GestureDetectorGPU
print(f"\n🙋 Testing GestureDetectorGPU...")
try:
    from src.core.gesture_detector_gpu import GestureDetectorGPU
    gesture = GestureDetectorGPU(batch_size=4)
    print("   GestureDetectorGPU initialization: ✅")
    del gesture
except Exception as e:
    print(f"   GestureDetectorGPU: ❌ {e}")

# Test 6: Memory cleanup
print(f"\n🧹 Testing GPU memory cleanup...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("   GPU cache cleared: ✅")

print(f"\n✨ All GPU component tests completed!")