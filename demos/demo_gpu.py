#!/usr/bin/env python3
"""GPU-accelerated demo of the restaurant analysis system."""

import sys
import os
import warnings
import torch
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main_gpu import RestaurantAnalyzerGPU

print("\n" + "="*80)
print("RESTAURANT SERVICE VIDEO ANALYSIS SYSTEM - GPU ACCELERATED DEMO")
print("="*80)
print("\nAnalyzing restaurant surveillance footage for service metrics...")
print("This demo showcases GPU acceleration with CUDA and TensorRT optimizations.\n")

# Check GPU availability
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🚀 GPU Acceleration: ENABLED")
    print(f"   Device: {gpu_name}")
    print(f"   Memory: {gpu_memory:.1f} GB")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    print("⚠️  GPU Acceleration: NOT AVAILABLE (falling back to CPU)")

print(f"   PyTorch Version: {torch.__version__}")

# Create GPU-optimized analyzer with aggressive settings for maximum performance
analyzer = RestaurantAnalyzerGPU(
    video_path="../data/video_salon_poco_gente.MP4",
    output_dir="../data/demo_output_gpu",
    skip_frames=15,  # Process every 15th frame (2 FPS from 30 FPS video)
    resize_factor=0.3,  # Smaller for faster processing
    batch_size=8,  # Moderate batch size for stability
    use_tensorrt=False,  # Disable TensorRT for this demo (requires additional setup)
    use_half_precision=True  # Enable FP16 for faster processing
)

# Process first 30 seconds of video
MAX_SECONDS = 30
max_frames = int(MAX_SECONDS * analyzer.video_processor.fps / analyzer.skip_frames)

print(f"\n🎬 Processing {MAX_SECONDS} seconds of video (~{max_frames} frames)")
print(f"📹 Original video: {analyzer.video_processor.fps:.0f} FPS")
print(f"⚡ Processing rate: {analyzer.video_processor.fps/analyzer.skip_frames:.1f} FPS")
print(f"📐 Resolution: {analyzer.video_processor.width}x{analyzer.video_processor.height}")
print(f"🔥 Batch size: {analyzer.batch_size}")
print(f"⏱️  TensorRT: {'Enabled' if analyzer.use_tensorrt else 'Disabled'}")
print(f"🎯 Half Precision: {'Enabled' if analyzer.use_half_precision else 'Disabled'}")

try:
    # Run GPU-accelerated analysis with timing
    import time
    
    print("\n" + "="*80)
    print("STARTING GPU-ACCELERATED ANALYSIS")
    print("="*80)
    
    start_time = time.time()
    
    # Calibrate tables
    print("\n📍 Configuring tables...")
    analyzer.calibrate_tables()
    print(f"✅ Configured {len(analyzer.table_mapper.tables)} tables")
    
    # Process frames with batch processing
    print(f"\n🚀 Processing frames with GPU acceleration...")
    
    # Custom processing loop for demo (limited frames)
    frame_count = 0
    batch_frames = []
    batch_frame_nums = []
    batch_timestamps = []
    
    for frame_num, frame, timestamp in analyzer.video_processor.process_frames():
        if frame_count >= max_frames:
            break
            
        batch_frames.append(frame)
        batch_frame_nums.append(frame_num)
        batch_timestamps.append(timestamp)
        
        # Process batch when full
        if len(batch_frames) >= analyzer.batch_size:
            analyzer._process_frame_batch(batch_frames, batch_frame_nums, batch_timestamps)
            frame_count += len(batch_frames)
            
            # Clear batch
            batch_frames.clear()
            batch_frame_nums.clear()
            batch_timestamps.clear()
            
            # Progress update
            if frame_count % (analyzer.batch_size * 4) == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Get GPU memory if available
                gpu_mem = ""
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**2
                    gpu_mem = f" | GPU: {memory_used:.0f}MB"
                
                print(f"  Processed {frame_count}/{max_frames} frames | "
                      f"Speed: {fps:.1f} FPS | "
                      f"Time: {elapsed_time:.1f}s{gpu_mem}")
    
    # Process remaining frames
    if batch_frames:
        analyzer._process_frame_batch(batch_frames, batch_frame_nums, batch_timestamps)
        frame_count += len(batch_frames)
    
    # Finalize processing
    final_timestamp = batch_timestamps[-1] if batch_timestamps else time.time()
    analyzer._finalize_processing(final_timestamp)
    
    processing_time = time.time() - start_time
    
    print(f"\n✅ Processed {frame_count} frames in {processing_time:.2f}s")
    print(f"   Average processing speed: {frame_count / processing_time:.2f} FPS")
    
    # Generate and save results
    print("\n📊 Generating analysis results...")
    analyzer.generate_results()
    analyzer.save_results()
    
    # Print detailed summary with GPU performance metrics
    print("\n" + "="*80)
    print("ANALYSIS RESULTS - GPU PERFORMANCE")
    print("="*80)
    
    results = analyzer.analysis_results
    metrics = results.get('metrics', {})
    gesture_stats = results.get('gesture_statistics', {})
    personnel_stats = results.get('personnel_statistics', {})
    performance = results.get('performance_metrics', {})
    
    print("\n🚀 GPU PERFORMANCE METRICS:")
    if torch.cuda.is_available():
        component_perf = performance.get('component_performance', {})
        
        # Person tracker performance
        tracker_perf = component_perf.get('person_tracker', {})
        if tracker_perf:
            detection_fps = tracker_perf.get('detection', {}).get('fps', 0)
            tracking_fps = tracker_perf.get('tracking', {}).get('fps', 0)
            print(f"  • Detection Speed: {detection_fps:.1f} FPS")
            print(f"  • Tracking Speed: {tracking_fps:.1f} FPS")
        
        # GPU memory usage
        gpu_memory = component_perf.get('gpu_memory', {})
        if gpu_memory:
            print(f"  • Peak GPU Memory: {gpu_memory.get('peak_memory_mb', 0):.1f} MB")
            print(f"  • Avg GPU Memory: {gpu_memory.get('avg_memory_mb', 0):.1f} MB")
        
        # Batch processing efficiency
        batch_times = performance.get('batch_processing_times', [])
        if batch_times:
            import numpy as np
            avg_batch_time = np.mean(batch_times)
            batch_throughput = analyzer.batch_size / avg_batch_time
            print(f"  • Batch Throughput: {batch_throughput:.1f} FPS")
            print(f"  • Avg Batch Time: {avg_batch_time:.3f}s")
    
    print(f"\n📈 SERVICE METRICS:")
    print(f"  • Total Requests: {metrics.get('total_requests', 0)}")
    print(f"  • Total Responses: {metrics.get('total_responses', 0)}")
    print(f"  • Response Rate: {metrics.get('response_rate', 0)*100:.1f}%")
    print(f"  • Avg Response Time: {metrics.get('avg_response_time', 0):.1f}s")
    print(f"  • Efficiency Score: {metrics.get('efficiency_score', 0):.1f}/100")
    
    print(f"\n👥 PERSONNEL DETECTED (GPU-ACCELERATED):")
    print(f"  • Total Persons: {personnel_stats.get('total_persons', 0)}")
    print(f"  • Waiters: {personnel_stats.get('waiters_detected', 0)}")
    print(f"  • Customers: {personnel_stats.get('customers_detected', 0)}")
    print(f"  • Unclassified: {personnel_stats.get('unclassified_persons', 0)}")
    
    print(f"\n🙋 GESTURE DETECTION (GPU-PROCESSED):")
    print(f"  • Total Gestures: {gesture_stats.get('total_gestures', 0)}")
    print(f"  • Responded: {gesture_stats.get('responded_gestures', 0)}")
    if gesture_stats.get('avg_response_time', 0) > 0:
        print(f"  • Avg Response: {gesture_stats.get('avg_response_time', 0):.1f}s")
    
    print(f"\n🪑 TABLE SERVICE:")
    table_attention = results.get('table_attention', {})
    tables_needing_attention = sum(1 for t in table_attention.values() if t.get('needs_attention', False))
    print(f"  • Tables Configured: {len(analyzer.table_mapper.tables)}")
    print(f"  • Tables Needing Attention: {tables_needing_attention}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  • GPU Results: {analyzer.output_dir}/analysis_results_gpu.json")
    print(f"  • Performance Report: {analyzer.output_dir}/performance_report_gpu.txt")
    print(f"  • Movement Data: {analyzer.output_dir}/movement_data_gpu.csv")
    print(f"  • GPU Heatmap: {analyzer.output_dir}/movement_heatmap_gpu.png")
    
    # Performance comparison hint
    total_time = performance.get('total_processing_time', processing_time)
    theoretical_cpu_time = frame_count * 0.4  # Estimate based on CPU demo
    speedup = theoretical_cpu_time / total_time if total_time > 0 else 1
    
    print(f"\n⚡ PERFORMANCE BOOST:")
    print(f"  • GPU Processing Time: {total_time:.2f}s")
    print(f"  • Estimated CPU Time: {theoretical_cpu_time:.2f}s")
    print(f"  • Approximate Speedup: {speedup:.1f}x faster")
    
    print("\n" + "="*80)
    print("✨ GPU-ACCELERATED DEMO COMPLETE!")
    print("="*80)
    
    print(f"\n🎯 Key Benefits Demonstrated:")
    print(f"  • Batch processing with {analyzer.batch_size} frames")
    print(f"  • GPU-accelerated person detection and tracking")
    print(f"  • Tensor-based movement analysis") 
    print(f"  • Memory-optimized processing pipeline")
    if analyzer.use_tensorrt:
        print(f"  • TensorRT model optimization")
    if analyzer.use_half_precision:
        print(f"  • FP16 half-precision acceleration")
    
    print(f"\n🚀 Next Steps:")
    print(f"  • Run full video analysis: python src/main_gpu.py data/video_salon_poco_gente.MP4")
    print(f"  • View interactive dashboard: streamlit run src/visualization/dashboard.py")
    print(f"  • Compare with CPU version: python demo.py")
    print("="*80 + "\n")
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    # Cleanup
    if 'analyzer' in locals():
        del analyzer
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()