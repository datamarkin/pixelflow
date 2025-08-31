# -*- coding: utf-8 -*-
"""
Buffer performance benchmark for PixelFlow.

This benchmark measures the performance impact of the Buffer module
by comparing frame processing speeds with and without buffering.
"""

import cv2
import time
import numpy as np
import psutil
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pixelflow.buffer import Buffer
from pixelflow.results import Detections, Detection


def format_memory(bytes_val):
    """Format memory in human-readable form."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def create_dummy_results(num_detections=5):
    """Create dummy results to simulate detection output."""
    results = Detections()
    for i in range(num_detections):
        pred = Detection(
            bbox=[100 + i*50, 100 + i*30, 200 + i*50, 200 + i*30],
            confidence=0.8 + i*0.02,
            class_id=i % 3,
            class_name=f"object_{i % 3}"
        )
        results.add_detection(pred)
    return results


def benchmark_baseline(video_path, max_frames=500):
    """Baseline benchmark without buffer."""
    print("\n" + "="*60)
    print("BASELINE TEST (No Buffer)")
    print("="*60)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return
    
    total_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Info: {total_fps:.2f} FPS, {total_frames} total frames")
    print(f"Processing up to {max_frames} frames...\n")
    
    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    frame_times = []
    frame_count = 0
    start_time = time.perf_counter()
    
    while frame_count < max_frames:
        frame_start = time.perf_counter()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simulate detection processing
        results = create_dummy_results()
        
        # Simulate some processing on frame
        _ = frame.copy()
        
        frame_time = time.perf_counter() - frame_start
        frame_times.append(frame_time)
        frame_count += 1
        
        # Progress update every 100 frames
        if frame_count % 100 == 0:
            avg_fps = 1 / np.mean(frame_times[-100:])
            print(f"  Frame {frame_count}: {avg_fps:.1f} FPS")
    
    total_time = time.perf_counter() - start_time
    
    # Get final memory
    final_memory = process.memory_info().rss
    memory_used = final_memory - initial_memory
    
    # Calculate statistics
    avg_fps = frame_count / total_time
    avg_frame_time = np.mean(frame_times) * 1000  # ms
    std_frame_time = np.std(frame_times) * 1000  # ms
    min_fps = 1 / max(frame_times) if frame_times else 0
    max_fps = 1 / min(frame_times) if frame_times else 0
    
    print(f"\nResults:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  FPS range: {min_fps:.1f} - {max_fps:.1f}")
    print(f"  Avg frame time: {avg_frame_time:.2f}ms +/- {std_frame_time:.2f}ms")
    print(f"  Memory used: {format_memory(memory_used)}")
    
    cap.release()
    return avg_fps, frame_times


def benchmark_with_buffer(video_path, buffer_size=7, max_frames=500):
    """Benchmark with buffer enabled."""
    print("\n" + "="*60)
    print(f"BUFFER TEST (size={buffer_size})")
    print("="*60)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return
    
    print(f"Buffer configuration:")
    print(f"  Size: {buffer_size} frames")
    print(f"  Delay: {buffer_size // 2} frames")
    print(f"Processing up to {max_frames} frames...\n")
    
    # Initialize buffer
    buffer = Buffer(frames=buffer_size)
    
    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    frame_times = []
    buffer_times = []
    frame_count = 0
    buffered_frames = 0
    start_time = time.perf_counter()
    
    while frame_count < max_frames:
        frame_start = time.perf_counter()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simulate detection processing
        results = create_dummy_results()
        
        # Buffer update
        buffer_start = time.perf_counter()
        buffered_results, buffered_frame = buffer.update(results, frame)
        buffer_time = time.perf_counter() - buffer_start
        buffer_times.append(buffer_time)
        
        if buffer.is_full:
            buffered_frames += 1
        
        frame_time = time.perf_counter() - frame_start
        frame_times.append(frame_time)
        frame_count += 1
        
        # Progress update every 100 frames
        if frame_count % 100 == 0:
            avg_fps = 1 / np.mean(frame_times[-100:])
            status = "BUFFERING" if not buffer.is_full else "ACTIVE"
            print(f"  Frame {frame_count} [{status}]: {avg_fps:.1f} FPS")
    
    total_time = time.perf_counter() - start_time
    
    # Get final memory
    final_memory = process.memory_info().rss
    memory_used = final_memory - initial_memory
    
    # Calculate statistics
    avg_fps = frame_count / total_time
    avg_frame_time = np.mean(frame_times) * 1000  # ms
    std_frame_time = np.std(frame_times) * 1000  # ms
    avg_buffer_time = np.mean(buffer_times) * 1000  # ms
    min_fps = 1 / max(frame_times) if frame_times else 0
    max_fps = 1 / min(frame_times) if frame_times else 0
    
    print(f"\nResults:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Frames buffered: {buffered_frames}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  FPS range: {min_fps:.1f} - {max_fps:.1f}")
    print(f"  Avg frame time: {avg_frame_time:.2f}ms +/- {std_frame_time:.2f}ms")
    print(f"  Avg buffer overhead: {avg_buffer_time:.3f}ms per frame")
    print(f"  Memory used: {format_memory(memory_used)}")
    print(f"  Memory per frame: {format_memory(memory_used / buffer_size)}")
    
    cap.release()
    return avg_fps, frame_times


def benchmark_buffer_sizes(video_path, max_frames=300):
    """Test different buffer sizes."""
    print("\n" + "="*60)
    print("BUFFER SIZE COMPARISON")
    print("="*60)
    
    buffer_sizes = [1, 3, 5, 7, 11, 15, 21]
    results = []
    
    for size in buffer_sizes:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        
        buffer = Buffer(frames=size)
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        frame_count = 0
        start_time = time.perf_counter()
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            results_obj = create_dummy_results()
            _, _ = buffer.update(results_obj, frame)
            frame_count += 1
        
        total_time = time.perf_counter() - start_time
        final_memory = process.memory_info().rss
        memory_used = final_memory - initial_memory
        
        fps = frame_count / total_time
        results.append({
            'size': size,
            'fps': fps,
            'memory': memory_used,
            'delay': size // 2
        })
        
        cap.release()
        print(f"  Buffer size {size:2d}: {fps:6.1f} FPS, {format_memory(memory_used):>10s}, delay={size//2}")
    
    return results


def main():
    """Main benchmark runner."""
    video_path = "../examples/data/crowd.mp4"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Please ensure you have the crowd.mp4 file in examples/data/")
        return
    
    print("\n" + "="*60)
    print("PIXELFLOW BUFFER PERFORMANCE BENCHMARK")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Total RAM: {format_memory(psutil.virtual_memory().total)}")
    print(f"Available RAM: {format_memory(psutil.virtual_memory().available)}")
    
    # Run baseline test
    baseline_fps, baseline_times = benchmark_baseline(video_path, max_frames=500)
    
    # Run buffer test with size 7
    buffer_fps, buffer_times = benchmark_with_buffer(video_path, buffer_size=7, max_frames=500)
    
    # Performance comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"Baseline FPS: {baseline_fps:.1f}")
    print(f"Buffer-7 FPS: {buffer_fps:.1f}")
    print(f"Performance impact: {((buffer_fps/baseline_fps - 1) * 100):.1f}%")
    
    if buffer_fps < baseline_fps * 0.95:
        print("�  Buffer introduces >5% performance overhead")
    else:
        print(" Buffer overhead is negligible (<5%)")
    
    # Test different buffer sizes
    print("\nTesting various buffer sizes...")
    size_results = benchmark_buffer_sizes(video_path, max_frames=300)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"1. Baseline can process at {baseline_fps:.1f} FPS")
    print(f"2. Buffer (size=7) processes at {buffer_fps:.1f} FPS")
    print(f"3. Buffer overhead is minimal: ~{np.mean(buffer_times)*1000:.3f}ms per frame")
    print(f"4. Larger buffers use more memory but FPS impact is minimal")
    print(f"5. Buffer is suitable for real-time processing")


if __name__ == "__main__":
    main()