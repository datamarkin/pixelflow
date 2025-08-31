"""
OpenCV Performance Benchmark for PixelFlow

Tests pure OpenCV operations performance including:
- Video frame reading
- Image resizing
- Color space conversions
- Drawing operations (rectangles, text, polygons)
- PixelFlow annotation functions
"""

import cv2
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import pixelflow
from pixelflow.results import Detections, Detection


class OpenCVBenchmark:
    def __init__(self, video_path="./examples/videos/paris.mp4", output_dir="benchmarks/results"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def benchmark_video_reading(self, num_frames=500):
        """Benchmark pure video frame reading speed."""
        print(f"\n📹 Benchmarking video reading ({num_frames} frames)...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open {self.video_path}")
            return
        
        start_time = time.time()
        frames_read = 0
        
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frames_read += 1
            
        elapsed = time.time() - start_time
        fps = frames_read / elapsed
        
        cap.release()
        
        self.results['video_reading'] = {
            'frames': frames_read,
            'time_seconds': elapsed,
            'fps': fps
        }
        
        print(f"   ✓ Read {frames_read} frames in {elapsed:.2f}s")
        print(f"   ✓ FPS: {fps:.1f}")
        
        return fps
    
    def benchmark_image_operations(self, num_iterations=1000):
        """Benchmark common image processing operations."""
        print(f"\n🖼️ Benchmarking image operations ({num_iterations} iterations)...")
        
        # Create test image
        test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        operations = {
            'resize_720p': lambda img: cv2.resize(img, (1280, 720)),
            'resize_480p': lambda img: cv2.resize(img, (640, 480)),
            'color_bgr2rgb': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            'color_bgr2gray': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            'blur_gaussian': lambda img: cv2.GaussianBlur(img, (15, 15), 0),
            'blur_motion': lambda img: cv2.filter2D(img, -1, np.ones((15, 15)) / 225)
        }
        
        results = {}
        
        for op_name, operation in operations.items():
            start_time = time.time()
            
            for _ in range(num_iterations):
                _ = operation(test_image)
            
            elapsed = time.time() - start_time
            ops_per_sec = num_iterations / elapsed
            
            results[op_name] = {
                'iterations': num_iterations,
                'time_seconds': elapsed,
                'ops_per_second': ops_per_sec
            }
            
            print(f"   ✓ {op_name}: {ops_per_sec:.1f} ops/sec")
        
        self.results['image_operations'] = results
        return results
    
    def benchmark_drawing_operations(self, num_iterations=10000):
        """Benchmark OpenCV drawing operations."""
        print(f"\n✏️ Benchmarking drawing operations ({num_iterations} iterations)...")
        
        # Create test canvas
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        operations = {
            'rectangle': lambda img: cv2.rectangle(img.copy(), (100, 100), (500, 400), (0, 255, 0), 2),
            'filled_rectangle': lambda img: cv2.rectangle(img.copy(), (100, 100), (500, 400), (0, 255, 0), -1),
            'circle': lambda img: cv2.circle(img.copy(), (960, 540), 100, (255, 0, 0), 2),
            'text': lambda img: cv2.putText(img.copy(), "Benchmark Text", (100, 100), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2),
            'polylines': lambda img: cv2.polylines(img.copy(), 
                                                  [np.array([(100, 100), (500, 100), (500, 400), (100, 400)])], 
                                                  True, (0, 255, 255), 2),
            'fillPoly': lambda img: cv2.fillPoly(img.copy(), 
                                                [np.array([(100, 100), (500, 100), (500, 400), (100, 400)])], 
                                                (0, 255, 255))
        }
        
        results = {}
        
        for op_name, operation in operations.items():
            start_time = time.time()
            
            for _ in range(num_iterations):
                _ = operation(canvas)
            
            elapsed = time.time() - start_time
            ops_per_sec = num_iterations / elapsed
            
            results[op_name] = {
                'iterations': num_iterations,
                'time_seconds': elapsed,
                'ops_per_second': ops_per_sec
            }
            
            print(f"   ✓ {op_name}: {ops_per_sec:.1f} ops/sec")
        
        self.results['drawing_operations'] = results
        return results
    
    def benchmark_pixelflow_annotations(self, num_frames=500):
        """Benchmark PixelFlow annotation functions with simulated detections."""
        print(f"\n🎨 Benchmarking PixelFlow annotations ({num_frames} frames)...")
        
        # Create synthetic results with multiple detections
        def create_mock_results(num_objects=10):
            predictions = []
            for i in range(num_objects):
                x = np.random.randint(100, 1800)
                y = np.random.randint(100, 900)
                predictions.append(Detection(
                    bbox=[x, y, x + 100, y + 100],
                    confidence=0.95,
                    class_id=i % 5,
                    class_name=f"object_{i % 5}",
                    tracker_id=i
                ))
            return Detections(detections=predictions)
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open {self.video_path}")
            return
        
        # Get first frame as template
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            cap.release()
            return
        
        annotation_functions = {
            'box': lambda img, res: pixelflow.annotate.box(img, res),
            'label': lambda img, res: pixelflow.annotate.label(img, res),
            'mask': lambda img, res: pixelflow.annotate.mask(img, res, opacity=0.3),
            'blur': lambda img, res: pixelflow.annotate.blur(img, res),
            'pixelate': lambda img, res: pixelflow.annotate.pixelate(img, res),
            'footprint': lambda img, res: pixelflow.annotate.footprint(img, res)
        }
        
        results = {}
        mock_results = create_mock_results(10)
        
        for func_name, annotation_func in annotation_functions.items():
            frame_copy = frame.copy()
            
            start_time = time.time()
            
            for _ in range(num_frames):
                _ = annotation_func(frame_copy.copy(), mock_results)
            
            elapsed = time.time() - start_time
            fps = num_frames / elapsed
            
            results[func_name] = {
                'frames': num_frames,
                'time_seconds': elapsed,
                'fps': fps
            }
            
            print(f"   ✓ {func_name}: {fps:.1f} FPS")
        
        cap.release()
        
        self.results['pixelflow_annotations'] = results
        return results
    
    def benchmark_full_pipeline(self, num_frames=300):
        """Benchmark a complete pipeline with video reading and annotations."""
        print(f"\n🚀 Benchmarking full pipeline ({num_frames} frames)...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open {self.video_path}")
            return
        
        # Create mock results
        def create_mock_results(num_objects=5):
            predictions = []
            for i in range(num_objects):
                x = np.random.randint(100, 1800)
                y = np.random.randint(100, 900)
                predictions.append(Detection(
                    bbox=[x, y, x + 100, y + 100],
                    confidence=0.95,
                    class_id=i % 3,
                    class_name=f"object_{i % 3}",
                    tracker_id=i
                ))
            return Detections(detections=predictions)
        
        mock_results = create_mock_results(5)
        
        start_time = time.time()
        frames_processed = 0
        
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            
            # Simulate complete pipeline
            frame = cv2.resize(frame, (1280, 720))  # Resize
            frame = pixelflow.annotate.box(frame, mock_results)
            frame = pixelflow.annotate.label(frame, mock_results)
            
            frames_processed += 1
        
        elapsed = time.time() - start_time
        fps = frames_processed / elapsed
        
        cap.release()
        
        self.results['full_pipeline'] = {
            'frames': frames_processed,
            'time_seconds': elapsed,
            'fps': fps
        }
        
        print(f"   ✓ Processed {frames_processed} frames in {elapsed:.2f}s")
        print(f"   ✓ FPS: {fps:.1f}")
        
        return fps
    
    def save_results(self):
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"opencv_benchmark_{timestamp}.json"
        
        # Add metadata
        self.results['metadata'] = {
            'timestamp': timestamp,
            'video_path': self.video_path,
            'opencv_version': cv2.__version__
        }
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return output_file
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "="*50)
        print("📊 BENCHMARK SUMMARY")
        print("="*50)
        
        if 'video_reading' in self.results:
            print(f"\n🎬 Video Reading: {self.results['video_reading']['fps']:.1f} FPS")
        
        if 'image_operations' in self.results:
            print("\n🖼️ Image Operations (ops/sec):")
            for op, data in self.results['image_operations'].items():
                print(f"   • {op}: {data['ops_per_second']:.1f}")
        
        if 'drawing_operations' in self.results:
            print("\n✏️ Drawing Operations (ops/sec):")
            for op, data in self.results['drawing_operations'].items():
                print(f"   • {op}: {data['ops_per_second']:.1f}")
        
        if 'pixelflow_annotations' in self.results:
            print("\n🎨 PixelFlow Annotations (FPS):")
            for func, data in self.results['pixelflow_annotations'].items():
                print(f"   • {func}: {data['fps']:.1f}")
        
        if 'full_pipeline' in self.results:
            print(f"\n🚀 Full Pipeline: {self.results['full_pipeline']['fps']:.1f} FPS")
        
        print("="*50)


def main():
    print("🏁 Starting OpenCV Performance Benchmark")
    print("-" * 50)
    
    # Check if video exists
    video_path = "../examples/videos/paris.mp4"
    if not Path(video_path).exists():
        print(f"⚠️  Warning: {video_path} not found. Using camera instead.")
        video_path = 0
    
    benchmark = OpenCVBenchmark(video_path=video_path)
    
    # Run benchmarks
    benchmark.benchmark_video_reading(num_frames=500)
    benchmark.benchmark_image_operations(num_iterations=1000)
    benchmark.benchmark_drawing_operations(num_iterations=10000)
    benchmark.benchmark_pixelflow_annotations(num_frames=500)
    benchmark.benchmark_full_pipeline(num_frames=300)
    
    # Save and display results
    benchmark.save_results()
    benchmark.print_summary()
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()