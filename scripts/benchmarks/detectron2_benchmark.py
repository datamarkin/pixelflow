"""
Detectron2 Performance Benchmark for PixelFlow

Tests Detectron2 model inference performance including:
- Different model architectures (Faster R-CNN, Mask R-CNN, RetinaNet)
- GPU vs CPU performance
- Different input resolutions
- PixelFlow integration overhead
- Batch processing capabilities
"""

import cv2
import time
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import pixelflow
from pixelflow.detections import from_detectron2
from pixelflow.tracker import ByteTracker


class Detectron2Benchmark:
    def __init__(self, video_path="../examples/videos/paris.mp4", output_dir="benchmarks/results"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🖥️  Using device: {self.device}")
        if self.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        
    def setup_model(self, model_config, threshold=0.5):
        """Setup a Detectron2 model with given configuration."""
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_config))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.DEVICE = self.device
        
        return DefaultPredictor(cfg), cfg
    
    def benchmark_model_loading(self):
        """Benchmark model loading times for different architectures."""
        print(f"\n📦 Benchmarking model loading times...")
        
        models = [
            ("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "Faster R-CNN R50"),
            ("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml", "Faster R-CNN R101"),
            ("COCO-Detection/retinanet_R_50_FPN_3x.yaml", "RetinaNet R50"),
            ("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", "Mask R-CNN R50"),
        ]
        
        loading_times = {}
        
        for model_config, model_name in models:
            try:
                start_time = time.time()
                predictor, cfg = self.setup_model(model_config)
                load_time = time.time() - start_time
                
                loading_times[model_name] = {
                    'config': model_config,
                    'load_time_seconds': load_time
                }
                
                print(f"   ✓ {model_name}: {load_time:.2f}s")
                
                # Clean up
                del predictor
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"   ✗ {model_name}: Failed - {str(e)}")
                loading_times[model_name] = {
                    'config': model_config,
                    'error': str(e)
                }
        
        self.results['model_loading'] = loading_times
        return loading_times
    
    def benchmark_inference_speed(self, num_frames=100):
        """Benchmark inference speed for different models."""
        print(f"\n⚡ Benchmarking inference speed ({num_frames} frames)...")
        
        models = [
            ("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "Faster R-CNN R50"),
            ("COCO-Detection/retinanet_R_50_FPN_3x.yaml", "RetinaNet R50"),
        ]
        
        # Load test frames
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open {self.video_path}")
            return
        
        test_frames = []
        for _ in range(min(num_frames, 10)):  # Load up to 10 unique frames
            ret, frame = cap.read()
            if ret:
                test_frames.append(frame)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.release()
        
        if not test_frames:
            print("Error: No frames could be loaded")
            return
        
        inference_results = {}
        
        for model_config, model_name in models:
            try:
                print(f"\n   Testing {model_name}...")
                predictor, cfg = self.setup_model(model_config, threshold=0.3)
                
                # Warmup
                for _ in range(3):
                    _ = predictor(test_frames[0])
                
                # Benchmark
                start_time = time.time()
                total_detections = 0
                
                for i in range(num_frames):
                    frame = test_frames[i % len(test_frames)]
                    outputs = predictor(frame)
                    total_detections += len(outputs["instances"])
                
                elapsed = time.time() - start_time
                fps = num_frames / elapsed
                avg_detections = total_detections / num_frames
                
                inference_results[model_name] = {
                    'frames': num_frames,
                    'time_seconds': elapsed,
                    'fps': fps,
                    'avg_detections_per_frame': avg_detections
                }
                
                print(f"   ✓ FPS: {fps:.2f}, Avg detections: {avg_detections:.1f}")
                
                # Clean up
                del predictor
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"   ✗ {model_name}: Failed - {str(e)}")
                inference_results[model_name] = {'error': str(e)}
        
        self.results['inference_speed'] = inference_results
        return inference_results
    
    def benchmark_resolution_impact(self):
        """Benchmark impact of different input resolutions."""
        print(f"\n📐 Benchmarking resolution impact...")
        
        resolutions = [
            (640, 480, "480p"),
            (1280, 720, "720p"),
            (1920, 1080, "1080p"),
        ]
        
        # Setup model
        model_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        try:
            predictor, cfg = self.setup_model(model_config)
        except Exception as e:
            print(f"Error setting up model: {e}")
            return
        
        # Create test images at different resolutions
        resolution_results = {}
        
        for width, height, name in resolutions:
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Warmup
            for _ in range(3):
                _ = predictor(test_image)
            
            # Benchmark
            num_iterations = 50
            start_time = time.time()
            
            for _ in range(num_iterations):
                _ = predictor(test_image)
            
            elapsed = time.time() - start_time
            fps = num_iterations / elapsed
            
            resolution_results[name] = {
                'resolution': f"{width}x{height}",
                'iterations': num_iterations,
                'time_seconds': elapsed,
                'fps': fps
            }
            
            print(f"   ✓ {name} ({width}x{height}): {fps:.2f} FPS")
        
        # Clean up
        del predictor
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        self.results['resolution_impact'] = resolution_results
        return resolution_results
    
    def benchmark_pixelflow_integration(self, num_frames=100):
        """Benchmark complete pipeline with PixelFlow integration."""
        print(f"\n🔄 Benchmarking PixelFlow integration ({num_frames} frames)...")
        
        # Setup model
        model_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        try:
            predictor, cfg = self.setup_model(model_config, threshold=0.5)
            metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            class_names = metadata.get("thing_classes", None)
        except Exception as e:
            print(f"Error setting up model: {e}")
            return
        
        # Setup tracker
        tracker = ByteTracker(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8
        )
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open {self.video_path}")
            return
        
        # Test different pipeline configurations
        pipelines = {
            'detection_only': {
                'description': 'Detection only',
                'process': lambda frame, pred: pred(frame)
            },
            'detection_conversion': {
                'description': 'Detection + PixelFlow conversion',
                'process': lambda frame, pred: from_detectron2(pred(frame))
            },
            'detection_tracking': {
                'description': 'Detection + Tracking',
                'process': lambda frame, pred: tracker.update(from_detectron2(pred(frame)))
            },
            'full_pipeline': {
                'description': 'Detection + Tracking + Annotations',
                'process': lambda frame, pred: self._full_pipeline(frame, pred, tracker)
            }
        }
        
        pipeline_results = {}
        
        for pipeline_name, pipeline_config in pipelines.items():
            print(f"\n   Testing: {pipeline_config['description']}")
            
            # Reset video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            start_time = time.time()
            frames_processed = 0
            
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                
                _ = pipeline_config['process'](frame, predictor)
                frames_processed += 1
            
            elapsed = time.time() - start_time
            fps = frames_processed / elapsed
            
            pipeline_results[pipeline_name] = {
                'description': pipeline_config['description'],
                'frames': frames_processed,
                'time_seconds': elapsed,
                'fps': fps
            }
            
            print(f"   ✓ FPS: {fps:.2f}")
        
        cap.release()
        
        # Clean up
        del predictor
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        self.results['pixelflow_integration'] = pipeline_results
        return pipeline_results
    
    def _full_pipeline(self, frame, predictor, tracker):
        """Helper method for full pipeline processing."""
        outputs = predictor(frame)
        results = from_detectron2(outputs)
        results = tracker.update(results)
        
        # Apply annotations
        frame = pixelflow.annotate.box(frame, results)
        frame = pixelflow.annotate.label(frame, results)
        
        return frame
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage for different models."""
        if self.device != "cuda":
            print("\n💾 Memory benchmark skipped (GPU not available)")
            return
        
        print(f"\n💾 Benchmarking GPU memory usage...")
        
        models = [
            ("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "Faster R-CNN R50"),
            ("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml", "Faster R-CNN R101"),
            ("COCO-Detection/retinanet_R_50_FPN_3x.yaml", "RetinaNet R50"),
        ]
        
        memory_results = {}
        
        for model_config, model_name in models:
            try:
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Load model
                predictor, cfg = self.setup_model(model_config)
                
                # Create test image
                test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
                
                # Run inference
                _ = predictor(test_image)
                
                # Get memory stats
                allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                reserved = torch.cuda.memory_reserved() / 1024**2  # MB
                peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
                
                memory_results[model_name] = {
                    'allocated_mb': allocated,
                    'reserved_mb': reserved,
                    'peak_mb': peak
                }
                
                print(f"   ✓ {model_name}: {peak:.1f} MB peak")
                
                # Clean up
                del predictor
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"   ✗ {model_name}: Failed - {str(e)}")
                memory_results[model_name] = {'error': str(e)}
        
        self.results['memory_usage'] = memory_results
        return memory_results
    
    def save_results(self):
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"detectron2_benchmark_{timestamp}.json"
        
        # Add metadata
        self.results['metadata'] = {
            'timestamp': timestamp,
            'video_path': self.video_path,
            'device': self.device,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            self.results['metadata']['gpu_name'] = torch.cuda.get_device_name(0)
            self.results['metadata']['cuda_version'] = torch.version.cuda
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return output_file
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "="*60)
        print("📊 DETECTRON2 BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"\n🖥️  Device: {self.device}")
        if self.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        if 'model_loading' in self.results:
            print("\n📦 Model Loading Times:")
            for model, data in self.results['model_loading'].items():
                if 'load_time_seconds' in data:
                    print(f"   • {model}: {data['load_time_seconds']:.2f}s")
        
        if 'inference_speed' in self.results:
            print("\n⚡ Inference Speed (FPS):")
            for model, data in self.results['inference_speed'].items():
                if 'fps' in data:
                    print(f"   • {model}: {data['fps']:.2f} FPS")
        
        if 'resolution_impact' in self.results:
            print("\n📐 Resolution Impact (FPS):")
            for res, data in self.results['resolution_impact'].items():
                print(f"   • {res}: {data['fps']:.2f}")
        
        if 'pixelflow_integration' in self.results:
            print("\n🔄 PixelFlow Integration (FPS):")
            for pipeline, data in self.results['pixelflow_integration'].items():
                print(f"   • {data['description']}: {data['fps']:.2f}")
        
        if 'memory_usage' in self.results and self.device == "cuda":
            print("\n💾 GPU Memory Usage (Peak MB):")
            for model, data in self.results['memory_usage'].items():
                if 'peak_mb' in data:
                    print(f"   • {model}: {data['peak_mb']:.1f} MB")
        
        print("="*60)


def main():
    print("🏁 Starting Detectron2 Performance Benchmark")
    print("-" * 60)
    
    # Check if video exists
    video_path = "../examples/videos/paris.mp4"
    if not Path(video_path).exists():
        print(f"⚠️  Warning: {video_path} not found.")
        print("Please ensure video file exists or modify path.")
        return
    
    benchmark = Detectron2Benchmark(video_path=video_path)
    
    # Run benchmarks
    try:
        benchmark.benchmark_model_loading()
        benchmark.benchmark_inference_speed(num_frames=50)
        benchmark.benchmark_resolution_impact()
        benchmark.benchmark_pixelflow_integration(num_frames=50)
        
        if torch.cuda.is_available():
            benchmark.benchmark_memory_usage()
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
    
    # Save and display results
    benchmark.save_results()
    benchmark.print_summary()
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()