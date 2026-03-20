#!/usr/bin/env python3
"""
Sliced Inference Demo for PixelFlow

This example demonstrates how to use the SlicedInference class for detecting
small objects in large images by slicing the image into overlapping tiles.

Usage:
    python slicer_demo.py [image_path] [model_type]
    
    image_path: Path to the image file (optional, uses sample image if not provided)
    model_type: 'yolo' or 'detectron2' (optional, defaults to mock detector)
"""

#TODO Is this the best way to implement slicer to PixelFlow?

import sys
import numpy as np
import cv2
from pathlib import Path

# Add the parent directory to Python path to import pixelflow
sys.path.insert(0, str(Path(__file__).parent.parent))

import pixelflow as pf
from pixelflow.slicer import SlicedInference, auto_slice_size
from pixelflow.detections import Detections, Detection


def mock_detector(image: np.ndarray, confidence=0.25) -> Detections:
    """
    Mock object detector that creates fake detections for testing.
    
    This simulates a real detector by finding bright regions in the image
    and creating bounding boxes around them.
    """
    results = Detections()
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Find bright regions (simulate object detection)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 100:  # Filter small areas
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create a prediction
            prediction = Detection(
                bbox=[x, y, x + w, y + h],
                class_id=0,
                class_name="bright_object",
                confidence=0.8 + 0.2 * np.random.random()  # Random confidence
            )
            results.add_detection(prediction)
    
    return results


def yolo_detector_wrapper(image: np.ndarray, model, confidence=0.25) -> Results:
    """
    Wrapper for YOLO detector that converts output to PixelFlow Results.
    
    Args:
        image: Input image
        model: YOLO model instance
        confidence: Confidence threshold
        
    Returns:
        Detections object with detections
    """
    try:
        # Run YOLO inference
        detections = model(image, conf=confidence, verbose=False)
        
        # Convert to PixelFlow format
        results = pf.results.from_ultralytics(detections)
        return results
    
    except Exception as e:
        print(f"YOLO detection failed: {e}")
        return Detections()


def detectron2_detector_wrapper(image: np.ndarray, predictor, confidence=0.25) -> Results:
    """
    Wrapper for Detectron2 detector that converts output to PixelFlow Results.
    
    Args:
        image: Input image
        predictor: Detectron2 predictor instance
        confidence: Confidence threshold
        
    Returns:
        Detections object with detections
    """
    try:
        # Run Detectron2 inference
        outputs = predictor(image)
        
        # Convert to PixelFlow format
        results = pf.results.from_detectron2(outputs)
        
        # Apply confidence filtering
        results = results.filter_by_confidence(confidence)
        return results
    
    except Exception as e:
        print(f"Detectron2 detection failed: {e}")
        return Detections()


def create_test_image(width: int = 2000, height: int = 1500) -> np.ndarray:
    """
    Create a synthetic test image with objects at various scales.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Test image as numpy array
    """
    # Create base image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image.fill(50)  # Dark gray background
    
    # Add some "objects" - bright rectangles and circles of various sizes
    objects = [
        # Large objects
        ((100, 100), (300, 250), (255, 100, 100)),  # Red rectangle
        ((width-400, 50), (width-50, 200), (100, 255, 100)),  # Green rectangle
        
        # Medium objects
        ((width//2-75, height//2-75), (width//2+75, height//2+75), (100, 100, 255)),  # Blue square
        ((500, 800), (650, 900), (255, 255, 100)),  # Yellow rectangle
        
        # Small objects scattered around
        ((800, 200), (850, 230), (255, 200, 200)),
        ((1200, 400), (1240, 440), (200, 255, 200)),
        ((300, 1000), (330, 1030), (200, 200, 255)),
        ((1500, 1200), (1520, 1220), (255, 255, 200)),
    ]
    
    # Draw rectangles
    for (x1, y1), (x2, y2), color in objects:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    
    # Add some circles
    circles = [
        ((400, 400), 60, (255, 150, 0)),
        ((1600, 600), 30, (0, 255, 255)),
        ((200, 1200), 40, (255, 0, 255)),
    ]
    
    for (x, y), radius, color in circles:
        cv2.circle(image, (x, y), radius, color, -1)
    
    return image


def main():
    """Main function to demonstrate sliced inference."""
    
    # Parse command line arguments
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    model_type = sys.argv[2] if len(sys.argv) > 2 else "mock"
    
    # Load or create test image
    if image_path and Path(image_path).exists():
        print(f"Loading image from: {image_path}")
        image = pf.read_image(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
    else:
        print("Creating synthetic test image...")
        image = create_test_image(22012, 15034)
        # Save test image for reference
        cv2.imwrite("test_image.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print("Saved test image as 'test_image.jpg'")
    
    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]} (WxH)")
    
    # Calculate optimal slice size
    slice_height, slice_width = auto_slice_size(image.shape[0], image.shape[1], target_size=640)
    print(f"Optimal slice size: {slice_width}x{slice_height}")
    
    # Setup detector
    detector_func = mock_detector  # Default to mock detector
    
    if model_type.lower() == "yolo":
        try:
            import ultralytics
            model = ultralytics.YOLO('yolov8n.pt')  # Load nano model
            detector_func = lambda img: yolo_detector_wrapper(img, model, confidence=0.25)
            print("Using YOLOv8 detector")
        except ImportError:
            print("Ultralytics not available, using mock detector")
            detector_func = mock_detector
    
    elif model_type.lower() == "detectron2":
        try:
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            predictor = DefaultPredictor(cfg)
            
            detector_func = lambda img: detectron2_detector_wrapper(img, predictor, confidence=0.25)
            print("Using Detectron2 Faster R-CNN")
        except ImportError:
            print("Detectron2 not available, using mock detector")
            detector_func = mock_detector
    
    else:
        print("Using mock detector")
    
    # Initialize sliced inference
    slicer = SlicedInference(
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_ratio_h=0.25,  # 25% overlap for better boundary detection
        overlap_ratio_w=0.25,
        iou_threshold=0.4,     # Lower threshold for slice boundary objects
        ios_threshold=0.5,     # For nested object handling
        merge_mode='nms'       # Use standard NMS
    )
    
    print(f"\nSliced Inference Configuration:")
    print(f"  Slice size: {slice_width}x{slice_height}")
    print(f"  Overlap: {slicer.overlap_ratio_h:.1%}h x {slicer.overlap_ratio_w:.1%}w")
    print(f"  IOU threshold: {slicer.iou_threshold}")
    print(f"  Merge mode: {slicer.merge_mode}")
    
    # Run regular inference for comparison
    print(f"\nRunning regular inference on full image...")
    regular_results = detector_func(image)
    print(f"Regular inference found: {len(regular_results)} objects")
    
    # Run sliced inference
    print(f"\nRunning sliced inference...")
    sliced_results = slicer.predict(image, detector_func, verbose=True)
    print(f"Sliced inference found: {len(sliced_results)} objects")
    
    # Visualize results
    print(f"\nVisualizing results...")
    
    # Create visualization of regular results
    regular_vis = image.copy()
    regular_vis = pf.annotate.box(regular_vis, regular_results, thickness=2)
    regular_vis = pf.annotate.label(regular_vis, regular_results, position='top_left')
    
    # Create visualization of sliced results
    sliced_vis = image.copy()
    sliced_vis = pf.annotate.box(sliced_vis, sliced_results, thickness=2)
    sliced_vis = pf.annotate.label(sliced_vis, sliced_results, position='top_left')
    
    # Create slice grid visualization
    grid_vis = image.copy()
    slices = slicer.generate_slices(image.shape[0], image.shape[1])
    for x1, y1, x2, y2, slice_id in slices:
        cv2.rectangle(grid_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(grid_vis, str(slice_id), (x1+5, y1+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Save results (convert RGB→BGR for OpenCV file output)
    cv2.imwrite("regular_inference.jpg", cv2.cvtColor(regular_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite("sliced_inference.jpg", cv2.cvtColor(sliced_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite("slice_grid.jpg", cv2.cvtColor(grid_vis, cv2.COLOR_RGB2BGR))
    
    print(f"\nResults saved:")
    print(f"  regular_inference.jpg - Regular inference results")
    print(f"  sliced_inference.jpg - Sliced inference results")  
    print(f"  slice_grid.jpg - Visualization of slice grid")
    
    # Print detection summary
    print(f"\nDetection Summary:")
    print(f"  Regular inference: {len(regular_results)} detections")
    print(f"  Sliced inference: {len(sliced_results)} detections")
    print(f"  Improvement: {len(sliced_results) - len(regular_results):+d} detections")
    
    # Display confidence statistics
    if len(sliced_results) > 0:
        confidences = [p.confidence for p in sliced_results if p.confidence is not None]
        if confidences:
            print(f"  Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"  Average confidence: {np.mean(confidences):.3f}")


if __name__ == "__main__":
    main()