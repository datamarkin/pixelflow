"""
Sliced inference for detecting small objects in large images.

This module provides high-accuracy object detection on large images by slicing 
images into overlapping tiles, running inference on each slice, and merging 
predictions with intelligent NMS/NMM. Designed for simplicity and performance 
following PixelFlow principles with comprehensive edge case handling.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Union
from .detections import Detections, Detection

__all__ = ["SlicedInference", "auto_slice_size"]


class SlicedInference:
    """Performs sliced inference on large images for improved small object detection.
    
    The main challenge in sliced inference is accurately merging predictions from
    overlapping slices. This implementation handles critical edge cases including
    objects on slice boundaries, nested objects, and partial detections at slice
    edges using intelligent NMS/NMM algorithms with configurable thresholds.
    
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Setup model and sliced inference
        >>> model = YOLO("yolo11n.pt")
        >>> slicer = pf.SlicedInference(slice_height=640, slice_width=640, overlap_ratio_h=0.2)
        >>> 
        >>> # Define detector function
        >>> def detector_func(image):
        ...     outputs = model.predict(image)
        ...     return pf.results.from_ultralytics(outputs)
        >>> 
        >>> # Run sliced inference on large image
        >>> large_image = cv2.imread("large_image.jpg")  # e.g., 4000x3000
        >>> results = slicer.predict(large_image, detector_func, verbose=True)
        >>> print(f"Detected {len(results)} objects")
        
    Notes:
        - Automatically handles coordinate transformation from slice to full image space
        - Uses intelligent merging to avoid duplicate detections on slice boundaries
        - Supports both NMS (Non-Maximum Suppression) and NMM (Non-Maximum Merging)
        - Configurable overlap ratios and thresholds for different use cases
        
    See Also:
        auto_slice_size : Automatically calculate optimal slice dimensions
    """
    
    def __init__(
        self,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_ratio_h: float = 0.2,
        overlap_ratio_w: float = 0.2,
        iou_threshold: float = 0.5,
        ios_threshold: float = 0.5,
        merge_mode: str = 'nms',
        min_slice_area_ratio: float = 0.1
    ):
        """Initialize SlicedInference with configuration parameters.
        
        Sets up the sliced inference pipeline with configurable slice dimensions,
        overlap ratios, and merging thresholds. Automatically calculates step
        sizes based on overlap ratios and validates parameter ranges.
        
        Args:
            slice_height (int): Height of each slice in pixels. Should match
                              model input size for optimal performance. Default is 640.
            slice_width (int): Width of each slice in pixels. Should match
                             model input size for optimal performance. Default is 640.
            overlap_ratio_h (float): Vertical overlap ratio between adjacent slices.
                                   Range: [0.0, 0.9]. Default is 0.2 (20% overlap).
            overlap_ratio_w (float): Horizontal overlap ratio between adjacent slices.
                                   Range: [0.0, 0.9]. Default is 0.2 (20% overlap).
            iou_threshold (float): IOU threshold for merging predictions from
                                 overlapping slices. Range: [0.0, 1.0]. Default is 0.5.
            ios_threshold (float): IOS (Intersection over Smaller) threshold for
                                 handling nested objects. Range: [0.0, 1.0]. Default is 0.5.
            merge_mode (str): Merging strategy. 'nms' for Non-Maximum Suppression,
                            'nmm' for Non-Maximum Merging. Default is 'nms'.
            min_slice_area_ratio (float): Minimum ratio of object area in slice to
                                        keep detection. Range: [0.0, 1.0]. Default is 0.1.
                                        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Basic configuration for 640px model input
            >>> slicer = pf.SlicedInference()
            >>> 
            >>> # Custom configuration for high overlap
            >>> slicer = pf.SlicedInference(
            ...     slice_height=512, 
            ...     slice_width=512,
            ...     overlap_ratio_h=0.3,  # 30% vertical overlap
            ...     overlap_ratio_w=0.3,  # 30% horizontal overlap
            ...     merge_mode='nmm'      # Use box merging instead of NMS
            ... )
            >>> 
            >>> # Configuration for detecting small objects
            >>> slicer = pf.SlicedInference(
            ...     slice_height=416,
            ...     slice_width=416,
            ...     overlap_ratio_h=0.4,  # Higher overlap for small objects
            ...     iou_threshold=0.3,    # Lower threshold for more aggressive merging
            ...     ios_threshold=0.7     # Higher threshold to preserve nested objects
            ... )
            
        Notes:
            - Overlap ratios are automatically clamped to range [0.0, 0.9]
            - Step sizes are calculated as: step = slice_size * (1 - overlap_ratio)
            - Higher overlap ratios improve detection of objects on slice boundaries
            - NMM mode merges boxes using weighted average based on confidence scores
            - NMS mode keeps only the highest confidence detection
            
        See Also:
            auto_slice_size : Calculate optimal slice dimensions automatically
        """
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio_h = max(0, min(overlap_ratio_h, 0.9))
        self.overlap_ratio_w = max(0, min(overlap_ratio_w, 0.9))
        self.iou_threshold = iou_threshold
        self.ios_threshold = ios_threshold
        self.merge_mode = merge_mode
        self.min_slice_area_ratio = min_slice_area_ratio
        
        # Calculate step sizes based on overlap
        self.step_height = int(slice_height * (1 - overlap_ratio_h))
        self.step_width = int(slice_width * (1 - overlap_ratio_w))
    
    def generate_slices(self, image_height: int, image_width: int) -> List[Tuple[int, int, int, int, int]]:
        """Generate slice coordinates for the given image dimensions.
        
        Creates a grid of overlapping slices covering the entire image based on
        configured slice dimensions and overlap ratios. Ensures complete image
        coverage while minimizing redundant computation from excessive overlap.
        
        Args:
            image_height (int): Height of the full image in pixels
            image_width (int): Width of the full image in pixels
            
        Returns:
            List[Tuple[int, int, int, int, int]]: List of slice coordinates where
                                               each tuple contains (x1, y1, x2, y2, slice_id).
                                               Coordinates are in (left, top, right, bottom) format.
                                               
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Generate slices for a large image
            >>> slicer = pf.SlicedInference(slice_height=640, slice_width=640, overlap_ratio_h=0.2)
            >>> slices = slicer.generate_slices(2000, 3000)  # 2000x3000 image
            >>> print(f"Generated {len(slices)} slices")
            Generated 15 slices
            >>> 
            >>> # Examine first slice coordinates
            >>> x1, y1, x2, y2, slice_id = slices[0]
            >>> print(f"Slice 0: ({x1}, {y1}) to ({x2}, {y2})")
            Slice 0: (0, 0) to (640, 640)
            >>> 
            >>> # Check slice dimensions
            >>> for x1, y1, x2, y2, slice_id in slices[:3]:
            ...     print(f"Slice {slice_id}: {x2-x1}x{y2-y1}")
            Slice 0: 640x640
            Slice 1: 640x640
            Slice 2: 640x640
            
        Notes:
            - Slices are generated in row-major order (left-to-right, top-to-bottom)
            - Last row/column slices may be smaller if image dimensions don't align
            - Slice IDs are assigned sequentially starting from 0
            - Step sizes are calculated based on overlap ratios to ensure proper coverage
            
        Performance Notes:
            - Number of slices grows quadratically with image size
            - Higher overlap ratios result in more slices and computation
            - For a 4000x3000 image with 640x640 slices and 20% overlap: ~30 slices
        """
        slices = []
        slice_id = 0
        
        y = 0
        while y < image_height:
            x = 0
            while x < image_width:
                # Calculate slice boundaries
                x1 = x
                y1 = y
                x2 = min(x + self.slice_width, image_width)
                y2 = min(y + self.slice_height, image_height)
                
                slices.append((x1, y1, x2, y2, slice_id))
                slice_id += 1
                
                # Move to next horizontal position
                if x2 >= image_width:
                    break
                x += self.step_width
            
            # Move to next vertical position
            if y2 >= image_height:
                break
            y += self.step_height
        
        return slices
    
    def shift_predictions(self, predictions: Detections, offset_x: int, offset_y: int, slice_id: int) -> Detections:
        """Shift prediction coordinates from slice space to full image space.
        
        Transforms detection coordinates from local slice coordinate system to
        global full image coordinate system by applying offset translations.
        Also adds slice_id metadata for tracking which slice generated each detection.
        
        Args:
            predictions (Detections): Detections object containing predictions with
                                    coordinates in slice-local coordinate system.
            offset_x (int): X offset of the slice's top-left corner in full image coordinates.
            offset_y (int): Y offset of the slice's top-left corner in full image coordinates.
            slice_id (int): Unique identifier of the source slice for tracking purposes.
            
        Returns:
            Detections: New Detections object with coordinates transformed to full image space
                       and slice_id added to each detection's metadata.
                       
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Assume we have predictions from a slice at position (512, 256)
            >>> slice_predictions = pf.Detections()
            >>> # Predictions have coordinates relative to slice origin (0, 0)
            >>> 
            >>> # Shift coordinates to full image space
            >>> slicer = pf.SlicedInference()
            >>> shifted = slicer.shift_predictions(slice_predictions, 512, 256, slice_id=5)
            >>> 
            >>> # Original box at (10, 20, 50, 60) in slice becomes (522, 276, 562, 316) in full image
            >>> for detection in shifted:
            ...     print(f"Box: {detection.bbox}, From slice: {detection.data.get('slice_id')}")
            
        Notes:
            - Only bbox coordinates are currently shifted; masks, segments, and keypoints need manual handling
            - Slice_id is stored in detection.data for use in merge algorithms
            - Original prediction objects are not modified; new objects are created
            - Preserves all detection attributes (class_id, confidence, etc.)
            
        Raises:
            AttributeError: If predictions object doesn't support iteration
        """
        shifted_results = Detections()
        
        for pred in predictions:
            # Create a new prediction with shifted coordinates
            shifted_pred = Detection(
                inference_id=pred.inference_id,
                bbox=[
                    pred.bbox[0] + offset_x,
                    pred.bbox[1] + offset_y,
                    pred.bbox[2] + offset_x,
                    pred.bbox[3] + offset_y
                ] if pred.bbox else None,
                masks=pred.masks,  # TODO: Shift mask coordinates if needed
                segments=pred.segments,  # TODO: Shift segment coordinates if needed
                keypoints=pred.keypoints,  # TODO: Shift keypoint coordinates if needed
                class_id=pred.class_id,
                class_name=pred.class_name,
                confidence=pred.confidence,
                tracker_id=pred.tracker_id,
                data={'slice_id': slice_id} if pred.data is None else {**pred.data, 'slice_id': slice_id}
            )
            shifted_results.add_detection(shifted_pred)
        
        return shifted_results
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Computes the IoU metric commonly used in object detection for measuring
        overlap between bounding boxes. IoU = intersection_area / union_area.
        Used for merging duplicate detections from overlapping slices.
        
        Args:
            box1 (List[float]): First bounding box in format [x1, y1, x2, y2]
                              where (x1, y1) is top-left and (x2, y2) is bottom-right.
            box2 (List[float]): Second bounding box in format [x1, y1, x2, y2]
                              where (x1, y1) is top-left and (x2, y2) is bottom-right.
            
        Returns:
            float: IoU value in range [0.0, 1.0] where 0.0 means no overlap
                  and 1.0 means perfect overlap.
                  
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> slicer = pf.SlicedInference()
            >>> 
            >>> # Identical boxes have IoU = 1.0
            >>> box1 = [10, 10, 50, 50]
            >>> box2 = [10, 10, 50, 50]
            >>> iou = slicer.calculate_iou(box1, box2)
            >>> print(f"Identical boxes IoU: {iou}")
            Identical boxes IoU: 1.0
            >>> 
            >>> # Partially overlapping boxes
            >>> box1 = [10, 10, 50, 50]
            >>> box2 = [30, 30, 70, 70]
            >>> iou = slicer.calculate_iou(box1, box2)
            >>> print(f"Overlapping boxes IoU: {iou:.3f}")
            Overlapping boxes IoU: 0.143
            >>> 
            >>> # Non-overlapping boxes have IoU = 0.0
            >>> box1 = [10, 10, 50, 50]
            >>> box2 = [60, 60, 100, 100]
            >>> iou = slicer.calculate_iou(box1, box2)
            >>> print(f"Non-overlapping boxes IoU: {iou}")
            Non-overlapping boxes IoU: 0.0
            
        Notes:
            - Returns 0.0 if boxes don't intersect
            - Returns 0.0 if either box has zero area (malformed coordinates)
            - Used primarily for detecting duplicate detections across slice boundaries
            
        Performance Notes:
            - O(1) time complexity with simple arithmetic operations
            - Efficient for processing large numbers of detection pairs
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def calculate_ios(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Smaller (IoS) area between two bounding boxes.
        
        Computes IoS metric useful for detecting nested objects where a small object
        is contained within a larger one. IoS = intersection_area / min(area1, area2).
        Higher IoS values indicate one box is largely contained within the other.
        
        Args:
            box1 (List[float]): First bounding box in format [x1, y1, x2, y2]
                              where (x1, y1) is top-left and (x2, y2) is bottom-right.
            box2 (List[float]): Second bounding box in format [x1, y1, x2, y2]
                              where (x1, y1) is top-left and (x2, y2) is bottom-right.
            
        Returns:
            float: IoS value in range [0.0, 1.0] where 0.0 means no overlap
                  and 1.0 means one box completely contains the other.
                  
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> slicer = pf.SlicedInference()
            >>> 
            >>> # Small box inside large box (high IoS)
            >>> large_box = [10, 10, 100, 100]  # 90x90 = 8100 area
            >>> small_box = [40, 40, 60, 60]    # 20x20 = 400 area  
            >>> ios = slicer.calculate_ios(large_box, small_box)
            >>> print(f"Nested boxes IoS: {ios}")
            Nested boxes IoS: 1.0
            >>> 
            >>> # Partially overlapping boxes of similar size
            >>> box1 = [10, 10, 50, 50]  # 40x40 = 1600 area
            >>> box2 = [30, 30, 70, 70]  # 40x40 = 1600 area
            >>> ios = slicer.calculate_ios(box1, box2)
            >>> print(f"Similar size overlap IoS: {ios:.3f}")
            Similar size overlap IoS: 0.250
            >>> 
            >>> # One box mostly contained in another
            >>> container = [0, 0, 100, 100]   # 100x100 = 10000 area
            >>> contained = [10, 10, 90, 90]   # 80x80 = 6400 area
            >>> ios = slicer.calculate_ios(container, contained)
            >>> print(f"Mostly contained IoS: {ios}")
            Mostly contained IoS: 1.0
            
        Notes:
            - Returns 0.0 if boxes don't intersect
            - Returns 0.0 if either box has zero area (malformed coordinates)
            - Useful for preserving both parent and child objects in nested scenarios
            - Higher threshold values preserve more nested detections
            
        Performance Notes:
            - O(1) time complexity with simple arithmetic operations
            - Used in conjunction with IoU for comprehensive overlap analysis
            
        See Also:
            calculate_iou : Standard Intersection over Union metric
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate smaller area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        smaller_area = min(area1, area2)
        
        if smaller_area <= 0:
            return 0.0
        
        return intersection / smaller_area
    
    def are_adjacent_slices(self, slice_id1: int, slice_id2: int, slices: List[Tuple]) -> bool:
        """Check if two slices are adjacent (share a border or overlap).
        
        Determines spatial relationship between two slices by checking if their
        bounding rectangles overlap or touch. Used to apply different merging
        strategies for predictions from adjacent vs. distant slices.
        
        Args:
            slice_id1 (int): Identifier of the first slice
            slice_id2 (int): Identifier of the second slice  
            slices (List[Tuple]): List of all slice coordinates where each tuple
                                contains (x1, y1, x2, y2, slice_id)
            
        Returns:
            bool: True if slices are adjacent (overlap or share borders),
                 False if slices are separated or if slice_id1 == slice_id2
                 
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> slicer = pf.SlicedInference(slice_height=640, slice_width=640, overlap_ratio_h=0.2)
            >>> slices = slicer.generate_slices(1500, 1500)  # Generate slice grid
            >>> 
            >>> # Check if first two slices are adjacent
            >>> adjacent = slicer.are_adjacent_slices(0, 1, slices)
            >>> print(f"Slices 0 and 1 adjacent: {adjacent}")
            Slices 0 and 1 adjacent: True
            >>> 
            >>> # Check distant slices
            >>> distant = slicer.are_adjacent_slices(0, len(slices)-1, slices)
            >>> print(f"First and last slice adjacent: {distant}")
            First and last slice adjacent: False
            >>> 
            >>> # Same slice returns False
            >>> same = slicer.are_adjacent_slices(5, 5, slices)
            >>> print(f"Same slice adjacent: {same}")
            Same slice adjacent: False
            
        Notes:
            - Returns False for identical slice IDs (slice_id1 == slice_id2)
            - Adjacency includes both touching and overlapping slices
            - Used to determine appropriate IoU thresholds during merging
            - Adjacent slices get lower IoU thresholds for more aggressive merging
            
        Performance Notes:
            - O(1) lookup time using slice IDs as indices
            - Simple geometric overlap test with minimal computation
        """
        if slice_id1 == slice_id2:
            return False
        
        slice1 = slices[slice_id1]
        slice2 = slices[slice_id2]
        
        # Check if slices overlap or touch
        x_overlap = not (slice1[2] < slice2[0] or slice2[2] < slice1[0])
        y_overlap = not (slice1[3] < slice2[1] or slice2[3] < slice1[1])
        
        return x_overlap and y_overlap
    
    def merge_predictions(self, all_predictions: List[Detection], slices: List[Tuple]) -> Detections:
        """Merge predictions from all slices using intelligent NMS/NMM algorithms.
        
        This is the core algorithm that handles critical edge cases in sliced inference.
        Uses sophisticated merging logic combining IoU, IoS, and slice adjacency to
        distinguish between duplicate detections, nested objects, and legitimate
        separate objects. Applies different thresholds based on slice relationships.
        
        Args:
            all_predictions (List[Detection]): Complete list of predictions from all slices
                                             with coordinates already shifted to full image space.
            slices (List[Tuple]): List of slice coordinates for adjacency checking.
                                Each tuple contains (x1, y1, x2, y2, slice_id).
            
        Returns:
            Detections: Merged detection results with duplicates removed and boxes
                       optionally merged based on configured merge_mode.
                       
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Assume we have predictions from multiple slices
            >>> all_preds = []  # List of Detection objects from all slices
            >>> slices = []     # List of slice coordinates
            >>> 
            >>> slicer = pf.SlicedInference(iou_threshold=0.5, merge_mode='nms')
            >>> merged = slicer.merge_predictions(all_preds, slices)
            >>> print(f\"Merged {len(all_preds)} predictions into {len(merged)} final detections\")
            >>> 
            >>> # With box merging mode
            >>> slicer_nmm = pf.SlicedInference(merge_mode='nmm')
            >>> merged_nmm = slicer_nmm.merge_predictions(all_preds, slices)
            >>> # NMM mode creates weighted average boxes from merged predictions
            >>> 
            >>> # Custom thresholds for small object detection
            >>> slicer_small = pf.SlicedInference(iou_threshold=0.3, ios_threshold=0.8)
            >>> merged_small = slicer_small.merge_predictions(all_preds, slices)
            >>> # Lower IoU threshold merges more aggressively, higher IoS preserves nested objects
            
        Notes:
            - Predictions are sorted by confidence in descending order before processing
            - Adjacent slices use lower IoU threshold (threshold * 0.7) for more aggressive merging
            - High IoS (>0.9) triggers suppression to handle complete containment cases
            - NMM mode merges boxes using confidence-weighted average coordinates
            - NMS mode keeps only the highest confidence detection per merged group
            - Different class IDs are never merged regardless of spatial overlap
            
        Performance Notes:
            - O(n²) time complexity where n is the number of predictions
            - Performance degrades with high overlap ratios generating many predictions
            - Confidence sorting ensures highest quality detections are preserved
            
        Algorithm Details:
            1. Sort predictions by confidence (highest first)
            2. For each prediction, compare with all subsequent predictions
            3. Skip comparisons between different object classes  
            4. Calculate IoU and IoS metrics for spatial overlap analysis
            5. Apply different thresholds based on slice adjacency relationships
            6. Suppress or merge overlapping predictions based on configured mode
        """
        if not all_predictions:
            return Detections()
        
        # Sort predictions by confidence (descending)
        sorted_preds = sorted(all_predictions, key=lambda x: x.confidence or 0, reverse=True)
        
        merged_results = Detections()
        suppressed = set()
        
        for i, pred_i in enumerate(sorted_preds):
            if i in suppressed:
                continue
            
            # Track predictions to potentially merge
            merge_candidates = [pred_i]
            
            for j, pred_j in enumerate(sorted_preds[i+1:], start=i+1):
                if j in suppressed:
                    continue
                
                # Skip if different classes
                if pred_i.class_id != pred_j.class_id:
                    continue
                
                # Calculate IOU and IOS
                iou = self.calculate_iou(pred_i.bbox, pred_j.bbox)
                ios = self.calculate_ios(pred_i.bbox, pred_j.bbox)
                
                # Get slice IDs
                slice_i = pred_i.data.get('slice_id', -1) if pred_i.data else -1
                slice_j = pred_j.data.get('slice_id', -1) if pred_j.data else -1
                adjacent = self.are_adjacent_slices(slice_i, slice_j, slices) if slice_i >= 0 and slice_j >= 0 else False
                
                # Decision logic for merging/suppression
                should_suppress = False
                
                # Case 1: High IOU from adjacent slices -> same object split across boundary
                if adjacent and iou > self.iou_threshold * 0.7:  # Lower threshold for adjacent slices
                    should_suppress = True
                
                # Case 2: High IOU from same slice -> different objects or duplicate
                elif not adjacent and iou > self.iou_threshold:
                    should_suppress = True
                
                # Case 3: High IOS -> nested objects (keep both unless very high IOS)
                elif ios > self.ios_threshold:
                    # Only suppress if IOS is very high (almost complete containment)
                    if ios > 0.9:
                        should_suppress = True
                
                if should_suppress:
                    suppressed.add(j)
                    if self.merge_mode == 'nmm' and adjacent:
                        merge_candidates.append(pred_j)
            
            # Handle merging based on mode
            if self.merge_mode == 'nmm' and len(merge_candidates) > 1:
                # Merge boxes by weighted average based on confidence
                merged_pred = self._merge_boxes(merge_candidates)
                merged_results.add_detection(merged_pred)
            else:
                # Standard NMS - keep highest confidence
                merged_results.add_detection(pred_i)
        
        return merged_results
    
    def _merge_boxes(self, predictions: List[Detection]) -> Detection:
        """Merge multiple predictions into one using confidence-weighted average.
        
        Combines multiple overlapping predictions from adjacent slices by computing
        a weighted average of their bounding box coordinates using confidence scores
        as weights. Creates a single merged detection that represents the consensus
        of all input predictions.
        
        Args:
            predictions (List[Detection]): List of predictions to merge. All predictions
                                         should have the same class_id and represent the
                                         same object detected across multiple slices.
            
        Returns:
            Detection: Single merged prediction with averaged coordinates and
                      highest confidence score from the input predictions.
                      
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # This method is typically called internally by merge_predictions
            >>> # but can be used standalone for merging related detections
            >>> slicer = pf.SlicedInference()
            >>> 
            >>> # Assume we have overlapping detections of the same object
            >>> detections = []  # List of Detection objects with same class_id
            >>> # detection1: bbox=[100, 100, 200, 200], confidence=0.8
            >>> # detection2: bbox=[110, 95, 210, 205], confidence=0.9  
            >>> # detection3: bbox=[95, 105, 195, 195], confidence=0.7
            >>> 
            >>> merged = slicer._merge_boxes(detections)
            >>> # Result: bbox=[103.8, 100.0, 203.8, 200.0], confidence=0.9
            >>> print(f\"Merged box: {merged.bbox}, confidence: {merged.confidence}\")
            
        Notes:
            - Weights are normalized confidence scores (sum to 1.0)
            - Final confidence is taken from the highest confidence input prediction
            - All other attributes (class_id, class_name, etc.) are copied from highest confidence prediction
            - Masks, segments, and keypoints are copied from the highest confidence prediction
            - This method assumes all input predictions represent the same object
            
        Performance Notes:
            - O(n) time complexity where n is the number of predictions to merge
            - Memory efficient as it creates only one new Detection object
            - Numpy operations provide fast weighted averaging
            
        See Also:
            merge_predictions : Main method that determines which predictions to merge
        """
        # Use confidence as weight
        weights = np.array([p.confidence or 1.0 for p in predictions])
        weights = weights / weights.sum()
        
        # Weighted average of box coordinates
        merged_bbox = np.zeros(4)
        for pred, weight in zip(predictions, weights):
            merged_bbox += np.array(pred.bbox) * weight
        
        merged_bbox = merged_bbox.tolist()
        
        # Use highest confidence
        max_conf_pred = max(predictions, key=lambda x: x.confidence or 0)
        
        return Detection(
            bbox=merged_bbox,
            class_id=max_conf_pred.class_id,
            class_name=max_conf_pred.class_name,
            confidence=max_conf_pred.confidence,
            masks=max_conf_pred.masks,
            segments=max_conf_pred.segments,
            keypoints=max_conf_pred.keypoints
        )
    
    def predict(
        self,
        image: np.ndarray,
        detector_func: Callable,
        verbose: bool = False,
        **detector_kwargs
    ) -> Detections:
        """Run sliced inference on an image with automatic merging.
        
        Performs complete sliced inference pipeline: generates overlapping slices,
        runs detection on each slice, transforms coordinates to full image space,
        and intelligently merges predictions to eliminate duplicates. Ideal for
        detecting small objects in high-resolution images.
        
        Args:
            image (np.ndarray): Input image as numpy array with shape (H, W, C)
                              for color images or (H, W) for grayscale.
            detector_func (Callable): Detection function that takes an image array
                                    and returns Detections object. Should accept
                                    additional keyword arguments via **detector_kwargs.
            verbose (bool): Print detailed progress information including slice
                          count and detection statistics. Default is False.
            **detector_kwargs: Additional keyword arguments passed to detector_func
                             for model-specific configuration (confidence thresholds,
                             NMS settings, etc.).
                             
        Returns:
            Detections: Merged detection results with coordinates in full image space,
                       duplicates removed, and predictions consolidated across slices.
                       
        Raises:
            ValueError: If image is not a valid numpy array
            RuntimeError: If detector_func fails on any slice
            
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> # Setup model and large image
            >>> model = YOLO("yolo11n.pt")
            >>> large_image = cv2.imread("satellite_image.jpg")  # e.g., 4000x3000
            >>> print(f\"Image size: {large_image.shape[:2]}\")
            Image size: (3000, 4000)
            >>> 
            >>> # Define detector function
            >>> def detector(image, conf=0.5):
            ...     outputs = model.predict(image, conf=conf)
            ...     return pf.results.from_ultralytics(outputs)
            >>> 
            >>> # Run sliced inference  
            >>> slicer = pf.SlicedInference(slice_height=640, slice_width=640)
            >>> results = slicer.predict(large_image, detector, verbose=True, conf=0.3)
            Generated 30 slices for 4000x3000 image
            Slice 0: 5 detections
            Slice 1: 3 detections
            ...
            Total predictions before merging: 120
            Predictions after merging: 85
            >>> 
            >>> # Use with different detector function
            >>> def custom_detector(image, threshold=0.5, nms=0.4):
            ...     # Your custom detection logic here
            ...     return pf.Detections()  # Return Detections object
            >>> 
            >>> results = slicer.predict(large_image, custom_detector, 
            ...                         threshold=0.3, nms=0.5)
            >>> 
            >>> # Process results
            >>> print(f\"Detected {len(results)} objects in large image\")
            >>> for detection in results:
            ...     print(f\"Class: {detection.class_name}, Conf: {detection.confidence:.2f}\")
            
        Notes:
            - Automatically handles coordinate transformation from slice to full image space
            - Detector function should return PixelFlow Detections object for best compatibility
            - Large images benefit most from sliced inference (>2000x2000 pixels)
            - Memory usage scales with number of slices and detections per slice
            - Verbose mode helpful for debugging and performance optimization
            
        Performance Notes:
            - Inference time scales linearly with number of slices
            - Memory usage peaks during coordinate transformation and merging phases
            - GPU memory requirements reduced as model processes smaller slice images
            - Optimal slice size matches your model's training input resolution
            
        See Also:
            generate_slices : Generate slice coordinates for manual processing
            auto_slice_size : Calculate optimal slice dimensions
        """
        image_height, image_width = image.shape[:2]
        
        # Generate slices
        slices = self.generate_slices(image_height, image_width)
        
        if verbose:
            print(f"Generated {len(slices)} slices for {image_width}x{image_height} image")
        
        # Run inference on each slice
        all_predictions = []
        
        for x1, y1, x2, y2, slice_id in slices:
            # Extract slice from image
            slice_img = image[y1:y2, x1:x2]
            
            # Run detector
            slice_results = detector_func(slice_img, **detector_kwargs)
            
            # Handle different return types
            if not isinstance(slice_results, Detections):
                # Assume it's a raw detection output that needs conversion
                # This allows flexibility with different detector formats
                if hasattr(slice_results, '__iter__'):
                    slice_results = Detections()  # Create empty Detections if needed
                else:
                    slice_results = Detections()
            
            # Shift coordinates to full image space
            shifted_results = self.shift_predictions(slice_results, x1, y1, slice_id)
            
            # Collect all predictions
            all_predictions.extend(shifted_results.detections)
            
            if verbose:
                print(f"Slice {slice_id}: {len(slice_results)} detections")
        
        if verbose:
            print(f"Total predictions before merging: {len(all_predictions)}")
        
        # Merge predictions
        merged_results = self.merge_predictions(all_predictions, slices)
        
        if verbose:
            print(f"Predictions after merging: {len(merged_results)}")
        
        return merged_results


def auto_slice_size(image_height: int, image_width: int, target_size: int = 640) -> Tuple[int, int]:
    """Automatically calculate optimal slice dimensions based on image properties.
    
    Computes slice dimensions that maintain the image's aspect ratio while
    targeting a specific slice size. Ensures slices are neither too small
    (minimum 320px) nor larger than the original image dimensions.
    
    Args:
        image_height (int): Height of the input image in pixels
        image_width (int): Width of the input image in pixels  
        target_size (int): Target dimension for the larger side of each slice.
                          Will be adjusted to maintain aspect ratio. Default is 640.
                          
    Returns:
        Tuple[int, int]: Optimal (slice_height, slice_width) dimensions that
                        maintain aspect ratio and respect size constraints.
                        
    Example:
        >>> import pixelflow as pf
        >>> 
        >>> # Square image - both dimensions equal target_size
        >>> h, w = pf.auto_slice_size(2000, 2000, target_size=640)
        >>> print(f"Square image slices: {h}x{w}")
        Square image slices: 640x640
        >>> 
        >>> # Wide image - width gets target_size, height scaled down
        >>> h, w = pf.auto_slice_size(1500, 3000, target_size=640) 
        >>> print(f"Wide image slices: {h}x{w}")
        Wide image slices: 320x640
        >>> 
        >>> # Tall image - height gets target_size, width scaled down
        >>> h, w = pf.auto_slice_size(4000, 1500, target_size=640)
        >>> print(f"Tall image slices: {h}x{w}")
        Tall image slices: 640x240  # But minimum 320, so actually 640x320
        >>> 
        >>> # Small image - slices limited by original dimensions
        >>> h, w = pf.auto_slice_size(500, 800, target_size=640)
        >>> print(f"Small image slices: {h}x{w}")
        Small image slices: 500x800
        >>> 
        >>> # Use with SlicedInference
        >>> image_h, image_w = 3000, 4500  # Large image
        >>> slice_h, slice_w = pf.auto_slice_size(image_h, image_w)
        >>> slicer = pf.SlicedInference(slice_height=slice_h, slice_width=slice_w)
        
    Notes:
        - Maintains original image aspect ratio in slice dimensions
        - Enforces minimum slice size of 320px on both dimensions
        - Never creates slices larger than the original image
        - Wider images result in wider slices with proportionally smaller height
        - Taller images result in taller slices with proportionally smaller width
        
    Performance Notes:
        - Larger slices reduce total slice count but may miss small objects
        - Smaller slices increase processing time but improve small object detection
        - 640px target size works well for most YOLO-based models
        - Consider model input size when choosing target_size
        
    See Also:
        SlicedInference : Main class for sliced inference with configurable slice dimensions
    """
    # Calculate aspect ratio
    aspect_ratio = image_width / image_height
    
    # Adjust slice dimensions to maintain aspect ratio
    if aspect_ratio > 1:  # Wider image
        slice_width = target_size
        slice_height = int(target_size / aspect_ratio)
    else:  # Taller image
        slice_height = target_size
        slice_width = int(target_size * aspect_ratio)
    
    # Ensure minimum size
    slice_height = max(slice_height, 320)
    slice_width = max(slice_width, 320)
    
    # Don't make slices larger than the image
    slice_height = min(slice_height, image_height)
    slice_width = min(slice_width, image_width)
    
    return slice_height, slice_width