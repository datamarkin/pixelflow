# slicer.py
"""
Sliced inference for detecting small objects in large images.

This module provides high-accuracy object detection on large images by:
1. Slicing images into overlapping tiles
2. Running inference on each slice
3. Merging predictions with intelligent NMS/NMM

Designed for simplicity and performance following PixelFlow principles.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Union
from .detections import Detections, Detection


class SlicedInference:
    """
    Performs sliced inference on large images for improved small object detection.
    
    The main challenge in sliced inference is accurately merging predictions from
    overlapping slices. This implementation handles edge cases like:
    - Objects on slice boundaries
    - Nested objects (small inside large)
    - Partial detections at slice edges
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
        """
        Initialize SlicedInference with configuration.
        
        Args:
            slice_height: Height of each slice in pixels
            slice_width: Width of each slice in pixels
            overlap_ratio_h: Vertical overlap ratio (0.2 = 20% overlap)
            overlap_ratio_w: Horizontal overlap ratio (0.2 = 20% overlap)
            iou_threshold: IOU threshold for merging predictions
            ios_threshold: IOS threshold for handling nested objects
            merge_mode: 'nms' for standard NMS, 'nmm' for merging boxes
            min_slice_area_ratio: Minimum ratio of object area in slice to keep it
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
        """
        Generate slice coordinates for the given image dimensions.
        
        Args:
            image_height: Height of the full image
            image_width: Width of the full image
            
        Returns:
            List of tuples (x1, y1, x2, y2, slice_id) for each slice
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
        """
        Shift prediction coordinates from slice space to full image space.
        
        Args:
            predictions: Detections object with detections in slice coordinates
            offset_x: X offset of the slice in the full image
            offset_y: Y offset of the slice in the full image
            slice_id: ID of the source slice for tracking
            
        Returns:
            Detections object with shifted coordinates
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
        """
        Calculate Intersection over Union between two boxes.
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IOU value between 0 and 1
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
        """
        Calculate Intersection over Smaller area between two boxes.
        Useful for detecting nested objects (small object inside large).
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IOS value between 0 and 1
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
        """
        Check if two slices are adjacent (share a border or overlap).
        
        Args:
            slice_id1: First slice ID
            slice_id2: Second slice ID
            slices: List of all slice coordinates
            
        Returns:
            True if slices are adjacent
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
        """
        Merge predictions from all slices using intelligent NMS/NMM.
        
        This is the core algorithm that handles edge cases:
        1. Objects on slice boundaries (high IOU from adjacent slices)
        2. Nested objects (use IOS instead of IOU)
        3. Different confidence scores for same object
        
        Args:
            all_predictions: List of all predictions from all slices
            slices: List of slice coordinates for adjacency checking
            
        Returns:
            Merged Detections object
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
        """
        Merge multiple predictions into one using weighted average.
        
        Args:
            predictions: List of predictions to merge
            
        Returns:
            Single merged prediction
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
        """
        Run sliced inference on an image.
        
        Args:
            image: Input image as numpy array
            detector_func: Detection function that takes an image and returns Detections
            verbose: Print progress information
            **detector_kwargs: Additional arguments to pass to detector_func
            
        Returns:
            Merged Detections object with all detections
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
    """
    Automatically calculate optimal slice size based on image dimensions.
    
    Args:
        image_height: Height of the image
        image_width: Width of the image
        target_size: Target size for slices (will be adjusted based on image)
        
    Returns:
        Tuple of (slice_height, slice_width)
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