"""
Smoother module for temporal detection smoothing and gap filling.

This module provides functions to reduce jitter and fill detection gaps
using temporal context from the Buffer module.
"""

from typing import Optional, List, Dict, Any
import numpy as np
from pixelflow.buffer import Buffer
from pixelflow.results import Results, Prediction


def smooth(buffer: Buffer, 
          max_gap_frames: int = 2,
          algorithm: str = 'exponential',
          alpha: float = 0.7,
          iou_threshold: float = 0.5) -> Results:
    """
    Smooth detection results using temporal context from buffer.
    
    This function analyzes past, current, and future detection results
    to reduce jitter and fill temporary detection gaps.
    
    Args:
        buffer: Buffer containing temporal context
        max_gap_frames: Maximum consecutive frames to fill gaps (default: 2)
        algorithm: Smoothing algorithm - 'exponential', 'linear', or 'none' (default: 'exponential')
        alpha: Weight for exponential smoothing, higher = less smoothing (default: 0.7)
        iou_threshold: IoU threshold for matching objects across frames (default: 0.5)
    
    Returns:
        Smoothed Results object for the middle frame
    """
    # Get temporal context from buffer
    context = buffer.get_temporal_context()
    if context is None:
        # Buffer not full yet, return middle results as-is
        middle_idx = buffer.buffer_size // 2
        if middle_idx < len(buffer.results_buffer):
            return buffer.results_buffer[middle_idx]
        return Results()
    
    # Extract temporal data
    past_results = context['past_results']
    current_results = context['current_results']
    future_results = context['future_results']
    
    # Handle empty current results - check for gap filling opportunity
    if current_results is None or len(current_results) == 0:
        current_results = _fill_gaps(past_results, future_results, max_gap_frames, iou_threshold)
        if current_results is None:
            return Results()
    
    # Apply smoothing algorithm
    if algorithm == 'exponential':
        return _exponential_smooth(past_results, current_results, future_results, alpha, iou_threshold)
    elif algorithm == 'linear':
        return _linear_smooth(past_results, current_results, future_results, iou_threshold)
    else:  # 'none' or unknown
        return current_results


def _exponential_smooth(past_results: List[Results], 
                        current_results: Results,
                        future_results: List[Results],
                        alpha: float,
                        iou_threshold: float) -> Results:
    """
    Apply exponential smoothing to detection bounding boxes and confidence scores.
    
    Args:
        past_results: List of past Results objects
        current_results: Current Results to smooth
        future_results: List of future Results objects  
        alpha: Weight factor (0-1), higher values give more weight to current frame
        iou_threshold: IoU threshold for object matching
        
    Returns:
        Smoothed Results object
    """
    smoothed = Results()
    
    # Process each prediction in current frame
    for current_pred in current_results.predictions:
        # Find matching predictions in temporal neighbors
        matches = _find_temporal_matches(current_pred, past_results, future_results, iou_threshold)
        
        if not matches['past'] and not matches['future']:
            # No matches found, keep original prediction
            smoothed.add_prediction(current_pred)
            continue
            
        # Calculate smoothed bbox and confidence
        smoothed_pred = _smooth_prediction(current_pred, matches, alpha)
        smoothed.add_prediction(smoothed_pred)
    
    # Check for gap filling opportunities - add missing detections
    missing_preds = _find_missing_detections(current_results, past_results, future_results, iou_threshold)
    for pred in missing_preds:
        smoothed.add_prediction(pred)
    
    return smoothed


def _linear_smooth(past_results: List[Results],
                  current_results: Results, 
                  future_results: List[Results],
                  iou_threshold: float) -> Results:
    """
    Apply linear interpolation smoothing to detections.
    
    Args:
        past_results: List of past Results objects
        current_results: Current Results to smooth
        future_results: List of future Results objects
        iou_threshold: IoU threshold for object matching
        
    Returns:
        Smoothed Results object
    """
    smoothed = Results()
    
    for current_pred in current_results.predictions:
        matches = _find_temporal_matches(current_pred, past_results, future_results, iou_threshold)
        
        if not matches['past'] and not matches['future']:
            smoothed.add_prediction(current_pred)
            continue
        
        # Linear interpolation for bbox
        smoothed_pred = _linear_interpolate_prediction(current_pred, matches)
        smoothed.add_prediction(smoothed_pred)
    
    # Add missing detections
    missing_preds = _find_missing_detections(current_results, past_results, future_results, iou_threshold)
    for pred in missing_preds:
        smoothed.add_prediction(pred)
    
    return smoothed


def _find_temporal_matches(target_pred: Prediction,
                          past_results: List[Results],
                          future_results: List[Results],
                          iou_threshold: float) -> Dict[str, List[Prediction]]:
    """
    Find matching predictions in past and future frames.
    
    Args:
        target_pred: Prediction to find matches for
        past_results: List of past Results objects
        future_results: List of future Results objects
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with 'past' and 'future' lists of matching predictions
    """
    matches = {'past': [], 'future': []}
    
    # Search in past frames (reverse order, nearest first)
    for results in reversed(past_results):
        if results is None or len(results) == 0:
            continue
        match = _find_best_match(target_pred, results, iou_threshold)
        if match:
            matches['past'].append(match)
    
    # Search in future frames
    for results in future_results:
        if results is None or len(results) == 0:
            continue
        match = _find_best_match(target_pred, results, iou_threshold)
        if match:
            matches['future'].append(match)
    
    return matches


def _find_best_match(target_pred: Prediction, 
                     results: Results,
                     iou_threshold: float) -> Optional[Prediction]:
    """
    Find the best matching prediction in a Results object.
    
    Args:
        target_pred: Prediction to match
        results: Results object to search in
        iou_threshold: Minimum IoU for a match
        
    Returns:
        Best matching Prediction or None
    """
    if not target_pred.bbox:
        return None
    
    best_match = None
    best_iou = iou_threshold
    
    for pred in results.predictions:
        # Check class compatibility
        if pred.class_id != target_pred.class_id:
            continue
            
        # Calculate IoU
        if pred.bbox:
            iou = _calculate_iou(target_pred.bbox, pred.bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = pred
    
    return best_match


def _calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union for two bounding boxes.
    
    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def _smooth_prediction(current_pred: Prediction,
                      matches: Dict[str, List[Prediction]], 
                      alpha: float) -> Prediction:
    """
    Apply exponential smoothing to a single prediction.
    
    Args:
        current_pred: Current prediction to smooth
        matches: Dictionary of matching predictions from past/future
        alpha: Smoothing weight (higher = less smoothing)
        
    Returns:
        Smoothed Prediction object
    """
    # Create new prediction with smoothed values
    smoothed_pred = Prediction(
        class_id=current_pred.class_id,
        class_name=current_pred.class_name,
        tracker_id=current_pred.tracker_id,
        data=current_pred.data
    )
    
    # Collect all bboxes and confidences for smoothing
    all_bboxes = []
    all_confidences = []
    
    # Add past matches
    for pred in matches['past']:
        if pred.bbox:
            all_bboxes.append(pred.bbox)
        if pred.confidence is not None:
            all_confidences.append(pred.confidence)
    
    # Add current
    if current_pred.bbox:
        all_bboxes.append(current_pred.bbox)
    if current_pred.confidence is not None:
        all_confidences.append(current_pred.confidence)
    
    # Add future matches
    for pred in matches['future']:
        if pred.bbox:
            all_bboxes.append(pred.bbox)
        if pred.confidence is not None:
            all_confidences.append(pred.confidence)
    
    # Apply exponential weighted average to bbox
    if all_bboxes:
        if len(all_bboxes) == 1:
            smoothed_pred.bbox = current_pred.bbox
        else:
            # Calculate weights (exponential decay from center)
            n = len(all_bboxes)
            center_idx = n // 2
            weights = []
            for i in range(n):
                distance = abs(i - center_idx)
                weight = alpha ** distance
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Apply weighted average to bbox coordinates
            bbox_array = np.array(all_bboxes)
            weights_array = np.array(weights)
            smoothed_bbox = np.average(bbox_array, axis=0, weights=weights_array)
            smoothed_pred.bbox = smoothed_bbox.tolist()
    
    # Apply exponential weighted average to confidence
    if all_confidences:
        if len(all_confidences) == 1:
            smoothed_pred.confidence = current_pred.confidence
        else:
            n = len(all_confidences)
            center_idx = n // 2
            weights = []
            for i in range(n):
                distance = abs(i - center_idx)
                weight = alpha ** distance
                weights.append(weight)
            
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            smoothed_confidence = np.average(all_confidences, weights=weights)
            smoothed_pred.confidence = float(smoothed_confidence)
    
    # Keep masks, segments, keypoints unchanged for now
    smoothed_pred.masks = current_pred.masks
    smoothed_pred.segments = current_pred.segments
    smoothed_pred.keypoints = current_pred.keypoints
    smoothed_pred.zones = current_pred.zones
    smoothed_pred.zone_names = current_pred.zone_names
    
    return smoothed_pred


def _linear_interpolate_prediction(current_pred: Prediction,
                                  matches: Dict[str, List[Prediction]]) -> Prediction:
    """
    Apply linear interpolation to smooth a prediction.
    
    Args:
        current_pred: Current prediction to smooth
        matches: Dictionary of matching predictions
        
    Returns:
        Smoothed Prediction object
    """
    smoothed_pred = Prediction(
        class_id=current_pred.class_id,
        class_name=current_pred.class_name,
        tracker_id=current_pred.tracker_id,
        data=current_pred.data
    )
    
    # For linear interpolation, use only nearest past and future
    past_pred = matches['past'][0] if matches['past'] else None
    future_pred = matches['future'][0] if matches['future'] else None
    
    # Interpolate bbox
    if current_pred.bbox:
        if past_pred and past_pred.bbox and future_pred and future_pred.bbox:
            # Full interpolation with past and future
            bbox_array = np.array([past_pred.bbox, current_pred.bbox, future_pred.bbox])
            smoothed_bbox = np.mean(bbox_array, axis=0)
            smoothed_pred.bbox = smoothed_bbox.tolist()
        elif past_pred and past_pred.bbox:
            # Average with past only
            bbox_array = np.array([past_pred.bbox, current_pred.bbox])
            smoothed_bbox = np.mean(bbox_array, axis=0)
            smoothed_pred.bbox = smoothed_bbox.tolist()
        elif future_pred and future_pred.bbox:
            # Average with future only
            bbox_array = np.array([current_pred.bbox, future_pred.bbox])
            smoothed_bbox = np.mean(bbox_array, axis=0)
            smoothed_pred.bbox = smoothed_bbox.tolist()
        else:
            smoothed_pred.bbox = current_pred.bbox
    
    # Interpolate confidence
    confidences = []
    if past_pred and past_pred.confidence is not None:
        confidences.append(past_pred.confidence)
    if current_pred.confidence is not None:
        confidences.append(current_pred.confidence)
    if future_pred and future_pred.confidence is not None:
        confidences.append(future_pred.confidence)
    
    if confidences:
        smoothed_pred.confidence = float(np.mean(confidences))
    
    # Keep other attributes unchanged
    smoothed_pred.masks = current_pred.masks
    smoothed_pred.segments = current_pred.segments
    smoothed_pred.keypoints = current_pred.keypoints
    smoothed_pred.zones = current_pred.zones
    smoothed_pred.zone_names = current_pred.zone_names
    
    return smoothed_pred


def _find_missing_detections(current_results: Results,
                            past_results: List[Results],
                            future_results: List[Results],
                            iou_threshold: float) -> List[Prediction]:
    """
    Find detections that appear in past and future but missing in current frame.
    
    Args:
        current_results: Current frame results
        past_results: List of past Results objects
        future_results: List of future Results objects
        iou_threshold: IoU threshold for matching
        
    Returns:
        List of interpolated predictions for missing detections
    """
    missing_predictions = []
    
    # Get the nearest past and future frames with detections
    past_frame = None
    for results in reversed(past_results):
        if results and len(results) > 0:
            past_frame = results
            break
    
    future_frame = None
    for results in future_results:
        if results and len(results) > 0:
            future_frame = results
            break
    
    if not past_frame or not future_frame:
        return missing_predictions
    
    # Check each detection in past frame
    for past_pred in past_frame.predictions:
        # Find match in future
        future_match = _find_best_match(past_pred, future_frame, iou_threshold * 0.8)  # Slightly lower threshold
        if not future_match:
            continue
        
        # Check if this detection exists in current frame
        current_match = _find_best_match(past_pred, current_results, iou_threshold)
        if current_match:
            continue  # Already exists in current frame
        
        # Create interpolated prediction for missing detection
        interpolated = _interpolate_missing(past_pred, future_match)
        if interpolated:
            missing_predictions.append(interpolated)
    
    return missing_predictions


def _interpolate_missing(past_pred: Prediction, future_pred: Prediction) -> Optional[Prediction]:
    """
    Create an interpolated prediction for a missing detection.
    
    Args:
        past_pred: Prediction from past frame
        future_pred: Prediction from future frame
        
    Returns:
        Interpolated Prediction or None
    """
    if not past_pred.bbox or not future_pred.bbox:
        return None
    
    # Linear interpolation for bbox
    bbox_past = np.array(past_pred.bbox)
    bbox_future = np.array(future_pred.bbox)
    interpolated_bbox = ((bbox_past + bbox_future) / 2).tolist()
    
    # Average confidence with decay factor
    confidence = None
    if past_pred.confidence is not None and future_pred.confidence is not None:
        # Apply decay factor for interpolated detection
        confidence = float((past_pred.confidence + future_pred.confidence) / 2 * 0.8)
    
    # Create interpolated prediction
    interpolated = Prediction(
        bbox=interpolated_bbox,
        confidence=confidence,
        class_id=past_pred.class_id,
        class_name=past_pred.class_name,
        data={'interpolated': True}  # Mark as interpolated
    )
    
    return interpolated


def _fill_gaps(past_results: List[Results],
               future_results: List[Results],
               max_gap_frames: int,
               iou_threshold: float) -> Optional[Results]:
    """
    Fill detection gaps when current frame has no detections.
    
    Args:
        past_results: List of past Results objects
        future_results: List of future Results objects
        max_gap_frames: Maximum gap size to fill
        iou_threshold: IoU threshold for matching
        
    Returns:
        Results object with filled detections or None
    """
    # Check if gap is within allowed range
    gap_size = 0
    
    # Count consecutive empty frames in past
    for results in reversed(past_results):
        if results is None or len(results) == 0:
            gap_size += 1
        else:
            break
    
    # Current frame is empty (that's why we're here)
    gap_size += 1
    
    # Count consecutive empty frames in future
    for results in future_results:
        if results is None or len(results) == 0:
            gap_size += 1
        else:
            break
    
    if gap_size > max_gap_frames:
        return None  # Gap too large to fill
    
    # Find nearest non-empty past and future frames
    past_frame = None
    for results in reversed(past_results):
        if results and len(results) > 0:
            past_frame = results
            break
    
    future_frame = None
    for results in future_results:
        if results and len(results) > 0:
            future_frame = results
            break
    
    if not past_frame or not future_frame:
        return None
    
    # Create filled results
    filled = Results()
    
    # Match detections between past and future
    for past_pred in past_frame.predictions:
        future_match = _find_best_match(past_pred, future_frame, iou_threshold * 0.8)
        if future_match:
            # Interpolate between past and future
            interpolated = _interpolate_missing(past_pred, future_match)
            if interpolated:
                filled.add_prediction(interpolated)
    
    return filled if len(filled) > 0 else None