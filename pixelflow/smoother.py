"""
Smoother module for temporal detection smoothing.

This module provides buffer-based smoothing to reduce bbox jitter
and confidence fluctuations for tracked objects using temporal context.
"""

from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
from collections import defaultdict
import numpy as np
from pixelflow.detections import Detections, Detection

if TYPE_CHECKING:
    from pixelflow.buffer import Buffer


def smooth(buffer: 'Buffer', temporal_weight_decay: float = 0.8) -> Detections:
    """
    Smooth detection results using buffer's temporal context.
    
    Uses past, current, and future frames for superior smoothing quality
    with temporal weighting and interpolation for missing detections.
    
    Args:
        buffer: Buffer containing temporal context with past/current/future frames
        temporal_weight_decay: Weight decay factor for temporal frames (default: 0.8)
                             Current frame = 1.0, ±1 frame = 0.8, ±2 frame = 0.64, etc.
        
    Returns:
        Smoothed Detections object for the current (middle) frame, or raw results if buffer not ready
    """
    # Always return the current middle frame results
    middle_idx = buffer.buffer_size // 2
    if middle_idx < len(buffer.results_buffer):
        raw_results = buffer.results_buffer[middle_idx]
    else:
        raw_results = Detections()
    
    # Only apply smoothing if buffer is full, otherwise return raw results
    context = buffer.get_temporal_context()
    if context is None:
        return raw_results
    
    # Group all detections by tracker_id across all temporal frames
    tracker_detections = _collect_temporal_detections(context)
    
    # Generate smoothed detections for each tracker
    smoothed_detections = Detections()
    for tracker_id, detections in tracker_detections.items():
        smoothed_detection = _smooth_temporal_detections(detections, temporal_weight_decay)
        if smoothed_detection is not None:
            smoothed_detections.add_detection(smoothed_detection)
    
    # Check for missing detections that can be interpolated
    missing_detections = _find_missing_detections(context, tracker_detections)
    for interpolated_detection in missing_detections:
        smoothed_detections.add_detection(interpolated_detection)
    
    return smoothed_detections


def _collect_temporal_detections(context: Dict) -> Dict[int, List[Tuple[str, int, Detection]]]:
    """
    Collect all detections by tracker_id across past/current/future frames.
    
    Args:
        context: Temporal context from buffer
        
    Returns:
        Dictionary mapping tracker_id to list of (frame_type, frame_offset, detection) tuples
    """
    tracker_detections = defaultdict(list)
    
    # Add past detections (negative offsets)
    for i, detections in enumerate(context['past_results']):
        if detections and len(detections) > 0:
            frame_offset = i - len(context['past_results'])  # -2, -1
            for detection in detections.detections:
                if detection.tracker_id is not None:
                    tracker_detections[detection.tracker_id].append(('past', frame_offset, detection))
    
    # Add current detection (offset 0)
    current_detections = context['current_results']
    if current_detections and len(current_detections) > 0:
        for detection in current_detections.detections:
            if detection.tracker_id is not None:
                tracker_detections[detection.tracker_id].append(('current', 0, detection))
    
    # Add future detections (positive offsets)  
    for i, detections in enumerate(context['future_results']):
        if detections and len(detections) > 0:
            frame_offset = i + 1  # +1, +2
            for detection in detections.detections:
                if detection.tracker_id is not None:
                    tracker_detections[detection.tracker_id].append(('future', frame_offset, detection))
    
    return tracker_detections


def _smooth_temporal_detections(detections: List[Tuple[str, int, Detection]], temporal_weight_decay: float) -> Optional[Detection]:
    """
    Smooth detections for a single tracker using temporal weighting.
    
    Args:
        detections: List of (frame_type, frame_offset, detection) tuples
        temporal_weight_decay: Weight decay factor for temporal frames
        
    Returns:
        Smoothed Detection or None if no valid data
    """
    if not detections:
        return None
    
    # Find current detection to use as template
    current_detection = None
    for frame_type, offset, detection in detections:
        if frame_type == 'current':
            current_detection = detection
            break
    
    # If no current detection, use the most recent one as template
    if current_detection is None:
        current_detection = detections[-1][2]
    
    # Collect valid detections with weights
    weighted_bboxes = []
    weighted_confidences = []
    
    for frame_type, frame_offset, detection in detections:
        if detection.bbox is None:
            continue
            
        # Calculate temporal weight based on distance from current frame
        weight = temporal_weight_decay ** abs(frame_offset)
        
        # Add bbox and confidence with weight
        weighted_bboxes.append((weight, detection.bbox))
        if detection.confidence is not None:
            weighted_confidences.append((weight, detection.confidence))
    
    # Calculate weighted averages
    smoothed_bbox = _weighted_average_bbox(weighted_bboxes)
    smoothed_confidence = _weighted_average_confidence(weighted_confidences)
    
    if smoothed_bbox is None:
        return None
    
    # Create smoothed detection
    return Detection(
        bbox=smoothed_bbox,
        confidence=smoothed_confidence,
        class_id=current_detection.class_id,
        class_name=current_detection.class_name,
        tracker_id=current_detection.tracker_id,
        masks=current_detection.masks,
        segments=current_detection.segments,
        keypoints=current_detection.keypoints,
        zones=current_detection.zones,
        zone_names=current_detection.zone_names,
        data=current_detection.data
    )


def _weighted_average_bbox(weighted_bboxes: List[Tuple[float, List[float]]]) -> Optional[List[float]]:
    """Calculate weighted average of bounding boxes."""
    if not weighted_bboxes:
        return None
    
    total_weight = sum(weight for weight, _ in weighted_bboxes)
    if total_weight == 0:
        return None
    
    # Calculate weighted sum
    weighted_sum = np.zeros(4)
    for weight, bbox in weighted_bboxes:
        weighted_sum += np.array(bbox) * weight
    
    # Return weighted average
    return (weighted_sum / total_weight).tolist()


def _weighted_average_confidence(weighted_confidences: List[Tuple[float, float]]) -> Optional[float]:
    """Calculate weighted average of confidence scores."""
    if not weighted_confidences:
        return None
    
    total_weight = sum(weight for weight, _ in weighted_confidences)
    if total_weight == 0:
        return None
    
    # Calculate weighted average
    weighted_sum = sum(weight * conf for weight, conf in weighted_confidences)
    return float(weighted_sum / total_weight)


def _find_missing_detections(context: Dict, tracker_detections: Dict[int, List[Tuple[str, int, Detection]]]) -> List[Detection]:
    """
    Find detections that appear in past/future but missing in current frame.
    
    Args:
        context: Temporal context from buffer
        tracker_detections: Already collected tracker detections
        
    Returns:
        List of interpolated detections for missing detections
    """
    current_tracker_ids = set()
    current_detections = context['current_results']
    
    # Get tracker IDs present in current frame
    if current_detections:
        current_tracker_ids = {detection.tracker_id for detection in current_detections.detections 
                             if detection.tracker_id is not None}
    
    missing_detections = []
    
    # Check each tracker in temporal detections
    for tracker_id, detections in tracker_detections.items():
        if tracker_id in current_tracker_ids:
            continue  # Already present in current frame
        
        # Check if tracker appears in both past and future
        has_past = any(frame_type == 'past' for frame_type, _, _ in detections)
        has_future = any(frame_type == 'future' for frame_type, _, _ in detections)
        
        if has_past and has_future:
            # Interpolate missing detection
            interpolated = _interpolate_missing_detection(detections)
            if interpolated is not None:
                missing_detections.append(interpolated)
    
    return missing_detections


def _interpolate_missing_detection(detections: List[Tuple[str, int, Detection]]) -> Optional[Detection]:
    """
    Interpolate a missing detection using past and future detections.
    
    Args:
        detections: List of (frame_type, frame_offset, detection) tuples
        
    Returns:
        Interpolated Detection or None if interpolation not possible
    """
    # Separate past and future detections
    past_detections = [(offset, detection) for frame_type, offset, detection in detections if frame_type == 'past']
    future_detections = [(offset, detection) for frame_type, offset, detection in detections if frame_type == 'future']
    
    if not past_detections or not future_detections:
        return None
    
    # Get the closest past and future detections
    closest_past = max(past_detections, key=lambda x: x[0])  # Highest negative offset (closest to 0)
    closest_future = min(future_detections, key=lambda x: x[0])  # Lowest positive offset (closest to 0)
    
    past_detection = closest_past[1]
    future_detection = closest_future[1]
    
    if not past_detection.bbox or not future_detection.bbox:
        return None
    
    # Linear interpolation for bbox (simple mid-point)
    past_bbox = np.array(past_detection.bbox)
    future_bbox = np.array(future_detection.bbox)
    interpolated_bbox = ((past_bbox + future_bbox) / 2).tolist()
    
    # Average confidence with decay factor for interpolated detections
    interpolated_confidence = None
    if past_detection.confidence is not None and future_detection.confidence is not None:
        avg_confidence = (past_detection.confidence + future_detection.confidence) / 2
        # Apply decay factor for interpolated detections to show lower confidence
        interpolated_confidence = float(avg_confidence * 0.85)
    
    # Create interpolated detection
    return Detection(
        bbox=interpolated_bbox,
        confidence=interpolated_confidence,
        class_id=past_detection.class_id,
        class_name=past_detection.class_name,
        tracker_id=past_detection.tracker_id,
        masks=past_detection.masks,
        segments=past_detection.segments,
        keypoints=past_detection.keypoints,
        zones=past_detection.zones,
        zone_names=past_detection.zone_names,
        data={'interpolated': True, 'interpolation_source': 'buffer_smoother'}
    )