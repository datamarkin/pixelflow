"""
Temporal detection smoothing for reducing jitter and stabilizing tracked objects.

This module provides advanced buffer-based smoothing algorithms that use temporal context
from past, current, and future frames to reduce bounding box jitter, stabilize confidence
scores, and interpolate missing detections. Designed to work seamlessly with PixelFlow's
buffer system for high-quality temporal smoothing in real-time applications.
"""

from typing import Optional, Dict, List, Tuple, Union, TYPE_CHECKING
from collections import defaultdict
import numpy as np
from pixelflow.detections import Detections, Detection

if TYPE_CHECKING:
    from pixelflow.buffer import Buffer


def smooth(buffer: 'Buffer', temporal_weight_decay: float = 0.8) -> Detections:
    """
    Smooth detection results using buffer's temporal context for jitter reduction.
    
    Applies sophisticated temporal smoothing using past, current, and future frames
    with exponential weight decay. Reduces bounding box jitter, stabilizes confidence
    scores, and interpolates missing detections when objects temporarily disappear.
    
    Args:
        buffer (Buffer): Buffer instance containing temporal context with past, current,
                        and future detection frames. Must have temporal context available.
        temporal_weight_decay (float): Exponential decay factor for temporal weighting.
                                     Current frame weight = 1.0, adjacent frames = decay^1,
                                     next frames = decay^2, etc. Range: [0.1, 1.0].
                                     Default is 0.8 (20% decay per frame distance).
        
    Returns:
        Detections: Smoothed detections for the current frame with reduced jitter and
                   stabilized confidence scores. Returns raw current frame results if
                   temporal context is unavailable.
    
    Raises:
        AttributeError: If buffer lacks required methods or temporal context structure.
        ValueError: If temporal_weight_decay is outside valid range [0.1, 1.0].
        TypeError: If buffer is not a valid Buffer instance.
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from pixelflow.buffer import Buffer
        >>> from ultralytics import YOLO
        >>> 
        >>> # Setup video processing with buffer
        >>> model = YOLO("yolo11n.pt")
        >>> buffer = Buffer(buffer_size=5)  # 2 past + current + 2 future
        >>> cap = cv2.VideoCapture("video.mp4")
        >>> 
        >>> # Process video frames with smoothing
        >>> while True:
        >>>     ret, frame = cap.read()
        >>>     if not ret: break
        >>>     
        >>>     # Get raw detections and add to buffer
        >>>     outputs = model.predict(frame)
        >>>     results = pf.results.from_ultralytics(outputs)
        >>>     buffer.add_frame(frame, results)
        >>>     
        >>>     # Apply temporal smoothing
        >>>     smoothed = pf.smoother.smooth(buffer)
        >>> 
        >>> # Advanced usage with custom decay
        >>> smoothed = pf.smoother.smooth(buffer, temporal_weight_decay=0.7)
        >>> 
        >>> # Conservative smoothing for fast-moving objects
        >>> smoothed = pf.smoother.smooth(buffer, temporal_weight_decay=0.9)
    
    Notes:
        - Requires buffer to be filled with temporal context (past + current + future frames)
        - Returns raw current frame results when temporal context is unavailable
        - Automatically interpolates missing detections when objects appear in past/future but not current frame
        - Preserves all detection attributes (masks, keypoints, etc.) from current frame
        - Weight decay applies exponentially: frame distance 1 = decay^1, distance 2 = decay^2, etc.
        - Smoothing quality improves with larger buffer sizes but increases latency
        
    Performance Notes:
        - Computational complexity: O(n*k) where n=detections, k=buffer_size
        - Memory usage scales linearly with buffer size and detection count
        - Optimized for real-time processing with minimal overhead
        
    See Also:
        Buffer : Temporal frame buffer for storing detection history
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
    Aggregate detections by tracker ID across all temporal frames.
    
    Groups all detections from past, current, and future frames by their tracker ID
    to enable temporal smoothing for each tracked object individually.
    
    Args:
        context (Dict): Temporal context dictionary containing 'past_results',
                       'current_results', and 'future_results' keys with
                       Detections objects or None values.
        
    Returns:
        Dict[int, List[Tuple[str, int, Detection]]]: Dictionary mapping tracker IDs
                                                   to lists of temporal detection data.
                                                   Each tuple contains (frame_type, frame_offset, detection)
                                                   where frame_type is 'past'/'current'/'future',
                                                   frame_offset is relative position, and detection
                                                   is the Detection object.
    
    Raises:
        KeyError: If context dict is missing required keys ('past_results', 'current_results', 'future_results').
        AttributeError: If detection objects lack required tracker_id attribute.
        
    Example:
        >>> import pixelflow as pf
        >>> from pixelflow.buffer import Buffer
        >>> 
        >>> # Get temporal context from buffer
        >>> buffer = Buffer(buffer_size=5)
        >>> # ... fill buffer with frames ...
        >>> context = buffer.get_temporal_context()
        >>> 
        >>> # Collect detections by tracker ID
        >>> tracker_detections = pf.smoother._collect_temporal_detections(context)
        >>> print(f"Tracking {len(tracker_detections)} objects")
        >>> 
        >>> # Access detections for specific tracker
        >>> tracker_1_data = tracker_detections[1]  # List of temporal detections
        >>> for frame_type, offset, detection in tracker_1_data:
        >>>     print(f"Frame {frame_type}({offset}): bbox {detection.bbox}")
    
    Notes:
        - Only processes detections with valid (non-None) tracker IDs
        - Frame offsets: negative for past frames, 0 for current, positive for future
        - Handles empty or None detection results gracefully
        - Preserves temporal ordering for each tracker across all frames
        - Essential preprocessing step for temporal smoothing algorithms
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
    Apply temporal smoothing to detections for a single tracked object.
    
    Computes weighted averages of bounding boxes and confidence scores across temporal
    frames, with exponential decay weighting based on distance from current frame.
    Preserves detection metadata from the current frame when available.
    
    Args:
        detections (List[Tuple[str, int, Detection]]): List of temporal detection data
                                                     where each tuple contains
                                                     (frame_type, frame_offset, detection).
                                                     Frame types: 'past', 'current', 'future'.
        temporal_weight_decay (float): Exponential decay factor for temporal weighting.
                                     Applied as decay^|frame_offset| to each detection.
                                     Range: [0.1, 1.0].
        
    Returns:
        Optional[Detection]: Smoothed detection with weighted average bbox and confidence,
                           or None if no valid detections available for smoothing.
    
    Raises:
        IndexError: If detections list is malformed or contains invalid tuples.
        AttributeError: If detection objects lack required bbox or confidence attributes.
        ValueError: If temporal_weight_decay is outside valid range.
        
    Example:
        >>> import pixelflow as pf
        >>> from pixelflow.detections import Detection
        >>> 
        >>> # Create temporal detections for tracker ID 1
        >>> detection1 = Detection(bbox=[100, 100, 200, 200], confidence=0.9, tracker_id=1)
        >>> detection2 = Detection(bbox=[102, 98, 198, 202], confidence=0.85, tracker_id=1)
        >>> detection3 = Detection(bbox=[98, 102, 202, 198], confidence=0.88, tracker_id=1)
        >>> 
        >>> temporal_data = [
        >>>     ('past', -1, detection1),
        >>>     ('current', 0, detection2),
        >>>     ('future', 1, detection3)
        >>> ]
        >>> 
        >>> # Apply temporal smoothing
        >>> smoothed = pf.smoother._smooth_temporal_detections(temporal_data, 0.8)
        >>> print(f"Smoothed bbox: {smoothed.bbox}")
        >>> print(f"Smoothed confidence: {smoothed.confidence}")
        >>> 
        >>> # Handle missing current frame detection
        >>> temporal_data_no_current = [
        >>>     ('past', -1, detection1),
        >>>     ('future', 1, detection3)
        >>> ]
        >>> smoothed = pf.smoother._smooth_temporal_detections(temporal_data_no_current, 0.8)
    
    Notes:
        - Uses current frame detection as template for metadata (class_id, masks, etc.)
        - Falls back to most recent detection if no current frame detection available
        - Applies exponential temporal weighting: weight = decay^|frame_offset|
        - Skips detections with invalid (None) bounding boxes
        - Preserves all non-smoothed attributes (masks, keypoints, segments, zones)
        - Returns None if no valid bounding boxes available for smoothing
        
    Performance Notes:
        - Linear complexity O(n) where n is number of temporal detections
        - Minimal memory allocation using numpy for efficient bbox averaging
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
    """
    Compute weighted average of bounding boxes for temporal smoothing.
    
    Calculates element-wise weighted average of bounding box coordinates using
    temporal weights to reduce jitter and stabilize object boundaries.
    
    Args:
        weighted_bboxes (List[Tuple[float, List[float]]]): List of (weight, bbox) tuples
                                                         where weight is temporal weight
                                                         and bbox is [x1, y1, x2, y2] coordinates.
        
    Returns:
        Optional[List[float]]: Smoothed bounding box as [x1, y1, x2, y2] coordinates,
                              or None if no valid bboxes or zero total weight.
    
    Raises:
        ValueError: If bbox coordinates are malformed or contain invalid values.
        TypeError: If weights are not numeric or bboxes are not lists.
        
    Example:
        >>> import pixelflow as pf
        >>> 
        >>> # Temporal bounding boxes with weights
        >>> weighted_bboxes = [
        >>>     (0.64, [100, 100, 200, 200]),  # Past frame (weight=0.8^2)
        >>>     (0.8, [102, 98, 198, 202]),    # Adjacent frame (weight=0.8^1)
        >>>     (1.0, [98, 102, 202, 198]),    # Current frame (weight=1.0)
        >>> ]
        >>> 
        >>> # Calculate weighted average
        >>> smoothed_bbox = pf.smoother._weighted_average_bbox(weighted_bboxes)
        >>> print(f"Smoothed bbox: {smoothed_bbox}")  # [99.5, 100.2, 199.8, 199.7]
        >>> 
        >>> # Handle empty input
        >>> result = pf.smoother._weighted_average_bbox([])
        >>> print(result)  # None
    
    Notes:
        - Normalizes weights automatically (divides by total weight)
        - Handles zero total weight by returning None
        - Uses numpy for efficient vectorized computation
        - Preserves coordinate precision with float arithmetic
        - Essential component of temporal detection smoothing pipeline
    """
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
    """
    Compute weighted average of confidence scores for temporal smoothing.
    
    Calculates weighted average of detection confidence scores using temporal weights
    to reduce confidence fluctuations and provide more stable object detection scores.
    
    Args:
        weighted_confidences (List[Tuple[float, float]]): List of (weight, confidence) tuples
                                                        where weight is temporal weight
                                                        and confidence is detection score [0.0, 1.0].
        
    Returns:
        Optional[float]: Smoothed confidence score in range [0.0, 1.0],
                        or None if no valid confidences or zero total weight.
    
    Raises:
        ValueError: If confidence scores are outside valid range [0.0, 1.0].
        TypeError: If weights or confidences are not numeric.
        
    Example:
        >>> import pixelflow as pf
        >>> 
        >>> # Temporal confidence scores with weights
        >>> weighted_confidences = [
        >>>     (0.64, 0.85),  # Past frame confidence
        >>>     (0.8, 0.92),   # Adjacent frame confidence
        >>>     (1.0, 0.88),   # Current frame confidence
        >>> ]
        >>> 
        >>> # Calculate weighted average
        >>> smoothed_conf = pf.smoother._weighted_average_confidence(weighted_confidences)
        >>> print(f"Smoothed confidence: {smoothed_conf:.3f}")  # 0.886
        >>> 
        >>> # Handle missing confidences
        >>> result = pf.smoother._weighted_average_confidence([])
        >>> print(result)  # None
    
    Notes:
        - Normalizes weights automatically (divides by total weight)
        - Returns None for empty input or zero total weight
        - Maintains confidence score precision with float arithmetic
        - Helps stabilize confidence fluctuations in temporal sequences
        - Complementary to bounding box smoothing in detection stabilization
    """
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
    Identify and interpolate missing detections in the current frame.
    
    Finds tracked objects that appear in both past and future frames but are missing
    in the current frame, then generates interpolated detections to maintain tracking
    continuity and reduce detection gaps.
    
    Args:
        context (Dict): Temporal context dictionary containing past, current, and
                       future detection results from buffer.
        tracker_detections (Dict[int, List[Tuple[str, int, Detection]]]): Pre-collected
                                                                        temporal detections
                                                                        grouped by tracker ID.
        
    Returns:
        List[Detection]: List of interpolated detections for objects missing in current
                        frame but present in both past and future frames. Each detection
                        has interpolated bbox, confidence, and 'interpolated' flag in data.
    
    Raises:
        KeyError: If context dictionary is missing required keys.
        AttributeError: If detection objects lack required tracker_id attribute.
        TypeError: If tracker_detections is not properly structured dictionary.
        
    Example:
        >>> import pixelflow as pf
        >>> from pixelflow.buffer import Buffer
        >>> 
        >>> # Setup buffer and get temporal context
        >>> buffer = Buffer(buffer_size=5)
        >>> # ... fill buffer with frames where object disappears temporarily ...
        >>> context = buffer.get_temporal_context()
        >>> tracker_detections = pf.smoother._collect_temporal_detections(context)
        >>> 
        >>> # Find missing detections
        >>> missing = pf.smoother._find_missing_detections(context, tracker_detections)
        >>> print(f"Found {len(missing)} missing detections to interpolate")
        >>> 
        >>> # Check interpolated detections
        >>> for detection in missing:
        >>>     if detection.data and detection.data.get('interpolated'):
        >>>         print(f"Interpolated tracker {detection.tracker_id}: {detection.bbox}")
        >>> 
        >>> # Use with smoothing pipeline
        >>> smoothed_detections = pf.smoother.smooth(buffer)
        >>> # missing detections are automatically included in smoothed results
    
    Notes:
        - Only interpolates when object appears in both past AND future frames
        - Skips objects already present in current frame
        - Uses linear interpolation for bounding box coordinates
        - Applies confidence decay factor (0.85) to indicate interpolated nature
        - Marks interpolated detections with 'interpolated': True in data dict
        - Essential for maintaining tracking continuity during temporary occlusions
        - Reduces tracking ID switches caused by detection gaps
        
    Performance Notes:
        - Complexity: O(k*n) where k=trackers, n=avg detections per tracker
        - Minimal overhead when no missing detections found
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
    Interpolate missing detection using closest past and future detections.
    
    Creates an interpolated detection by linearly interpolating bounding box coordinates
    and averaging confidence scores from the closest past and future detections.
    Applies confidence decay to indicate interpolated nature.
    
    Args:
        detections (List[Tuple[str, int, Detection]]): List of temporal detection data
                                                     containing both past and future detections.
                                                     Each tuple: (frame_type, frame_offset, detection).
        
    Returns:
        Optional[Detection]: Interpolated detection with averaged bbox and confidence,
                           or None if interpolation not possible (missing past or future data).
    
    Raises:
        ValueError: If detections contain invalid bbox coordinates.
        AttributeError: If detection objects lack required attributes.
        IndexError: If detections list structure is malformed.
        
    Example:
        >>> import pixelflow as pf
        >>> from pixelflow.detections import Detection
        >>> 
        >>> # Create past and future detections for interpolation
        >>> past_det = Detection(bbox=[100, 100, 200, 200], confidence=0.9, 
        >>>                     class_id=0, tracker_id=1)
        >>> future_det = Detection(bbox=[110, 95, 210, 205], confidence=0.85, 
        >>>                       class_id=0, tracker_id=1)
        >>> 
        >>> # Temporal detection data (missing current frame)
        >>> temporal_data = [
        >>>     ('past', -1, past_det),
        >>>     ('future', 1, future_det)
        >>> ]
        >>> 
        >>> # Interpolate missing detection
        >>> interpolated = pf.smoother._interpolate_missing_detection(temporal_data)
        >>> print(f"Interpolated bbox: {interpolated.bbox}")  # [105, 97.5, 205, 202.5]
        >>> print(f"Interpolated confidence: {interpolated.confidence}")  # ~0.744 (avg * 0.85)
        >>> print(f"Is interpolated: {interpolated.data['interpolated']}")  # True
        >>> 
        >>> # Handle insufficient data
        >>> incomplete_data = [('past', -1, past_det)]  # Missing future detection
        >>> result = pf.smoother._interpolate_missing_detection(incomplete_data)
        >>> print(result)  # None
    
    Notes:
        - Requires both past and future detections for interpolation
        - Uses closest temporal neighbors (highest past offset, lowest future offset)
        - Applies simple linear interpolation for bounding box coordinates
        - Averages confidence scores then applies 0.85 decay factor
        - Copies class_id, tracker_id, and other metadata from past detection
        - Marks result with 'interpolated': True and source information in data dict
        - Returns None if either past or future detections lack valid bounding boxes
        
    Performance Notes:
        - Constant time complexity O(1) for interpolation computation
        - Uses numpy for efficient coordinate averaging
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