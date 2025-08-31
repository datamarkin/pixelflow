"""
Smoother module for temporal detection smoothing.

This module provides simple, tracker-based smoothing to reduce bbox jitter
and confidence fluctuations for tracked objects.
"""

from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
from collections import defaultdict, deque
import numpy as np
from pixelflow.detections import Detections, Detection

if TYPE_CHECKING:
    from pixelflow.buffer import Buffer


class DetectionsSmoother:
    """
    Simple tracker-based smoother for PixelFlow detections.
    
    Maintains a history of detections for each tracker_id and provides
    smoothed predictions based on simple averaging.
    """
    
    def __init__(self, length: int = 5):
        """
        Initialize smoother.
        
        Args:
            length: Maximum number of frames to consider for smoothing (default: 5)
        """
        self.length = length
        self.tracks = defaultdict(lambda: deque(maxlen=length))
    
    def smooth(self, results: Results) -> Results:
        """
        Smooth detection results using tracker-based history.
        
        Args:
            results: Results object containing predictions with tracker_id
            
        Returns:
            Smoothed Results object
        """
        if not results or len(results) == 0:
            return results
        
        # Update tracks with current detections
        self._update_tracks(results)
        
        # Generate smoothed results
        return self._get_smoothed_results()
    
    def _update_tracks(self, results: Results):
        """Update track history with current detections."""
        current_tracker_ids = set()
        
        # Add current detections to their respective tracks
        for pred in results.predictions:
            if pred.tracker_id is not None:
                current_tracker_ids.add(pred.tracker_id)
                self.tracks[pred.tracker_id].append(pred)
        
        # Add None for missing tracker IDs (temporarily lost tracks)
        for tracker_id in list(self.tracks.keys()):
            if tracker_id not in current_tracker_ids:
                self.tracks[tracker_id].append(None)
        
        # Clean up tracks that are completely empty
        for tracker_id in list(self.tracks.keys()):
            if all(d is None for d in self.tracks[tracker_id]):
                del self.tracks[tracker_id]
    
    def _get_smoothed_results(self) -> Results:
        """Generate smoothed results from track history."""
        smoothed_results = Results()
        
        for tracker_id in self.tracks:
            smoothed_pred = self._get_smoothed_prediction(tracker_id)
            if smoothed_pred is not None:
                smoothed_results.add_prediction(smoothed_pred)
        
        return smoothed_results
    
    def _get_smoothed_prediction(self, tracker_id: int) -> Optional[Prediction]:
        """
        Generate smoothed prediction for a specific tracker.
        
        Args:
            tracker_id: Tracker ID to smooth
            
        Returns:
            Smoothed Prediction or None if no valid data
        """
        track = self.tracks.get(tracker_id, None)
        if track is None:
            return None
        
        # Filter out None values (missing detections)
        valid_preds = [pred for pred in track if pred is not None]
        if len(valid_preds) == 0:
            return None
        
        # Use the most recent prediction as template
        template_pred = valid_preds[-1]
        
        # Smooth bbox using simple averaging
        smoothed_bbox = None
        if template_pred.bbox is not None:
            bboxes = [pred.bbox for pred in valid_preds if pred.bbox is not None]
            if bboxes:
                smoothed_bbox = np.mean(bboxes, axis=0).tolist()
        
        # Smooth confidence using simple averaging
        smoothed_confidence = None
        if template_pred.confidence is not None:
            confidences = [pred.confidence for pred in valid_preds 
                         if pred.confidence is not None]
            if confidences:
                smoothed_confidence = float(np.mean(confidences))
        
        # Create smoothed prediction
        return Prediction(
            bbox=smoothed_bbox,
            confidence=smoothed_confidence,
            class_id=template_pred.class_id,
            class_name=template_pred.class_name,
            tracker_id=tracker_id,
            masks=template_pred.masks,
            segments=template_pred.segments,
            keypoints=template_pred.keypoints,
            zones=template_pred.zones,
            zone_names=template_pred.zone_names,
            data=template_pred.data
        )


class BufferDetectionsSmoother:
    """
    Enhanced buffer-based smoother that uses temporal context (past + current + future).
    
    This smoother leverages the Buffer's temporal context to smooth detections
    using both historical and future information, providing superior smoothing
    quality compared to the basic DetectionsSmoother.
    """
    
    def __init__(self, temporal_weight_decay: float = 0.8):
        """
        Initialize buffer-based smoother.
        
        Args:
            temporal_weight_decay: Weight decay factor for temporal frames (default: 0.8)
                                 Current frame = 1.0, ±1 frame = 0.8, ±2 frame = 0.64, etc.
        """
        self.temporal_weight_decay = temporal_weight_decay
    
    def smooth(self, buffer: 'Buffer') -> Results:
        """
        Smooth detection results using buffer's temporal context.
        
        Args:
            buffer: Buffer containing temporal context with past/current/future frames
            
        Returns:
            Smoothed Results object for the current (middle) frame
        """
        context = buffer.get_temporal_context()
        if context is None:
            # Buffer not full yet, return middle results as-is
            middle_idx = buffer.buffer_size // 2
            if middle_idx < len(buffer.results_buffer):
                return buffer.results_buffer[middle_idx]
            return Results()
        
        # Group all predictions by tracker_id across all temporal frames
        tracker_predictions = self._collect_temporal_predictions(context)
        
        # Generate smoothed predictions for each tracker
        smoothed_results = Results()
        for tracker_id, predictions in tracker_predictions.items():
            smoothed_pred = self._smooth_temporal_predictions(predictions)
            if smoothed_pred is not None:
                smoothed_results.add_prediction(smoothed_pred)
        
        # Check for missing detections that can be interpolated
        missing_predictions = self._find_missing_detections(context, tracker_predictions)
        for interpolated_pred in missing_predictions:
            smoothed_results.add_prediction(interpolated_pred)
        
        return smoothed_results
    
    def _collect_temporal_predictions(self, context: Dict) -> Dict[int, List[Tuple[str, int, Prediction]]]:
        """
        Collect all predictions by tracker_id across past/current/future frames.
        
        Args:
            context: Temporal context from buffer
            
        Returns:
            Dictionary mapping tracker_id to list of (frame_type, frame_offset, prediction) tuples
        """
        tracker_predictions = defaultdict(list)
        
        # Add past predictions (negative offsets)
        for i, results in enumerate(context['past_results']):
            if results and len(results) > 0:
                frame_offset = i - len(context['past_results'])  # -2, -1
                for pred in results.predictions:
                    if pred.tracker_id is not None:
                        tracker_predictions[pred.tracker_id].append(('past', frame_offset, pred))
        
        # Add current prediction (offset 0)
        current_results = context['current_results']
        if current_results and len(current_results) > 0:
            for pred in current_results.predictions:
                if pred.tracker_id is not None:
                    tracker_predictions[pred.tracker_id].append(('current', 0, pred))
        
        # Add future predictions (positive offsets)  
        for i, results in enumerate(context['future_results']):
            if results and len(results) > 0:
                frame_offset = i + 1  # +1, +2
                for pred in results.predictions:
                    if pred.tracker_id is not None:
                        tracker_predictions[pred.tracker_id].append(('future', frame_offset, pred))
        
        return tracker_predictions
    
    def _smooth_temporal_predictions(self, predictions: List[Tuple[str, int, Prediction]]) -> Optional[Prediction]:
        """
        Smooth predictions for a single tracker using temporal weighting.
        
        Args:
            predictions: List of (frame_type, frame_offset, prediction) tuples
            
        Returns:
            Smoothed Prediction or None if no valid data
        """
        if not predictions:
            return None
        
        # Find current prediction to use as template
        current_pred = None
        for frame_type, offset, pred in predictions:
            if frame_type == 'current':
                current_pred = pred
                break
        
        # If no current prediction, use the most recent one as template
        if current_pred is None:
            current_pred = predictions[-1][2]
        
        # Collect valid predictions with weights
        weighted_bboxes = []
        weighted_confidences = []
        
        for frame_type, frame_offset, pred in predictions:
            if pred.bbox is None:
                continue
                
            # Calculate temporal weight based on distance from current frame
            weight = self.temporal_weight_decay ** abs(frame_offset)
            
            # Add bbox and confidence with weight
            weighted_bboxes.append((weight, pred.bbox))
            if pred.confidence is not None:
                weighted_confidences.append((weight, pred.confidence))
        
        # Calculate weighted averages
        smoothed_bbox = self._weighted_average_bbox(weighted_bboxes)
        smoothed_confidence = self._weighted_average_confidence(weighted_confidences)
        
        if smoothed_bbox is None:
            return None
        
        # Create smoothed prediction
        return Prediction(
            bbox=smoothed_bbox,
            confidence=smoothed_confidence,
            class_id=current_pred.class_id,
            class_name=current_pred.class_name,
            tracker_id=current_pred.tracker_id,
            masks=current_pred.masks,
            segments=current_pred.segments,
            keypoints=current_pred.keypoints,
            zones=current_pred.zones,
            zone_names=current_pred.zone_names,
            data=current_pred.data
        )
    
    def _weighted_average_bbox(self, weighted_bboxes: List[Tuple[float, List[float]]]) -> Optional[List[float]]:
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
    
    def _weighted_average_confidence(self, weighted_confidences: List[Tuple[float, float]]) -> Optional[float]:
        """Calculate weighted average of confidence scores."""
        if not weighted_confidences:
            return None
        
        total_weight = sum(weight for weight, _ in weighted_confidences)
        if total_weight == 0:
            return None
        
        # Calculate weighted average
        weighted_sum = sum(weight * conf for weight, conf in weighted_confidences)
        return float(weighted_sum / total_weight)
    
    def _find_missing_detections(self, context: Dict, tracker_predictions: Dict[int, List[Tuple[str, int, Prediction]]]) -> List[Prediction]:
        """
        Find detections that appear in past/future but missing in current frame.
        
        Args:
            context: Temporal context from buffer
            tracker_predictions: Already collected tracker predictions
            
        Returns:
            List of interpolated predictions for missing detections
        """
        current_tracker_ids = set()
        current_results = context['current_results']
        
        # Get tracker IDs present in current frame
        if current_results:
            current_tracker_ids = {pred.tracker_id for pred in current_results.predictions 
                                 if pred.tracker_id is not None}
        
        missing_predictions = []
        
        # Check each tracker in temporal predictions
        for tracker_id, predictions in tracker_predictions.items():
            if tracker_id in current_tracker_ids:
                continue  # Already present in current frame
            
            # Check if tracker appears in both past and future
            has_past = any(frame_type == 'past' for frame_type, _, _ in predictions)
            has_future = any(frame_type == 'future' for frame_type, _, _ in predictions)
            
            if has_past and has_future:
                # Interpolate missing detection
                interpolated = self._interpolate_missing_detection(predictions)
                if interpolated is not None:
                    missing_predictions.append(interpolated)
        
        return missing_predictions
    
    def _interpolate_missing_detection(self, predictions: List[Tuple[str, int, Prediction]]) -> Optional[Prediction]:
        """
        Interpolate a missing detection using past and future predictions.
        
        Args:
            predictions: List of (frame_type, frame_offset, prediction) tuples
            
        Returns:
            Interpolated Prediction or None if interpolation not possible
        """
        # Separate past and future predictions
        past_preds = [(offset, pred) for frame_type, offset, pred in predictions if frame_type == 'past']
        future_preds = [(offset, pred) for frame_type, offset, pred in predictions if frame_type == 'future']
        
        if not past_preds or not future_preds:
            return None
        
        # Get the closest past and future predictions
        closest_past = max(past_preds, key=lambda x: x[0])  # Highest negative offset (closest to 0)
        closest_future = min(future_preds, key=lambda x: x[0])  # Lowest positive offset (closest to 0)
        
        past_pred = closest_past[1]
        future_pred = closest_future[1]
        
        if not past_pred.bbox or not future_pred.bbox:
            return None
        
        # Linear interpolation for bbox (simple mid-point)
        past_bbox = np.array(past_pred.bbox)
        future_bbox = np.array(future_pred.bbox)
        interpolated_bbox = ((past_bbox + future_bbox) / 2).tolist()
        
        # Average confidence with decay factor for interpolated detections
        interpolated_confidence = None
        if past_pred.confidence is not None and future_pred.confidence is not None:
            avg_confidence = (past_pred.confidence + future_pred.confidence) / 2
            # Apply decay factor for interpolated detections to show lower confidence
            interpolated_confidence = float(avg_confidence * 0.85)
        
        # Create interpolated prediction
        return Prediction(
            bbox=interpolated_bbox,
            confidence=interpolated_confidence,
            class_id=past_pred.class_id,
            class_name=past_pred.class_name,
            tracker_id=past_pred.tracker_id,
            masks=past_pred.masks,
            segments=past_pred.segments,
            keypoints=past_pred.keypoints,
            zones=past_pred.zones,
            zone_names=past_pred.zone_names,
            data={'interpolated': True, 'interpolation_source': 'buffer_smoother'}
        )


# Legacy function for buffer-based smoothing (enhanced)
def smooth(buffer, **kwargs) -> Results:
    """
    Legacy buffer-based smoothing function.
    
    Now uses the enhanced BufferDetectionsSmoother for superior smoothing
    with temporal context (past + current + future frames).
    
    Args:
        buffer: Buffer containing temporal context
        **kwargs: Deprecated parameters (ignored for backward compatibility)
                 - max_gap_frames: No longer used
                 - algorithm: No longer used
                 - alpha: No longer used
                 - iou_threshold: No longer used
        
    Returns:
        Smoothed Results from middle frame using temporal context
    """
    # Use enhanced buffer-based smoother
    buffer_smoother = BufferDetectionsSmoother(temporal_weight_decay=0.8)
    return buffer_smoother.smooth(buffer)
