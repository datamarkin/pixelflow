"""
Smoother module for temporal detection smoothing.

This module provides simple, tracker-based smoothing to reduce bbox jitter
and confidence fluctuations for tracked objects.
"""

from typing import Optional, Dict
from collections import defaultdict, deque
import numpy as np
from pixelflow.results import Results, Prediction


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


# Legacy function for buffer-based smoothing (simplified)
def smooth(buffer, **kwargs) -> Results:
    """
    Legacy buffer-based smoothing function.
    
    Args:
        buffer: Buffer containing temporal context
        **kwargs: Ignored for backward compatibility
        
    Returns:
        Smoothed Results from middle frame
    """
    context = buffer.get_temporal_context()
    if context is None:
        middle_idx = buffer.buffer_size // 2
        if middle_idx < len(buffer.results_buffer):
            return buffer.results_buffer[middle_idx]
        return Results()
    
    current_results = context['current_results']
    if current_results is None or len(current_results) == 0:
        return Results()
    
    # Use simple smoother on current results
    smoother = DetectionsSmoother(length=3)
    return smoother.smooth(current_results)
