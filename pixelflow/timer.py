"""
Time tracking module for PixelFlow.

Provides unified time tracking for object detection, zone analysis, and line crossing events.
Works with both frame-based and clock-based timing for maximum flexibility.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import numpy as np


class TimeTracker:
    """
    Unified time tracker that works with frames or system clock.
    Automatically tracks overall time, zone time, and line crossing time.
    
    This class provides time tracking for:
    - Total time since first detection of each object
    - Time spent in specific zones (if zone info is present in detections)
    - Time since line crossings (if line crossing info is present)
    
    The tracker automatically switches between frame-based and clock-based timing
    based on whether an FPS value is provided during initialization.
    
    Examples:
        # Clock-based timing (default)
        time_tracker = TimeTracker()
        
        # Frame-based timing
        time_tracker = TimeTracker(fps=30.0)
        
        # Usage in processing loop
        times = time_tracker.update(detections)
        total_times = times['total']  # Array of total times for each detection
        zone_times = times['zones']   # Dict of zone_id -> time arrays
    """
    
    def __init__(self, fps: Optional[float] = None):
        """
        Initialize the TimeTracker.
        
        Args:
            fps: If provided, uses frame-based timing. If None, uses system clock timing.
                Frame-based timing is more accurate for video processing where frame rate
                is known and consistent.
        """
        self.fps = fps
        self.frame_count = 0
        
        # Overall time tracking - when each tracker was first seen
        self.first_seen: Dict[int, Union[datetime, int]] = {}
        
        # Zone-specific time tracking - when tracker entered each zone
        self.zone_times: Dict[tuple, Union[datetime, int]] = {}  # (tracker_id, zone_id) -> start_time
        
        # Line crossing time tracking - when tracker crossed each line
        self.line_cross_times: Dict[tuple, Union[datetime, int]] = {}  # (tracker_id, line_id, direction) -> time
        
        # Set timing mode
        if fps is None:
            self.use_clock = True
            self.get_current_time = datetime.now
        else:
            self.use_clock = False
            self.get_current_time = lambda: self.frame_count
    
    def update(self, detections):
        """
        Update time tracking for all detections.
        
        This method processes all detections and updates each Detection object with:
        - first_seen_time: When the object was first detected
        - total_time: Total time since first detection
        
        Also maintains internal tracking for zones and line crossings for detailed analysis.
        
        Args:
            detections: Detections object containing detection results
            
        Returns:
            Updated Detections object (modified in-place)
        """
        if not self.use_clock:
            self.frame_count += 1
        
        current_time = self.get_current_time()
        
        # Process each detection and update with time information
        for detection in detections:
            if detection.tracker_id is None:
                detection.total_time = 0.0
                continue
            
            tid = detection.tracker_id
            
            # Track total time since first detection
            if tid not in self.first_seen:
                self.first_seen[tid] = current_time
                detection.first_seen_time = current_time
            
            # Update detection with time info
            total_duration = self._calculate_duration(self.first_seen[tid], current_time)
            detection.total_time = total_duration
            
            # Track zone times (if detection has zone information) for internal analysis
            if hasattr(detection, 'zones') and detection.zones:
                for zone_id in detection.zones:
                    zone_key = (tid, zone_id)
                    # Record when tracker entered this zone
                    if zone_key not in self.zone_times:
                        self.zone_times[zone_key] = current_time
            
            # Track line crossing times (if detection has crossing information) for internal analysis
            if hasattr(detection, 'line_crossings') and detection.line_crossings:
                for crossing in detection.line_crossings:
                    line_id = crossing['line_id']
                    direction = crossing['direction']
                    crossing_key = (tid, line_id, direction)
                    # Record crossing time if this is a new crossing
                    if crossing_key not in self.line_cross_times:
                        self.line_cross_times[crossing_key] = current_time
        
        return detections
    
    def get_detailed_stats(self, detections) -> Dict[str, Any]:
        """
        Get detailed timing statistics in dictionary format for advanced analysis.
        
        This method provides the same detailed information that was previously returned
        by update(), useful for advanced users who need zone-specific times and 
        line crossing analysis.
        
        Args:
            detections: Detections object to analyze
            
        Returns:
            Dictionary containing:
                - 'total': np.ndarray of total time for each detection
                - 'zones': Dict[zone_id, List[float]] - time arrays for objects in each zone
                - 'since_line_crossing': Dict[(line_id, direction), List[float]] - time arrays
        """
        current_time = self.get_current_time()
        
        # Initialize results
        total_times = []
        zone_times_dict = {}
        line_times_dict = {}
        
        for detection in detections:
            if detection.tracker_id is None:
                total_times.append(0.0)
                continue
            
            tid = detection.tracker_id
            total_times.append(detection.total_time)
            
            # Calculate zone times
            if hasattr(detection, 'zones') and detection.zones:
                for zone_id in detection.zones:
                    zone_key = (tid, zone_id)
                    if zone_key in self.zone_times:
                        if zone_id not in zone_times_dict:
                            zone_times_dict[zone_id] = []
                        zone_duration = self._calculate_duration(self.zone_times[zone_key], current_time)
                        zone_times_dict[zone_id].append(zone_duration)
            
            # Calculate line crossing times
            if hasattr(detection, 'line_crossings') and detection.line_crossings:
                for crossing in detection.line_crossings:
                    line_id = crossing['line_id']
                    direction = crossing['direction']
                    crossing_key = (tid, line_id, direction)
                    
                    if crossing_key in self.line_cross_times:
                        result_key = (line_id, direction)
                        if result_key not in line_times_dict:
                            line_times_dict[result_key] = []
                        crossing_duration = self._calculate_duration(
                            self.line_cross_times[crossing_key], current_time
                        )
                        line_times_dict[result_key].append(crossing_duration)
        
        return {
            'total': np.array(total_times),
            'zones': zone_times_dict,
            'since_line_crossing': line_times_dict
        }
    
    def _calculate_duration(self, start_time: Union[datetime, int], current_time: Union[datetime, int]) -> float:
        """
        Calculate duration between start and current time based on timing mode.
        
        Args:
            start_time: Start time (datetime or frame number)
            current_time: Current time (datetime or frame number)
            
        Returns:
            Duration in seconds
        """
        if self.use_clock:
            return (current_time - start_time).total_seconds()
        else:
            return (current_time - start_time) / self.fps
    
    def reset(self, tracker_ids: Optional[List[int]] = None):
        """
        Reset tracking for specific tracker IDs or all trackers.
        
        This is useful when objects leave the scene or when you want to
        restart timing for specific objects.
        
        Args:
            tracker_ids: List of tracker IDs to reset. If None, resets all trackers.
        """
        if tracker_ids is None:
            # Reset everything
            self.first_seen.clear()
            self.zone_times.clear()
            self.line_cross_times.clear()
        else:
            # Reset specific trackers
            for tid in tracker_ids:
                # Remove from first_seen
                self.first_seen.pop(tid, None)
                
                # Remove from zone_times (keys are tuples with tracker_id as first element)
                keys_to_remove = [k for k in self.zone_times.keys() if k[0] == tid]
                for key in keys_to_remove:
                    del self.zone_times[key]
                
                # Remove from line_cross_times (keys are tuples with tracker_id as first element)
                keys_to_remove = [k for k in self.line_cross_times.keys() if k[0] == tid]
                for key in keys_to_remove:
                    del self.line_cross_times[key]
    
    def get_tracker_stats(self, tracker_id: int) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific tracker.
        
        Args:
            tracker_id: ID of the tracker to get stats for
            
        Returns:
            Dictionary containing tracker statistics
        """
        stats = {
            'tracker_id': tracker_id,
            'first_seen': self.first_seen.get(tracker_id),
            'total_time': 0.0,
            'zones': {},
            'line_crossings': {}
        }
        
        # Calculate total time if tracker exists
        if tracker_id in self.first_seen:
            current_time = self.get_current_time()
            stats['total_time'] = self._calculate_duration(
                self.first_seen[tracker_id], current_time
            )
        
        # Get zone times
        for (tid, zone_id), start_time in self.zone_times.items():
            if tid == tracker_id:
                current_time = self.get_current_time()
                zone_duration = self._calculate_duration(start_time, current_time)
                stats['zones'][zone_id] = zone_duration
        
        # Get line crossing times
        for (tid, line_id, direction), cross_time in self.line_cross_times.items():
            if tid == tracker_id:
                current_time = self.get_current_time()
                crossing_duration = self._calculate_duration(cross_time, current_time)
                key = f"{line_id}_{direction}"
                stats['line_crossings'][key] = crossing_duration
        
        return stats
    
    def cleanup_inactive_trackers(self, active_tracker_ids: List[int]):
        """
        Remove tracking data for trackers that are no longer active.
        
        This helps prevent memory leaks in long-running applications where
        tracker IDs might be reused or objects permanently leave the scene.
        
        Args:
            active_tracker_ids: List of currently active tracker IDs
        """
        active_set = set(active_tracker_ids)
        
        # Clean first_seen
        inactive_trackers = [tid for tid in self.first_seen.keys() if tid not in active_set]
        for tid in inactive_trackers:
            del self.first_seen[tid]
        
        # Clean zone_times
        keys_to_remove = [k for k in self.zone_times.keys() if k[0] not in active_set]
        for key in keys_to_remove:
            del self.zone_times[key]
        
        # Clean line_cross_times
        keys_to_remove = [k for k in self.line_cross_times.keys() if k[0] not in active_set]
        for key in keys_to_remove:
            del self.line_cross_times[key]