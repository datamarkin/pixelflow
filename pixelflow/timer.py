"""
Time tracking utilities for computer vision applications.

Provides comprehensive time tracking for object detection workflows including total
detection time, zone occupancy duration, and line crossing analysis. Supports both
frame-based timing for video processing and real-time clock-based timing for live streams.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import numpy as np


class TimeTracker:
    """
    Unified time tracker for object detection and tracking workflows.
    
    Provides comprehensive time tracking capabilities including total detection time,
    zone occupancy duration, and line crossing analysis. Automatically adapts between
    frame-based timing (for video processing) and real-time clock-based timing (for live streams).
    
    The tracker maintains internal state for all tracked objects and provides both
    simple integration through detection object modification and detailed analytics
    through comprehensive statistics methods.
    
    Args:
        fps (Optional[float]): Frame rate for frame-based timing. If None, uses system
                              clock timing. Frame-based timing provides more accurate
                              results for video processing with consistent frame rates.
                              Range: > 0.0. Default is None (clock-based timing).
    
    Attributes:
        fps (Optional[float]): Frame rate used for timing calculations.
        frame_count (int): Current frame number (frame-based timing only).
        use_clock (bool): Whether using system clock (True) or frame-based (False) timing.
        first_seen (Dict[int, Union[datetime, int]]): First detection time for each tracker ID.
        zone_times (Dict[tuple, Union[datetime, int]]): Zone entry times keyed by (tracker_id, zone_id).
        line_cross_times (Dict[tuple, Union[datetime, int]]): Line crossing times keyed by (tracker_id, line_id, direction).
    
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Basic setup with clock-based timing
        >>> model = YOLO("yolo11n.pt")
        >>> time_tracker = pf.timer.TimeTracker()
        >>> 
        >>> # Process video frame
        >>> frame = cv2.imread("frame.jpg")
        >>> outputs = model.track(frame)  # Enable tracking
        >>> results = pf.results.from_ultralytics(outputs)
        >>> 
        >>> # Update timing (modifies detections in-place)
        >>> time_tracker.update(results)
        >>> print(f"Detection times: {[d.total_time for d in results]}")
        >>> 
        >>> # Frame-based timing for video processing
        >>> time_tracker = pf.timer.TimeTracker(fps=30.0)
        >>> cap = cv2.VideoCapture("video.mp4")
        >>> while True:
        >>>     ret, frame = cap.read()
        >>>     if not ret: break
        >>>     outputs = model.track(frame)
        >>>     results = pf.results.from_ultralytics(outputs)
        >>>     time_tracker.update(results)  # Automatically tracks frame-based timing
        >>> 
        >>> # Get detailed statistics for analysis
        >>> stats = time_tracker.get_detailed_stats(results)
        >>> total_times = stats['total']  # numpy array of total times
        >>> zone_times = stats['zones']   # dict of zone-specific times
        >>> 
        >>> # Reset specific trackers when objects leave scene
        >>> inactive_ids = [1, 3, 5]
        >>> time_tracker.reset(inactive_ids)
    
    Notes:
        - Detection objects are modified in-place with timing information
        - Frame-based timing requires consistent FPS throughout processing
        - Zone and line crossing data requires corresponding attributes in detection objects
        - Memory usage scales with number of unique tracker IDs over time
        - Use cleanup_inactive_trackers() periodically for long-running applications
        
    Performance Notes:
        - O(n) time complexity where n is number of detections per frame
        - Memory usage grows with unique tracker count and zone/line complexity
        - Frame-based timing is more CPU efficient than datetime calculations
    """
    
    def __init__(self, fps: Optional[float] = None):
        """
        Initialize the TimeTracker with specified timing mode.
        
        Sets up timing infrastructure and determines whether to use frame-based or
        clock-based timing based on the fps parameter. Frame-based timing provides
        more consistent results for video processing workflows.
        
        Args:
            fps (Optional[float]): Frame rate for frame-based timing. If provided,
                                  timing calculations use frame counts divided by FPS.
                                  If None, uses system clock timestamps.
                                  Range: > 0.0. Default is None.
        
        Raises:
            ValueError: If fps is provided but is <= 0.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Clock-based timing for live streams
            >>> tracker = pf.timer.TimeTracker()
            >>> 
            >>> # Frame-based timing for video processing
            >>> tracker = pf.timer.TimeTracker(fps=30.0)
            >>> 
            >>> # High frame rate video
            >>> tracker = pf.timer.TimeTracker(fps=120.0)
        
        Notes:
            - Frame-based timing assumes consistent frame rate throughout processing
            - Clock-based timing adapts to variable processing speeds automatically
            - All internal tracking dictionaries are initialized as empty
        """
        if fps is not None and fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")
            
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
        Update time tracking for all detections in current frame.
        
        Processes each detection and updates the Detection objects with timing information.
        Modifies detection objects in-place by adding total_time and first_seen_time attributes.
        Also maintains internal state for zone occupancy and line crossing analysis.
        
        Args:
            detections: Collection of detection objects with tracker_id attributes.
                       Each detection should have a tracker_id (int or None) for tracking.
                       Optional zone and line_crossing attributes enable advanced timing.
        
        Returns:
            Same detections object with updated timing attributes (modified in-place).
            Each detection gains:
            - total_time (float): Seconds since first detection of this tracker
            - first_seen_time (Union[datetime, int]): When tracker was first detected
        
        Raises:
            AttributeError: If detections object doesn't support iteration.
        
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> # Setup tracking
            >>> model = YOLO("yolo11n.pt")
            >>> tracker = pf.timer.TimeTracker(fps=30.0)
            >>> 
            >>> # Process video frames
            >>> cap = cv2.VideoCapture("video.mp4")
            >>> ret, frame = cap.read()
            >>> outputs = model.track(frame)  # Enable tracking for consistent IDs
            >>> results = pf.results.from_ultralytics(outputs)
            >>> 
            >>> # Update timing (modifies results in-place)
            >>> tracker.update(results)
            >>> for detection in results:
            >>>     print(f"Object {detection.tracker_id}: {detection.total_time:.2f}s")
            >>> 
            >>> # Advanced usage with zones (requires zone-enabled detections)
            >>> # Zone detection must be added separately via zone filtering
            >>> results_with_zones = pf.zones.filter_detections(results, zones)
            >>> tracker.update(results_with_zones)  # Now tracks zone times internally
            >>> 
            >>> # Access detailed stats after update
            >>> stats = tracker.get_detailed_stats(results_with_zones)
            >>> print(f"Zone occupancy times: {stats['zones']}")
        
        Notes:
            - Detections without tracker_id get total_time set to 0.0
            - Frame count increments automatically for frame-based timing
            - Zone and line crossing tracking requires corresponding attributes in detections
            - First detection of each tracker_id initializes timing baseline
            - Internal state persists across multiple update() calls for continuous tracking
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
        Calculate comprehensive timing statistics for current detections.
        
        Analyzes current detections and returns detailed timing information including
        total detection times, zone occupancy durations, and time since line crossings.
        Provides structured data for advanced analytics and reporting workflows.
        
        Args:
            detections: Collection of detection objects to analyze. Should contain
                       detections with tracker_id attributes. Zone and line crossing
                       analysis requires corresponding attributes in detection objects.
        
        Returns:
            Dict[str, Any]: Comprehensive timing statistics containing:
                - 'total' (np.ndarray): Total time in seconds for each detection
                - 'zones' (Dict[Any, List[float]]): Zone occupancy times grouped by zone_id
                - 'since_line_crossing' (Dict[Tuple, List[float]]): Time since crossing
                  grouped by (line_id, direction) tuples
        
        Raises:
            AttributeError: If detections object doesn't support iteration.
        
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> import numpy as np
            >>> 
            >>> # Setup with zone and line tracking
            >>> model = YOLO("yolo11n.pt")
            >>> tracker = pf.timer.TimeTracker(fps=25.0)
            >>> 
            >>> # Process frame with zone detection
            >>> frame = cv2.imread("frame.jpg")
            >>> outputs = model.track(frame)
            >>> results = pf.results.from_ultralytics(outputs)
            >>> tracker.update(results)  # Update timing first
            >>> 
            >>> # Get comprehensive statistics
            >>> stats = tracker.get_detailed_stats(results)
            >>> total_times = stats['total']  # numpy array of times
            >>> print(f"Average detection time: {np.mean(total_times):.2f}s")
            >>> 
            >>> # Analyze zone occupancy (if zones present)
            >>> if stats['zones']:
            >>>     for zone_id, times in stats['zones'].items():
            >>>         avg_time = np.mean(times)
            >>>         print(f"Zone {zone_id} average occupancy: {avg_time:.2f}s")
            >>> 
            >>> # Line crossing analysis (if crossings present)
            >>> for (line_id, direction), times in stats['since_line_crossing'].items():
            >>>     recent_crossing = np.min(times)  # Most recent crossing
            >>>     print(f"Line {line_id} ({direction}): {recent_crossing:.2f}s ago")
            >>> 
            >>> # Export for external analysis
            >>> import json
            >>> exportable_stats = {
            >>>     'total_times': stats['total'].tolist(),
            >>>     'zone_analysis': {str(k): v for k, v in stats['zones'].items()}
            >>> }
            >>> with open('timing_report.json', 'w') as f:
            >>>     json.dump(exportable_stats, f)
        
        Notes:
            - Call update() before get_detailed_stats() for current timing data
            - Zone and line crossing stats require corresponding detection attributes
            - Total times array corresponds 1:1 with input detections order
            - Empty lists in zone/crossing stats indicate no current activity
            - Statistics reflect current frame state, not historical maximums
        
        Performance Notes:
            - O(n) complexity where n is number of detections
            - Memory usage proportional to number of active zones and crossings
            - Numpy array creation provides efficient numerical analysis interface
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
        Calculate duration between two time points based on timing mode.
        
        Handles both clock-based timing (using datetime objects) and frame-based timing
        (using frame numbers and FPS conversion). Provides consistent duration calculation
        regardless of timing mode selected during initialization.
        
        Args:
            start_time (Union[datetime, int]): Starting time point. datetime object for
                                              clock-based timing or frame number for
                                              frame-based timing.
            current_time (Union[datetime, int]): Current time point. Must be same type
                                                as start_time and represent later time.
        
        Returns:
            float: Duration in seconds between start_time and current_time.
                  Always positive for valid inputs.
        
        Raises:
            TypeError: If start_time and current_time are not compatible types.
            ZeroDivisionError: If fps is 0 (should not occur with proper initialization).
        
        Example:
            >>> import pixelflow as pf
            >>> from datetime import datetime, timedelta
            >>> 
            >>> # Clock-based timing
            >>> tracker = pf.timer.TimeTracker()
            >>> start = datetime.now()
            >>> end = start + timedelta(seconds=5.5)
            >>> duration = tracker._calculate_duration(start, end)
            >>> print(f"Duration: {duration:.1f}s")  # 5.5s
            >>> 
            >>> # Frame-based timing
            >>> tracker = pf.timer.TimeTracker(fps=30.0)
            >>> start_frame = 100
            >>> end_frame = 190  # 90 frames later
            >>> duration = tracker._calculate_duration(start_frame, end_frame)
            >>> print(f"Duration: {duration:.1f}s")  # 3.0s (90/30)
        
        Notes:
            - Internal method not intended for direct public use
            - Timing mode consistency enforced by initialization logic
            - Clock-based timing uses datetime.total_seconds() for precision
            - Frame-based timing assumes consistent frame rate throughout
        """
        if self.use_clock:
            return (current_time - start_time).total_seconds()
        else:
            return (current_time - start_time) / self.fps
    
    def reset(self, tracker_ids: Optional[List[int]] = None):
        """
        Reset timing data for specified trackers or all tracked objects.
        
        Removes timing history for specific tracker IDs or clears all tracking data.
        Useful for restarting timing when objects leave the scene permanently or for
        periodic cleanup in long-running applications to prevent memory accumulation.
        
        Args:
            tracker_ids (Optional[List[int]]): List of specific tracker IDs to reset.
                                              If None, resets all tracking data.
                                              Empty list resets nothing.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Setup tracker
            >>> tracker = pf.timer.TimeTracker(fps=30.0)
            >>> 
            >>> # After processing several frames...
            >>> # Reset specific objects that left the scene
            >>> inactive_trackers = [1, 5, 8]
            >>> tracker.reset(inactive_trackers)
            >>> 
            >>> # Reset all tracking data (fresh start)
            >>> tracker.reset()
            >>> 
            >>> # Reset nothing (empty list)
            >>> tracker.reset([])
            >>> 
            >>> # Verify reset worked
            >>> stats = tracker.get_tracker_stats(1)  # Should show no data
            >>> print(f"Tracker 1 exists: {stats['first_seen'] is not None}")
        
        Notes:
            - Does not affect frame counter or timing mode settings
            - Removing active trackers may cause timing discontinuities
            - Use cleanup_inactive_trackers() for automatic memory management
            - Reset affects all timing aspects: total time, zones, and line crossings
            - No-op for tracker IDs that don't exist in tracking data
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
        Retrieve comprehensive statistics for a single tracked object.
        
        Provides detailed timing analysis for a specific tracker including total detection
        time, zone occupancy durations, and line crossing history. Useful for individual
        object analysis and debugging tracking behavior.
        
        Args:
            tracker_id (int): Unique identifier of the tracker to analyze.
                             Must be a valid integer tracker ID.
        
        Returns:
            Dict[str, Any]: Comprehensive tracker statistics containing:
                - 'tracker_id' (int): The requested tracker ID
                - 'first_seen' (Union[datetime, int, None]): When tracker first detected
                - 'total_time' (float): Total time in seconds since first detection
                - 'zones' (Dict[Any, float]): Zone occupancy times by zone_id
                - 'line_crossings' (Dict[str, float]): Time since crossings by "line_direction"
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Setup tracker and process some frames
            >>> tracker = pf.timer.TimeTracker(fps=30.0)
            >>> # ... process frames and update tracker ...
            >>> 
            >>> # Get detailed stats for specific object
            >>> stats = tracker.get_tracker_stats(tracker_id=42)
            >>> 
            >>> # Basic timing information
            >>> if stats['first_seen'] is not None:
            >>>     print(f"Tracker 42 detected for {stats['total_time']:.2f}s")
            >>>     print(f"First seen at: {stats['first_seen']}")
            >>> else:
            >>>     print("Tracker 42 not found in tracking data")
            >>> 
            >>> # Zone analysis
            >>> for zone_id, duration in stats['zones'].items():
            >>>     print(f"Zone {zone_id}: {duration:.2f}s occupancy")
            >>> 
            >>> # Line crossing analysis
            >>> for crossing_key, time_since in stats['line_crossings'].items():
            >>>     print(f"Crossing {crossing_key}: {time_since:.2f}s ago")
            >>> 
            >>> # Export individual tracker report
            >>> import json
            >>> report = {
            >>>     'tracker_id': stats['tracker_id'],
            >>>     'total_detection_time': stats['total_time'],
            >>>     'zone_summary': len(stats['zones']),
            >>>     'crossing_events': len(stats['line_crossings'])
            >>> }
            >>> with open(f"tracker_{tracker_id}_report.json", 'w') as f:
            >>>     json.dump(report, f)
        
        Notes:
            - Returns empty/default values for unknown tracker IDs (no errors)
            - Zone and line crossing data depends on detection attributes
            - Times are calculated relative to current frame/clock time
            - Line crossing keys formatted as "line_id_direction" for uniqueness
            - Useful for debugging individual object behavior patterns
        
        Performance Notes:
            - O(z + c) complexity where z=zones, c=crossings for the tracker
            - Minimal memory overhead as it doesn't store historical snapshots
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
        Remove tracking data for objects no longer present in current detections.
        
        Automatically identifies and removes timing data for tracker IDs not present
        in the active list. Essential for long-running applications to prevent memory
        accumulation from objects that have permanently left the scene.
        
        Args:
            active_tracker_ids (List[int]): List of tracker IDs currently active
                                           in the scene. Should contain all tracker
                                           IDs present in current frame detections.
                                           Empty list removes all tracking data.
        
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> # Long-running video processing
            >>> model = YOLO("yolo11n.pt")
            >>> tracker = pf.timer.TimeTracker(fps=30.0)
            >>> cap = cv2.VideoCapture("long_video.mp4")
            >>> 
            >>> frame_count = 0
            >>> while True:
            >>>     ret, frame = cap.read()
            >>>     if not ret: break
            >>>     
            >>>     # Process frame
            >>>     outputs = model.track(frame)
            >>>     results = pf.results.from_ultralytics(outputs)
            >>>     tracker.update(results)
            >>>     
            >>>     # Periodic cleanup to prevent memory growth
            >>>     if frame_count % 300 == 0:  # Every 10 seconds at 30fps
            >>>         current_ids = [d.tracker_id for d in results if d.tracker_id is not None]
            >>>         tracker.cleanup_inactive_trackers(current_ids)
            >>>         print(f"Cleaned up inactive trackers at frame {frame_count}")
            >>>     
            >>>     frame_count += 1
            >>> 
            >>> # Manual cleanup based on scene analysis
            >>> # Get currently visible objects
            >>> visible_ids = []
            >>> for detection in results:
            >>>     if detection.tracker_id is not None and detection.confidence > 0.5:
            >>>         visible_ids.append(detection.tracker_id)
            >>> tracker.cleanup_inactive_trackers(visible_ids)
            >>> 
            >>> # Complete cleanup (remove all tracking data)
            >>> tracker.cleanup_inactive_trackers([])
        
        Notes:
            - Should be called periodically in long-running applications
            - Only removes data for tracker IDs not in the active list
            - Does not affect frame counter or timing mode settings  
            - Safe to call frequently as it only processes dictionary differences
            - Consider calling every few seconds or minutes depending on use case
        
        Performance Notes:
            - O(n + m) complexity where n=tracked objects, m=active objects
            - Memory freed proportional to number of inactive trackers removed
            - Recommended frequency: every 100-1000 frames depending on scene dynamics
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