"""
Line zone detection and counting for object tracking.

This module provides functionality for counting objects that cross defined lines
in video streams, with support for directional counting (in/out) and per-class tracking.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import numpy as np

from .strategies import TriggerStrategy, get_anchor_position


class Line:
    """
    Counts objects crossing a defined line with directional tracking.
    
    This class tracks objects crossing a line from one side to another,
    maintaining separate counts for "in" and "out" directions. It requires
    tracker IDs to prevent double-counting.
    
    Attributes:
        in_count: Total number of objects that crossed from right to left
        out_count: Total number of objects that crossed from left to right
        in_count_per_class: Per-class counts for right-to-left crossings
        out_count_per_class: Per-class counts for left-to-right crossings
    """
    
    def __init__(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        line_id: Union[int, str] = 0,
        name: str = "",
        color: Optional[Tuple[int, int, int]] = None,
        triggering_anchor: Union[str, List[str], TriggerStrategy, List[TriggerStrategy]] = "center",
        minimum_crossing_threshold: int = 1,
        mode: Literal["any", "all"] = "all",
        boundary_margin: float = 50.0,
        debounce_time: int = 30,
        minimum_distance: float = 10.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a Line.
        
        Args:
            start: Starting point of the line (x, y)
            end: Ending point of the line (x, y)
            line_id: Unique identifier for the line
            name: Human-readable name for the line
            color: RGB color tuple for visualization
            triggering_anchor: Anchor point to check for crossing 
                              (default: "center", options: "center", "bottom_center", 
                               "top_left", "top_right", "bottom_left", "bottom_right")
            minimum_crossing_threshold: Number of frames object must be on
                                       opposite side to count as crossed
            mode: "all" (AND logic) or "any" (OR logic) for multiple strategies
            boundary_margin: Distance in pixels from line endpoints to still consider valid (default: 50.0)
            debounce_time: Frames to wait before allowing another crossing for same tracker (default: 30)
            minimum_distance: Minimum distance in pixels the object must travel to be valid (default: 10.0)
            metadata: Additional custom data
        """
        self.start = start
        self.end = end
        self.line_id = line_id
        self.name = name or f"Line {line_id}"
        self.color = color or self._generate_color(line_id)
        self.boundary_margin = max(0.0, boundary_margin)
        self.debounce_time = max(0, debounce_time)
        self.minimum_distance = max(0.0, minimum_distance)
        self.metadata = metadata or {}
        
        # Temporal consistency tracking
        self.last_crossing_time: Dict[int, int] = {}  # tracker_id -> frame_count
        self.tracker_positions: Dict[int, List[Tuple[float, float]]] = defaultdict(list)  # For distance tracking
        self.frame_count = 0
        
        # Convert string anchor to TriggerStrategy if needed
        if isinstance(triggering_anchor, str):
            try:
                triggering_anchor = TriggerStrategy(triggering_anchor)
            except ValueError:
                valid_strategies = [s.value for s in TriggerStrategy]
                raise ValueError(
                    f"Invalid triggering_anchor '{triggering_anchor}'. "
                    f"Valid options are: {', '.join(valid_strategies)}"
                )
        
        self.minimum_crossing_threshold = max(1, minimum_crossing_threshold)
        
        # Crossing history for stability
        self.crossing_history_length = max(2, minimum_crossing_threshold + 1)
        
        # Store configuration for simplified system
        self.triggering_anchor = triggering_anchor
        self.mode = mode
        self.use_multiple_anchors = isinstance(triggering_anchor, (list, tuple))
        
        # Single anchor history (existing behavior)
        self.crossing_state_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.crossing_history_length)
        )
        
        # Counting
        self._in_count_per_class: Counter = Counter()
        self._out_count_per_class: Counter = Counter()
        self.class_id_to_name: Dict[int, str] = {}
        
        # Calculate line vector and perpendicular for side detection
        self._calculate_line_geometry()
    
    def _generate_color(self, line_id) -> Tuple[int, int, int]:
        """Generate a consistent color based on line_id."""
        import colorsys
        golden_ratio = 0.618033988749895
        hue = (hash(str(line_id)) * golden_ratio) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        return tuple(int(c * 255) for c in rgb)
    
    def _calculate_line_geometry(self):
        """Calculate line vector and perpendicular for side detection."""
        self.dx = self.end[0] - self.start[0]
        self.dy = self.end[1] - self.start[1]
        
        # Line equation: ax + by + c = 0
        # For line from (x1,y1) to (x2,y2): (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
        self.a = self.dy
        self.b = -self.dx
        self.c = self.dx * self.start[1] - self.dy * self.start[0]
    
    @property
    def in_count(self) -> int:
        """Total number of objects that crossed into the zone."""
        return sum(self._in_count_per_class.values())
    
    @property
    def out_count(self) -> int:
        """Total number of objects that crossed out of the zone."""
        return sum(self._out_count_per_class.values())
    
    @property
    def in_count_per_class(self) -> Dict[int, int]:
        """Per-class counts for objects crossing in."""
        return dict(self._in_count_per_class)
    
    @property
    def out_count_per_class(self) -> Dict[int, int]:
        """Per-class counts for objects crossing out."""
        return dict(self._out_count_per_class)
    
    
    def _point_side_of_line(self, point: Tuple[float, float]) -> int:
        """
        Determine which side of the line a point is on.
        
        Returns:
            1 if point is on the left/positive side
            -1 if point is on the right/negative side
            0 if point is on the line
        """
        x, y = point
        value = self.a * x + self.b * y + self.c
        if value > 0:
            return 1
        elif value < 0:
            return -1
        else:
            return 0
    
    def _is_point_near_line_segment(self, point: Tuple[float, float], margin: Optional[float] = None) -> bool:
        """
        Check if a point is within the bounds of the line segment (with optional margin).
        
        This prevents counting objects that pass beside the line segment
        (outside the start/end points) but would cross the infinite line extension.
        
        Args:
            point: The point to check (x, y)
            margin: Additional margin beyond line endpoints (uses instance boundary_margin if None)
        
        Returns:
            True if point is near the line segment, False otherwise
        """
        if margin is None:
            margin = self.boundary_margin
        x, y = point
        x1, y1 = self.start
        x2, y2 = self.end
        
        # Calculate the parameter t for the closest point on the line
        # Using parametric form: P(t) = start + t * (end - start)
        line_length_sq = self.dx * self.dx + self.dy * self.dy
        
        if line_length_sq == 0:
            # Start and end are the same point
            dist_sq = (x - x1) ** 2 + (y - y1) ** 2
            return dist_sq <= margin * margin
        
        # Calculate t parameter for projection of point onto line
        t = ((x - x1) * self.dx + (y - y1) * self.dy) / line_length_sq
        
        # Check if projection falls within segment bounds (with margin)
        margin_ratio = margin / (line_length_sq ** 0.5)
        return -margin_ratio <= t <= 1.0 + margin_ratio
    
    def trigger(self, detections) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check for line crossings and update counts.
        
        Args:
            detections: Detection results with tracker_id and bbox
            
        Returns:
            Tuple of (crossed_in, crossed_out) boolean arrays
        """
        # Increment frame counter for temporal tracking
        self.frame_count += 1
        n_detections = len(detections.detections)
        crossed_in = np.full(n_detections, False)
        crossed_out = np.full(n_detections, False)
        
        if n_detections == 0:
            return crossed_in, crossed_out
        
        # Process each detection
        predictions = detections.detections
        
        for i, prediction in enumerate(predictions):
            # Skip if no tracker_id
            if prediction.tracker_id is None:
                continue
            
            # Skip if no bbox
            if prediction.bbox is None:
                continue
            
            # Get tracker ID early for use in boundary checking
            tracker_id = prediction.tracker_id
            
            # TODO: Implement full multi-anchor logic for simplified system
            # For now, always use single anchor behavior
            if True:
                # Single anchor (existing behavior)
                point = get_anchor_position(prediction.bbox, self.triggering_anchor)
                
                # First check if the point is near the line segment
                # This prevents counting objects that pass beside the line
                if not self._is_point_near_line_segment(point):
                    # Clear history for trackers that move away from the line
                    if tracker_id in self.crossing_state_history:
                        self.crossing_state_history[tracker_id].clear()
                    continue
                
                side = self._point_side_of_line(point)
                
                if side == 0:  # Point is exactly on the line, skip
                    continue
                
                # Determine position: True for left side, False for right side
                tracker_state = side > 0
            
            # Update crossing history
            class_id = prediction.class_id if prediction.class_id is not None else -1
            
            # Store class name mapping
            if hasattr(prediction, 'class_name') and prediction.class_name:
                self.class_id_to_name[class_id] = prediction.class_name
            elif class_id not in self.class_id_to_name:
                self.class_id_to_name[class_id] = str(class_id)
            
            # Use appropriate history tracking based on mode
            if self.use_multiple_anchors:
                # For multiple anchors, we simplify by using the combined result
                # Create a simple history for the combined state
                if not hasattr(self, 'multi_anchor_history'):
                    self.multi_anchor_history: Dict[int, deque] = defaultdict(
                        lambda: deque(maxlen=self.crossing_history_length)
                    )
                crossing_history = self.multi_anchor_history[tracker_id]
            else:
                crossing_history = self.crossing_state_history[tracker_id]
            
            crossing_history.append(tracker_state)
            
            # Check if we have enough history
            if len(crossing_history) < self.crossing_history_length:
                continue
            
            # Check if object crossed the line
            oldest_state = crossing_history[0]
            newest_state = crossing_history[-1]
            
            # Only trigger if oldest state appears exactly once
            if crossing_history.count(oldest_state) > 1:
                continue
            
            # Must have different states to be a crossing
            if oldest_state == newest_state:
                continue
            
            # Get the current position for validation
            if self.use_multiple_anchors:
                # Use center point for distance tracking when multiple anchors
                current_point = get_anchor_position(prediction.bbox, TriggerStrategy.CENTER)
            else:
                current_point = point
            
            # Validate temporal consistency before counting
            if not self._validate_crossing_timing(tracker_id):
                continue
                
            # Validate minimum distance if enabled
            if not self._validate_minimum_distance(tracker_id, current_point):
                continue
            
            # Crossing detected - use newest state to determine direction
            if newest_state:  # Moved from right to left (in)
                self._in_count_per_class[class_id] += 1
                crossed_in[i] = True
                self.last_crossing_time[tracker_id] = self.frame_count
            else:  # Moved from left to right (out)
                self._out_count_per_class[class_id] += 1
                crossed_out[i] = True
                self.last_crossing_time[tracker_id] = self.frame_count
        
        return crossed_in, crossed_out
    
    def _validate_crossing_timing(self, tracker_id: int) -> bool:
        """
        Validate that enough time has passed since the last crossing for this tracker.
        
        Args:
            tracker_id: ID of the tracker to validate
            
        Returns:
            True if crossing is allowed, False if still in debounce period
        """
        if self.debounce_time <= 0:
            return True  # No debouncing
            
        if tracker_id not in self.last_crossing_time:
            return True  # First crossing for this tracker
            
        frames_since_last = self.frame_count - self.last_crossing_time[tracker_id]
        return frames_since_last >= self.debounce_time
    
    def _validate_minimum_distance(self, tracker_id: int, current_point: Tuple[float, float]) -> bool:
        """
        Validate that the tracker has moved a minimum distance since we started tracking it.
        
        Args:
            tracker_id: ID of the tracker to validate
            current_point: Current position of the tracker
            
        Returns:
            True if distance requirement is met, False otherwise
        """
        if self.minimum_distance <= 0:
            return True  # No distance requirement
            
        positions = self.tracker_positions[tracker_id]
        positions.append(current_point)
        
        # Keep only last few positions to avoid memory issues
        if len(positions) > 10:
            positions.pop(0)
        
        if len(positions) < 2:
            return True  # Need at least 2 positions to calculate distance
        
        # Calculate total distance traveled
        total_distance = 0.0
        for i in range(1, len(positions)):
            prev_x, prev_y = positions[i-1]
            curr_x, curr_y = positions[i]
            distance = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5
            total_distance += distance
        
        return total_distance >= self.minimum_distance
    
    def reset_counts(self):
        """Reset all counting statistics."""
        self._in_count_per_class.clear()
        self._out_count_per_class.clear()
        
        if self.use_multiple_anchors:
            if hasattr(self, 'multi_anchor_history'):
                self.multi_anchor_history.clear()
            if hasattr(self, 'anchor_crossing_histories'):
                self.anchor_crossing_histories.clear()
        else:
            self.crossing_state_history.clear()
            
        self.class_id_to_name.clear()
        
        # Reset temporal consistency tracking
        self.last_crossing_time.clear()
        self.tracker_positions.clear()
        self.frame_count = 0


class Lines:
    """
    Manages multiple Line instances for complex crossing detection scenarios.
    
    This class provides centralized management of multiple lines,
    allowing for easy processing of detections against all lines at once.
    """
    
    def __init__(self):
        """Initialize the Lines manager."""
        self.lines: List[Line] = []
        self._line_dict: Dict[Union[int, str], Line] = {}
    
    def add_line(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        line_id: Optional[Union[int, str]] = None,
        name: str = "",
        color: Optional[Tuple[int, int, int]] = None,
        triggering_anchor: Union[str, List[str], TriggerStrategy, List[TriggerStrategy]] = "center",
        minimum_crossing_threshold: int = 1,
        mode: Literal["any", "all"] = "all",
        boundary_margin: float = 50.0,
        debounce_time: int = 30,
        minimum_distance: float = 10.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Line:
        """
        Add a new line.
        
        Args:
            start: Starting point of the line (x, y)
            end: Ending point of the line (x, y)
            line_id: Unique identifier (auto-generated if None)
            name: Human-readable name
            color: RGB color tuple for visualization
            triggering_anchor: Anchor point to check for crossing (default: "center")
            minimum_crossing_threshold: Frames required for crossing
            mode: "all" (AND logic) or "any" (OR logic) for multiple strategies
            boundary_margin: Distance in pixels from line endpoints to still consider valid (default: 50.0)
            debounce_time: Frames to wait before allowing another crossing for same tracker (default: 30)
            minimum_distance: Minimum distance in pixels the object must travel to be valid (default: 10.0)
            metadata: Additional custom data
            
        Returns:
            The created Line object
        """
        # Auto-generate line_id if not provided
        if line_id is None:
            line_id = len(self.lines)
            while line_id in self._line_dict:
                line_id += 1
        
        # Check for duplicate line_id
        if line_id in self._line_dict:
            raise ValueError(f"Line with ID {line_id} already exists")
        
        # Create line
        line = Line(
            start=start,
            end=end,
            line_id=line_id,
            name=name,
            color=color,
            triggering_anchor=triggering_anchor,
            minimum_crossing_threshold=minimum_crossing_threshold,
            mode=mode,
            boundary_margin=boundary_margin,
            debounce_time=debounce_time,
            minimum_distance=minimum_distance,
            metadata=metadata
        )
        
        # Add to manager
        self.lines.append(line)
        self._line_dict[line_id] = line
        
        return line
    
    def remove_line(self, line_id: Union[int, str]):
        """Remove a line by its ID."""
        if line_id in self._line_dict:
            line = self._line_dict.pop(line_id)
            self.lines.remove(line)
    
    def get_line(self, line_id: Union[int, str]) -> Optional[Line]:
        """Get a line by its ID."""
        return self._line_dict.get(line_id)
    
    def update(self, results):
        """
        Process detections against all lines and update results with crossing information.
        
        This method checks each detection against all lines and updates:
        - prediction.line_crossings: List of crossing events for each detection
        - Line statistics (in_count, out_count, per-class counts)
        
        Args:
            results: Results object containing predictions
            
        Returns:
            Updated Results object (modified in-place)
        """        
        for line in self.lines:
            crossed_in, crossed_out = line.trigger(results)
            
            # Update detections with crossing info
            for i, detection in enumerate(results.detections):
                if not hasattr(detection, 'line_crossings'):
                    detection.line_crossings = []
                
                if crossed_in[i]:
                    detection.line_crossings.append({
                        'line_id': line.line_id,
                        'line_name': line.name,
                        'direction': 'in'
                    })
                elif crossed_out[i]:
                    detection.line_crossings.append({
                        'line_id': line.line_id,
                        'line_name': line.name,
                        'direction': 'out'
                    })
        
        return results
    
    def get_line_counts(self) -> Dict[Union[int, str], Dict[str, int]]:
        """
        Get current counts for all lines.
        
        Returns:
            Dictionary mapping line_id to counts
        """
        counts = {}
        for line in self.lines:
            counts[line.line_id] = {
                'name': line.name,
                'in_count': line.in_count,
                'out_count': line.out_count,
                'in_count_per_class': line.in_count_per_class,
                'out_count_per_class': line.out_count_per_class
            }
        return counts
    
    def reset_all_counts(self):
        """Reset counts for all lines."""
        for line in self.lines:
            line.reset_counts()
    
    def clear_lines(self):
        """Remove all lines."""
        self.lines.clear()
        self._line_dict.clear()