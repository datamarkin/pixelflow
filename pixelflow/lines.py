"""
Line zone detection and counting for object tracking.

This module provides functionality for counting objects that cross defined lines
in video streams, with support for directional counting (in/out) and per-class tracking.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union
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
        triggering_anchor: Union[TriggerStrategy, str] = "center",
        minimum_crossing_threshold: int = 1,
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
            metadata: Additional custom data
        """
        self.start = start
        self.end = end
        self.line_id = line_id
        self.name = name or f"Line {line_id}"
        self.color = color or self._generate_color(line_id)
        self.metadata = metadata or {}
        
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
        
        self.triggering_anchor = triggering_anchor
        self.minimum_crossing_threshold = max(1, minimum_crossing_threshold)
        
        # Crossing history for stability
        self.crossing_history_length = max(2, minimum_crossing_threshold + 1)
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
    
    def trigger(self, detections) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check for line crossings and update counts.
        
        Args:
            detections: Detection results with tracker_id and bbox
            
        Returns:
            Tuple of (crossed_in, crossed_out) boolean arrays
        """
        n_detections = len(detections.predictions) if hasattr(detections, 'predictions') else 0
        crossed_in = np.full(n_detections, False)
        crossed_out = np.full(n_detections, False)
        
        if n_detections == 0:
            return crossed_in, crossed_out
        
        # Process each detection
        predictions = detections.predictions if hasattr(detections, 'predictions') else []
        
        for i, prediction in enumerate(predictions):
            # Skip if no tracker_id
            if prediction.tracker_id is None:
                continue
            
            # Skip if no bbox
            if prediction.bbox is None:
                continue
            
            # Check which side of the line the triggering anchor is on
            point = get_anchor_position(prediction.bbox, self.triggering_anchor)
            side = self._point_side_of_line(point)
            
            if side == 0:  # Point is exactly on the line, skip
                continue
            
            # Determine position: True for left side, False for right side
            tracker_state = side > 0
            
            # Update crossing history
            tracker_id = prediction.tracker_id
            class_id = prediction.class_id if prediction.class_id is not None else -1
            
            # Store class name mapping
            if hasattr(prediction, 'class_name') and prediction.class_name:
                self.class_id_to_name[class_id] = prediction.class_name
            elif class_id not in self.class_id_to_name:
                self.class_id_to_name[class_id] = str(class_id)
            
            crossing_history = self.crossing_state_history[tracker_id]
            crossing_history.append(tracker_state)
            
            # Check if we have enough history
            if len(crossing_history) < self.crossing_history_length:
                continue
            
            # Check if object crossed the line
            oldest_state = crossing_history[0]
            newest_state = crossing_history[-1]
            
            # Object must have been consistently on one side and now consistently on the other
            if crossing_history.count(oldest_state) == 1 and oldest_state != newest_state:
                # Crossing detected
                if newest_state:  # Moved from right to left (in)
                    self._in_count_per_class[class_id] += 1
                    crossed_in[i] = True
                else:  # Moved from left to right (out)
                    self._out_count_per_class[class_id] += 1
                    crossed_out[i] = True
        
        return crossed_in, crossed_out
    
    def reset_counts(self):
        """Reset all counting statistics."""
        self._in_count_per_class.clear()
        self._out_count_per_class.clear()
        self.crossing_state_history.clear()
        self.class_id_to_name.clear()


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
        triggering_anchor: Union[TriggerStrategy, str] = "center",
        minimum_crossing_threshold: int = 1,
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
            
            # Update predictions with crossing info
            if hasattr(results, 'predictions'):
                for i, prediction in enumerate(results.predictions):
                    if not hasattr(prediction, 'line_crossings'):
                        prediction.line_crossings = []
                    
                    if crossed_in[i]:
                        prediction.line_crossings.append({
                            'line_id': line.line_id,
                            'line_name': line.name,
                            'direction': 'in'
                        })
                    elif crossed_out[i]:
                        prediction.line_crossings.append({
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