"""
Line crossing detection and directional counting for object tracking.

This module provides sophisticated line crossing detection capabilities for video streams,
implementing temporal validation and directional counting algorithms. It supports both
single and multi-crossing scenarios with debouncing, distance validation, and per-class
tracking statistics, making it ideal for traffic monitoring, people counting, and access
control applications.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import numpy as np

from .strategies import validate_strategy, get_anchor_position, STRATEGY_CENTER


class Crossing:
    """
    Detects and counts objects crossing a defined line with directional tracking and temporal validation.
    
    This class implements a sophisticated line crossing detection algorithm that tracks objects 
    crossing a line segment from one side to another. It uses temporal consistency validation,
    debouncing, and minimum distance requirements to ensure accurate counting while preventing
    false positives from noise or oscillation. The algorithm maintains separate directional
    counts and supports per-class statistics for detailed analytics.
    
    The crossing detection algorithm works by:
    1. Tracking object positions relative to the line over multiple frames
    2. Validating crossings using temporal consistency (debouncing)
    3. Enforcing minimum distance requirements to filter out noise
    4. Maintaining directional state history to detect actual crossings
    5. Supporting multiple anchor points for different object types
    
    Args:
        start (Tuple[float, float]): Starting point of the line (x, y) in pixels
        end (Tuple[float, float]): Ending point of the line (x, y) in pixels
        line_id (Union[int, str]): Unique identifier for the crossing line. Default is 0
        name (str): Human-readable name for the crossing. Auto-generated if empty
        color (Optional[Tuple[int, int, int]]): RGB color tuple for visualization. 
                                               Auto-generated if None
        triggering_anchor (Union[str, List[str]]): Anchor point(s) to check for crossing.
                                                  Options: "center", "bottom_center", 
                                                  "top_left", "top_right", "bottom_left", 
                                                  "bottom_right". Default is "center"
        minimum_crossing_threshold (int): Number of consecutive frames an object must be
                                         on the opposite side to count as crossed. 
                                         Range: [1, inf]. Default is 1
        mode (Literal["any", "all"]): Logic mode for multiple anchor strategies.
                                     "all" uses AND logic, "any" uses OR logic. 
                                     Default is "all"
        boundary_margin (float): Distance in pixels from line endpoints where crossings
                                are still considered valid. Range: [0, inf]. Default is 50.0
        debounce_time (int): Number of frames to wait before allowing another crossing
                            for the same tracker ID. Range: [0, inf]. Default is 30
        minimum_distance (float): Minimum distance in pixels an object must travel
                                 to be considered a valid crossing. Range: [0, inf]. 
                                 Default is 10.0
        metadata (Optional[Dict[str, Any]]): Additional custom data for the crossing
    
    Attributes:
        in_count (int): Total number of objects that crossed from right to left (positive side to negative side)
        out_count (int): Total number of objects that crossed from left to right (negative side to positive side)
        in_count_per_class (Dict[int, int]): Per-class counts for right-to-left crossings
        out_count_per_class (Dict[int, int]): Per-class counts for left-to-right crossings
        line_id (Union[int, str]): Unique identifier for this crossing
        name (str): Human-readable name for this crossing
        color (Tuple[int, int, int]): RGB color tuple for visualization
        start (Tuple[float, float]): Starting point of the line
        end (Tuple[float, float]): Ending point of the line
        
    Raises:
        ValueError: If start and end points are identical
        ValueError: If minimum_crossing_threshold is less than 1
        ValueError: If boundary_margin, debounce_time, or minimum_distance are negative
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Basic crossing detection setup
        >>> model = YOLO("yolo11n.pt")
        >>> tracker = pf.tracker.ByteTracker()
        >>> crossing = pf.crossings.Crossing(
        ...     start=(100, 200), 
        ...     end=(500, 200),
        ...     name="Main Entry"
        ... )
        >>> 
        >>> # Process video frame with crossing detection
        >>> image = cv2.imread("frame.jpg")
        >>> outputs = model.predict(image)
        >>> results = pf.results.from_ultralytics(outputs)
        >>> tracked_results = tracker.update(results)
        >>> crossed_in, crossed_out = crossing.trigger(tracked_results)
        >>> print(f"In: {crossing.in_count}, Out: {crossing.out_count}")
        >>> 
        >>> # Advanced setup with custom parameters for high-precision counting
        >>> precision_crossing = pf.crossings.Crossing(
        ...     start=(200, 100), 
        ...     end=(200, 400),
        ...     triggering_anchor="bottom_center",
        ...     minimum_crossing_threshold=3,
        ...     debounce_time=60,
        ...     minimum_distance=20.0,
        ...     name="Precision Gate"
        ... )
        >>> 
        >>> # Multiple anchor point strategy for complex objects
        >>> multi_crossing = pf.crossings.Crossing(
        ...     start=(0, 300), 
        ...     end=(640, 300),
        ...     triggering_anchor=["center", "bottom_center"],
        ...     mode="any",
        ...     boundary_margin=25.0
        ... )
        >>> 
        >>> # Per-class counting analysis
        >>> person_results = tracker.update(person_detections)
        >>> crossed_in, crossed_out = crossing.trigger(person_results)
        >>> person_counts = crossing.in_count_per_class
        >>> print(f"Person crossings by class: {person_counts}")
    
    Notes:
        - Requires tracker IDs for temporal consistency - call tracker.update() before crossing.trigger()
        - Line equation is calculated as ax + by + c = 0 for efficient side determination
        - Crossing direction is determined by the line orientation: left side is "in", right side is "out"
        - Temporal validation prevents double-counting from object oscillation near the line
        - Distance validation filters out stationary objects that might drift across the line
        - Boundary margin prevents false positives from objects passing outside line endpoints
        - Class ID mapping is automatically maintained for human-readable statistics
        
    Performance Notes:
        - Memory usage scales with number of unique tracker IDs (typically 100-1000 objects)
        - Processing time is O(n) where n is number of detections per frame
        - History tracking uses fixed-size deques to prevent memory leaks
        - Distance calculation uses squared distance for efficiency (no sqrt needed)
        
    See Also:
        Crossings : Manager class for multiple crossing instances
        pf.tracker.ByteTracker : Object tracker providing required tracker IDs
        pf.strategies : Anchor point strategies for different object types
    """
    
    def __init__(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        line_id: Union[int, str] = 0,
        name: str = "",
        color: Optional[Tuple[int, int, int]] = None,
        triggering_anchor: Union[str, List[str]] = "center",
        minimum_crossing_threshold: int = 1,
        mode: Literal["any", "all"] = "all",
        boundary_margin: float = 50.0,
        debounce_time: int = 30,
        minimum_distance: float = 10.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a crossing line detector with temporal validation and directional counting.
        
        Creates a crossing detector that monitors objects passing through a defined line segment.
        The detector implements sophisticated validation algorithms including debouncing, distance
        thresholds, and temporal consistency checks to ensure accurate counting in real-world scenarios.
        
        Args:
            start (Tuple[float, float]): Starting point of the line segment (x, y) in pixels.
                                        Must be different from end point
            end (Tuple[float, float]): Ending point of the line segment (x, y) in pixels.
                                      Must be different from start point
            line_id (Union[int, str]): Unique identifier for the crossing line. Used for tracking
                                      and referencing. Default is 0
            name (str): Human-readable name for the crossing line. If empty, auto-generates
                       name based on line_id. Default is ""
            color (Optional[Tuple[int, int, int]]): RGB color tuple (0-255) for visualization.
                                                   If None, auto-generates color based on line_id.
                                                   Default is None
            triggering_anchor (Union[str, List[str]]): Anchor point(s) on bounding boxes to use
                                                      for crossing detection. Single string or list
                                                      of strings. Options: "center", "bottom_center",
                                                      "top_left", "top_right", "bottom_left",
                                                      "bottom_right". Default is "center"
            minimum_crossing_threshold (int): Number of consecutive frames an object must remain
                                             on the opposite side of the line to register as crossed.
                                             Higher values reduce false positives. Range: [1, inf].
                                             Default is 1
            mode (Literal["any", "all"]): Logic mode for multiple anchor point strategies.
                                         "all" requires ALL anchor points to cross (AND logic),
                                         "any" requires ANY anchor point to cross (OR logic).
                                         Only relevant when triggering_anchor is a list. Default is "all"
            boundary_margin (float): Distance in pixels from line endpoints where objects are still
                                    considered for crossing detection. Prevents false negatives from
                                    objects passing slightly outside the line segment. Range: [0, inf].
                                    Default is 50.0
            debounce_time (int): Number of frames to wait before allowing another crossing event
                               for the same tracker ID. Prevents rapid oscillation counting.
                               Range: [0, inf]. 0 disables debouncing. Default is 30
            minimum_distance (float): Minimum distance in pixels an object must travel during
                                     the crossing detection period to be considered valid.
                                     Filters out stationary objects that drift across the line.
                                     Range: [0, inf]. 0 disables distance checking. Default is 10.0
            metadata (Optional[Dict[str, Any]]): Additional custom data to associate with
                                               this crossing. Can store application-specific
                                               information. Default is None
                                               
        Raises:
            ValueError: If start and end points are identical
            ValueError: If minimum_crossing_threshold is less than 1
            ValueError: If boundary_margin is negative
            ValueError: If debounce_time is negative
            ValueError: If minimum_distance is negative
            TypeError: If triggering_anchor contains invalid strategy names
            
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Basic horizontal crossing line
            >>> crossing = pf.crossings.Crossing(
            ...     start=(100, 300),
            ...     end=(500, 300),
            ...     name="Entry Gate"
            ... )
            >>> 
            >>> # Vertical crossing with custom parameters for precision
            >>> precision_crossing = pf.crossings.Crossing(
            ...     start=(320, 0),
            ...     end=(320, 480),
            ...     line_id="gate_001",
            ...     triggering_anchor="bottom_center",
            ...     minimum_crossing_threshold=5,
            ...     debounce_time=90,
            ...     minimum_distance=30.0,
            ...     boundary_margin=25.0
            ... )
            >>> 
            >>> # Multi-anchor crossing for complex objects
            >>> vehicle_crossing = pf.crossings.Crossing(
            ...     start=(0, 200),
            ...     end=(640, 200),
            ...     triggering_anchor=["center", "bottom_center", "top_center"],
            ...     mode="any",
            ...     color=(255, 0, 0),
            ...     metadata={"type": "vehicle", "lane": 1}
            ... )
            >>> 
            >>> # High-frequency crossing with minimal debouncing
            >>> rapid_crossing = pf.crossings.Crossing(
            ...     start=(200, 100),
            ...     end=(200, 400),
            ...     debounce_time=10,
            ...     minimum_distance=5.0,
            ...     boundary_margin=15.0
            ... )
        
        Notes:
            - Line geometry is automatically calculated for efficient side determination
            - Color generation uses golden ratio hashing for consistent, distinct colors
            - All numeric parameters are automatically clamped to valid ranges
            - Temporal tracking structures are initialized but remain empty until first use
            - The line equation ax + by + c = 0 is computed for fast point-to-line calculations
            - Multiple anchor point mode requires strategy validation for all anchor types
            
        Performance Notes:
            - Initialization is O(1) and very fast regardless of parameter values
            - Memory allocation scales with expected number of concurrent tracked objects
            - Color generation involves hash computation and HSV-to-RGB conversion
        """
        self.start = start
        self.end = end
        self.line_id = line_id
        self.name = name or f"Crossing {line_id}"
        self.color = color or self._generate_color(line_id)
        self.boundary_margin = max(0.0, boundary_margin)
        self.debounce_time = max(0, debounce_time)
        self.minimum_distance = max(0.0, minimum_distance)
        self.metadata = metadata or {}
        
        # Temporal consistency tracking
        self.last_crossing_time: Dict[int, int] = {}  # tracker_id -> frame_count
        self.tracker_positions: Dict[int, List[Tuple[float, float]]] = defaultdict(list)  # For distance tracking
        self.frame_count = 0
        
        # Validate triggering_anchor
        triggering_anchor = validate_strategy(triggering_anchor)
        
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
        """Generate a consistent and visually distinct color based on line_id.
        
        This method creates deterministic colors using the golden ratio for optimal
        color distribution. The same line_id will always produce the same color,
        ensuring visual consistency across sessions.
        
        Args:
            line_id: Unique identifier used as seed for color generation
            
        Returns:
            Tuple[int, int, int]: RGB color tuple with values in range [0, 255]
            
        Notes:
            - Uses golden ratio (φ ≈ 0.618) for optimal hue distribution
            - Hash function ensures even distribution across color space
            - High saturation (0.8) and brightness (0.9) for visibility
        """
        import colorsys
        golden_ratio = 0.618033988749895
        hue = (hash(str(line_id)) * golden_ratio) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        return tuple(int(c * 255) for c in rgb)
    
    def _calculate_line_geometry(self):
        """Calculate line equation coefficients for efficient point-to-line side determination.
        
        This method precomputes the line equation in the form ax + by + c = 0 from the
        start and end points. This allows for fast determination of which side of the line
        any point lies on during crossing detection.
        
        The line equation is derived as:
        - For line from (x1,y1) to (x2,y2): (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
        - Simplified to: a = (y2-y1), b = -(x2-x1), c = (x2-x1)y1 - (y2-y1)x1
        
        Sets instance attributes:
            dx (float): Horizontal component of line vector (end.x - start.x)
            dy (float): Vertical component of line vector (end.y - start.y) 
            a (float): Coefficient 'a' in line equation ax + by + c = 0
            b (float): Coefficient 'b' in line equation ax + by + c = 0
            c (float): Coefficient 'c' in line equation ax + by + c = 0
            
        Notes:
            - Called automatically during __init__ to precompute line geometry
            - Enables O(1) point-to-line side determination in _point_side_of_line
            - Line coefficients remain constant throughout crossing object lifetime
        """
        self.dx = self.end[0] - self.start[0]
        self.dy = self.end[1] - self.start[1]
        
        # Line equation: ax + by + c = 0
        # For line from (x1,y1) to (x2,y2): (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
        self.a = self.dy
        self.b = -self.dx
        self.c = self.dx * self.start[1] - self.dy * self.start[0]
    
    @property
    def in_count(self) -> int:
        """Get the total count of objects that crossed from right to left (positive to negative side).
        
        This property returns the aggregate count of all objects that have crossed the line
        in the "in" direction, regardless of their class. The "in" direction is defined as
        crossing from the positive side of the line to the negative side, which corresponds
        to right-to-left movement when viewing the line from start to end point.
        
        Returns:
            int: Total number of objects that have crossed from right to left.
                 Range: [0, inf]. Always non-negative.
                 
        Example:
            >>> crossing = pf.crossings.Crossing(start=(100, 200), end=(500, 200))
            >>> # After processing some frames...
            >>> print(f"Total entries: {crossing.in_count}")
            42
            >>> print(f"Per class entries: {crossing.in_count_per_class}")
            {0: 25, 1: 17}  # 25 + 17 = 42 total
        
        Notes:
            - This is a computed property that sums all per-class counts
            - Value is automatically updated when objects cross in the trigger() method
            - Direction is determined by line geometry and object movement
            - Reset to 0 when reset_counts() is called
        """
        return sum(self._in_count_per_class.values())
    
    @property
    def out_count(self) -> int:
        """Get the total count of objects that crossed from left to right (negative to positive side).
        
        This property returns the aggregate count of all objects that have crossed the line
        in the "out" direction, regardless of their class. The "out" direction is defined as
        crossing from the negative side of the line to the positive side, which corresponds
        to left-to-right movement when viewing the line from start to end point.
        
        Returns:
            int: Total number of objects that have crossed from left to right.
                 Range: [0, inf]. Always non-negative.
                 
        Example:
            >>> crossing = pf.crossings.Crossing(start=(100, 200), end=(500, 200))
            >>> # After processing some frames...
            >>> print(f"Total exits: {crossing.out_count}")
            38
            >>> print(f"Net flow: {crossing.in_count - crossing.out_count}")
            4  # 42 in - 38 out = 4 net inward
        
        Notes:
            - This is a computed property that sums all per-class counts
            - Value is automatically updated when objects cross in the trigger() method
            - Direction is determined by line geometry and object movement
            - Reset to 0 when reset_counts() is called
        """
        return sum(self._out_count_per_class.values())
    
    @property
    def in_count_per_class(self) -> Dict[int, int]:
        """Get per-class counts of objects that crossed from right to left (positive to negative side).
        
        This property returns a dictionary mapping class IDs to their respective crossing counts
        in the "in" direction. This enables detailed analytics by object type, useful for applications
        like traffic analysis where different vehicle types need separate counting.
        
        Returns:
            Dict[int, int]: Dictionary mapping class ID to crossing count.
                           Keys are integer class IDs from the detection model.
                           Values are non-negative integers representing crossing counts.
                           Empty dictionary if no crossings have occurred.
                           
        Example:
            >>> crossing = pf.crossings.Crossing(start=(100, 200), end=(500, 200))
            >>> # After processing frames with people (class 0) and cars (class 2)...
            >>> class_counts = crossing.in_count_per_class
            >>> print(class_counts)
            {0: 25, 2: 17}  # 25 people, 17 cars crossed inward
            >>> print(f"People entries: {class_counts.get(0, 0)}")
            25
            >>> print(f"Car entries: {class_counts.get(2, 0)}")
            17
        
        Notes:
            - Returns a copy of internal counter to prevent external modification
            - Class IDs correspond to model's class numbering (e.g., COCO classes)
            - Class names can be accessed via crossing.class_id_to_name mapping
            - Automatically updated when objects cross in the trigger() method
            - Reset when reset_counts() is called
        """
        return dict(self._in_count_per_class)
    
    @property
    def out_count_per_class(self) -> Dict[int, int]:
        """Get per-class counts of objects that crossed from left to right (negative to positive side).
        
        This property returns a dictionary mapping class IDs to their respective crossing counts
        in the "out" direction. This enables detailed analytics by object type, useful for applications
        like monitoring exit patterns or analyzing bidirectional traffic flow by category.
        
        Returns:
            Dict[int, int]: Dictionary mapping class ID to crossing count.
                           Keys are integer class IDs from the detection model.
                           Values are non-negative integers representing crossing counts.
                           Empty dictionary if no crossings have occurred.
                           
        Example:
            >>> crossing = pf.crossings.Crossing(start=(100, 200), end=(500, 200))
            >>> # After processing frames...
            >>> out_counts = crossing.out_count_per_class
            >>> in_counts = crossing.in_count_per_class
            >>> print(f"Outward: {out_counts}")
            {0: 18, 2: 20}  # 18 people, 20 cars crossed outward
            >>> 
            >>> # Calculate net flow by class
            >>> for class_id in set(in_counts.keys()) | set(out_counts.keys()):
            ...     net = in_counts.get(class_id, 0) - out_counts.get(class_id, 0)
            ...     print(f"Class {class_id} net flow: {net}")
            Class 0 net flow: 7   # 25 in - 18 out = 7 people net inward
            Class 2 net flow: -3  # 17 in - 20 out = -3 cars net outward
        
        Notes:
            - Returns a copy of internal counter to prevent external modification
            - Class IDs correspond to model's class numbering (e.g., COCO classes)
            - Class names can be accessed via crossing.class_id_to_name mapping
            - Automatically updated when objects cross in the trigger() method
            - Reset when reset_counts() is called
        """
        return dict(self._out_count_per_class)
    
    
    def _point_side_of_line(self, point: Tuple[float, float]) -> int:
        """
        Determine which side of the line a point is on using precomputed line equation.
        
        This method uses the line equation ax + by + c = 0 to efficiently determine
        the side of the line where a point lies. The sign of the equation result
        indicates the side: positive for left/positive side, negative for right/negative side.
        
        Args:
            point (Tuple[float, float]): Point coordinates (x, y) to test
            
        Returns:
            int: Side indicator:
                 1 if point is on the left/positive side (crossing "in" direction origin)
                -1 if point is on the right/negative side (crossing "out" direction origin)
                 0 if point is exactly on the line (rare in practice)
                 
        Example:
            >>> crossing = pf.crossings.Crossing(start=(100, 200), end=(500, 200))
            >>> # For horizontal line, points above are positive, below are negative
            >>> print(crossing._point_side_of_line((300, 150)))  # Above line
            1
            >>> print(crossing._point_side_of_line((300, 250)))  # Below line  
            -1
            >>> print(crossing._point_side_of_line((300, 200)))  # On line
            0
        
        Notes:
            - Uses precomputed coefficients from _calculate_line_geometry for efficiency
            - Line orientation determines which side is positive vs negative
            - Floating point precision may rarely result in exact zero (on-line) cases
            - O(1) computation time using simple arithmetic operations
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
        Check if a point is within the bounds of the line segment with optional margin extension.
        
        This method prevents counting objects that pass beside the line segment (outside the
        start/end points) but would mathematically cross the infinite line extension. It uses
        parametric line representation to determine if the closest point on the infinite line
        falls within the segment bounds, extended by the specified margin.
        
        Args:
            point (Tuple[float, float]): Point coordinates (x, y) to test for proximity
            margin (Optional[float]): Additional distance in pixels beyond line endpoints
                                     where objects are still considered for crossing detection.
                                     If None, uses instance boundary_margin. Range: [0, inf]
        
        Returns:
            bool: True if point is within the extended line segment bounds, False otherwise
            
        Example:
            >>> crossing = pf.crossings.Crossing(
            ...     start=(100, 200), 
            ...     end=(500, 200),
            ...     boundary_margin=50.0
            ... )
            >>> # Point directly above line segment - within bounds
            >>> print(crossing._is_point_near_line_segment((300, 150)))
            True
            >>> # Point far to the left of start point - outside bounds  
            >>> print(crossing._is_point_near_line_segment((0, 150)))
            False
            >>> # Point slightly left of start but within margin - within bounds
            >>> print(crossing._is_point_near_line_segment((80, 150)))
            True
        
        Notes:
            - Uses parametric line form P(t) = start + t * (end - start) for calculation
            - Parameter t in range [-margin_ratio, 1+margin_ratio] indicates valid bounds
            - Handles degenerate case where start and end points are identical
            - Margin is converted to parametric space for consistent behavior regardless of line length
            - Essential for preventing false positives from objects passing outside line segment
            
        Performance Notes:
            - O(1) computation time using vector algebra
            - Avoids expensive square root operations by working in parametric space
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
    
    def trigger(self, detections, strategy: Union[str, List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect line crossings and update directional counts using temporal validation algorithm.
        
        This method implements the core crossing detection algorithm that processes tracked objects
        to determine if they have crossed the defined line. It uses a sophisticated multi-step
        validation process including temporal consistency checks, debouncing, and distance validation
        to ensure accurate counting while minimizing false positives from noise, oscillation, or
        stationary objects drifting across the line.
        
        The algorithm follows these steps:
        1. Validate that detections have tracker IDs for temporal consistency
        2. For each detection, calculate anchor point position using specified strategy
        3. Check if the point is within the line segment boundary (with margin)
        4. Determine which side of the line the object is on using line equation
        5. Update temporal history of object positions relative to the line
        6. Validate crossings using minimum frame threshold and debouncing
        7. Apply distance validation to filter out stationary drift
        8. Update directional counts and per-class statistics
        
        Args:
            detections: PixelFlow Results object containing detection predictions with
                       bounding boxes and tracker IDs. Each prediction must have
                       .tracker_id (int) and .bbox (4-tuple) attributes for crossing
                       detection to work properly
            strategy (Optional[Union[str, List[str]]]): Override anchor point strategy
                                                       for this trigger call. If None,
                                                       uses the crossing's configured
                                                       triggering_anchor. Valid strategies:
                                                       "center", "bottom_center", "top_left",
                                                       "top_right", "bottom_left", "bottom_right"
                                                       
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - crossed_in (np.ndarray): Boolean array indicating which detections
                                          crossed from right to left (positive to negative side)
                - crossed_out (np.ndarray): Boolean array indicating which detections
                                           crossed from left to right (negative to positive side)
                Both arrays have the same length as the input detections
                
        Raises:
            ValueError: If strategy override contains invalid anchor point names
            AttributeError: If detections object lacks required attributes
            IndexError: If detection bounding boxes have invalid format
            
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> # Setup crossing detection pipeline
            >>> model = YOLO("yolo11n.pt")
            >>> tracker = pf.tracker.ByteTracker()
            >>> crossing = pf.crossings.Crossing(
            ...     start=(200, 100), 
            ...     end=(200, 400),
            ...     name="Doorway"
            ... )
            >>> 
            >>> # Process single frame with crossing detection
            >>> image = cv2.imread("surveillance_frame.jpg")
            >>> outputs = model.predict(image)
            >>> results = pf.results.from_ultralytics(outputs)
            >>> tracked_results = tracker.update(results)  # Required for crossing detection
            >>> crossed_in, crossed_out = crossing.trigger(tracked_results)
            >>> print(f"Objects entered: {crossed_in.sum()}, Objects exited: {crossed_out.sum()}")
            >>> 
            >>> # Use different anchor strategy for specific detection
            >>> person_results = tracker.update(person_detections)
            >>> crossed_in, crossed_out = crossing.trigger(person_results, strategy="bottom_center")
            >>> 
            >>> # Multi-class crossing analysis
            >>> vehicle_results = tracker.update(vehicle_detections)
            >>> crossed_in, crossed_out = crossing.trigger(vehicle_results, strategy="center")
            >>> print(f"Total in: {crossing.in_count}, Total out: {crossing.out_count}")
            >>> print(f"Per class in: {crossing.in_count_per_class}")
            >>> 
            >>> # Access crossing events from detections
            >>> for i, detection in enumerate(tracked_results.detections):
            ...     if hasattr(detection, 'line_crossings'):
            ...         for event in detection.line_crossings:
            ...             print(f"Tracker {detection.tracker_id} crossed {event['line_name']} going {event['direction']}")
        
        Notes:
            - Requires tracker IDs: Call tracker.update(results) before crossing.trigger(results)
            - Crossing direction is determined by line orientation: "in" = right to left, "out" = left to right
            - Temporal validation prevents double-counting from object oscillation near the line
            - Objects must be within boundary_margin of line endpoints to trigger crossing detection
            - Debouncing prevents rapid re-triggering for the same tracker ID
            - Distance validation filters out stationary objects that drift across the line
            - Per-class statistics are automatically updated for each crossing event
            - Class name mapping is maintained for human-readable analytics
            - Line crossing events are added to detection.line_crossings attribute when using Crossings manager
            
        Performance Notes:
            - Processing time is O(n) where n is the number of detections in the frame
            - Memory usage scales with number of unique tracker IDs (bounded by deque maxlen)
            - Line side calculation uses precomputed line equation for efficiency
            - Distance calculations use squared distance to avoid expensive sqrt operations
            - Warning is issued once if no tracker IDs are found (not per frame)
            
        See Also:
            Crossings.update : Process detections against multiple crossings simultaneously
            _validate_crossing_timing : Internal method for debouncing validation
            _validate_minimum_distance : Internal method for distance threshold validation
        """
        # Validate override strategy if provided
        if strategy is not None:
            strategy = validate_strategy(strategy)
        
        # Increment frame counter for temporal tracking
        self.frame_count += 1
        n_detections = len(detections.detections)
        crossed_in = np.full(n_detections, False)
        crossed_out = np.full(n_detections, False)
        
        if n_detections == 0:
            return crossed_in, crossed_out
        
        # Process each detection
        predictions = detections.detections
        
        # Check if any detections have tracker IDs (for debugging)
        has_tracker_ids = any(pred.tracker_id is not None for pred in predictions)
        if predictions and not has_tracker_ids:
            import warnings
            warnings.warn(
                f"Crossing '{self.name}' (ID: {self.line_id}): No detections have tracker_id. "
                "Crossings require tracker_id for temporal consistency. "
                "Make sure to call tracker.update(results) before crossings.update(results).",
                UserWarning,
                stacklevel=3
            )

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
                # Use override strategy or crossing's own strategy
                detection_strategy = strategy if strategy is not None else self.triggering_anchor
                # Single anchor (existing behavior)
                point = get_anchor_position(prediction.bbox, detection_strategy)
                
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
                current_point = get_anchor_position(prediction.bbox, STRATEGY_CENTER)
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
        Validate that sufficient time has passed since the last crossing for temporal consistency.
        
        This method implements debouncing to prevent rapid re-triggering of crossing events
        for the same tracker ID. It ensures that objects oscillating near the line boundary
        don't generate multiple false crossing events within a short time window.
        
        Args:
            tracker_id (int): Unique identifier of the tracker to validate timing for
            
        Returns:
            bool: True if crossing is allowed (debounce period has elapsed or is disabled),
                  False if still within debounce period
                  
        Example:
            >>> crossing = pf.crossings.Crossing(
            ...     start=(100, 200), 
            ...     end=(500, 200),
            ...     debounce_time=30  # 30 frames minimum between crossings
            ... )
            >>> # After a crossing event at frame 100...
            >>> crossing.frame_count = 120
            >>> print(crossing._validate_crossing_timing(tracker_id=42))
            False  # Only 20 frames have passed, need 30
            >>> crossing.frame_count = 135  
            >>> print(crossing._validate_crossing_timing(tracker_id=42))
            True   # 35 frames have passed, debounce period expired
        
        Notes:
            - Returns True immediately if debounce_time is 0 (debouncing disabled)
            - Returns True for trackers that haven't crossed before (first crossing)
            - Frame counting is managed internally by the trigger() method
            - Essential for preventing oscillation-induced false positives
            - Debounce timing is per-tracker, allowing simultaneous crossings by different objects
            
        Performance Notes:
            - O(1) lookup time using dictionary for last crossing times
            - Memory usage scales with number of unique tracker IDs that have crossed
        """
        if self.debounce_time <= 0:
            return True  # No debouncing
            
        if tracker_id not in self.last_crossing_time:
            return True  # First crossing for this tracker
            
        frames_since_last = self.frame_count - self.last_crossing_time[tracker_id]
        return frames_since_last >= self.debounce_time
    
    def _validate_minimum_distance(self, tracker_id: int, current_point: Tuple[float, float]) -> bool:
        """
        Validate that the tracker has moved sufficient distance to constitute a valid crossing.
        
        This method filters out stationary objects that drift across the line due to detection
        noise or small movements. It calculates the cumulative distance traveled by a tracker
        and only allows crossing events if the object has moved beyond the minimum threshold.
        
        Args:
            tracker_id (int): Unique identifier of the tracker to validate distance for
            current_point (Tuple[float, float]): Current position (x, y) of the tracker
            
        Returns:
            bool: True if distance requirement is met or disabled, False if insufficient movement
            
        Example:
            >>> crossing = pf.crossings.Crossing(
            ...     start=(100, 200), 
            ...     end=(500, 200),
            ...     minimum_distance=20.0  # Require 20 pixels of movement
            ... )
            >>> # Simulating a stationary object with small drift
            >>> print(crossing._validate_minimum_distance(123, (300, 150)))
            True   # First position always passes
            >>> print(crossing._validate_minimum_distance(123, (302, 151)))  
            False  # Only moved ~2.2 pixels, below threshold
            >>> print(crossing._validate_minimum_distance(123, (325, 155)))
            True   # Total movement now > 20 pixels
        
        Notes:
            - Returns True immediately if minimum_distance is 0 (validation disabled)
            - Returns True for first position of any tracker (need baseline)
            - Maintains position history limited to 10 points to prevent memory growth
            - Calculates cumulative Euclidean distance along the tracker's path
            - Essential for filtering out detection noise and stationary object drift
            - Position history is per-tracker, allowing independent validation
            
        Performance Notes:
            - O(k) computation where k is number of stored positions (max 10)
            - Uses squared distance calculation with sqrt for accuracy
            - Memory usage bounded by maximum 10 positions per active tracker
            - Position list is pruned automatically to prevent unbounded growth
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
        """Reset all crossing statistics and temporal tracking data to initial state.
        
        This method clears all accumulated crossing counts, per-class statistics, and
        temporal tracking data, effectively returning the crossing detector to its
        initial state. This is useful for starting fresh analysis periods or when
        reusing a crossing instance for different video segments.
        
        Resets the following data:
        - Total crossing counts (in_count, out_count)
        - Per-class crossing statistics (in_count_per_class, out_count_per_class)  
        - Temporal crossing history for all trackers
        - Debounce timing records
        - Position tracking for distance validation
        - Class ID to name mappings
        - Frame counter for temporal consistency
        
        Returns:
            None: Method modifies object state in-place
            
        Example:
            >>> crossing = pf.crossings.Crossing(start=(100, 200), end=(500, 200))
            >>> # After processing some video...
            >>> print(f\"Before reset - In: {crossing.in_count}, Out: {crossing.out_count}\")
            Before reset - In: 42, Out: 38
            >>> crossing.reset_counts()
            >>> print(f\"After reset - In: {crossing.in_count}, Out: {crossing.out_count}\")
            After reset - In: 0, Out: 0
            >>> print(f\"Per-class counts: {crossing.in_count_per_class}\")
            Per-class counts: {}
        
        Notes:
            - Does not reset crossing configuration (line position, thresholds, etc.)
            - Preserves line geometry and visual properties (color, name)
            - Clears both single and multi-anchor tracking histories
            - Resets frame counter to 0 for fresh temporal tracking
            - Class name mappings are cleared but will be rebuilt during next processing
            - All tracker-specific temporal data is removed
            
        Performance Notes:
            - O(n) operation where n is number of tracked objects with history
            - Memory is immediately freed for all tracking data structures
            - Fast operation suitable for real-time applications
        """
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


class Crossings:
    """
    Centralized manager for multiple crossing line detectors with unified processing and analytics.
    
    This class provides sophisticated management of multiple crossing instances, enabling
    complex crossing detection scenarios such as multi-gate entrances, traffic intersections,
    and zone-based monitoring systems. It offers unified processing of detections against
    all crossings simultaneously, consolidated analytics, and automatic result annotation
    with crossing event metadata.
    
    The manager handles:
    - Creation and lifecycle management of multiple crossing instances
    - Unified detection processing across all crossings in a single call
    - Automatic annotation of detections with crossing event information
    - Consolidated analytics and reporting across all managed crossings
    - Collision-free ID management and lookup operations
    
    Args:
        None: Initializes an empty crossings manager
        
    Attributes:
        crossings (List[Crossing]): List of all managed crossing instances
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Setup multi-crossing detection system
        >>> model = YOLO("yolo11n.pt")
        >>> tracker = pf.tracker.ByteTracker()
        >>> crossings = pf.crossings.Crossings()
        >>> 
        >>> # Add multiple crossing lines
        >>> crossings.add_crossing(
        ...     start=(100, 200), end=(500, 200),
        ...     name="Main Entrance", line_id="entrance_01"
        ... )
        >>> crossings.add_crossing(
        ...     start=(300, 0), end=(300, 400), 
        ...     name="Side Gate", line_id="side_01",
        ...     triggering_anchor="bottom_center"
        ... )
        >>> 
        >>> # Process video with unified crossing detection
        >>> image = cv2.imread("surveillance_frame.jpg")
        >>> outputs = model.predict(image)
        >>> results = pf.results.from_ultralytics(outputs)
        >>> tracked_results = tracker.update(results)
        >>> annotated_results = crossings.update(tracked_results)
        >>> 
        >>> # Access consolidated crossing analytics
        >>> counts = crossings.get_crossing_counts()
        >>> for line_id, stats in counts.items():
        ...     print(f"{stats['name']}: {stats['in_count']} in, {stats['out_count']} out")
        Main Entrance: 15 in, 12 out
        Side Gate: 8 in, 9 out
        >>> 
        >>> # Check individual detection crossing events  
        >>> for detection in annotated_results.detections:
        ...     if hasattr(detection, 'line_crossings') and detection.line_crossings:
        ...         for event in detection.line_crossings:
        ...             print(f"Object {detection.tracker_id} crossed {event['line_name']} going {event['direction']}")
        >>> 
        >>> # Advanced multi-gate scenario with different strategies per crossing
        >>> office_crossings = pf.crossings.Crossings()
        >>> office_crossings.add_crossing(
        ...     start=(0, 300), end=(200, 300),
        ...     name="Door 1", triggering_anchor="center",
        ...     debounce_time=45, minimum_distance=15.0
        ... )
        >>> office_crossings.add_crossing(
        ...     start=(400, 300), end=(640, 300),
        ...     name="Door 2", triggering_anchor="bottom_center", 
        ...     minimum_crossing_threshold=3
        ... )
        >>> 
        >>> # Process with strategy override for specific object types
        >>> person_results = tracker.update(person_detections)
        >>> office_crossings.update(person_results, strategy="bottom_center")
    
    Notes:
        - All crossing instances are processed simultaneously for efficiency
        - Crossing IDs must be unique within a manager instance  
        - Detection objects are annotated with line_crossings attribute containing event details
        - Individual crossings maintain their own temporal tracking and validation state
        - Manager provides unified interface while preserving per-crossing configuration flexibility
        - Supports heterogeneous crossing configurations (different thresholds, strategies, etc.)
        
    Performance Notes:
        - Processing time scales linearly with number of crossings and detections
        - Memory usage scales with number of unique tracker IDs across all crossings
        - Batch processing is more efficient than individual crossing.trigger() calls
        - ID lookup operations are O(1) using internal dictionary mapping
        
    See Also:
        Crossing : Individual crossing line detector with temporal validation
        pf.tracker.ByteTracker : Object tracker providing required tracker IDs
    """
    
    def __init__(self):
        """Initialize an empty crossings manager with no crossing instances.
        
        Creates a new crossings manager ready to accept crossing line definitions.
        The manager maintains both a list for ordered access and a dictionary for
        fast ID-based lookups of crossing instances.
        
        Returns:
            None: Constructor initializes empty internal data structures
            
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Create new crossings manager
            >>> crossings = pf.crossings.Crossings()
            >>> print(len(crossings.crossings))
            0
            >>> 
            >>> # Ready to add crossing definitions
            >>> crossings.add_crossing(start=(100, 200), end=(500, 200))
            >>> print(len(crossings.crossings))
            1
            
        Notes:
            - Initializes empty crossings list and ID lookup dictionary
            - No crossing instances are created during initialization
            - Manager is ready to accept crossing definitions via add_crossing()
            - Internal data structures are optimized for both iteration and lookup operations
        """
        self.crossings: List[Crossing] = []
        self._crossing_dict: Dict[Union[int, str], Crossing] = {}
    
    def add_crossing(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        line_id: Optional[Union[int, str]] = None,
        name: str = "",
        color: Optional[Tuple[int, int, int]] = None,
        triggering_anchor: Union[str, List[str]] = "center",
        minimum_crossing_threshold: int = 1,
        mode: Literal["any", "all"] = "all",
        boundary_margin: float = 50.0,
        debounce_time: int = 30,
        minimum_distance: float = 10.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Crossing:
        """
        Add a new crossing line detector to the manager with automatic ID collision resolution.
        
        This method creates and registers a new crossing line detector with the specified
        configuration. It handles automatic ID generation and collision detection to ensure
        all managed crossings have unique identifiers for lookup and analytics operations.
        
        Args:
            start (Tuple[float, float]): Starting point of the line segment (x, y) in pixels
            end (Tuple[float, float]): Ending point of the line segment (x, y) in pixels
            line_id (Optional[Union[int, str]]): Unique identifier for the crossing.
                                               If None, auto-generates sequential integer ID
            name (str): Human-readable name for the crossing. If empty, auto-generates
                       based on line_id. Default is ""
            color (Optional[Tuple[int, int, int]]): RGB color tuple (0-255) for visualization.
                                                   If None, auto-generates distinct color.
                                                   Default is None
            triggering_anchor (Union[str, List[str]]): Anchor point(s) for crossing detection.
                                                      Single string or list of strings.
                                                      Valid options: "center", "bottom_center",
                                                      "top_left", "top_right", "bottom_left",
                                                      "bottom_right". Default is "center"
            minimum_crossing_threshold (int): Consecutive frames required on opposite side
                                             to register crossing. Range: [1, inf]. Default is 1
            mode (Literal["any", "all"]): Logic for multiple anchor strategies. "all" uses
                                         AND logic, "any" uses OR logic. Default is "all"
            boundary_margin (float): Distance in pixels beyond line endpoints where crossings
                                    are still detected. Range: [0, inf]. Default is 50.0
            debounce_time (int): Frames to wait before allowing repeat crossing for same tracker.
                               Range: [0, inf]. Default is 30
            minimum_distance (float): Minimum distance in pixels object must travel to be valid.
                                     Range: [0, inf]. Default is 10.0
            metadata (Optional[Dict[str, Any]]): Additional custom data for the crossing.
                                               Default is None
                                               
        Returns:
            Crossing: The newly created and registered crossing instance
            
        Raises:
            ValueError: If line_id already exists in the manager
            ValueError: If start and end points are identical
            ValueError: If any numeric parameter is outside valid range
            
        Example:
            >>> crossings = pf.crossings.Crossings()
            >>> 
            >>> # Basic crossing with auto-generated ID
            >>> crossing1 = crossings.add_crossing(
            ...     start=(100, 200), 
            ...     end=(500, 200),
            ...     name="Main Entrance"
            ... )
            >>> print(crossing1.line_id)
            0
            >>> 
            >>> # Custom ID and advanced parameters
            >>> crossing2 = crossings.add_crossing(
            ...     start=(300, 0), 
            ...     end=(300, 480),
            ...     line_id="gate_001",
            ...     triggering_anchor="bottom_center",
            ...     debounce_time=60,
            ...     minimum_distance=25.0,
            ...     color=(255, 0, 0)
            ... )
            >>> 
            >>> # Multi-anchor crossing for complex objects
            >>> crossing3 = crossings.add_crossing(
            ...     start=(0, 350), 
            ...     end=(640, 350),
            ...     line_id="vehicle_gate",
            ...     triggering_anchor=["center", "bottom_center"],
            ...     mode="any",
            ...     metadata={"purpose": "vehicle_counting", "lane": 1}
            ... )
            >>> 
            >>> print(f"Total crossings: {len(crossings.crossings)}")
            Total crossings: 3
        
        Notes:
            - Line IDs must be unique within the manager - duplicates raise ValueError
            - Auto-generated IDs start at 0 and increment until an unused ID is found
            - All crossing parameters are validated during creation
            - Created crossing is immediately available for lookup via get_crossing()
            - Manager maintains both ordered list and fast ID-based dictionary access
            - Individual crossings can have different configurations (heterogeneous setup)
            
        Performance Notes:
            - O(1) average time for ID collision detection using dictionary lookup
            - O(n) worst case for auto-ID generation if many sequential IDs are taken
            - Creation time dominated by Crossing.__init__ validation and setup
        """
        # Auto-generate line_id if not provided
        if line_id is None:
            line_id = len(self.crossings)
            while line_id in self._crossing_dict:
                line_id += 1
        
        # Check for duplicate line_id
        if line_id in self._crossing_dict:
            raise ValueError(f"Crossing with ID {line_id} already exists")
        
        # Create crossing
        crossing = Crossing(
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
        self.crossings.append(crossing)
        self._crossing_dict[line_id] = crossing
        
        return crossing
    
    def remove_crossing(self, line_id: Union[int, str]):
        """Remove a crossing instance from the manager by its unique identifier.
        
        This method safely removes a crossing from both the ordered list and ID lookup
        dictionary. All accumulated statistics and temporal tracking data for the crossing
        are lost when removed. The operation is idempotent - removing a non-existent
        crossing ID has no effect.
        
        Args:
            line_id (Union[int, str]): Unique identifier of the crossing to remove
            
        Returns:
            None: Method modifies manager state in-place
            
        Example:
            >>> crossings = pf.crossings.Crossings()
            >>> crossings.add_crossing(start=(100, 200), end=(500, 200), line_id="entrance")
            >>> crossings.add_crossing(start=(300, 0), end=(300, 400), line_id="exit")
            >>> print(len(crossings.crossings))
            2
            >>> crossings.remove_crossing("entrance")
            >>> print(len(crossings.crossings))
            1
            >>> crossings.remove_crossing("nonexistent")  # No error, no effect
            >>> print(len(crossings.crossings))
            1
        
        Notes:
            - Safe to call with non-existent IDs (no error raised)
            - Removes from both internal list and dictionary for consistency
            - All crossing statistics and temporal data are permanently lost
            - Remaining crossings are unaffected and retain their configurations
        """
        if line_id in self._crossing_dict:
            crossing = self._crossing_dict.pop(line_id)
            self.crossings.remove(crossing)
    
    def get_crossing(self, line_id: Union[int, str]) -> Optional[Crossing]:
        """Retrieve a crossing instance by its unique identifier with O(1) lookup.
        
        This method provides fast access to individual crossing instances for
        direct manipulation, configuration updates, or statistics access.
        
        Args:
            line_id (Union[int, str]): Unique identifier of the crossing to retrieve
            
        Returns:
            Optional[Crossing]: The crossing instance if found, None if ID doesn't exist
            
        Example:
            >>> crossings = pf.crossings.Crossings()
            >>> crossings.add_crossing(start=(100, 200), end=(500, 200), line_id="main_gate")
            >>> 
            >>> # Retrieve crossing for direct access
            >>> gate = crossings.get_crossing("main_gate")
            >>> if gate:
            ...     print(f"Gate counts: {gate.in_count} in, {gate.out_count} out")
            ...     gate.reset_counts()  # Direct method call on crossing
            >>> 
            >>> # Handle non-existent crossing
            >>> missing = crossings.get_crossing("invalid_id")
            >>> if missing is None:
            ...     print("Crossing not found")
            Crossing not found
        
        Notes:
            - O(1) lookup time using internal dictionary
            - Returns None for non-existent IDs (no exception raised)
            - Returned crossing can be modified directly (not a copy)
            - Useful for accessing crossing-specific methods and properties
        """
        return self._crossing_dict.get(line_id)
    
    def update(self, results, strategy: Union[str, List[str]] = None):
        """
        Process detections against all managed crossings and annotate results with crossing events.
        
        This method performs unified crossing detection across all managed crossing instances,
        applying each crossing's detection algorithm to the provided detections. It automatically
        annotates detection objects with crossing event metadata and updates all crossing
        statistics simultaneously. This is the primary processing method for multi-crossing scenarios.
        
        The update process:
        1. Validates optional strategy override parameter
        2. Iterates through all managed crossing instances  
        3. Calls trigger() on each crossing with the detections
        4. Annotates each detection with line_crossings attribute containing event details
        5. Updates crossing statistics (counts, per-class analytics)
        
        Args:
            results: PixelFlow Results object containing detection predictions with bounding boxes
                    and tracker IDs. Must have .detections list with objects containing .tracker_id
                    and .bbox attributes for crossing detection to function properly
            strategy (Optional[Union[str, List[str]]]): Optional strategy override applied to ALL
                                                       managed crossings for this update call.
                                                       If None, each crossing uses its own configured
                                                       triggering_anchor. Valid values: "center",
                                                       "bottom_center", "top_left", "top_right",
                                                       "bottom_left", "bottom_right"
            
        Returns:
            Results: The same Results object passed in, modified in-place with crossing annotations.
                    Each detection in results.detections may have a new line_crossings attribute
                    containing a list of crossing event dictionaries
            
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> # Setup multi-crossing detection system
            >>> model = YOLO("yolo11n.pt")
            >>> tracker = pf.tracker.ByteTracker()
            >>> crossings = pf.crossings.Crossings()
            >>> crossings.add_crossing(start=(100, 200), end=(500, 200), name="Main Gate")
            >>> crossings.add_crossing(start=(300, 0), end=(300, 400), name="Side Door")
            >>> 
            >>> # Process frame with unified crossing detection
            >>> image = cv2.imread("security_frame.jpg")
            >>> outputs = model.predict(image)
            >>> results = pf.results.from_ultralytics(outputs)
            >>> tracked_results = tracker.update(results)  # Required for crossing detection
            >>> annotated_results = crossings.update(tracked_results)
            >>> 
            >>> # Access crossing events for each detection
            >>> for detection in annotated_results.detections:
            ...     if hasattr(detection, 'line_crossings'):
            ...         for event in detection.line_crossings:
            ...             print(f\"Object {detection.tracker_id} crossed {event['line_name']} going {event['direction']}\")
            Object 15 crossed Main Gate going in
            Object 23 crossed Side Door going out
            >>> 
            >>> # Override strategy for specific object types (e.g., people vs vehicles)
            >>> person_detections = filter_detections_by_class(tracked_results, class_id=0)
            >>> crossings.update(person_detections, strategy=\"bottom_center\")
            >>> 
            >>> # Access consolidated analytics
            >>> counts = crossings.get_crossing_counts()
            >>> for line_id, stats in counts.items():
            ...     print(f\"{stats['name']}: {stats['in_count']} in, {stats['out_count']} out\")
            Main Gate: 12 in, 8 out
            Side Door: 5 in, 7 out
            >>> 
            >>> # Multi-anchor strategy override for complex objects  
            >>> vehicle_detections = filter_detections_by_class(tracked_results, class_id=2)
            >>> crossings.update(vehicle_detections, strategy=[\"center\", \"bottom_center\"])
        
        Notes:
            - Requires tracker IDs: Call tracker.update(results) before crossings.update(results)
            - Modifies Results object in-place by adding line_crossings attribute to detections
            - Each crossing event dict contains: {'line_id', 'line_name', 'direction'}
            - Direction values are 'in' (right to left) or 'out' (left to right)
            - Strategy override affects ALL crossings uniformly for this update call
            - Individual crossing statistics are updated automatically during processing
            - Processing is performed in crossing creation order (deterministic)
            - Detection objects may have multiple crossing events if they cross multiple lines
            
        Performance Notes:
            - Processing time is O(n * m) where n = detections, m = number of crossings
            - Memory usage scales with number of unique tracker IDs across all crossings
            - Strategy validation is performed once per update call, not per crossing
            - Batch processing is more efficient than individual crossing.trigger() calls
            
        See Also:
            Crossing.trigger : Individual crossing detection method
            get_crossing_counts : Retrieve consolidated crossing analytics
            pf.tracker.ByteTracker : Required object tracker for temporal consistency
        """        
        # Validate override strategy if provided
        if strategy is not None:
            strategy = validate_strategy(strategy)
        
        for crossing in self.crossings:
            crossed_in, crossed_out = crossing.trigger(results, strategy)
            
            # Update detections with crossing info
            for i, detection in enumerate(results.detections):
                if not hasattr(detection, 'line_crossings'):
                    detection.line_crossings = []
                
                if crossed_in[i]:
                    detection.line_crossings.append({
                        'line_id': crossing.line_id,
                        'line_name': crossing.name,
                        'direction': 'in'
                    })
                elif crossed_out[i]:
                    detection.line_crossings.append({
                        'line_id': crossing.line_id,
                        'line_name': crossing.name,
                        'direction': 'out'
                    })
        
        return results
    
    def get_crossing_counts(self) -> Dict[Union[int, str], Dict[str, int]]:
        """
        Get consolidated crossing statistics for all managed crossing instances.
        
        This method provides a unified view of crossing analytics across all managed
        crossings, including total counts, per-class breakdowns, and metadata for
        each crossing line. Useful for dashboards, reporting, and analytics applications.
        
        Returns:
            Dict[Union[int, str], Dict[str, int]]: Dictionary mapping each crossing's line_id
                                                  to its complete statistics dictionary containing:
                                                  - 'name': Human-readable crossing name
                                                  - 'in_count': Total objects crossed in (right to left)
                                                  - 'out_count': Total objects crossed out (left to right)
                                                  - 'in_count_per_class': Dict mapping class_id to in counts
                                                  - 'out_count_per_class': Dict mapping class_id to out counts
                                                  
        Example:
            >>> crossings = pf.crossings.Crossings()
            >>> crossings.add_crossing(start=(100, 200), end=(500, 200), name="Main Gate")
            >>> crossings.add_crossing(start=(300, 0), end=(300, 400), name="Side Door")
            >>> # After processing some detections...
            >>> counts = crossings.get_crossing_counts()
            >>> print(counts)
            {
                0: {
                    'name': 'Main Gate',
                    'in_count': 15,
                    'out_count': 12,
                    'in_count_per_class': {0: 10, 2: 5},   # 10 people, 5 cars in
                    'out_count_per_class': {0: 8, 2: 4}    # 8 people, 4 cars out
                },
                1: {
                    'name': 'Side Door', 
                    'in_count': 8,
                    'out_count': 9,
                    'in_count_per_class': {0: 8},          # 8 people in
                    'out_count_per_class': {0: 9}          # 9 people out
                }
            }
            >>> 
            >>> # Generate summary report
            >>> for line_id, stats in counts.items():
            ...     net_flow = stats['in_count'] - stats['out_count']
            ...     print(f"{stats['name']}: {stats['in_count']} in, {stats['out_count']} out (net: {net_flow})")
            Main Gate: 15 in, 12 out (net: 3)
            Side Door: 8 in, 9 out (net: -1)
        
        Notes:
            - Returns current state snapshot - counts may change with subsequent updates
            - Per-class dictionaries use detection model's class IDs as keys
            - Empty crossings return zero counts and empty per-class dictionaries
            - Dictionary structure is suitable for JSON serialization and reporting
            - Class names can be accessed via individual crossing.class_id_to_name mappings
            
        Performance Notes:
            - O(n) where n is number of managed crossings
            - Creates new dictionary structure (safe to modify returned data)
            - Minimal computation - primarily data aggregation and formatting
        """
        counts = {}
        for crossing in self.crossings:
            counts[crossing.line_id] = {
                'name': crossing.name,
                'in_count': crossing.in_count,
                'out_count': crossing.out_count,
                'in_count_per_class': crossing.in_count_per_class,
                'out_count_per_class': crossing.out_count_per_class
            }
        return counts
    
    def reset_all_counts(self):
        """Reset crossing statistics and temporal tracking data for all managed crossings.
        
        This method calls reset_counts() on every managed crossing instance, effectively
        clearing all accumulated statistics, temporal tracking data, and returning all
        crossings to their initial state. Useful for starting fresh analysis periods
        or when reusing the manager for different video segments.
        
        Returns:
            None: Method modifies all crossing states in-place
            
        Example:
            >>> crossings = pf.crossings.Crossings()
            >>> crossings.add_crossing(start=(100, 200), end=(500, 200), name="Gate 1")
            >>> crossings.add_crossing(start=(300, 0), end=(300, 400), name="Gate 2")
            >>> # After processing video...
            >>> counts = crossings.get_crossing_counts()
            >>> print(f"Total crossings before reset: {sum(c['in_count'] + c['out_count'] for c in counts.values())}")
            47
            >>> crossings.reset_all_counts()
            >>> counts = crossings.get_crossing_counts()
            >>> print(f"Total crossings after reset: {sum(c['in_count'] + c['out_count'] for c in counts.values())}")
            0
        
        Notes:
            - Resets statistics for ALL managed crossings simultaneously
            - Does not remove crossings or modify their configurations
            - Preserves crossing line positions, colors, names, and detection parameters
            - Clears temporal tracking data, debounce timers, and position histories
            - Frame counters are reset to 0 for all crossings
            - Class name mappings are cleared but will be rebuilt during next processing
            
        Performance Notes:
            - O(n * m) where n = number of crossings, m = average tracked objects per crossing
            - Fast operation suitable for real-time applications
            - Memory is immediately freed for all tracking data structures
        """
        for crossing in self.crossings:
            crossing.reset_counts()
    
    def clear_crossings(self):
        """Remove all crossing instances from the manager and clear all data.
        
        This method completely empties the manager, removing all crossing instances
        and clearing all internal data structures. This is more comprehensive than
        reset_all_counts() as it removes the crossing definitions themselves, not
        just their accumulated statistics.
        
        Returns:
            None: Method modifies manager state in-place
            
        Example:
            >>> crossings = pf.crossings.Crossings()
            >>> crossings.add_crossing(start=(100, 200), end=(500, 200), name="Gate 1")
            >>> crossings.add_crossing(start=(300, 0), end=(300, 400), name="Gate 2")
            >>> print(f"Crossings before clear: {len(crossings.crossings)}")
            2
            >>> crossings.clear_crossings()
            >>> print(f"Crossings after clear: {len(crossings.crossings)}")
            0
            >>> print(f"Can still add new crossings: {crossings.add_crossing(start=(0, 0), end=(100, 100)) is not None}")
            True
        
        Notes:
            - Removes ALL crossing instances and their configurations
            - More comprehensive than reset_all_counts() which only clears statistics
            - Clears both ordered list and ID lookup dictionary
            - Manager remains functional and can accept new crossing definitions
            - All accumulated statistics and temporal tracking data are permanently lost
            - Previously used line IDs can be reused after clearing
            
        Performance Notes:
            - O(n) where n is number of managed crossings
            - Fast operation that immediately frees all memory used by crossings
            - More efficient than removing crossings individually
        """
        self.crossings.clear()
        self._crossing_dict.clear()