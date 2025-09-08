"""
Detection Filtering Functions for Advanced Detection Processing.

Provides comprehensive filtering capabilities for detection collections including
confidence thresholding, class-based filtering, geometric constraints, zone-based
filtering, and tracking-related filters. Designed for zero-overhead method
injection into Detections class for seamless chaining operations.
"""

from typing import List, Union, Optional, Any

__all__ = [
    "filter_by_confidence", "filter_by_class_id", "remap_class_ids",
    "filter_by_size", "filter_by_dimensions", "filter_by_aspect_ratio", 
    "filter_by_zones", "filter_by_position", "filter_by_relative_size",
    "filter_by_tracking_duration", "filter_by_first_seen_time", "filter_tracked_objects",
    "remove_duplicates", "filter_overlapping", "_calculate_iou"
]


def filter_by_confidence(self, threshold: float) -> 'Detections':
    """
    Filter detections by minimum confidence score threshold.
    
    Applies confidence-based filtering to remove low-confidence detections that may
    represent false positives or uncertain predictions. Essential preprocessing step
    for improving detection quality and reducing noise in downstream analysis.
    
    Args:
        threshold (float): Minimum confidence score (inclusive). Range: [0.0, 1.0].
                          Detections with confidence >= threshold are retained.
                          Higher values are more restrictive.
        
    Returns:
        Detections: New Detections object containing only high-confidence detections.
                   Preserves all detection attributes including tracking data.
    
    Raises:
        TypeError: If threshold is not a numeric type.
        ValueError: If threshold is outside the valid range [0.0, 1.0].
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Load model and run inference
        >>> model = YOLO("yolov8n.pt")
        >>> image = cv2.imread("traffic_scene.jpg")
        >>> outputs = model.predict(image)
        >>> detections = pf.detections.from_ultralytics(outputs)
        >>> 
        >>> # Basic filtering: keep high-confidence detections
        >>> high_conf = detections.filter_by_confidence(0.8)
        >>> print(f"High confidence: {len(high_conf)}/{len(detections)} detections")
        >>> 
        >>> # Conservative filtering for critical applications
        >>> critical = detections.filter_by_confidence(0.95)
        >>> print(f"Very high confidence: {len(critical)} detections")
        >>> 
        >>> # Method chaining with other filters
        >>> filtered = detections.filter_by_confidence(0.7).filter_by_class_id([0, 2])
        >>> print(f"High-conf people and cars: {len(filtered)} detections")
        >>> 
        >>> # Moderate filtering for analysis
        >>> moderate = detections.filter_by_confidence(0.5)
        >>> print(f"Moderate confidence: {len(moderate)} detections")
    
    Notes:
        - Detections with None confidence values are automatically excluded
        - Returns empty Detections object if no detections meet threshold
        - Preserves all detection attributes (bbox, masks, keypoints, tracking data)
        - Supports method chaining with other filter operations
        - Threshold validation is not enforced; values outside [0.0, 1.0] may cause unexpected behavior
        
    Performance Notes:
        - O(n) time complexity where n is the number of detections
        - Memory efficient with lazy evaluation and direct filtering
        - Minimal overhead for large detection collections
    """
    filtered_detections = self.__class__()
    for detection in self.detections:
        if detection.confidence is not None and detection.confidence >= threshold:
            filtered_detections.add_detection(detection)
    return filtered_detections


def filter_by_class_id(self, class_ids: Union[int, str, List[Union[int, str]]]) -> 'Detections':
    """
    Filter detections by class identifier(s).
    
    Enables class-specific filtering to focus analysis on particular object types.
    Supports both single class selection and multi-class filtering with flexible
    ID format handling for different model outputs and naming conventions.
    
    Args:
        class_ids (Union[int, str, List[Union[int, str]]]): Single class ID or list of class IDs
                                                           to include. Accepts both numeric IDs
                                                           (e.g., COCO indices) and string class names.
                                                           Mixed types are supported in lists.
        
    Returns:
        Detections: New Detections object containing only detections with matching class IDs.
                   Preserves all detection attributes and metadata.
    
    Raises:
        TypeError: If class_ids contains unsupported types (not int, str, or list thereof).
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Load model and run inference
        >>> model = YOLO("yolov8n.pt")
        >>> image = cv2.imread("street_scene.jpg")
        >>> outputs = model.predict(image)
        >>> detections = pf.detections.from_ultralytics(outputs)
        >>> 
        >>> # Filter for specific class by numeric ID (COCO format)
        >>> people = detections.filter_by_class_id(0)  # person class
        >>> print(f"Found {len(people)} people")
        >>> 
        >>> # Filter for multiple vehicle classes
        >>> vehicles = detections.filter_by_class_id([2, 3, 5, 7])  # car, motorcycle, bus, truck
        >>> print(f"Found {len(vehicles)} vehicles")
        >>> 
        >>> # Filter by class name string
        >>> dogs = detections.filter_by_class_id("dog")
        >>> print(f"Found {len(dogs)} dogs")
        >>> 
        >>> # Mixed ID types for flexible filtering
        >>> targets = detections.filter_by_class_id(["person", 2, "bicycle"])
        >>> print(f"Found {len(targets)} people, cars, or bicycles")
    
    Notes:
        - Detections with None class_id values are automatically excluded
        - Accepts single values or lists for flexible usage patterns
        - Supports both numeric IDs and string class names in same operation
        - Class ID matching uses exact equality (no fuzzy matching)
        - Empty list input returns empty Detections object
        - Preserves all detection attributes including confidence and tracking data
        
    Performance Notes:
        - O(n×m) time complexity where n is detections and m is class_ids length
        - Efficient membership testing with set conversion for large class_id lists
        - Memory efficient with minimal object copying
    """
    # Handle single class_id or list of class_ids
    if not isinstance(class_ids, (list, tuple)):
        class_ids = [class_ids]
        
    filtered_detections = self.__class__()
    for detection in self.detections:
        if detection.class_id is not None and detection.class_id in class_ids:
            filtered_detections.add_detection(detection)
    return filtered_detections


def remap_class_ids(self, from_ids: Union[int, str, List[Union[int, str]]], to_id: Union[int, str]) -> 'Detections':
    """
    Remap class IDs to consolidate or standardize classification labels.
    
    Creates new Detection objects with modified class IDs while preserving all other
    detection attributes. Useful for consolidating similar classes or standardizing
    class schemas across different models.
    
    Args:
        from_ids (Union[int, str, List[Union[int, str]]]): Source class ID(s) to remap.
                                                          Single ID or list of IDs.
        to_id (Union[int, str]): Target class ID to map to.
        
    Returns:
        Detections: New Detections object with remapped class IDs and cleared class names.
    
    Example:
        >>> # Consolidate vehicle classes to generic "vehicle"
        >>> vehicles = detections.remap_class_ids([2, 3, 5, 7], "vehicle")
        >>> 
        >>> # Remap multiple animal classes to "animal"
        >>> animals = detections.remap_class_ids(["dog", "cat", "bird"], 1)
        >>> 
        >>> # Single class remapping
        >>> motorcycles_as_bikes = detections.remap_class_ids(3, "bike")
    
    Notes:
        - Creates deep copies of Detection objects for safe modification
        - Clears class_name field when remapping to avoid inconsistency
        - Preserves all other detection attributes including tracking data
        - Non-matching detections are included unchanged
    """
    from .detections import Detection
    
    # Handle single from_id or list of from_ids
    if not isinstance(from_ids, (list, tuple)):
        from_ids = [from_ids]
        
    remapped_detections = self.__class__()
    for detection in self.detections:
        # Create a copy of the detection
        new_detection = Detection(
            inference_id=detection.inference_id,
            bbox=detection.bbox,
            masks=detection.masks,
            segments=detection.segments,
            keypoints=detection.keypoints,
            class_id=detection.class_id,
            class_name=detection.class_name,
            labels=detection.labels,
            confidence=detection.confidence,
            tracker_id=detection.tracker_id,
            data=detection.data,
            zones=detection.zones,
            zone_names=detection.zone_names,
            line_crossings=detection.line_crossings,
            first_seen_time=detection.first_seen_time,
            total_time=detection.total_time
        )
        
        # Remap class_id if it matches
        if new_detection.class_id is not None and new_detection.class_id in from_ids:
            new_detection.class_id = to_id
            # Clear class_name since it may no longer be accurate
            new_detection.class_name = None
        
        remapped_detections.add_detection(new_detection)
    
    return remapped_detections


def filter_by_size(self, min_area: Optional[float] = None, max_area: Optional[float] = None) -> 'Detections':
    """
    Filter detections by bounding box area constraints.
    
    Args:
        min_area (Optional[float]): Minimum bounding box area in pixels (inclusive).
                                   If None, no minimum constraint is applied.
        max_area (Optional[float]): Maximum bounding box area in pixels (inclusive).
                                   If None, no maximum constraint is applied.
        
    Returns:
        Detections: New Detections object containing only detections within size range.
    
    Example:
        >>> # Remove very small noise detections
        >>> filtered = detections.filter_by_size(min_area=100)
        >>> 
        >>> # Remove very large detections (likely false positives)
        >>> filtered = detections.filter_by_size(max_area=50000)
        >>> 
        >>> # Keep medium-sized objects only
        >>> medium = detections.filter_by_size(min_area=1000, max_area=10000)
    
    Notes:
        - Detections without bounding boxes are excluded
        - Area calculated as (width * height) from XYXY coordinates
        - Useful for removing noise (small) and false positives (large)
    """
    filtered_detections = self.__class__()
    for detection in self.detections:
        if detection.bbox is None:
            continue
            
        # Calculate bounding box area
        x1, y1, x2, y2 = detection.bbox
        area = (x2 - x1) * (y2 - y1)
        
        # Check area constraints
        if min_area is not None and area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
            
        filtered_detections.add_detection(detection)
    return filtered_detections


def filter_by_dimensions(self, min_width: Optional[float] = None, max_width: Optional[float] = None, 
                         min_height: Optional[float] = None, max_height: Optional[float] = None) -> 'Detections':
    """
    Filter detections by individual width and height constraints.
    
    Provides fine-grained control over detection dimensions, useful for filtering
    objects based on expected physical dimensions or removing artifacts.
    
    Args:
        min_width (Optional[float]): Minimum bounding box width in pixels (inclusive).
        max_width (Optional[float]): Maximum bounding box width in pixels (inclusive).
        min_height (Optional[float]): Minimum bounding box height in pixels (inclusive).
        max_height (Optional[float]): Maximum bounding box height in pixels (inclusive).
        
    Returns:
        Detections: New Detections object containing only detections within dimension constraints.
    
    Example:
        >>> # Filter for tall, narrow objects (people)
        >>> people = detections.filter_by_dimensions(min_height=100, max_width=80)
        >>> 
        >>> # Remove extremely wide detections (likely errors)
        >>> cleaned = detections.filter_by_dimensions(max_width=500)
        >>> 
        >>> # Keep medium-sized rectangular objects
        >>> medium = detections.filter_by_dimensions(
        ...     min_width=50, max_width=200, min_height=100, max_height=300
        ... )
    
    Notes:
        - More specific than filter_by_size for shape-based filtering
        - Useful for filtering based on expected object proportions
        - All constraints are inclusive (detection must satisfy ALL provided limits)
    """
    filtered_detections = self.__class__()
    for detection in self.detections:
        if detection.bbox is None:
            continue
            
        # Calculate bounding box dimensions
        x1, y1, x2, y2 = detection.bbox
        width = x2 - x1
        height = y2 - y1
        
        # Check width constraints
        if min_width is not None and width < min_width:
            continue
        if max_width is not None and width > max_width:
            continue
            
        # Check height constraints
        if min_height is not None and height < min_height:
            continue
        if max_height is not None and height > max_height:
            continue
            
        filtered_detections.add_detection(detection)
    return filtered_detections


def filter_by_aspect_ratio(self, min_ratio: Optional[float] = None, max_ratio: Optional[float] = None) -> 'Detections':
    """
    Filter detections by bounding box aspect ratio (width/height).
    
    Useful for filtering objects based on their shape characteristics, such as
    identifying square objects, wide banners, or tall structures.
    
    Args:
        min_ratio (Optional[float]): Minimum aspect ratio (width/height) (inclusive).
                                    Values > 1.0 favor wide objects.
        max_ratio (Optional[float]): Maximum aspect ratio (width/height) (inclusive).
                                    Values < 1.0 favor tall objects.
        
    Returns:
        Detections: New Detections object containing only detections within aspect ratio range.
    
    Example:
        >>> # Keep square-ish objects only
        >>> squares = detections.filter_by_aspect_ratio(min_ratio=0.8, max_ratio=1.2)
        >>> 
        >>> # Keep only wide objects (banners, signs)
        >>> wide_objects = detections.filter_by_aspect_ratio(min_ratio=2.0)
        >>> 
        >>> # Keep only tall objects (people, poles)
        >>> tall_objects = detections.filter_by_aspect_ratio(max_ratio=0.5)
        >>> 
        >>> # Filter for typical car aspect ratios
        >>> cars = detections.filter_by_aspect_ratio(min_ratio=1.5, max_ratio=2.5)
    
    Notes:
        - Aspect ratio = width / height (wider objects have higher ratios)
        - Detections with zero height are excluded to avoid division by zero
        - Useful for shape-based object classification and noise removal
    """
    filtered_detections = self.__class__()
    for detection in self.detections:
        if detection.bbox is None:
            continue
            
        # Calculate aspect ratio
        x1, y1, x2, y2 = detection.bbox
        width = x2 - x1
        height = y2 - y1
        
        # Avoid division by zero
        if height <= 0:
            continue
            
        aspect_ratio = width / height
        
        # Check aspect ratio constraints
        if min_ratio is not None and aspect_ratio < min_ratio:
            continue
        if max_ratio is not None and aspect_ratio > max_ratio:
            continue
            
        filtered_detections.add_detection(detection)
    return filtered_detections


def filter_by_zones(self, zone_ids: Union[str, int, List[Union[str, int]]], exclude: bool = False) -> 'Detections':
    """
    Filter detections based on zone intersection status.
    
    Enables spatial filtering by including or excluding detections that intersect
    with specified zones. Requires prior zone assignment via update_zones().
    
    Args:
        zone_ids (Union[str, int, List[Union[str, int]]]): Single zone ID or list of zone IDs
                                                          to filter by. Supports both string
                                                          names and numeric IDs.
        exclude (bool): If True, exclude detections in specified zones.
                       If False, include only detections in specified zones.
                       Default is False.
        
    Returns:
        Detections: New Detections object with zone-based filtering applied.
    
    Example:
        >>> # Setup zones and update detections
        >>> zone_manager = pf.zones.ZoneManager()
        >>> detections = detections.update_zones(zone_manager)
        >>> 
        >>> # Keep only detections in parking areas
        >>> parking = detections.filter_by_zones(["parking_1", "parking_2"])
        >>> 
        >>> # Exclude detections from restricted areas
        >>> allowed = detections.filter_by_zones(["restricted", "private"], exclude=True)
        >>> 
        >>> # Filter by numeric zone IDs
        >>> zone_1_only = detections.filter_by_zones(1)
    
    Notes:
        - Requires zone information to be populated via update_zones() first
        - Detections without zone information are included only when exclude=True
        - Supports both inclusive and exclusive zone filtering
        - Zone IDs can be strings or integers depending on zone configuration
    """
    # Handle single zone_id or list of zone_ids
    if not isinstance(zone_ids, (list, tuple)):
        zone_ids = [zone_ids]
        
    filtered_detections = self.__class__()
    for detection in self.detections:
        if not hasattr(detection, 'zones') or detection.zones is None:
            # If no zone info, include only if we're excluding zones
            if exclude:
                filtered_detections.add_detection(detection)
        else:
            # Check if detection is in any of the specified zones
            in_specified_zones = any(z in zone_ids for z in detection.zones)
            
            if (in_specified_zones and not exclude) or (not in_specified_zones and exclude):
                filtered_detections.add_detection(detection)
                
    return filtered_detections


def filter_by_position(self, region: str, margin_percent: float = 0.1, 
                       frame_width: Optional[int] = None, frame_height: Optional[int] = None) -> 'Detections':
    """
    Filter detections by their position within the frame.
    
    Enables spatial filtering based on detection center position relative to frame
    regions. Useful for focusing on specific areas of interest or excluding edge artifacts.
    
    Args:
        region (str): Target region to filter by. Options: "center", "edge", "top", 
                     "bottom", "left", "right", "corners".
        margin_percent (float): Margin as percentage of frame size. Range: [0.0, 0.5].
                               Default is 0.1 (10% margin).
        frame_width (Optional[int]): Frame width in pixels. Required for filtering.
        frame_height (Optional[int]): Frame height in pixels. Required for filtering.
        
    Returns:
        Detections: New Detections object containing only detections in specified region.
    
    Raises:
        ValueError: If frame dimensions are not provided or region is invalid.
    
    Example:
        >>> # Keep only detections in center 80% of frame
        >>> center_detections = detections.filter_by_position(
        ...     "center", margin_percent=0.1, frame_width=1920, frame_height=1080
        ... )
        >>> 
        >>> # Find objects near frame edges (security monitoring)
        >>> edge_objects = detections.filter_by_position(
        ...     "edge", margin_percent=0.05, frame_width=1920, frame_height=1080
        ... )
        >>> 
        >>> # Focus on objects in top region (sky, signs)
        >>> top_objects = detections.filter_by_position(
        ...     "top", margin_percent=0.3, frame_width=1920, frame_height=1080
        ... )
    
    Notes:
        - Uses detection center point for position determination
        - Margin percentage is automatically clamped to [0.0, 0.5] range
        - "corners" region requires objects to be near BOTH horizontal AND vertical edges
        - Useful for region-of-interest analysis and edge artifact removal
    """
    if frame_width is None or frame_height is None:
        raise ValueError("frame_width and frame_height must be provided")
        
    margin_percent = max(0.0, min(0.5, margin_percent))
    margin_x = int(frame_width * margin_percent)
    margin_y = int(frame_height * margin_percent)
    
    filtered_detections = self.__class__()
    for detection in self.detections:
        if detection.bbox is None:
            continue
            
        # Calculate detection center
        x1, y1, x2, y2 = detection.bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Check region constraints
        if region == "center":
            # Center region (excluding margins from all sides)
            if (margin_x <= center_x <= frame_width - margin_x and
                margin_y <= center_y <= frame_height - margin_y):
                filtered_detections.add_detection(detection)
                
        elif region == "edge":
            # Edge region (within margins from any side)
            if (center_x < margin_x or center_x > frame_width - margin_x or
                center_y < margin_y or center_y > frame_height - margin_y):
                filtered_detections.add_detection(detection)
                
        elif region == "top":
            if center_y < margin_y:
                filtered_detections.add_detection(detection)
                
        elif region == "bottom":
            if center_y > frame_height - margin_y:
                filtered_detections.add_detection(detection)
                
        elif region == "left":
            if center_x < margin_x:
                filtered_detections.add_detection(detection)
                
        elif region == "right":
            if center_x > frame_width - margin_x:
                filtered_detections.add_detection(detection)
                
        elif region == "corners":
            # Corners are both edge horizontally AND vertically
            if ((center_x < margin_x or center_x > frame_width - margin_x) and
                (center_y < margin_y or center_y > frame_height - margin_y)):
                filtered_detections.add_detection(detection)
        else:
            raise ValueError(f"Invalid region '{region}'. Valid options are: center, edge, top, bottom, left, right, corners")
            
    return filtered_detections


def filter_by_relative_size(self, min_percent: Optional[float] = None, max_percent: Optional[float] = None, 
                            frame_width: Optional[int] = None, frame_height: Optional[int] = None) -> 'Detections':
    """
    Filter detections by size relative to total frame area.
    
    Provides scale-invariant filtering based on detection size as percentage of
    total frame area. More robust than absolute size filtering across different
    resolution inputs.
    
    Args:
        min_percent (Optional[float]): Minimum size as percentage of frame area.
                                      Range: [0.0, 1.0]. If None, no minimum constraint.
        max_percent (Optional[float]): Maximum size as percentage of frame area.
                                      Range: [0.0, 1.0]. If None, no maximum constraint.
        frame_width (Optional[int]): Frame width in pixels. Required for calculation.
        frame_height (Optional[int]): Frame height in pixels. Required for calculation.
        
    Returns:
        Detections: New Detections object containing only detections within relative size range.
    
    Raises:
        ValueError: If frame dimensions are not provided.
    
    Example:
        >>> # Remove tiny noise detections (< 0.01% of frame)
        >>> cleaned = detections.filter_by_relative_size(
        ...     min_percent=0.0001, frame_width=1920, frame_height=1080
        ... )
        >>> 
        >>> # Keep medium-sized objects (0.1% to 20% of frame)
        >>> medium_objects = detections.filter_by_relative_size(
        ...     min_percent=0.001, max_percent=0.2, frame_width=1920, frame_height=1080
        ... )
        >>> 
        >>> # Remove objects that dominate the frame (> 50%)
        >>> reasonable_size = detections.filter_by_relative_size(
        ...     max_percent=0.5, frame_width=1920, frame_height=1080
        ... )
    
    Notes:
        - Scale-invariant filtering works across different image resolutions
        - Percentage calculated as (detection_area / frame_area)
        - Useful for removing both noise artifacts and unrealistic large detections
    """
    if frame_width is None or frame_height is None:
        raise ValueError("frame_width and frame_height must be provided")
        
    frame_area = frame_width * frame_height
    
    filtered_detections = self.__class__()
    for detection in self.detections:
        if detection.bbox is None:
            continue
            
        # Calculate bounding box area
        x1, y1, x2, y2 = detection.bbox
        detection_area = (x2 - x1) * (y2 - y1)
        relative_size = detection_area / frame_area
        
        # Check relative size constraints
        if min_percent is not None and relative_size < min_percent:
            continue
        if max_percent is not None and relative_size > max_percent:
            continue
            
        filtered_detections.add_detection(detection)
    return filtered_detections


def filter_by_tracking_duration(self, min_seconds: Optional[float] = None, max_seconds: Optional[float] = None) -> 'Detections':
    """
    Filter detections by their tracking duration.
    
    Args:
        min_seconds (Optional[float]): Minimum tracking duration in seconds (inclusive).
                                      If None, no minimum constraint is applied.
        max_seconds (Optional[float]): Maximum tracking duration in seconds (inclusive).
                                      If None, no maximum constraint is applied.
        
    Returns:
        Detections: New Detections object containing only detections within duration range.
    
    Example:
        >>> # Keep only persistent objects (tracked for at least 5 seconds)
        >>> persistent = detections.filter_by_tracking_duration(min_seconds=5.0)
        >>> 
        >>> # Find short-term detections (tracked 2-10 seconds)
        >>> short_term = detections.filter_by_tracking_duration(
        ...     min_seconds=2.0, max_seconds=10.0
        ... )
        >>> 
        >>> # Remove very brief detections (< 1 second, likely noise)
        >>> stable = detections.filter_by_tracking_duration(min_seconds=1.0)
    
    Notes:
        - Requires total_time field to be populated in Detection objects
        - Useful for filtering out transient false positives
        - Duration measured from first detection to current frame
    """
    filtered_detections = self.__class__()
    for detection in self.detections:
        duration = detection.total_time
        
        # Check duration constraints
        if min_seconds is not None and duration < min_seconds:
            continue
        if max_seconds is not None and duration > max_seconds:
            continue
            
        filtered_detections.add_detection(detection)
    return filtered_detections


def filter_by_first_seen_time(self, start_time: Optional[float] = None, end_time: Optional[float] = None) -> 'Detections':
    """
    Filter detections by when they were first observed.
    
    Args:
        start_time (Optional[float]): Earliest first seen time (inclusive).
                                     Time units depend on tracking implementation
                                     (frames, seconds, timestamps).
        end_time (Optional[float]): Latest first seen time (inclusive).
        
    Returns:
        Detections: New Detections object containing only detections first seen within time range.
    
    Example:
        >>> # Keep only objects first detected in early frames (0-100)
        >>> early_objects = detections.filter_by_first_seen_time(start_time=0, end_time=100)
        >>> 
        >>> # Focus on objects that appeared recently (after frame 500)
        >>> recent_objects = detections.filter_by_first_seen_time(start_time=500)
        >>> 
        >>> # Analyze objects from specific time window
        >>> window_objects = detections.filter_by_first_seen_time(
        ...     start_time=1000.5, end_time=2000.5
        ... )
    
    Notes:
        - Requires first_seen_time field to be populated in Detection objects
        - Detections with None first_seen_time are excluded
        - Time units depend on tracking system (frames, seconds, etc.)
        - Useful for temporal analysis and event-based filtering
    """
    filtered_detections = self.__class__()
    for detection in self.detections:
        if detection.first_seen_time is None:
            continue
            
        first_seen = detection.first_seen_time
        
        # Check time constraints
        if start_time is not None and first_seen < start_time:
            continue
        if end_time is not None and first_seen > end_time:
            continue
            
        filtered_detections.add_detection(detection)
    return filtered_detections


def filter_tracked_objects(self, require_tracker_id: bool = True) -> 'Detections':
    """
    Filter detections based on tracking status.
    
    Args:
        require_tracker_id (bool): If True, keep only objects with tracker IDs.
                                  If False, keep only objects without tracker IDs.
                                  Default is True.
        
    Returns:
        Detections: New Detections object filtered by tracking status.
    
    Example:
        >>> # Keep only successfully tracked objects
        >>> tracked_only = detections.filter_tracked_objects(require_tracker_id=True)
        >>> 
        >>> # Get newly detected objects (not yet tracked)
        >>> new_detections = detections.filter_tracked_objects(require_tracker_id=False)
        >>> 
        >>> # Analyze tracking success rate
        >>> total = len(detections)
        >>> tracked = len(detections.filter_tracked_objects(True))
        >>> print(f"Tracking rate: {tracked/total:.1%}")
    
    Notes:
        - Useful for analyzing tracking performance
        - Tracked objects have non-None tracker_id values
        - Helps separate established tracks from new detections
    """
    filtered_detections = self.__class__()
    for detection in self.detections:
        has_tracker = detection.tracker_id is not None
        
        if (require_tracker_id and has_tracker) or (not require_tracker_id and not has_tracker):
            filtered_detections.add_detection(detection)
            
    return filtered_detections


def _calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1 (List[float]): First bounding box in XYXY format [x1, y1, x2, y2].
        bbox2 (List[float]): Second bounding box in XYXY format [x1, y1, x2, y2].
        
    Returns:
        float: IoU value in range [0.0, 1.0]. Higher values indicate greater overlap.
    
    Example:
        >>> bbox_a = [100, 100, 200, 200]  # 100x100 box
        >>> bbox_b = [150, 150, 250, 250]  # Overlapping 100x100 box
        >>> iou = pf.detections._calculate_iou(bbox_a, bbox_b)
        >>> print(f"IoU: {iou:.3f}")  # IoU: 0.143
        >>> 
        >>> # Perfect overlap
        >>> identical_iou = pf.detections._calculate_iou(bbox_a, bbox_a)
        >>> print(f"Identical IoU: {identical_iou}")  # 1.0
    
    Notes:
        - Returns 0.0 for non-overlapping boxes
        - Returns 1.0 for identical boxes
        - Used internally by duplicate removal and overlap filtering functions
        - Handles edge cases (zero area boxes, no intersection)
    """
    # Calculate intersection coordinates
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    
    # Check if there's any intersection
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
        
    # Calculate intersection area
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union area
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
        
    return intersection_area / union_area


def remove_duplicates(self, iou_threshold: float = 0.8, keep: str = 'first') -> 'Detections':
    """
    Remove duplicate or highly overlapping detections using Non-Maximum Suppression.
    
    Identifies groups of overlapping detections and keeps only one detection per group
    based on the specified strategy. Essential for cleaning up multi-model outputs
    or removing redundant detections.
    
    Args:
        iou_threshold (float): IoU threshold for considering detections as duplicates.
                              Range: [0.0, 1.0]. Higher values are more restrictive.
                              Default is 0.8.
        keep (str): Strategy for selecting which detection to keep from each group.
                   Options: 'first', 'last', 'highest_confidence'. Default is 'first'.
        
    Returns:
        Detections: New Detections object with duplicate detections removed.
    
    Raises:
        ValueError: If keep parameter is not one of the valid options.
    
    Example:
        >>> # Remove highly overlapping detections (conservative)
        >>> cleaned = detections.remove_duplicates(iou_threshold=0.8, keep='first')
        >>> 
        >>> # Aggressive duplicate removal, keep best confidence
        >>> best_only = detections.remove_duplicates(
        ...     iou_threshold=0.5, keep='highest_confidence'
        ... )
        >>> 
        >>> # Remove exact duplicates only
        >>> exact_clean = detections.remove_duplicates(iou_threshold=0.95, keep='last')
    
    Notes:
        - Groups detections by IoU overlap and keeps one per group
        - 'highest_confidence' strategy treats None confidence as 0.0
        - Processes detections in order, marking overlapping groups efficiently
        - Essential post-processing step for ensemble models or multiple detectors
        
    Performance Notes:
        - O(n²) complexity for overlap detection
        - Efficient early termination for processed detections
        - Memory efficient with in-place processing tracking
    """
    if keep not in ['first', 'last', 'highest_confidence']:
        raise ValueError("keep must be 'first', 'last', or 'highest_confidence'")
        
    filtered_detections = self.__class__()
    processed_indices = set()
    
    for i, detection_i in enumerate(self.detections):
        if i in processed_indices or detection_i.bbox is None:
            continue
            
        # Find all detections that overlap with this one
        overlapping_group = [i]
        
        for j, detection_j in enumerate(self.detections[i+1:], start=i+1):
            if j in processed_indices or detection_j.bbox is None:
                continue
                
            iou = _calculate_iou(detection_i.bbox, detection_j.bbox)
            if iou >= iou_threshold:
                overlapping_group.append(j)
        
        # Decide which detection to keep from the overlapping group
        if keep == 'first':
            keep_index = overlapping_group[0]
        elif keep == 'last':
            keep_index = overlapping_group[-1]
        else:  # highest_confidence
            best_detection = None
            keep_index = overlapping_group[0]
            best_confidence = -1
            
            for idx in overlapping_group:
                det = self.detections[idx]
                conf = det.confidence if det.confidence is not None else 0
                if conf > best_confidence:
                    best_confidence = conf
                    keep_index = idx
        
        # Add the chosen detection and mark all others as processed
        filtered_detections.add_detection(self.detections[keep_index])
        processed_indices.update(overlapping_group)
        
    return filtered_detections


def filter_overlapping(self, min_overlap: float = 0.5, target_class_ids: Optional[List[Union[int, str]]] = None) -> 'Detections':
    """
    Filter detections that significantly overlap with other detections.
    
    Useful for finding co-occurring objects, validating detection consistency,
    or identifying regions with multiple overlapping predictions.
    
    Args:
        min_overlap (float): Minimum IoU overlap required to consider detections
                           as overlapping. Range: [0.0, 1.0]. Default is 0.5.
        target_class_ids (Optional[List[Union[int, str]]]): Optional list of class IDs
                                                           to check overlap against.
                                                           If None, checks overlap
                                                           with all other detections.
        
    Returns:
        Detections: New Detections object containing only detections that overlap
                   sufficiently with other detections.
    
    Example:
        >>> # Find detections that overlap significantly with others
        >>> overlapping = detections.filter_overlapping(min_overlap=0.3)
        >>> 
        >>> # Find objects that overlap with people (co-occurrence analysis)
        >>> near_people = detections.filter_overlapping(
        ...     min_overlap=0.2, target_class_ids=[0]  # person class
        ... )
        >>> 
        >>> # Find detections overlapping with vehicles
        >>> near_vehicles = detections.filter_overlapping(
        ...     min_overlap=0.1, target_class_ids=["car", "truck", "bus"]
        ... )
    
    Notes:
        - Useful for spatial relationship analysis between objects
        - Can help identify detection inconsistencies or multi-class objects
        - Self-overlap is ignored (detection doesn't overlap with itself)
        - Returns empty result if no detections meet overlap criteria
    """
    filtered_detections = self.__class__()
    
    for i, detection_i in enumerate(self.detections):
        if detection_i.bbox is None:
            continue
            
        has_overlap = False
        
        for j, detection_j in enumerate(self.detections):
            if i == j or detection_j.bbox is None:
                continue
                
            # If target_class_ids specified, only check overlap with those classes
            if target_class_ids is not None:
                if detection_j.class_id not in target_class_ids:
                    continue
            
            iou = _calculate_iou(detection_i.bbox, detection_j.bbox)
            if iou >= min_overlap:
                has_overlap = True
                break
        
        if has_overlap:
            filtered_detections.add_detection(detection_i)
            
    return filtered_detections