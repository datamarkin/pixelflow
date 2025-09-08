"""
Zone management system for spatial region-based object detection filtering.

This module provides tools for defining geometric zones and automatically filtering 
detections based on their spatial relationship to these zones. Supports multiple 
trigger strategies and statistical tracking for computer vision applications.
"""

from shapely.geometry import Polygon, Point, box
from typing import List, Optional, Tuple, Dict, Any, Literal, Union
import numpy as np
from .strategies import validate_strategy, check_detection_in_region


class Zone:
    """
    Represents a single detection zone with polygon boundary and configurable trigger strategies.
    
    A Zone defines a spatial region for filtering object detections based on their 
    bounding box relationship to the zone polygon. Supports multiple trigger strategies
    including center point, overlap percentage, and intersection detection.
    """
    
    def __init__(
        self,
        polygon: Union[List[Tuple[float, float]], np.ndarray],
        zone_id: Union[int, str],
        name: str = "",
        color: Optional[Tuple[int, int, int]] = None,
        trigger_strategy: Union[str, List[str]] = "center",
        overlap_threshold: float = 0.5,
        mode: Literal["any", "all"] = "all",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize a Zone with polygon boundary and trigger configuration.
        
        Creates a detection zone with specified geometry and trigger behavior.
        The zone tracks statistics including current count and total entries.
        
        Args:
            polygon (Union[List[Tuple[float, float]], np.ndarray]): Zone boundary as 
                    list of (x, y) coordinate tuples or numpy array in pixel coordinates. 
                    Points should form a closed polygon.
            zone_id (Union[int, str]): Unique identifier for the zone. Used for 
                    tracking and referencing in zone operations.
            name (str): Human-readable name for the zone. Defaults to "Zone {zone_id}" 
                   if empty string provided.
            color (Optional[Tuple[int, int, int]]): RGB color tuple (0-255) for 
                  visualization. Auto-generated based on zone_id if None.
            trigger_strategy (Union[str, List[str]]): Strategy for detection matching. 
                             Options: "center", "overlap", "percentage", "bottom_center", 
                             "top_center", "left_center", "right_center". 
                             Default is "center".
            overlap_threshold (float): Threshold for percentage-based strategies. 
                                     Range: [0.0, 1.0]. Default is 0.5 (50% overlap).
            mode (Literal["any", "all"]): Logic mode for multiple strategies.
                 "all" uses AND logic, "any" uses OR logic. Default is "all".
            metadata (Optional[Dict[str, Any]]): Additional custom data for the zone.
                     Can store application-specific information.
                     
        Raises:
            ValueError: If trigger_strategy is invalid or polygon cannot be created.
            TypeError: If polygon coordinates are not numeric types.
            
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> 
            >>> # Basic rectangular zone
            >>> zone = pf.Zone(
            ...     polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
            ...     zone_id="entrance",
            ...     name="Main Entrance"
            ... )
            >>> 
            >>> # Zone with percentage-based triggering
            >>> zone = pf.Zone(
            ...     polygon=[(0, 0), (50, 0), (50, 50), (0, 50)],
            ...     zone_id=1,
            ...     trigger_strategy="percentage",
            ...     overlap_threshold=0.3
            ... )
            >>> 
            >>> # Zone with multiple strategies (OR logic)
            >>> zone = pf.Zone(
            ...     polygon=[(300, 300), (400, 300), (400, 400), (300, 400)],
            ...     zone_id="parking",
            ...     trigger_strategy=["center", "bottom_center"],
            ...     mode="any"
            ... )
            >>> 
            >>> # Zone with custom metadata
            >>> zone = pf.Zone(
            ...     polygon=[(500, 500), (600, 500), (600, 600), (500, 600)],
            ...     zone_id="restricted",
            ...     metadata={"priority": "high", "alert_threshold": 2}
            ... )
            
        Notes:
            - Overlap threshold is automatically clamped to [0.0, 1.0] range
            - Color is generated using golden ratio for consistent distribution
            - Zone tracks unique object IDs to prevent double-counting
            - Polygon is converted to Shapely Polygon for geometric operations
        """
        # Convert polygon to Shapely Polygon
        if isinstance(polygon, np.ndarray):
            polygon = polygon.tolist()
        self.polygon = Polygon(polygon)
        
        self.zone_id = zone_id
        self.name = name or f"Zone {zone_id}"
        self.color = color or self._generate_color(zone_id)
        
        # Trigger configuration
        if trigger_strategy is None:
            trigger_strategy = "center"
        self.trigger_strategy = validate_strategy(trigger_strategy)
        self.overlap_threshold = max(0.0, min(1.0, overlap_threshold))
        self.mode = mode
        
        # Metadata for custom use cases
        self.metadata = metadata or {}
        
        # Statistics
        self.current_count = 0
        self.total_entered = 0
        self._tracked_ids = set()
    
    def _generate_color(self, zone_id: Union[int, str]) -> Tuple[int, int, int]:
        """
        Generate a consistent color based on zone_id.
        
        Uses the golden ratio method to distribute colors evenly across the color
        spectrum for visual distinction between zones.
        
        Args:
            zone_id (Union[int, str]): Zone identifier to generate color from.
            
        Returns:
            Tuple[int, int, int]: RGB color tuple with values in range [0, 255].
        """
        # Use golden ratio for better color distribution
        golden_ratio = 0.618033988749895
        hue = (hash(str(zone_id)) * golden_ratio) % 1.0
        
        # Convert HSV to RGB (simplified)
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        return tuple(int(c * 255) for c in rgb)
    
    def check_detection(self, bbox: List[float], tracker_id: Optional[int] = None) -> bool:
        """
        Check if a detection is within this zone based on the configured trigger strategy.
        
        Evaluates whether a bounding box satisfies the zone's trigger conditions
        and updates zone statistics for unique tracked objects.
        
        Args:
            bbox (List[float]): Bounding box coordinates in [x1, y1, x2, y2] format.
                              Coordinates should be in same units as zone polygon.
            tracker_id (Optional[int]): Unique tracker ID for object counting.
                                      If provided, updates total_entered count for 
                                      first-time entries only.
                                      
        Returns:
            bool: True if detection satisfies zone trigger conditions, False otherwise.
            
        Raises:
            ValueError: If bbox format is invalid or contains non-numeric values.
            AttributeError: If zone polygon is not properly initialized.
            
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> # Setup zone and model
            >>> zone = pf.Zone(
            ...     polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            ...     zone_id="test"
            ... )
            >>> model = YOLO("yolo11n.pt")
            >>> 
            >>> # Basic detection checking
            >>> bbox = [40, 40, 60, 60]  # Center at (50, 50)
            >>> is_in_zone = zone.check_detection(bbox)
            >>> print(f"Detection in zone: {is_in_zone}")  # True
            >>> 
            >>> # Check with tracker ID for counting
            >>> is_in_zone = zone.check_detection(bbox, tracker_id=123)
            >>> print(f"Total entered: {zone.total_entered}")  # 1
            >>> 
            >>> # Multiple checks with same tracker ID (no double-counting)
            >>> zone.check_detection([45, 45, 65, 65], tracker_id=123)
            >>> print(f"Total entered: {zone.total_entered}")  # Still 1
            >>> 
            >>> # Edge case: bbox outside zone
            >>> outside_bbox = [200, 200, 220, 220]
            >>> is_outside = zone.check_detection(outside_bbox)
            >>> print(f"Outside detection: {is_outside}")  # False
            
        Notes:
            - Uses centralized strategy logic from check_detection_in_region
            - Automatically prevents double-counting of tracked objects
            - Statistics are updated only when object enters zone for first time
            - Bounding box coordinates are validated for numeric types
        """
        # Use centralized strategy logic
        in_zone = check_detection_in_region(
            bbox, 
            self.trigger_strategy, 
            self.polygon, 
            self.overlap_threshold,
            self.mode
        )
        
        # Update tracking if object is in zone
        if in_zone and tracker_id is not None:
            if tracker_id not in self._tracked_ids:
                self._tracked_ids.add(tracker_id)
                self.total_entered += 1
        
        return in_zone
    
    def reset_counts(self) -> None:
        """
        Reset all zone statistics to zero.
        
        Clears current count, total entered count, and tracked object IDs.
        Useful for resetting statistics between analysis sessions.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> zone = pf.Zone(
            ...     polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            ...     zone_id="test"
            ... )
            >>> 
            >>> # After some detections
            >>> zone.check_detection([40, 40, 60, 60], tracker_id=1)
            >>> zone.check_detection([50, 50, 70, 70], tracker_id=2)
            >>> print(f"Before reset: {zone.total_entered}")  # 2
            >>> 
            >>> # Reset all statistics
            >>> zone.reset_counts()
            >>> print(f"After reset: {zone.current_count}")  # 0
            >>> print(f"After reset: {zone.total_entered}")  # 0
        """
        self.current_count = 0
        self.total_entered = 0
        self._tracked_ids.clear()


class Zones:
    """
    Manages multiple detection zones and automatically updates results with zone information.
    
    The Zones manager handles a collection of Zone objects and provides batch processing
    of detections against all zones. Updates detection objects with zone membership 
    information and maintains zone statistics.
    """
    
    def __init__(self) -> None:
        """
        Initialize an empty zone manager.
        
        Creates containers for zone storage and fast lookup by ID. The manager
        maintains both a list for ordered access and a dictionary for fast lookups.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Create empty zone manager
            >>> zones = pf.Zones()
            >>> print(len(zones.zones))  # 0
            >>> 
            >>> # Manager is ready for zone additions
            >>> zones.add_zone([(0, 0), (100, 0), (100, 100), (0, 100)])
            >>> print(len(zones.zones))  # 1
        """
        self.zones: List[Zone] = []
        self._zone_dict: Dict[Union[int, str], Zone] = {}
    
    def add_zone(
        self,
        polygon: Union[List[Tuple[float, float]], np.ndarray],
        zone_id: Optional[Union[int, str]] = None,
        name: str = "",
        color: Optional[Tuple[int, int, int]] = None,
        trigger_strategy: Union[str, List[str]] = "center",
        overlap_threshold: float = 0.5,
        mode: Literal["any", "all"] = "all",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Zone:
        """
        Add a new zone to the manager with automatic ID generation.
        
        Creates and registers a new Zone object with the manager. Handles ID 
        conflicts and provides fast lookup capabilities.
        
        Args:
            polygon (Union[List[Tuple[float, float]], np.ndarray]): Zone boundary 
                    coordinates as list of (x, y) tuples or numpy array in pixel coordinates.
            zone_id (Optional[Union[int, str]]): Unique zone identifier. 
                    Auto-generated as sequential integer if None.
            name (str): Human-readable zone name. Defaults to "Zone {zone_id}" if empty.
            color (Optional[Tuple[int, int, int]]): RGB color tuple (0-255) for visualization.
                  Auto-generated if None.
            trigger_strategy (Union[str, List[str]]): Detection matching strategy. 
                             Options: "center", "overlap", "percentage", etc. Default is "center".
            overlap_threshold (float): Threshold for percentage-based strategies.
                                     Range: [0.0, 1.0]. Default is 0.5 (50% overlap).
            mode (Literal["any", "all"]): Logic mode for multiple strategies.
                 "all" for AND logic, "any" for OR logic. Default is "all".
            metadata (Optional[Dict[str, Any]]): Additional zone metadata dictionary.
            
        Returns:
            Zone: The created and registered Zone object.
            
        Raises:
            ValueError: If zone_id already exists in the manager or trigger_strategy is invalid.
            TypeError: If polygon coordinates are not numeric types.
            
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> 
            >>> zones = pf.Zones()
            >>> 
            >>> # Add zone with auto-generated ID
            >>> zone1 = zones.add_zone([(0, 0), (100, 0), (100, 100), (0, 100)])
            >>> print(f"Zone ID: {zone1.zone_id}")  # 0
            >>> 
            >>> # Add zone with custom configuration
            >>> zone2 = zones.add_zone(
            ...     polygon=[(200, 200), (300, 200), (300, 300), (200, 300)],
            ...     zone_id="parking_lot",
            ...     name="Parking Lot Entrance",
            ...     trigger_strategy="percentage",
            ...     overlap_threshold=0.3
            ... )
            >>> 
            >>> # Add zone with multiple strategies
            >>> zone3 = zones.add_zone(
            ...     polygon=[(400, 400), (500, 400), (500, 500), (400, 500)],
            ...     zone_id="security",
            ...     trigger_strategy=["center", "bottom_center"],
            ...     mode="any"
            ... )
            >>> 
            >>> # Add zone with metadata
            >>> zone4 = zones.add_zone(
            ...     polygon=[(600, 600), (700, 600), (700, 700), (600, 700)],
            ...     zone_id="vip",
            ...     metadata={"priority": "high", "max_capacity": 10}
            ... )
            
        Notes:
            - Zone IDs must be unique within the manager
            - Auto-generated IDs start at 0 and increment to avoid conflicts
            - All Zone constructor parameters are supported
            - Zones are stored in both list and dictionary for different access patterns
        """
        # Auto-generate zone_id if not provided
        if zone_id is None:
            zone_id = len(self.zones)
            while zone_id in self._zone_dict:
                zone_id += 1
        
        # Check for duplicate zone_id
        if zone_id in self._zone_dict:
            raise ValueError(f"Zone with ID {zone_id} already exists")
        
        # Create zone
        zone = Zone(
            polygon=polygon,
            zone_id=zone_id,
            name=name,
            color=color,
            trigger_strategy=trigger_strategy,
            overlap_threshold=overlap_threshold,
            mode=mode,
            metadata=metadata
        )
        
        # Add to manager
        self.zones.append(zone)
        self._zone_dict[zone_id] = zone
        
        return zone
    
    def remove_zone(self, zone_id: Union[int, str]) -> None:
        """
        Remove a zone from the manager by its ID.
        
        Removes the zone from both the list and dictionary storage. If the zone
        does not exist, the operation is silently ignored.
        
        Args:
            zone_id (Union[int, str]): ID of the zone to remove. Must match exactly
                                     the zone_id used when creating the zone.
                                     
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> zones = pf.Zones()
            >>> zones.add_zone([(0, 0), (100, 100)], zone_id="entrance")
            >>> print(len(zones.zones))  # 1
            >>> 
            >>> # Remove zone by ID
            >>> zones.remove_zone("entrance")
            >>> print(len(zones.zones))  # 0
            >>> 
            >>> # Removing non-existent zone is safe
            >>> zones.remove_zone("non_existent")  # No error
        """
        if zone_id in self._zone_dict:
            zone = self._zone_dict.pop(zone_id)
            self.zones.remove(zone)
    
    def get_zone(self, zone_id: Union[int, str]) -> Optional[Zone]:
        """
        Retrieve a zone object by its ID.
        
        Provides fast O(1) lookup of zones by their unique identifier.
        
        Args:
            zone_id (Union[int, str]): ID of the zone to retrieve. Must match exactly
                                     the zone_id used when creating the zone.
            
        Returns:
            Optional[Zone]: Zone object if found, None if zone_id doesn't exist.
            
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> zones = pf.Zones()
            >>> zones.add_zone([(0, 0), (100, 100)], zone_id="entrance", name="Main Entrance")
            >>> 
            >>> # Retrieve existing zone
            >>> zone = zones.get_zone("entrance")
            >>> if zone:
            ...     print(f"Found zone: {zone.name}")  # Found zone: Main Entrance
            >>> 
            >>> # Handle non-existent zone
            >>> missing_zone = zones.get_zone("non_existent")
            >>> print(missing_zone)  # None
        """
        return self._zone_dict.get(zone_id)
    
    def update(self, results, strategy: Optional[Union[str, List[str]]] = None):
        """
        Update detection results with zone membership information.
        
        Processes all detections against all managed zones, updating detection
        objects with zone information and maintaining zone statistics. This is
        the primary method for batch zone processing.
        
        Args:
            results: Detections object containing detection list with bbox attributes.
                   Each detection should have a .bbox attribute in [x1, y1, x2, y2] format.
            strategy (Optional[Union[str, List[str]]]): Override strategy for all zones. 
                     If None, each zone uses its own strategy. Can be single string or 
                     list of strings. Will be validated against available strategies.
                   
        Returns:
            Detections: The same Detections object with updated zone information
                       (modified in-place for performance).
                       
        Raises:
            ValueError: If override strategy is invalid or results format is incorrect.
            AttributeError: If results object doesn't have expected attributes.
                       
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> # Setup zones and model
            >>> zones = pf.Zones()
            >>> zones.add_zone([(100, 100), (200, 100), (200, 200), (100, 200)], zone_id="entrance")
            >>> zones.add_zone([(300, 300), (400, 300), (400, 400), (300, 400)], zone_id="exit")
            >>> model = YOLO("yolo11n.pt")
            >>> 
            >>> # Process detections with zones
            >>> image = cv2.imread("image.jpg")
            >>> outputs = model.predict(image)
            >>> results = pf.results.from_ultralytics(outputs)
            >>> 
            >>> # Update with zone-specific strategies
            >>> updated_results = zones.update(results)
            >>> 
            >>> # Override strategy for specific detection types
            >>> person_results = zones.update(person_detections, strategy="bottom_center")
            >>> car_results = zones.update(car_detections, strategy="center")
            >>> 
            >>> # Multiple strategies with OR logic
            >>> flexible_results = zones.update(results, strategy=["center", "overlap"])
            >>> 
            >>> # Access zone information
            >>> for detection in results.detections:
            ...     if hasattr(detection, 'zones') and detection.zones:
            ...         print(f"Detection in zones: {detection.zones}")
            ...         print(f"Zone names: {detection.zone_names}")
            ...         print(f"Zone counts: {zones.get_zone_counts()}")
                        
        Notes:
            - Updates detection.zones with list of matching zone IDs
            - Updates detection.zone_names with corresponding zone names
            - Resets current_count for all zones before processing
            - Updates zone statistics including current_count and total_entered
            - Skips detections without bbox attribute
            - Strategy override applies to all zones uniformly
            - Maintains tracking statistics for unique object counting
        """
        # Validate override strategy if provided
        if strategy is not None:
            strategy = validate_strategy(strategy)
        
        # Reset current counts for all zones
        for zone in self.zones:
            zone.current_count = 0
        
        # Check each detection against all zones
        for detection in results.detections:
            if detection.bbox is None:
                continue
            
            # Find which zones this detection is in
            zones_in = []
            zone_names = []
            
            for zone in self.zones:
                # Use override strategy or zone's own strategy
                detection_strategy = strategy if strategy is not None else zone.trigger_strategy
                
                # Check if detection is in zone using the strategy
                in_zone = check_detection_in_region(
                    detection.bbox, 
                    detection_strategy, 
                    zone.polygon, 
                    zone.overlap_threshold,
                    zone.mode
                )
                
                if in_zone:
                    # Update tracking if object is in zone
                    if detection.tracker_id is not None:
                        if detection.tracker_id not in zone._tracked_ids:
                            zone._tracked_ids.add(detection.tracker_id)
                            zone.total_entered += 1
                    
                    zones_in.append(zone.zone_id)
                    zone_names.append(zone.name)
                    zone.current_count += 1
            
            # Update detection with zone information
            detection.zones = zones_in
            detection.zone_names = zone_names
        
        return results
    
    def get_zone_counts(self) -> Dict[Union[int, str], int]:
        """
        Get current detection count for each managed zone.
        
        Returns current count of detections in each zone from the most recent
        update() call. Useful for real-time monitoring and analytics.
        
        Returns:
            Dict[Union[int, str], int]: Dictionary mapping zone_id to current 
                                      detection count in that zone.
                                      
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> zones = pf.Zones()
            >>> zones.add_zone([(0, 0), (100, 0), (100, 100), (0, 100)], zone_id="zone1")
            >>> zones.add_zone([(200, 200), (300, 200), (300, 300), (200, 300)], zone_id="zone2")
            >>> model = YOLO("yolo11n.pt")
            >>> 
            >>> # Process frame and get counts
            >>> image = cv2.imread("frame.jpg")
            >>> outputs = model.predict(image)
            >>> results = pf.results.from_ultralytics(outputs)
            >>> zones.update(results)
            >>> 
            >>> counts = zones.get_zone_counts()
            >>> print(counts)  # {'zone1': 3, 'zone2': 1}
            >>> 
            >>> # Monitor changes over time
            >>> for frame_path in frame_paths:
            ...     image = cv2.imread(frame_path)
            ...     outputs = model.predict(image)
            ...     results = pf.results.from_ultralytics(outputs)
            ...     zones.update(results)
            ...     current_counts = zones.get_zone_counts()
            ...     print(f"Frame counts: {current_counts}")
        """
        return {zone.zone_id: zone.current_count for zone in self.zones}
    
    def get_zone_stats(self) -> Dict[Union[int, str], Dict[str, Any]]:
        """
        Get comprehensive statistics for all managed zones.
        
        Returns detailed information including current counts, total entries,
        trigger strategies, and metadata for each zone.
        
        Returns:
            Dict[Union[int, str], Dict[str, Any]]: Nested dictionary with zone_id
                                                  as key and statistics dictionary
                                                  as value containing:
                                                  - name: Zone name
                                                  - current_count: Current detections
                                                  - total_entered: Cumulative unique entries
                                                  - trigger_strategy: Strategy configuration
                                                  - metadata: Custom zone data
                                                  
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> zones = pf.Zones()
            >>> zones.add_zone(
            ...     polygon=[(0, 0), (100, 0), (100, 100), (0, 100)], 
            ...     zone_id="entrance", 
            ...     name="Main Entrance",
            ...     metadata={"priority": "high"}
            ... )
            >>> model = YOLO("yolo11n.pt")
            >>> 
            >>> # Process detections over multiple frames
            >>> for frame_path in frame_paths:
            ...     image = cv2.imread(frame_path)
            ...     outputs = model.predict(image)
            ...     results = pf.results.from_ultralytics(outputs)
            ...     zones.update(results)
            >>> 
            >>> # Get comprehensive statistics
            >>> stats = zones.get_zone_stats()
            >>> entrance_stats = stats["entrance"]
            >>> print(f"Name: {entrance_stats['name']}")  # Main Entrance
            >>> print(f"Current: {entrance_stats['current_count']}")  # Current detections
            >>> print(f"Total: {entrance_stats['total_entered']}")   # Total unique entries
            >>> print(f"Strategy: {entrance_stats['trigger_strategy']}")  # center
            >>> print(f"Metadata: {entrance_stats['metadata']}")  # {'priority': 'high'}
            >>> 
            >>> # Export statistics for analysis
            >>> import json
            >>> with open("zone_analytics.json", "w") as f:
            ...     json.dump(stats, f, indent=2)
        """
        stats = {}
        for zone in self.zones:
            stats[zone.zone_id] = {
                'name': zone.name,
                'current_count': zone.current_count,
                'total_entered': zone.total_entered,
                'trigger_strategy': zone.trigger_strategy,
                'metadata': zone.metadata
            }
        return stats
    
    def filter_by_zones(self, results, zone_ids: List[Union[int, str]], exclude: bool = False):
        """
        Filter detection results to include/exclude detections in specified zones.
        
        Creates a new Detections object containing only detections that match
        the zone filtering criteria. Useful for focusing analysis on specific
        spatial regions or excluding restricted areas.
        
        Args:
            results: Detections object to filter. Should have been processed by
                    update() method to have zone information attached.
            zone_ids (List[Union[int, str]]): List of zone IDs to filter by.
                     Must match existing zone IDs in the manager.
            exclude (bool): If True, exclude detections in specified zones.
                          If False, include only detections in specified zones.
                          Default is False (include mode).
                          
        Returns:
            Detections: New filtered Detections object containing subset of
                       original detections based on zone criteria.
                       
        Raises:
            AttributeError: If results object doesn't have expected structure.
            ValueError: If zone_ids contains invalid zone identifiers.
                       
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> zones = pf.Zones()
            >>> zones.add_zone([(0, 0), (200, 0), (200, 200), (0, 200)], zone_id="entrance")
            >>> zones.add_zone([(300, 300), (500, 300), (500, 500), (300, 500)], zone_id="exit")
            >>> zones.add_zone([(600, 600), (800, 600), (800, 800), (600, 800)], zone_id="restricted")
            >>> model = YOLO("yolo11n.pt")
            >>> 
            >>> # Process detections
            >>> image = cv2.imread("image.jpg")
            >>> outputs = model.predict(image)
            >>> results = pf.results.from_ultralytics(outputs)
            >>> results = zones.update(results)
            >>> 
            >>> # Get only entrance detections
            >>> entrance_detections = zones.filter_by_zones(results, ["entrance"])
            >>> print(f"Entrance detections: {len(entrance_detections.detections)}")
            >>> 
            >>> # Get detections NOT in restricted areas
            >>> allowed_detections = zones.filter_by_zones(results, ["restricted"], exclude=True)
            >>> print(f"Allowed detections: {len(allowed_detections.detections)}")
            >>> 
            >>> # Filter by multiple zones (include mode)
            >>> entry_exit_detections = zones.filter_by_zones(results, ["entrance", "exit"])
            >>> 
            >>> # Complex filtering: exclude multiple zones
            >>> public_detections = zones.filter_by_zones(
            ...     results, ["restricted", "private"], exclude=True
            ... )
            
        Notes:
            - Results must be processed with update() first to have zone information
            - Detections without zone information are included when exclude=True
            - Returns empty Detections object if no detections match criteria
            - Original Detections object is not modified
            - Filtering preserves all detection attributes and metadata
        """
        from pixelflow.detections import Detections, Detection
        
        filtered = Detections()
        
        for detection in results.detections:
            if not hasattr(detection, 'zones') or detection.zones is None:
                # If no zone info, include if we're excluding
                if exclude:
                    filtered.detections.append(detection)
            else:
                # Check if detection is in any of the specified zones
                in_specified_zones = any(z in zone_ids for z in detection.zones)
                
                if (in_specified_zones and not exclude) or (not in_specified_zones and exclude):
                    filtered.detections.append(detection)
        
        return filtered
    
    def clear_zones(self) -> None:
        """
        Remove all zones from the manager.
        
        Clears both the zone list and lookup dictionary, effectively resetting
        the manager to its initial empty state.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> zones = pf.Zones()
            >>> zones.add_zone([(0, 0), (100, 100)], zone_id="zone1")
            >>> zones.add_zone([(200, 200), (300, 300)], zone_id="zone2")
            >>> print(len(zones.zones))  # 2
            >>> 
            >>> # Clear all zones
            >>> zones.clear_zones()
            >>> print(len(zones.zones))  # 0
            >>> print(len(zones._zone_dict))  # 0
            >>> 
            >>> # Manager is ready for new zones
            >>> zones.add_zone([(400, 400), (500, 500)])  # Works normally
        """
        self.zones.clear()
        self._zone_dict.clear()
    
    def reset_all_counts(self) -> None:
        """
        Reset statistics for all managed zones.
        
        Calls reset_counts() on each zone to clear current and total counts
        as well as tracked object IDs. Useful for starting fresh analysis sessions.
        
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> from ultralytics import YOLO
            >>> 
            >>> zones = pf.Zones()
            >>> zones.add_zone([(0, 0), (100, 100)], zone_id="zone1")
            >>> zones.add_zone([(200, 200), (300, 300)], zone_id="zone2")
            >>> model = YOLO("yolo11n.pt")
            >>> 
            >>> # Process some frames
            >>> for frame_path in frame_paths[:10]:
            ...     image = cv2.imread(frame_path)
            ...     outputs = model.predict(image)
            ...     results = pf.results.from_ultralytics(outputs)
            ...     zones.update(results)
            >>> 
            >>> print(zones.get_zone_stats())  # Shows accumulated counts
            >>> 
            >>> # Reset for new analysis session
            >>> zones.reset_all_counts()
            >>> stats_after_reset = zones.get_zone_stats()
            >>> for zone_id, stats in stats_after_reset.items():
            ...     print(f"{zone_id}: current={stats['current_count']}, total={stats['total_entered']}")  # All zeros
        """
        for zone in self.zones:
            zone.reset_counts()