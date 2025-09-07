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
    ):
        """
        Initialize a Zone with polygon boundary and trigger configuration.
        
        Creates a detection zone with specified geometry and trigger behavior.
        The zone tracks statistics including current count and total entries.
        
        Args:
            polygon (Union[List[Tuple[float, float]], np.ndarray]): Zone boundary as 
                    list of (x, y) coordinate tuples or numpy array. Points should 
                    form a closed polygon.
            zone_id (Union[int, str]): Unique identifier for the zone. Used for 
                    tracking and referencing.
            name (str): Human-readable name for the zone. Defaults to "Zone {zone_id}" 
                   if empty.
            color (Optional[Tuple[int, int, int]]): RGB color tuple (0-255) for 
                  visualization. Auto-generated based on zone_id if None.
            trigger_strategy (Union[str, List[str]]): 
                             Strategy for detection matching. Options: "center", "overlap", 
                             "percentage". Default is "center".
            overlap_threshold (float): Threshold for PERCENTAGE strategy. 
                                     Range: [0.0, 1.0]. Default is 0.5 (50% overlap).
            mode (Literal["any", "all"]): Logic mode for multiple strategies.
                 "all" uses AND logic, "any" uses OR logic. Default is "all".
            metadata (Optional[Dict[str, Any]]): Additional custom data for the zone.
                     
        Raises:
            ValueError: If trigger_strategy is invalid or polygon cannot be created.
            
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Create a rectangular zone
            >>> zone = pf.Zone(
            ...     polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
            ...     zone_id="entrance",
            ...     name="Main Entrance"
            ... )
            >>> 
            >>> # Create zone with custom trigger strategy
            >>> zone = pf.Zone(
            ...     polygon=[(0, 0), (50, 0), (50, 50), (0, 50)],
            ...     zone_id=1,
            ...     trigger_strategy="percentage",
            ...     overlap_threshold=0.3
            ... )
            
        Notes:
            - Overlap threshold is automatically clamped to [0.0, 1.0] range
            - Color is generated using golden ratio for consistent distribution
            - Zone tracks unique object IDs to prevent double-counting
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
    
    def _generate_color(self, zone_id) -> Tuple[int, int, int]:
        """Generate a consistent color based on zone_id."""
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
            
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> zone = pf.Zone(
            ...     polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            ...     zone_id="test"
            ... )
            >>> 
            >>> # Check if bounding box is in zone
            >>> bbox = [40, 40, 60, 60]  # Center at (50, 50)
            >>> is_in_zone = zone.check_detection(bbox)
            >>> 
            >>> # Check with tracker ID for counting
            >>> is_in_zone = zone.check_detection(bbox, tracker_id=123)
            
        Notes:
            - Uses centralized strategy logic from check_detection_in_region
            - Automatically prevents double-counting of tracked objects
            - Statistics are updated only when object enters zone for first time
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
    
    def reset_counts(self):
        """
        Reset all zone statistics to zero.
        
        Clears current count, total entered count, and tracked object IDs.
        Useful for resetting statistics between analysis sessions.
        
        Example:
            >>> zone.reset_counts()
            >>> print(zone.current_count)  # 0
            >>> print(zone.total_entered)  # 0
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
    
    def __init__(self):
        """
        Initialize an empty zone manager.
        
        Creates containers for zone storage and fast lookup by ID.
        
        Example:
            >>> import pixelflow as pf
            >>> zones = pf.Zones()
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
                    coordinates as list of (x, y) tuples or numpy array.
            zone_id (Optional[Union[int, str]]): Unique zone identifier. 
                    Auto-generated as sequential integer if None.
            name (str): Human-readable zone name. Defaults to "Zone {zone_id}".
            color (Optional[Tuple[int, int, int]]): RGB color tuple for visualization.
                  Auto-generated if None.
            trigger_strategy (Union[str, List[str]]): 
                             Detection matching strategy. Default is "center".
            overlap_threshold (float): Threshold for percentage-based strategies.
                                     Range: [0.0, 1.0]. Default is 0.5.
            mode (Literal["any", "all"]): Logic mode for multiple strategies.
                 "all" for AND logic, "any" for OR logic. Default is "all".
            metadata (Optional[Dict[str, Any]]): Additional zone metadata.
            
        Returns:
            Zone: The created and registered Zone object.
            
        Raises:
            ValueError: If zone_id already exists in the manager.
            
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> zones = pf.Zones()
            >>> 
            >>> # Add zone with auto-generated ID
            >>> zone1 = zones.add_zone([(0, 0), (100, 0), (100, 100), (0, 100)])
            >>> 
            >>> # Add zone with custom configuration
            >>> zone2 = zones.add_zone(
            ...     polygon=[(200, 200), (300, 200), (300, 300), (200, 300)],
            ...     zone_id="parking_lot",
            ...     name="Parking Lot Entrance",
            ...     trigger_strategy="percentage",
            ...     overlap_threshold=0.3
            ... )
            
        Notes:
            - Zone IDs must be unique within the manager
            - Auto-generated IDs start at 0 and increment to avoid conflicts
            - All Zone constructor parameters are supported
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
    
    def remove_zone(self, zone_id: Union[int, str]):
        """
        Remove a zone from the manager by its ID.
        
        Args:
            zone_id (Union[int, str]): ID of the zone to remove.
                                     
        Example:
            >>> zones.remove_zone("entrance")
        """
        if zone_id in self._zone_dict:
            zone = self._zone_dict.pop(zone_id)
            self.zones.remove(zone)
    
    def get_zone(self, zone_id: Union[int, str]) -> Optional[Zone]:
        """
        Retrieve a zone object by its ID.
        
        Args:
            zone_id (Union[int, str]): ID of the zone to retrieve.
            
        Returns:
            Optional[Zone]: Zone object if found, None otherwise.
            
        Example:
            >>> zone = zones.get_zone("entrance")
            >>> if zone:
            ...     print(zone.name)
        """
        return self._zone_dict.get(zone_id)
    
    def update(self, results, strategy: Union[str, List[str]] = None):
        """
        Update detection results with zone membership information.
        
        Processes all detections against all managed zones, updating detection
        objects with zone information and maintaining zone statistics. This is
        the primary method for batch zone processing.
        
        Args:
            results: Detections object containing detection list with bbox attributes.
                   Each detection should have a .bbox attribute in [x1, y1, x2, y2] format.
            strategy: Optional override strategy for all zones. If None, each zone uses its own strategy.
                     Can be a single string or list of strings. Will be validated.
                   
        Returns:
            Detections: The same Detections object with updated zone information
                       (modified in-place for performance).
                       
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> 
            >>> # Setup zones and get detection results
            >>> zones = pf.Zones()
            >>> zones.add_zone([(100, 100), (200, 100), (200, 200), (100, 200)])
            >>> 
            >>> image = cv2.imread("image.jpg")
            >>> outputs = model.predict(image)
            >>> results = pf.results.from_ultralytics(outputs)
            >>> 
            >>> # Update results with zone information using zone's own strategies
            >>> updated_results = zones.update(results)
            >>> 
            >>> # Override strategy for all zones (for flexible multi-class usage)
            >>> person_results = zones.update(person_detections, strategy="bottom_center")
            >>> car_results = zones.update(car_detections, strategy="center")
            >>> 
            >>> # Access zone information
            >>> for detection in results.detections:
            ...     if detection.zones:
            ...         print(f"Detection in zones: {detection.zones}")
            ...         print(f"Zone names: {detection.zone_names}")
                        
        Notes:
            - Updates detection.zones with list of matching zone IDs
            - Updates detection.zone_names with corresponding zone names
            - Resets current_count for all zones before processing
            - Updates zone statistics including current_count and total_entered
            - Skips detections without bbox attribute
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
            >>> zones = pf.Zones()
            >>> zones.add_zone([(0, 0), (100, 100)], zone_id="zone1")
            >>> zones.add_zone([(200, 200), (300, 300)], zone_id="zone2")
            >>> 
            >>> # After processing detections
            >>> zones.update(results)
            >>> counts = zones.get_zone_counts()
            >>> print(counts)  # {'zone1': 3, 'zone2': 1}
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
                                                  - trigger_strategy: Strategy used
                                                  - metadata: Custom zone data
                                                  
        Example:
            >>> zones = pf.Zones()
            >>> zones.add_zone([(0, 0), (100, 100)], zone_id="entrance", name="Main Entrance")
            >>> 
            >>> # After processing detections
            >>> zones.update(results)
            >>> stats = zones.get_zone_stats()
            >>> print(stats["entrance"]["current_count"])  # Current detections
            >>> print(stats["entrance"]["total_entered"])   # Total unique entries
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
        spatial regions.
        
        Args:
            results: Detections object to filter. Should have been processed by
                    update() method to have zone information.
            zone_ids (List[Union[int, str]]): List of zone IDs to filter by.
                     Must match existing zone IDs in the manager.
            exclude (bool): If True, exclude detections in specified zones.
                          If False, include only detections in specified zones.
                          Default is False (include mode).
                          
        Returns:
            Detections: New filtered Detections object containing subset of
                       original detections based on zone criteria.
                       
        Example:
            >>> import cv2
            >>> import pixelflow as pf
            >>> 
            >>> zones = pf.Zones()
            >>> zones.add_zone([(0, 0), (100, 100), (200, 100), (200, 200)], zone_id="entrance")
            >>> zones.add_zone([(300, 300), (400, 300), (400, 400), (300, 400)], zone_id="exit")
            >>> 
            >>> image = cv2.imread("image.jpg")
            >>> outputs = model.predict(image)
            >>> results = pf.results.from_ultralytics(outputs)
            >>> results = zones.update(results)
            >>> 
            >>> # Get only detections in entrance zone
            >>> entrance_detections = zones.filter_by_zones(results, ["entrance"])
            >>> 
            >>> # Get detections NOT in exit zone
            >>> non_exit_detections = zones.filter_by_zones(results, ["exit"], exclude=True)
            
        Notes:
            - Results must be processed with update() first to have zone information
            - Detections without zone information are included when exclude=True
            - Returns empty Detections object if no detections match criteria
            - Original Detections object is not modified
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
    
    def clear_zones(self):
        """
        Remove all zones from the manager.
        
        Clears both the zone list and lookup dictionary.
        
        Example:
            >>> zones.clear_zones()
            >>> len(zones.zones)  # 0
        """
        self.zones.clear()
        self._zone_dict.clear()
    
    def reset_all_counts(self):
        """
        Reset statistics for all managed zones.
        
        Calls reset_counts() on each zone to clear current and total counts.
        
        Example:
            >>> zones.reset_all_counts()
        """
        for zone in self.zones:
            zone.reset_counts()