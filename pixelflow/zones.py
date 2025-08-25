# zones.py

from shapely.geometry import Polygon, Point, box
from typing import List, Optional, Tuple, Dict, Any, Literal, Union
import numpy as np
from .strategies import TriggerStrategy


class Zone:
    """Represents a single detection zone with polygon boundary."""
    
    def __init__(
        self,
        polygon: List[Tuple[float, float]] | np.ndarray,
        zone_id: int | str,
        name: str = "",
        color: Optional[Tuple[int, int, int]] = None,
        trigger_strategy: Union[TriggerStrategy, Literal["center", "bottom_center", "top_left", "top_right", "bottom_left", "bottom_right", "any_corner", "all_corners", "overlap", "contains", "percentage"], None] = "center",
        overlap_threshold: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a Zone.
        
        Args:
            polygon: List of (x, y) tuples or numpy array defining zone boundary
            zone_id: Unique identifier for the zone
            name: Human-readable name for the zone
            color: RGB color tuple for visualization (default: auto-generated)
            trigger_strategy: Strategy for determining if detection is in zone
            overlap_threshold: Threshold for PERCENTAGE strategy (0.0 to 1.0)
            metadata: Additional custom data associated with the zone
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
        if isinstance(trigger_strategy, str):
            try:
                trigger_strategy = TriggerStrategy(trigger_strategy)
            except ValueError:
                valid_strategies = [s.value for s in TriggerStrategy]
                raise ValueError(
                    f"Invalid trigger_strategy '{trigger_strategy}'. "
                    f"Valid options are: {', '.join(valid_strategies)}"
                )
        self.trigger_strategy = trigger_strategy
        self.overlap_threshold = max(0.0, min(1.0, overlap_threshold))
        
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
        Check if a detection is within this zone based on the trigger strategy.
        
        Args:
            bbox: Bounding box in [x1, y1, x2, y2] format
            tracker_id: Optional tracker ID for counting unique objects
            
        Returns:
            True if detection is in zone, False otherwise
        """
        x1, y1, x2, y2 = bbox
        
        # Create bbox polygon
        bbox_poly = box(x1, y1, x2, y2)
        
        # Check based on strategy
        if self.trigger_strategy == TriggerStrategy.CENTER:
            center = Point((x1 + x2) / 2, (y1 + y2) / 2)
            in_zone = self.polygon.contains(center)
            
        elif self.trigger_strategy == TriggerStrategy.BOTTOM_CENTER:
            bottom_center = Point((x1 + x2) / 2, y2)
            in_zone = self.polygon.contains(bottom_center)
            
        elif self.trigger_strategy == TriggerStrategy.TOP_LEFT:
            in_zone = self.polygon.contains(Point(x1, y1))
            
        elif self.trigger_strategy == TriggerStrategy.TOP_RIGHT:
            in_zone = self.polygon.contains(Point(x2, y1))
            
        elif self.trigger_strategy == TriggerStrategy.BOTTOM_LEFT:
            in_zone = self.polygon.contains(Point(x1, y2))
            
        elif self.trigger_strategy == TriggerStrategy.BOTTOM_RIGHT:
            in_zone = self.polygon.contains(Point(x2, y2))
            
        elif self.trigger_strategy == TriggerStrategy.ANY_CORNER:
            corners = [Point(x1, y1), Point(x2, y1), Point(x1, y2), Point(x2, y2)]
            in_zone = any(self.polygon.contains(corner) for corner in corners)
            
        elif self.trigger_strategy == TriggerStrategy.ALL_CORNERS:
            corners = [Point(x1, y1), Point(x2, y1), Point(x1, y2), Point(x2, y2)]
            in_zone = all(self.polygon.contains(corner) for corner in corners)
            
        elif self.trigger_strategy == TriggerStrategy.OVERLAP:
            in_zone = self.polygon.intersects(bbox_poly)
            
        elif self.trigger_strategy == TriggerStrategy.CONTAINS:
            in_zone = self.polygon.contains(bbox_poly)
            
        elif self.trigger_strategy == TriggerStrategy.PERCENTAGE:
            if self.polygon.intersects(bbox_poly):
                intersection = self.polygon.intersection(bbox_poly)
                overlap_ratio = intersection.area / bbox_poly.area
                in_zone = overlap_ratio >= self.overlap_threshold
            else:
                in_zone = False
        else:
            in_zone = False
        
        # Update tracking if object is in zone
        if in_zone and tracker_id is not None:
            if tracker_id not in self._tracked_ids:
                self._tracked_ids.add(tracker_id)
                self.total_entered += 1
        
        return in_zone
    
    def reset_counts(self):
        """Reset zone statistics."""
        self.current_count = 0
        self.total_entered = 0
        self._tracked_ids.clear()


class Zones:
    """Manages multiple zones and updates detection results with zone information."""
    
    def __init__(self):
        """Initialize the ZoneManager."""
        self.zones: List[Zone] = []
        self._zone_dict: Dict[int | str, Zone] = {}
    
    def add_zone(
        self,
        polygon: List[Tuple[float, float]] | np.ndarray,
        zone_id: Optional[int | str] = None,
        name: str = "",
        color: Optional[Tuple[int, int, int]] = None,
        trigger_strategy: Union[TriggerStrategy, Literal["center", "bottom_center", "top_left", "top_right", "bottom_left", "bottom_right", "any_corner", "all_corners", "overlap", "contains", "percentage"], None] = "center",
        overlap_threshold: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Zone:
        """
        Add a new zone to the manager.
        
        Args:
            polygon: List of (x, y) tuples or numpy array defining zone boundary
            zone_id: Unique identifier (auto-generated if None)
            name: Human-readable name for the zone
            color: RGB color tuple for visualization
            trigger_strategy: Strategy for determining if detection is in zone
            overlap_threshold: Threshold for PERCENTAGE strategy
            metadata: Additional custom data
            
        Returns:
            The created Zone object
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
            metadata=metadata
        )
        
        # Add to manager
        self.zones.append(zone)
        self._zone_dict[zone_id] = zone
        
        return zone
    
    def remove_zone(self, zone_id: int | str):
        """Remove a zone by its ID."""
        if zone_id in self._zone_dict:
            zone = self._zone_dict.pop(zone_id)
            self.zones.remove(zone)
    
    def get_zone(self, zone_id: int | str) -> Optional[Zone]:
        """Get a zone by its ID."""
        return self._zone_dict.get(zone_id)
    
    def update(self, results):
        """
        Update detection results with zone information.
        
        This method checks each detection against all zones and updates:
        - prediction.zones: List of zone IDs the detection is in
        - prediction.zone_names: List of zone names (for convenience)
        - Zone statistics (current_count, total_entered)
        
        Args:
            results: Results object containing predictions
            
        Returns:
            Updated Results object (modified in-place)
        """
        # Reset current counts for all zones
        for zone in self.zones:
            zone.current_count = 0
        
        # Check each prediction against all zones
        for prediction in results.predictions:
            if prediction.bbox is None:
                continue
            
            # Find which zones this detection is in
            zones_in = []
            zone_names = []
            
            for zone in self.zones:
                if zone.check_detection(prediction.bbox, prediction.tracker_id):
                    zones_in.append(zone.zone_id)
                    zone_names.append(zone.name)
                    zone.current_count += 1
            
            # Update prediction with zone information
            prediction.zones = zones_in
            prediction.zone_names = zone_names
        
        return results
    
    def get_zone_counts(self) -> Dict[int | str, int]:
        """
        Get current detection count for each zone.
        
        Returns:
            Dictionary mapping zone_id to current count
        """
        return {zone.zone_id: zone.current_count for zone in self.zones}
    
    def get_zone_stats(self) -> Dict[int | str, Dict[str, Any]]:
        """
        Get detailed statistics for each zone.
        
        Returns:
            Dictionary with zone statistics including counts and metadata
        """
        stats = {}
        for zone in self.zones:
            stats[zone.zone_id] = {
                'name': zone.name,
                'current_count': zone.current_count,
                'total_entered': zone.total_entered,
                'trigger_strategy': zone.trigger_strategy.value,
                'metadata': zone.metadata
            }
        return stats
    
    def filter_by_zones(self, results, zone_ids: List[int | str], exclude: bool = False):
        """
        Filter results to only include detections in specified zones.
        
        Args:
            results: Results object to filter
            zone_ids: List of zone IDs to filter by
            exclude: If True, exclude detections in specified zones (default: include)
            
        Returns:
            New filtered Results object
        """
        from pixelflow.results import Results, Prediction
        
        filtered = Results()
        
        for prediction in results.predictions:
            if not hasattr(prediction, 'zones') or prediction.zones is None:
                # If no zone info, include if we're excluding
                if exclude:
                    filtered.predictions.append(prediction)
            else:
                # Check if prediction is in any of the specified zones
                in_specified_zones = any(z in zone_ids for z in prediction.zones)
                
                if (in_specified_zones and not exclude) or (not in_specified_zones and exclude):
                    filtered.predictions.append(prediction)
        
        return filtered
    
    def clear_zones(self):
        """Remove all zones."""
        self.zones.clear()
        self._zone_dict.clear()
    
    def reset_all_counts(self):
        """Reset statistics for all zones."""
        for zone in self.zones:
            zone.reset_counts()