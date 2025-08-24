# zones/__init__.py

from shapely.geometry import Polygon
from typing import List, Optional, Tuple


class Zone:
    def __init__(self, polygon: Polygon, zone_id: int, name: str = "", color: Optional[Tuple[int, int, int]] = None):
        """
        Represents a zone with an associated polygon, ID, name, and color.

        Args:
            polygon (Polygon): The geometric polygon representing the zone.
            zone_id (int): A unique identifier for the zone.
            name (str): A human-readable name for the zone (default: "").
            color (Tuple[int, int, int], optional): The color associated with the zone (default: None).
        """
        self.polygon = polygon
        self.zone_id = zone_id
        self.name = name
        self.color = color or (255, 255, 255)  # Default color is white (RGB)


class Zones:
    def __init__(self):
        self.zones: List[Zone] = []

    def add_zone(self, zone: Zone):
        """Add a zone to the zones list."""
        self.zones.append(zone)

    def remove_zone(self, zone_id: int):
        """Remove a zone from the zones list by its ID."""
        self.zones = [zone for zone in self.zones if zone.zone_id != zone_id]

    def is_inside(self, bbox: List[float], masks: List[List[float]]) -> bool:
        """
        Check if the bounding box or masks fall within any of the zones.
        """
        bbox_polygon = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])
        for zone in self.zones:
            if zone.polygon.contains(bbox_polygon) or zone.polygon.intersects(bbox_polygon):
                return True

        #TODO verify with real world logic if this is necessary
        if masks:
            for mask in masks:
                mask_polygon = Polygon(mask)
                for zone in self.zones:
                    if zone.polygon.contains(mask_polygon) or zone.polygon.intersects(mask_polygon):
                        return True

        return False
