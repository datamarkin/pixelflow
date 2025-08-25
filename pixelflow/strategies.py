"""
Strategies for detection and trigger logic used across PixelFlow components.
"""

from enum import Enum


class TriggerStrategy(Enum):
    """
    Strategies for determining if a detection triggers an event (e.g., enters a zone).
    
    These strategies can be used to define how bounding boxes interact with spatial regions
    like zones, boundaries, or other geometric areas.
    
    Attributes:
        CENTER: Check if the center point of the bounding box is in the region
        BOTTOM_CENTER: Check if the bottom center point is in the region (useful for ground-based tracking)
        TOP_LEFT: Check if the top-left corner is in the region
        TOP_RIGHT: Check if the top-right corner is in the region
        BOTTOM_LEFT: Check if the bottom-left corner is in the region
        BOTTOM_RIGHT: Check if the bottom-right corner is in the region
        ANY_CORNER: Check if any corner of the bounding box is in the region
        ALL_CORNERS: Check if all corners of the bounding box are in the region
        OVERLAP: Check if the bounding box overlaps with the region at all
        CONTAINS: Check if the region fully contains the entire bounding box
        PERCENTAGE: Check if the overlap percentage exceeds a threshold
    
    Examples:
        Using with zones:
        >>> zones.add_zone(polygon=points, trigger_strategy="center")
        >>> zones.add_zone(polygon=points, trigger_strategy="bottom_center")  # For foot tracking
        >>> zones.add_zone(polygon=points, trigger_strategy="percentage", overlap_threshold=0.5)
        
        Available string values:
        - "center", "bottom_center", "top_left", "top_right"
        - "bottom_left", "bottom_right", "any_corner", "all_corners"
        - "overlap", "contains", "percentage"
    """
    CENTER = "center"  # Check if center point is in region
    BOTTOM_CENTER = "bottom_center"  # Check if bottom center is in region
    TOP_LEFT = "top_left"  # Check if top-left corner is in region
    TOP_RIGHT = "top_right"  # Check if top-right corner is in region
    BOTTOM_LEFT = "bottom_left"  # Check if bottom-left corner is in region
    BOTTOM_RIGHT = "bottom_right"  # Check if bottom-right corner is in region
    ANY_CORNER = "any_corner"  # Check if any corner is in region
    ALL_CORNERS = "all_corners"  # Check if all corners are in region
    OVERLAP = "overlap"  # Check if bbox overlaps with region
    CONTAINS = "contains"  # Check if region fully contains the bbox
    PERCENTAGE = "percentage"  # Check if overlap percentage exceeds threshold
    