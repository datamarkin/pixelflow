"""
Strategies for detection and trigger logic used across PixelFlow components.
"""

from enum import Enum
from typing import List, Tuple, Optional, Union, Literal
from shapely.geometry import Polygon, Point, box


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
    MULTI_ANCHOR = "multi_anchor"  # Check multiple anchor points with AND/OR logic
    TOP_CENTER = "top_center"  # Check if top center point is in region
    LEFT_CENTER = "left_center"  # Check if left center point is in region  
    RIGHT_CENTER = "right_center"  # Check if right center point is in region


class AnchorConfig:
    """
    Configuration for multiple anchor point checking.
    
    This class defines how multiple anchor points should be evaluated together.
    Useful for cases where you need both corners of a bounding box to cross a line,
    or when you want any of several anchor points to trigger a zone detection.
    
    Examples:
        # Both bottom-right and top-left must cross the line
        config = AnchorConfig(["bottom_right", "top_left"], mode="all")
        
        # Either center or bottom center can trigger the zone
        config = AnchorConfig(["center", "bottom_center"], mode="any")
        
        # All four corners must be inside the zone
        config = AnchorConfig(["top_left", "top_right", "bottom_left", "bottom_right"], mode="all")
    """
    
    def __init__(
        self, 
        anchors: List[Union[TriggerStrategy, str]], 
        mode: Literal["all", "any"] = "all"
    ):
        """
        Initialize anchor configuration.
        
        Args:
            anchors: List of anchor strategies to check
            mode: "all" requires all anchors to pass (AND logic)
                  "any" requires at least one anchor to pass (OR logic)
        """
        # Convert string anchors to TriggerStrategy enum
        self.anchors = []
        for anchor in anchors:
            if isinstance(anchor, str):
                try:
                    self.anchors.append(TriggerStrategy(anchor))
                except ValueError:
                    valid_strategies = [s.value for s in TriggerStrategy if s != TriggerStrategy.MULTI_ANCHOR]
                    raise ValueError(
                        f"Invalid anchor '{anchor}'. "
                        f"Valid options are: {', '.join(valid_strategies)}"
                    )
            else:
                self.anchors.append(anchor)
        
        if not self.anchors:
            raise ValueError("At least one anchor must be specified")
            
        if mode not in ["all", "any"]:
            raise ValueError("Mode must be 'all' or 'any'")
            
        self.mode = mode


def get_anchor_position(bbox: List[float], strategy: TriggerStrategy) -> Tuple[float, float]:
    """
    Get the position of a specific anchor point on the bounding box.
    
    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format
        strategy: The trigger strategy defining which anchor point to use
        
    Returns:
        Tuple of (x, y) coordinates for the anchor point
    """
    x1, y1, x2, y2 = bbox
    
    if strategy == TriggerStrategy.CENTER:
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    elif strategy == TriggerStrategy.BOTTOM_CENTER:
        return ((x1 + x2) / 2, y2)
    elif strategy == TriggerStrategy.TOP_LEFT:
        return (x1, y1)
    elif strategy == TriggerStrategy.TOP_RIGHT:
        return (x2, y1)
    elif strategy == TriggerStrategy.BOTTOM_LEFT:
        return (x1, y2)
    elif strategy == TriggerStrategy.BOTTOM_RIGHT:
        return (x2, y2)
    elif strategy == TriggerStrategy.TOP_CENTER:
        return ((x1 + x2) / 2, y1)
    elif strategy == TriggerStrategy.LEFT_CENTER:
        return (x1, (y1 + y2) / 2)
    elif strategy == TriggerStrategy.RIGHT_CENTER:
        return (x2, (y1 + y2) / 2)
    else:
        # Default to center if unknown
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def check_multiple_anchors(
    bbox: List[float],
    anchor_config: AnchorConfig,
    region: Polygon,
    overlap_threshold: float = 0.5
) -> bool:
    """
    Check multiple anchor points with AND/OR logic.
    
    This function evaluates multiple anchor points according to the provided
    configuration, allowing for complex detection scenarios like requiring
    both corners to cross a line or any of several points to be in a zone.
    
    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format
        anchor_config: Configuration specifying which anchors to check and how
        region: Shapely Polygon representing the region
        overlap_threshold: Threshold for PERCENTAGE strategy (0.0 to 1.0)
        
    Returns:
        True if detection meets the anchor configuration criteria, False otherwise
    """
    results = []
    
    for anchor_strategy in anchor_config.anchors:
        # Use the existing single anchor checking logic
        result = check_detection_in_region(bbox, anchor_strategy, region, overlap_threshold)
        results.append(result)
    
    # Apply AND/OR logic based on mode
    if anchor_config.mode == "all":
        return all(results)
    else:  # mode == "any"
        return any(results)


def check_detection_in_region(
    bbox: List[float], 
    strategy: TriggerStrategy, 
    region: Polygon, 
    overlap_threshold: float = 0.5,
    anchor_config: Optional[AnchorConfig] = None
) -> bool:
    """
    Check if a detection is within a region based on the trigger strategy.
    
    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format
        strategy: Strategy for determining if detection is in region
        region: Shapely Polygon representing the region
        overlap_threshold: Threshold for PERCENTAGE strategy (0.0 to 1.0)
        anchor_config: Configuration for MULTI_ANCHOR strategy (required when strategy is MULTI_ANCHOR)
        
    Returns:
        True if detection is in region according to strategy, False otherwise
    """
    x1, y1, x2, y2 = bbox
    
    # Handle multi-anchor strategy first
    if strategy == TriggerStrategy.MULTI_ANCHOR:
        if anchor_config is None:
            raise ValueError("anchor_config is required when using MULTI_ANCHOR strategy")
        return check_multiple_anchors(bbox, anchor_config, region, overlap_threshold)
    
    # Create bbox polygon for geometric operations
    bbox_poly = box(x1, y1, x2, y2)
    
    # Check based on strategy
    if strategy == TriggerStrategy.CENTER:
        center = Point((x1 + x2) / 2, (y1 + y2) / 2)
        return region.contains(center)
        
    elif strategy == TriggerStrategy.BOTTOM_CENTER:
        bottom_center = Point((x1 + x2) / 2, y2)
        return region.contains(bottom_center)
        
    elif strategy == TriggerStrategy.TOP_LEFT:
        return region.contains(Point(x1, y1))
        
    elif strategy == TriggerStrategy.TOP_RIGHT:
        return region.contains(Point(x2, y1))
        
    elif strategy == TriggerStrategy.BOTTOM_LEFT:
        return region.contains(Point(x1, y2))
        
    elif strategy == TriggerStrategy.BOTTOM_RIGHT:
        return region.contains(Point(x2, y2))
        
    elif strategy == TriggerStrategy.TOP_CENTER:
        top_center = Point((x1 + x2) / 2, y1)
        return region.contains(top_center)
        
    elif strategy == TriggerStrategy.LEFT_CENTER:
        left_center = Point(x1, (y1 + y2) / 2)
        return region.contains(left_center)
        
    elif strategy == TriggerStrategy.RIGHT_CENTER:
        right_center = Point(x2, (y1 + y2) / 2)
        return region.contains(right_center)
        
    elif strategy == TriggerStrategy.ANY_CORNER:
        corners = [Point(x1, y1), Point(x2, y1), Point(x1, y2), Point(x2, y2)]
        return any(region.contains(corner) for corner in corners)
        
    elif strategy == TriggerStrategy.ALL_CORNERS:
        corners = [Point(x1, y1), Point(x2, y1), Point(x1, y2), Point(x2, y2)]
        return all(region.contains(corner) for corner in corners)
        
    elif strategy == TriggerStrategy.OVERLAP:
        return region.intersects(bbox_poly)
        
    elif strategy == TriggerStrategy.CONTAINS:
        return region.contains(bbox_poly)
        
    elif strategy == TriggerStrategy.PERCENTAGE:
        if region.intersects(bbox_poly):
            intersection = region.intersection(bbox_poly)
            overlap_ratio = intersection.area / bbox_poly.area
            return overlap_ratio >= overlap_threshold
        else:
            return False
    else:
        return False
    