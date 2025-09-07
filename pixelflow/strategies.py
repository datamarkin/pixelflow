"""
Strategies for detection and trigger logic used across PixelFlow components.
"""

from typing import List, Tuple, Union, Literal
from shapely.geometry import Polygon, Point, box

# Valid strategy constants
STRATEGY_CENTER = "center"
STRATEGY_BOTTOM_CENTER = "bottom_center"
STRATEGY_TOP_LEFT = "top_left"
STRATEGY_TOP_RIGHT = "top_right"
STRATEGY_BOTTOM_LEFT = "bottom_left"
STRATEGY_BOTTOM_RIGHT = "bottom_right"
STRATEGY_TOP_CENTER = "top_center"
STRATEGY_LEFT_CENTER = "left_center"
STRATEGY_RIGHT_CENTER = "right_center"
STRATEGY_ANY_CORNER = "any_corner"
STRATEGY_ALL_CORNERS = "all_corners"
STRATEGY_OVERLAP = "overlap"
STRATEGY_CONTAINS = "contains"
STRATEGY_PERCENTAGE = "percentage"

# All valid strategy strings
VALID_STRATEGIES = {
    STRATEGY_CENTER,
    STRATEGY_BOTTOM_CENTER,
    STRATEGY_TOP_LEFT,
    STRATEGY_TOP_RIGHT,
    STRATEGY_BOTTOM_LEFT,
    STRATEGY_BOTTOM_RIGHT,
    STRATEGY_TOP_CENTER,
    STRATEGY_LEFT_CENTER,
    STRATEGY_RIGHT_CENTER,
    STRATEGY_ANY_CORNER,
    STRATEGY_ALL_CORNERS,
    STRATEGY_OVERLAP,
    STRATEGY_CONTAINS,
    STRATEGY_PERCENTAGE,
}


def validate_strategy(strategy: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Validate that strategy string(s) are valid.
    
    Args:
        strategy: Single strategy string or list of strategy strings
        
    Returns:
        The validated strategy (unchanged if valid)
        
    Raises:
        ValueError: If any strategy is invalid
        
    Examples:
        >>> validate_strategy("center")  # Returns "center"
        >>> validate_strategy(["center", "bottom_center"])  # Returns ["center", "bottom_center"]
        >>> validate_strategy("invalid")  # Raises ValueError
    """
    if isinstance(strategy, (list, tuple)):
        for s in strategy:
            if s not in VALID_STRATEGIES:
                raise ValueError(
                    f"Invalid strategy '{s}'. "
                    f"Valid options are: {', '.join(sorted(VALID_STRATEGIES))}"
                )
        return strategy
    else:
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Valid options are: {', '.join(sorted(VALID_STRATEGIES))}"
            )
        return strategy




def get_anchor_position(bbox: List[float], strategy: str) -> Tuple[float, float]:
    """
    Get the position of a specific anchor point on the bounding box.
    
    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format
        strategy: The trigger strategy string defining which anchor point to use
        
    Returns:
        Tuple of (x, y) coordinates for the anchor point
        
    Examples:
        >>> bbox = [10, 20, 50, 80]
        >>> get_anchor_position(bbox, "center")
        (30.0, 50.0)
        >>> get_anchor_position(bbox, "bottom_center")
        (30.0, 80.0)
    """
    x1, y1, x2, y2 = bbox
    
    if strategy == STRATEGY_CENTER:
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    elif strategy == STRATEGY_BOTTOM_CENTER:
        return ((x1 + x2) / 2, y2)
    elif strategy == STRATEGY_TOP_LEFT:
        return (x1, y1)
    elif strategy == STRATEGY_TOP_RIGHT:
        return (x2, y1)
    elif strategy == STRATEGY_BOTTOM_LEFT:
        return (x1, y2)
    elif strategy == STRATEGY_BOTTOM_RIGHT:
        return (x2, y2)
    elif strategy == STRATEGY_TOP_CENTER:
        return ((x1 + x2) / 2, y1)
    elif strategy == STRATEGY_LEFT_CENTER:
        return (x1, (y1 + y2) / 2)
    elif strategy == STRATEGY_RIGHT_CENTER:
        return (x2, (y1 + y2) / 2)
    else:
        # Default to center if unknown
        return ((x1 + x2) / 2, (y1 + y2) / 2)




def check_detection_in_region(
    bbox: List[float], 
    strategy: Union[str, List[str]], 
    region: Polygon, 
    overlap_threshold: float = 0.5,
    mode: Literal["any", "all"] = "all"
) -> bool:
    """
    Check if a detection is within a region based on the trigger strategy.
    
    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format
        strategy: Strategy for determining if detection is in region. Can be:
                 - Single string (e.g., "center")
                 - List of strings (e.g., ["center", "bottom_center"])
        region: Shapely Polygon representing the region
        overlap_threshold: Threshold for PERCENTAGE strategy (0.0 to 1.0)
        mode: "all" (AND logic) or "any" (OR logic) for multiple strategies. Default: "all"
        
    Returns:
        True if detection is in region according to strategy, False otherwise
        
    Examples:
        >>> from shapely.geometry import box
        >>> bbox = [10, 10, 50, 50]
        >>> region = box(0, 0, 100, 100)
        >>> check_detection_in_region(bbox, "center", region)
        True
    """
    # Convert strategy to list of strings for uniform processing
    if isinstance(strategy, (list, tuple)):
        strategies_to_check = [validate_strategy(s) for s in strategy]
    else:
        strategies_to_check = [validate_strategy(strategy)]
    
    # Check each strategy
    results = []
    for single_strategy in strategies_to_check:
        result = _check_single_strategy(bbox, single_strategy, region, overlap_threshold)
        results.append(result)
    
    # Apply AND/OR logic
    if mode == "all":
        return all(results)
    else:  # mode == "any"
        return any(results)


def _check_single_strategy(
    bbox: List[float],
    strategy: str,
    region: Polygon,
    overlap_threshold: float = 0.5
) -> bool:
    """
    Check a single strategy against a region.
    
    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format
        strategy: Single strategy string to check
        region: Shapely Polygon representing the region
        overlap_threshold: Threshold for PERCENTAGE strategy (0.0 to 1.0)
        
    Returns:
        True if detection meets the strategy criteria, False otherwise
    """
    x1, y1, x2, y2 = bbox
    
    # Create bbox polygon for geometric operations
    bbox_poly = box(x1, y1, x2, y2)
    
    # Check based on strategy
    if strategy == STRATEGY_CENTER:
        center = Point((x1 + x2) / 2, (y1 + y2) / 2)
        return region.contains(center)
        
    elif strategy == STRATEGY_BOTTOM_CENTER:
        bottom_center = Point((x1 + x2) / 2, y2)
        return region.contains(bottom_center)
        
    elif strategy == STRATEGY_TOP_LEFT:
        return region.contains(Point(x1, y1))
        
    elif strategy == STRATEGY_TOP_RIGHT:
        return region.contains(Point(x2, y1))
        
    elif strategy == STRATEGY_BOTTOM_LEFT:
        return region.contains(Point(x1, y2))
        
    elif strategy == STRATEGY_BOTTOM_RIGHT:
        return region.contains(Point(x2, y2))
        
    elif strategy == STRATEGY_TOP_CENTER:
        top_center = Point((x1 + x2) / 2, y1)
        return region.contains(top_center)
        
    elif strategy == STRATEGY_LEFT_CENTER:
        left_center = Point(x1, (y1 + y2) / 2)
        return region.contains(left_center)
        
    elif strategy == STRATEGY_RIGHT_CENTER:
        right_center = Point(x2, (y1 + y2) / 2)
        return region.contains(right_center)
        
    elif strategy == STRATEGY_ANY_CORNER:
        corners = [Point(x1, y1), Point(x2, y1), Point(x1, y2), Point(x2, y2)]
        return any(region.contains(corner) for corner in corners)
        
    elif strategy == STRATEGY_ALL_CORNERS:
        corners = [Point(x1, y1), Point(x2, y1), Point(x1, y2), Point(x2, y2)]
        return all(region.contains(corner) for corner in corners)
        
    elif strategy == STRATEGY_OVERLAP:
        return region.intersects(bbox_poly)
        
    elif strategy == STRATEGY_CONTAINS:
        return region.contains(bbox_poly)
        
    elif strategy == STRATEGY_PERCENTAGE:
        if region.intersects(bbox_poly):
            intersection = region.intersection(bbox_poly)
            overlap_ratio = intersection.area / bbox_poly.area
            return overlap_ratio >= overlap_threshold
        else:
            return False
    else:
        return False
    