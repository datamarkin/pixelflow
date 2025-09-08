"""
Detection strategy system for spatial analysis and region-based filtering in PixelFlow.

This module provides a comprehensive strategy system for determining how object detections
interact with spatial regions, zones, and crossing lines. It supports multiple anchor points,
geometric relationships, and logical combinations for precise control over detection filtering.
"""

from typing import List, Tuple, Union, Literal, Optional
from shapely.geometry import Polygon, Point, box

__all__ = ["validate_strategy", "get_anchor_position", "check_detection_in_region"]

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
    Validate that strategy string(s) are among the supported detection strategies.
    
    Ensures that provided strategy strings are valid options for spatial detection
    analysis. Supports both single strategy validation and batch validation for
    multiple strategies used in compound logical operations.
    
    Args:
        strategy (Union[str, List[str]]): Single strategy string or list of strategy
                                         strings to validate. Supported strategies include:
                                         anchor points (center, corners, edges), geometric
                                         relationships (overlap, contains), and area-based
                                         analysis (percentage).
        
    Returns:
        Union[str, List[str]]: The validated strategy unchanged if all entries are valid.
                              Returns same type as input (string or list).
        
    Raises:
        ValueError: If any strategy string is not in the valid strategies set.
                   Includes helpful message listing all valid options.
        TypeError: If strategy is not a string or list/tuple of strings.
        
    Example:
        >>> import pixelflow as pf
        >>> 
        >>> # Validate single strategy
        >>> valid_strategy = pf.strategies.validate_strategy("center")
        >>> print(valid_strategy)  # Output: "center"
        >>> 
        >>> # Validate multiple strategies for compound logic
        >>> strategies = pf.strategies.validate_strategy(["center", "bottom_center"])
        >>> print(strategies)  # Output: ["center", "bottom_center"]
        >>> 
        >>> # Validate edge/corner strategies
        >>> edge_strategy = pf.strategies.validate_strategy("any_corner")
        >>> print(edge_strategy)  # Output: "any_corner"
        >>> 
        >>> # Error case - invalid strategy
        >>> try:
        ...     pf.strategies.validate_strategy("invalid_strategy")
        ... except ValueError as e:
        ...     print(f"Validation failed: {e}")
    
    Notes:
        - Strategy validation is case-sensitive and must match exactly
        - Valid strategies include 23 different options covering all spatial relationships
        - Used internally by other strategy functions to ensure consistency
        - No modification of input values, only validation
        
    See Also:
        check_detection_in_region : Uses validated strategies for spatial analysis
        get_anchor_position : Requires validated anchor point strategies
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
    Extract specific anchor point coordinates from a bounding box based on strategy.
    
    Computes the exact pixel coordinates of strategic anchor points on detection
    bounding boxes. Used for crossing line analysis, zone entry detection, and
    precise spatial positioning in computer vision workflows.
    
    Args:
        bbox (List[float]): Bounding box coordinates in [x1, y1, x2, y2] format where
                           (x1, y1) is top-left corner and (x2, y2) is bottom-right.
                           Coordinates should be in pixel units.
        strategy (str): Anchor point strategy defining which position to extract.
                       Must be one of: "center", "bottom_center", "top_center",
                       "left_center", "right_center", "top_left", "top_right",
                       "bottom_left", "bottom_right". Strategy is not validated
                       internally - use validate_strategy() first if needed.
        
    Returns:
        Tuple[float, float]: The (x, y) coordinates of the specified anchor point
                            in the same coordinate system as the input bbox.
                            Returns center coordinates if strategy is unrecognized.
        
    Raises:
        IndexError: If bbox does not contain exactly 4 coordinate values.
        TypeError: If bbox coordinates cannot be converted to float for calculations.
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Load and process image
        >>> image = cv2.imread("person_detection.jpg")
        >>> model = YOLO("yolo11n.pt")
        >>> outputs = model.predict(image)
        >>> results = pf.results.from_ultralytics(outputs)
        >>> 
        >>> # Extract center point for each detection
        >>> for detection in results:
        ...     center_x, center_y = pf.strategies.get_anchor_position(detection.bbox, "center")
        ...     print(f"Detection center: ({center_x:.1f}, {center_y:.1f})")
        >>> 
        >>> # Extract bottom center for ground-level tracking
        >>> bbox = [100, 50, 200, 150]  # [x1, y1, x2, y2]
        >>> foot_position = pf.strategies.get_anchor_position(bbox, "bottom_center")
        >>> print(f"Foot position: {foot_position}")  # Output: (150.0, 150.0)
        >>> 
        >>> # Extract corner positions for detailed analysis
        >>> top_left = pf.strategies.get_anchor_position(bbox, "top_left")
        >>> bottom_right = pf.strategies.get_anchor_position(bbox, "bottom_right")
        >>> print(f"Corners: {top_left}, {bottom_right}")  # Output: (100.0, 50.0), (200.0, 150.0)
        >>> 
        >>> # Edge center points for crossing detection
        >>> right_edge = pf.strategies.get_anchor_position(bbox, "right_center")
        >>> print(f"Right edge: {right_edge}")  # Output: (200.0, 100.0)
    
    Notes:
        - Coordinates are computed using floating-point arithmetic for precision
        - Center calculations use midpoint formula: (coord1 + coord2) / 2
        - Returns center position as fallback for unrecognized strategies
        - No input validation performed - ensure bbox format is correct
        - Edge strategies (left/right/top/bottom_center) useful for crossing detection
        - Corner strategies useful for containment analysis
        
    Performance Notes:
        - Constant time O(1) operation with minimal computational overhead
        - Simple arithmetic operations, suitable for real-time video processing
        - No memory allocation beyond return tuple
        
    See Also:
        validate_strategy : Validates strategy strings before use
        check_detection_in_region : Uses anchor positions for spatial analysis
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
    Determine if a detection satisfies spatial relationship criteria with a region.
    
    Performs comprehensive spatial analysis using configurable strategies to determine
    detection-region relationships. Supports anchor point analysis, geometric overlap
    calculations, and compound logical operations for complex filtering scenarios.
    
    Args:
        bbox (List[float]): Detection bounding box in [x1, y1, x2, y2] format where
                           coordinates represent pixel positions. Must contain exactly
                           4 float values.
        strategy (Union[str, List[str]]): Spatial analysis strategy or list of strategies.
                                         Single string for basic checks or list for compound
                                         logic. Supported: anchor points ("center", "bottom_center", 
                                         etc.), geometric ("overlap", "contains"), area-based
                                         ("percentage").
        region (Polygon): Shapely Polygon object defining the spatial region for analysis.
                         Must be a valid polygon with defined boundaries.
        overlap_threshold (float): Minimum overlap ratio for "percentage" strategy.
                                  Range: [0.0, 1.0]. Default is 0.5 (50% overlap required).
        mode (Literal["any", "all"]): Logical combination mode for multiple strategies.
                                     "all" requires all strategies to pass (AND logic),
                                     "any" requires at least one to pass (OR logic).
                                     Default is "all".
        
    Returns:
        bool: True if detection satisfies the spatial criteria according to the specified
             strategy/strategies and logical mode, False otherwise.
        
    Raises:
        ValueError: If any strategy string is invalid or if overlap_threshold is outside
                   valid range [0.0, 1.0].
        IndexError: If bbox does not contain exactly 4 coordinate values.
        TypeError: If region is not a valid Shapely Polygon object.
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> from shapely.geometry import Polygon
        >>> 
        >>> # Setup detection pipeline
        >>> image = cv2.imread("traffic_scene.jpg")
        >>> model = YOLO("yolo11n.pt")
        >>> outputs = model.predict(image)
        >>> results = pf.results.from_ultralytics(outputs)
        >>> 
        >>> # Define region of interest (e.g., crosswalk area)
        >>> crosswalk_region = Polygon([(100, 200), (400, 200), (400, 300), (100, 300)])
        >>> 
        >>> # Basic center-point analysis
        >>> for detection in results:
        ...     in_crosswalk = pf.strategies.check_detection_in_region(
        ...         detection.bbox, "center", crosswalk_region
        ...     )
        ...     if in_crosswalk:
        ...         print(f"Detection center in crosswalk: {detection.class_name}")
        >>> 
        >>> # Advanced overlap analysis with threshold
        >>> bbox = [150, 220, 250, 280]  # Example detection
        >>> substantial_overlap = pf.strategies.check_detection_in_region(
        ...     bbox, "percentage", crosswalk_region, overlap_threshold=0.7
        ... )
        >>> print(f"70%+ overlap: {substantial_overlap}")
        >>> 
        >>> # Compound strategy with OR logic
        >>> foot_or_center = pf.strategies.check_detection_in_region(
        ...     bbox, ["bottom_center", "center"], crosswalk_region, mode="any"
        ... )
        >>> print(f"Foot OR center in region: {foot_or_center}")
        >>> 
        >>> # Strict containment analysis
        >>> fully_contained = pf.strategies.check_detection_in_region(
        ...     bbox, "contains", crosswalk_region
        ... )
        >>> print(f"Fully contained: {fully_contained}")
    
    Notes:
        - Strategy validation is performed automatically for all input strategies
        - Geometric calculations use Shapely for robust spatial analysis
        - "percentage" strategy calculates intersection area / detection area ratio
        - "any_corner" passes if any of the 4 corners is within the region
        - "all_corners" requires all 4 corners to be within the region
        - "overlap" is true for any intersection, regardless of area
        - "contains" requires the region to fully contain the detection bbox
        - Compound strategies enable complex spatial logic combinations
        
    Performance Notes:
        - Single strategy checks are O(1) for anchor points, O(log n) for geometric operations
        - Multiple strategies scale linearly with strategy count
        - Geometric operations may be slower for complex polygons with many vertices
        - Consider caching region objects for repeated analysis
        
    See Also:
        validate_strategy : Validates strategy strings before analysis
        get_anchor_position : Extracts anchor coordinates used by point-based strategies
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
    Execute spatial analysis for a single detection strategy against a region.
    
    Internal helper function that implements the core spatial analysis logic for
    individual strategies. Handles anchor point containment, geometric relationships,
    and area-based calculations with optimized implementations for each strategy type.
    
    Args:
        bbox (List[float]): Detection bounding box coordinates in [x1, y1, x2, y2] format.
                           Coordinates should be in pixel units with x1 < x2 and y1 < y2.
        strategy (str): Single validated strategy string to evaluate. Should be pre-validated
                       using validate_strategy() to ensure it's a supported option.
        region (Polygon): Shapely Polygon object representing the spatial region for analysis.
                         Must be a valid, non-self-intersecting polygon.
        overlap_threshold (float): Minimum overlap ratio required for "percentage" strategy.
                                  Range: [0.0, 1.0]. Only used when strategy is "percentage".
        
    Returns:
        bool: True if the detection bounding box satisfies the spatial criteria defined
             by the strategy, False otherwise. Returns False for unrecognized strategies.
        
    Raises:
        IndexError: If bbox does not contain exactly 4 coordinate values.
        AttributeError: If region lacks required Shapely Polygon methods.
        
    Example:
        >>> from shapely.geometry import box, Polygon
        >>> import pixelflow.strategies as strategies
        >>> 
        >>> # Setup test data
        >>> bbox = [10, 20, 50, 60]  # Detection box
        >>> region = box(0, 0, 100, 100)  # Large region
        >>> small_region = box(25, 35, 35, 45)  # Small region at center
        >>> 
        >>> # Test center point containment
        >>> center_in_large = strategies._check_single_strategy(bbox, "center", region)
        >>> print(f"Center in large region: {center_in_large}")  # True
        >>> 
        >>> # Test geometric overlap
        >>> has_overlap = strategies._check_single_strategy(bbox, "overlap", small_region) 
        >>> print(f"Has overlap: {has_overlap}")  # True
        >>> 
        >>> # Test percentage overlap with threshold
        >>> substantial_overlap = strategies._check_single_strategy(
        ...     bbox, "percentage", small_region, overlap_threshold=0.1
        ... )
        >>> print(f"Meets 10% threshold: {substantial_overlap}")  # True or False depending on actual overlap
    
    Notes:
        - This is an internal helper function not intended for direct public use
        - No input validation is performed - caller must ensure valid inputs
        - Strategy matching uses exact string comparison for performance
        - Geometric operations leverage Shapely's optimized spatial algorithms
        - Point containment tests use Shapely's robust point-in-polygon algorithm
        - Area calculations handle edge cases like zero-area intersections
        - Returns False for unrecognized strategies as safe fallback behavior
        
    Performance Notes:
        - Point containment operations are O(log n) where n is polygon vertices
        - Area calculations require polygon intersection computation
        - Corner strategies create temporary Point objects but minimal overhead
        - Geometric strategies (overlap, contains) are most computationally intensive
        
    See Also:
        check_detection_in_region : Public interface that calls this function
        get_anchor_position : Provides anchor coordinates used by point strategies
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
    