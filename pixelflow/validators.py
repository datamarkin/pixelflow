import math

from typing import List, Optional
from shapely.geometry import Polygon as Shapely_Polygon


# How much precision pixelflow keeps on each numeric field. Truncating bbox
# coordinates to whole pixels shifts every box half a pixel in a fixed
# direction, which reads as a systematic translation rather than noise and
# costs ~1 mAP concentrated in the high-IoU bins and on small objects. Two
# decimals measures indistinguishable from full precision while keeping
# float32 artefacts like 10.699999809265137 out of the JSON payload.
BBOX_DECIMALS = 2
CONFIDENCE_DECIMALS = 3


def validate_bbox(bbox):
    """
    Ensure that bbox contains exactly 4 finite coordinates.
    If bbox is not valid, return None.

    Coordinates are returned as floats rounded to ``BBOX_DECIMALS`` places,
    matching the ``List[float]`` that Detection declares. Non-finite values
    (NaN, ±inf) are rejected rather than passed through: they would poison IoU
    maths silently, and strict JSON encoders refuse to serialize them.

    Accepts any 4-element sequence — list, tuple, or numpy array — since
    framework converters routinely hand back numpy rows.
    """
    # str/bytes are sequences too, but never a valid bbox.
    if isinstance(bbox, (str, bytes)):
        return None

    # Unpacking materializes the sequence and length-checks it in one step:
    # a non-iterable raises TypeError, a wrong length raises ValueError.
    try:
        x1, y1, x2, y2 = bbox
    except (TypeError, ValueError):
        return None

    coords = []
    try:
        for value in (x1, y1, x2, y2):
            coord = float(value)
            if not math.isfinite(coord):
                return None
            coords.append(round(coord, BBOX_DECIMALS))
    except (ValueError, TypeError):
        return None

    return coords


def validate_masks(mask: Optional[list]) -> Optional[list]:
    """
    Validates that the mask data is a list of lists of tuples (each tuple representing a polygon).
    Each tuple must contain exactly two integers.

    Args:
        mask (list): The mask data to be validated, expected to be a list of lists of tuples of integers.

    Returns:
        list: The validated mask data if all values are lists of tuples containing two integers.
        None: If the data is not in the expected format.
    """
    if not isinstance(mask, list):
        return None

    for sublist in mask:
        if not isinstance(sublist, list):
            return None
        for value in sublist:
            if not (isinstance(value, tuple) and len(value) == 2 and all(isinstance(i, int) for i in value)):
                return None

    return mask


def convert_flattened_to_tuples(flat_list: list) -> list:
    """
    Converts a flattened list of coordinates into a list of tuples (polygon points).
    Each tuple contains two integers representing a point.

    Args:
        flat_list (list): A flattened list of coordinates (e.g., [x1, y1, x2, y2, ...]).

    Returns:
        list: A list of tuples containing integer polygon points.
    """
    # Ensure the list length is even to form coordinate pairs
    if len(flat_list) % 2 != 0:
        raise ValueError("The flattened list must contain an even number of values to form coordinate pairs.")

    return [(round(flat_list[i]), round(flat_list[i + 1])) for i in range(0, len(flat_list), 2)]


def convert_datamarkin_masks(mask: list) -> list:
    """
    Converts a list of lists of flattened coordinates into lists of tuples (polygon points).
    Each sublist will be processed separately, with each tuple containing two integers representing a point.

    Args:
        mask (list): The mask data to be converted, expected to be a list of lists of floats.

    Returns:
        list: A list of lists of tuples containing integer polygon points.
    """
    validated_mask = [convert_flattened_to_tuples(sublist) for sublist in mask]
    return validated_mask


def round_to_decimal(value, decimals=CONFIDENCE_DECIMALS):
    """
    Rounds the given value to the specified number of decimal places.

    Args:
        value (float or None): The value to be rounded.
        decimals (int): The number of decimal places (default CONFIDENCE_DECIMALS).

    Returns:
        float or None: The rounded value or None if the input is None.
    """
    if value is not None:
        return round(float(value), decimals)
    return None


def simplify_polygon(polygon_points: list, tolerance: float = 2.0, preserve_topology: bool = True) -> list:
    """
    Simplifies a single polygon using Shapely.

    Args:
        polygon_points (list): A list of tuples representing a polygon.
        tolerance (float): The tolerance factor for simplification (higher = more simplified).
        preserve_topology (bool): If True, the function will try to preserve the polygon's topology.

    Returns:
        list: A simplified polygon represented as a list of tuples.
    """
    # Convert the list of tuples to a Shapely Polygon
    polygon = Shapely_Polygon(polygon_points)

    # Simplify the polygon using the specified tolerance
    simplified_polygon = polygon.simplify(tolerance=tolerance, preserve_topology=preserve_topology)

    # Return the simplified coordinates as a list of tuples
    return list(simplified_polygon.exterior.coords)


def simplify_polygons(polygons: list, tolerance: float = 1.0, preserve_topology: bool = True) -> list:
    """
    Simplifies a list of polygons using Shapely by calling the simplify_single_polygon function.

    Args:
        polygons (list): A list of lists of tuples representing multiple polygons.
        tolerance (float): The tolerance factor for simplification (higher = more simplified).
        preserve_topology (bool): If True, the function will try to preserve the polygons' topology.

    Returns:
        list: A list of simplified polygons, each represented as a list of tuples.
    """
    simplified_polygons = []

    for polygon_points in polygons:
        # Call the simplify_single_polygon function for each polygon
        simplified_polygons.append(simplify_polygon(polygon_points, tolerance, preserve_topology))

    return simplified_polygons
