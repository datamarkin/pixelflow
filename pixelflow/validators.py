"""Validation and precision policy for the values a Detection stores.

Every pixel coordinate pixelflow keeps - bounding boxes, keypoints and polygon
vertices - rounds to the same precision here, so no single converter or
transform gets to pick its own. Malformed geometry degrades to None for the
Optional fields; see round_coord for why a KeyPoint raises instead.
"""

import math

import numpy as np
from shapely.geometry import Polygon as Shapely_Polygon


# How much precision pixelflow keeps. Truncating pixel coordinates to whole
# numbers shifts every one of them toward zero, which reads as a systematic
# translation rather than as noise and costs ~1 mAP concentrated in the
# high-IoU bins and on small objects. Two decimals measures indistinguishable
# from full precision while keeping float32 artefacts like 10.699999809265137
# out of the JSON payload.
COORD_DECIMALS = 2
CONFIDENCE_DECIMALS = 3


def round_coord(value):
    """
    Round one pixel coordinate to ``COORD_DECIMALS`` places.

    Bounding boxes, keypoints and polygon vertices all measure the same thing,
    so they all round here rather than each picking their own precision.

    Raises on anything that is not a finite number: NaN and ±inf poison IoU
    maths silently, and strict JSON encoders refuse to serialize them. What to
    do about that is the caller's to decide - validate_bbox degrades to None,
    KeyPoint lets it propagate.
    """
    coord = float(value)
    if not math.isfinite(coord):
        raise ValueError(f"pixel coordinate must be finite, got {value!r}")
    return round(coord, COORD_DECIMALS)


def validate_bbox(bbox):
    """
    Ensure that bbox contains exactly 4 finite coordinates.
    If bbox is not valid, return None.

    Coordinates come back as floats at the shared pixel precision, matching the
    ``List[float]`` that Detection declares.

    Accepts any 4-element sequence — list, tuple, or numpy array — since
    framework converters routinely hand back numpy rows.
    """
    # str/bytes are sequences too, but never a valid bbox.
    if isinstance(bbox, (str, bytes)):
        return None

    try:
        # Unpacking checks iterability and length in one step; round_coord
        # checks that each element is a finite number.
        x1, y1, x2, y2 = bbox
        return [round_coord(v) for v in (x1, y1, x2, y2)]
    except (TypeError, ValueError):
        return None


def validate_segments(segments):
    """
    Normalize one polygon to a list of ``[x, y]`` points at the shared precision.
    If the polygon is not valid, return None.

    Converters hand polygons over in whichever shape their framework used - an
    ``(N, 2)`` numpy array, a list of tuples, a list of lists. Settling on one
    shape here is what lets every caller that moves a polygon write a single
    comprehension instead of branching on how it arrived.

    Rounding here also keeps float32 vertices from reaching ``tolist()``, where
    the nearest float32 to 10.7 prints as 10.699999809265137.

    One polygon, not several: an instance split by occlusion keeps its extra
    parts in ``masks``, and a nested list of polygons is rejected rather than
    flattened. If that ever needs to change, this is the function to change.
    """
    if segments is None:
        return None

    try:
        points = np.asarray(segments, dtype=float)
    except (TypeError, ValueError):
        return None

    # A polygon is N points of (x, y). `size` rather than `ndim` alone catches
    # the vertexless (0, 2) array that ultralytics emits for an empty mask.
    if points.ndim != 2 or points.shape[1] != 2 or points.size == 0:
        return None
    if not np.isfinite(points).all():
        return None

    return points.round(COORD_DECIMALS).tolist()


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
