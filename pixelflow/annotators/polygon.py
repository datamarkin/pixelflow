from typing import TYPE_CHECKING, Optional, List, Tuple

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from .utils import _get_adaptive_params
from ..colors import _get_color_for_prediction


def polygon(
    image: np.ndarray, 
    detections: 'Detections', 
    thickness: Optional[int] = None, 
    colors: Optional[List[Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draw polygon outlines on detected objects with segmentation masks.
    
    Renders precise polygon boundaries around detected objects using their segmentation 
    data. Automatically adapts line thickness based on image size and supports custom 
    color schemes for visual distinction between different object classes.
    
    Args:
        image (np.ndarray): Input image to draw polygons on. Must be a valid BGR image array.
        detections (Detections): Detections object containing segmentation data.
                                Each detection must have a 'segments' attribute with polygon coordinates.
        thickness (Optional[int]): Line thickness for polygon outlines in pixels.
                                 If None, automatically calculated based on image dimensions.
                                 Range: [1-50]. Default is adaptive (typically 1-6).
        colors (Optional[List[Tuple[int, int, int]]]): List of BGR color tuples to override default colors.
                                                     Colors are cycled through unique class_ids in order of appearance.
                                                     Each tuple should be (B, G, R) with values [0-255].
                                                     If None, uses default ColorManager colors.
        
    Returns:
        np.ndarray: Image with polygon outlines drawn. Modifies the input image in-place
                   and returns the same array reference.
    
    Raises:
        AssertionError: If image is not a numpy array.
        AttributeError: If detections don't contain segments data.
        ValueError: If polygon coordinates are invalid or out of bounds.
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> 
        >>> # Load image and run segmentation model
        >>> image = cv2.imread("path/to/image.jpg")
        >>> model = YOLO("yolo11n-seg.pt")  # Segmentation model
        >>> outputs = model.predict(image)
        >>> results = pf.results.from_ultralytics(outputs)
        >>> 
        >>> # Draw polygon outlines with default settings
        >>> annotated = pf.annotators.polygon(image, results)
        >>> 
        >>> # Customize line thickness and colors
        >>> custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red
        >>> annotated = pf.annotators.polygon(image, results, thickness=3, colors=custom_colors)
        >>> 
        >>> # Use adaptive thickness on high-resolution image
        >>> annotated = pf.annotators.polygon(image, results, thickness=None)
    
    Notes:
        - Modifies the input image in-place for memory efficiency
        - Polygon coordinates are automatically converted to integer format for OpenCV
        - Line thickness is adaptively calculated based on image size when not specified
        - Colors are assigned consistently based on class_id to maintain visual coherence
        - Requires segmentation model outputs (not just bounding boxes)
        
    See Also:
        filled_polygon : Fill polygon areas instead of drawing outlines
        box : Draw bounding boxes around detections
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."
    
    # Get adaptive thickness if not specified
    if thickness is None:
        params = _get_adaptive_params(image)
        thickness = params['thickness']

    for result in detections:
        # Iterate over the segments in the result
        # Convert the points to a NumPy array and reshape for OpenCV
        polygon = np.array(result.segments, dtype=np.int32).reshape((-1, 1, 2))
        # Draw the polygon on the canvas
        color = _get_color_for_prediction(result, colors)
        cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=thickness)

    return image