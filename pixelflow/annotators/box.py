from typing import List, Optional
import cv2
import numpy as np
from .utils import _get_adaptive_params


def box(
    image: np.ndarray, 
    results: List, 
    thickness: Optional[int] = None, 
    colors: Optional[List[tuple]] = None,
    filled: bool = False,
    opacity: Optional[float] = None
) -> np.ndarray:
    """
    Draw bounding boxes on detected objects.
    
    Simple and efficient bounding box visualization for object detection results.
    Automatically adapts line thickness based on image dimensions for optimal visibility.
    Can optionally draw filled boxes with opacity instead of or in addition to borders.
    
    Args:
        image (np.ndarray): Input image to draw boxes on (BGR format)
        results (List): List of detection results containing bounding boxes.
                       Each result must have a 'bbox' attribute with (x1, y1, x2, y2) coordinates.
        thickness (Optional[int]): Line thickness for bounding boxes in pixels.
                                  If None, automatically determined based on image size.
        colors (Optional[List[tuple]]): List of BGR color tuples to override default colors.
                                       Colors are mapped to unique class_ids in order of appearance.
                                       If None, uses default ColorManager colors.
        filled (bool): If True, draws filled boxes with opacity instead of borders.
                      Default is False (draws only borders).
        opacity (Optional[float]): Fill opacity when filled=True. Range: [0.0-1.0].
                                  Only used when filled=True. If None, auto-determined.
    
    Returns:
        np.ndarray: Image with bounding boxes drawn. The input image is modified in-place.
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> 
        >>> # Load image and get model predictions
        >>> image = cv2.imread("path/to/image.jpg")
        >>> outputs = model.predict(image)  # Raw model outputs
        >>> results = pf.results.from_ultralytics(outputs)  # Convert to PixelFlow format
        >>> 
        >>> # Draw boxes with default colors (borders only)
        >>> annotated = pf.annotate.box(image, results)
        >>> 
        >>> # Draw filled boxes with automatic opacity
        >>> annotated = pf.annotate.box(image, results, filled=True)
        >>> 
        >>> # Draw filled boxes with custom opacity
        >>> annotated = pf.annotate.box(image, results, filled=True, opacity=0.3)
        >>> 
        >>> # Override with custom colors for specific classes
        >>> custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red
        >>> annotated = pf.annotate.box(image, results, colors=custom_colors)
        >>> 
        >>> # Use custom thickness for smaller images
        >>> annotated = pf.annotate.box(image, results, thickness=1)
    """
    # If filled is True, delegate to filled_box function
    if filled:
        from .filled_box import filled_box
        return filled_box(image, results, opacity=opacity, colors=colors)
    
    from ..colors import get_color_for_prediction
    from .. import draw
    
    # Get adaptive thickness if not specified
    if thickness is None:
        params = _get_adaptive_params(image)
        thickness = params['thickness']
    
    for result in results:
        box = result.bbox
        x1, y1, x2, y2 = map(int, box)

        color = get_color_for_prediction(result, colors)

        draw.rectangle(image, (x1, y1), (x2, y2), line_color=color, thickness=thickness)

    return image