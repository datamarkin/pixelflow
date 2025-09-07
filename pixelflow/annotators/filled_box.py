from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from .utils import _get_adaptive_params
from ..colors import _get_color_for_prediction


def filled_box(
    image: np.ndarray, 
    detections: 'Detections', 
    opacity: Optional[float] = None, 
    colors: Optional[List[tuple]] = None
) -> np.ndarray:
    """
    Draw filled bounding boxes with opacity on detected objects.
    
    Creates semi-transparent filled rectangles for clean overlays.
    Automatically adapts opacity based on image dimensions for optimal visibility.
    
    Args:
        image (np.ndarray): Input image to draw filled boxes on (BGR format)
        detections (Detections): Detections object containing bounding boxes.
                                Each detection must have a 'bbox' attribute with (x1, y1, x2, y2) coordinates.
        opacity (Optional[float]): Fill opacity for bounding boxes.
                                  Range: [0.0-1.0] where 0 is transparent, 1 is opaque.
                                  If None, automatically determined based on image size.
        colors (Optional[List[tuple]]): List of BGR color tuples to override default colors.
                                       Colors are mapped to unique class_ids in order of appearance.
                                       If None, uses default ColorManager colors.
    
    Returns:
        np.ndarray: Image with filled bounding boxes drawn. The input image is modified in-place.
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> 
        >>> # Load image and get model predictions
        >>> image = cv2.imread("path/to/image.jpg")
        >>> outputs = model.predict(image)  # Raw model outputs
        >>> detections = pf.results.from_ultralytics(outputs)  # Convert to PixelFlow format
        >>> 
        >>> # Draw filled boxes with automatic opacity
        >>> annotated = pf.annotate.filled_box(image, detections)
        >>> 
        >>> # Use custom opacity for subtle overlay
        >>> annotated = pf.annotate.filled_box(image, detections, opacity=0.3)
        >>> 
        >>> # Override with custom colors for specific classes
        >>> custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red
        >>> annotated = pf.annotate.filled_box(image, detections, opacity=0.5, colors=custom_colors)
    
    Notes:
        - Filled rectangles are drawn with transparency
        - Opacity is automatically calculated based on image size if not specified
        - Typical auto-calculated opacity ranges from 0.3 to 0.5 for optimal visibility
        - Uses alpha blending for smooth transparency effect
        - Multiple overlapping boxes will create cumulative opacity effect
    """
    # Get adaptive opacity if not specified
    if opacity is None:
        params = _get_adaptive_params(image)
        # Calculate opacity based on image size - larger images get slightly lower opacity
        base_scale = np.sqrt(image.shape[0] * image.shape[1]) / 1000
        opacity = min(0.5, max(0.3, 0.4 - (base_scale - 1) * 0.05))
    
    # Clamp opacity to valid range
    opacity = max(0.0, min(1.0, opacity))
    
    # First draw the filled boxes
    for result in detections:
        bbox = result.bbox
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get color for this prediction
        color = _get_color_for_prediction(result, colors)
        
        # Create overlay for this box
        overlay = image.copy()
        
        # Draw filled rectangle on overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=cv2.FILLED)
        
        # Blend overlay with original image
        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)
    
    return image