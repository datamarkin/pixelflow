from typing import TYPE_CHECKING, Optional, List, Tuple, Union

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from .utils import _get_adaptive_params
from ..colors import _get_color_for_prediction


def oval(
    image: np.ndarray, 
    detections: 'Detections', 
    thickness: Optional[int] = None, 
    start_angle: int = -45, 
    end_angle: int = 235, 
    colors: Optional[List[Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draws elliptical footprints at the bottom of detected objects.
    
    Creates ground-plane footprint visualization by drawing partial ellipses at the bottom 
    center of each bounding box. The elliptical shape provides a natural representation of 
    object presence on the ground plane, making it ideal for spatial awareness, shadow 
    effects, and ground-based tracking visualizations.
    
    Args:
        image (np.ndarray): Input image array with shape (H, W, 3) in BGR color format.
        detections (Detections): PixelFlow detections object containing bounding boxes. 
                                Each detection must have a `.bbox` attribute with 
                                (x1, y1, x2, y2) coordinates.
        thickness (Optional[int]): Line thickness for ellipse drawing in pixels. 
                                  If None, automatically adapts based on image size 
                                  (typically 1-4 pixels). Default is None.
        start_angle (int): Starting angle of the ellipse arc in degrees. 
                          Range: -180 to 180. Default is -45 (bottom-left quadrant).
        end_angle (int): Ending angle of the ellipse arc in degrees. 
                        Range: -180 to 360. Default is 235 (bottom-right, creating 
                        bottom arc spanning ~280 degrees).
        colors (Optional[List[Tuple[int, int, int]]]): List of BGR color tuples 
                                                      (B, G, R) where each value is 
                                                      0-255. Colors are mapped to unique 
                                                      class_ids in order of appearance. 
                                                      If None, uses default ColorManager colors.
        
    Returns:
        np.ndarray: Image array with elliptical footprints drawn at the bottom center 
                   of each detected object. Original image is modified in-place.
    
    Raises:
        AssertionError: If image is not a numpy.ndarray.
        AttributeError: If detections object doesn't have required bbox attributes.
        ValueError: If angle parameters are invalid or cause OpenCV drawing errors.
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Load image and model
        >>> image = cv2.imread("pedestrians.jpg")
        >>> model = YOLO("yolo11n.pt")
        >>> outputs = model.predict(image)
        >>> results = pf.results.from_ultralytics(outputs)
        >>> 
        >>> # Basic footprint annotation with adaptive sizing
        >>> annotated = pf.annotators.oval(image, results)
        >>> 
        >>> # Custom thickness and colors for specific visualization
        >>> custom_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]  # Cyan, Magenta, Yellow
        >>> annotated = pf.annotators.oval(image, results, thickness=3, colors=custom_colors)
        >>> 
        >>> # Full ellipse instead of bottom arc
        >>> full_ellipse = pf.annotators.oval(image, results, start_angle=0, end_angle=360)
        >>> 
        >>> # Narrow bottom arc for subtle ground indication
        >>> subtle_footprint = pf.annotators.oval(image, results, start_angle=30, end_angle=150)
    
    Notes:
        - Ellipse width automatically matches the full width of each bounding box
        - Ellipse height is set to 25% of the width for natural ground-plane proportions
        - Center point is positioned at the bottom-center of each bounding box (x_center, y2)
        - Uses anti-aliased line drawing (cv2.LINE_AA) for smooth curves
        - Thickness automatically adapts to image resolution when not specified
        - Colors cycle through the provided list if there are more detections than colors
        - Angles follow OpenCV convention: 0° = right, 90° = down, 180° = left, 270° = up
        
    See Also:
        box : Draw rectangular bounding boxes around detections
        blur : Apply blur effects to detected regions
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."
    
    # Get adaptive thickness if not specified
    if thickness is None:
        params = _get_adaptive_params(image)
        thickness = params['thickness']
    
    for result in detections:
        box = result.bbox
        x1, y1, x2, y2 = map(int, box)
        
        # Get color for this detection
        color = _get_color_for_prediction(result, colors)
        
        # Calculate ellipse parameters
        center = (int((x1 + x2) / 2), y2)  # Bottom center of bbox
        width = x2 - x1
        height = int(0.25 * width)  # Height is 25% of width for natural look
        
        # Draw the ellipse (partial arc from start_angle to end_angle)
        cv2.ellipse(
            image,
            center=center,
            axes=(int(width / 2), height),  # Semi-major and semi-minor axes
            angle=0.0,
            startAngle=start_angle,
            endAngle=end_angle,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA  # Anti-aliased for smooth curves
        )
    
    return image




# Alias for backward compatibility
footprint = oval