from typing import List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from ..colors import _get_color_for_prediction


def mask(
    frame: np.ndarray,
    detections: 'Detections',
    opacity: float = 0.5,
    colors: Optional[List[tuple]] = None
) -> np.ndarray:
    """
    Overlay segmentation masks on a video frame with support for both binary masks and polygon formats.
    
    Provides efficient mask visualization using optimized single-pass rendering and OpenCV's 
    accelerated blending operations. Supports both binary mask arrays and polygon coordinate 
    lists, with automatic format detection and conversion for seamless integration.
    
    Args:
        frame (np.ndarray): Input video frame in BGR format (height, width, 3).
                           Modified in-place with overlaid masks.
        detections (Detections): Detections object containing mask data.
                                Each detection's masks attribute can contain binary arrays
                                or polygon coordinate lists.
        opacity (float): Opacity level for blending masks with the frame.
                        Range: [0.0, 1.0]. Default is 0.5 (semi-transparent).
                        Value of 1.0 creates opaque masks, 0.0 makes them invisible.
        colors (Optional[List[tuple]]): List of BGR color tuples to override default colors.
                                       Colors are mapped to unique class_ids in order of appearance.
                                       If None, uses default ColorManager colors.
        
    Returns:
        np.ndarray: Frame with masks overlaid using the specified opacity.
                   The input frame is modified in-place for memory efficiency.
    
    Raises:
        ValueError: If binary mask dimensions do not match frame dimensions.
        AttributeError: If detection objects lack required 'masks' attribute.
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Load image and get segmentation predictions
        >>> image = cv2.imread("path/to/image.jpg")
        >>> model = YOLO("yolo11n-seg.pt")  # Segmentation model
        >>> outputs = model.predict(image)  # Raw model outputs
        >>> detections = pf.results.from_ultralytics(outputs)  # Convert to PixelFlow format
        >>> 
        >>> # Apply masks with default semi-transparent overlay
        >>> annotated = pf.annotators.mask(image, detections)
        >>> 
        >>> # Use custom colors for specific classes
        >>> custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red
        >>> annotated = pf.annotators.mask(image, detections, colors=custom_colors)
        >>> 
        >>> # Create opaque masks for clear segmentation boundaries
        >>> annotated = pf.annotators.mask(image, detections, opacity=1.0)
        >>> 
        >>> # Subtle overlay for background preservation
        >>> annotated = pf.annotators.mask(image, detections, opacity=0.2)
    
    Notes:
        - Input frame is modified in-place for memory efficiency
        - Supports both binary mask arrays (boolean or uint8) and polygon coordinate lists
        - Binary masks must match frame dimensions exactly
        - Polygon coordinates are automatically converted to binary masks using cv2.fillPoly
        - Uses single-pass rendering for optimal performance with multiple masks
        - OpenCV's addWeighted function provides hardware-accelerated blending
        - Empty or None mask data is automatically skipped
        
    Performance Notes:
        - Single overlay creation minimizes memory allocation
        - Batch blending operation reduces computational overhead
        - Direct pixel assignment used for full opacity (opacity=1.0) to bypass blending
        - Polygon-to-mask conversion cached within the function scope
    """
    # Create a single overlay for all masks
    overlay = np.zeros_like(frame, dtype=np.uint8)
    has_mask = np.zeros(frame.shape[:2], dtype=bool)
    
    # Draw all masks to the overlay in a single pass
    for result in detections:
        if result.masks is None:
            continue
            
        color = _get_color_for_prediction(result, colors)
        
        for mask_data in result.masks:
            binary_mask = None
            
            if isinstance(mask_data, np.ndarray):
                # Binary mask format
                if mask_data.dtype == bool:
                    if mask_data.shape[:2] != frame.shape[:2]:
                        raise ValueError(f"Mask dimensions {mask_data.shape[:2]} do not match frame dimensions {frame.shape[:2]}.")
                    binary_mask = mask_data
                else:
                    binary_mask = mask_data.astype(bool)
                    if binary_mask.shape[:2] != frame.shape[:2]:
                        raise ValueError(f"Mask dimensions {binary_mask.shape[:2]} do not match frame dimensions {frame.shape[:2]}.")
            
            elif isinstance(mask_data, list) and len(mask_data) > 0:
                # Polygon format - convert to binary mask
                mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
                points = np.array(mask_data, dtype=np.int32)
                if len(points.shape) == 2 and points.shape[1] == 2:
                    points = points.reshape((-1, 1, 2))
                    cv2.fillPoly(mask_img, [points], 1)
                    binary_mask = mask_img.astype(bool)
            
            if binary_mask is not None:
                # Draw mask to overlay and track which pixels have masks
                overlay[binary_mask] = color
                has_mask |= binary_mask
    
    # Single blend operation for all masks
    if has_mask.any():  # Only blend if there are masks
        if opacity >= 1.0:
            # Direct copy for full opacity
            frame[has_mask] = overlay[has_mask]
        else:
            # Use OpenCV's optimized blending for partial opacity
            blended = cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0)
            frame[has_mask] = blended[has_mask]
    
    return frame