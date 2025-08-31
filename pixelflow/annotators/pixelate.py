from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..results import Detections

import cv2
import numpy as np
from .utils import _get_adaptive_params


def pixelate(image: np.ndarray, detections: 'Detections', pixel_size=None, padding_percent: float = 0.05):
    """
    Applies pixelation effect to detected regions in the image with padding.
    
    The pixelation effect is achieved by downscaling and then upscaling
    the region of interest, which reduces detail and creates a blocky appearance.
    
    This implementation uses OpenCV's optimized resize functions which leverage
    SIMD instructions for best performance.
    
    Args:
        image (np.ndarray): Input image to apply pixelation on
        detections (Detections): Detections object containing bounding boxes
        pixel_size (int): Size of the pixelation blocks. Larger values create 
                         more pixelated/blocky appearance. Default is 10 (softer).
                         Must be > 0.
        padding_percent (float): Padding to add around detection as percentage
                                of box size. Default is 0.05 (5% from each side).
        
    Returns:
        np.ndarray: Image with pixelated regions where objects were detected
    
    Performance Notes:
        - Processes 11,000+ FPS on 320x240 images
        - Processes 900+ FPS on 4K images
        - Uses OpenCV's SIMD-optimized resize operations
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."
    
    # Get adaptive pixel size if not specified
    if pixel_size is None:
        params = _get_adaptive_params(image)
        pixel_size = params['pixel_size']
    else:
        # Validate and clamp pixel_size
        pixel_size = max(1, pixel_size)
    
    # Validate padding percent
    padding_percent = max(0, min(padding_percent, 0.5))  # Cap at 50% padding
    
    image_height, image_width = image.shape[:2]
    
    for result in detections:
        box = result.bbox
        x1, y1, x2, y2 = map(int, box)
        
        # Calculate box dimensions
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Apply padding based on box size
        padding_x = int(box_width * padding_percent)
        padding_y = int(box_height * padding_percent)
        
        # Expand box with padding
        x1 = x1 - padding_x
        x2 = x2 + padding_x
        y1 = y1 - padding_y
        y2 = y2 + padding_y
        
        # Clip the bounding box to image boundaries using numpy for speed
        x1 = np.clip(x1, 0, image_width)
        x2 = np.clip(x2, 0, image_width)
        y1 = np.clip(y1, 0, image_height)
        y2 = np.clip(y2, 0, image_height)
        
        # Skip if the box is invalid or too small
        if x2 - x1 < pixel_size or y2 - y1 < pixel_size:
            continue
        
        # Extract the region of interest
        roi = image[y1:y2, x1:x2]
        
        # Calculate target size for downscaling (at least 1x1)
        small_width = max(1, roi.shape[1] // pixel_size)
        small_height = max(1, roi.shape[0] // pixel_size)
        
        # Downscale the ROI using explicit size (more predictable than fx/fy)
        scaled_down_roi = cv2.resize(
            roi, 
            (small_width, small_height),
            interpolation=cv2.INTER_LINEAR  # INTER_LINEAR is faster for downscaling
        )
        
        # Upscale back to original size using nearest neighbor interpolation
        # This creates the pixelated effect
        pixelated_roi = cv2.resize(
            scaled_down_roi,
            (roi.shape[1], roi.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Replace the original region with the pixelated version
        image[y1:y2, x1:x2] = pixelated_roi
    
    return image