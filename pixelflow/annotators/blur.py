"""
Blur annotator for privacy-preserving object detection visualization.

This module provides Gaussian blur effects for detected regions, commonly used
for privacy protection, aesthetic effects, or focus redirection in computer vision applications.
The blur effect maintains natural appearance while obscuring sensitive details.
"""

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..results import Detections

import cv2
import numpy as np
from .utils import _get_adaptive_params

__all__ = ["blur"]


def blur(
    image: np.ndarray, 
    detections: 'Detections', 
    kernel_size: Optional[int] = None, 
    padding_percent: float = 0.05
) -> np.ndarray:
    """
    Applies blur effect to detected regions in the image with padding.
    
    The blur effect is achieved using Gaussian blur to obscure details
    in detected regions while maintaining a natural appearance.
    
    Args:
        image (np.ndarray): Input image to apply blur on
        detections (Detections): Detections object containing bounding boxes.
                                Each detection must have a 'bbox' attribute with (x1, y1, x2, y2) coordinates.
        kernel_size (Optional[int]): Size of the blur kernel. Larger values create
                                   stronger blur effect. Must be odd and > 0. 
                                   If None, uses adaptive sizing based on image dimensions.
        padding_percent (float): Padding to add around detection as percentage
                               of box size. Range: 0.0-0.5. Default is 0.05 (5% from each side).
        
    Returns:
        np.ndarray: Image with blurred regions where objects were detected.
                   The input image is modified in-place for memory efficiency.
    
    Raises:
        AssertionError: If image is not a NumPy array
        AttributeError: If results objects don't have 'bbox' attribute
        IndexError: If bbox coordinates are invalid or outside image bounds
    
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> 
        >>> # Load image and get model predictions
        >>> image = cv2.imread("path/to/image.jpg")
        >>> outputs = model.predict(image)  # Raw model outputs
        >>> detections = pf.results.from_ultralytics(outputs)  # Convert to PixelFlow format
        >>> 
        >>> # Apply blur with default settings
        >>> blurred_image = pf.annotators.blur(image, detections)
        >>> 
        >>> # Apply stronger blur with custom kernel size
        >>> blurred_image = pf.annotators.blur(image, detections, kernel_size=25)
        >>> 
        >>> # Apply blur with more padding around detections
        >>> blurred_image = pf.annotators.blur(image, detections, padding_percent=0.1)
    
    Notes:
        - Modifies the input image in-place for memory efficiency
        - Automatically adapts kernel size based on image dimensions when not specified
        - Skips regions too small for the specified kernel size
        - Padding is automatically clamped to prevent excessive expansion
        
    Performance Notes:
        - Uses OpenCV's optimized Gaussian blur implementation
        - Efficient for real-time processing
        - Scales well with image size
        
    See Also:
        pixelate : Alternative privacy protection method using pixelation
        oval : Shaped region effects for selective blurring
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."
    
    # Get adaptive kernel size if not specified
    if kernel_size is None:
        params = _get_adaptive_params(image)
        kernel_size = params['blur_kernel']
    else:
        # Validate and ensure kernel_size is odd and positive
        kernel_size = max(1, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Make it odd
    
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
        
        # Clip the bounding box to image boundaries
        x1 = np.clip(x1, 0, image_width)
        x2 = np.clip(x2, 0, image_width)
        y1 = np.clip(y1, 0, image_height)
        y2 = np.clip(y2, 0, image_height)
        
        # Skip if the box is invalid or too small
        if x2 - x1 < kernel_size or y2 - y1 < kernel_size:
            continue
        
        # Extract the region of interest
        roi = image[y1:y2, x1:x2]
        
        # Apply Gaussian blur to the region
        blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        
        # Replace the original region with the blurred version
        image[y1:y2, x1:x2] = blurred_roi
    
    return image


