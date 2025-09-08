"""
Blur annotator for privacy-preserving object detection visualization.

This module provides Gaussian blur effects for detected regions, commonly used
for privacy protection, aesthetic effects, or focus redirection in computer vision applications.
The blur effect maintains natural appearance while obscuring sensitive details.
"""

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from .utils import _get_adaptive_params


def blur(
    image: np.ndarray, 
    detections: 'Detections', 
    kernel_size: Optional[int] = None, 
    padding_percent: float = 0.05
) -> np.ndarray:
    """
    Applies Gaussian blur effect to detected regions for privacy protection.
    
    Creates a natural-looking privacy filter by applying Gaussian blur to detected object
    regions while preserving the rest of the image. The function automatically handles
    boundary conditions and provides adaptive kernel sizing for optimal results.
    
    Args:
        image (np.ndarray): Input image as BGR or RGB array with shape (H, W, C).
                           Must be a valid NumPy array.
        detections (Detections): PixelFlow Detections object containing bounding boxes.
                                Each detection must have a 'bbox' attribute with 
                                (x1, y1, x2, y2) coordinates in pixel units.
        kernel_size (Optional[int]): Size of the Gaussian blur kernel in pixels.
                                   Larger values create stronger blur effect.
                                   Must be positive and odd. If None, uses adaptive 
                                   sizing based on image dimensions. Default is None.
        padding_percent (float): Additional padding around each detection as percentage
                               of bounding box size. Range: 0.0-0.5. Default is 0.05 
                               (5% padding on each side).
        
    Returns:
        np.ndarray: Image with blurred regions where objects were detected.
                   The input image is modified in-place for memory efficiency.
    
    Raises:
        AssertionError: If image is not a NumPy array.
        AttributeError: If detections objects don't have required 'bbox' attribute.
        IndexError: If bbox coordinates are invalid or outside image bounds.
    
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Load image and get model predictions
        >>> image = cv2.imread("faces_in_crowd.jpg")
        >>> model = YOLO("yolo11n.pt")
        >>> outputs = model.predict(image)  # Raw model outputs
        >>> detections = pf.results.from_ultralytics(outputs)  # Convert to PixelFlow format
        >>> 
        >>> # Apply blur with default adaptive settings
        >>> blurred_image = pf.annotators.blur(image, detections)
        >>> 
        >>> # Apply stronger blur with custom kernel size
        >>> strong_blur = pf.annotators.blur(image, detections, kernel_size=25)
        >>> 
        >>> # Apply blur with expanded padding for better coverage
        >>> padded_blur = pf.annotators.blur(image, detections, padding_percent=0.15)
        >>> 
        >>> # Minimal blur for subtle privacy protection
        >>> subtle_blur = pf.annotators.blur(image, detections, kernel_size=7, padding_percent=0.02)
    
    Notes:
        - Modifies the input image in-place for memory efficiency
        - Automatically adapts kernel size based on image dimensions when not specified
        - Even kernel sizes are automatically converted to odd numbers (required for Gaussian blur)
        - Skips regions smaller than the kernel size to prevent processing errors
        - Padding is automatically clamped to range [0.0, 0.5] to prevent excessive expansion
        - Bounding boxes are clipped to image boundaries to handle edge cases
        
    Performance Notes:
        - Uses OpenCV's highly optimized GaussianBlur implementation
        - Efficient for real-time video processing applications  
        - Processing time scales linearly with number of detections and blur strength
        - Memory usage is minimal due to in-place modification
        
    See Also:
        pixelate : Alternative privacy protection method using pixelation effects
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


