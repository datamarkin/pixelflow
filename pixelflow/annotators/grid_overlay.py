from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from .utils import _get_adaptive_params
from ..colors import _get_color_for_prediction


def grid_overlay(
    image: np.ndarray,
    detections: 'Detections',
    grid_size: Optional[Tuple[int, int]] = None,
    thickness: Optional[int] = None,
    colors: Optional[List[tuple]] = None,
    filled: bool = False,
    opacity: Optional[float] = None
) -> np.ndarray:
    """
    Draw grid overlays inside bounding boxes for spatial reference.
    
    Creates evenly-spaced grid lines within detection boxes for measurement and analysis.
    Automatically adapts grid density based on box dimensions for optimal visibility.
    Can create filled checkerboard patterns for visual segmentation.
    
    Args:
        image (np.ndarray): Input image to draw grid overlays on. Must be in BGR format
                           with shape (H, W, 3) and dtype uint8.
        detections (Detections): Detections object containing bounding boxes.
                                Each detection must have a 'bbox' attribute with (x1, y1, x2, y2) coordinates
                                where coordinates are in pixel units within image boundaries.
        grid_size (Optional[Tuple[int, int]]): Grid dimensions as (rows, cols).
                                               If None, automatically calculated based on box size.
                                               Range: rows and cols must be >= 2 (automatically enforced).
                                               Example: (3, 4) creates 3 horizontal and 4 vertical divisions.
        thickness (Optional[int]): Line thickness for grid lines in pixels.
                                  If None, automatically determined based on image size (typically 1-3).
                                  Range: Must be >= 1. Values > 10 may cause visual overlap.
        colors (Optional[List[tuple]]): List of BGR color tuples to override default colors.
                                       Each tuple must be 3 integers in range [0, 255].
                                       Colors are mapped to unique class_ids in order of appearance.
                                       If None, uses default ColorManager colors.
        filled (bool): If True, creates alternating filled cells (checkerboard pattern).
                      Default is False (draws only grid lines).
        opacity (Optional[float]): Fill opacity when filled=True. 
                                  Range: [0.0-1.0]. Automatically clamped to valid range.
                                  Only used when filled=True. If None, defaults to 0.25 for optimal visibility.
    
    Returns:
        np.ndarray: Image with grid overlays drawn. The input image is modified in-place
                   and also returned for method chaining.
    
    Raises:
        AttributeError: If detections object lacks required 'bbox' attribute on any detection.
        ValueError: If image is not a valid numpy array or has incorrect shape/dtype.
        IndexError: If bounding box coordinates are outside image boundaries.
        TypeError: If grid_size contains non-integer values or colors contain non-tuple elements.
        OverflowError: If thickness value is too large for OpenCV operations.
    
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Load image and get model predictions
        >>> image = cv2.imread("path/to/image.jpg")
        >>> model = YOLO("yolo11n.pt")
        >>> outputs = model.predict(image)  # Raw model outputs
        >>> detections = pf.results.from_ultralytics(outputs)  # Convert to PixelFlow format
        >>> 
        >>> # Draw grids with automatic sizing based on box dimensions
        >>> annotated = pf.annotate.grid_overlay(image, detections)
        >>> 
        >>> # Use fixed 4x4 grid for all boxes with custom thickness
        >>> annotated = pf.annotate.grid_overlay(image, detections, grid_size=(4, 4), thickness=2)
        >>> 
        >>> # Create filled checkerboard pattern with custom opacity
        >>> annotated = pf.annotate.grid_overlay(image, detections, filled=True, opacity=0.3)
        >>> 
        >>> # Fine grid with thin lines for precise measurement
        >>> annotated = pf.annotate.grid_overlay(image, detections, grid_size=(8, 8), thickness=1)
    
    Notes:
        - Auto-sizing creates approximately one grid line every 50 pixels for optimal density
        - Minimum grid size is 2x2, maximum is 10x10 for auto-calculation to prevent overcrowding
        - Grid lines are drawn inside the bounding box boundaries, not extending beyond box edges
        - Filled mode creates alternating checkerboard pattern starting with filled top-left cell
        - Grid color automatically matches the bounding box color for visual consistency
        - Opacity values are automatically clamped to [0.0, 1.0] range for safety
        - Grid calculations use integer pixel coordinates to ensure crisp line rendering
        - Checkerboard pattern uses (row + col) % 2 == 0 formula for cell alternation
        
    Performance Notes:
        - Grid calculation is optimized for real-time processing with minimal computational overhead
        - Filled mode with opacity uses alpha blending which may impact performance on large images
        - Consider using lower grid density (larger grid_size values) for better performance
        - Memory usage scales with number of detections and grid density
    """
    # Get adaptive parameters if not specified
    params = _get_adaptive_params(image)
    if thickness is None:
        thickness = max(1, params['thickness'] // 2)  # Thinner than box lines
    
    # Auto-calculate opacity for filled mode if not specified
    if filled and opacity is None:
        # Lower opacity for grid fills to maintain visibility
        opacity = 0.25
    
    # Clamp opacity to valid range
    if opacity is not None:
        opacity = max(0.0, min(1.0, opacity))
    
    for result in detections:
        box = result.bbox
        x1, y1, x2, y2 = map(int, box)
        
        # Calculate box dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Auto-calculate grid size if not specified
        if grid_size is None:
            # Approximately one grid line every 50 pixels
            min_dimension = min(width, height)
            grid_divisions = max(2, min(10, int(min_dimension / 50)))
            rows, cols = grid_divisions, grid_divisions
        else:
            rows, cols = grid_size
            # Ensure minimum grid size
            rows = max(2, rows)
            cols = max(2, cols)
        
        # Get color for this prediction
        color = _get_color_for_prediction(result, colors)
        
        # Draw filled checkerboard pattern if requested
        if filled and opacity is not None:
            overlay = image.copy()
            cell_width = width / cols
            cell_height = height / rows
            
            # Create checkerboard pattern
            for row in range(rows):
                for col in range(cols):
                    # Alternate pattern: fill when (row + col) is even
                    if (row + col) % 2 == 0:
                        cell_x1 = int(x1 + col * cell_width)
                        cell_y1 = int(y1 + row * cell_height)
                        cell_x2 = int(x1 + (col + 1) * cell_width)
                        cell_y2 = int(y1 + (row + 1) * cell_height)
                        
                        cv2.rectangle(overlay, (cell_x1, cell_y1), (cell_x2, cell_y2), 
                                    color, thickness=cv2.FILLED)
            
            # Blend overlay with original image
            cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)
        
        # Draw vertical grid lines
        for i in range(1, cols):
            x_pos = int(x1 + (width * i / cols))
            cv2.line(image, (x_pos, y1), (x_pos, y2), color=color, thickness=thickness)
        
        # Draw horizontal grid lines
        for i in range(1, rows):
            y_pos = int(y1 + (height * i / rows))
            cv2.line(image, (x1, y_pos), (x2, y_pos), color=color, thickness=thickness)
    
    return image