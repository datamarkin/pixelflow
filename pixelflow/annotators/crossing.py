from typing import Optional, Tuple
import cv2
import numpy as np
from .utils import _get_adaptive_params


def crossing(
    image: np.ndarray,
    crossing_line,
    thickness: Optional[int] = None,
    color: Optional[Tuple[int, int, int]] = None,
    text_thickness: Optional[int] = None,
    text_color: Optional[Tuple[int, int, int]] = None,
    text_scale: Optional[float] = None,
    text_offset: Optional[int] = None,
    text_padding: Optional[int] = None,
    custom_in_text: Optional[str] = None,
    custom_out_text: Optional[str] = None,
    display_in_count: bool = True,
    display_out_count: bool = True,
    display_text_box: bool = True,
    text_centered: bool = True,
) -> np.ndarray:
    """
    Draw a crossing line with directional count annotations on an image.
    
    This function visualizes a crossing line by drawing the line itself, endpoint markers,
    and count text labels showing in/out crossing statistics with customizable styling.
    The annotation adapts automatically to image resolution for optimal visibility.
    
    Args:
        image (np.ndarray): Input BGR image to annotate. Must be a 3-channel NumPy array
                           with shape (height, width, 3).
        crossing_line: Crossing object from pixelflow.crossings containing line geometry,
                      counts, and color. Must have attributes: start, end, color,
                      in_count, out_count.
        thickness (Optional[int]): Line thickness in pixels. If None, uses adaptive
                                  sizing based on image resolution. Range: [1, ∞].
        color (Optional[Tuple[int, int, int]]): Line color as BGR tuple (0-255 each).
                                              If None, uses crossing_line.color.
        text_thickness (Optional[int]): Text stroke thickness in pixels. If None,
                                       uses adaptive sizing. Range: [1, ∞].
        text_color (Optional[Tuple[int, int, int]]): Text color as BGR tuple (0-255 each).
                                                   Default is white (255, 255, 255).
        text_scale (Optional[float]): Text scale factor. If None, uses adaptive sizing.
                                     Range: (0, ∞). Default ~0.5.
        text_offset (Optional[int]): Distance between text labels in pixels. If None,
                                    uses adaptive sizing. Range: [1, ∞]. Default ~20.
        text_padding (Optional[int]): Padding around text background in pixels.
                                     If None, uses adaptive sizing. Range: [0, ∞].
        custom_in_text (Optional[str]): Custom label for inward crossing count.
                                       Default is "in".
        custom_out_text (Optional[str]): Custom label for outward crossing count.
                                        Default is "out".
        display_in_count (bool): Whether to display the inward crossing count.
                                Default is True.
        display_out_count (bool): Whether to display the outward crossing count.
                                 Default is True.
        display_text_box (bool): Whether to draw colored background behind text.
                                Default is True.
        text_centered (bool): Whether to center text on the line. If False,
                             positions text near the end point. Default is True.
        
    Returns:
        np.ndarray: The input image with crossing line and count annotations drawn.
                   Same shape and dtype as input image. Image is modified in-place.
    
    Raises:
        AssertionError: If image is not a NumPy array.
        AttributeError: If crossing_line lacks required attributes (start, end,
                       color, in_count, out_count).
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> 
        >>> # Load image and set up tracking system
        >>> image = cv2.imread("traffic.jpg")
        >>> crossings_manager = pf.Crossings()
        >>> crossing_line = crossings_manager.add_crossing(
        ...     start=(100, 300), end=(500, 300), name="Street Line"
        ... )
        >>> 
        >>> # Basic crossing annotation
        >>> annotated = pf.annotate.crossing(image, crossing_line)
        >>> 
        >>> # Custom styling with thick line and green text
        >>> annotated = pf.annotate.crossing(
        ...     image, crossing_line,
        ...     thickness=5, text_color=(0, 255, 0)
        ... )
        >>> 
        >>> # Custom labels without text backgrounds
        >>> annotated = pf.annotate.crossing(
        ...     image, crossing_line,
        ...     custom_in_text="Entered", custom_out_text="Exited",
        ...     display_text_box=False
        ... )
        >>> 
        >>> # Show only outward count with right-aligned text
        >>> annotated = pf.annotate.crossing(
        ...     image, crossing_line,
        ...     display_in_count=False, text_centered=False
        ... )
    
    Notes:
        - All None parameters are automatically replaced with adaptive values based
          on image resolution using _get_adaptive_params()
        - The line is drawn with anti-aliasing (cv2.LINE_AA) for smooth appearance
        - Endpoint markers are drawn as filled circles to clearly indicate line direction
        - Text backgrounds use the same color as the crossing line for visual consistency
        - When both counts are displayed, they are vertically offset from the line center
        - The function modifies the input image in-place and returns it for convenience
        
    See Also:
        crossings : Convenience function for annotating multiple crossing lines
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."
    
    # Import colors locally to avoid circular dependency
    from ..colors import PASTEL_PALETTE
    
    # Get adaptive parameters
    params = _get_adaptive_params(image)
    
    # Use adaptive values if not specified
    if thickness is None:
        thickness = params['thickness']
    if text_thickness is None:
        text_thickness = params['font_thickness']
    if text_scale is None:
        text_scale = params['font_scale']
    if text_offset is None:
        text_offset = params['text_offset']
    if text_padding is None:
        text_padding = params['padding']
    
    # Get crossing color
    crossing_color = color if color else crossing_line.color
    
    # Get text color
    if text_color is None:
        text_color = (255, 255, 255)  # White text
    
    # Draw the crossing line
    start_point = tuple(map(int, crossing_line.start))
    end_point = tuple(map(int, crossing_line.end))
    cv2.line(image, start_point, end_point, crossing_color, thickness, cv2.LINE_AA)
    
    # Draw end point markers (scale with image size)
    marker_size = max(3, int(params['thickness'] * 2.5))
    cv2.circle(image, start_point, marker_size, text_color, -1, cv2.LINE_AA)
    cv2.circle(image, end_point, marker_size, text_color, -1, cv2.LINE_AA)
    
    # Calculate crossing line center for text placement
    center_x = (crossing_line.start[0] + crossing_line.end[0]) / 2
    center_y = (crossing_line.start[1] + crossing_line.end[1]) / 2
    
    # Prepare count texts
    in_text = custom_in_text if custom_in_text else "in"
    out_text = custom_out_text if custom_out_text else "out"
    
    texts = []
    if display_in_count:
        texts.append(f"{in_text}: {crossing_line.in_count}")
    if display_out_count:
        texts.append(f"{out_text}: {crossing_line.out_count}")
    
    # Draw text for each count
    for i, text in enumerate(texts):
        # Calculate text size
        text_size, _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
        )
        text_width, text_height = text_size
        
        # Calculate text position
        if text_centered:
            text_x = int(center_x - text_width / 2)
        else:
            text_x = int(end_point[0] - text_width - text_padding)
        
        # Offset for multiple texts
        offset_y = text_offset * (i - len(texts) / 2 + 0.5)
        text_y = int(center_y + text_height / 2 + offset_y)
        
        # Draw text background if enabled
        if display_text_box:
            bg_x1 = text_x - text_padding
            bg_y1 = text_y - text_height - text_padding
            bg_x2 = text_x + text_width + text_padding
            bg_y2 = text_y + text_padding
            cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), crossing_color, -1)
        
        # Draw text
        cv2.putText(
            image, text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, text_scale,
            text_color, text_thickness, cv2.LINE_AA
        )
    
    return image


def crossings(image: np.ndarray, crossings_manager) -> np.ndarray:
    """
    Draw multiple crossing lines with count annotations on an image.
    
    This convenience function iterates through all crossing lines managed by a
    Crossings object and draws each one with its individual styling and count
    information using the crossing() function.
    
    Args:
        image (np.ndarray): Input BGR image to annotate. Must be a 3-channel NumPy array
                           with shape (height, width, 3).
        crossings_manager: Crossings manager object containing multiple crossing lines.
                          Must have a 'crossings' attribute that is iterable,
                          containing Crossing objects.
        
    Returns:
        np.ndarray: The input image with all crossing lines and count annotations drawn.
                   Same shape and dtype as input image. Image is modified in-place.
    
    Raises:
        AttributeError: If crossings_manager lacks 'crossings' attribute or if
                       individual crossing objects lack required attributes.
        AssertionError: If image is not a NumPy array (raised by underlying
                       crossing() calls).
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> 
        >>> # Load image and create crossing manager
        >>> image = cv2.imread("intersection.jpg")
        >>> crossings_manager = pf.Crossings()
        >>> 
        >>> # Add multiple crossing lines
        >>> crossings_manager.add_crossing(
        ...     start=(0, 500), end=(1920, 500),
        ...     name="Horizontal Line", color=(255, 0, 0)
        ... )
        >>> crossings_manager.add_crossing(
        ...     start=(960, 0), end=(960, 1080),
        ...     name="Vertical Line", color=(0, 255, 0)
        ... )
        >>> 
        >>> # Draw all crossings at once
        >>> annotated = pf.annotate.crossings(image, crossings_manager)
        >>> 
        >>> # Alternative workflow with tracking integration
        >>> tracker = pf.tracker.ByteTracker()
        >>> for frame in frames:
        ...     outputs = yolo_model.predict(frame)
        ...     results = pf.detections.from_ultralytics(outputs)
        ...     results = tracker.update(results)
        ...     results = crossings_manager.update(results)  # Update counts
        ...     annotated = pf.annotate.crossings(frame, crossings_manager)
    
    Notes:
        - Each crossing line is drawn with its individual color, name, and styling
        - The function processes crossings in the order they appear in the manager
        - All crossing lines share the same adaptive sizing parameters based on image resolution
        - The input image is modified in-place for efficiency
        - This is equivalent to calling crossing() individually for each line
        
    See Also:
        crossing : Function for annotating a single crossing line
    """
    for crossing_line in crossings_manager.crossings:
        image = crossing(image, crossing_line)
    return image