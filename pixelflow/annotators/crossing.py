import cv2
import numpy as np
from .utils import _get_adaptive_params


def crossing(
    image,
    crossing_line,
    thickness=None,
    color=None,
    text_thickness=None,
    text_color=None,
    text_scale=None,
    text_offset=None,
    text_padding=None,
    custom_in_text=None,
    custom_out_text=None,
    display_in_count=True,
    display_out_count=True,
    display_text_box=True,
    text_centered=True,
):
    """
    Annotate a crossing line on an image with crossing counts.
    
    This function draws a line and displays the in/out crossing counts
    with customizable styling.
    
    Args:
        image (np.ndarray): Input image to annotate
        crossing_line: Crossing object from pixelflow.crossings
        thickness (int): Line thickness. Default 2.
        color (tuple, optional): Line color RGB. If None, uses line's color.
        text_thickness (int): Text thickness. Default 2.
        text_color (tuple, optional): Text color RGB. If None, uses UI text color.
        text_scale (float): Text scale factor. Default 0.5.
        text_offset (int): Distance of text from line center. Default 20.
        text_padding (int): Padding around text. Default 10.
        custom_in_text (str, optional): Custom label for "in" count.
        custom_out_text (str, optional): Custom label for "out" count.
        display_in_count (bool): Whether to show in count. Default True.
        display_out_count (bool): Whether to show out count. Default True.
        display_text_box (bool): Whether to show text background. Default True.
        text_centered (bool): Whether to center text on line. Default True.
    
    Returns:
        np.ndarray: Annotated image
    
    Examples:
        # Simple usage
        annotated = crossing(image, crossing_line)
        
        # Custom styling
        annotated = crossing(image, crossing_line, thickness=3, text_color=(0, 255, 0))
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


def crossings(image, crossings_manager):
    """
    Annotate multiple crossing lines on an image.
    
    This is a convenience function that draws all crossings managed by a Crossings object.
    
    Args:
        image (np.ndarray): Input image to annotate
        crossings_manager: Crossings manager object containing multiple crossings
    
    Returns:
        np.ndarray: Annotated image with all lines drawn
    
    Example:
        crossings_obj = Crossings()
        crossings_obj.add_crossing(start=(0, 500), end=(1920, 500))
        crossings_obj.add_crossing(start=(960, 0), end=(960, 1080))
        annotated = crossings(image, crossings_obj)
    """
    for crossing_line in crossings_manager.crossings:
        image = crossing(image, crossing_line)
    return image