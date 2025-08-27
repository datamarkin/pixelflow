import cv2
import numpy as np
from .utils import _get_adaptive_params


def line_zone(
    image,
    line,
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
    Annotate a line zone on an image with crossing counts.
    
    This function draws a line and displays the in/out crossing counts
    with customizable styling.
    
    Args:
        image (np.ndarray): Input image to annotate
        line: Line object from pixelflow.lines
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
        annotated = line_zone(image, line)
        
        # Custom styling
        annotated = line_zone(image, line, thickness=3, text_color=(0, 255, 0))
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."
    
    # Import colors locally to avoid circular dependency
    from .. import colors as color_module
    colors = color_module.ColorManager()
    
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
    
    # Get line color
    line_color = color if color else line.color
    
    # Get text color
    if text_color is None:
        text_color = colors.ui('text')
    
    # Draw the line
    start_point = tuple(map(int, line.start))
    end_point = tuple(map(int, line.end))
    cv2.line(image, start_point, end_point, line_color, thickness, cv2.LINE_AA)
    
    # Draw end point markers (scale with image size)
    marker_size = max(3, int(params['thickness'] * 2.5))
    cv2.circle(image, start_point, marker_size, text_color, -1, cv2.LINE_AA)
    cv2.circle(image, end_point, marker_size, text_color, -1, cv2.LINE_AA)
    
    # Calculate line center for text placement
    center_x = (line.start[0] + line.end[0]) / 2
    center_y = (line.start[1] + line.end[1]) / 2
    
    # Prepare count texts
    in_text = custom_in_text if custom_in_text else "in"
    out_text = custom_out_text if custom_out_text else "out"
    
    texts = []
    if display_in_count:
        texts.append(f"{in_text}: {line.in_count}")
    if display_out_count:
        texts.append(f"{out_text}: {line.out_count}")
    
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
            cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), line_color, -1)
        
        # Draw text
        cv2.putText(
            image, text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, text_scale,
            text_color, text_thickness, cv2.LINE_AA
        )
    
    return image


def line_zones(image, lines_manager):
    """
    Annotate multiple line zones on an image.
    
    This is a convenience function that draws all lines managed by a Lines object.
    
    Args:
        image (np.ndarray): Input image to annotate
        lines_manager: Lines manager object containing multiple lines
    
    Returns:
        np.ndarray: Annotated image with all lines drawn
    
    Example:
        lines = Lines()
        lines.add_line(start=(0, 500), end=(1920, 500))
        lines.add_line(start=(960, 0), end=(960, 1080))
        annotated = line_zones(image, lines)
    """
    for line in lines_manager.lines:
        image = line_zone(image, line)
    return image