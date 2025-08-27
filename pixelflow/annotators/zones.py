import cv2
import numpy as np
from .utils import _get_adaptive_params


def zones(
    image,
    zone_manager,
    opacity=0.3,
    border_thickness=None,
    show_counts=True,
    show_names=True,
    font_scale=None,
    font_thickness=None,
    text_color=None,
    text_bg_color=None,
    text_bg_opacity=0.7,
    count_position='center',
    draw_filled=True,
    draw_border=True
):
    """
    Annotate zones on an image with superior visualization capabilities.
    
    This function draws polygon zones with customizable appearance, automatic
    counting, and labeling.
    
    Args:
        image (np.ndarray): Input image to annotate
        zone_manager: ZoneManager instance containing zones to draw
        opacity (float): Fill opacity for zones (0.0-1.0). Default 0.3.
        border_thickness (int): Thickness of zone borders. Default 2.
        show_counts (bool): Display detection count in each zone. Default True.
        show_names (bool): Display zone names. Default True.
        font_scale (float): Scale of text labels. Default 0.7.
        font_thickness (int): Thickness of text. Default 2.
        text_color (tuple, optional): RGB color for text. If None, uses UI text color.
        text_bg_color (tuple, optional): RGB color for text background. If None, uses UI background.
        text_bg_opacity (float): Opacity of text background. Default 0.7.
        count_position (str): Position for count display ('center', 'top', 'bottom').
        draw_filled (bool): Whether to fill zones with color. Default True.
        draw_border (bool): Whether to draw zone borders. Default True.
    
    Returns:
        np.ndarray: Annotated image with zones visualized
    
    Examples:
        # Simple usage - draws all zones with default settings
        annotated = zones(image, zone_manager)
        
        # Transparent zones with counts
        annotated = zones(image, zone_manager, opacity=0.2, show_counts=True)
        
        # Border-only zones with names
        annotated = zones(image, zone_manager, draw_filled=False, show_names=True)
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."
    
    if zone_manager is None or not hasattr(zone_manager, 'zones'):
        return image
    
    # Import colors locally to avoid circular dependency
    from .. import colors as color_module
    colors = color_module.ColorManager()
    
    # Get adaptive parameters
    params = _get_adaptive_params(image)
    
    # Use adaptive values if not specified
    if border_thickness is None:
        border_thickness = params['thickness']
    if font_scale is None:
        font_scale = params['font_scale']
    if font_thickness is None:
        font_thickness = params['font_thickness']
    
    # Get default colors if not specified
    if text_color is None:
        text_color = colors.ui('text')
    if text_bg_color is None:
        text_bg_color = colors.ui('background')
    
    # Create overlay for transparency effects
    overlay = image.copy()
    
    for zone in zone_manager.zones:
        # Convert Shapely polygon to numpy array of points
        points = np.array(zone.polygon.exterior.coords[:-1], dtype=np.int32)
        
        # Draw filled zone if enabled
        if draw_filled and opacity > 0:
            # Create zone mask
            zone_overlay = image.copy()
            cv2.fillPoly(zone_overlay, [points], zone.color)
            # Blend with original
            cv2.addWeighted(zone_overlay, opacity, overlay, 1 - opacity, 0, overlay)
        
        # Draw zone border if enabled
        if draw_border:
            cv2.polylines(overlay, [points], True, zone.color, border_thickness, cv2.LINE_AA)
        
        # Prepare text labels
        labels = []
        if show_names:
            labels.append(zone.name)
        if show_counts:
            count_text = f"Count: {zone.current_count}"
            if hasattr(zone, 'total_entered') and zone.total_entered > 0:
                count_text += f" (Total: {zone.total_entered})"
            labels.append(count_text)
        
        # Draw text if we have labels
        if labels:
            # Calculate text position based on zone centroid
            centroid = zone.polygon.centroid
            text_x = int(centroid.x)
            text_y = int(centroid.y)
            
            # Adjust position based on count_position parameter
            if count_position == 'top':
                # Find topmost point of polygon
                min_y = np.min(points[:, 1])
                text_y = min_y + 30
            elif count_position == 'bottom':
                # Find bottommost point of polygon
                max_y = np.max(points[:, 1])
                text_y = max_y - 30
            
            # Draw each label line
            y_offset = 0
            for label in labels:
                # Calculate text size
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                # Calculate text position (centered)
                label_x = text_x - text_size[0] // 2
                label_y = text_y + y_offset
                
                # Draw text background if opacity > 0
                if text_bg_opacity > 0:
                    # Create background rectangle with padding
                    bg_padding = params['padding']
                    bg_x1 = label_x - bg_padding
                    bg_y1 = label_y - text_size[1] - bg_padding
                    bg_x2 = label_x + text_size[0] + bg_padding
                    bg_y2 = label_y + baseline + bg_padding
                    
                    # Draw semi-transparent background
                    text_overlay = overlay.copy()
                    cv2.rectangle(text_overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), text_bg_color, -1)
                    cv2.addWeighted(text_overlay, text_bg_opacity, overlay, 1 - text_bg_opacity, 0, overlay)
                
                # Draw text
                cv2.putText(
                    overlay,
                    label,
                    (label_x, label_y),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA
                )
                
                y_offset += text_size[1] + baseline + 10  # Space between lines
    
    # Copy overlay back to image
    image[:] = overlay
    return image