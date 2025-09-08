from typing import Optional, Tuple, Union
import cv2
import numpy as np
from .utils import _get_adaptive_params


def zones(
    image: np.ndarray,
    zone_manager,
    opacity: float = 0.3,
    border_thickness: Optional[int] = None,
    show_counts: bool = True,
    show_names: bool = True,
    font_scale: Optional[float] = None,
    font_thickness: Optional[int] = None,
    text_color: Optional[Tuple[int, int, int]] = None,
    text_bg_color: Optional[Tuple[int, int, int]] = None,
    text_bg_opacity: float = 0.7,
    count_position: str = 'center',
    draw_filled: bool = True,
    draw_border: bool = True
) -> np.ndarray:
    """
    Draw polygon zones on images with customizable styling and intelligent labeling.
    
    Renders defined zones as filled polygons with optional borders, displaying zone names, 
    detection counts, and total entry statistics. Automatically adapts text size and 
    thickness based on image resolution for optimal visibility across different image sizes.
    
    Args:
        image (np.ndarray): Input image to annotate. Must be 3-channel BGR format.
        zone_manager: ZoneManager instance containing polygon zones with associated metadata.
                     Must have 'zones' attribute containing Zone objects with polygon, color, 
                     name, and count attributes.
        opacity (float): Fill opacity for zone polygons. Range: [0.0, 1.0]. 
                        Default is 0.3 (30% transparency).
        border_thickness (Optional[int]): Pixel thickness of zone borders. If None,
                                        automatically calculated based on image size.
                                        Range: [1, 20]. Default adapts to image resolution.
        show_counts (bool): Whether to display current detection count in each zone.
                          Shows both current count and total entries if available.
                          Default is True.
        show_names (bool): Whether to display zone names as text labels.
                          Default is True.
        font_scale (Optional[float]): Scale factor for text size. If None, automatically
                                    calculated based on image resolution. Range: [0.3, 2.0].
                                    Default adapts to image size.
        font_thickness (Optional[int]): Thickness of text stroke in pixels. If None,
                                      automatically calculated based on image size.
                                      Range: [1, 5]. Default adapts to font scale.
        text_color (Optional[Tuple[int, int, int]]): RGB color tuple for text labels.
                                                   If None, uses white (255, 255, 255).
                                                   Range: [0, 255] per channel.
        text_bg_color (Optional[Tuple[int, int, int]]): RGB color tuple for text background.
                                                      If None, uses black (0, 0, 0).
                                                      Range: [0, 255] per channel.
        text_bg_opacity (float): Opacity of text background rectangles. Range: [0.0, 1.0].
                               Default is 0.7 (70% opacity).
        count_position (str): Vertical position for count display relative to zone.
                            Options: 'center' (zone centroid), 'top' (above zone), 
                            'bottom' (below zone). Default is 'center'.
        draw_filled (bool): Whether to fill zone polygons with semi-transparent color.
                          Default is True.
        draw_border (bool): Whether to draw zone polygon borders. Default is True.
        
    Returns:
        np.ndarray: Annotated image with zones visualized. Modifies input image in-place
                   and returns the modified array.
    
    Raises:
        AssertionError: If image is not a numpy ndarray.
        AttributeError: If zone_manager lacks 'zones' attribute or zones lack required 
                       properties (polygon, color, name, current_count).
        IndexError: If zone polygon coordinates are outside image boundaries during
                   text positioning calculations.
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from shapely.geometry import Polygon
        >>> 
        >>> # Load image and create zone manager with defined areas
        >>> image = cv2.imread("surveillance_feed.jpg")
        >>> zone_manager = pf.ZoneManager()
        >>> zone_manager.add_zone("entrance", Polygon([(100, 100), (300, 100), (300, 200), (100, 200)]))
        >>> zone_manager.add_zone("restricted", Polygon([(400, 150), (600, 150), (600, 300), (400, 300)]))
        >>> 
        >>> # Basic usage - draw all zones with default semi-transparent fill
        >>> annotated = pf.annotators.zones(image, zone_manager)
        >>> 
        >>> # High-visibility zones with custom opacity and thick borders
        >>> annotated = pf.annotators.zones(image, zone_manager, opacity=0.5, 
        ...                                border_thickness=4, show_counts=True)
        >>> 
        >>> # Border-only zones with names positioned at bottom
        >>> annotated = pf.annotators.zones(image, zone_manager, draw_filled=False, 
        ...                                show_names=True, count_position='bottom')
        >>> 
        >>> # Minimal zones with custom colors and no background
        >>> annotated = pf.annotators.zones(image, zone_manager, text_color=(255, 255, 0),
        ...                                text_bg_opacity=0.0, show_counts=False)
    
    Notes:
        - Modifies the input image in-place for memory efficiency
        - Text size and thickness automatically adapt to image resolution when not specified
        - Zone colors are taken from individual Zone objects in the zone_manager
        - Count display shows format "Count: X" or "Count: X (Total: Y)" when total_entered available
        - Text positioning uses zone polygon centroid with automatic adjustment for top/bottom modes
        - Semi-transparent overlays are created using OpenCV's addWeighted for smooth blending
        - Polygon rendering uses anti-aliased lines for smooth appearance
        - Text background rectangles include padding calculated from adaptive parameters
        
    Performance Notes:
        - Processing time scales with number of zones and image resolution
        - Memory usage is optimized through in-place image modification
        - Overlay operations create temporary image copies for transparency blending
        - Text rendering involves multiple OpenCV operations per label
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."
    
    if zone_manager is None or not hasattr(zone_manager, 'zones'):
        return image
    
    # Import colors locally to avoid circular dependency
    from ..colors import PASTEL_PALETTE
    
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
        text_color = (255, 255, 255)  # White text
    if text_bg_color is None:
        text_bg_color = (0, 0, 0)  # Black background
    
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