import cv2
from . import colors
from . import draw
import numpy as np

colors = colors.ColorManager()

def box_fill(image, results, thickness: int = 2, ):
    # TODO: Implement filled bounding boxes without borders
    return image


def blur(image, results, kernel_size: int = 15, padding_percent: float = 0.05):
    """
    Applies blur effect to detected regions in the image with padding.
    
    The blur effect is achieved using Gaussian blur to obscure details
    in detected regions while maintaining a natural appearance.
    
    Args:
        image (np.ndarray): Input image to apply blur on
        results: List of detection results containing bounding boxes
        kernel_size (int): Size of the blur kernel. Larger values create
                          stronger blur effect. Default is 15. Must be odd and > 0.
        padding_percent (float): Padding to add around detection as percentage
                                of box size. Default is 0.05 (5% from each side).
        
    Returns:
        np.ndarray: Image with blurred regions where objects were detected
    
    Performance Notes:
        - Uses OpenCV's optimized Gaussian blur implementation
        - Efficient for real-time processing
        - Scales well with image size
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."
    
    # Validate and ensure kernel_size is odd and positive
    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Make it odd
    
    # Validate padding percent
    padding_percent = max(0, min(padding_percent, 0.5))  # Cap at 50% padding
    
    image_height, image_width = image.shape[:2]
    
    for result in results:
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


def pixelate(image, results, pixel_size: int = 10, padding_percent: float = 0.05):
    """
    Applies pixelation effect to detected regions in the image with padding.
    
    The pixelation effect is achieved by downscaling and then upscaling
    the region of interest, which reduces detail and creates a blocky appearance.
    
    This implementation uses OpenCV's optimized resize functions which leverage
    SIMD instructions for best performance.
    
    Args:
        image (np.ndarray): Input image to apply pixelation on
        results: List of detection results containing bounding boxes
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
    
    # Validate and clamp pixel_size early (reduced default from 20 to 10)
    pixel_size = max(1, pixel_size)
    
    # Validate padding percent
    padding_percent = max(0, min(padding_percent, 0.5))  # Cap at 50% padding
    
    image_height, image_width = image.shape[:2]
    
    for result in results:
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


def footprint(image, results, thickness: int = 2, start_angle: int = -45, end_angle: int = 235):
    """
    Draws elliptical footprints at the bottom of detected objects.
    
    Creates ground-plane footprint visualization by drawing partial ellipses
    at the bottom center of each bounding box. Useful for showing object
    presence on the ground plane or creating shadow-like effects.
    
    Args:
        image (np.ndarray): Input image to draw footprints on
        results: List of detection results containing bounding boxes
        thickness (int): Thickness of the ellipse lines. Default is 2.
        start_angle (int): Starting angle of the ellipse in degrees. 
                          Default is -45 (bottom-left).
        end_angle (int): Ending angle of the ellipse in degrees.
                        Default is 235 (bottom-right, creating bottom arc).
        
    Returns:
        np.ndarray: Image with elliptical footprints drawn at object bases
    
    Notes:
        - Ellipse width matches the bounding box width
        - Ellipse height is 25% of the width for natural proportions
        - Center point is at bottom-center of bounding box
        - Useful for ground plane visualization and spatial awareness
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."
    
    for result in results:
        box = result.bbox
        x1, y1, x2, y2 = map(int, box)
        
        # Get color for this detection
        color = colors.get_color(result.class_id)
        
        # Calculate ellipse parameters
        center = (int((x1 + x2) / 2), y2)  # Bottom center of bbox
        width = x2 - x1
        height = int(0.25 * width)  # Height is 25% of width for natural look
        
        # Draw the ellipse (partial arc from start_angle to end_angle)
        cv2.ellipse(
            image,
            center=center,
            axes=(int(width / 2), height),  # Semi-major and semi-minor axes
            angle=0.0,
            startAngle=start_angle,
            endAngle=end_angle,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA  # Anti-aliased for smooth curves
        )
    
    return image


def motion_trails(image, results, thickness: int = 2, ):
    # TODO: Implement motion trail visualization for tracked objects
    return image


def motion_dots(image, results, thickness: int = 2, ):
    # TODO: Implement motion dots/breadcrumbs for object paths
    return image


def heatmap(image, results, thickness: int = 2, ):
    # TODO: Implement heatmap visualization for detection density
    return image


def dot(image, results, thickness: int = 2, ):
    # TODO: Implement center dot annotation for detected objects
    return image


def keypoint(image, results, thickness: int = 2, ):
    # TODO: Implement keypoint visualization (e.g., pose estimation)
    return image


def keypoint_skeleton(image, results, thickness: int = 2, ):
    # TODO: Implement skeleton connections between keypoints
    return image


def grid_overlay(image, results, thickness: int = 2, ):
    # TODO: Implement grid overlay for spatial reference
    return image


def scale_bar(image, results, thickness: int = 2, ):
    # TODO: Implement scale bar for size reference
    return image


def fps_counter(image, results, thickness: int = 2, ):
    import time

    # Get current time
    current_time = time.time()

    # Calculate FPS
    if not hasattr(fps_counter, 'prev_time'):
        fps_counter.prev_time = current_time
        fps_counter.fps = 0
    else:
        # Calculate time difference
        time_diff = current_time - fps_counter.prev_time
        if time_diff > 0:
            fps_counter.fps = 1 / time_diff
        fps_counter.prev_time = current_time

    # Draw FPS text
    fps_text = f"FPS: {int(fps_counter.fps)}"
    cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, colors.ui('fps'), 2)

    return image


def box(image, results, thickness: int = 2, ):
    for result in results:
        box = result.bbox
        x1, y1, x2, y2 = map(int, box)

        color = colors.get_color(result.class_id)

        draw.rectangle(image, (x1, y1), (x2, y2), line_color=color, thickness=thickness)

    return image


def polygon(image: np.ndarray, results, color: tuple = None, thickness: int = 2) -> np.ndarray:
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."

    for result in results:
        # Iterate over the segments in the result
        # Convert the points to a NumPy array and reshape for OpenCV
        polygon = np.array(result.segments, dtype=np.int32).reshape((-1, 1, 2))
        # Draw the polygon on the canvas
        color = colors.get_color(result.class_id)
        cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=2)

    return image


def mask(frame: np.ndarray,
         results,
         color_order='class_id',
         opacity: float = 0.5) -> np.ndarray:
    """
    Overlays binary masks on a video frame with improved performance.

    Args:
        frame (np.ndarray): The video frame (BGR format).
        results: List of results containing masks (e.g., from a model's output).
        color (tuple): The color for the mask overlay (BGR format). Default is green (0, 255, 0).
        opacity (float): Opacity level for blending masks with the frame (0.0 to 1.0).

    Returns:
        np.ndarray: The frame with masks overlaid.
    """
    # Create a shared overlay array (same as frame) for all masks
    overlay = np.zeros_like(frame, dtype=np.uint8)

    # Process all masks at once
    for result in results:
        for mask in result.masks:
            # Ensure mask dimensions match the frame
            if mask.shape[:2] != frame.shape[:2]:
                raise ValueError("Mask dimensions do not match frame dimensions.")

            # Apply the color only to the masked regions
            color = colors.get_color(result.class_id)
            overlay[mask] = color

    # Blend the overlay with the original frame in a single operation
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, dst=frame)

    return frame


def label(
    image,
    results,
    labels=None,
    position='top_left',
    template=None,
    style=None,
    auto_arrange=True,
    padding=5,
    margin=2,
    max_width=None,
    opacity=0.7,
    rounded=0,
    shadow=False,
    gradient=False
):
    """
    Enhanced label annotator with advanced features and simpler API.
    
    This function provides a superior labeling system with automatic overlap prevention,
    template-based formatting, and flexible positioning - all while being simpler to use
    than traditional annotators.
    
    Args:
        image (np.ndarray): Input image to annotate
        results: List of detection results containing bounding boxes
        labels (list, optional): Custom labels for each detection. If None, auto-generates
                                from class_name and confidence
        position (str): Label position relative to bbox. Options:
                       'top_left', 'top_center', 'top_right',
                       'center_left', 'center', 'center_right',
                       'bottom_left', 'bottom_center', 'bottom_right'
        template (str, optional): Format template for auto-generating labels.
                                 E.g., "{class_name}: {confidence:.1%}"
                                 Available fields: class_name, confidence, class_id, tracker_id
        style (dict, optional): Advanced styling options:
                               {'font_scale': 0.5, 'font_thickness': 1,
                                'font_color': (255,255,255), 'bg_color': 'auto',
                                'border_color': None, 'border_width': 0}
        auto_arrange (bool): Automatically prevent label overlaps. Default True.
        padding (int): Padding inside label background. Default 5.
        margin (int): Margin between label and bbox edge. Default 2.
        max_width (int, optional): Maximum label width in pixels before text wrapping.
        opacity (float): Background opacity (0.0-1.0). Default 0.7.
        rounded (int): Corner radius for rounded rectangles. 0 = sharp corners.
        shadow (bool): Add drop shadow effect. Default False.
        gradient (bool): Use gradient background. Default False.
    
    Returns:
        np.ndarray: Annotated image with enhanced labels
    
    Examples:
        # Simple usage with auto-generated labels
        enhanced_label(image, results)
        
        # Custom template with tracker ID
        enhanced_label(image, results, template="{class_name} #{tracker_id} ({confidence:.0%})")
        
        # Advanced styling
        enhanced_label(image, results, position='bottom_center',
                      style={'font_scale': 0.7, 'bg_color': (0, 100, 200)},
                      rounded=5, shadow=True)
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."
    
    # Default style
    default_style = {
        'font_scale': 0.5,
        'font_thickness': 1,
        'font_color': colors.ui('text'),
        'bg_color': 'auto',
        'border_color': None,
        'border_width': 0
    }
    
    # Merge user style with defaults
    if style:
        default_style.update(style)
    style = default_style
    
    # Position mapping
    position_map = {
        'top_left': (0, 0),
        'top_center': (0.5, 0),
        'top_right': (1, 0),
        'center_left': (0, 0.5),
        'center': (0.5, 0.5),
        'center_right': (1, 0.5),
        'bottom_left': (0, 1),
        'bottom_center': (0.5, 1),
        'bottom_right': (1, 1)
    }
    
    pos_ratio = position_map.get(position.lower(), (0, 0))
    
    # Generate labels if not provided
    if labels is None:
        labels = []
        for result in results:
            if template:
                # Use template formatting with safe defaults
                try:
                    label_text = template.format(
                        class_name=getattr(result, 'class_name', 'Object'),
                        confidence=getattr(result, 'confidence', 0.0),
                        class_id=getattr(result, 'class_id', 0),
                        tracker_id=getattr(result, 'tracker_id', '')
                    )
                except (KeyError, ValueError):
                    # Fallback if template formatting fails
                    label_text = getattr(result, 'class_name', 'Object')
            else:
                # Default format
                class_name = getattr(result, 'class_name', 'Object')
                confidence = getattr(result, 'confidence', None)
                if confidence is not None:
                    label_text = f"{class_name}: {confidence:.2f}"
                else:
                    label_text = class_name
            labels.append(label_text)
    
    # Collect label positions and sizes for overlap prevention
    label_boxes = []
    
    for idx, (result, label_text) in enumerate(zip(results, labels)):
        if not label_text:
            continue
            
        bbox = result.bbox
        if bbox is None:
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Handle multi-line text if max_width is specified
        lines = []
        if max_width:
            words = label_text.split()
            current_line = []
            for word in words:
                test_line = ' '.join(current_line + [word])
                text_size = cv2.getTextSize(test_line, font, style['font_scale'], style['font_thickness'])[0]
                if text_size[0] <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
            if current_line:
                lines.append(' '.join(current_line))
        else:
            lines = [label_text]
        
        # Calculate total text dimensions
        max_text_width = 0
        total_text_height = 0
        line_heights = []
        
        for line in lines:
            text_size, baseline = cv2.getTextSize(line, font, style['font_scale'], style['font_thickness'])
            max_text_width = max(max_text_width, text_size[0])
            line_height = text_size[1] + baseline
            line_heights.append(line_height)
            total_text_height += line_height
        
        # Add line spacing for multi-line text
        if len(lines) > 1:
            total_text_height += (len(lines) - 1) * 2
        
        # Calculate label background box
        label_width = max_text_width + 2 * padding
        label_height = total_text_height + 2 * padding
        
        # Calculate label position based on anchor
        anchor_x = x1 + (x2 - x1) * pos_ratio[0]
        anchor_y = y1 + (y2 - y1) * pos_ratio[1]
        
        # Adjust for label dimensions
        label_x1 = int(anchor_x - label_width * pos_ratio[0])
        label_y1 = int(anchor_y - label_height * pos_ratio[1])
        
        # Apply margin
        if position.startswith('top'):
            label_y1 -= (label_height + margin)
        elif position.startswith('bottom'):
            label_y1 += margin
        
        if 'left' in position:
            label_x1 += margin
        elif 'right' in position:
            label_x1 -= margin
        
        label_x2 = label_x1 + label_width
        label_y2 = label_y1 + label_height
        
        # Store for overlap prevention
        label_boxes.append({
            'idx': idx,
            'x1': label_x1,
            'y1': label_y1,
            'x2': label_x2,
            'y2': label_y2,
            'lines': lines,
            'line_heights': line_heights
        })
    
    # Prevent overlaps if auto_arrange is enabled
    if auto_arrange and len(label_boxes) > 1:
        # Simple overlap prevention: adjust overlapping labels vertically
        for i in range(len(label_boxes)):
            for j in range(i + 1, len(label_boxes)):
                box_i = label_boxes[i]
                box_j = label_boxes[j]
                
                # Check for overlap
                if (box_i['x1'] < box_j['x2'] and box_i['x2'] > box_j['x1'] and
                    box_i['y1'] < box_j['y2'] and box_i['y2'] > box_j['y1']):
                    # Move the second box down
                    overlap_height = box_i['y2'] - box_j['y1']
                    box_j['y1'] += overlap_height + 2
                    box_j['y2'] += overlap_height + 2
    
    # Draw labels
    for label_info in label_boxes:
        idx = label_info['idx']
        result = results[idx]
        
        # Get background color
        bg_color = style['bg_color']
        if bg_color == 'auto':
            bg_color = colors.get_color(result.class_id)
        
        # Create overlay for transparency
        overlay = image.copy()
        
        # Draw shadow if enabled
        if shadow:
            shadow_offset = 2
            shadow_color = colors.ui('shadow')
            if rounded > 0:
                _draw_rounded_rectangle(
                    overlay,
                    label_info['x1'] + shadow_offset,
                    label_info['y1'] + shadow_offset,
                    label_info['x2'] + shadow_offset,
                    label_info['y2'] + shadow_offset,
                    shadow_color,
                    rounded
                )
            else:
                cv2.rectangle(
                    overlay,
                    (label_info['x1'] + shadow_offset, label_info['y1'] + shadow_offset),
                    (label_info['x2'] + shadow_offset, label_info['y2'] + shadow_offset),
                    shadow_color,
                    -1
                )
        
        # Draw background
        if gradient:
            # Create gradient effect
            _draw_gradient_rectangle(
                overlay,
                label_info['x1'],
                label_info['y1'],
                label_info['x2'],
                label_info['y2'],
                bg_color,
                rounded
            )
        elif rounded > 0:
            _draw_rounded_rectangle(
                overlay,
                label_info['x1'],
                label_info['y1'],
                label_info['x2'],
                label_info['y2'],
                bg_color,
                rounded
            )
        else:
            cv2.rectangle(
                overlay,
                (label_info['x1'], label_info['y1']),
                (label_info['x2'], label_info['y2']),
                bg_color,
                -1
            )
        
        # Draw border if specified
        if style['border_color'] and style['border_width'] > 0:
            if rounded > 0:
                _draw_rounded_rectangle(
                    overlay,
                    label_info['x1'],
                    label_info['y1'],
                    label_info['x2'],
                    label_info['y2'],
                    style['border_color'],
                    rounded,
                    style['border_width']
                )
            else:
                cv2.rectangle(
                    overlay,
                    (label_info['x1'], label_info['y1']),
                    (label_info['x2'], label_info['y2']),
                    style['border_color'],
                    style['border_width']
                )
        
        # Apply transparency
        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)
        
        # Draw text (always opaque)
        y_offset = label_info['y1'] + padding
        for line, line_height in zip(label_info['lines'], label_info['line_heights']):
            text_x = label_info['x1'] + padding
            text_y = y_offset + line_height - 3  # Adjust for baseline
            
            cv2.putText(
                image,
                line,
                (text_x, text_y),
                font,
                style['font_scale'],
                style['font_color'],
                style['font_thickness'],
                cv2.LINE_AA
            )
            y_offset += line_height + 2  # Add line spacing
    
    return image


def _draw_rounded_rectangle(image, x1, y1, x2, y2, color, radius, thickness=-1):
    """Helper function to draw rounded rectangle."""
    # Clip radius to valid range
    width = x2 - x1
    height = y2 - y1
    radius = min(radius, min(width, height) // 2)
    
    if radius <= 0:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        return
    
    # Draw rounded rectangle using circles and rectangles
    if thickness == -1:
        # Filled rounded rectangle
        # Draw rectangles
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        
        # Draw circles at corners
        cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        # Outlined rounded rectangle
        # Draw lines
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw arcs at corners
        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def _draw_gradient_rectangle(image, x1, y1, x2, y2, base_color, radius=0):
    """Helper function to draw gradient-filled rectangle."""
    height = y2 - y1
    
    # Create gradient
    for y in range(y1, y2):
        # Calculate gradient factor (darker at top, lighter at bottom)
        factor = (y - y1) / height if height > 0 else 0
        gradient_color = tuple(
            int(c * (0.7 + 0.3 * factor)) for c in base_color
        )
        
        if radius > 0 and (y < y1 + radius or y > y2 - radius):
            # Handle rounded corners by using the rounded rectangle function
            continue
        
        cv2.line(image, (x1, y), (x2, y), gradient_color, 1)
    
    # If rounded, draw the corners with base color
    if radius > 0:
        _draw_rounded_rectangle(image, x1, y1, x2, y2, base_color, radius)


def line_zone(
    image,
    line,
    thickness=2,
    color=None,
    text_thickness=2,
    text_color=None,
    text_scale=0.5,
    text_offset=20,
    text_padding=10,
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
    
    # Get line color
    line_color = color if color else line.color
    
    # Get text color
    if text_color is None:
        text_color = colors.ui('text')
    
    # Draw the line
    start_point = tuple(map(int, line.start))
    end_point = tuple(map(int, line.end))
    cv2.line(image, start_point, end_point, line_color, thickness, cv2.LINE_AA)
    
    # Draw end point markers
    cv2.circle(image, start_point, 5, text_color, -1, cv2.LINE_AA)
    cv2.circle(image, end_point, 5, text_color, -1, cv2.LINE_AA)
    
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


def zones(
    image,
    zone_manager,
    opacity=0.3,
    border_thickness=2,
    show_counts=True,
    show_names=True,
    font_scale=0.7,
    font_thickness=2,
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
                    padding = 5
                    bg_x1 = label_x - padding
                    bg_y1 = label_y - text_size[1] - padding
                    bg_x2 = label_x + text_size[0] + padding
                    bg_y2 = label_y + baseline + padding
                    
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

