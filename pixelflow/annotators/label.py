import cv2
import numpy as np
from .. import colors
from .utils import _get_adaptive_params

color_manager = colors.ColorManager()


def label(
        image,
        results,
        labels=None,
        position='top_left',
        template=None,
        style=None,
        auto_arrange=True,
        padding=None,
        margin=None,
        max_width=None,
        opacity=0.7,
        rounded=None,
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

    # Get adaptive parameters
    params = _get_adaptive_params(image)

    # Use adaptive values if not specified
    if padding is None:
        padding = params['padding']
    if margin is None:
        margin = params['margin']
    if rounded is None:
        rounded = params['corner_radius']

    # Default style with adaptive font parameters
    default_style = {
        'font_scale': params['font_scale'],
        'font_thickness': params['font_thickness'],
        'font_color': color_manager.ui('text'),
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
            bg_color = color_manager.get_color(result.class_id)

        # Create overlay for transparency
        overlay = image.copy()

        # Draw shadow if enabled
        if shadow:
            shadow_offset = params['shadow_offset']
            shadow_color = color_manager.ui('shadow')
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


