from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from ..colors import get_color_for_prediction
from .utils import _get_adaptive_params


def label(
    image: np.ndarray,
    detections: 'Detections',
    texts: Union[str, List[str]] = None,
    position: str = 'top_left',
    font_scale: float = None,
    padding: int = 6,
    line_spacing: int = 2,
    bg_color: Union[tuple, str] = None,
    text_color: tuple = (255, 255, 255)
) -> np.ndarray:
    """
    Fast, simplified label annotator with template support and multi-line text.

    Args:
        image (np.ndarray): Input image to annotate
        detections (Detections): Detections object containing bounding boxes
        texts (Union[str, List[str]], optional): Text template or list of labels.
            - None: Auto-generates from class_name and confidence
            - str: Template with placeholders like "{class_name}: {confidence:.1%}"
            - List[str]: Custom labels for each detection
        position (str): Label position relative to bbox. Options:
                       'top_left', 'top_center', 'top_right',
                       'center_left', 'center', 'center_right',
                       'bottom_left', 'bottom_center', 'bottom_right'
        font_scale (float, optional): Font scale. If None, uses adaptive scaling
        padding (int): Padding around text inside background rectangle
        line_spacing (int): Space between lines for multi-line text
        bg_color (tuple or str, optional): Background color. If None, uses auto color from class_id
        text_color (tuple): Text color (BGR format)

    Returns:
        np.ndarray: Annotated image with labels

    Examples:
        # Auto-generated labels
        label(image, detections)

        # Template with placeholders
        label(image, detections, "{class_name}: {confidence:.1%}")
        
        # Multi-line template
        template = \"\"\"{class_name}
Confidence: {confidence:.2f}
Track: {tracker_id}\"\"\"
        label(image, detections, template)

        # Custom list of labels
        label(image, detections, ["Person", "Car"], position='top_center')
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."

    if len(detections) == 0:
        return image

    # Get adaptive font scale if not provided
    if font_scale is None:
        params = _get_adaptive_params(image)
        font_scale = params['font_scale']


    # Handle template strings, lists, or auto-generation
    if texts is None:
        # Auto-generate from detection properties
        texts = []
        for detection in detections:
            class_name = getattr(detection, 'class_name', 'Object')
            confidence = getattr(detection, 'confidence', None)
            if confidence is not None:
                text = f"{class_name}: {confidence:.2f}"
            else:
                text = class_name
            texts.append(text)
    elif isinstance(texts, str):
        # Template mode - format for each detection
        template = texts
        texts = []
        for detection in detections:
            # Build context dict from detection attributes
            context = {
                'class_name': getattr(detection, 'class_name', 'Unknown'),
                'confidence': getattr(detection, 'confidence', 0.0),
                'class_id': getattr(detection, 'class_id', -1),
                'tracker_id': getattr(detection, 'tracker_id', ''),
                'bbox': getattr(detection, 'bbox', None),
            }
            try:
                formatted_text = template.format(**context)
            except (KeyError, ValueError):
                # Fallback if template formatting fails
                formatted_text = template
            texts.append(formatted_text)

    # Process each detection
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1

    for idx, (detection, text) in enumerate(zip(detections, texts)):
        if not text:
            continue

        bbox = detection.bbox
        if bbox is None:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # Handle multi-line text by splitting on newlines
        lines = text.split('\n') if text else ['']
        
        # Calculate dimensions for all lines
        max_width = 0
        total_text_height = 0
        line_heights = []
        
        for line in lines:
            (line_width, line_height), line_baseline = cv2.getTextSize(
                line if line else ' ',  # Use space for empty lines
                font, font_scale, font_thickness
            )
            max_width = max(max_width, line_width)
            line_heights.append(line_height)
            total_text_height += line_height
        
        # Add spacing between lines (except for single line)
        if len(lines) > 1:
            total_text_height += line_spacing * (len(lines) - 1)
        
        # Calculate rectangle dimensions
        top_padding = padding + 2  # Extra padding on top
        rect_width = max_width + 2 * padding
        rect_height = total_text_height + padding + top_padding

        # Calculate label position based on bbox and position
        if position.startswith('top'):
            label_y = y1 - rect_height  # No gap - label bottom touches bbox top
        elif position.startswith('bottom'):
            label_y = y2  # No gap - label top touches bbox bottom
        else:  # center
            label_y = y1 + (y2 - y1 - rect_height) // 2

        if 'left' in position:
            label_x = x1  # Align with bbox left edge
        elif 'right' in position:
            label_x = x2 - rect_width  # Align with bbox right edge
        else:  # center
            label_x = x1 + (x2 - x1 - rect_width) // 2


        # Get background color
        if bg_color is None:
            bg_color_final = get_color_for_prediction(detection)
        else:
            bg_color_final = bg_color

        # Draw background rectangle directly
        cv2.rectangle(
            image,
            (label_x, label_y),
            (label_x + rect_width, label_y + rect_height),
            bg_color_final,
            -1
        )

        # Draw each line of text
        text_x = label_x + padding
        current_y = label_y + top_padding
        
        for i, line in enumerate(lines):
            if line:  # Only draw non-empty lines
                # Calculate baseline position for this line
                line_height = line_heights[i]
                text_y = current_y + line_height
                
                cv2.putText(
                    image,
                    line,
                    (text_x, text_y),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA
                )
            
            # Move to next line position
            current_y += line_heights[i] + line_spacing

    return image


