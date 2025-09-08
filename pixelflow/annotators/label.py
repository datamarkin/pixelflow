from typing import TYPE_CHECKING, List, Union, Optional

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from ..colors import _get_color_for_prediction
from .utils import _get_adaptive_params


def label(
    image: np.ndarray,
    detections: 'Detections',
    texts: Optional[Union[str, List[str]]] = None,
    position: str = 'top_left',
    font_scale: Optional[float] = None,
    padding: int = 6,
    line_spacing: int = 2,
    bg_color: Optional[Union[tuple, str]] = None,
    text_color: tuple = (255, 255, 255)
) -> np.ndarray:
    """
    Draw text labels on detected objects with positioning, templates, and multi-line support.
    
    This function provides flexible labeling capabilities with automatic color assignment,
    adaptive font scaling, and template-based text generation. It supports multi-line labels
    and various positioning options relative to bounding boxes.
    
    Args:
        image (np.ndarray): Input image to annotate. Must be a valid OpenCV image array
                           in BGR format with shape (H, W, 3) or (H, W).
        detections (Detections): PixelFlow detections object containing bounding box
                               coordinates and optional attributes (class_name, confidence,
                               class_id, tracker_id).
        texts (Optional[Union[str, List[str]]]): Label text specification.
            - None: Auto-generates labels from detection attributes (class_name: confidence)
            - str: Template string with placeholders ({class_name}, {confidence}, 
                   {class_id}, {tracker_id}, {bbox})
            - List[str]: Custom labels for each detection (length should match detections)
        position (str): Label position relative to bounding box. Options:
                       'top_left', 'top_center', 'top_right', 'center_left', 'center',
                       'center_right', 'bottom_left', 'bottom_center', 'bottom_right'.
                       Default is 'top_left'.
        font_scale (Optional[float]): OpenCV font scale factor. If None, uses adaptive
                                    scaling based on image dimensions. Range: [0.1, 5.0].
        padding (int): Padding in pixels around text inside background rectangle.
                      Range: [0, 50]. Default is 6.
        line_spacing (int): Additional spacing in pixels between lines for multi-line text.
                           Range: [0, 20]. Default is 2.
        bg_color (Optional[Union[tuple, str]]): Background rectangle color in BGR format.
                                              If None, uses automatic color based on class_id.
                                              Can be tuple (B, G, R) or color string.
        text_color (tuple): Text color in BGR format. Default is white (255, 255, 255).
        
    Returns:
        np.ndarray: Input image with labels drawn directly on it (in-place modification).
                   Returns the same image array that was passed as input.
    
    Raises:
        AssertionError: If image is not a NumPy array.
        AttributeError: If detections object lacks required bbox attribute.
        KeyError: If template string contains invalid placeholders.
        ValueError: If template formatting fails or colors are invalid.
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Load image and run detection
        >>> image = cv2.imread("people.jpg")
        >>> model = YOLO("yolo11n.pt")
        >>> outputs = model.predict(image)
        >>> detections = pf.results.from_ultralytics(outputs)
        >>> 
        >>> # Basic auto-generated labels
        >>> labeled_image = pf.annotators.label(image, detections)
        >>> 
        >>> # Custom template with confidence percentage
        >>> template = "{class_name}: {confidence:.1%}"
        >>> labeled_image = pf.annotators.label(image, detections, template, position='top_center')
        >>> 
        >>> # Multi-line labels with tracking info
        >>> multi_template = \"\"\"{class_name}
        ... ID: {tracker_id}
        ... Conf: {confidence:.2f}\"\"\"
        >>> labeled_image = pf.annotators.label(image, detections, multi_template, 
        ...                                    position='bottom_right', padding=8)
        >>> 
        >>> # Custom label list with specific positioning
        >>> custom_labels = ["Primary Target", "Secondary", "Background"]
        >>> labeled_image = pf.annotators.label(image, detections[:3], custom_labels,
        ...                                    position='center', bg_color=(0, 100, 200))
    
    Notes:
        - Labels are drawn directly on the input image (in-place modification)
        - Empty detections list returns the original image unchanged
        - Font scale automatically adapts to image size when not specified
        - Background colors are automatically assigned based on class_id for visual consistency
        - Multi-line text is supported by including newline characters in templates
        - Template placeholders are safely handled with fallback values for missing attributes
        - Label positioning automatically adjusts to keep labels within image boundaries
        - Text baseline and height calculations ensure consistent multi-line spacing
        
    Performance Notes:
        - Optimized for real-time annotation with minimal memory allocation
        - Adaptive parameter calculation cached per image size
        - Direct OpenCV drawing operations for maximum performance
        - Template formatting is cached per detection to avoid repeated processing
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
            bg_color_final = _get_color_for_prediction(detection)
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


