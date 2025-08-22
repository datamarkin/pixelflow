import cv2
from . import colors
from . import draw
import numpy as np

colors = colors.ColorManager()


def draw_box(image, results, thickness: int = 2):
    """
    Draws filled boxes with borders around detected objects.
    
    Args:
        image (np.ndarray): Input image to draw on
        results: List of detection results containing bounding boxes
        thickness (int): Border thickness of the boxes
        
    Returns:
        np.ndarray: Image with filled boxes drawn
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."
    
    for result in results:
        box = result.bbox
        x1, y1, x2, y2 = map(int, box)
        
        color = colors.get_color(result.class_id)
        
        # Create a semi-transparent fill
        alpha = 0.3
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # Filled rectangle
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Draw the border
        draw.rectangle(image, (x1, y1), (x2, y2), line_color=color, thickness=thickness)

    return image



def box_fill(image, results, thickness: int = 2, ):
    return image


def blur(image, results, thickness: int = 2, ):
    return image


def pixelate(image, results, thickness: int = 2, ):
    return image


def motion_trails(image, results, thickness: int = 2, ):
    return image


def motion_dots(image, results, thickness: int = 2, ):
    return image


def heatmap(image, results, thickness: int = 2, ):
    return image


def dot(image, results, thickness: int = 2, ):
    return image


def keypoint(image, results, thickness: int = 2, ):
    return image


def keypoint_skeleton(image, results, thickness: int = 2, ):
    return image


def grid_overlay(image, results, thickness: int = 2, ):
    return image


def scale_bar(image, results, thickness: int = 2, ):
    return image


def fps_counter(image, results, thickness: int = 2, ):
    import time
    from datetime import datetime

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
    cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image


def box(image, results, thickness: int = 2, ):
    for result in results:
        box = result.bbox
        x1, y1, x2, y2 = map(int, box)

        color = colors.get_color(result.class_id)

        draw.rectangle(image, (x1, y1), (x2, y2), line_color=color, thickness=thickness)

    return image


def polygon(image: np.ndarray, results, color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
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


def label(image, results, labels=None, font_scale=0.5, font_thickness=1, font_color=(255, 255, 255),
          box_color=(0, 0, 0)):
    """
    Draws labels on the image for detected objects.
    
    This function takes detection results and draws text labels above each detected object's 
    bounding box. Each label is drawn with a background box for better visibility against
    any image background.

    Args:
        image (np.ndarray): The input image in BGR format to draw labels on.
        results: List of detection results, where each result contains a bounding box and 
                optionally a label for the detected object.
        labels (list, optional): Override labels to use instead of the ones in results.
                                Must match length of results if provided. Defaults to None.
        font_scale (float): Controls text size. Larger values = bigger text. Defaults to 0.5.
        font_thickness (int): Controls text stroke width. Larger values = bolder text. 
                            Defaults to 1.
        font_color (tuple): BGR color tuple for the label text. Defaults to white (255,255,255).
        box_color (tuple): BGR color tuple for label background box. Defaults to black (0,0,0).

    Returns:
        np.ndarray: Copy of input image with labels drawn above detected objects.
    """
    assert isinstance(image, np.ndarray), "Input image must be a NumPy array."

    for idx, result in enumerate(results):
        # Get the bounding box and label
        result_dict = result.to_dict()
        bbox = result_dict.get("bbox", None)
        label = result_dict.get("label", None) if labels is None else labels[idx]

        if not bbox or not label:
            continue  # Skip if bbox or label is missing

        x1, y1, x2, y2 = map(int, bbox)

        # Define the text size and position
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x, text_y = x1, y1 - 10  # Position the text slightly above the bbox

        # Draw a filled rectangle as a background for the text
        cv2.rectangle(image, (text_x, text_y - text_size[1] - 4), (text_x + text_size[0] + 4, text_y), box_color, -1)

        # Draw the text on top of the rectangle
        cv2.putText(image, label, (text_x + 2, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color,
                    font_thickness)

    return image

## Quick Start
