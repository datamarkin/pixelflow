import cv2
import numpy as np


def polygon(image, points, line_color=(0, 255, 0), fill_color=(0, 255, 0),
            line_thickness=3, line_opacity=0.8, fill_opacity=0.4):
    """
    Draw a polygon with different opacities for the outline and the fill on an image.

    Parameters:
    - image: The original image (NumPy array).
    - points: A list of points defining the polygon's vertices.
    - line_color: The color of the polygon outline (BGR tuple).
    - fill_color: The color of the polygon fill (BGR tuple).
    - line_thickness: The thickness of the polygon outline.
    - line_opacity: The opacity of the polygon outline (0.0 to 1.0).
    - fill_opacity: The opacity of the polygon fill (0.0 to 1.0).

    Returns:
    - The image with the polygon drawn with the specified opacities.
    """
    # Create two overlay images
    overlay_fill = image.copy()
    overlay_line = image.copy()

    # Convert points to the correct shape
    points = np.array(points, np.int32).reshape((-1, 1, 2))

    # Draw the filled polygon on the overlay_fill
    cv2.fillPoly(overlay_fill, [points], color=fill_color)

    # Draw the polygon outline on the overlay_line
    cv2.polylines(overlay_line, [points], isClosed=True, color=line_color, thickness=line_thickness)

    # Blend the filled polygon with the original image
    cv2.addWeighted(overlay_fill, fill_opacity, image, 1 - fill_opacity, 0, image)

    # Blend the polygon outline with the original image
    cv2.addWeighted(overlay_line, line_opacity, image, 1 - line_opacity, 0, image)

    return image
