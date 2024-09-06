# annotate/__init__.py

import os
import cv2
import ast
import numpy as np
import pandas as pd
import pixelflow.draw
import pixelflow.predictions


def frame(image, predictions: pixelflow.predictions.Predictions):
    """
    Draw polygons on the image using the masks from a Predictions object.

    Parameters:
    - image: The original image (NumPy array).
    - predictions: A Predictions object containing multiple Prediction objects with polygon masks.

    This function will draw the polygons defined by each prediction's mask and bounding box onto the image.
    """
    # Loop over all predictions
    for prediction in predictions:
        # Draw the bounding box using the custom rectangle function
        if prediction.bbox:
            # Unpack the bounding box coordinates
            xmin, ymin, xmax, ymax = prediction.bbox
            # Call the custom rectangle function to draw the bounding box with opacity
            image = pixelflow.draw.rectangle(image, top_left=(xmin, ymin), bottom_right=(xmax, ymax))

        # Get the mask (polygon points) from the prediction
        if prediction.masks:
            for mask in prediction.masks:
                # Draw the polygon on the image using the polygon points (mask)
                image = pixelflow.draw.polygon(image, mask)

    return image

