# visualise/__init__.py

import os
import cv2
import ast
import numpy as np
import pandas as pd
import pixelflow.draw
import pixelflow.predictions


def from_dtm_csv(csv_path, image_dir):
    # Load data from CSV
    data = pd.read_csv(csv_path)

    # Group data by 'id' column which represents the image file
    grouped_data = data.groupby('id')

    # Get the list of group keys (file names)
    group_keys = list(grouped_data.groups.keys())

    # Shuffle the group keys to randomize the order
    # random.shuffle(group_keys)

    # Iterate through each group (each unique image)
    for file_name in group_keys:
        group = grouped_data.get_group(file_name)
        image_path = os.path.join(image_dir, file_name)

        # Load image
        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        datamarkin_format = pixelflow.predictions.from_datamarkin_csv(group, height, width)
        print(datamarkin_format)

        # Iterate through rows for the current image
        for index, row in group.iterrows():
            # Get the bounding box coordinates and denormalize them
            xmin = int(row['xmin'] * width)
            ymin = int(row['ymin'] * height)
            xmax = int(row['xmax'] * width)
            ymax = int(row['ymax'] * height)

            pixelflow.draw.rectangle(img, (xmin, ymin), (xmax, ymax))

            # Convert normalized points to pixel coordinates
            segmentation_list = ast.literal_eval(row['segmentation'])

            segmentation_points = []
            for i in range(0, len(segmentation_list), 2):
                x = int(segmentation_list[i] * width)
                y = int(segmentation_list[i + 1] * height)
                segmentation_points.append((x, y))

            pixelflow.draw.polygon(img,segmentation_points)

            # Convert to a NumPy array and reshape to the required format for polylines
            # segmentation_points = np.array(segmentation_points, dtype=np.int32)
            # segmentation_points = segmentation_points.reshape((-1, 1, 2))

            # Draw the polygon on the image
            # cv2.polylines(img, [segmentation_points], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow(str(file_name), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def frame(image, predictions: pixelflow.predictions.Predictions):
    """
    Draw polygons on the image using the masks from a Predictions object and display the result.

    Parameters:
    - image: The original image (NumPy array).
    - predictions: A Predictions object containing multiple Prediction objects with polygon masks.

    This function will draw the polygons defined by each prediction's mask onto the image
    and then display the resulting image.
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
        for mask in prediction.masks:
            # Draw the polygon on the image using the polygon points (mask)
            image = pixelflow.draw.polygon(image, mask)

    # Display the image with the polygons
    cv2.imshow("Predictions Visualization", image)

    # Wait indefinitely until a key is pressed
    cv2.waitKey(0)

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

