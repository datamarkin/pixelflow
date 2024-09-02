# visualise/__init__.py

import os
import cv2
import ast
import numpy as np
import pandas as pd
import pixelflow.draw
import pixelflow.annotations


def from_datamarkin_csv(csv_path, image_dir):
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

        datamarkin_format = pixelflow.annotations.dtm_csv_to_dtm(group, height, width)
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


def image(image, points):
    """
    Draw a polygon on the image using the provided points and display the result.

    Parameters:
    - image: The original image (NumPy array).
    - points: A list of points defining the polygon's vertices.

    This function will use the existing polygon() function to draw the polygon
    and then display the resulting image in a window.
    """
    # Draw the polygon on the image using the existing polygon() function
    image_with_polygon = pixelflow.draw.polygon(image, points)

    # Display the image with the polygon
    cv2.imshow("Polygon Visualization", image_with_polygon)

    # Wait indefinitely until a key is pressed
    cv2.waitKey(0)

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()
