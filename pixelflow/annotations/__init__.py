# annotations/__init__.py

import cv2
import numpy as np
import pixelflow.draw
import pandas as pd
import ast


def dtm_csv_to_dtm(group, height, width):
    annotations = {}

    objects = []
    for index, row in group.iterrows():

        # Get the bounding box coordinates and denormalize them
        xmin = int(row['xmin'] * width)
        ymin = int(row['ymin'] * height)
        xmax = int(row['xmax'] * width)
        ymax = int(row['ymax'] * height)

        # Convert normalized points to pixel coordinates
        segmentation_list = ast.literal_eval(row['segmentation'])

        segmentation_points = []
        for i in range(0, len(segmentation_list), 2):
            x = int(segmentation_list[i] * width)
            y = int(segmentation_list[i + 1] * height)
            segmentation_points.append([x, y])

        x_object = {}
        x_object['bbox'] = [xmin, ymin, xmax, ymax]
        x_object['class'] = row['class']
        x_object['segmentation'] = segmentation_points

        # x_object['keypoints'] = [
        #     [
        #         {
        #             "name": "p0",
        #             "point": [
        #                 0.2026,
        #                 0.2128
        #             ]
        #         },
        #         {
        #             "name": "p1",
        #             "point": [
        #                 0.2719,
        #                 0.25
        #             ]
        #         }
        #     ]]

        objects.append(x_object)
    annotations['objects'] = objects

    return annotations
