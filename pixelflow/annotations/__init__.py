# annotations/__init__.py

import cv2
import json
import numpy as np
import pixelflow.draw
import pandas as pd
import ast
from typing import List, Iterator


# Object-oriented approach instead of a NumPy array-based approach
# Let's see how it goes


class KeyPoint:
    def __init__(self, x: int, y: int, name: str, visibility: bool):
        self.x = x
        self.y = y
        self.name = name
        self.visibility = visibility

    def to_dict(self):
        """
        Convert the KeyPoint object to a dictionary that can be easily converted to JSON.
        """
        return {
            "x": self.x,
            "y": self.y,
            "name": self.name,
            "visibility": self.visibility
        }


class Prediction:
    def __init__(self, inference_id=None, bbox=None, mask=None, keypoints: List[KeyPoint] = None, class_id=None,
                 confidence=None, tracker_id=None, data=None):
        self.inference_id = inference_id
        self.bbox = bbox
        self.mask = mask
        self.keypoints = keypoints if keypoints is not None else []
        self.class_id = class_id
        self.confidence = confidence
        self.tracker_id = tracker_id
        self.data = data

    def to_dict(self):
        """
        Convert the Prediction object to a dictionary that can be easily converted to JSON.
        """
        return {
            "inference_id": self.inference_id,
            "bbox": self.bbox,
            "mask": self.mask,
            "keypoints": [kp.to_dict() for kp in self.keypoints],
            "class_id": self.class_id,
            "confidence": self.confidence,
            "tracker_id": self.tracker_id
        }


class Predictions:
    def __init__(self):
        self.predictions: List[Prediction] = []

    def add_prediction(self, prediction: Prediction):
        self.predictions.append(prediction)

    def __len__(self):
        return len(self.predictions)

    def __iter__(self) -> Iterator[Prediction]:
        return iter(self.predictions)

    def __getitem__(self, index: int) -> Prediction:
        return self.predictions[index]

    def filter_by_confidence(self, threshold: float) -> 'Predictions':
        """
        Returns a new Predictions object containing only predictions
        with a confidence score greater than or equal to the given threshold.
        """
        filtered_predictions = Predictions()
        for prediction in self.predictions:
            if prediction.confidence is not None and prediction.confidence >= threshold:
                filtered_predictions.add_prediction(prediction)
        return filtered_predictions

    def to_json(self) -> str:
        """
        Converts the list of predictions into a JSON string.
        """
        predictions_dict = [prediction.to_dict() for prediction in self.predictions]
        return json.dumps(predictions_dict, indent=4)


# # Example usage
# pred1 = Prediction(inference_id=1, bbox=[0, 0, 10, 10], class_id=0, confidence=0.9)
# pred2 = Prediction(inference_id=2, bbox=[15, 15, 25, 25], class_id=1, confidence=0.8)
# pred3 = Prediction(inference_id=3, bbox=[30, 30, 40, 40], class_id=2, confidence=0.7)
#
# predictions = Predictions()
# predictions.add_prediction(pred1)
# predictions.add_prediction(pred2)
# predictions.add_prediction(pred3)
#
# for pred in predictions:
#     print(pred.bbox, pred.confidence)


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
