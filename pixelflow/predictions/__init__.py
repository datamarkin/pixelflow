# predictions/__init__.py

import json
import ast
from pixelflow.validators import (validate_bbox,
                                  validate_masks,
                                  round_to_decimal,
                                  convert_datamarkin_masks,
                                  simplify_polygon)
from typing import (List,
                    Iterator)


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
        self.bbox = validate_bbox(bbox)
        self.masks = validate_masks(mask)
        self.keypoints = keypoints if keypoints is not None else None
        self.class_id = class_id
        self.confidence = round_to_decimal(confidence)
        self.tracker_id = tracker_id
        self.data = data

    def to_dict(self):
        """
        Convert the Prediction object to a dictionary that can be easily converted to JSON.
        """
        return {
            "inference_id": self.inference_id,
            "bbox": self.bbox,
            "mask": self.masks,
            "keypoints": [kp.to_dict() for kp in self.keypoints] if self.keypoints is not None else None,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "tracker_id": self.tracker_id
        }

    def simplify_masks(self, tolerance: float = 2.0, preserve_topology: bool = True):
        """
        Simplifies the polygon masks using Shapely.

        Args:
            tolerance (float): The tolerance factor for simplification (higher = more simplified).
            preserve_topology (bool): If True, the function will try to preserve the polygon's topology.
        """
        if self.masks:
            # Apply the simplify function to each mask (assuming self.masks is a list of polygons)
            self.masks = [simplify_polygon(mask, tolerance, preserve_topology) for mask in self.masks]


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

    def simplify(self, tolerance: float = 2.0, preserve_topology: bool = True):
        """
        Simplifies the masks of all predictions in the Predictions object.

        Args:
            tolerance (float): The tolerance factor for simplification.
            preserve_topology (bool): Whether to preserve the topology.
        """
        for prediction in self.predictions:
            prediction.simplify_masks(tolerance=tolerance, preserve_topology=preserve_topology)
        return self

    def to_json(self) -> str:
        """
        Converts the list of predictions into a JSON string.
        """
        predictions_dict = [prediction.to_dict() for prediction in self.predictions]
        return json.dumps(predictions_dict, indent=4)

    def to_json_with_metrics(self) -> str:
        """
        Converts the list of predictions into a JSON string.
        More to come here
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


def from_datamarkin_api(api_response: dict) -> Predictions:
    """
    Converts the Datamarkin API response to a `Predictions` object.

    Args:
        api_response (dict): The API response in dictionary format.

    Returns:
        Predictions: The corresponding Predictions object.
    """

    # if hasattr(api_response, 'predictions'):
    # if hasattr(api_response, 'errors'):
    # TODO: Add some checks when basic idea starts working

    predictions_obj = Predictions()

    for obj in api_response.get("predictions", {}).get("objects", []):
        bbox = obj.get("bbox", [])
        mask = obj.get("mask", [])
        keypoints = obj.get("keypoints", [])
        class_name = obj.get("class", "")
        confidence = obj.get("bbox_score", None)

        # Create the Prediction object
        prediction = Prediction(
            bbox=bbox,
            mask=convert_datamarkin_masks(mask),
            keypoints=keypoints,
            class_id=class_name,
            confidence=confidence,
        )

        # Add to the predictions list
        predictions_obj.add_prediction(prediction)

    return predictions_obj


def from_detectron2(detectron2_results):
    pass


def from_ultralytics(ultralytics_results) -> Predictions:
    """
    Converts Ultralytics YOLOv8 inference results to a `Predictions` object.

    Args:
        ultralytics_results: The YOLOv8 inference results from the Ultralytics model.

    Returns:
        Predictions: A `Predictions` object containing all the detections from the model.
    """
    predictions_obj = Predictions()

    # Loop through all detections in ultralytics_results
    for result in ultralytics_results:
        # Get bounding boxes in xyxy format
        bboxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes as [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
        masks = None

        # If there are masks (segmentation), extract them
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # Masks in boolean array

        # Loop through all detections in the current result
        for bbox, confidence, class_id in zip(bboxes, confidences, class_ids):
            # Create a Prediction object
            prediction = Prediction(
                bbox=bbox.tolist(),
                mask=masks,  # If masks exist, provide them
                keypoints=None,  # Keypoints not used in this example
                class_id=str(class_id),  # Convert class_id to string for compatibility
                confidence=float(confidence)  # Convert confidence to float
            )

            # Add to the predictions list
            predictions_obj.add_prediction(prediction)

    return predictions_obj


def from_transformers(transformers_results):
    pass


def from_sam(sam_results):
    pass


def from_datamarkin_csv(group, height, width) -> Predictions:
    """
    Converts CSV data to a `Predictions` object.

    Args:
        group: The pandas DataFrame group with the CSV rows.
        height: Image height to denormalize the bounding box and segmentation coordinates.
        width: Image width to denormalize the bounding box and segmentation coordinates.

    Returns:
        Predictions: The corresponding Predictions object.
    """

    predictions_obj = Predictions()

    for index, row in group.iterrows():
        # Get the bounding box coordinates and denormalize them
        xmin = int(row['xmin'] * width)
        ymin = int(row['ymin'] * height)
        xmax = int(row['xmax'] * width)
        ymax = int(row['ymax'] * height)

        # Convert normalized points to pixel coordinates for the mask
        segmentation_list = ast.literal_eval(row['segmentation'])
        segmentation_points = []
        for i in range(0, len(segmentation_list), 2):
            x = int(segmentation_list[i] * width)
            y = int(segmentation_list[i + 1] * height)
            segmentation_points.append((x, y))  # Convert to tuple for polygon points

        # Create the Prediction object
        prediction = Prediction(
            bbox=[xmin, ymin, xmax, ymax],
            mask=[segmentation_points],  # Add mask as list of lists of tuples
            keypoints=None,  # TODO
            class_id=row['class'],
            confidence=row.get('confidence', None)  # Add confidence if available
        )

        # Add the prediction to the predictions list
        predictions_obj.add_prediction(prediction)

    return predictions_obj
