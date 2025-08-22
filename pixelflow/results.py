# results/__init__.py

import json
import ast
from pixelflow.validators import (validate_bbox,
                                  validate_masks,
                                  round_to_decimal,
                                  convert_datamarkin_masks,
                                  simplify_polygon)
from pixelflow.zones import Zones
from typing import (List,
                    Iterator)
import cv2
import numpy as np
from typing import Optional


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
    def __init__(self, inference_id=None, bbox=None, masks=None, segments=None, keypoints: List[KeyPoint] = None, class_id=None,
                 class_name=None, labels=None, confidence=None, tracker_id=None, data=None):
        self.inference_id = inference_id
        self.bbox = validate_bbox(bbox)
        self.masks = masks
        self.segments = segments
        self.keypoints = keypoints if keypoints is not None else None
        self.class_id = class_id
        self.class_name = class_name
        self.labels = labels
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
            "segments": self.segments,
            "keypoints": [kp.to_dict() for kp in self.keypoints] if self.keypoints is not None else None,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "labels": self.labels,
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


class Results:
    def __init__(self, zones: 'Zones' = None):
        self.predictions: List[Prediction] = []
        self.zones = zones  # Store the zones object if provided

    def show(self):
        # Display the annotated image
        pass

    def add_prediction(self, prediction: Prediction):
        """
        Add a prediction to the list, applying zone filtering if zones are provided.
        """
        if self.zones:
            # Check if the prediction is excluded or not included in zones
            if self.zones.is_excluded(prediction.bbox, prediction.masks):
                return  # Skip if the prediction is in an excluded zone
            if not self.zones.is_included(prediction.bbox, prediction.masks):
                return  # Skip if the prediction is outside of included zones

        # Add the prediction if it passes the zone checks
        self.predictions.append(prediction)

    def __len__(self):
        return len(self.predictions)

    def __iter__(self) -> Iterator[Prediction]:
        return iter(self.predictions)

    def __getitem__(self, index: int) -> Prediction:
        return self.predictions[index]

    def filter_by_confidence(self, threshold: float) -> 'Results':
        """
        Returns a new Predictions object containing only predictions
        with a confidence score greater than or equal to the given threshold.
        """
        filtered_predictions = Results()
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

    def to_json(self):
        """
        Converts the list of predictions into a JSON string.
        """
        predictions_dict = [prediction.to_dict() for prediction in self.predictions]
        return json.dumps(predictions_dict, indent=4)

    def to_dict(self):
        """
        Converts the list of predictions into a JSON string.
        """
        return [prediction.to_dict() for prediction in self.predictions]

    def to_json_with_metrics(self) -> str:
        """
        Converts the list of predictions into a JSON string.
        More to come here
        """
        predictions_dict = [prediction.to_dict() for prediction in self.predictions]
        return json.dumps(predictions_dict, indent=4)


def from_datamarkin_api(api_response: dict, zones: Zones = None) -> Results:
    """
    Converts the Datamarkin API response to a `Predictions` object, filtering based on included/excluded zones.

    Args:
        api_response (dict): The API response in dictionary format.
        zones (Zones, optional): The Zones object for managing included/excluded zones. If None, no filtering is applied.

    Returns:
        Predictions: The corresponding Predictions object.
    """

    predictions_obj = Results()

    for obj in api_response.get("predictions", {}).get("objects", []):
        bbox = obj.get("bbox", [])
        mask = obj.get("mask", [])
        keypoints = obj.get("keypoints", [])
        class_name = obj.get("class", "")
        confidence = obj.get("bbox_score", None)

        # Create the Prediction object
        prediction = Prediction(
            bbox=bbox,
            mask=mask,
            keypoints=keypoints,
            class_id=class_name,
            confidence=confidence,
        )

        # Add the prediction to the list
        predictions_obj.add_prediction(prediction)

    return predictions_obj



def from_detectron2(detectron2_results) -> Results:
    """
    Converts Detectron2 results to a custom Results object.

    Args:
        detectron2_results: Detectron2 inference results containing instances with prediction data.

    Returns:
        Results: A unified Results object containing predictions.
    """
    predictions_obj = Results()

    # Extract data from Detectron2 results
    bboxes = detectron2_results["instances"].pred_boxes.tensor.cpu().numpy()  # Bounding boxes
    confidences = detectron2_results["instances"].scores.cpu().numpy()       # Confidence scores
    class_ids = detectron2_results["instances"].pred_classes.cpu().numpy().astype(int)  # Class IDs
    masks = (
        detectron2_results["instances"].pred_masks.cpu().numpy()
        if hasattr(detectron2_results["instances"], "pred_masks")
        else None
    )  # Optional segmentation masks

    # Iterate over each detection
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]
        mask = masks[i] if masks is not None else None

        # Create a Prediction object
        prediction = Prediction(
            bbox=bbox.tolist(),
            masks=mask if mask is None else mask.astype(bool),  # Ensure masks are binary
            segments=None,  # Detectron2 does not provide polygon segments
            keypoints=None,  # Add if keypoints are part of your format
            class_id=int(class_id),
            confidence=float(confidence)
        )

        # Add the prediction to the Results object
        predictions_obj.add_prediction(prediction)

    return predictions_obj


def from_ultralytics(ultralytics_results) -> Results:
    """
    Converts Ultralytics YOLO results to a custom Results object.

    Args:
        ultralytics_results: YOLOv8 results from the Ultralytics library.

    Returns:
        Results: A unified Results object containing predictions.
    """
    predictions_obj = Results()

    # Loop through all detections in ultralytics_results
    for ultralytics_result in ultralytics_results:
        # Get bounding boxes in xyxy format
        box = ultralytics_result.boxes.xyxy.cpu().numpy()[0]
        confidence = ultralytics_result.boxes.conf.cpu().numpy()[0]
        class_id = ultralytics_result.boxes.cls.cpu().numpy().astype(int)[0]
        segments = ultralytics_result.masks.xy[0]
        masks = extract_pixel_perfect_ultralytics_masks(ultralytics_result)

        prediction = Prediction(
            bbox=box.tolist(),
            masks=masks,
            segments=segments.astype(int).tolist(),
            keypoints=None,
            class_id=int(class_id),
            confidence=float(confidence)
        )

        # Add to the predictions list
        predictions_obj.add_prediction(prediction)

    return predictions_obj


def from_transformers(transformers_results):
    pass


def from_sam(sam_results):
    pass


def from_datamarkin_csv(group, height, width) -> Results:
    """
    Converts CSV data to a `Predictions` object.

    Args:
        group: The pandas DataFrame group with the CSV rows.
        height: Image height to denormalize the bounding box and segmentation coordinates.
        width: Image width to denormalize the bounding box and segmentation coordinates.

    Returns:
        Predictions: The corresponding Predictions object.
    """

    predictions_obj = Results()

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

def extract_ultralytics_masks(yolo_results) -> Optional[np.ndarray]:
    """
    Resizes segmentation masks from YOLO results to match the original image dimensions.

    Args:
        yolo_results: YOLO inference results containing segmentation masks.

    Returns:
        Optional[np.ndarray]: Array of binary masks resized to the original image dimensions,
                              or None if no masks are available.
    """
    # Check if masks are available in the YOLO results
    if not yolo_results.masks:
        return None

    # Get the original image shape
    original_shape = yolo_results.orig_shape

    # Extract masks from the YOLO results and resize them
    raw_masks = yolo_results.masks.data.cpu().numpy()
    resized_masks = [
        cv2.resize(mask, (original_shape[1], original_shape[0])) > 0.5  # Resize and threshold
        for mask in raw_masks
    ]

    # Convert to binary masks and return
    return np.asarray(resized_masks, dtype=bool)

def extract_pixel_perfect_ultralytics_masks(yolo_results) -> Optional[np.ndarray]:
    """
    Extracts segmentation masks from YOLO results and resizes them to match the original image dimensions,
    including handling padding for pixel-perfect alignment.

    Args:
        yolo_results: YOLO inference results containing segmentation masks.

    Returns:
        Optional[np.ndarray]: Array of binary masks resized to the original image dimensions,
                              or None if no masks are available.
    """
    # Check if masks are available in the YOLO results
    if not yolo_results.masks:
        return None

    # Get the original image shape and inference shape
    original_shape = yolo_results.orig_shape  # (height, width)
    inference_shape = tuple(yolo_results.masks.data.shape[1:])  # (height, width)

    # Calculate gain and padding used during preprocessing
    gain = min(inference_shape[0] / original_shape[0], inference_shape[1] / original_shape[1])
    pad_w = (inference_shape[1] - original_shape[1] * gain) / 2  # Width padding
    pad_h = (inference_shape[0] - original_shape[0] * gain) / 2  # Height padding

    # Convert padding values to integers
    pad_w = int(pad_w)
    pad_h = int(pad_h)

    # Extract masks from YOLO results
    raw_masks = yolo_results.masks.data.cpu().numpy()

    # Initialize an empty list to store resized masks
    aligned_masks = []

    for mask in raw_masks:
        # Remove padding and resize to the original shape
        cropped_mask = mask[pad_h:inference_shape[0] - pad_h, pad_w:inference_shape[1] - pad_w]

        # Resize to the original image dimensions
        resized_mask = cv2.resize(cropped_mask, (original_shape[1], original_shape[0]))

        # Threshold to create a binary mask
        binary_mask = resized_mask > 0.5

        # Add the binary mask to the list
        aligned_masks.append(binary_mask)

    # Convert the list of masks to a NumPy array and return
    return np.asarray(aligned_masks, dtype=bool)